"""BGRL (Bootstrapped Graph Representation Learning) with Half-Hop augmentation.

Implements:
  - Two GCN encoders (online + target) with EMA weight update
  - Half-Hop as one of the two graph augmentations
  - A linear probing evaluation protocol (standard for SSL GNNs)

Reference:
  Thakoor et al., "Large-Scale Representation Learning on Graphs via
  Bootstrapping", ICLR 2022.

  Zhao et al., "Half-Hop: A graph upsampling approach for slowing down
  message passing", ICML 2023.
"""

import argparse
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from tqdm import trange

from halfhop.datasets import load_dataset
from halfhop.halfhop import HalfHop
from halfhop.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device(
    os.environ.get('HALFHOP_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class GCNEncoder(nn.Module):
    """2-layer GCN encoder used for BGRL."""

    def __init__(self, in_channels: int, hidden_channels: int = 512,
                 out_channels: int = 256):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.prelu(x, weight=torch.tensor(0.25).to(x.device))
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Predictor MLP (online network only)
# ---------------------------------------------------------------------------
class Predictor(nn.Module):
    """2-layer MLP predictor head."""

    def __init__(self, in_channels: int, hidden_channels: int = 512,
                 out_channels: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.prelu(x, weight=torch.tensor(0.25).to(x.device))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Half-Hop augmentation helper
# ---------------------------------------------------------------------------
def halfhop_augment(data: Data, alpha: float = 0.5, p: float = 0.75) -> Data:
    """Apply Half-Hop transformation and return augmented Data."""
    hh = HalfHop(alpha=alpha, p=p, inplace=False)
    aug = hh(data)
    # Return only the original-node features and new edge index
    return aug


def edge_dropout(data: Data, drop_rate: float = 0.3) -> Data:
    """Random edge dropout augmentation."""
    data = copy.deepcopy(data)
    num_edges = data.edge_index.size(1)
    mask = torch.rand(num_edges, device=data.edge_index.device) > drop_rate
    data.edge_index = data.edge_index[:, mask]
    return data


# ---------------------------------------------------------------------------
# BGRL model
# ---------------------------------------------------------------------------
class BGRL(nn.Module):
    """BGRL: Bootstrap Your Own Graph Representations.

    Uses two augmented views:
      - View 1: Half-Hop augmentation (the key contribution from the paper)
      - View 2: Edge dropout augmentation

    Features:
      - Multi-GPU: Automatically places Online network on GPU 0 and Target network on GPU 1
        when 2+ GPUs are available.
      - Memory efficiency: Computes target representations first (no autograd graph),
        then backpropagates View 1 and View 2 sequentially to avoid holding both
        activation graphs in GPU memory at once.
    """

    def __init__(self, encoder: GCNEncoder, predictor: Predictor,
                 online_device=None, target_device=None):
        super().__init__()
        self.online_device = online_device or DEVICE
        self.target_device = target_device or self.online_device

        self.online_encoder = encoder.to(self.online_device)
        self.predictor = predictor.to(self.online_device)

        # Target encoder: EMA copy, no gradients
        self.target_encoder = copy.deepcopy(encoder).to(self.target_device)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau: float = 0.99):
        """EMA update: theta_target = tau * theta_target + (1-tau) * theta_online."""
        for online_p, target_p in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_p.data.mul_(tau).add_(
                online_p.data.to(target_p.device), alpha=1.0 - tau
            )

    def train_step(self, x1, edge1, x2, edge2, original_mask=None, use_amp=True):
        """Memory-efficient training step.
        Computes target representations first under no_grad, then runs
        each online branch and backpropagates sequentially to minimize peak memory.
        """
        device_type = 'cuda' if self.online_device.type == 'cuda' else 'cpu'
        amp_enabled = use_amp and (device_type == 'cuda')

        # 1. Target representations (no autograd graph stored)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                t1 = self.target_encoder(x1.to(self.target_device), edge1.to(self.target_device))
                if original_mask is not None:
                    t1 = t1[original_mask.to(self.target_device)]
                t1 = t1.to(self.online_device)

                t2 = self.target_encoder(x2.to(self.target_device), edge2.to(self.target_device))
                t2 = t2.to(self.online_device)

        # 2. Online View 1 -> predicts t2
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            z1 = self.online_encoder(x1.to(self.online_device), edge1.to(self.online_device))
            q1 = self.predictor(z1)
            if original_mask is not None:
                q1 = q1[original_mask.to(self.online_device)]
            loss1 = _cosine_loss(q1, t2)

        loss1.backward()
        loss1_val = loss1.item()
        del z1, q1, loss1

        # 3. Online View 2 -> predicts t1
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            z2 = self.online_encoder(x2.to(self.online_device), edge2.to(self.online_device))
            q2 = self.predictor(z2)
            loss2 = _cosine_loss(q2, t1)

        loss2.backward()
        loss2_val = loss2.item()
        del z2, q2, loss2, t1, t2

        return loss1_val + loss2_val

    def forward(self, x1, edge1, x2, edge2, original_mask=None):
        """Standard forward method for compatibility."""
        device_type = 'cuda' if self.online_device.type == 'cuda' else 'cpu'
        with torch.no_grad():
            t1 = self.target_encoder(x1.to(self.target_device), edge1.to(self.target_device))
            if original_mask is not None:
                t1 = t1[original_mask.to(self.target_device)]
            t1 = t1.to(self.online_device)
            t2 = self.target_encoder(x2.to(self.target_device), edge2.to(self.target_device)).to(self.online_device)

        z1 = self.online_encoder(x1.to(self.online_device), edge1.to(self.online_device))
        q1 = self.predictor(z1)
        if original_mask is not None:
            q1 = q1[original_mask.to(self.online_device)]

        z2 = self.online_encoder(x2.to(self.online_device), edge2.to(self.online_device))
        q2 = self.predictor(z2)

        loss = _cosine_loss(q1, t2.detach()) + _cosine_loss(q2, t1.detach())
        return loss

    @torch.no_grad()
    def get_embeddings(self, x, edge_index):
        """Get node embeddings using the online encoder."""
        self.online_encoder.eval()
        return self.online_encoder(x.to(self.online_device), edge_index.to(self.online_device))


def _cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity loss (BGRL objective)."""
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Linear evaluation protocol
# ---------------------------------------------------------------------------
def linear_eval(embeddings: np.ndarray, labels: np.ndarray,
                train_mask: np.ndarray, val_mask: np.ndarray,
                test_mask: np.ndarray) -> tuple:
    """Logistic regression linear evaluation."""
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, solver='lbfgs',
                                  multi_class='auto', C=1.0))
    ])
    clf.fit(embeddings[train_mask], labels[train_mask])
    val_acc = clf.score(embeddings[val_mask], labels[val_mask])
    test_acc = clf.score(embeddings[test_mask], labels[test_mask])
    return val_acc, test_acc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_bgrl(dataset_name: str = "amazon-computers",
             hidden: int = 512, out_channels: int = 256,
             epochs: int = 1000, lr: float = 1e-5,
             tau: float = 0.99, hh_alpha: float = 0.5, hh_p: float = 0.75,
             edge_drop: float = 0.3, seed: int = 42,
             use_amp: bool = True) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    set_seed(seed)
    dataset, data = load_dataset(dataset_name)

    in_channels = dataset.num_features
    num_classes = dataset.num_classes

    # Multi-GPU support (e.g. 2x T4 on Kaggle)
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        online_device = torch.device('cuda:0')
        target_device = torch.device('cuda:1')
        print(f"Detected {num_gpus} GPUs! Dual-GPU enabled:")
        print(f"  - Online Network (Backprop): {torch.cuda.get_device_name(0)} ({online_device})")
        print(f"  - Target Network (EMA):      {torch.cuda.get_device_name(1)} ({target_device})")
    elif torch.cuda.is_available():
        online_device = torch.device('cuda:0')
        target_device = torch.device('cuda:0')
        print(f"Using Single GPU: {torch.cuda.get_device_name(0)} ({online_device})")
    else:
        online_device = torch.device('cpu')
        target_device = torch.device('cpu')
        print("Using CPU")

    encoder = GCNEncoder(in_channels, hidden, out_channels)
    predictor = Predictor(out_channels, hidden, out_channels)
    model = BGRL(encoder, predictor, online_device=online_device, target_device=target_device)

    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) +
        list(model.predictor.parameters()),
        lr=lr, weight_decay=1e-5
    )

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    print(f"Training BGRL on {dataset_name} | {in_channels} feats | "
          f"{num_classes} classes | {data.num_nodes} nodes | AMP: {use_amp}")

    model.train()
    progress = trange(1, epochs + 1, desc=f"BGRL {dataset_name}", unit="epoch")
    for epoch in progress:
        optimizer.zero_grad()

        # View 1: Half-Hop augmented graph
        aug1 = halfhop_augment(data, alpha=hh_alpha, p=hh_p)
        x1 = aug1.x
        e1 = aug1.edge_index

        # View 2: Edge dropout on original graph
        aug2 = edge_dropout(data, drop_rate=edge_drop)
        x2 = aug2.x
        e2 = aug2.edge_index

        original_mask = ~aug1.slow_node_mask if hasattr(aug1, 'slow_node_mask') else None
        loss_val = model.train_step(x1, e1, x2, e2, original_mask=original_mask, use_amp=use_amp)
        optimizer.step()
        model.update_target(tau=tau)
        scheduler.step()

        del aug1, aug2, x1, e1, x2, e2

        if epoch % 100 == 0 or epoch == 1:
            progress.set_postfix(loss=f"{loss_val:.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Linear evaluation
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index)
    embeddings = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()

    # Use data masks if available, else create a simple random split
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 1:
        train_mask = data.train_mask.cpu().numpy().astype(bool)
        val_mask = data.val_mask.cpu().numpy().astype(bool)
        test_mask = data.test_mask.cpu().numpy().astype(bool)
    else:
        # Create a 60/20/20 random split for evaluation
        n = data.num_nodes
        idx = np.random.permutation(n)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[idx[:int(0.6 * n)]] = True
        val_mask[idx[int(0.6 * n):int(0.8 * n)]] = True
        test_mask[idx[int(0.8 * n):]] = True

    val_acc, test_acc = linear_eval(embeddings, labels,
                                    train_mask, val_mask, test_mask)

    print("\n" + "=" * 60)
    print(f"FINAL SSL RESULTS: HH-BGRL on {dataset_name}")
    print("=" * 60)
    print(f"Val Accuracy:  {val_acc:.4f} ({val_acc:.2%})")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    return {"val_acc": val_acc, "test_acc": test_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BGRL with Half-Hop")
    parser.add_argument("--dataset", type=str, default="amazon-computers")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--out_channels", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--hh_alpha", type=float, default=0.5)
    parser.add_argument("--hh_p", type=float, default=0.75)
    parser.add_argument("--edge_drop", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    run_bgrl(
        dataset_name=args.dataset,
        hidden=args.hidden,
        out_channels=args.out_channels,
        epochs=args.epochs,
        lr=args.lr,
        tau=args.tau,
        hh_alpha=args.hh_alpha,
        hh_p=args.hh_p,
        edge_drop=args.edge_drop,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
