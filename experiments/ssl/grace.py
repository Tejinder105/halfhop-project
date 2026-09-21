"""GRACE (Graph Contrastive Representation Learning) with Half-Hop augmentation.

Implements:
  - Two augmented views of the graph
  - View 1: Half-Hop augmentation (p=0.75, alpha=0.5)
  - View 2: Feature masking
  - Node-level InfoNCE contrastive loss (NT-Xent style)
  - Linear probing evaluation

Reference:
  Zhu et al., "Deep Graph Contrastive Representation Learning", 2020.

  Zhao et al., "Half-Hop: A graph upsampling approach for slowing down
  message passing", ICML 2023.
"""

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

from halfhop.datasets import load_dataset
from halfhop.halfhop import HalfHop
from halfhop.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class GCNEncoder(nn.Module):
    """2-layer GCN encoder for GRACE."""

    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 out_channels: int = 128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """2-layer MLP projection head."""

    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
def halfhop_augment(data: Data, alpha: float = 0.5, p: float = 0.75) -> Data:
    """Apply Half-Hop transformation; returns augmented Data object."""
    hh = HalfHop(alpha=alpha, p=p, inplace=False)
    return hh(data)


def feature_mask(data: Data, mask_rate: float = 0.3) -> Data:
    """Zero-mask a random fraction of feature dimensions."""
    data = copy.deepcopy(data)
    num_feats = data.x.size(1)
    mask = torch.rand(num_feats, device=data.x.device) > mask_rate
    data.x = data.x * mask.float().unsqueeze(0)
    return data


# ---------------------------------------------------------------------------
# InfoNCE / NT-Xent loss
# ---------------------------------------------------------------------------
def grace_loss(z1: torch.Tensor, z2: torch.Tensor,
               tau: float = 0.4) -> torch.Tensor:
    """Node-level NT-Xent loss.

    Positive pair: (z1[i], z2[i]) — same node, different augmented views.
    Negatives: all other nodes in both views.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Similarity matrix: [N, N]
    sim_12 = torch.mm(z1, z2.t()) / tau
    sim_11 = torch.mm(z1, z1.t()) / tau
    sim_22 = torch.mm(z2, z2.t()) / tau

    n = z1.size(0)
    mask = torch.eye(n, dtype=torch.bool, device=z1.device)

    # Remove self-similarities from intra-view matrices
    sim_11 = sim_11.masked_fill(mask, float('-inf'))
    sim_22 = sim_22.masked_fill(mask, float('-inf'))

    # Numerator: exp(sim(z1_i, z2_i))
    # Denominator: sum over all negatives (all j != i in both views)
    pos_12 = sim_12.diag()
    neg_12 = torch.cat([sim_11, sim_12], dim=1)
    neg_21 = torch.cat([sim_22, sim_12.t()], dim=1)

    loss_12 = -pos_12 + torch.logsumexp(neg_12, dim=1)
    loss_21 = -sim_12.t().diag() + torch.logsumexp(neg_21, dim=1)

    return (loss_12 + loss_21).mean()


# ---------------------------------------------------------------------------
# Linear evaluation
# ---------------------------------------------------------------------------
def linear_eval(embeddings: np.ndarray, labels: np.ndarray,
                train_mask, val_mask, test_mask) -> tuple:
    """Logistic regression linear evaluation."""
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, solver='lbfgs',
                                  multi_class='auto', C=0.1))
    ])
    clf.fit(embeddings[train_mask], labels[train_mask])
    val_acc = clf.score(embeddings[val_mask], labels[val_mask])
    test_acc = clf.score(embeddings[test_mask], labels[test_mask])
    return val_acc, test_acc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_grace(dataset_name: str = "amazon_photo",
              hidden: int = 256, proj_dim: int = 128,
              epochs: int = 1000, lr: float = 0.001,
              tau: float = 0.4,
              hh_alpha: float = 0.5, hh_p: float = 0.75,
              feat_mask_rate: float = 0.3,
              seed: int = 42) -> dict:
    set_seed(seed)
    dataset, data = load_dataset(dataset_name)

    in_channels = dataset.num_features
    num_classes = dataset.num_classes
    data = data.to(DEVICE)

    encoder = GCNEncoder(in_channels, hidden, hidden).to(DEVICE)
    projector = ProjectionHead(hidden, proj_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    print(f"Training GRACE on {dataset_name} | {in_channels} feats | "
          f"{num_classes} classes | {data.num_nodes} nodes")
    print(f"Device: {DEVICE}")

    encoder.train()
    projector.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # View 1: Half-Hop augmented
        aug1 = halfhop_augment(data, alpha=hh_alpha, p=hh_p)
        x1, e1 = aug1.x, aug1.edge_index
        # View 2: Feature masking on original graph
        aug2 = feature_mask(data, mask_rate=feat_mask_rate)
        x2, e2 = aug2.x, aug2.edge_index

        # Encode both views
        h1 = encoder(x1, e1)
        h2 = encoder(x2, e2)

        # GRACE uses original node embeddings only (discard slow nodes from view 1)
        if hasattr(aug1, 'slow_node_mask'):
            h1 = h1[~aug1.slow_node_mask]

        # Project to contrastive space
        z1 = projector(h1)
        z2 = projector(h2)

        loss = grace_loss(z1, z2, tau=tau)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    # Linear evaluation using encoder (not projector)
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index).cpu().numpy()
    labels = data.y.cpu().numpy()

    if hasattr(data, 'train_mask') and data.train_mask.dim() == 1:
        train_mask = data.train_mask.cpu().numpy().astype(bool)
        val_mask = data.val_mask.cpu().numpy().astype(bool)
        test_mask = data.test_mask.cpu().numpy().astype(bool)
    else:
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
    print(f"FINAL SSL RESULTS: HH-GRACE on {dataset_name}")
    print("=" * 60)
    print(f"Val Accuracy:  {val_acc:.4f} ({val_acc:.2%})")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    return {"val_acc": val_acc, "test_acc": test_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRACE with Half-Hop")
    parser.add_argument("--dataset", type=str, default="amazon_photo")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--hh_alpha", type=float, default=0.5)
    parser.add_argument("--hh_p", type=float, default=0.75)
    parser.add_argument("--feat_mask_rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_grace(
        dataset_name=args.dataset,
        hidden=args.hidden,
        proj_dim=args.proj_dim,
        epochs=args.epochs,
        lr=args.lr,
        tau=args.tau,
        hh_alpha=args.hh_alpha,
        hh_p=args.hh_p,
        feat_mask_rate=args.feat_mask_rate,
        seed=args.seed,
    )
