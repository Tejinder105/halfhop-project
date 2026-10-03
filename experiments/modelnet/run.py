"""ModelNet 3D mesh graph classification with optional Half-Hop.

Meshes are converted to graphs (FaceToEdge). Compare GCN / GraphSAGE
with and without Half-Hop on the same protocol.

Kaggle (2x T4): official training is one process per GPU. Run two models
in parallel, e.g. baseline on GPU 0 and Half-Hop on GPU 1.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.transforms import Compose, FaceToEdge, NormalizeScale
from tqdm import tqdm

from halfhop.gcn import GCN
from halfhop.graphsage import GraphSAGE
from halfhop.halfhop import HalfHop
from halfhop.reproducibility import set_seed


class UsePosAsFeatures:
    """ModelNet meshes often have ``pos`` but no ``x``; GNN needs ``x``."""

    def __call__(self, data):
        if data.x is None:
            data.x = data.pos
        return data


class MeshClassifier(nn.Module):
    """Node encoder → (drop slow nodes) → global mean pool → class logits."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_channels: int,
        num_classes: int,
        use_halfhop: bool = False,
        alpha: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__()
        self.halfhop = (
            HalfHop(alpha=alpha, p=p, inplace=False) if use_halfhop else None
        )
        self.encoder = encoder
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        if self.halfhop is not None:
            data = self.halfhop(data)

        x = self.encoder(data.x, data.edge_index)

        if hasattr(data, "slow_node_mask") and data.slow_node_mask is not None:
            keep = ~data.slow_node_mask
            x = x[keep]
            batch = data.batch[keep]
        else:
            batch = data.batch

        return self.lin(global_mean_pool(x, batch))


def build_model(
    model_name: str,
    in_channels: int,
    hidden: int,
    num_classes: int,
    depth: int,
    dropout: float,
    alpha: float,
    p: float,
) -> MeshClassifier:
    name = model_name.lower()
    use_hh = name.startswith("hh-")
    backbone = name.replace("hh-", "")

    if backbone in ("gcn",):
        encoder = GCN(in_channels, hidden, hidden, dropout=dropout, depth=depth)
    elif backbone in ("sage", "graphsage"):
        encoder = GraphSAGE(
            in_channels, hidden, hidden, dropout=dropout, depth=depth
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Use gcn, sage, hh-gcn, or hh-sage."
        )

    return MeshClassifier(
        encoder=encoder,
        hidden_channels=hidden,
        num_classes=num_classes,
        use_halfhop=use_hh,
        alpha=alpha,
        p=p,
    )


def accuracy(loader, model, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).argmax(dim=-1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
    return correct / max(total, 1)


def train_one_epoch(loader, model, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / max(len(loader.dataset), 1)


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def main():
    parser = argparse.ArgumentParser(
        description="ModelNet mesh graph classification ± Half-Hop"
    )
    parser.add_argument(
        "--dataset",
        default="10",
        choices=["10", "40"],
        help="ModelNet10 or ModelNet40",
    )
    parser.add_argument(
        "--model",
        default="hh-gcn",
        choices=["gcn", "sage", "hh-gcn", "hh-sage"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Half-Hop probability. Use <1.0 on T4 to limit VRAM.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="data/ModelNet")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to append FINAL RESULTS line",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(device)} "
            f"(visible {torch.cuda.device_count()}). "
            "Pin one job per T4 with CUDA_VISIBLE_DEVICES."
        )

    pre_transform = Compose([NormalizeScale(), FaceToEdge(), UsePosAsFeatures()])
    root = args.data_dir

    print(f"Loading ModelNet{args.dataset} (download on first run)...")
    train_dataset = ModelNet(
        root, name=args.dataset, train=True, pre_transform=pre_transform
    )
    test_dataset = ModelNet(
        root, name=args.dataset, train=False, pre_transform=pre_transform
    )

    # Hold out 10% of train graphs for validation.
    from torch.utils.data import Subset

    n_train = len(train_dataset)
    n_val = max(1, n_train // 10)
    perm = torch.randperm(n_train)
    val_dataset = Subset(train_dataset, perm[:n_val].tolist())
    train_subset = Subset(train_dataset, perm[n_val:].tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    in_channels = train_dataset.num_features
    num_classes = train_dataset.num_classes
    print(
        f"ModelNet{args.dataset}: train={len(train_subset)} "
        f"val={len(val_dataset)} test={len(test_dataset)} "
        f"feats={in_channels} classes={num_classes}"
    )

    model = build_model(
        args.model,
        in_channels,
        args.hidden,
        num_classes,
        args.depth,
        args.dropout,
        args.alpha,
        args.p,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_state = None
    best_test = 0.0

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"ModelNet {args.model}"):
        loss = train_one_epoch(train_loader, model, optimizer, device)
        val_acc = accuracy(val_loader, model, device)
        test_acc = accuracy(test_loader, model, device)
        if val_acc >= best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d} loss={loss:.4f} "
                f"val={val_acc:.4f} test={test_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = accuracy(test_loader, model, device)

    line = (
        f"FINAL RESULTS: {args.model.upper()} on ModelNet{args.dataset} | "
        f"best_val={best_val:.4f} test_at_best_val={best_test:.4f} "
        f"final_test={final_test:.4f} seed={args.seed} p={args.p} alpha={args.alpha}"
    )
    print(line)

    out = args.output
    if out is None:
        out = str(
            Path("experiments/results")
            / f"modelnet{args.dataset}_{args.model.replace('-', '')}.txt"
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    # Avoid OpenMP oversubscription next to DataLoader workers on Kaggle.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
