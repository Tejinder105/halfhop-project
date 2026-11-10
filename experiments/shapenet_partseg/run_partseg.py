"""Geometry-Aware Half-Hop for ShapeNet Part Segmentation.

Arms: none | boundary | random
  none      plain KNN, no Half-Hop
  boundary  BoundaryHalfHop (slow nodes on inter-part boundary edges only)
  random    uniform random Half-Hop at same fraction as boundary produces

python -m experiments.shapenet_partseg.run_partseg --category Chair --condition boundary --epochs 100
python -m experiments.shapenet_partseg.run_partseg --category Chair --sweep
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from experiments.shapenet_partseg.model import EdgeConvSegModel
from halfhop.boundary_halfhop import BoundaryHalfHop
from halfhop.halfhop import HalfHop
from halfhop.reproducibility import set_seed

# Mirror URL for Kaggle / offline environments
ShapeNet.url = (
    "https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/"
    "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
)

CONDITIONS = ('none', 'boundary', 'random')


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _knn(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Build a symmetric KNN edge_index via cdist. Returns [2, N*k]."""
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float('inf'))
    nb = dist.topk(k, largest=False, dim=1).indices  # [N, k]
    dst = torch.arange(pos.size(0), device=pos.device).repeat_interleave(k)
    src = nb.reshape(-1)
    return torch.stack([src, dst], dim=0).long()


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class GraphBuild:
    """Pre-process a ShapeNet Data object into a graph ready for training."""

    def __init__(self, condition: str, k: int, budget: float, split: str):
        assert condition in CONDITIONS
        assert split in ('train', 'val', 'test')
        self.condition = condition
        self.k = k
        self.budget = budget
        self.split = split

    def __call__(self, data):
        data = copy.copy(data)

        pos = data.pos.float()  # [N, 3]
        normal = getattr(data, 'normal', None)

        # Node features: pos + normal if available, else just pos
        if normal is not None:
            x = torch.cat([pos, normal.float()], dim=-1)  # [N, 6]
        else:
            x = pos  # [N, 3]

        data.x = x
        data.y = data.y.view(-1).long()
        data.edge_index = _knn(pos, self.k)

        if self.condition == 'none':
            return data

        if self.condition == 'boundary':
            # Let BoundaryHalfHop handle edge construction from pos + y
            # It needs pos and y; pass data with both set.
            data = BoundaryHalfHop(
                knn=self.k,
                alpha=0.4,
                use_labels=(self.split == 'train'),
            )(data)
            return data

        if self.condition == 'random':
            data.edge_score = torch.ones(data.edge_index.size(1))
            data = HalfHop(selection='edge', budget=self.budget, inplace=True)(data)
            return data

        return data  # unreachable


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def instance_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_classes: list[int],
) -> float:
    """Per-shape mIoU over valid_classes only."""
    ious = []
    for cls in valid_classes:
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            # Class not present and not predicted — skip
            continue
        ious.append(inter / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


@torch.no_grad()
def evaluate(loader, model, device, valid_classes: list[int]):
    """Returns (accuracy, instance_mIoU) over the loader."""
    model.eval()
    total_correct = 0
    total_nodes = 0
    shape_mious = []

    for data in loader:
        data = data.to(device)
        logits = model(data)

        # Build target — slow nodes are masked out in model.forward
        mask = getattr(data, 'slow_node_mask', None)
        if mask is not None:
            y = data.y[~mask]
        else:
            y = data.y

        pred = logits.argmax(dim=-1)
        total_correct += (pred == y).sum().item()
        total_nodes += y.numel()

        # Per-graph mIoU (batch may contain multiple graphs)
        batch_vec = getattr(data, 'batch', None)
        if batch_vec is None:
            batch_vec = torch.zeros(y.size(0), dtype=torch.long, device=device)
        if mask is not None:
            batch_vec = batch_vec[~mask]

        for g in batch_vec.unique():
            m = batch_vec == g
            shape_mious.append(instance_miou(pred[m], y[m], valid_classes))

    acc = total_correct / max(total_nodes, 1)
    miou = float(sum(shape_mious) / len(shape_mious)) if shape_mious else 0.0
    return acc, miou


def train_one_epoch(loader, model, optimizer, device) -> float:
    """Train for one epoch; return mean cross-entropy loss."""
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)

        mask = getattr(data, 'slow_node_mask', None)
        if mask is not None:
            y = data.y[~mask]
        else:
            y = data.y

        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_graphs += 1

    return total_loss / max(total_graphs, 1)


# ---------------------------------------------------------------------------
# Single-condition run
# ---------------------------------------------------------------------------

def run_condition(args, condition: str, seed: int, device: torch.device) -> dict:
    set_seed(seed)

    root = str(Path(args.data_dir) / "ShapeNet")
    category = args.category

    print(f"\n=== condition={condition}  category={category}  seed={seed} ===")

    # Load raw datasets
    raw_train = ShapeNet(root=root, categories=[category], split='train',
                         include_normals=True)
    raw_val = ShapeNet(root=root, categories=[category], split='val',
                       include_normals=True)
    raw_test = ShapeNet(root=root, categories=[category], split='test',
                        include_normals=True)

    # Collect valid part classes from training set
    all_labels: set[int] = set()
    for d in raw_train:
        all_labels.update(d.y.view(-1).tolist())
    valid_classes = sorted(all_labels)

    # Eagerly apply transform
    def build(raw_ds, split):
        t = GraphBuild(condition, args.k, args.budget, split)
        return [t(d) for d in tqdm(raw_ds, desc=f"build {split}")]

    train_ds = build(raw_train, 'train')
    val_ds = build(raw_val, 'val')
    test_ds = build(raw_test, 'test')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Determine actual in_channels from first sample
    in_channels = train_ds[0].x.size(-1) if train_ds else 6

    model = EdgeConvSegModel(
        in_channels=in_channels,
        hidden=args.hidden,
        num_classes=50,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_miou = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(train_loader, model, optimizer, device)
        scheduler.step()
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            val_acc, val_miou = evaluate(val_loader, model, device, valid_classes)
            print(f"  epoch {epoch:3d}  loss={loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_mIoU={val_miou:.4f}")
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_state = copy.deepcopy(model.state_dict())

    # Reload best and evaluate test
    model.load_state_dict(best_state)
    test_acc, test_miou = evaluate(test_loader, model, device, valid_classes)
    print(f"  TEST  acc={test_acc:.4f}  mIoU={test_miou:.4f}")

    return dict(
        condition=condition,
        category=category,
        seed=seed,
        best_val_miou=best_val_miou,
        test_miou=test_miou,
        test_acc=test_acc,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ShapeNet Part Segmentation — Half-Hop ablation"
    )
    parser.add_argument('--category', default='Chair',
                        help='ShapeNet category name (e.g. Chair, Airplane)')
    parser.add_argument('--condition', default='none', choices=CONDITIONS,
                        help='Half-Hop arm to run (ignored when --sweep is set)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run all conditions sequentially')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=16,
                        help='KNN neighbourhood size')
    parser.add_argument('--budget', type=float, default=0.3,
                        help='Fraction of edges to half-hop (random condition)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-dir', default='data',
                        help='Root directory for datasets')
    parser.add_argument('--output', default='experiments/results/partseg.jsonl',
                        help='JSONL file to append results to')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Download data and exit (no training)')
    parser.add_argument('--device', default='auto',
                        help='cuda / cuda:0 / cuda:1 / cpu / auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if args.prepare_only:
        root = str(Path(args.data_dir) / "ShapeNet")
        print(f"Downloading ShapeNet [{args.category}] to {root} ...")
        ShapeNet(root=root, categories=[args.category], split='train',
                 include_normals=True)
        ShapeNet(root=root, categories=[args.category], split='val',
                 include_normals=True)
        ShapeNet(root=root, categories=[args.category], split='test',
                 include_normals=True)
        print("Download complete.")
        return

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = CONDITIONS if args.sweep else (args.condition,)
    for cond in conditions:
        result = run_condition(args, cond, args.seed, device)
        line = json.dumps(result)
        with open(out_path, 'a') as f:
            f.write(line + '\n')
        print(line)


if __name__ == '__main__':
    main()
