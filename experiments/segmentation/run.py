"""ShapeNet part segmentation: plain GCN vs edge-level Half-Hop.

Three arms, same kNN graph and the same fraction of slowed edges:

    gcn        no Half-Hop
    hh-random  that fraction, chosen uniformly
    hh-geom    that fraction, highest normal disagreement

Half-Hop runs on each shape before batching, so the budget is per shape.
Normals are used only to score edges. The GCN sees point coordinates.

    python -m experiments.segmentation.run --category Chair --sweep

The ShapeNet zip must already be present (see experiments.segmentation.proxy).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import experiments.segmentation.proxy  # noqa: F401  sets ShapeNet.url
from experiments.pointcloud.run import knn_edge_index
from halfhop.gcn import GCN
from halfhop.halfhop import HalfHop
from halfhop.reproducibility import set_seed

# proxy.py points ShapeNet.url at the Hugging Face mirror.
ARMS = ('gcn', 'hh-random', 'hh-geom')


class Segmenter(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, depth, dropout):
        super().__init__()
        self.encoder = GCN(
            in_channels, hidden, num_classes, dropout=dropout, depth=depth
        )

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index)
        mask = getattr(data, 'slow_node_mask', None)
        if mask is not None:
            x = x[~mask]
        return x


class GraphBuild:
    """kNN graph, coordinate features, optional per-shape Half-Hop."""

    def __init__(self, k: int, arm: str, budget: float):
        self.k = k
        self.arm = arm
        self.budget = budget

    def __call__(self, data):
        normals = data.x
        data.x = data.pos.clone()
        data.edge_index = knn_edge_index(data.pos, self.k)
        n = F.normalize(normals, dim=-1)
        n = torch.nan_to_num(n, nan=0.0)
        src, dst = data.edge_index
        data.edge_score = 1.0 - (n[src] * n[dst]).sum(dim=-1).abs().clamp(max=1.0)
        if self.arm == 'hh-random':
            data = HalfHop(selection='edge', budget=self.budget, inplace=False)(data)
        elif self.arm == 'hh-geom':
            data = HalfHop(selection='score', budget=self.budget, inplace=False)(data)
        return data


def instance_miou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ious = []
    for c in target.unique().tolist():
        gt = target == c
        pr = pred == c
        union = int((gt | pr).sum())
        if union == 0:
            continue
        ious.append((gt & pr).sum().float() / union)
    if not ious:
        return pred.new_zeros(())
    return torch.stack(ious).mean()


def original_batch(data):
    mask = getattr(data, 'slow_node_mask', None)
    if mask is None:
        return data.batch
    return data.batch[~mask]


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    correct = total = 0
    miou_sum = 0.0
    graphs = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=-1)
        target = data.y.view(-1)
        batch = original_batch(data)
        correct += int((pred == target).sum())
        total += target.numel()
        for g in batch.unique().tolist():
            m = batch == g
            miou_sum += float(instance_miou(pred[m], target[m]))
            graphs += 1
    return correct / max(total, 1), miou_sum / max(graphs, 1)


def train_one_epoch(loader, model, optimizer, device) -> float:
    model.train()
    total = 0.0
    seen = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        n = data.y.numel()
        total += loss.item() * n
        seen += n
    return total / max(seen, 1)


def load_dataset(root, category, split, transform):
    return ShapeNet(
        root,
        categories=[category],
        include_normals=True,
        split=split,
        transform=transform,
    )


def run_key(row: dict) -> tuple:
    return (
        row['stage'], row['model'], row['category'], row['seed'],
        row['budget'], row['k'], row['depth'], row['hidden'],
        row['dropout'], row['epochs'], row['lr'], row['batch_size'],
    )


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            done.add(run_key(json.loads(line)))
    return done


def train_arm(args, category, arm, seed, device):
    set_seed(seed)
    transform = GraphBuild(args.k, arm, args.budget)
    train_ds = load_dataset(args.data_dir, category, 'train', transform)
    val_ds = load_dataset(args.data_dir, category, 'val', transform)
    test_ds = load_dataset(args.data_dir, category, 'test', transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = Segmenter(
        in_channels=3,
        hidden=args.hidden,
        num_classes=50,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_state = None
    for epoch in tqdm(range(1, args.epochs + 1), desc=f'{category} {arm} s{seed}'):
        loss = train_one_epoch(train_loader, model, optimizer, device)
        val_acc, val_miou = evaluate(val_loader, model, device)
        if val_miou > best_val:
            best_val = val_miou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(
                f'{category} {arm} seed {seed} epoch {epoch:03d} '
                f'loss={loss:.4f} val_acc={val_acc:.4f} val_miou={val_miou:.4f}'
            )

    model.load_state_dict(best_state)
    test_acc, test_miou = evaluate(test_loader, model, device)
    return best_val, test_acc, test_miou


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', default='Chair')
    parser.add_argument('--model', default='gcn', choices=ARMS)
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run gcn, hh-random, and hh-geom for this category and seed.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--budget', type=float, default=0.1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-dir', default='data/ShapeNet')
    parser.add_argument(
        '--output', default='experiments/results/segmentation.jsonl'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available() else
        ('cpu' if args.device == 'auto' else args.device)
    )
    arms = ARMS if args.sweep else (args.model,)
    out = Path(args.output)
    done = load_done(out)
    proto = {
        'stage': 'seg',
        'category': args.category,
        'seed': args.seed,
        'budget': args.budget,
        'k': args.k,
        'depth': args.depth,
        'hidden': args.hidden,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
    }
    print(f'Device: {device}')
    for arm in arms:
        row = {**proto, 'model': arm}
        if run_key(row) in done:
            print(f'skip {args.category} {arm} seed {args.seed}')
            continue
        best_val, test_acc, test_miou = train_arm(
            args, args.category, arm, args.seed, device
        )
        row.update(
            best_val_miou=round(best_val, 6),
            test_acc=round(test_acc, 6),
            test_miou=round(test_miou, 6),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(
            f'FINAL {args.category} {arm} seed {args.seed} '
            f'val_miou={best_val:.4f} test_acc={test_acc:.4f} test_miou={test_miou:.4f}'
        )


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
