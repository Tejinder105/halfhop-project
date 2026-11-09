"""Step 1: Reproduction diagnostic for Wisconsin (target: 72.55%) and Chameleon (target: 62.64%).

This script tests 5 hypotheses that could explain the gap:
  H1: alpha convention — revert to paper's alpha=0.9937 on Wisconsin
  H2: epochs — paper may use more than 200 (test 500, 1000)
  H3: slow-node resampling — HalfHop is called each forward pass (p<1 stochastic);
      p=0.714 on Wisconsin means ~28.6% of nodes are NOT half-hopped each step.
  H4: Chameleon duplicate-node contamination — paper used contaminated geom-gcn splits
  H5: lr schedule — test with cosine annealing or step decay

Outputs a JSON file per dataset with per-split results for each hypothesis.

Usage (Kaggle):
    python experiments/reproduce_wisconsin_chameleon.py --dataset wisconsin --device cuda:0
    python experiments/reproduce_wisconsin_chameleon.py --dataset chameleon --device cuda:1
"""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import copy
import json
import time

import torch
import torch.nn.functional as F

from halfhop.datasets import load_dataset
from halfhop.splits import get_split
from halfhop.hh_gcn import HHGCN
from halfhop.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Canonical paper configs (Table 5, SOURCE_NOTES.md §10)
# ---------------------------------------------------------------------------
PAPER_CONFIGS = {
    "wisconsin": {
        "lr": 0.0105, "wd": 0.0002, "depth": 3, "hidden": 128,
        "dropout": 0.6612, "alpha": 0.9937, "p": 0.7140,
    },
    "chameleon": {
        "lr": 0.0318, "wd": 0.0057, "depth": 2, "hidden": 128,
        "dropout": 0.8040, "alpha": 0.0510, "p": 0.9986,
    },
}


def _move_data(data, device):
    """Shallow-copy data and move all tensors to device."""
    data = copy.copy(data)
    for key in data.keys():
        val = getattr(data, key, None)
        if isinstance(val, torch.Tensor):
            setattr(data, key, val.to(device))
    return data


def _forward(model, data):
    return model(data)


def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = _forward(model, data)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, train_mask, val_mask, test_mask):
    model.eval()
    out = _forward(model, data)
    pred = out.argmax(dim=-1)
    t_acc = (pred[train_mask] == data.y[train_mask]).float().mean().item()
    v_acc = (pred[val_mask]   == data.y[val_mask]).float().mean().item()
    te_acc = (pred[test_mask]  == data.y[test_mask]).float().mean().item()
    return t_acc, v_acc, te_acc


def run_split(model, data, train_mask, val_mask, test_mask,
              epochs, device, scheduler=None):
    """Train for `epochs` epochs and return (best_val, test_at_best_val)."""
    model = model.to(device)
    data = _move_data(data, device)
    train_mask = train_mask.to(device)
    val_mask    = val_mask.to(device)
    test_mask   = test_mask.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=model._lr, weight_decay=model._wd
    )
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        sched = None

    best_val = 0.0
    best_test = 0.0
    for epoch in range(1, epochs + 1):
        train_epoch(model, data, optimizer, train_mask)
        _, val_acc, test_acc = evaluate(model, data, train_mask, val_mask, test_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if sched is not None:
            sched.step()

    return best_val, best_test


def build_model(cfg, num_features, num_classes):
    """Construct HHGCN with paper config; attach lr/wd as attributes for run_split."""
    model = HHGCN(
        in_channels=num_features,
        hidden_channels=cfg["hidden"],
        out_channels=num_classes,
        alpha=cfg["alpha"],
        p=cfg["p"],
        dropout=cfg["dropout"],
        depth=cfg["depth"],
        inplace=False,
    )
    model._lr = cfg["lr"]
    model._wd = cfg["wd"]
    return model


def run_experiment(name, cfg_override, dataset_name, dataset, data,
                   num_splits, epochs, device, scheduler=None):
    """Run all 10 splits for one experimental condition."""
    base_cfg = PAPER_CONFIGS[dataset_name].copy()
    base_cfg.update(cfg_override)

    num_features = dataset.num_features
    num_classes  = dataset.num_classes

    scores = []
    for split_id in range(num_splits):
        set_seed(42 + split_id)
        train_mask, val_mask, test_mask = get_split(data, split_id)
        model = build_model(base_cfg, num_features, num_classes)
        _, test_acc = run_split(
            model, data, train_mask, val_mask, test_mask,
            epochs=epochs, device=device, scheduler=scheduler,
        )
        scores.append(test_acc)
        print(f"  [{name}] split {split_id:02d}: {test_acc*100:.2f}%")

    mean = sum(scores) / len(scores) * 100
    std  = (sum((s*100 - mean)**2 for s in scores) / len(scores)) ** 0.5
    print(f"  [{name}] MEAN={mean:.2f} ± {std:.2f}%\n")
    return {"scores": [s * 100 for s in scores], "mean": mean, "std": std,
            "config": base_cfg, "epochs": epochs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wisconsin", "chameleon"],
                        default="wisconsin")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dataset_name = args.dataset
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Dataset: {dataset_name}  Device: {device}")

    dataset, data = load_dataset(dataset_name)
    num_splits = data.train_mask.size(1) if data.train_mask.dim() == 2 else 10

    results = {"dataset": dataset_name, "experiments": {}}

    # -------------------------------------------------------------------
    # E0  Baseline — paper alpha but only 200 epochs (reproduce our gap)
    # -------------------------------------------------------------------
    print("\n=== E0: Paper alpha, 200 epochs ===")
    results["experiments"]["E0_paper_alpha_200ep"] = run_experiment(
        "E0", {}, dataset_name, dataset, data, num_splits, epochs=200, device=device
    )

    # -------------------------------------------------------------------
    # E1  Corrected alpha (what we used previously)
    # -------------------------------------------------------------------
    if dataset_name == "wisconsin":
        corrected_alpha = 0.0063
    else:
        corrected_alpha = PAPER_CONFIGS[dataset_name]["alpha"]  # unchanged for chameleon

    print("\n=== E1: 'Corrected' alpha (prev. run), 200 epochs ===")
    results["experiments"]["E1_corrected_alpha_200ep"] = run_experiment(
        "E1", {"alpha": corrected_alpha}, dataset_name, dataset, data,
        num_splits, epochs=200, device=device
    )

    # -------------------------------------------------------------------
    # E2  Paper alpha, 500 epochs
    # -------------------------------------------------------------------
    print("\n=== E2: Paper alpha, 500 epochs ===")
    results["experiments"]["E2_paper_alpha_500ep"] = run_experiment(
        "E2", {}, dataset_name, dataset, data, num_splits, epochs=500, device=device
    )

    # -------------------------------------------------------------------
    # E3  Paper alpha, 1000 epochs
    # -------------------------------------------------------------------
    print("\n=== E3: Paper alpha, 1000 epochs ===")
    results["experiments"]["E3_paper_alpha_1000ep"] = run_experiment(
        "E3", {}, dataset_name, dataset, data, num_splits, epochs=1000, device=device
    )

    # -------------------------------------------------------------------
    # E4  Paper alpha, 500 epochs, cosine LR annealing
    # -------------------------------------------------------------------
    print("\n=== E4: Paper alpha, 500 epochs, cosine LR ===")
    results["experiments"]["E4_paper_alpha_500ep_cosine"] = run_experiment(
        "E4", {}, dataset_name, dataset, data,
        num_splits, epochs=500, device=device, scheduler="cosine"
    )

    # -------------------------------------------------------------------
    # E5  Paper alpha, 1000 epochs, cosine LR annealing
    # -------------------------------------------------------------------
    print("\n=== E5: Paper alpha, 1000 epochs, cosine LR ===")
    results["experiments"]["E5_paper_alpha_1000ep_cosine"] = run_experiment(
        "E5", {}, dataset_name, dataset, data,
        num_splits, epochs=1000, device=device, scheduler="cosine"
    )

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    out_dir = "experiments/results/reproduction"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_reproduction.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"SUMMARY: {dataset_name.upper()}")
    if dataset_name == "wisconsin":
        print("Paper target: 72.55% (Table 2, HH-GCN)")
    else:
        print("Paper target: 62.64% (Table 2, HH-GCN)")
    print("=" * 60)
    for exp_name, exp_data in results["experiments"].items():
        print(f"  {exp_name:35s}: {exp_data['mean']:5.2f} ± {exp_data['std']:.2f}%")


if __name__ == "__main__":
    main()

