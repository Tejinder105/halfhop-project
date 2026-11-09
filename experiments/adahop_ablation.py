"""AdaHop ablation experiment runner.

Runs the A1 component ablation (which component matters?) on a chosen dataset
across all 10 splits. Results are saved as JSON.

Usage:
    python experiments/adahop_ablation.py --dataset texas --epochs 200
    python experiments/adahop_ablation.py --dataset wisconsin --epochs 200
"""

import argparse
import json
import os
import torch
import numpy as np

from halfhop.datasets import load_dataset
from halfhop.splits import get_split
from halfhop.training import run_training_loop
from halfhop.reproducibility import set_seed
from halfhop.evaluation import format_results

from halfhop.gcn import GCN
from halfhop.hh_gcn import HHGCN
from halfhop.adahop import (
    AdaHopGCN,
    SoftAdaHopGCN,
    StatefulHHGCN,
    MultiDelayGCN,
)

RESULTS_DIR = "experiments/results/adahop"


def build_models(num_features, num_classes, cfg, model_names=None):
    """Build the full A1 ablation model set, or filter by model_names."""
    h = cfg["hidden"]
    d = cfg["depth"]
    dr = cfg["dropout"]
    alpha = cfg.get("alpha", 0.5)
    p = cfg.get("p", 1.0)

    all_models = {
        # Baseline
        "gcn": GCN(num_features, h, num_classes, dropout=dr, depth=d),
        # Standard Half-Hop (reference)
        "hh-gcn": HHGCN(num_features, h, num_classes,
                         alpha=alpha, p=p, dropout=dr, depth=d),
        # +A: learned alpha
        "adahop-gcn-A": AdaHopGCN(num_features, h, num_classes,
                                    alpha_hidden=32, entropy_reg=0.0,
                                    p=p, dropout=dr, depth=d),
        # +A with entropy reg
        "adahop-gcn-A-ent": AdaHopGCN(num_features, h, num_classes,
                                        alpha_hidden=32, entropy_reg=0.01,
                                        p=p, dropout=dr, depth=d),
        # +B: soft gate
        "soft-adahop-gcn-B": SoftAdaHopGCN(num_features, h, num_classes,
                                             alpha=alpha, gate_hidden=32,
                                             dropout=dr, depth=d),
        # +C: stateful GRU slow nodes
        "stateful-hh-gcn-C": StatefulHHGCN(num_features, h, num_classes,
                                             alpha=alpha, p=p,
                                             dropout=dr, depth=d),
        # +D: k=2 delay
        "multidelay-gcn-D-k2": MultiDelayGCN(num_features, h, num_classes,
                                               k=2, alpha=alpha, p=p,
                                               dropout=dr, depth=d),
        # +D: k=3 delay
        "multidelay-gcn-D-k3": MultiDelayGCN(num_features, h, num_classes,
                                               k=3, alpha=alpha, p=p,
                                               dropout=dr, depth=d),
    }
    if model_names:
        selected = {m.strip() for m in model_names}
        return {k: v for k, v in all_models.items() if k in selected}
    return all_models


def run_dataset(dataset_name, epochs, seed, lr=0.01, wd=5e-4,
                hidden=64, depth=2, dropout=0.5, device=None, model_names=None, tag=None):
    device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"[{dataset_name}] Using device: {device}")
    dataset, data = load_dataset(dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    cfg = dict(hidden=hidden, depth=depth, dropout=dropout)

    # Try to load HH config for alpha / p
    import yaml
    cfg_path = f"configs/{dataset_name}.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        hh_cfg = raw.get("hh-gcn", {})
        cfg["alpha"] = hh_cfg.get("alpha", 0.5)
        cfg["p"] = hh_cfg.get("p", 1.0)
        lr = hh_cfg.get("lr", lr)
        wd = hh_cfg.get("wd", wd)
        cfg["hidden"] = hh_cfg.get("hidden", hidden)
        cfg["depth"] = hh_cfg.get("depth", depth)
        cfg["dropout"] = hh_cfg.get("dropout", dropout)

    num_splits = data.train_mask.size(1) if data.train_mask.dim() == 2 else 1

    all_results = {}

    for split_id in range(num_splits):
        print(f"\n--- Split {split_id}/{num_splits - 1} ---")
        set_seed(seed + split_id)
        train_mask, val_mask, test_mask = get_split(data, split_id)

        models = build_models(num_features, num_classes, cfg, model_names=model_names)

        for model_name, model in models.items():
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=wd
            )
            best_val, best_test = run_training_loop(
                model=model,
                data=data,
                optimizer=optimizer,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                epochs=epochs,
                device=device,
            )
            if model_name not in all_results:
                all_results[model_name] = []
            all_results[model_name].append(best_test)
            print(f"  {model_name}: val={best_val:.4f}  test={best_test:.4f}")

    # Summarise
    print(f"\n{'='*60}")
    print(f"ABLATION RESULTS on {dataset_name} ({num_splits} splits)")
    print(f"{'='*60}")
    summary = {}
    for model_name, results in all_results.items():
        fmt = format_results(results)
        summary[model_name] = {
            "mean": float(np.mean(results) * 100),
            "std": float(np.std(results) * 100),
            "per_split": [float(r * 100) for r in results],
            "formatted": fmt,
        }
        print(f"  {model_name:30s}: {fmt}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_ablation{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_name, "epochs": epochs,
                   "seed": seed, "device": str(device),
                   "config": cfg, "results": summary}, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="AdaHop A1 ablation runner")
    parser.add_argument("--dataset", type=str, default="texas",
                        help="Dataset name (default: texas)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g. cuda:0, cuda:1, cpu)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: all)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for output filename")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")] if args.models else None
    run_dataset(args.dataset, args.epochs, args.seed,
                device=args.device, model_names=model_names, tag=args.tag)


if __name__ == "__main__":
    main()

