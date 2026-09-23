"""Run official BGRL, with optional Half-Hop, for Half-Hop Table 3.

Kaggle (2x T4, 15GB each): official BGRL is single-GPU. Pin one job per GPU:

    CUDA_VISIBLE_DEVICES=0 python run_bgrl_experiments.py --dataset computers --augmentation feat_edge --seed 0
    CUDA_VISIBLE_DEVICES=1 python run_bgrl_experiments.py --dataset computers --augmentation none --seed 0

Amazon Computers baseline (no Half-Hop, official config):

    python run_bgrl_experiments.py --dataset computers --augmentation feat_edge --seed 0 --device cuda

Use --low-memory for Half-Hop runs if a 15GB T4 OOMs.
Do not launch the 20-seed sweep until the single-run baseline looks reasonable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halfhop_bgrl.configs import AUGMENTATIONS  # noqa: E402
from halfhop_bgrl.train import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Thakoor BGRL + optional Half-Hop (Table 3)"
    )
    parser.add_argument("--dataset", type=str, default="computers")
    parser.add_argument(
        "--augmentation",
        type=str,
        default="feat_edge",
        choices=list(AUGMENTATIONS),
        help="none | feat_edge | hh | feat_edge_hh",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Linear-eval split seed. Default: same as --seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override official epoch count (Amazon Computers = 10000).",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="data")
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=None,
        help="Evaluate every N epochs. 0 = final eval only. Default: official 250.",
    )
    parser.add_argument("--eval-jobs", type=int, default=5)
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Sequential backward of the two BGRL terms. Same objective, less VRAM.",
    )
    parser.add_argument(
        "--no-debug-counts",
        action="store_true",
        help="Do not print first-epoch node/edge counts.",
    )
    parser.add_argument("--no-save-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        dataset=args.dataset,
        augmentation=args.augmentation,
        seed=args.seed,
        data_seed=args.data_seed,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        eval_epochs=args.eval_epochs,
        eval_jobs=args.eval_jobs,
        low_memory=args.low_memory,
        debug_counts=not args.no_debug_counts,
        save_embeddings=not args.no_save_embeddings,
    )
    print(
        f"test_accuracy={metrics['test_accuracy']:.4f} "
        f"saved to {metrics.get('device')} run dir"
    )


if __name__ == "__main__":
    main()
