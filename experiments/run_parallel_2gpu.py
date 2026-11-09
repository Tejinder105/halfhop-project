"""Parallel Multi-GPU runner for Kaggle (2x T4) and multi-GPU nodes.

Distributes datasets across available GPUs (e.g. GPU 0 and GPU 1) so that
both GPUs run simultaneously at 100% utilization, cutting overall wall-clock
time in half.

Usage on Kaggle:
    python -m experiments.run_parallel_2gpu --datasets texas,cornell,wisconsin,chameleon --epochs 200
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
import torch


def worker(gpu_id: int, dataset: str, epochs: int, seed: int, models: str | None):
    print(f"\n[GPU {gpu_id}] >>> Starting {dataset} (epochs={epochs}, seed={seed})")
    cmd = [
        sys.executable,
        "-m",
        "experiments.adahop_ablation",
        "--dataset",
        dataset,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--device",
        f"cuda:{gpu_id}",
    ]
    if models:
        cmd.extend(["--models", models])

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[GPU {gpu_id}] !!! Error running {dataset} (exit code {res.returncode})")
    else:
        print(f"[GPU {gpu_id}] <<< Finished {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU parallel experiment runner")
    parser.add_argument(
        "--datasets",
        type=str,
        default="texas,cornell,wisconsin,chameleon",
        help="Comma-separated list of datasets to run",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated models (default: all ablation models)",
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"=== Detected {num_gpus} CUDA device(s) ===")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        print("No datasets provided.")
        return

    if num_gpus < 2:
        print("\nSingle GPU or CPU detected. Running sequentially on default device.")
        for ds in datasets:
            worker(0 if num_gpus == 1 else "cpu", ds, args.epochs, args.seed, args.models)
        return

    print(f"\n2+ GPUs detected! Distributing {len(datasets)} dataset(s) across {min(num_gpus, 2)} GPUs in parallel.\n")

    # Queue-based multi-processing pool across GPU 0 and GPU 1
    # Pair jobs with alternating GPU ids: GPU 0, GPU 1, GPU 0, GPU 1...
    with mp.Pool(processes=min(num_gpus, 2)) as pool:
        tasks = []
        for idx, ds in enumerate(datasets):
            gpu_id = idx % 2
            tasks.append((gpu_id, ds, args.epochs, args.seed, args.models))
        pool.starmap(worker, tasks)

    print("\n=== All parallel multi-GPU runs completed! ===")


if __name__ == "__main__":
    main()

