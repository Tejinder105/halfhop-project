"""Concurrent Multi-GPU Runner for ShapeNet Part Segmentation.

Automatically detects available GPUs (e.g. 2x T4 on Kaggle), pre-downloads
and verifies datasets to prevent extraction race conditions, dynamically balances
tasks across GPUs from a shared queue, streams live epoch logs in real-time,
and safely appends results to experiments/results/segmentation.jsonl.

Usage:
    python -m experiments.segmentation.run_2gpu
    python -m experiments.segmentation.run_2gpu --categories Airplane,Chair,Lamp,Table --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

ARMS = ("gcn", "hh-random", "hh-geom")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        default="Airplane,Chair,Lamp,Table",
        help="Comma-separated category names.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seed integers.",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--data-dir", default="data/ShapeNet")
    parser.add_argument(
        "--output", default="experiments/results/segmentation.jsonl"
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="Comma-separated GPU IDs to use (e.g. '0,1') or 'auto'.",
    )
    return parser.parse_args()


def get_gpu_list(arg_gpus: str) -> list[int]:
    if arg_gpus != "auto":
        return [int(g.strip()) for g in arg_gpus.split(",") if g.strip()]
    if not torch.cuda.is_available():
        return []
    count = torch.cuda.device_count()
    return list(range(count))


def load_completed_keys(output_file: Path) -> set[tuple]:
    if not output_file.exists():
        return set()
    completed = set()
    for line in output_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            completed.add((d["category"], int(d["seed"]), d["model"]))
        except Exception:
            continue
    return completed


def prepare_datasets(categories: list[str], data_dir: str):
    """Download and extract datasets sequentially in main process to prevent race conditions."""
    print("=" * 60)
    print("Pre-checking & preparing ShapeNet datasets (prevents multi-GPU unzip collision)...")
    print("=" * 60)
    for cat in categories:
        print(f"Verifying dataset for {cat}...")
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "experiments.segmentation.run",
            "--category",
            cat,
            "--prepare-only",
            "--data-dir",
            data_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Warning on preparing {cat}: {res.stderr[-300:]}")
        else:
            print(f"✓ {cat} dataset ready.")
    print("All datasets prepared and verified!\n")


def stream_reader(pipe, prefix: str):
    for line in iter(pipe.readline, ""):
        line_s = line.strip()
        if line_s:
            # Print significant lines: epochs, loss, final results, or progress
            if (
                "epoch" in line_s
                or "FINAL" in line_s
                or "Building graphs" in line_s
                or "skip" in line_s
                or "Error" in line_s
            ):
                print(f"[{prefix}] {line_s}", flush=True)
    pipe.close()


def run_arm(args, category: str, seed: int, arm: str, gpu_id: int | None) -> int:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.segmentation.run",
        "--category",
        category,
        "--model",
        arm,
        "--seed",
        str(seed),
        "--depth",
        str(args.depth),
        "--budget",
        str(args.budget),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden",
        str(args.hidden),
        "--dropout",
        str(args.dropout),
        "--k",
        str(args.k),
        "--data-dir",
        args.data_dir,
        "--output",
        args.output,
        "--device",
        device_str,
    ]

    prefix = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    t = threading.Thread(target=stream_reader, args=(proc.stdout, prefix))
    t.daemon = True
    t.start()
    returncode = proc.wait()
    t.join(timeout=2.0)
    return returncode


def main():
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    gpus = get_gpu_list(args.gpus)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Available GPUs: {gpus if gpus else 'None (using CPU)'}")
    print(f"Categories: {categories}")
    print(f"Seeds: {seeds}")
    print(f"Arms: {list(ARMS)}")
    print(f"Depth: {args.depth} | Budget: {args.budget} | Epochs: {args.epochs}")
    print(f"Results output: {out_path}\n")

    # Step 1: Pre-download & extract datasets to guarantee no race conditions
    prepare_datasets(categories, args.data_dir)

    # Step 2: Build task queue of incomplete arms
    completed = load_completed_keys(out_path)
    all_tasks = []
    for cat in categories:
        for seed in seeds:
            for arm in ARMS:
                if (cat, seed, arm) in completed:
                    continue
                all_tasks.append((cat, seed, arm))

    print(f"Total tasks: {len(categories) * len(seeds) * len(ARMS)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining tasks to run: {len(all_tasks)}\n")

    if not all_tasks:
        print("All tasks already completed! Nothing to run.")
        return

    # If only 1 GPU or CPU, run sequentially
    if len(gpus) <= 1:
        gpu_id = gpus[0] if gpus else None
        print(f"Running sequentially on GPU {gpu_id}...")
        for idx, (cat, seed, arm) in enumerate(all_tasks, 1):
            print(f"\n--- [{idx}/{len(all_tasks)}] Starting {cat} seed={seed} arm={arm} on GPU {gpu_id} ---", flush=True)
            code = run_arm(args, cat, seed, arm, gpu_id)
            if code != 0:
                print(f"Task {cat} seed={seed} arm={arm} returned non-zero code {code}")
        print("\nAll tasks finished!")
        return

    # Step 3: Multi-GPU dynamic queue dispatch
    task_queue = queue.Queue()
    for task in all_tasks:
        task_queue.put(task)

    total_tasks = len(all_tasks)
    finished_count = 0
    lock = threading.Lock()

    def worker(gpu_id: int):
        nonlocal finished_count
        while True:
            try:
                cat, seed, arm = task_queue.get_nowait()
            except queue.Empty:
                break

            with lock:
                curr_task = total_tasks - task_queue.qsize()
                print(
                    f"\n>>> [GPU {gpu_id}] [{curr_task}/{total_tasks}] Starting {cat} seed={seed} arm={arm} <<<",
                    flush=True,
                )

            code = run_arm(args, cat, seed, arm, gpu_id)

            with lock:
                finished_count += 1
                if code == 0:
                    print(
                        f"✓ [GPU {gpu_id}] Finished {cat} seed={seed} arm={arm} ({finished_count}/{total_tasks} done)",
                        flush=True,
                    )
                else:
                    print(
                        f"✗ [GPU {gpu_id}] Failed {cat} seed={seed} arm={arm} (exit code {code})",
                        flush=True,
                    )

            task_queue.task_done()

    threads = []
    print(f"Spawning {len(gpus)} worker threads for GPUs: {gpus}...\n")
    for gpu_id in gpus:
        t = threading.Thread(target=worker, args=(gpu_id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED ACROSS ALL GPUS!")
    print(f"Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

