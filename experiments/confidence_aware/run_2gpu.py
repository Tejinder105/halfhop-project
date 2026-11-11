"""Concurrent Multi-GPU Runner for Confidence-Aware Adaptive Half-Hop (CAHH).

Enables full 2x T4 utilization on Kaggle:
  1. Sequentially trains the lightweight edge-confidence MLP for requested categories.
  2. Dynamically balances GNN benchmark jobs (gcn, hh-random, hh-geom, hh-cahh)
     across all available GPUs (e.g. GPU 0 and GPU 1) from a shared task queue.
  3. Pre-checks and prepares ShapeNet datasets sequentially to eliminate race conditions.
  4. Appends all results to experiments/results/cahh_segmentation.jsonl in real-time.

Usage:
    python -m experiments.confidence_aware.run_2gpu
    python -m experiments.confidence_aware.run_2gpu --categories Airplane,Chair,Lamp,Table --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

ARMS = ("gcn", "hh-random", "hh-geom", "hh-cahh")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        default="Chair,Airplane,Table,Lamp",
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
        "--output", default="experiments/results/cahh_segmentation.jsonl"
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
    return list(range(torch.cuda.device_count()))


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


def prepare_datasets_and_mlps(categories: list[str], args):
    """Sequential prep: ensures zero download collision and trains MLPs."""
    print("=" * 65, flush=True)
    print("PREPARATION STEP: Verifying ShapeNet datasets & training MLPs...", flush=True)
    print("=" * 65, flush=True)
    
    ckpt_dir = Path("experiments/results")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories:
        # Pre-verify dataset download
        print(f"[{cat}] Checking dataset presence / downloading...", flush=True)
        cmd_prep = [
            sys.executable, "-u", "-m", "experiments.segmentation.run",
            "--category", cat, "--prepare-only", "--data-dir", args.data_dir
        ]
        # Run without suppressing output so download/extract progress is visible
        p_prep = subprocess.run(cmd_prep, text=True)
        if p_prep.returncode != 0:
            print(f"[{cat}] Note/warning during dataset preparation (rc={p_prep.returncode})", flush=True)
        else:
            print(f"[{cat}] Dataset verified on disk.", flush=True)

        # Train confidence MLP if not already present
        mlp_path = ckpt_dir / f"cahh_mlp_{cat}.pt"
        if mlp_path.exists():
            print(f"[{cat}] Confidence MLP checkpoint already exists at {mlp_path}", flush=True)
        else:
            mlp_device = "cuda:0" if (args.gpus != "cpu" and torch.cuda.is_available()) else "cpu"
            print(f"[{cat}] Training Edge-Confidence MLP on {mlp_device} (batch_size=65536, high GPU throughput)...", flush=True)
            cmd_mlp = [
                sys.executable, "-u", "-m", "halfhop.confidence.train_gate",
                "--category", cat,
                "--data-dir", args.data_dir,
                "--k", str(args.k),
                "--epochs", "25",
                "--lr", "0.002",
                "--batch-size", "65536",
                "--dropout", "0.2",
                "--budget", str(args.budget),
                "--device", mlp_device,
                "--output", str(mlp_path),
            ]
            p_mlp = subprocess.run(cmd_mlp, text=True)
            if p_mlp.returncode != 0:
                print(f"[{cat}] Warning during MLP training: returncode={p_mlp.returncode}", flush=True)
            else:
                print(f"[{cat}] ✓ Edge-Confidence MLP trained and saved on {mlp_device}.", flush=True)
    print("=" * 65, flush=True)
    print("All MLPs and datasets ready! Starting multi-GPU dispatch...\n", flush=True)


def stream_reader(pipe, prefix: str):
    for line in iter(pipe.readline, ""):
        line_s = line.strip()
        if line_s:
            print(f"[{prefix}] {line_s}", flush=True)
    pipe.close()


def run_arm(args, category: str, seed: int, arm: str, gpu_id: int | None) -> int:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    mlp_path = f"experiments/results/cahh_mlp_{category}.pt"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.confidence_aware.train_cahh",
        "--category", category,
        "--model", arm,
        "--seed", str(seed),
        "--depth", str(args.depth),
        "--budget", str(args.budget),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--hidden", str(args.hidden),
        "--dropout", str(args.dropout),
        "--k", str(args.k),
        "--data-dir", args.data_dir,
        "--output", args.output,
        "--device", device_str,
    ]
    if arm == "hh-cahh":
        cmd.extend(["--mlp-ckpt", mlp_path])

    prefix = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    prefix = f"{prefix} | {category}-{arm}-s{seed}"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    t_out = threading.Thread(target=stream_reader, args=(proc.stdout, prefix))
    t_err = threading.Thread(target=stream_reader, args=(proc.stderr, f"{prefix}-ERR"))
    t_out.start()
    t_err.start()

    rc = proc.wait()
    t_out.join()
    t_err.join()
    return rc


def worker(gpu_id: int | None, task_queue: queue.Queue, args, completed_lock: threading.Lock):
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        category, seed, arm = task
        print(f"--> [GPU {gpu_id}] Picking up: {category} | {arm} | seed {seed}", flush=True)
        rc = run_arm(args, category, seed, arm, gpu_id)
        if rc == 0:
            print(f"<-- [GPU {gpu_id}] Finished: {category} | {arm} | seed {seed}", flush=True)
        else:
            print(f"[!] [GPU {gpu_id}] Failed (rc={rc}): {category} | {arm} | seed {seed}", flush=True)
        task_queue.task_done()


def main():
    args = parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    gpus = get_gpu_list(args.gpus)

    print("=" * 65)
    print("Multi-GPU Runner for CAHH (Confidence-Aware Adaptive Half-Hop)")
    print(f"Categories : {cats}")
    print(f"Seeds      : {seeds}")
    print(f"Models     : {list(ARMS)}")
    print(f"GPUs       : {gpus if gpus else 'CPU mode'}")
    print(f"Output     : {args.output}")
    print("=" * 65)

    prepare_datasets_and_mlps(cats, args)

    output_path = Path(args.output)
    completed = load_completed_keys(output_path)
    print(f"Found {len(completed)} already completed run(s).")

    task_queue = queue.Queue()
    pending = []
    for cat in cats:
        for seed in seeds:
            for arm in ARMS:
                key = (cat, seed, arm)
                if key in completed:
                    continue
                pending.append(key)
                task_queue.put(key)

    print(f"Total tasks queued: {len(pending)} (skipped {len(completed)})\n")
    if not pending:
        print("All requested tasks are already completed. Exiting.")
        return

    workers = []
    num_workers = max(1, len(gpus))
    completed_lock = threading.Lock()

    start_time = time.time()
    for i in range(num_workers):
        gpu_id = gpus[i] if gpus else None
        t = threading.Thread(
            target=worker,
            args=(gpu_id, task_queue, args, completed_lock),
            name=f"Worker-GPU-{gpu_id}",
        )
        t.start()
        workers.append(t)

    for t in workers:
        t.join()

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"All jobs completed in {elapsed/60:.2f} minutes!")
    print(f"Results appended to: {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()

