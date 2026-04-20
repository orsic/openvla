"""
run_libero_eval_parallel.py

Runs LIBERO evaluation in parallel across multiple GPUs by distributing tasks
across worker processes. Each worker loads a separate model instance on its
assigned GPU and evaluates a contiguous slice of the task suite.

EGL constraint: This script sets CUDA_VISIBLE_DEVICES=0,1 (all GPUs) and
MUJOCO_EGL_DEVICE_ID=0 for all workers so MuJoCo rendering always uses the
single EGL device (physical GPU 0). Model inference is steered to different
GPUs via `cuda_device_index` (passed to from_pretrained's device_map).

Usage:
    python experiments/robot/libero/run_libero_eval_parallel.py \\
        --model_family openvla \\
        --pretrained_checkpoint <CHECKPOINT_PATH> \\
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \\
        --num_gpus <N> \\
        --run_id_note <OPTIONAL_TAG>

    # Example: 2-GPU run for libero_spatial
    python experiments/robot/libero/run_libero_eval_parallel.py \\
        --model_family openvla \\
        --pretrained_checkpoint /hf_cache/hub/models--openvla--openvla-7b-finetuned-libero-spatial/... \\
        --task_suite_name libero_spatial \\
        --num_gpus 2
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def split_tasks(num_tasks: int, num_workers: int) -> list[tuple[int, int]]:
    """Split `num_tasks` into `num_workers` contiguous slices as evenly as possible."""
    base, remainder = divmod(num_tasks, num_workers)
    slices = []
    start = 0
    for i in range(num_workers):
        size = base + (1 if i < remainder else 0)
        slices.append((start, start + size))
        start += size
    return slices


def get_num_tasks(task_suite_name: str) -> int:
    """Return number of tasks in the given LIBERO suite."""
    suite_sizes = {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    if task_suite_name in suite_sizes:
        return suite_sizes[task_suite_name]
    # Fall back to importing LIBERO if not in the table.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    return bd[task_suite_name]().n_tasks


def aggregate_logs(log_paths: list[str], suite_name: str) -> None:
    """Parse worker log files and print combined success stats."""
    task_pattern = re.compile(r"Task: (.+)")
    success_pattern = re.compile(r"Success: (True|False)")

    all_task_results: dict[str, list[bool]] = {}
    current_task = None

    for log_path in sorted(log_paths):
        if not os.path.exists(log_path):
            print(f"  WARNING: log file not found: {log_path}")
            continue
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                m = task_pattern.match(line)
                if m:
                    current_task = m.group(1)
                    if current_task not in all_task_results:
                        all_task_results[current_task] = []
                m = success_pattern.match(line)
                if m and current_task is not None:
                    all_task_results[current_task].append(m.group(1) == "True")

    if not all_task_results:
        print("  No results parsed from log files.")
        return

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS — {suite_name}")
    print(f"{'='*60}")
    total_ep, total_suc = 0, 0
    for task, results in all_task_results.items():
        n, s = len(results), sum(results)
        total_ep += n
        total_suc += s
        print(f"  {task[:60]:<60}  {s}/{n} = {s/n*100:.1f}%")
    print(f"{'='*60}")
    print(f"  TOTAL: {total_suc}/{total_ep} = {total_suc/total_ep*100:.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Parallel LIBERO evaluation launcher")
    parser.add_argument("--model_family", default="openvla")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--center_crop", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of parallel GPU workers to launch")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated physical GPU indices to use (e.g. '0,1'). "
                             "Defaults to 0..num_gpus-1.")
    parser.add_argument("--run_id_note", default=None)
    parser.add_argument("--local_log_dir", default="./experiments/logs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Resolve GPU IDs.
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))
    num_workers = len(gpu_ids)

    # Resolve task split.
    num_tasks = get_num_tasks(args.task_suite_name)
    slices = split_tasks(num_tasks, num_workers)

    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    run_id_note = f"parallel{num_workers}gpu"
    if args.run_id_note:
        run_id_note += f"_{args.run_id_note}"

    print(f"Launching {num_workers} workers for {args.task_suite_name} ({num_tasks} tasks)")
    print(f"GPU IDs: {gpu_ids}")
    for i, (start, end) in enumerate(slices):
        print(f"  Worker {i} (GPU {gpu_ids[i]}): tasks {start}–{end-1}")
    print()

    # All workers share the same CUDA_VISIBLE_DEVICES to keep GPU 0 visible for EGL.
    all_gpu_ids_str = ",".join(str(g) for g in sorted(set(gpu_ids)))
    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = all_gpu_ids_str
    base_env["MUJOCO_EGL_DEVICE_ID"] = "0"

    # Build eval script path relative to repo root.
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_libero_eval.py",
    )

    # Launch worker subprocesses.
    procs = []
    log_paths = []
    for i, (start, end) in enumerate(slices):
        gpu_id = gpu_ids[i]
        # Each worker writes to its own log file.
        worker_note = f"worker{i}_gpu{gpu_id}_tasks{start}-{end}"
        if args.run_id_note:
            worker_note += f"_{args.run_id_note}"

        cmd = [
            sys.executable, script,
            "--model_family", args.model_family,
            "--pretrained_checkpoint", args.pretrained_checkpoint,
            "--task_suite_name", args.task_suite_name,
            "--num_trials_per_task", str(args.num_trials_per_task),
            "--center_crop", str(args.center_crop),
            "--task_id_start", str(start),
            "--task_id_end", str(end),
            "--cuda_device_index", str(gpu_id),
            "--run_id_note", worker_note,
            "--local_log_dir", args.local_log_dir,
            "--seed", str(args.seed),
        ]

        worker_log = os.path.join(args.local_log_dir, f"worker_{i}_stdout_{timestamp}.txt")
        log_paths.append(None)  # will be filled from the worker's own log file

        print(f"[Worker {i}] Starting: GPU {gpu_id}, tasks {start}–{end-1}")
        print(f"[Worker {i}] Stdout -> {worker_log}")
        f = open(worker_log, "w")
        proc = subprocess.Popen(cmd, env=base_env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((i, proc, f, worker_log, start, end, gpu_id))

    print(f"\nAll {num_workers} workers launched. Waiting for completion...\n")

    # Wait for all workers and report exit codes.
    results = []
    for i, proc, f, worker_log, start, end, gpu_id in procs:
        proc.wait()
        f.close()
        rc = proc.returncode
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[Worker {i}] GPU {gpu_id}, tasks {start}–{end-1}: {status} — stdout: {worker_log}")
        results.append((i, rc, worker_log))

    # Find the actual eval log files written by each worker and aggregate.
    # Workers name their logs: EVAL-{suite}-{model}-{timestamp}--tasks{S}-{E}--worker{i}...txt
    # The worker_note embedded in each log name lets us identify our workers unambiguously.
    print("\nAggregating results from worker log files...")
    os.makedirs(args.local_log_dir, exist_ok=True)
    worker_logs = []
    for i, (start, end) in enumerate(slices):
        gpu_id = gpu_ids[i]
        # The note each worker uses (must match how we pass --run_id_note above).
        worker_note = f"worker{i}_gpu{gpu_id}_tasks{start}-{end}"
        if args.run_id_note:
            worker_note += f"_{args.run_id_note}"
        matches = sorted(Path(args.local_log_dir).glob(f"EVAL-{args.task_suite_name}-*--*{worker_note}*.txt"))
        if matches:
            worker_logs.append(str(matches[-1]))  # use most recent if multiple
        else:
            print(f"  WARNING: no log file found for worker {i} (note={worker_note})")
    print(f"Found {len(worker_logs)} worker eval log(s).")
    aggregate_logs(worker_logs, args.task_suite_name)


if __name__ == "__main__":
    main()
