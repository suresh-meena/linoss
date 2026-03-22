"""Multi-GPU parameter sweep using functional programming (FP) principles.
ThreadPoolExecutor + isolated subprocesses processing task groups."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import NamedTuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from run_experiment import _build_run_args
from sweep_slinoss.sweep_slinoss import (
    DEFAULT_DATASETS,
    HYPERPARAM_GRID,
    _iter_grid,
    _apply_sweep_params_to_config,
    _resolve_seeds,
    _expected_output_path,
    _is_completed_run,
    _validate_slinoss_sweep_params,
    _write_failure_marker,
    _failure_marker_path,
    PreflightValidationError,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class TaskGroup(NamedTuple):
    dataset: str
    include_time: bool
    seed: int
    tasks: list[dict]

def get_pending_task_groups(
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    skip_existing: bool,
) -> list[TaskGroup]:
    combinations = _iter_grid(HYPERPARAM_GRID)
    groups = defaultdict(list)

    for dataset_name in datasets:
        config_path = os.path.join(experiment_folder, "SLinOSS", f"{dataset_name}.json")
        if not os.path.exists(config_path):
            print(f"Warning: Config not found for {dataset_name}. Skipping.")
            continue
            
        with open(config_path, "r", encoding="utf-8") as file:
            base_config = json.load(file)

        for params in combinations:
            try:
                _validate_slinoss_sweep_params(params)
            except PreflightValidationError:
                continue

            run_config = _apply_sweep_params_to_config(base_config, params)
            run_args, _ = _build_run_args("SLinOSS", dataset_name, run_config)
            effective_seeds = _resolve_seeds(
                config=run_config, seeds_per_config=seeds_per_config, seeds=seeds
            )

            for seed in effective_seeds:
                target_dir = _expected_output_path(seed, dataset_name, run_args)
                
                # Check for an abandoned active marker from a previous segfault
                active_marker = os.path.join(target_dir, "_active_run.json")
                if os.path.exists(active_marker) and not _is_completed_run(target_dir):
                    _write_failure_marker(
                        target_dir,
                        dataset_name=dataset_name,
                        seed=seed,
                        run_args=run_args,
                        failure_kind="worker_crash",
                        exc=RuntimeError("Worker process crashed (segfault/OOM) while running this task."),
                    )
                    os.remove(active_marker)

                if skip_existing:
                    if _is_completed_run(target_dir):
                        continue
                    if os.path.exists(_failure_marker_path(target_dir)):
                        continue

                key = (dataset_name, params["include_time"], seed)
                groups[key].append({
                    "params": params,
                    "target_dir": target_dir
                })

    return [
        TaskGroup(dataset=ds, include_time=inc, seed=sd, tasks=ts) 
        for (ds, inc, sd), ts in groups.items()
    ]

def _execute_task_group(
    group: TaskGroup,
    gpu_queue: queue.Queue,
    experiment_folder: str,
    skip_existing: bool,
) -> int:
    gpu_id = gpu_queue.get()
    try:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "dataset": group.dataset,
                "include_time": group.include_time,
                "seed": group.seed,
                "experiment_folder": experiment_folder,
                "skip_existing": skip_existing,
                "tasks": group.tasks
            }, f)
            payload_path = f.name
        
        cmd = [
            sys.executable,
            os.path.join(repo_root, "sweep_slinoss", "dataset_worker.py"),
            "--payload", payload_path
        ]

        log_dir = os.path.join(experiment_folder, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"worker_gpu{gpu_id}_{group.dataset}_seed{group.seed}.log")

        with open(log_file, "w") as f_out:
            result = subprocess.run(cmd, env=env, stdout=f_out, stderr=subprocess.STDOUT, text=True)
        
        success = result.returncode == 0
        os.remove(payload_path)
        
        if not success:
            print(f"\n[GPU {gpu_id}] Group Crashed: Dataset={group.dataset}, Time={group.include_time}, Seed={group.seed}")
            print(f"Check log for details: {log_file}")
            # Try to print the last few lines of the log for immediate feedback
            try:
                with open(log_file, "r") as f_log:
                    lines = f_log.readlines()
                    print("".join(lines[-10:]))
            except Exception:
                pass
        else:
            # Even if the worker finished successfully, check if any internal runs failed
            try:
                with open(log_file, "r") as f_log:
                    content = f_log.read()
                    if "[Worker] Run failed" in content:
                        print(f"\n[GPU {gpu_id}] Group Finished with some failed runs: Dataset={group.dataset}, Seed={group.seed}. See log: {log_file}")
            except Exception:
                pass
            
        return len(group.tasks)
    finally:
        gpu_queue.put(gpu_id)

def run_concurrent_sweep(
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    skip_existing: bool,
    gpu_ids: list[int],
    show_progress: bool,
):
    print("Initializing multi-GPU grouped dataset execution...")
    
    total_processed = 0
    iteration = 1
    
    while True:
        pending_groups = get_pending_task_groups(
            experiment_folder, datasets, seeds_per_config, seeds, skip_existing
        )
        
        if not pending_groups:
            print("\nNo pending tasks remaining.")
            break
            
        total_tasks_this_round = sum(len(g.tasks) for g in pending_groups)
        print(f"\n--- Iteration {iteration} ---")
        print(f"Found {len(pending_groups)} task groups ({total_tasks_this_round} total tasks).")

        gpu_queue = queue.Queue()
        for gid in gpu_ids:
            gpu_queue.put(gid)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = [
                executor.submit(
                    _execute_task_group, group, gpu_queue, experiment_folder, skip_existing
                ) for group in pending_groups
            ]

            if show_progress and tqdm is not None:
                pbar = tqdm(
                    total=total_tasks_this_round, 
                    desc="Sweep", 
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    leave=True,
                    smoothing=0.1
                )
                
                # Smooth progress polling
                all_target_dirs = [task["target_dir"] for group in pending_groups for task in group.tasks]
                last_completed = 0
                
                while True:
                    done_count = sum(1 for f in futures if f.done())
                    
                    current_completed = 0
                    for target_dir in all_target_dirs:
                        if _is_completed_run(target_dir) or os.path.exists(_failure_marker_path(target_dir)):
                            current_completed += 1
                            
                    if current_completed > last_completed:
                        pbar.update(current_completed - last_completed)
                        total_processed += (current_completed - last_completed)
                        last_completed = current_completed
                        
                    if done_count == len(futures):
                        for future in futures:
                            future.result() 
                        break
                        
                    time.sleep(2.0)
                    
                pbar.close()
            else:
                for future in concurrent.futures.as_completed(futures):
                    tasks_in_group = future.result()
                    total_processed += tasks_in_group
                    print(f"Group finished ({tasks_in_group} configs attempted). Total processed: {total_processed}")
                    
        iteration += 1

    print(f"\nSweep complete! Total configs processed across all iterations: {total_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the SLinOSS sweep grouped by dataset for extreme efficiency.",
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="experiment_configs/repeats",
        help="Directory that contains the SLinOSS dataset config JSON files.",
    )
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to include in the sweep.",
    )
    parser.add_argument(
        "--seeds_per_config",
        type=int,
        default=None,
        help="Optional cap on number of seeds per hyperparameter combination.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated list of seed values to use for all configs. Overrides JSON seeds.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose output directory already exists.",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=[0, 1],
        metavar="GPU_ID",
        help="List of GPU ids to use for concurrent execution.",
    )
    parser.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bar when tqdm is installed.",
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=0,
        help="Ignored. Always runs with 0 workers per FP dataloader architecture.",
    )
    args = parser.parse_args()

    user_seeds = None
    if args.seeds is not None:
        user_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_concurrent_sweep(
        experiment_folder=args.experiment_folder,
        datasets=args.dataset_name,
        seeds_per_config=args.seeds_per_config,
        seeds=user_seeds,
        skip_existing=args.skip_existing,
        gpu_ids=args.gpu_ids,
        show_progress=args.show_progress,
    )
