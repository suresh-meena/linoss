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


def _clear_retryable_failure_marker(target_dir: str) -> None:
    failure_marker = _failure_marker_path(target_dir)
    if os.path.exists(failure_marker):
        os.remove(failure_marker)

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
                failure_marker = _failure_marker_path(target_dir)
                
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

                # Failed runs should be retried, not treated as completed.
                # Remove the stale marker so this iteration starts from a clean state.
                if os.path.exists(failure_marker):
                    _clear_retryable_failure_marker(target_dir)

                key = (dataset_name, params["include_time"], seed)
                groups[key].append({
                    "params": params,
                    "target_dir": target_dir
                })

    return [
        TaskGroup(dataset=ds, include_time=inc, seed=sd, tasks=ts) 
        for (ds, inc, sd), ts in groups.items()
    ]

def _log_file_path(experiment_folder: str, gpu_id: int, group: TaskGroup) -> str:
    log_dir = os.path.join(experiment_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"worker_gpu{gpu_id}_{group.dataset}_seed{group.seed}.log")


def _write_json(path: str, payload: dict[str, object]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _start_persistent_worker(gpu_id: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        "-u",
        os.path.join(repo_root, "sweep_slinoss", "persistent_dataset_worker.py"),
    ]
    process = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ready_line = process.stdout.readline()
    if not ready_line:
        raise RuntimeError(f"Persistent worker for GPU {gpu_id} exited before ready signal.")
    ready = json.loads(ready_line)
    if ready.get("status") != "ready":
        raise RuntimeError(f"Persistent worker for GPU {gpu_id} returned invalid ready signal: {ready}")
    return process


def _shutdown_persistent_worker(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
        process.stdin.flush()
        process.wait(timeout=5)
    except Exception:
        process.kill()
        process.wait(timeout=5)


def _execute_task_group_via_worker(
    process: subprocess.Popen,
    group: TaskGroup,
    gpu_id: int,
    experiment_folder: str,
    skip_existing: bool,
) -> tuple[dict[str, object], str]:
    payload = {
        "dataset": group.dataset,
        "include_time": group.include_time,
        "seed": group.seed,
        "experiment_folder": experiment_folder,
        "skip_existing": skip_existing,
        "tasks": group.tasks,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as payload_file:
        json.dump(payload, payload_file)
        payload_path = payload_file.name

    log_file = _log_file_path(experiment_folder, gpu_id, group)
    with open(log_file, "w", encoding="utf-8"):
        pass

    try:
        if process.poll() is not None:
            raise RuntimeError(f"Persistent worker for GPU {gpu_id} is not running.")
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(
            json.dumps(
                {
                    "type": "run",
                    "payload_path": payload_path,
                    "log_path": log_file,
                }
            )
            + "\n"
        )
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"Persistent worker for GPU {gpu_id} exited while running a group.")
        return json.loads(response_line), log_file
    finally:
        if os.path.exists(payload_path):
            os.remove(payload_path)


def _gpu_worker_loop(
    gpu_id: int,
    group_queue: queue.Queue,
    experiment_folder: str,
    skip_existing: bool,
    result_queue: queue.Queue | None = None,
) -> int:
    worker = _start_persistent_worker(gpu_id)
    completed_groups = 0
    try:
        while True:
            try:
                group = group_queue.get_nowait()
            except queue.Empty:
                break

            try:
                response, log_file = _execute_task_group_via_worker(
                    worker,
                    group,
                    gpu_id,
                    experiment_folder,
                    skip_existing,
                )
            except Exception:
                _shutdown_persistent_worker(worker)
                worker = _start_persistent_worker(gpu_id)
                print(
                    f"\n[GPU {gpu_id}] Group Crashed: Dataset={group.dataset}, Time={group.include_time}, Seed={group.seed}"
                )
                print(f"Check log for details: { _log_file_path(experiment_folder, gpu_id, group) }")
                try:
                    with open(_log_file_path(experiment_folder, gpu_id, group), "r", encoding="utf-8") as f_log:
                        lines = f_log.readlines()
                        print("".join(lines[-10:]))
                except Exception:
                    pass
                continue

            if response.get("status") != "ok":
                print(
                    f"\n[GPU {gpu_id}] Group Failed: Dataset={group.dataset}, Time={group.include_time}, Seed={group.seed}"
                )
                print(f"Check log for details: {log_file}")
                try:
                    with open(log_file, "r", encoding="utf-8") as f_log:
                        lines = f_log.readlines()
                        print("".join(lines[-10:]))
                except Exception:
                    pass
                continue

            if result_queue is not None:
                records = response.get("records", [])
                for record in records:
                    record["worker_log_path"] = log_file
                result_queue.put(records)

            if response.get("had_failures"):
                print(
                    f"\n[GPU {gpu_id}] Group Finished with some failed runs: Dataset={group.dataset}, Seed={group.seed}. See log: {log_file}"
                )
            completed_groups += 1
    finally:
        _shutdown_persistent_worker(worker)

    return completed_groups

def run_concurrent_sweep(
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    skip_existing: bool,
    gpu_ids: list[int],
    show_progress: bool,
    report_path: str,
):
    print("Initializing multi-GPU grouped dataset execution...")
    
    total_processed = 0
    iteration = 1
    all_records: list[dict[str, object]] = []
    
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

        group_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()
        for group in pending_groups:
            group_queue.put(group)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = [
                executor.submit(
                    _gpu_worker_loop,
                    gpu_id,
                    group_queue,
                    experiment_folder,
                    skip_existing,
                    result_queue,
                ) for gpu_id in gpu_ids
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

            while True:
                done_count = sum(1 for f in futures if f.done())

                while True:
                    try:
                        records = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    all_records.extend(records)
                    delta = len(records)
                    total_processed += delta
                    if show_progress and tqdm is not None:
                        pbar.update(delta)

                if done_count == len(futures):
                    for future in futures:
                        future.result()
                    while True:
                        try:
                            records = result_queue.get_nowait()
                        except queue.Empty:
                            break
                        all_records.extend(records)
                        delta = len(records)
                        total_processed += delta
                        if show_progress and tqdm is not None:
                            pbar.update(delta)
                    break

                time.sleep(2.0)

            if show_progress and tqdm is not None:
                pbar.close()
                    
        iteration += 1

    completed = sum(1 for record in all_records if record.get("status") == "completed")
    failed = sum(1 for record in all_records if record.get("status") == "failed")
    skipped = sum(1 for record in all_records if record.get("status") == "skipped_existing")
    failure_breakdown: dict[str, int] = {}
    for record in all_records:
        failure_kind = record.get("failure_kind")
        if failure_kind is None:
            continue
        name = str(failure_kind)
        failure_breakdown[name] = failure_breakdown.get(name, 0) + 1

    _write_json(
        report_path,
        {
            "report_path": report_path,
            "datasets": datasets,
            "total_records": len(all_records),
            "completed": completed,
            "failed": failed,
            "skipped_existing": skipped,
            "failure_breakdown": failure_breakdown,
            "records": all_records,
        },
    )

    print(f"\nSweep complete! Total configs processed across all iterations: {total_processed}")
    print(f"JSON run report: {report_path}")

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
        "--report_path",
        type=str,
        default="outputs/SLinOSS/sweep_run_report.json",
        help="Path to the aggregated JSON report for all sweep runs.",
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
        report_path=args.report_path,
    )
