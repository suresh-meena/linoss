"""Dry-run validator for the SLinOSS hyperparameter grid."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import NamedTuple
import json
import os
import queue
import subprocess
import sys
import time
import tempfile
import concurrent.futures

import torch

# Torch-only entrypoint: prevent accidental JAX GPU preallocation if JAX is
# imported transitively by a dependency stack.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Ensure repository root is on sys.path when running from sweep_slinoss folder directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from run_experiment import _build_run_args
from sweep_slinoss.sweep_slinoss import (
    DEFAULT_DATASETS,
    HYPERPARAM_GRID,
    _apply_sweep_params_to_config,
    _iter_grid,
    _resolve_seeds,
    _validate_slinoss_sweep_params,
    PreflightValidationError,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _write_jsonl(path: str, record: dict[str, object]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True))
        file.write("\n")


def _write_summary(path: str, summary: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)


def _write_json(path: str, payload: dict[str, object]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


class DryRunTaskGroup(NamedTuple):
    dataset: str
    include_time: bool
    seed: int
    data_dir: str
    use_presplit: bool
    T: float
    batch_size: int
    datasetkey: int
    model_seed: int
    tasks: list[dict[str, object]]


def _load_jsonl(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _build_dry_run_task_groups(
    *,
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    max_tasks_per_worker: int,
) -> tuple[list[DryRunTaskGroup], int]:
    combinations = _iter_grid(HYPERPARAM_GRID)
    base_configs: dict[str, dict] = {}
    groups: dict[tuple[str, bool, int], list[dict[str, object]]] = defaultdict(list)
    total_runs = 0

    for dataset_name in datasets:
        config_path = os.path.join(experiment_folder, "SLinOSS", f"{dataset_name}.json")
        with open(config_path, "r", encoding="utf-8") as file:
            base_config = json.load(file)
        base_configs[dataset_name] = base_config

        for combo_idx, params in enumerate(combinations, start=1):
            try:
                _validate_slinoss_sweep_params(params)
            except PreflightValidationError:
                continue

            run_config = _apply_sweep_params_to_config(base_config, params)
            run_args, _ = _build_run_args("SLinOSS", dataset_name, run_config)
            effective_seeds = _resolve_seeds(
                config=run_config,
                seeds_per_config=seeds_per_config,
                seeds=seeds,
            )

            for seed in effective_seeds:
                key = (dataset_name, bool(run_args["include_time"]), seed)
                groups[key].append(
                    {
                        "combo_index": combo_idx,
                        "params": params,
                        "model_args": run_args["model_args"],
                    }
                )
                total_runs += 1

    if max_tasks_per_worker < 1:
        raise ValueError(
            f"max_tasks_per_worker must be >= 1. Got {max_tasks_per_worker}."
        )

    task_groups: list[DryRunTaskGroup] = []
    for (dataset_name, include_time, seed), tasks in groups.items():
        base_config = base_configs[dataset_name]
        base_run_args, _ = _build_run_args("SLinOSS", dataset_name, base_config)
        gen = torch.Generator().manual_seed(seed)
        datasetkey = torch.randint(0, 2**32, (1,), generator=gen).item()
        model_seed = torch.randint(0, 2**32, (1,), generator=gen).item()

        for start in range(0, len(tasks), max_tasks_per_worker):
            task_groups.append(
                DryRunTaskGroup(
                    dataset=dataset_name,
                    include_time=include_time,
                    seed=seed,
                    data_dir=str(base_config["data_dir"]),
                    use_presplit=bool(base_run_args["use_presplit"]),
                    T=float(base_config["T"]),
                    batch_size=int(base_config["batch_size"]),
                    datasetkey=datasetkey,
                    model_seed=model_seed,
                    tasks=tasks[start : start + max_tasks_per_worker],
                )
            )

    return task_groups, total_runs


def _execute_dry_run_group(group: DryRunTaskGroup, gpu_id: int) -> list[dict[str, object]]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("JAX_PLATFORMS", "cpu")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as payload_file:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as result_file:
            payload = {
                "dataset": group.dataset,
                "include_time": group.include_time,
                "seed": group.seed,
                "data_dir": group.data_dir,
                "use_presplit": group.use_presplit,
                "T": group.T,
                "batch_size": group.batch_size,
                "datasetkey": group.datasetkey,
                "model_seed": group.model_seed,
                "tasks": group.tasks,
                "result_path": result_file.name,
            }
            json.dump(payload, payload_file)
            payload_path = payload_file.name
            result_path = result_file.name

    try:
        cmd = [
            sys.executable,
            os.path.join(repo_root, "sweep_slinoss", "dry_run_dataset_worker.py"),
            "--payload",
            payload_path,
        ]
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        records = _load_jsonl(result_path)
        if result.returncode != 0 and not records:
            return [
                {
                    "dataset_name": group.dataset,
                    "seed": group.seed,
                    "combo_index": task["combo_index"],
                    "params": task["params"],
                    "model_args": task["model_args"],
                    "elapsed_sec": 0.0,
                    "status": "failed",
                    "failure_kind": "worker_crash",
                    "error_type": "Crash",
                    "error_message": (
                        f"Dry-run worker crashed with exitcode {result.returncode}."
                    ),
                    "traceback": result.stdout[-4000:],
                }
                for task in group.tasks
            ]
        return records
    finally:
        if os.path.exists(payload_path):
            os.remove(payload_path)
        if os.path.exists(result_path):
            os.remove(result_path)


def run_dry_run_grid(
    *,
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    report_path: str,
    gpu_ids: list[int],
    show_progress: bool,
    max_tasks_per_worker: int,
) -> None:
    task_groups, total_runs = _build_dry_run_task_groups(
        experiment_folder=experiment_folder,
        datasets=datasets,
        seeds_per_config=seeds_per_config,
        seeds=seeds,
        max_tasks_per_worker=max_tasks_per_worker,
    )

    if os.path.exists(report_path):
        os.remove(report_path)
    summary_path = os.path.splitext(report_path)[0] + "_summary.json"
    json_path = os.path.splitext(report_path)[0] + ".json"

    gpu_queue = queue.Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    passed = 0
    failed = 0
    failure_breakdown: dict[str, int] = {}
    all_records: list[dict[str, object]] = []

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(
            total=total_runs, 
            desc="Dry Run", 
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=True,
            smoothing=0.1
        )

    def _worker_task(group: DryRunTaskGroup):
        gid = gpu_queue.get()
        try:
            return _execute_dry_run_group(group, gid)
        finally:
            gpu_queue.put(gid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        future_to_group = {executor.submit(_worker_task, group): group for group in task_groups}
        for future in concurrent.futures.as_completed(future_to_group):
            records = future.result()
            for record in records:
                if record["status"] == "passed":
                    passed += 1
                else:
                    failed += 1
                    failure_kind = str(record["failure_kind"])
                    failure_breakdown[failure_kind] = failure_breakdown.get(failure_kind, 0) + 1
                    if not show_progress:
                        print(
                            f"\nDry run failed (dataset={record['dataset_name']}, seed={record['seed']}, "
                            f"combo={record['combo_index']}, failure_kind={failure_kind}): {record['error_message']}"
                        )

                _write_jsonl(report_path, record)
                all_records.append(record)
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    summary = {
        "report_path": report_path,
        "summary_path": summary_path,
        "datasets": datasets,
        "total_runs": total_runs,
        "passed": passed,
        "failed": failed,
        "failure_breakdown": failure_breakdown,
    }
    _write_summary(summary_path, summary)
    _write_json(
        json_path,
        {
            "report_path": report_path,
            "json_path": json_path,
            "summary_path": summary_path,
            "datasets": datasets,
            "total_runs": total_runs,
            "passed": passed,
            "failed": failed,
            "failure_breakdown": failure_breakdown,
            "records": all_records,
        },
    )

    print(
        f"\nDry-run summary: passed={passed}, failed={failed}, total_runs={total_runs}, "
        f"failure_breakdown={failure_breakdown or 'none'}"
    )
    print(f"Detailed report: {report_path}")
    print(f"JSON run report: {json_path}")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a dry forward/backward validation over the full SLinOSS sweep grid.",
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
        help="One or more dataset names to validate.",
    )
    parser.add_argument(
        "--seeds_per_config",
        type=int,
        default=1,
        help="Optional cap on number of seeds per hyperparameter combination.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed override. Overrides JSON seeds and seeds_per_config.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="outputs/SLinOSS/dry_run_report.jsonl",
        help="Path to the JSONL report file.",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=[0, 1],
        help="List of GPU ids to use for concurrent dry-run execution.",
    )
    parser.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a tqdm progress bar when tqdm is installed.",
    )
    parser.add_argument(
        "--max_tasks_per_worker",
        type=int,
        default=1,
        help=(
            "Maximum number of dry-run configs to execute in a single worker subprocess. "
            "Lower values reduce GPU VRAM growth from long-lived workers."
        ),
    )
    args = parser.parse_args()

    user_seeds = None
    if args.seeds is not None:
        user_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_dry_run_grid(
        experiment_folder=args.experiment_folder,
        datasets=args.dataset_name,
        seeds_per_config=args.seeds_per_config,
        seeds=user_seeds,
        report_path=args.report_path,
        gpu_ids=args.gpu_ids,
        show_progress=args.show_progress,
        max_tasks_per_worker=args.max_tasks_per_worker,
    )
