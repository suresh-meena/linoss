"""Dry-run validator for the SLinOSS hyperparameter grid."""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import sys
import time
import traceback
import multiprocessing as mp
import concurrent.futures

import torch

# Ensure repository root is on sys.path when running from sweep_slinoss folder directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_dir.torch_datasets import create_torch_dataset
from run_experiment import _build_run_args
from sweep_slinoss.sweep_slinoss import (
    DEFAULT_DATASETS,
    HYPERPARAM_GRID,
    _apply_sweep_params_to_config,
    _classify_failure,
    _iter_grid,
    _resolve_seeds,
    _safe_error_message,
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


def _run_single_dry_run_worker(q: mp.Queue, kwargs: dict, gpu_id: int) -> None:
    # Set GPU isolation BEFORE importing torch in the worker process
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        from train_torch import _key_to_seed, _set_torch_seed
        from models.generate_torch_model import create_torch_model
        from models.SLinOSS import ensure_slinoss_cuda_ready
        from data_dir.torch_datasets import create_torch_dataset
        import torch
        from torch.nn import functional as F

        ensure_slinoss_cuda_ready()
        device = torch.device("cuda")

        dataset = create_torch_dataset(
            kwargs["data_dir"],
            kwargs["dataset_name"],
            kwargs["use_presplit"],
            kwargs["include_time"],
            kwargs["T"],
            key=kwargs["datasetkey"],
        )

        _set_torch_seed(kwargs["model_seed"])
        model = create_torch_model(
            "SLinOSS",
            dataset.data_dim,
            dataset.label_dim,
            model_args=kwargs["model_args"],
        ).to(device)
        model.train()

        batch_size = kwargs["batch_size"]
        if batch_size > len(dataset.train):
            raise ValueError(f"Batch size {batch_size} larger than training dataset size {len(dataset.train)}.")

        x_all, lengths_all, labels_all = dataset.train.tensors
        x = x_all[:batch_size].contiguous().to(device=device, non_blocking=True)
        lengths = lengths_all[:batch_size].contiguous().to(device=device, non_blocking=True)
        labels = labels_all[:batch_size].contiguous().to(device=device, non_blocking=True)

        logits = model(x, lengths)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.cuda.synchronize(device=device)

        q.put({"status": "passed"})
    except Exception as exc:
        q.put({
            "status": "failed",
            "failure_kind": _classify_failure(exc),
            "error_type": exc.__class__.__name__,
            "error_message": _safe_error_message(exc),
            "traceback": traceback.format_exc(limit=25),
        })


def _execute_isolated_dry_run(kwargs: dict, gpu_id: int) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_single_dry_run_worker, args=(q, kwargs, gpu_id))
    p.start()

    res = None
    while p.is_alive():
        try:
            res = q.get(timeout=0.1)
            break
        except queue.Empty:
            continue

    p.join()

    if res is None:
        try:
            res = q.get(block=False)
        except queue.Empty:
            res = {
                "status": "failed",
                "failure_kind": "worker_crash",
                "error_type": "Crash",
                "error_message": f"Worker crashed with exitcode {p.exitcode}. Likely a C-level Segfault or severe OOM.",
                "traceback": "",
            }
    return res


def run_dry_run_grid(
    *,
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    report_path: str,
    gpu_ids: list[int],
    show_progress: bool,
) -> None:
    combinations = _iter_grid(HYPERPARAM_GRID)
    base_configs: dict[str, dict] = {}
    total_runs = 0
    for dataset_name in datasets:
        config_path = os.path.join(experiment_folder, "SLinOSS", f"{dataset_name}.json")
        with open(config_path, "r", encoding="utf-8") as file:
            base_configs[dataset_name] = json.load(file)
        effective_seeds = _resolve_seeds(
            config=base_configs[dataset_name],
            seeds_per_config=seeds_per_config,
            seeds=seeds,
        )
        total_runs += len(combinations) * len(effective_seeds)

    if os.path.exists(report_path):
        os.remove(report_path)
    summary_path = os.path.splitext(report_path)[0] + "_summary.json"

    gpu_queue = queue.Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    dry_run_tasks = []
    for dataset_name in datasets:
        base_config = base_configs[dataset_name]
        for combo_idx, params in enumerate(combinations, start=1):
            run_config = _apply_sweep_params_to_config(base_config, params)
            run_args, _ = _build_run_args("SLinOSS", dataset_name, run_config)
            effective_seeds = _resolve_seeds(
                config=run_config,
                seeds_per_config=seeds_per_config,
                seeds=seeds,
            )

            for seed in effective_seeds:
                gen = torch.Generator().manual_seed(seed)
                datasetkey = torch.randint(0, 2**32, (1,), generator=gen).item()
                modelkey = torch.randint(0, 2**32, (1,), generator=gen).item()

                kwargs = {
                    "data_dir": run_args["data_dir"],
                    "dataset_name": dataset_name,
                    "use_presplit": run_args["use_presplit"],
                    "include_time": bool(run_args["include_time"]),
                    "T": run_args["T"],
                    "datasetkey": datasetkey,
                    "model_args": run_args["model_args"],
                    "batch_size": int(run_args["batch_size"]),
                    "model_seed": modelkey,
                }
                
                dry_run_tasks.append({
                    "kwargs": kwargs,
                    "dataset_name": dataset_name,
                    "seed": seed,
                    "combo_index": combo_idx,
                    "params": params,
                })

    passed = 0
    failed = 0
    failure_breakdown: dict[str, int] = {}

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

    def _worker_task(task):
        gid = gpu_queue.get()
        start_time = time.time()
        try:
            res = _execute_isolated_dry_run(task["kwargs"], gid)
            elapsed_sec = round(time.time() - start_time, 4)
            
            record = {
                "dataset_name": task["dataset_name"],
                "seed": task["seed"],
                "combo_index": task["combo_index"],
                "params": task["params"],
                "model_args": task["kwargs"]["model_args"],
                "elapsed_sec": elapsed_sec,
            }
            record.update(res)
            return record
        finally:
            gpu_queue.put(gid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        future_to_task = {executor.submit(_worker_task, task): task for task in dry_run_tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            record = future.result()
            
            if record["status"] == "passed":
                passed += 1
            else:
                failed += 1
                failure_kind = record["failure_kind"]
                failure_breakdown[failure_kind] = failure_breakdown.get(failure_kind, 0) + 1
                if not show_progress:
                    print(
                        f"\nDry run failed (dataset={record['dataset_name']}, seed={record['seed']}, "
                        f"combo={record['combo_index']}, failure_kind={failure_kind}): {record['error_message']}"
                    )

            _write_jsonl(report_path, record)
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

    print(
        f"\nDry-run summary: passed={passed}, failed={failed}, total_runs={total_runs}, "
        f"failure_breakdown={failure_breakdown or 'none'}"
    )
    print(f"Detailed report: {report_path}")
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
    )
