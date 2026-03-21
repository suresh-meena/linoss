"""Dry-run validator for the SLinOSS hyperparameter grid."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

import jax.random as jr
import torch
from torch.nn import functional as F

# Ensure repository root is on sys.path when running from sweep_slinoss folder directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_dir.torch_datasets import create_torch_dataset
from models.generate_torch_model import create_torch_model
from models.SLinOSS import ensure_slinoss_cuda_ready
from run_experiment import _build_run_args
from train_torch import _key_to_seed, _set_torch_seed
from sweep_slinoss.sweep_slinoss import (
    DEFAULT_DATASETS,
    HYPERPARAM_GRID,
    _apply_sweep_params_to_config,
    _classify_failure,
    _cleanup_after_failure,
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


def _take_batch(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_size > len(dataset.train):
        raise ValueError(
            f"Batch size larger than training dataset size: batch_size={batch_size}, "
            f"train_size={len(dataset.train)}."
        )
    x_all, lengths_all, labels_all = dataset.train.tensors
    return (
        x_all[:batch_size].contiguous(),
        lengths_all[:batch_size].contiguous(),
        labels_all[:batch_size].contiguous(),
    )


def _run_single_dry_run(
    *,
    dataset,
    model_args: dict,
    batch_size: int,
    model_seed,
    device: torch.device,
) -> None:
    _set_torch_seed(_key_to_seed(model_seed))
    model = create_torch_model(
        "SLinOSS",
        dataset.data_dim,
        dataset.label_dim,
        model_args=model_args,
    ).to(device)
    model.train()

    x, lengths, labels = _take_batch(dataset, batch_size)
    x = x.to(device=device, non_blocking=True)
    lengths = lengths.to(device=device, non_blocking=True)
    labels = labels.to(device=device, non_blocking=True)

    logits = model(x, lengths)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    torch.cuda.synchronize(device=device)

    del loss
    del logits
    del x
    del lengths
    del labels
    del model


def run_dry_run_grid(
    *,
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    report_path: str,
    show_progress: bool,
) -> None:
    ensure_slinoss_cuda_ready()
    device = torch.device("cuda")

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

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=total_runs, desc="SLinOSS Dry Run", dynamic_ncols=True)

    dataset_cache: dict[tuple[str, bool, int], object] = {}
    passed = 0
    failed = 0
    failure_breakdown: dict[str, int] = {}

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
                start_time = time.time()
                key = jr.PRNGKey(seed)
                datasetkey, modelkey, _, _ = jr.split(key, 4)
                include_time = bool(run_args["include_time"])
                cache_key = (dataset_name, include_time, int(seed))

                try:
                    dataset = dataset_cache.get(cache_key)
                    if dataset is None:
                        dataset = create_torch_dataset(
                            run_args["data_dir"],
                            dataset_name,
                            run_args["use_presplit"],
                            include_time,
                            run_args["T"],
                            key=datasetkey,
                        )
                        dataset_cache[cache_key] = dataset

                    _run_single_dry_run(
                        dataset=dataset,
                        model_args=run_args["model_args"],
                        batch_size=int(run_args["batch_size"]),
                        model_seed=modelkey,
                        device=device,
                    )
                    record = {
                        "status": "passed",
                        "dataset_name": dataset_name,
                        "seed": seed,
                        "combo_index": combo_idx,
                        "params": params,
                        "model_args": run_args["model_args"],
                        "elapsed_sec": round(time.time() - start_time, 4),
                    }
                    passed += 1
                except Exception as exc:
                    failure_kind = _classify_failure(exc)
                    failure_breakdown[failure_kind] = failure_breakdown.get(failure_kind, 0) + 1
                    failed += 1
                    _cleanup_after_failure("")
                    record = {
                        "status": "failed",
                        "dataset_name": dataset_name,
                        "seed": seed,
                        "combo_index": combo_idx,
                        "params": params,
                        "model_args": run_args["model_args"],
                        "failure_kind": failure_kind,
                        "error_type": exc.__class__.__name__,
                        "error_message": _safe_error_message(exc),
                        "traceback": traceback.format_exc(limit=25),
                        "elapsed_sec": round(time.time() - start_time, 4),
                    }
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                _write_jsonl(report_path, record)
                if progress is not None:
                    progress.update(1)
                elif not show_progress:
                    if record["status"] == "failed":
                        print(
                            "Dry run failed "
                            f"(dataset={dataset_name}, seed={seed}, combo={combo_idx}, "
                            f"failure_kind={record['failure_kind']}): {record['error_message']}"
                        )

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
        f"Dry-run summary: passed={passed}, failed={failed}, total_runs={total_runs}, "
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
        show_progress=args.show_progress,
    )
