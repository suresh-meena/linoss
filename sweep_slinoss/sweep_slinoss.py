"""Hyperparameter sweep runner for SLinOSS experiments only."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import shutil
import sys

# Ensure repository root is on sys.path when running from sweep_slinoss folder directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from run_experiment import _build_run_args
from train_torch import _build_output_dir

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


HYPERPARAM_GRID = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "hidden_dimension": [16, 64, 128],
    "ssm_dimension": [16, 64, 256],
    "num_ssm_blocks": [2, 4, 6],
    "include_time": [True, False],
}

HEAD_DIMS_BY_HIDDEN_DIM = {
    16: [16],
    64: [32],
    128: [64],
}

DEFAULT_DATASETS = [
    "EigenWorms",
    "EthanolConcentration",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]

def _iter_grid(grid: dict[str, list]) -> list[dict[str, object]]:
    keys = tuple(grid.keys())
    values = tuple(grid[k] for k in keys)
    combinations: list[dict[str, object]] = []
    for combo in itertools.product(*values):
        base_params = dict(zip(keys, combo))
        hidden_dim = int(base_params["hidden_dimension"])
        for head_dim in HEAD_DIMS_BY_HIDDEN_DIM[hidden_dim]:
            params = dict(base_params)
            params["head_dim"] = head_dim
            combinations.append(params)
    return combinations


def _apply_sweep_params_to_config(base_config: dict, params: dict[str, object]) -> dict:
    run_config = dict(base_config)
    run_config.update(
        {
            "lr": params["learning_rate"],
            "d_model": params["hidden_dimension"],
            "d_head": params["head_dim"],
            "d_state": params["ssm_dimension"],
            "n_layers": params["num_ssm_blocks"],
            "time": params["include_time"],
        }
    )
    return run_config


def _expected_output_path(seed: int, dataset_name: str, run_args: dict) -> str:
    output_parent_dir = os.path.join(
        run_args["output_parent_dir"],
        "outputs",
        "SLinOSS",
        dataset_name,
    )
    output_dir = _build_output_dir(
        seed=seed,
        T=run_args["T"],
        include_time=run_args["include_time"],
        num_steps=run_args["num_steps"],
        lr=run_args["lr"],
        model_name="SLinOSS",
        stepsize=run_args["stepsize"],
        logsig_depth=run_args["logsig_depth"],
        model_args=run_args["model_args"],
    )
    return os.path.join(output_parent_dir, output_dir)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or "outofmemoryerror" in message
        or "out of memory" in message and "cuda" in message
    )


def _is_nonfinite_training_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "non-finite" in message
        or "nan" in message
        or "inf" in message and "loss" in message
    )


def _cleanup_after_oom(path: str) -> None:
    try:
        import torch
    except Exception:
        torch = None

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _resolve_seeds(
    *,
    config: dict,
    seeds_per_config: int | None,
    seeds: list[int] | None,
) -> list[int]:
    if seeds is not None:
        return list(seeds)

    available = list(config.get("seeds", []))
    if not available:
        raise ValueError("No seeds found in config and no --seeds override provided.")

    num_seeds = config.get("num_seeds", 3)
    available = available[: int(num_seeds)]

    if seeds_per_config is not None:
        available = available[:seeds_per_config]

    return available


def run_sweep(
    *,
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None = None,
    skip_existing: bool,
    show_progress: bool = True,
    progress_desc: str = "Sweep",
    progress_position: int = 0,
    tqdm_lock=None,
) -> None:
    combinations = _iter_grid(HYPERPARAM_GRID)

    if not show_progress:
        print(f"Total hyperparameter combinations: {len(combinations)}")

    base_configs: dict[str, dict] = {}
    total_runs = 0
    for dataset_name in datasets:
        config_path = os.path.join(experiment_folder, "SLinOSS", f"{dataset_name}.json")
        with open(config_path, "r", encoding="utf-8") as file:
            base_config = json.load(file)
        base_configs[dataset_name] = base_config
        effective_seeds = _resolve_seeds(
            config=base_config,
            seeds_per_config=seeds_per_config,
            seeds=seeds,
        )
        total_runs += len(combinations) * len(effective_seeds)

    if tqdm_lock is not None and tqdm is not None:
        tqdm.set_lock(tqdm_lock)

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(
            total=total_runs,
            desc=progress_desc,
            position=progress_position,
            dynamic_ncols=True,
            leave=True,
        )

    skipped_runs = 0
    completed_runs = 0
    oom_failed_runs = 0
    nonfinite_failed_runs = 0

    for dataset_name in datasets:
        base_config = base_configs[dataset_name]

        for combo_idx, params in enumerate(combinations, start=1):
            run_config = _apply_sweep_params_to_config(base_config, params)

            run_args, run_fn = _build_run_args("SLinOSS", dataset_name, run_config)
            run_args["print_steps"] = max(int(run_args["print_steps"]), 500)
            effective_seeds = _resolve_seeds(
                config=run_config,
                seeds_per_config=seeds_per_config,
                seeds=seeds,
            )

            for seed in effective_seeds:
                target_dir = _expected_output_path(seed, dataset_name, run_args)
                if skip_existing and os.path.isdir(target_dir):
                    skipped_runs += 1
                    if progress is not None:
                        progress.update(1)
                    continue
                try:
                    run_fn(
                        seed=seed,
                        overwrite_output_dir=False,
                        auto_confirm_output_dir=False,
                        verbose=not show_progress,
                        **run_args,
                    )
                    completed_runs += 1
                except Exception as exc:
                    if _is_cuda_oom_error(exc):
                        oom_failed_runs += 1
                        if not show_progress:
                            print(f"Skipping run after CUDA OOM (dataset={dataset_name}, seed={seed})")
                        _cleanup_after_oom(target_dir)
                        if progress is not None:
                            progress.update(1)
                        continue
                    if _is_nonfinite_training_error(exc):
                        nonfinite_failed_runs += 1
                        if not show_progress:
                            print(f"Skipping run after non-finite values (dataset={dataset_name}, seed={seed})")
                        _cleanup_after_oom(target_dir)
                        if progress is not None:
                            progress.update(1)
                        continue
                    raise
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    if not show_progress:
        print(
            f"{progress_desc} summary: completed={completed_runs}, skipped={skipped_runs}, "
            f"oom_failed={oom_failed_runs}, "
            f"nonfinite_failed={nonfinite_failed_runs}, "
            f"total_runs={completed_runs + skipped_runs + oom_failed_runs + nonfinite_failed_runs}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an SLinOSS-only hyperparameter sweep over a fixed grid.",
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
        help="One or more dataset names to sweep.",
    )
    parser.add_argument(
        "--seeds_per_config",
        type=int,
        default=None,
        help="Optional cap on number of seeds per hyperparameter combination.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose output directory already exists.",
    )
    parser.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a tqdm progress bar when tqdm is installed.",
    )
    args = parser.parse_args()

    run_sweep(
        experiment_folder=args.experiment_folder,
        datasets=args.dataset_name,
        seeds_per_config=args.seeds_per_config,
        skip_existing=args.skip_existing,
        show_progress=args.show_progress,
    )