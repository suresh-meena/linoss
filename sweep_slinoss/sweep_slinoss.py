"""Hyperparameter sweep runner for SLinOSS experiments only."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import shutil
import sys
import traceback

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

COMPLETION_MARKER = "_sweep_complete.json"
FAILURE_MARKER = "_sweep_failed.json"
REQUIRED_OUTPUT_FILES = (
    "steps.npy",
    "all_train_metric.npy",
    "all_val_metric.npy",
    "all_time.npy",
    "test_metric.npy",
)

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


def _completion_marker_path(output_dir: str) -> str:
    return os.path.join(output_dir, COMPLETION_MARKER)


def _failure_marker_path(output_dir: str) -> str:
    return os.path.join(output_dir, FAILURE_MARKER)


def _has_required_output_files(output_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(output_dir, filename)) for filename in REQUIRED_OUTPUT_FILES)


def _is_completed_run(output_dir: str) -> bool:
    marker_path = _completion_marker_path(output_dir)
    if not os.path.isfile(marker_path):
        return False
    if not _has_required_output_files(output_dir):
        return False

    try:
        with open(marker_path, "r", encoding="utf-8") as file:
            marker = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return marker.get("status") == "complete"


def _write_completion_marker(
    output_dir: str,
    *,
    dataset_name: str,
    seed: int,
    run_args: dict,
) -> None:
    marker_path = _completion_marker_path(output_dir)
    marker = {
        "status": "complete",
        "dataset_name": dataset_name,
        "seed": seed,
        "num_steps": int(run_args["num_steps"]),
        "print_steps": int(run_args["print_steps"]),
    }
    with open(marker_path, "w", encoding="utf-8") as file:
        json.dump(marker, file, indent=2, sort_keys=True)


def _sanitize_filename_component(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_"}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "default"


def _safe_error_message(exc: BaseException, *, limit: int = 2000) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) <= limit:
        return message
    return f"{message[:limit]}..."


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


def _classify_failure(exc: BaseException) -> str:
    message = str(exc).lower()
    if _is_cuda_oom_error(exc):
        return "cuda_oom"
    if _is_nonfinite_training_error(exc):
        return "nonfinite"
    if "cuda_error_invalid_value" in message or "invalid value" in message and "cuda" in message:
        return "cuda_invalid_value"
    if "illegal memory access" in message and "cuda" in message:
        return "cuda_illegal_memory_access"
    if "cuda" in message:
        return "cuda_runtime_error"
    if "cutlass" in message:
        return "cutlass_error"
    return exc.__class__.__name__.lower()


def _cleanup_after_failure(path: str) -> None:
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


def _write_failure_marker(
    output_dir: str,
    *,
    dataset_name: str,
    seed: int,
    run_args: dict,
    failure_kind: str,
    exc: BaseException,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    marker_path = _failure_marker_path(output_dir)
    marker = {
        "status": "failed",
        "dataset_name": dataset_name,
        "seed": seed,
        "failure_kind": failure_kind,
        "error_type": exc.__class__.__name__,
        "error_message": _safe_error_message(exc),
        "num_steps": int(run_args["num_steps"]),
        "print_steps": int(run_args["print_steps"]),
        "model_args": run_args.get("model_args", {}),
        "traceback": traceback.format_exc(limit=25),
    }
    with open(marker_path, "w", encoding="utf-8") as file:
        json.dump(marker, file, indent=2, sort_keys=True)


def _append_failure_log(failure_log_path: str | None, record: dict[str, object]) -> None:
    if failure_log_path is None:
        return
    parent = os.path.dirname(failure_log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(failure_log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True))
        file.write("\n")


def _default_failure_log_path(base_configs: dict[str, dict], progress_desc: str) -> str:
    first_config = next(iter(base_configs.values()))
    output_parent_dir = str(first_config.get("output_parent_dir", ""))
    filename = f"_sweep_failures_{_sanitize_filename_component(progress_desc)}.jsonl"
    return os.path.join(output_parent_dir, "outputs", "SLinOSS", filename)


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
    progress_queue=None,
    failure_log_path: str | None = None,
) -> None:
    def emit_progress(value: float = 1.0) -> None:
        if progress is not None:
            progress.update(value)
        if progress_queue is not None:
            progress_queue.put({"type": "update", "value": value})

    combinations = _iter_grid(HYPERPARAM_GRID)

    if not show_progress and progress_queue is None:
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

    if failure_log_path is None:
        failure_log_path = _default_failure_log_path(base_configs, progress_desc)

    if tqdm_lock is not None and tqdm is not None:
        tqdm.set_lock(tqdm_lock)

    progress = None
    if progress_queue is not None:
        show_progress = False
        progress_queue.put({"type": "total", "value": total_runs, "worker": progress_desc})
    elif show_progress and tqdm is not None:
        progress = tqdm(
            total=total_runs,
            desc=progress_desc,
            position=progress_position,
            dynamic_ncols=True,
            leave=True,
        )

    skipped_runs = 0
    completed_runs = 0
    failed_runs: dict[str, int] = {}

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
                reported_run_progress = 0.0

                def report_run_progress(completed_steps: int, total_steps: int) -> None:
                    nonlocal reported_run_progress
                    if total_steps <= 0:
                        return
                    run_progress = min(max(completed_steps / total_steps, 0.0), 1.0)
                    delta = run_progress - reported_run_progress
                    if delta <= 0:
                        return
                    reported_run_progress = run_progress
                    emit_progress(delta)

                target_dir = _expected_output_path(seed, dataset_name, run_args)
                run_already_completed = _is_completed_run(target_dir)

                if skip_existing and run_already_completed:
                    skipped_runs += 1
                    emit_progress(1.0)
                    continue
                overwrite_existing = os.path.isdir(target_dir) and (
                    not skip_existing or not run_already_completed
                )
                try:
                    run_fn(
                        seed=seed,
                        overwrite_output_dir=overwrite_existing,
                        auto_confirm_output_dir=False,
                        verbose=not (show_progress or progress_queue is not None),
                        progress_callback=report_run_progress,
                        **run_args,
                    )
                    _write_completion_marker(
                        target_dir,
                        dataset_name=dataset_name,
                        seed=seed,
                        run_args=run_args,
                    )
                    completed_runs += 1
                except Exception as exc:
                    failure_kind = _classify_failure(exc)
                    failed_runs[failure_kind] = failed_runs.get(failure_kind, 0) + 1
                    _cleanup_after_failure(target_dir)
                    _write_failure_marker(
                        target_dir,
                        dataset_name=dataset_name,
                        seed=seed,
                        run_args=run_args,
                        failure_kind=failure_kind,
                        exc=exc,
                    )
                    _append_failure_log(
                        failure_log_path,
                        {
                            "dataset_name": dataset_name,
                            "seed": seed,
                            "combo_index": combo_idx,
                            "params": params,
                            "failure_kind": failure_kind,
                            "error_type": exc.__class__.__name__,
                            "error_message": _safe_error_message(exc),
                            "target_dir": target_dir,
                        },
                    )
                    if not show_progress and progress_queue is None:
                        print(
                            "Skipping failed run "
                            f"(dataset={dataset_name}, seed={seed}, combo={combo_idx}, "
                            f"failure_kind={failure_kind}): {_safe_error_message(exc, limit=300)}"
                        )
                    emit_progress(1.0 - reported_run_progress)
                    continue
                emit_progress(1.0 - reported_run_progress)

    if progress is not None:
        progress.close()

    if not show_progress and progress_queue is None:
        total_failed_runs = sum(failed_runs.values())
        failure_summary = ", ".join(
            f"{name}={count}" for name, count in sorted(failed_runs.items())
        ) or "none"
        print(
            f"{progress_desc} summary: completed={completed_runs}, skipped={skipped_runs}, "
            f"failed={total_failed_runs}, failure_breakdown={failure_summary}, "
            f"total_runs={completed_runs + skipped_runs + total_failed_runs}"
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
