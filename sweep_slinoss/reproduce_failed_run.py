"""Reproduce a failed SLinOSS sweep run from its failure marker."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

# Torch-only entrypoint: prevent accidental JAX GPU preallocation if JAX is
# imported transitively by a dependency stack.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from run_experiment import _build_run_args
from train_torch import create_dataset_model_and_train_torch


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _infer_from_run_dir_name(run_dir_name: str) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    time_match = re.search(r"_time(True|False)_", run_dir_name)
    if time_match:
        inferred["time"] = _parse_bool(time_match.group(1))

    lr_match = re.search(r"_lr([^_]+)", run_dir_name)
    if lr_match:
        inferred["lr"] = float(lr_match.group(1))

    steps_match = re.search(r"_steps(\d+)", run_dir_name)
    if steps_match:
        inferred["num_steps"] = int(steps_match.group(1))

    return inferred


def _load_failure_marker(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _build_repro_config(
    *,
    failure_marker: dict[str, Any],
    failure_json_path: str,
    experiment_folder: str,
    include_time_override: bool | None,
    lr_override: float | None,
    output_parent_dir: str,
) -> tuple[dict[str, Any], int, str]:
    dataset_name = str(failure_marker["dataset_name"])
    seed = int(failure_marker["seed"])

    config_path = os.path.join(experiment_folder, "SLinOSS", f"{dataset_name}.json")
    with open(config_path, "r", encoding="utf-8") as file:
        base_config = json.load(file)

    parent_run_dir = os.path.basename(os.path.dirname(os.path.abspath(failure_json_path)))
    inferred = _infer_from_run_dir_name(parent_run_dir)

    include_time = include_time_override
    if include_time is None:
        include_time = inferred.get("time")
    if include_time is None:
        raise ValueError(
            "Could not infer include_time from the failure marker path. "
            "Pass --include_time True/False explicitly."
        )

    lr = lr_override
    if lr is None:
        lr = inferred.get("lr")
    if lr is None:
        raise ValueError(
            "Could not infer lr from the failure marker path. "
            "Pass --lr explicitly."
        )

    model_args = dict(failure_marker.get("model_args", {}))
    if not model_args:
        raise ValueError("Failure marker does not contain model_args.")

    repro_config = dict(base_config)
    repro_config.update(model_args)
    repro_config["lr"] = lr
    repro_config["time"] = include_time
    repro_config["num_steps"] = int(failure_marker.get("num_steps", inferred.get("num_steps", base_config["num_steps"])))
    repro_config["print_steps"] = int(failure_marker.get("print_steps", base_config["print_steps"]))
    repro_config["output_parent_dir"] = output_parent_dir
    repro_config["n_layers"] = int(model_args["n_layers"])

    return repro_config, seed, dataset_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run a failed SLinOSS sweep configuration from _sweep_failed.json.",
    )
    parser.add_argument(
        "--failure_json",
        type=str,
        required=True,
        help="Path to the _sweep_failed.json marker for the failed run.",
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="experiment_configs/repeats",
        help="Directory containing the SLinOSS dataset config JSON files.",
    )
    parser.add_argument(
        "--include_time",
        type=str,
        default=None,
        help="Optional explicit include_time override (True/False).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional explicit learning-rate override.",
    )
    parser.add_argument(
        "--output_parent_dir",
        type=str,
        default="repro_outputs",
        help="Where to write the reproduced run outputs.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Force torch.compile on for the reproduced run.",
    )
    args = parser.parse_args()

    include_time_override = None
    if args.include_time is not None:
        include_time_override = _parse_bool(args.include_time)

    failure_marker = _load_failure_marker(args.failure_json)
    repro_config, seed, dataset_name = _build_repro_config(
        failure_marker=failure_marker,
        failure_json_path=args.failure_json,
        experiment_folder=args.experiment_folder,
        include_time_override=include_time_override,
        lr_override=args.lr,
        output_parent_dir=args.output_parent_dir,
    )
    if args.torch_compile:
        repro_config["torch_compile"] = True

    run_args, _ = _build_run_args("SLinOSS", dataset_name, repro_config)

    print("Reproducing failed SLinOSS run with:")
    print(json.dumps(
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "include_time": run_args["include_time"],
            "lr": run_args["lr"],
            "num_steps": run_args["num_steps"],
            "print_steps": run_args["print_steps"],
            "model_args": run_args["model_args"],
            "output_parent_dir": run_args["output_parent_dir"],
            "torch_compile": run_args.get("torch_compile", False),
        },
        indent=2,
        sort_keys=True,
    ))

    create_dataset_model_and_train_torch(
        seed=seed,
        overwrite_output_dir=True,
        auto_confirm_output_dir=True,
        check_numerics=True,
        verbose=True,
        **run_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
