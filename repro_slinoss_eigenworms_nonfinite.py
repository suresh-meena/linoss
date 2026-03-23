"""Exact repo-root repro for the known SLinOSS EigenWorms non-finite failure.

This script uses the real processed EigenWorms dataset already expected by the
repo and runs the exact failing SLinOSS configuration directly, without going
through the sweep machinery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from run_experiment import _build_run_args
from train_torch import create_dataset_model_and_train_torch


DATASET_NAME = "EigenWorms"
SEED = 3456

EXACT_FAILING_OVERRIDES = {
    "time": False,
    "lr": 1e-3,
    "num_steps": 100000,
    "print_steps": 500,
    "d_model": 128,
    "n_layers": 2,
    "d_state": 64,
    "expand": 2,
    "d_head": 64,
    "d_conv": 4,
    "chunk_size": 64,
    "dropout": 0.0,
    "ffn_mult": 2,
    "dt_min": 1e-4,
    "dt_max": 1e-1,
    "dt_init_floor": 1e-4,
    "r_min": 0.9,
    "r_max": 1.0,
    "theta_bound": 3.141592653589793,
    "k_max": 0.5,
    "eps": 1e-8,
    "torch_compile": False,
}


def _load_base_config(experiment_folder: str) -> dict:
    config_path = os.path.join(experiment_folder, "SLinOSS", f"{DATASET_NAME}.json")
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the known SLinOSS non-finite EigenWorms failure.",
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="experiment_configs/repeats",
        help="Directory containing the SLinOSS dataset config JSON files.",
    )
    parser.add_argument(
        "--output_parent_dir",
        type=str,
        default="repro_outputs",
        help="Where to write this repro run's outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed to use. Default matches the known failing run.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Optionally enable torch.compile for this repro.",
    )
    args = parser.parse_args()

    config = _load_base_config(args.experiment_folder)
    config.update(EXACT_FAILING_OVERRIDES)
    config["output_parent_dir"] = args.output_parent_dir
    if args.torch_compile:
        config["torch_compile"] = True

    run_args, _ = _build_run_args("SLinOSS", DATASET_NAME, config)

    print("Running exact SLinOSS repro with config:")
    print(
        json.dumps(
            {
                "dataset_name": DATASET_NAME,
                "seed": args.seed,
                "output_parent_dir": run_args["output_parent_dir"],
                "include_time": run_args["include_time"],
                "num_steps": run_args["num_steps"],
                "print_steps": run_args["print_steps"],
                "lr": run_args["lr"],
                "batch_size": run_args["batch_size"],
                "torch_compile": run_args.get("torch_compile", False),
                "model_args": run_args["model_args"],
                "expected_failure": "FloatingPointError: Encountered non-finite model logits",
            },
            indent=2,
            sort_keys=True,
        )
    )

    create_dataset_model_and_train_torch(
        seed=args.seed,
        overwrite_output_dir=True,
        auto_confirm_output_dir=True,
        check_numerics=True,
        verbose=True,
        **run_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
