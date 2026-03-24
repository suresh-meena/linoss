"""Run JAX and Torch experiments from the shared JSON config format."""

from __future__ import annotations

import argparse
import json
import math
from typing import Callable

from train_torch import create_dataset_model_and_train_torch


TORCH_MODELS = {"SLinOSS"}
_MISSING = object()


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _first_present(config, keys, default=_MISSING):
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    if default is not _MISSING:
        return default
    key_list = ", ".join(keys)
    raise KeyError(f"Missing required config key. Expected one of: {key_list}.")


def _build_jax_model_args(model_name: str, config: dict) -> tuple[dict, int, int, str | None]:
    if model_name == "LinOSS":
        linoss_discretization = config["linoss_discretization"]
    else:
        linoss_discretization = None

    if model_name in {"lru", "S5", "S6", "mamba", "LinOSS"}:
        dt0 = None
    else:
        dt0 = float(config["dt0"])

    hidden_dim = int(config["hidden_dim"])
    scale = config["scale"]

    if model_name in {"log_ncde", "nrde", "ncde"}:
        vf_depth = int(config["vf_depth"])
        vf_width = int(config["vf_width"])
        if model_name in {"log_ncde", "nrde"}:
            logsig_depth = int(config["depth"])
            stepsize = int(float(config["stepsize"]))
        else:
            logsig_depth = 1
            stepsize = 1
        lambd = float(config["lambd"]) if model_name == "log_ncde" else None
        ssm_dim = None
        num_blocks = None
    else:
        vf_depth = None
        vf_width = None
        logsig_depth = 1
        stepsize = 1
        lambd = None
        ssm_dim = int(config["ssm_dim"])
        num_blocks = int(config["num_blocks"])

    if model_name in {"S5", "LinOSS"}:
        ssm_blocks = int(config["ssm_blocks"])
    else:
        ssm_blocks = None

    model_args = {
        "num_blocks": num_blocks,
        "hidden_dim": hidden_dim,
        "vf_depth": vf_depth,
        "vf_width": vf_width,
        "ssm_dim": ssm_dim,
        "ssm_blocks": ssm_blocks,
        "dt0": dt0,
        "solver": diffrax.Heun(),
        "stepsize_controller": diffrax.ConstantStepSize(),
        "scale": scale,
        "lambd": lambd,
    }
    return model_args, stepsize, logsig_depth, linoss_discretization


def _build_slinoss_model_args(config: dict) -> tuple[dict, int, int, None]:
    d_model = int(_first_present(config, ("d_model", "hidden_dim")))
    n_layers = int(_first_present(config, ("n_layers", "num_blocks")))
    d_state = int(_first_present(config, ("d_state", "ssm_dim"), default=128))
    expand = int(config.get("expand", 2))
    d_head = int(config.get("d_head", d_model))
    d_conv = int(config.get("d_conv", 4))
    chunk_size = int(config.get("chunk_size", 64))
    dropout = float(config.get("dropout", 0.0))
    ffn_mult = int(config.get("ffn_mult", 2))

    model_args = {
        "d_model": d_model,
        "n_layers": n_layers,
        "d_state": d_state,
        "expand": expand,
        "d_head": d_head,
        "d_conv": d_conv,
        "chunk_size": chunk_size,
        "dropout": dropout,
        "ffn_mult": ffn_mult,
        "dt_min": float(config.get("dt_min", 1e-4)),
        "dt_max": float(config.get("dt_max", 1e-1)),
        "dt_init_floor": float(config.get("dt_init_floor", 1e-4)),
        "r_min": float(config.get("r_min", 0.9)),
        "r_max": float(config.get("r_max", 1.0)),
        "theta_bound": float(config.get("theta_bound", math.pi)),
        "k_max": float(config.get("k_max", 0.5)),
        "eps": float(config.get("eps", 1e-8)),
    }
    return model_args, 1, 1, None


def _build_run_args(
    model_name: str,
    dataset_name: str,
    config: dict,
) -> tuple[dict, Callable]:
    if model_name in TORCH_MODELS:
        model_args, stepsize, logsig_depth, linoss_discretization = _build_slinoss_model_args(
            config
        )
        run_fn = create_dataset_model_and_train_torch
    else:
        model_args, stepsize, logsig_depth, linoss_discretization = _build_jax_model_args(
            model_name,
            config,
        )
        run_fn = create_dataset_model_and_train

    run_args = {
        "data_dir": config["data_dir"],
        "use_presplit": _parse_bool(config["use_presplit"]),
        "dataset_name": dataset_name,
        "output_step": int(config["output_step"]) if dataset_name == "ppg" else 1,
        "metric": config["metric"],
        "include_time": _parse_bool(config["time"]),
        "T": float(config["T"]),
        "model_name": model_name,
        "stepsize": stepsize,
        "logsig_depth": logsig_depth,
        "linoss_discretization": linoss_discretization,
        "model_args": model_args,
        "num_steps": int(config["num_steps"]),
        "print_steps": int(config["print_steps"]),
        "lr": float(config["lr"]),
        "lr_scheduler": eval(config["lr_scheduler"]),
        "batch_size": int(config["batch_size"]),
        "output_parent_dir": config["output_parent_dir"],
        "id": config.get("id"),
    }
    if model_name in TORCH_MODELS:
        run_args["torch_compile"] = _parse_bool(config.get("torch_compile", False))
        run_args["torch_compile_mode"] = config.get("torch_compile_mode", "reduce-overhead")
        run_args["allow_tf32"] = _parse_bool(config.get("allow_tf32", False))
        run_args["mixed_precision"] = _parse_bool(config.get("mixed_precision", False))
        run_args["check_numerics"] = _parse_bool(config.get("check_numerics", True))
        dataloader_workers = config.get("dataloader_workers", 0)
        run_args["dataloader_workers"] = int(dataloader_workers)
    return run_args, run_fn


def run_experiments(model_names, dataset_names, experiment_folder):
    for model_name in model_names:
        for dataset_name in dataset_names:
            config_path = f"{experiment_folder}/{model_name}/{dataset_name}.json"
            with open(config_path, "r") as file:
                config = json.load(file)

            run_args, run_fn = _build_run_args(model_name, dataset_name, config)
            for seed in config["seeds"]:
                print(f"Running experiment with seed: {seed}")
                run_fn(seed=seed, **run_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=["LinOSS"],
        help="One or more model names to run.",
    )
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["EigenWorms"],
        help="One or more dataset names to run.",
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="experiment_configs/repeats",
        help="Directory that contains per-model config subfolders.",
    )
    args = parser.parse_args()

    run_experiments(args.model_name, args.dataset_name, args.experiment_folder)
