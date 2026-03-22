"""Worker script that loads a dataset once onto the GPU and evaluates many configs."""

import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import Callable

# Torch-only worker: prevent accidental JAX GPU preallocation if JAX is imported
# transitively by a dependency stack.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch.utils.data import TensorDataset
from data_dir.torch_datasets import create_torch_dataset
from models.generate_torch_model import create_torch_model
from train_torch import _prepare_output_dir, train_torch_model, _set_torch_seed
from models.SLinOSS import ensure_slinoss_cuda_ready
from run_experiment import _build_run_args
from sweep_slinoss.sweep_slinoss import (
    _apply_sweep_params_to_config,
    _cleanup_after_failure,
    _cleanup_after_run,
    _classify_failure,
    _is_completed_run,
    _write_completion_marker,
    _write_failure_marker,
)


FALLBACK_PATTERNS = (
    "falling back to 'reference'",
    'falling back to "reference"',
    "failed warmup with CuTe/CUDA runtime error",
)


class _LineLoggerStream:
    """Capture stdout/stderr lines, forward them to the worker log, and keep a copy."""

    def __init__(self, logger: Callable[[str], None]) -> None:
        self._logger = logger
        self._parts: list[str] = []
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._parts.append(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._logger(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._logger(self._buffer)
            self._buffer = ""

    def get_text(self) -> str:
        return "".join(self._parts)


def _write_scan_backend_status(
    target_dir: str,
    *,
    used_reference_fallback: bool,
    matching_lines: list[str],
    status: str | None = None,
) -> None:
    status_path = os.path.join(target_dir, "_scan_backend_status.json")
    payload = {
        "scan_backend_requested": "CuTe",
        "used_reference_fallback": used_reference_fallback,
        "status": status
        if status is not None
        else ("reference_fallback" if used_reference_fallback else "cute_ok"),
        "matching_lines": matching_lines,
    }
    with open(status_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def move_dataset_to_gpu(dataset, device):
    def _move(td):
        return TensorDataset(*[t.to(device, non_blocking=True) for t in td.tensors])
    dataset.train = _move(dataset.train)
    dataset.val = _move(dataset.val)
    dataset.test = _move(dataset.test)

def run_task_group(
    payload: dict[str, object],
    *,
    logger: Callable[[str], None] = print,
) -> dict[str, object]:
    dataset_name = payload["dataset"]
    include_time = payload["include_time"]
    seed = payload["seed"]
    tasks = payload["tasks"]
    experiment_folder = payload["experiment_folder"]
    skip_existing = payload["skip_existing"]

    config_path = os.path.join(experiment_folder, "SLinOSS", f"{dataset_name}.json")
    with open(config_path, "r", encoding="utf-8") as file:
        base_config = json.load(file)

    ensure_slinoss_cuda_ready()
    device = torch.device("cuda")

    T = float(base_config.get("T", 1.0))
    data_dir = base_config.get("data_dir", "data_dir")
    use_presplit_str = str(base_config.get("use_presplit", "False")).lower()
    use_presplit = use_presplit_str in ("true", "1", "yes")

    # Generate deterministic keys identically to train_torch.py
    gen = torch.Generator().manual_seed(seed)
    datasetkey = torch.randint(0, 2**32, (1,), generator=gen).item()
    modelkey = torch.randint(0, 2**32, (1,), generator=gen).item()
    shufflekey = torch.randint(0, 2**32, (1,), generator=gen).item()

    logger(f"[Worker] Loading dataset {dataset_name} (time={include_time}, seed={seed}) onto GPU...")
    dataset = create_torch_dataset(
        data_dir,
        dataset_name,
        use_presplit,
        include_time,
        T,
        key=datasetkey,
    )
    move_dataset_to_gpu(dataset, device)
    logger(f"[Worker] Dataset loaded. Starting {len(tasks)} runs...")

    had_failures = False

    for i, task in enumerate(tasks):
        params = task["params"]
        target_dir = task["target_dir"]
        stream = _LineLoggerStream(logger)
        stream_text = ""
        matching_lines: list[str] = []
        scan_status = "unknown_due_to_failure"
        used_reference_fallback = False

        if skip_existing and _is_completed_run(target_dir):
            continue

        run_config = _apply_sweep_params_to_config(base_config, params)
        run_args, _ = _build_run_args("SLinOSS", dataset_name, run_config)
        run_args["print_steps"] = max(int(run_args["print_steps"]), 500)

        overwrite_existing = os.path.isdir(target_dir) and not skip_existing

        logger(f"[Worker] Run {i+1}/{len(tasks)}: {os.path.basename(target_dir)}")
        
        os.makedirs(target_dir, exist_ok=True)
        active_marker = os.path.join(target_dir, "_active_run.json")
        with open(active_marker, "w") as f:
            json.dump({"status": "running"}, f)

        try:
            _prepare_output_dir(
                target_dir,
                overwrite=overwrite_existing,
                auto_confirm=True, # We already created it above, so auto-confirm
                verbose=False,
            )

            # Re-create active marker inside the now-clean directory
            with open(active_marker, "w") as f:
                json.dump({"status": "running"}, f)

            _set_torch_seed(modelkey)

            model = create_torch_model(
                "SLinOSS",
                dataset.data_dim,
                dataset.label_dim,
                model_args=run_args["model_args"],
            ).to(device)

            if run_args.get("torch_compile", False):
                model = torch.compile(model, mode=run_args.get("torch_compile_mode", "reduce-overhead"), dynamic=True)

            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                trained_model = train_torch_model(
                    dataset,
                    model,
                    num_steps=run_args["num_steps"],
                    print_steps=run_args["print_steps"],
                    lr=run_args["lr"],
                    lr_scheduler=run_args["lr_scheduler"],
                    batch_size=run_args["batch_size"],
                    shuffle_seed=shufflekey,
                    output_dir=target_dir,
                    device=device,
                    dataloader_workers=0,
                    mixed_precision=False,
                    verbose=False,
                    progress_callback=None,
                )
            scan_status = "cute_ok"
            if used_reference_fallback:
                logger(
                    "[Worker] Scan backend fallback detected; "
                    f"reference backend was used for {os.path.basename(target_dir)}"
                )
            
            if os.path.exists(active_marker):
                os.remove(active_marker)

            _write_completion_marker(
                target_dir,
                dataset_name=dataset_name,
                seed=seed,
                run_args=run_args,
            )
            
            del model
            del trained_model

        except Exception as exc:
            if os.path.exists(active_marker):
                os.remove(active_marker)
            failure_kind = _classify_failure(exc)
            _cleanup_after_failure(target_dir)
            _write_failure_marker(
                target_dir,
                dataset_name=dataset_name,
                seed=seed,
                run_args=run_args,
                failure_kind=failure_kind,
                exc=exc,
            )
            had_failures = True
            logger(f"[Worker] Run failed ({failure_kind}): {exc}")
            logger(traceback.format_exc())
        finally:
            stream.flush()
            stream_text = stream.get_text()
            matching_lines = [
                line
                for line in stream_text.splitlines()
                if any(pattern in line for pattern in FALLBACK_PATTERNS)
            ]
            used_reference_fallback = bool(matching_lines)
            if used_reference_fallback:
                scan_status = "reference_fallback"
            _write_scan_backend_status(
                target_dir,
                used_reference_fallback=used_reference_fallback,
                matching_lines=matching_lines,
                status=scan_status,
            )
            _cleanup_after_run()

    return {"tasks": len(tasks), "had_failures": had_failures}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    args = parser.parse_args()

    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)

    run_task_group(payload)

if __name__ == "__main__":
    main()
