"""Worker script that loads one dataset once and validates many dry-run configs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# Torch-only worker: prevent accidental JAX GPU preallocation if JAX is imported
# transitively by a dependency stack.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from data_dir.torch_datasets import create_torch_dataset
from models.SLinOSS import ensure_slinoss_cuda_ready
from models.generate_torch_model import create_torch_model
from train_torch import _set_torch_seed
from sweep_slinoss.sweep_slinoss import (
    _classify_failure,
    _cleanup_after_run,
    _safe_error_message,
)


def _move_dataset_to_gpu(dataset, device: torch.device) -> None:
    def _move(td):
        return TensorDataset(*[t.to(device, non_blocking=True) for t in td.tensors])

    dataset.train = _move(dataset.train)
    dataset.val = _move(dataset.val)
    dataset.test = _move(dataset.test)


def _write_jsonl(path: str, record: dict[str, object]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True))
        file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    args = parser.parse_args()

    with open(args.payload, "r", encoding="utf-8") as file:
        payload = json.load(file)

    result_path = payload["result_path"]
    if os.path.exists(result_path):
        os.remove(result_path)

    dataset_name = payload["dataset"]
    include_time = bool(payload["include_time"])
    seed = int(payload["seed"])
    data_dir = payload["data_dir"]
    use_presplit = bool(payload["use_presplit"])
    T = float(payload["T"])
    batch_size = int(payload["batch_size"])
    datasetkey = int(payload["datasetkey"])
    model_seed = int(payload["model_seed"])
    tasks = payload["tasks"]

    ensure_slinoss_cuda_ready()
    device = torch.device("cuda")

    dataset = create_torch_dataset(
        data_dir,
        dataset_name,
        use_presplit,
        include_time,
        T,
        key=datasetkey,
    )
    _move_dataset_to_gpu(dataset, device)

    if batch_size > len(dataset.train):
        raise ValueError(
            f"Batch size {batch_size} larger than training dataset size {len(dataset.train)}."
        )

    x_all, lengths_all, labels_all = dataset.train.tensors
    x = x_all[:batch_size].contiguous()
    lengths = lengths_all[:batch_size].contiguous()
    labels = labels_all[:batch_size].contiguous()

    for task in tasks:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            _set_torch_seed(model_seed)
            model = create_torch_model(
                "SLinOSS",
                dataset.data_dim,
                dataset.label_dim,
                model_args=task["model_args"],
            ).to(device)
            model.train()

            logits = model(x, lengths)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            end_event.record()
            torch.cuda.synchronize(device=device)

            elapsed_sec = round(start_event.elapsed_time(end_event) / 1000.0, 4)
            record = {
                "dataset_name": dataset_name,
                "seed": seed,
                "combo_index": task["combo_index"],
                "params": task["params"],
                "model_args": task["model_args"],
                "elapsed_sec": elapsed_sec,
                "status": "passed",
            }
            _write_jsonl(result_path, record)
            del model
        except Exception as exc:
            end_event.record()
            torch.cuda.synchronize(device=device)
            elapsed_sec = round(start_event.elapsed_time(end_event) / 1000.0, 4)
            record = {
                "dataset_name": dataset_name,
                "seed": seed,
                "combo_index": task["combo_index"],
                "params": task["params"],
                "model_args": task["model_args"],
                "elapsed_sec": elapsed_sec,
                "status": "failed",
                "failure_kind": _classify_failure(exc),
                "error_type": exc.__class__.__name__,
                "error_message": _safe_error_message(exc),
                "traceback": traceback.format_exc(limit=25),
            }
            _write_jsonl(result_path, record)
        finally:
            _cleanup_after_run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
