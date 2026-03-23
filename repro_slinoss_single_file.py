#!/usr/bin/env python3
"""Single-file SLinOSS repro for the known EigenWorms non-finite failure.

Requirements for shipped repro usage
------------------------------------
- Python 3.10+
- PyTorch with CUDA
- A working local ``slinoss`` installation with its CUDA extensions
- The frozen dataset bundle produced on the original machine, for example:
  ``eigenworms_repro_bundle.pt``

The maintainer-facing repro command is:

    python repro_slinoss_single_file.py repro --bundle eigenworms_repro_bundle.pt

In ``repro`` mode this script does not import any linoss model, sweep, or
training wrapper. The model wrapper and training loop are defined inline here.

This script also has a local helper mode:

1. ``dump-bundle``: run locally in this repo to freeze the exact train/val/test
   split into a single ``.pt`` file.
2. ``repro``: run from that bundle with the inline SLinOSS model wrapper and
   inline training loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from slinoss.layers import CuteScanBackend, CuteScanPrepBackend, SLinOSSMixer
    from slinoss.ops.cconv1d import (
        cconv1d_cuda_supported,
        cconv1d_is_available,
        cconv1d_load_error,
    )
except Exception as exc:  # pragma: no cover - depends on local slinoss install.
    CuteScanBackend = None
    CuteScanPrepBackend = None
    SLinOSSMixer = None
    cconv1d_cuda_supported = None
    cconv1d_is_available = None
    cconv1d_load_error = None
    _SLINOSS_IMPORT_ERROR = exc
else:
    _SLINOSS_IMPORT_ERROR = None


DATASET_NAME = "EigenWorms"
TOP_LEVEL_SEED = 3456
T = 1.0

FAILING_CONFIG = {
    "include_time": False,
    "lr": 1e-3,
    "num_steps": 100000,
    "print_steps": 500,
    "batch_size": 4,
    "mixed_precision": False,
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
    "theta_bound": math.pi,
    "k_max": 0.5,
    "eps": 1e-8,
}


@dataclass
class LocalDataset:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    data_dim: int
    label_dim: int


def _derive_seeds(seed: int) -> tuple[int, int, int]:
    gen = torch.Generator().manual_seed(seed)
    dataset_seed = int(torch.randint(0, 2**32, (1,), generator=gen).item())
    model_seed = int(torch.randint(0, 2**32, (1,), generator=gen).item())
    shuffle_seed = int(torch.randint(0, 2**32, (1,), generator=gen).item())
    return dataset_seed, model_seed, shuffle_seed


def ensure_slinoss_cuda_ready() -> None:
    if _SLINOSS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Could not import slinoss. Reproduce mode requires a working local "
            "`slinoss` package plus its CUDA dependencies."
        ) from _SLINOSS_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("Reproduce mode requires CUDA. torch.cuda.is_available() is False.")
    if not cconv1d_is_available():
        detail = cconv1d_load_error()
        raise RuntimeError(
            "The slinoss CUDA causal-conv extension is unavailable."
        ) from detail


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"Expected {name} on CUDA. Got {tensor.device}.")


class TransposedBatchNormEMA(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model, affine=True, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model: int, *, mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class StrictCudaCConv1dBackend:
    def __call__(
        self,
        owner: SLinOSSMixer,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_cuda_tensor("inputs", x)
        if conv_state is not None:
            _require_cuda_tensor("convolution state", conv_state)
        if not cconv1d_is_available():
            detail = cconv1d_load_error()
            raise RuntimeError(
                "SLinOSS causal-conv CUDA extension is unavailable."
            ) from detail
        if not cconv1d_cuda_supported(
            x.transpose(1, 2),
            owner.dw_weight,
            initial_states=conv_state,
            activation=None,
        ):
            raise RuntimeError(
                "Current configuration is unsupported by the slinoss CUDA "
                f"causal-conv kernel: input device={x.device}, input dtype={x.dtype}, "
                f"weight dtype={owner.dw_weight.dtype}, d_conv={owner.d_conv}."
            )
        return owner._apply_cconv_cuda(x, conv_state)


class SLinOSSBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        dropout: float,
        ffn_mult: int,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm = TransposedBatchNormEMA(d_model)
        self.mixer = SLinOSSMixer(
            d_model,
            d_state=d_state,
            expand=expand,
            d_head=d_head,
            d_conv=d_conv,
            chunk_size=chunk_size,
            scanprep_backend=CuteScanPrepBackend(),
            cconv_backend=StrictCudaCConv1dBackend(),
            backend=CuteScanBackend(),
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            r_min=r_min,
            r_max=r_max,
            theta_bound=theta_bound,
            k_max=k_max,
            eps=eps,
            normalize_bc=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = SiLUFeedForward(d_model, mult=ffn_mult, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = F.silu(x)
        x = self.dropout1(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        return residual + x


class SLinOSSClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        d_model: int,
        n_layers: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
        dropout: float,
        ffn_mult: int,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
    ) -> None:
        super().__init__()
        ensure_slinoss_cuda_ready()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                SLinOSSBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    d_head=d_head,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    dt_init_floor=dt_init_floor,
                    r_min=r_min,
                    r_max=r_max,
                    theta_bound=theta_bound,
                    k_max=k_max,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = TransposedBatchNormEMA(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        timesteps = x.shape[1]
        idx = torch.arange(timesteps, device=x.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = lengths.clamp_min(1).to(dtype=x.dtype).unsqueeze(1)
        return (x * mask).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _require_cuda_tensor("inputs", x)
        _require_cuda_tensor("sequence lengths", lengths)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        pooled = self._masked_mean(x, lengths.long())
        return self.head(pooled)


def _first_nonfinite_index(tensor: torch.Tensor) -> tuple[int, ...] | None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return None
    bad = (~finite_mask).nonzero(as_tuple=False)
    return tuple(int(i) for i in bad[0].tolist())


def _make_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


def _infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, lengths, labels in loader:
        x = x.to(device=device, non_blocking=True)
        lengths = lengths.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        logits = model(x, lengths)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.shape[0])
    return float(correct / total)


def _make_tensor_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> TensorDataset:
    x = x.detach().cpu().to(dtype=torch.float32).contiguous()
    y = y.detach().cpu().to(dtype=torch.int64).contiguous()
    if lengths is None:
        lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.int64)
    else:
        lengths = lengths.detach().cpu().to(dtype=torch.int64).contiguous()
    return TensorDataset(x, lengths, y)


def _load_bundle(path: str) -> tuple[LocalDataset, dict]:
    bundle = torch.load(path, map_location="cpu")
    required = ["train_x", "train_y", "val_x", "val_y", "test_x", "test_y"]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"Bundle missing required keys: {missing}")

    train = _make_tensor_dataset(bundle["train_x"], bundle["train_y"], bundle.get("train_lengths"))
    val = _make_tensor_dataset(bundle["val_x"], bundle["val_y"], bundle.get("val_lengths"))
    test = _make_tensor_dataset(bundle["test_x"], bundle["test_y"], bundle.get("test_lengths"))
    label_dim = int(
        max(
            int(train.tensors[2].max().item()),
            int(val.tensors[2].max().item()),
            int(test.tensors[2].max().item()),
        )
        + 1
    )
    dataset = LocalDataset(
        train=train,
        val=val,
        test=test,
        data_dim=int(train.tensors[0].shape[-1]),
        label_dim=label_dim,
    )
    meta = {
        "dataset_name": bundle.get("dataset_name", DATASET_NAME),
        "source": bundle.get("source", "unknown"),
        "include_time": bool(bundle.get("include_time", FAILING_CONFIG["include_time"])),
        "seed": int(bundle.get("seed", TOP_LEVEL_SEED)),
    }
    return dataset, meta


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train(
    dataset: LocalDataset,
    model: nn.Module,
    *,
    num_steps: int,
    print_steps: int,
    lr: float,
    batch_size: int,
    shuffle_seed: int,
    device: torch.device,
) -> None:
    train_generator = torch.Generator().manual_seed(shuffle_seed)
    train_loader = _make_loader(dataset.train, batch_size=batch_size, shuffle=True, drop_last=True, generator=train_generator)
    val_loader = _make_loader(dataset.val, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = _make_loader(dataset.test, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batches = _infinite_batches(train_loader)
    start = time.time()
    running_loss = 0.0
    best_val = float("-inf")
    best_state = None
    no_val_improvement = 0

    try:
        for step in range(num_steps):
            model.train()
            x, lengths, labels = next(batches)
            x = x.to(device=device, non_blocking=True)
            lengths = lengths.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            bad_input = _first_nonfinite_index(x)
            if bad_input is not None:
                raise FloatingPointError(
                    "Encountered non-finite input batch values "
                    f"at training step {step + 1}, tensor index {bad_input}."
                )

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            bad_logits = _first_nonfinite_index(logits)
            if bad_logits is not None:
                raise FloatingPointError(
                    "Encountered non-finite model logits "
                    f"at training step {step + 1}, tensor index {bad_logits}."
                )

            loss = F.cross_entropy(logits, labels)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"Encountered non-finite loss at training step {step + 1}: loss={loss_value}."
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm_value = float(grad_norm)
            if not math.isfinite(grad_norm_value):
                raise FloatingPointError(
                    "Encountered non-finite gradient norm "
                    f"at training step {step + 1}: grad_norm={grad_norm_value}."
                )

            optimizer.step()
            running_loss += loss_value

            if (step + 1) % print_steps != 0:
                continue

            val_acc = _evaluate_accuracy(model, val_loader, device)
            elapsed = time.time() - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Validation metric: {val_acc}, Time: {elapsed}",
                flush=True,
            )

            if step > 0:
                if val_acc <= best_val:
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        print("Early stopping triggered.")
                        break
                else:
                    no_val_improvement = 0

            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

            running_loss = 0.0
            start = time.time()

        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = _evaluate_accuracy(model, test_loader, device)
        print(f"Test metric: {test_acc}", flush=True)
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()


def _dump_bundle(args: argparse.Namespace) -> int:
    repo_root = os.path.abspath(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from data_dir.torch_datasets import create_torch_dataset

    dataset_seed, _, _ = _derive_seeds(args.seed)
    dataset = create_torch_dataset(
        data_dir=args.data_dir,
        name=args.dataset_name,
        use_presplit=False,
        include_time=args.include_time,
        T=T,
        key=dataset_seed,
    )

    bundle = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "include_time": args.include_time,
        "source": "linoss create_torch_dataset(use_presplit=False)",
        "train_x": dataset.train.tensors[0].clone(),
        "train_lengths": dataset.train.tensors[1].clone(),
        "train_y": dataset.train.tensors[2].clone(),
        "val_x": dataset.val.tensors[0].clone(),
        "val_lengths": dataset.val.tensors[1].clone(),
        "val_y": dataset.val.tensors[2].clone(),
        "test_x": dataset.test.tensors[0].clone(),
        "test_lengths": dataset.test.tensors[1].clone(),
        "test_y": dataset.test.tensors[2].clone(),
    }
    torch.save(bundle, args.bundle)
    print(
        json.dumps(
            {
                "bundle": os.path.abspath(args.bundle),
                "dataset_name": args.dataset_name,
                "seed": args.seed,
                "include_time": args.include_time,
                "train_shape": tuple(bundle["train_x"].shape),
                "val_shape": tuple(bundle["val_x"].shape),
                "test_shape": tuple(bundle["test_x"].shape),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _repro(args: argparse.Namespace) -> int:
    ensure_slinoss_cuda_ready()
    dataset, meta = _load_bundle(args.bundle)
    _, model_seed, shuffle_seed = _derive_seeds(args.seed)
    _set_torch_seed(model_seed)

    device = torch.device("cuda")
    model = SLinOSSClassifier(
        input_dim=dataset.data_dim,
        num_classes=dataset.label_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        d_head=args.d_head,
        d_conv=args.d_conv,
        chunk_size=args.chunk_size,
        dropout=args.dropout,
        ffn_mult=args.ffn_mult,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        dt_init_floor=args.dt_init_floor,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_bound=args.theta_bound,
        k_max=args.k_max,
        eps=args.eps,
    ).to(device)

    if args.torch_compile:
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    print(
        json.dumps(
            {
                "bundle": os.path.abspath(args.bundle),
                "bundle_meta": meta,
                "expected_failure": "FloatingPointError: Encountered non-finite model logits",
                "seed": args.seed,
                "model_seed": model_seed,
                "shuffle_seed": shuffle_seed,
                "config": {
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "num_steps": args.num_steps,
                    "print_steps": args.print_steps,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "d_state": args.d_state,
                    "expand": args.expand,
                    "d_head": args.d_head,
                    "d_conv": args.d_conv,
                    "chunk_size": args.chunk_size,
                    "dropout": args.dropout,
                    "ffn_mult": args.ffn_mult,
                    "dt_min": args.dt_min,
                    "dt_max": args.dt_max,
                    "dt_init_floor": args.dt_init_floor,
                    "r_min": args.r_min,
                    "r_max": args.r_max,
                    "theta_bound": args.theta_bound,
                    "k_max": args.k_max,
                    "eps": args.eps,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )

    _train(
        dataset,
        model,
        num_steps=args.num_steps,
        print_steps=args.print_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        shuffle_seed=shuffle_seed,
        device=device,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file SLinOSS repro for the known EigenWorms failure.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser(
        "dump-bundle",
        help="Freeze the exact EigenWorms split used by linoss into one .pt file.",
    )
    dump_parser.add_argument(
        "--data_dir",
        default="data_dir",
        help="Repo data directory containing processed/UEA/<dataset>.",
    )
    dump_parser.add_argument(
        "--dataset_name",
        default=DATASET_NAME,
        help="Dataset to bundle.",
    )
    dump_parser.add_argument(
        "--seed",
        type=int,
        default=TOP_LEVEL_SEED,
        help="Top-level seed used to derive the dataset split key.",
    )
    dump_parser.add_argument(
        "--include_time",
        action="store_true",
        help="Include a time channel when creating the bundle.",
    )
    dump_parser.add_argument(
        "--bundle",
        default="eigenworms_repro_bundle.pt",
        help="Output .pt file for the frozen split.",
    )
    dump_parser.set_defaults(func=_dump_bundle)

    repro_parser = subparsers.add_parser(
        "repro",
        help="Run the known failing SLinOSS config from a frozen .pt bundle.",
    )
    repro_parser.add_argument(
        "--bundle",
        default="eigenworms_repro_bundle.pt",
        help="Path to the frozen dataset bundle created by dump-bundle.",
    )
    repro_parser.add_argument(
        "--seed",
        type=int,
        default=TOP_LEVEL_SEED,
        help="Top-level seed. Default matches the failing run.",
    )
    repro_parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Optionally enable torch.compile.",
    )
    for key, value in FAILING_CONFIG.items():
        if isinstance(value, bool):
            continue
        repro_parser.add_argument(
            f"--{key}",
            type=type(value),
            default=value,
            help=f"Override {key}. Default matches the failing run.",
        )
    repro_parser.set_defaults(func=_repro)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
