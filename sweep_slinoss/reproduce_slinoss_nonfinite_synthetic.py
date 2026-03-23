"""Synthetic one-file repro for SLinOSS non-finite forward failures.

This script does not rely on the UEA data loader. It creates a deterministic
finite dataset with EigenWorms-like tensor shape and scale, then trains the
same SLinOSS configuration through the normal Torch training loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

# Torch-only entrypoint: prevent accidental JAX GPU preallocation if JAX is
# imported transitively by a dependency stack.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_dir.torch_datasets import TorchDataset
from models.SLinOSS import ensure_slinoss_cuda_ready
from models.generate_torch_model import create_torch_model
from train_torch import _build_output_dir, _prepare_output_dir, _set_torch_seed, train_torch_model


@dataclass
class ReproConfig:
    seed: int = 3456
    dataset_name: str = "SyntheticEigenWormsLike"
    include_time: bool = False
    T: float = 1.0
    num_steps: int = 100000
    print_steps: int = 500
    lr: float = 1e-3
    batch_size: int = 4
    num_classes: int = 5
    num_channels: int = 6
    sequence_length: int = 17984
    train_size: int = 165
    val_size: int = 35
    test_size: int = 36
    base_scale: float = 4.84
    spike_scale: float = 220.0
    spikes_per_channel: int = 6
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    output_parent_dir: str = "synthetic_repro_outputs"


MODEL_ARGS = {
    "chunk_size": 64,
    "d_conv": 4,
    "d_head": 64,
    "d_model": 128,
    "d_state": 64,
    "dropout": 0.0,
    "dt_init_floor": 0.0001,
    "dt_max": 0.1,
    "dt_min": 0.0001,
    "eps": 1e-08,
    "expand": 2,
    "ffn_mult": 2,
    "k_max": 0.5,
    "n_layers": 2,
    "r_max": 1.0,
    "r_min": 0.9,
    "theta_bound": float(np.pi),
}


def _split_key_triplet(seed: int) -> tuple[int, int, int]:
    gen = torch.Generator().manual_seed(seed)
    datasetkey = torch.randint(0, 2**32, (1,), generator=gen).item()
    modelkey = torch.randint(0, 2**32, (1,), generator=gen).item()
    shufflekey = torch.randint(0, 2**32, (1,), generator=gen).item()
    return int(datasetkey), int(modelkey), int(shufflekey)


def _prepend_time_channel(data: np.ndarray, T: float) -> np.ndarray:
    timesteps = data.shape[1]
    ts = (T / timesteps) * np.broadcast_to(
        np.arange(timesteps, dtype=np.float32),
        (data.shape[0], timesteps),
    )
    return np.concatenate([ts[:, :, None], data], axis=2)


def _make_synthetic_block(
    *,
    rng: np.random.Generator,
    size: int,
    sequence_length: int,
    num_channels: int,
    num_classes: int,
    base_scale: float,
    spike_scale: float,
    spikes_per_channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    labels = rng.integers(0, num_classes, size=size, dtype=np.int64)
    data = rng.normal(
        loc=0.05,
        scale=base_scale,
        size=(size, sequence_length, num_channels),
    ).astype(np.float32)

    # Add deterministic class-structured oscillations so this is not pure noise.
    for i in range(size):
        label = int(labels[i])
        for ch in range(num_channels):
            freq = 1.0 + 0.35 * (label + 1) * (ch + 1)
            phase = 0.17 * (label + ch)
            amplitude = base_scale * (0.55 + 0.12 * ch)
            signal = amplitude * np.sin((2.0 * np.pi * freq * t) + phase)
            signal += 0.35 * amplitude * np.cos((np.pi * (label + 1) * t) + 0.5 * phase)
            data[i, :, ch] += signal.astype(np.float32)

            # Inject a handful of spikes so the dynamic range resembles EigenWorms.
            spike_positions = rng.integers(0, sequence_length, size=spikes_per_channel)
            spike_values = rng.normal(
                loc=0.0,
                scale=spike_scale * (1.0 + 0.1 * ch),
                size=spikes_per_channel,
            ).astype(np.float32)
            data[i, spike_positions, ch] += spike_values

    return data, labels


def _to_tensor_dataset(data: np.ndarray, labels: np.ndarray) -> TensorDataset:
    lengths = np.full((data.shape[0],), data.shape[1], dtype=np.int64)
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)),
        torch.from_numpy(lengths),
        torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
    )


def _build_dataset(config: ReproConfig, dataset_seed: int) -> TorchDataset:
    rng = np.random.default_rng(dataset_seed)

    train_data, train_labels = _make_synthetic_block(
        rng=rng,
        size=config.train_size,
        sequence_length=config.sequence_length,
        num_channels=config.num_channels,
        num_classes=config.num_classes,
        base_scale=config.base_scale,
        spike_scale=config.spike_scale,
        spikes_per_channel=config.spikes_per_channel,
    )
    val_data, val_labels = _make_synthetic_block(
        rng=rng,
        size=config.val_size,
        sequence_length=config.sequence_length,
        num_channels=config.num_channels,
        num_classes=config.num_classes,
        base_scale=config.base_scale,
        spike_scale=config.spike_scale,
        spikes_per_channel=config.spikes_per_channel,
    )
    test_data, test_labels = _make_synthetic_block(
        rng=rng,
        size=config.test_size,
        sequence_length=config.sequence_length,
        num_channels=config.num_channels,
        num_classes=config.num_classes,
        base_scale=config.base_scale,
        spike_scale=config.spike_scale,
        spikes_per_channel=config.spikes_per_channel,
    )

    if config.include_time:
        train_data = _prepend_time_channel(train_data, config.T)
        val_data = _prepend_time_channel(val_data, config.T)
        test_data = _prepend_time_channel(test_data, config.T)

    return TorchDataset(
        name=config.dataset_name,
        train=_to_tensor_dataset(train_data, train_labels),
        val=_to_tensor_dataset(val_data, val_labels),
        test=_to_tensor_dataset(test_data, test_labels),
        data_dim=int(train_data.shape[-1]),
        label_dim=config.num_classes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce SLinOSS non-finite failure without UEA data.",
    )
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--print_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_scale", type=float, default=4.84)
    parser.add_argument("--spike_scale", type=float, default=220.0)
    parser.add_argument("--spikes_per_channel", type=int, default=6)
    parser.add_argument("--output_parent_dir", type=str, default="synthetic_repro_outputs")
    parser.add_argument("--torch_compile", action="store_true")
    args = parser.parse_args()

    config = ReproConfig(
        seed=args.seed,
        num_steps=args.num_steps,
        print_steps=args.print_steps,
        batch_size=args.batch_size,
        base_scale=args.base_scale,
        spike_scale=args.spike_scale,
        spikes_per_channel=args.spikes_per_channel,
        output_parent_dir=args.output_parent_dir,
        torch_compile=args.torch_compile,
    )

    ensure_slinoss_cuda_ready()
    device = torch.device("cuda")

    datasetkey, modelkey, shufflekey = _split_key_triplet(config.seed)
    dataset = _build_dataset(config, datasetkey)

    print("Synthetic SLinOSS repro configuration:")
    print(
        json.dumps(
            {
                "repro_config": asdict(config),
                "model_args": MODEL_ARGS,
                "dataset_seed": datasetkey,
                "model_seed": modelkey,
                "shuffle_seed": shufflekey,
                "data_dim": dataset.data_dim,
                "label_dim": dataset.label_dim,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        "Synthetic dataset stats:",
        json.dumps(
            {
                "train_shape": tuple(int(v) for v in dataset.train.tensors[0].shape),
                "train_min": float(dataset.train.tensors[0].amin().item()),
                "train_max": float(dataset.train.tensors[0].amax().item()),
                "train_mean": float(dataset.train.tensors[0].mean().item()),
                "train_std": float(dataset.train.tensors[0].std().item()),
            },
            indent=2,
            sort_keys=True,
        ),
    )

    _set_torch_seed(modelkey)
    model = create_torch_model(
        "SLinOSS",
        dataset.data_dim,
        dataset.label_dim,
        model_args=MODEL_ARGS,
    ).to(device)
    if config.torch_compile:
        model = torch.compile(model, mode=config.torch_compile_mode, dynamic=True)

    output_dir = os.path.join(
        config.output_parent_dir,
        "outputs",
        "SLinOSS",
        config.dataset_name,
        _build_output_dir(
            seed=config.seed,
            T=config.T,
            include_time=config.include_time,
            num_steps=config.num_steps,
            lr=config.lr,
            model_name="SLinOSS",
            stepsize=1,
            logsig_depth=1,
            model_args=MODEL_ARGS,
        ),
    )
    _prepare_output_dir(
        output_dir,
        overwrite=True,
        auto_confirm=True,
        verbose=True,
    )

    try:
        train_torch_model(
            dataset,
            model,
            num_steps=config.num_steps,
            print_steps=config.print_steps,
            lr=config.lr,
            lr_scheduler=lambda lr: lr,
            batch_size=config.batch_size,
            shuffle_seed=shufflekey,
            output_dir=output_dir,
            device=device,
            dataloader_workers=0,
            mixed_precision=False,
            verbose=True,
            progress_callback=None,
        )
    except Exception as exc:
        print(f"\nReproduced failure: {type(exc).__name__}: {exc}")
        raise

    print("\nRun completed without reproducing a failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
