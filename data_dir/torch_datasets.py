"""Torch dataset helpers for raw UEA sequence classification."""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass
class TorchDataset:
    """Train/validation/test splits backed by CPU tensors."""

    name: str
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    data_dim: int
    label_dim: int


def _load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _prepend_time_channel(data: np.ndarray, T: float) -> np.ndarray:
    timesteps = data.shape[1]
    ts = (T / timesteps) * np.broadcast_to(
        np.arange(timesteps, dtype=np.float32),
        (data.shape[0], timesteps),
    )
    return np.concatenate([ts[:, :, None], data], axis=2)


def _labels_to_indices(labels: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return labels.argmax(axis=1).astype(np.int64, copy=False), int(labels.shape[1])
    labels = labels.astype(np.int64, copy=False)
    return labels, int(labels.max()) + 1


def _to_tensor_dataset(data: np.ndarray, labels: np.ndarray) -> TensorDataset:
    data = np.ascontiguousarray(data, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    lengths = np.full((data.shape[0],), data.shape[1], dtype=np.int64)
    return TensorDataset(
        torch.from_numpy(data),
        torch.from_numpy(lengths),
        torch.from_numpy(labels),
    )


def _load_presplit_uea_dataset(
    data_dir: str,
    name: str,
    include_time: bool,
    T: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    int,
]:
    required_files = (
        "X_train.pkl",
        "y_train.pkl",
        "X_val.pkl",
        "y_val.pkl",
        "X_test.pkl",
        "y_test.pkl",
    )
    base = os.path.join(data_dir, "processed", "UEA", name)
    missing = [
        filename
        for filename in required_files
        if not os.path.exists(os.path.join(base, filename))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing pre-split UEA files for dataset {name}: {missing}. "
            "Either generate X_train/X_val/X_test splits first or set use_presplit to false."
        )

    train_data = np.asarray(
        _load_pickle(os.path.join(base, "X_train.pkl")),
        dtype=np.float32,
    )
    val_data = np.asarray(
        _load_pickle(os.path.join(base, "X_val.pkl")),
        dtype=np.float32,
    )
    test_data = np.asarray(
        _load_pickle(os.path.join(base, "X_test.pkl")),
        dtype=np.float32,
    )
    train_labels, train_label_dim = _labels_to_indices(
        _load_pickle(os.path.join(base, "y_train.pkl"))
    )
    val_labels, val_label_dim = _labels_to_indices(
        _load_pickle(os.path.join(base, "y_val.pkl"))
    )
    test_labels, test_label_dim = _labels_to_indices(
        _load_pickle(os.path.join(base, "y_test.pkl"))
    )

    if include_time:
        train_data = _prepend_time_channel(train_data, T)
        val_data = _prepend_time_channel(val_data, T)
        test_data = _prepend_time_channel(test_data, T)

    label_dim = max(train_label_dim, val_label_dim, test_label_dim)
    return (
        (train_data, val_data, test_data),
        (train_labels, val_labels, test_labels),
        label_dim,
    )


def _load_random_split_uea_dataset(
    data_dir: str,
    name: str,
    include_time: bool,
    T: float,
    *,
    key,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    int,
]:
    def _cache_token(value) -> str:
        values = np.asarray(value, dtype=np.uint32).reshape(-1)
        if values.size == 0:
            raise ValueError("Random split key must contain at least one value.")
        return "_".join(str(int(v)) for v in values.tolist())

    def _cache_path(base_dir: str, cache_key) -> str:
        cache_dir = os.path.join(base_dir, "_split_index_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{_cache_token(cache_key)}.npz")

    def _valid_indices(indices: np.ndarray, *, size: int) -> bool:
        if indices.ndim != 1:
            return False
        if indices.size == 0:
            return True
        return bool(indices.min() >= 0 and indices.max() < size)

    base = os.path.join(data_dir, "processed", "UEA", name)
    data = np.asarray(_load_pickle(os.path.join(base, "data.pkl")), dtype=np.float32)
    labels, label_dim = _labels_to_indices(
        _load_pickle(os.path.join(base, "labels.pkl"))
    )
    if include_time:
        data = _prepend_time_channel(data, T)

    size = len(data)
    bound1 = int(size * 0.7)
    bound2 = int(size * 0.85)

    split_cache_path = _cache_path(base, key)
    train_idx = val_idx = test_idx = None
    if os.path.exists(split_cache_path):
        try:
            with np.load(split_cache_path) as cached:
                train_idx = np.asarray(cached["train_idx"], dtype=np.int64)
                val_idx = np.asarray(cached["val_idx"], dtype=np.int64)
                test_idx = np.asarray(cached["test_idx"], dtype=np.int64)
            expected_total = size
            actual_total = train_idx.size + val_idx.size + test_idx.size
            if actual_total != expected_total:
                train_idx = val_idx = test_idx = None
            elif not (
                _valid_indices(train_idx, size=size)
                and _valid_indices(val_idx, size=size)
                and _valid_indices(test_idx, size=size)
            ):
                train_idx = val_idx = test_idx = None
        except Exception:
            train_idx = val_idx = test_idx = None

    if train_idx is None or val_idx is None or test_idx is None:
        rng = np.random.default_rng(key)
        perm = rng.permutation(size)

        train_idx = np.asarray(perm[:bound1], dtype=np.int64)
        val_idx = np.asarray(perm[bound1:bound2], dtype=np.int64)
        test_idx = np.asarray(perm[bound2:], dtype=np.int64)

        cache_dir = os.path.dirname(split_cache_path)
        with tempfile.NamedTemporaryFile(
            dir=cache_dir,
            prefix=".split_",
            suffix=".npz",
            delete=False,
        ) as tmp:
            temp_path = tmp.name
        try:
            np.savez(
                temp_path,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
            os.replace(temp_path, split_cache_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return (
        (data[train_idx], data[val_idx], data[test_idx]),
        (labels[train_idx], labels[val_idx], labels[test_idx]),
        label_dim,
    )


def create_torch_dataset(
    data_dir: str,
    name: str,
    use_presplit: bool,
    include_time: bool,
    T: float,
    *,
    key,
) -> TorchDataset:
    processed_uea_dir = os.path.join(data_dir, "processed", "UEA")
    if not os.path.isdir(processed_uea_dir):
        raise FileNotFoundError(
            f"Processed UEA directory not found at {processed_uea_dir}. "
            "Run the UEA preprocessing pipeline before training Torch models."
        )
    available_uea_datasets = {
        entry.name for entry in os.scandir(processed_uea_dir) if entry.is_dir()
    }
    if name == "ppg":
        raise ValueError(
            "SLinOSS is wired up for UEA classification datasets only. "
            "The PPG regression pipeline is not implemented for the Torch trainer."
        )
    if name not in available_uea_datasets:
        raise ValueError(f"Dataset {name} not found in {processed_uea_dir}.")

    if use_presplit:
        split_data, split_labels, label_dim = _load_presplit_uea_dataset(
            data_dir,
            name,
            include_time,
            T,
        )
    else:
        split_data, split_labels, label_dim = _load_random_split_uea_dataset(
            data_dir,
            name,
            include_time,
            T,
            key=key,
        )

    train_data, val_data, test_data = split_data
    train_labels, val_labels, test_labels = split_labels
    data_dim = int(train_data.shape[-1])
    return TorchDataset(
        name=name,
        train=_to_tensor_dataset(train_data, train_labels),
        val=_to_tensor_dataset(val_data, val_labels),
        test=_to_tensor_dataset(test_data, test_labels),
        data_dim=data_dim,
        label_dim=label_dim,
    )
