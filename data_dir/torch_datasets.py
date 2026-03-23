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
    """Train/validation/test splits backed by tensors on a single device."""

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


def move_torch_dataset_to_device(dataset: TorchDataset, device: torch.device) -> TorchDataset:
    """Move all TorchDataset splits onto a target device."""

    def _move_tensor_dataset(tensor_dataset: TensorDataset) -> TensorDataset:
        tensors = getattr(tensor_dataset, "tensors", ())
        if not tensors:
            raise ValueError("TensorDataset must contain at least one tensor.")
        return TensorDataset(
            *tuple(tensor.to(device, non_blocking=device.type == "cuda") for tensor in tensors)
        )

    dataset.train = _move_tensor_dataset(dataset.train)
    dataset.val = _move_tensor_dataset(dataset.val)
    dataset.test = _move_tensor_dataset(dataset.test)
    return dataset


def _split_cache_token(value) -> str:
    values = np.asarray(value, dtype=np.uint32).reshape(-1)
    if values.size == 0:
        raise ValueError("Random split key must contain at least one value.")
    return "_".join(str(int(v)) for v in values.tolist())


def _split_cache_path(base_dir: str, cache_key, *, include_time: bool) -> str:
    cache_dir = os.path.join(base_dir, "_split_cache")
    os.makedirs(cache_dir, exist_ok=True)
    time_flag = "time1" if include_time else "time0"
    return os.path.join(cache_dir, f"v2_{time_flag}_{_split_cache_token(cache_key)}.npz")


def _load_cached_random_split(cache_path: str):
    if not os.path.exists(cache_path):
        return None

    required_keys = (
        "train_data",
        "val_data",
        "test_data",
        "train_labels",
        "val_labels",
        "test_labels",
        "label_dim",
    )
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if any(key not in cached.files for key in required_keys):
                return None
            train_data = np.asarray(cached["train_data"], dtype=np.float32)
            val_data = np.asarray(cached["val_data"], dtype=np.float32)
            test_data = np.asarray(cached["test_data"], dtype=np.float32)
            train_labels = np.asarray(cached["train_labels"], dtype=np.int64)
            val_labels = np.asarray(cached["val_labels"], dtype=np.int64)
            test_labels = np.asarray(cached["test_labels"], dtype=np.int64)
            label_dim = int(np.asarray(cached["label_dim"], dtype=np.int64).item())
    except Exception:
        return None

    if label_dim <= 0:
        return None
    if not (
        train_data.ndim == val_data.ndim == test_data.ndim == 3
        and train_labels.ndim == val_labels.ndim == test_labels.ndim == 1
        and train_data.shape[0] == train_labels.shape[0]
        and val_data.shape[0] == val_labels.shape[0]
        and test_data.shape[0] == test_labels.shape[0]
    ):
        return None

    return (
        (train_data, val_data, test_data),
        (train_labels, val_labels, test_labels),
        label_dim,
    )


def _save_cached_random_split(
    cache_path: str,
    split_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    split_labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    label_dim: int,
) -> None:
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)

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
            train_data=np.ascontiguousarray(split_data[0], dtype=np.float32),
            val_data=np.ascontiguousarray(split_data[1], dtype=np.float32),
            test_data=np.ascontiguousarray(split_data[2], dtype=np.float32),
            train_labels=np.ascontiguousarray(split_labels[0], dtype=np.int64),
            val_labels=np.ascontiguousarray(split_labels[1], dtype=np.int64),
            test_labels=np.ascontiguousarray(split_labels[2], dtype=np.int64),
            label_dim=np.asarray(label_dim, dtype=np.int64),
        )
        os.replace(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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
    base = os.path.join(data_dir, "processed", "UEA", name)
    split_cache_path = _split_cache_path(base, key, include_time=include_time)
    cached_split = _load_cached_random_split(split_cache_path)

    if cached_split is None:
        data = np.asarray(_load_pickle(os.path.join(base, "data.pkl")), dtype=np.float32)
        labels, label_dim = _labels_to_indices(
            _load_pickle(os.path.join(base, "labels.pkl"))
        )
        size = len(data)
        bound1 = int(size * 0.7)
        bound2 = int(size * 0.85)

        rng = np.random.default_rng(key)
        perm = rng.permutation(size)

        train_idx = np.asarray(perm[:bound1], dtype=np.int64)
        val_idx = np.asarray(perm[bound1:bound2], dtype=np.int64)
        test_idx = np.asarray(perm[bound2:], dtype=np.int64)

        split_data = (
            data[train_idx],
            data[val_idx],
            data[test_idx],
        )
        split_labels = (
            labels[train_idx],
            labels[val_idx],
            labels[test_idx],
        )
        if include_time:
            split_data = tuple(_prepend_time_channel(data, T) for data in split_data)
        _save_cached_random_split(split_cache_path, split_data, split_labels, label_dim)
    else:
        split_data, split_labels, label_dim = cached_split

    return split_data, split_labels, label_dim


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
