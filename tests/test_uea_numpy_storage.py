from __future__ import annotations

import pickle

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from data_dir.datasets import create_uea_dataset
from data_dir.process_uea import rewrite_processed_pickles_to_numpy
from data_dir.torch_datasets import create_torch_dataset


def _save_pickle(path, value) -> None:
    with open(path, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_numpy_uea_dataset(tmp_path) -> None:
    dataset_dir = tmp_path / "processed" / "UEA" / "ToySet"
    dataset_dir.mkdir(parents=True)

    data = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
    labels = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
    original_idxs = (
        np.arange(3, dtype=np.int64),
        np.arange(3, 5, dtype=np.int64),
    )

    _save_pickle(dataset_dir / "data.pkl", data)
    _save_pickle(dataset_dir / "labels.pkl", labels)
    _save_pickle(dataset_dir / "original_idxs.pkl", original_idxs)


def test_rewrite_processed_uea_pickles_to_numpy(tmp_path) -> None:
    dataset_dir = tmp_path / "processed" / "UEA" / "ToySet"
    dataset_dir.mkdir(parents=True)

    _save_pickle(
        dataset_dir / "data.pkl",
        jnp.arange(5 * 4 * 2, dtype=jnp.float32).reshape(5, 4, 2),
    )
    _save_pickle(
        dataset_dir / "labels.pkl",
        jnp.asarray([0, 1, 0, 1, 0], dtype=jnp.int32),
    )
    _save_pickle(
        dataset_dir / "original_idxs.pkl",
        (
            jnp.arange(3, dtype=jnp.int32),
            jnp.arange(3, 5, dtype=jnp.int32),
        ),
    )

    rewrite_processed_pickles_to_numpy(str(tmp_path))

    with open(dataset_dir / "data.pkl", "rb") as handle:
        data = pickle.load(handle)
    with open(dataset_dir / "labels.pkl", "rb") as handle:
        labels = pickle.load(handle)
    with open(dataset_dir / "original_idxs.pkl", "rb") as handle:
        original_idxs = pickle.load(handle)

    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(original_idxs, tuple)
    assert all(isinstance(indices, np.ndarray) for indices in original_idxs)


def test_torch_loader_reads_numpy_backed_uea_dataset(tmp_path) -> None:
    _write_numpy_uea_dataset(tmp_path)

    dataset = create_torch_dataset(
        str(tmp_path),
        "ToySet",
        use_presplit=False,
        include_time=False,
        T=1.0,
        key=np.asarray([1, 2], dtype=np.uint32),
    )

    assert len(dataset.train) == 3
    assert len(dataset.val) == 1
    assert len(dataset.test) == 1
    sample, label = dataset.train[0]
    assert tuple(sample.shape) == (4, 2)
    assert label.item() in {0, 1}


def test_jax_loader_reads_numpy_backed_uea_dataset(tmp_path) -> None:
    _write_numpy_uea_dataset(tmp_path)

    dataset = create_uea_dataset(
        str(tmp_path),
        "ToySet",
        use_idxs=False,
        use_presplit=False,
        stepsize=2,
        depth=2,
        include_time=False,
        T=1.0,
        key=jr.PRNGKey(0),
    )

    assert dataset.data_dim == 2
    assert dataset.label_dim == 2
    assert dataset.raw_dataloaders["train"].size == 3
    batch_data, batch_labels = next(dataset.raw_dataloaders["train"].loop_epoch(2))
    assert not isinstance(batch_data, tuple)
    assert tuple(batch_data.shape[1:]) == (4, 2)
    assert batch_labels.shape[1] == 2
