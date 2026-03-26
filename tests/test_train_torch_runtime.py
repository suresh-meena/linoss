from __future__ import annotations

import importlib

import train_torch


def test_train_torch_forces_jax_cpu_runtime(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

    importlib.reload(train_torch)

    assert train_torch.os.environ["JAX_PLATFORMS"] == "cpu"
    assert train_torch.os.environ["JAX_PLATFORM_NAME"] == "cpu"
    assert train_torch.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
