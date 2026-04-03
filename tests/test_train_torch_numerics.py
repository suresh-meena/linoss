from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import train_torch


class BadModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[0, 0] = float("nan")
        return x


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.bad = BadModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bad(self.proj(x))


def test_forward_nonfinite_tracker_reports_module_name() -> None:
    model = TinyModel()
    tracker = train_torch._ForwardNonfiniteTracker(model)
    try:
        tracker.reset()
        _ = model(torch.ones(1, 2))
        summary = tracker.summary()
    finally:
        tracker.close()

    assert summary is not None
    assert "bad (BadModule)" in summary
    assert "first_bad_index=(0, 0)" in summary


def test_describe_nonfinite_gradients_reports_parameter_owner() -> None:
    model = TinyModel()
    model.proj.weight.grad = torch.zeros_like(model.proj.weight)
    model.proj.weight.grad[0, 0] = float("nan")

    summary = train_torch._describe_nonfinite_gradients(model)

    assert summary is not None
    assert "gradient proj.weight" in summary
    assert "proj (Linear)" in summary


def test_describe_nonfinite_parameters_reports_parameter_owner() -> None:
    model = TinyModel()
    with torch.no_grad():
        model.proj.bias[0] = float("inf")

    summary = train_torch._describe_nonfinite_parameters(model)

    assert summary is not None
    assert "parameter proj.bias" in summary
    assert "proj (Linear)" in summary
