"""CUDA-only SLinOSS classifier blocks for UEA experiments."""

from __future__ import annotations

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

try:
    from slinoss.layers import CuteScanBackend, CuteScanPrepBackend, SLinOSSMixer
    from slinoss.ops.cconv1d import (
        cconv1d_cuda_supported,
        cconv1d_is_available,
        cconv1d_load_error,
    )
except Exception as exc:  # pragma: no cover - exercised on misconfigured installs.
    CuteScanBackend = None
    CuteScanPrepBackend = None
    SLinOSSMixer = None
    cconv1d_cuda_supported = None
    cconv1d_is_available = None
    cconv1d_load_error = None
    _SLINOSS_IMPORT_ERROR = exc
else:
    _SLINOSS_IMPORT_ERROR = None


def ensure_slinoss_cuda_ready() -> None:
    """Fail fast when the required CUDA runtime is not available."""

    if _SLINOSS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "SLinOSS requires the published `slinoss` wheel and its CUDA dependencies "
            "to import cleanly. Reinstall from requirements.txt and verify that the "
            "wheel matches this machine's Python and platform."
        ) from _SLINOSS_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SLinOSS requires a CUDA-capable GPU with a CUDA-enabled PyTorch runtime. "
            "torch.cuda.is_available() is False on this machine."
        )
    if not cconv1d_is_available():
        detail = cconv1d_load_error()
        raise RuntimeError(
            "SLinOSS requires the compiled slinoss CUDA causal-conv extension. "
            "Reinstall the published CUDA wheel and verify that the NVIDIA driver "
            "and CUDA runtime are available."
        ) from detail
    try:
        import_module("cuda")
    except Exception as exc:
        raise RuntimeError(
            "SLinOSS requires the CUDA Python bindings used by the CuTe scan backend. "
            "Verify that `cuda-python` from the `slinoss[cuda]` dependency set is "
            "installed correctly."
        ) from exc
    try:
        import_module("cutlass")
    except Exception as exc:
        raise RuntimeError(
            "SLinOSS requires the NVIDIA CUTLASS DSL runtime for the CuTe scan "
            "backend. This usually means the `nvidia-cutlass-dsl` package or its "
            "JAX compatibility stack is not installed correctly."
        ) from exc


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise RuntimeError(
            f"SLinOSS requires {name} to live on a CUDA device. "
            f"Got device={tensor.device!s}."
        )


class TokenBatchNormEMA(nn.Module):
    """BatchNorm1d with EMA running stats over flattened token features."""

    def __init__(self, d_model: int, *, momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(
            d_model,
            affine=True,
            track_running_stats=True,
            momentum=momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, channels = x.shape
        x = x.reshape(batch * timesteps, channels)
        x = self.bn(x)
        return x.reshape(batch, timesteps, channels)


class SiLUFeedForward(nn.Module):
    """Two-layer feed-forward network with SiLU activations."""

    def __init__(
        self,
        d_model: int,
        *,
        mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
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
    """CUDA-only causal-conv backend that never falls back silently."""

    def __init__(self) -> None:
        self._last_support_key: tuple[object, ...] | None = None

    def __call__(
        self,
        owner: SLinOSSMixer,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_cuda_tensor("inputs", x)
        if conv_state is not None:
            _require_cuda_tensor("convolution state", conv_state)
        support_key = (
            tuple(x.shape),
            x.dtype,
            tuple(owner.dw_weight.shape),
            owner.dw_weight.dtype,
            None if conv_state is None else (tuple(conv_state.shape), conv_state.dtype),
        )
        if support_key != self._last_support_key:
            if not cconv1d_cuda_supported(
                x.transpose(1, 2),
                owner.dw_weight,
                initial_states=conv_state,
                activation=None,
            ):
                raise RuntimeError(
                    "SLinOSS causal-conv CUDA execution is unsupported for the current "
                    "configuration. Expected CUDA float16/bfloat16/float32 tensors and "
                    f"d_conv in {{2, 3, 4}}. Got input device={x.device}, "
                    f"input dtype={x.dtype}, weight dtype={owner.dw_weight.dtype}, "
                    f"d_conv={owner.d_conv}."
                )
            self._last_support_key = support_key
        return owner._apply_cconv_cuda(x, conv_state)


class SLinOSSBlock(nn.Module):
    """Pre-norm residual SLinOSS block with a SiLU feed-forward tail."""

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
        ffn_mult: int = 2,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_bound: float = math.pi,
        k_max: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if d_state % 8 != 0:
            raise ValueError(
                "SLinOSS uses the CuTe scan backend exclusively, which currently "
                f"requires d_state to be a multiple of 8. Got d_state={d_state}."
            )
        if d_conv not in (2, 3, 4):
            raise ValueError(
                "SLinOSS uses the CUDA causal-conv kernel exclusively, which "
                f"currently supports d_conv in {{2, 3, 4}}. Got d_conv={d_conv}."
            )

        self.norm = TokenBatchNormEMA(d_model)
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
        self.feed_forward = SiLUFeedForward(
            d_model,
            mult=ffn_mult,
            dropout=dropout,
        )
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
    """SLinOSS-based classifier for fixed-length UEA time-series inputs."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        d_model: int,
        n_layers: int,
        d_state: int = 128,
        expand: int = 2,
        d_head: int = 64,
        d_conv: int = 4,
        chunk_size: int = 64,
        dropout: float = 0.0,
        ffn_mult: int = 2,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_bound: float = math.pi,
        k_max: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        ensure_slinoss_cuda_ready()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.Sequential(
            *[
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
        self.output_norm = TokenBatchNormEMA(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        _require_cuda_tensor("inputs", x)
        if x.ndim != 3:
            raise ValueError(
                f"Expected inputs with shape (batch, time, channels). Got {tuple(x.shape)}."
            )

        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


SLinOSS = SLinOSSClassifier

__all__ = ["SLinOSS", "SLinOSSClassifier", "ensure_slinoss_cuda_ready"]
