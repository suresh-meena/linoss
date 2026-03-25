"""CUDA-only SLinOSS model blocks for Torch experiments."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

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
    if cconv1d_is_available is None or cconv1d_load_error is None:
        raise RuntimeError(
            "SLinOSS cconv1d CUDA helpers are unavailable. Reinstall the published "
            "`slinoss` wheel and verify that its CUDA extras import cleanly."
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


class BatchNormEMA(nn.Module):
    """Equinox-style EMA BatchNorm over batch and time with no affine parameters."""

    running_mean: torch.Tensor
    running_var: torch.Tensor
    _initialized: torch.Tensor

    def __init__(
        self,
        d_model: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.register_buffer("_initialized", torch.tensor(False, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "BatchNormEMA expects (batch, time, channels) inputs. "
                f"Got shape {tuple(x.shape)}."
            )
        batch, timesteps, channels = x.shape
        flat_x = x.reshape(batch * timesteps, channels)
        stats_x = flat_x.to(dtype=torch.float32)

        if self.training:
            batch_mean = stats_x.mean(dim=0)
            centered = stats_x - batch_mean
            batch_var = centered.square().mean(dim=0).clamp_min_(0.0)
            initialized = bool(self._initialized.item())

            with torch.no_grad():
                if initialized:
                    self.running_mean.mul_(self.momentum).add_(
                        batch_mean.detach(),
                        alpha=1.0 - self.momentum,
                    )
                    self.running_var.mul_(self.momentum).add_(
                        batch_var.detach(),
                        alpha=1.0 - self.momentum,
                    )
                else:
                    self.running_mean.copy_(batch_mean.detach())
                    self.running_var.copy_(batch_var.detach())
                    self._initialized.fill_(True)

            mean = batch_mean
            var = batch_var
        else:
            if bool(self._initialized.item()):
                mean = self.running_mean
                var = self.running_var
            else:
                mean = stats_x.mean(dim=0)
                centered = stats_x - mean
                var = centered.square().mean(dim=0).clamp_min_(0.0)

        normalized = (stats_x - mean) / torch.sqrt(var + self.eps)
        return normalized.to(dtype=x.dtype).reshape(batch, timesteps, channels)


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

    def __call__(
        self,
        owner: Any,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_cuda_tensor("inputs", x)
        if conv_state is not None:
            _require_cuda_tensor("convolution state", conv_state)
        if cconv1d_is_available is None or cconv1d_load_error is None:
            raise RuntimeError(
                "SLinOSS cconv1d CUDA helpers are unavailable. Reinstall the published "
                "`slinoss` wheel and verify that its CUDA extras import cleanly."
            )
        if cconv1d_cuda_supported is None:
            raise RuntimeError(
                "SLinOSS cconv1d CUDA capability checks are unavailable. Reinstall the "
                "published `slinoss` wheel and verify that its CUDA extras import cleanly."
            )
        if not cconv1d_is_available():
            detail = cconv1d_load_error()
            raise RuntimeError(
                "SLinOSS requires the compiled slinoss CUDA causal-conv extension. "
                "The extension is not importable, so the model refuses to fall back "
                "to the reference implementation."
            ) from detail
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
        if (
            SLinOSSMixer is None
            or CuteScanPrepBackend is None
            or CuteScanBackend is None
        ):
            raise RuntimeError(
                "SLinOSS CuTe scan components are unavailable. Reinstall the published "
                "`slinoss` wheel and verify that its CUDA extras import cleanly."
            )

        self.norm = BatchNormEMA(d_model)
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


class SLinOSS(nn.Module):
    """SLinOSS sequence model for classification and stepped sequence outputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        d_model: int,
        n_layers: int,
        classification: bool = True,
        output_step: int = 1,
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
        if classification and output_dim <= 1:
            raise ValueError(
                "SLinOSS classification requires output_dim >= 2. "
                f"Got output_dim={output_dim}."
            )
        if output_step <= 0:
            raise ValueError(
                f"output_step must be positive. Got output_step={output_step}."
            )

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
        self.classification = classification
        self.output_step = output_step
        self.output_norm = BatchNormEMA(d_model)
        self.head = nn.Linear(d_model, output_dim)

    @staticmethod
    def _mean_pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def _downsample_valid_steps(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_outputs = x[:, self.output_step - 1 :: self.output_step]
        if step_outputs.shape[1] == 0:
            empty_mask = torch.zeros(
                (x.shape[0], 0),
                dtype=torch.bool,
                device=x.device,
            )
            return step_outputs, empty_mask

        valid_counts = torch.clamp(
            torch.div(
                lengths - self.output_step,
                self.output_step,
                rounding_mode="floor",
            )
            + 1,
            min=0,
        )
        time_index = torch.arange(step_outputs.shape[1], device=x.device)
        valid_mask = time_index.unsqueeze(0) < valid_counts.unsqueeze(1)
        return step_outputs, valid_mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _require_cuda_tensor("inputs", x)
        if x.ndim != 3:
            raise ValueError(
                f"Expected inputs with shape (batch, time, channels). Got {tuple(x.shape)}."
            )

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        if self.classification:
            pooled = self._mean_pool(x)
            return self.head(pooled)

        _require_cuda_tensor("sequence lengths", lengths)
        if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
            raise ValueError(
                "Expected lengths with shape (batch,). "
                f"Got {tuple(lengths.shape)} for batch size {x.shape[0]}."
            )
        if torch.any(lengths <= 0):
            raise ValueError("All sequence lengths must be positive.")
        if torch.any(lengths > x.shape[1]):
            raise ValueError(
                "Sequence lengths cannot exceed the input time dimension. "
                f"Got max length {int(lengths.max().item())} for time dimension {x.shape[1]}."
            )
        step_outputs, valid_mask = self._downsample_valid_steps(x, lengths)
        predictions = torch.tanh(self.head(step_outputs))
        return predictions.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


SLinOSSClassifier = SLinOSS
TokenBatchNormEMA = BatchNormEMA

__all__ = [
    "SLinOSS",
    "SLinOSSClassifier",
    "BatchNormEMA",
    "TokenBatchNormEMA",
    "ensure_slinoss_cuda_ready",
]
