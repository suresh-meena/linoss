"""Typed models for SLinOSS hyperparameter sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: str = "data_dir"
    use_presplit: bool = False
    include_time: bool = True
    T: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    num_steps: int = 100_000
    print_steps: int = 1_000
    lr: float = 1e-3
    lr_scheduler: str = "identity"
    batch_size: int = 2
    allow_tf32: bool = False
    mixed_precision: bool = False
    check_numerics: bool = True
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    early_stopping_patience: int | None = 10
    min_steps_before_early_stop: int | None = None
    dataloader_workers: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_layers: int
    d_state: int = 128
    expand: int = 2
    d_head: int = 64
    d_conv: int = 4
    chunk_size: int = 64
    dropout: float = 0.0
    ffn_mult: int = 2
    dt_min: float = 1e-4
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    r_min: float = 0.9
    r_max: float = 1.0
    theta_bound: float = 3.141592653589793
    k_max: float = 0.5
    eps: float = 1e-8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSweepDefinition:
    seeds: tuple[int, ...]
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    dataset_grid: dict[str, tuple[object, ...]]
    training_grid: dict[str, tuple[object, ...]]
    model_grid: dict[str, tuple[object, ...]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SweepDefinition:
    name: str
    output_root: str
    datasets: tuple[DatasetSweepDefinition, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrialSpec:
    index: int
    trial_id: str
    family_id: str
    group_id: str
    seed: int
    dataset_seed: int
    model_seed: int
    shuffle_seed: int
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrialGroup:
    index: int
    group_id: str
    dataset: DatasetConfig
    dataset_seed: int
    trials: tuple[TrialSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SweepPlan:
    name: str
    output_root: str
    trials: tuple[TrialSpec, ...]
    groups: tuple[TrialGroup, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    family_id: str
    group_id: str
    status: Literal["success", "failed", "skipped"]
    dataset_name: str
    seed: int
    device: str
    output_dir: str
    runner_id: str
    started_at: str
    finished_at: str
    duration_sec: float
    completed_steps: int | None = None
    best_validation_metric: float | None = None
    test_metric: float | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
