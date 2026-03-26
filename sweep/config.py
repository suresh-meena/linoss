"""Config parsing for the SLinOSS sweep package."""

from __future__ import annotations

import json
import os
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, TypeVar, cast

from sweep.types import (
    DatasetConfig,
    DatasetSweepDefinition,
    ModelConfig,
    ResourceProfile,
    SweepDefinition,
    TrialResourceMatch,
    TrialResourceRule,
    TrainingConfig,
)

T = TypeVar("T")


def _merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_nested(base_value, value)
        else:
            merged[key] = value
    return merged


def _normalize_grid(raw_grid: dict[str, Any] | None) -> dict[str, tuple[object, ...]]:
    if not raw_grid:
        return {}

    grid: dict[str, tuple[object, ...]] = {}
    for key, values in raw_grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Grid axis '{key}' must be a non-empty JSON list. Got: {values!r}"
            )
        grid[key] = tuple(values)
    return grid


def _build_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    dataclass_fields = tuple(fields(cast(Any, cls)))
    field_names = {field.name for field in dataclass_fields}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")

    required = [
        field.name
        for field in dataclass_fields
        if field.default is MISSING and field.default_factory is MISSING
    ]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing required {cls.__name__} fields: {missing}")
    return cls(**values)


def _load_resource_profile(
    raw_profile: str | dict[str, Any] | None,
    *,
    config_path: Path,
) -> ResourceProfile | None:
    if raw_profile is None:
        return None

    if isinstance(raw_profile, str):
        profile_path = Path(raw_profile)
        if not profile_path.is_absolute():
            profile_path = config_path.parent / profile_path
        with profile_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif isinstance(raw_profile, dict):
        payload = raw_profile
    else:
        raise ValueError(
            "Top-level 'resource_profile' must be a JSON object or a path string."
        )

    default_tier = payload.get("default_tier")
    if default_tier is not None and not isinstance(default_tier, str):
        raise ValueError("resource_profile.default_tier must be a string.")

    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("resource_profile.rules must be a JSON list.")

    rules: list[TrialResourceRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("Each resource profile rule must be a JSON object.")
        raw_match = raw_rule.get("match")
        if not isinstance(raw_match, dict):
            raise ValueError("Each resource profile rule must define a match object.")
        resource_tier = raw_rule.get("resource_tier")
        if not isinstance(resource_tier, str) or not resource_tier:
            raise ValueError(
                "Each resource profile rule must define a non-empty resource_tier."
            )
        estimated_peak_vram_gb = raw_rule.get("estimated_peak_vram_gb")
        if estimated_peak_vram_gb is not None and not isinstance(
            estimated_peak_vram_gb, (int, float)
        ):
            raise ValueError(
                "resource_profile rule estimated_peak_vram_gb must be numeric."
            )
        note = raw_rule.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError("resource_profile rule note must be a string.")

        rules.append(
            TrialResourceRule(
                match=_build_dataclass(TrialResourceMatch, raw_match),
                resource_tier=resource_tier,
                estimated_peak_vram_gb=(
                    float(estimated_peak_vram_gb)
                    if estimated_peak_vram_gb is not None
                    else None
                ),
                note=note,
            )
        )

    return ResourceProfile(default_tier=default_tier, rules=tuple(rules))


def load_sweep_definition(path: str | os.PathLike[str]) -> SweepDefinition:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Sweep config must define a non-empty top-level 'name'.")

    output_root = raw.get("output_root")
    if output_root is None:
        output_root = os.path.join("outputs", "sweeps", name)
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("Top-level 'output_root' must be a non-empty string.")

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Top-level 'defaults' must be a JSON object when present.")
    global_grid = raw.get("grid", {})
    if not isinstance(global_grid, dict):
        raise ValueError("Top-level 'grid' must be a JSON object when present.")

    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("Sweep config must define a non-empty 'datasets' list.")

    default_seeds = raw.get("seeds")
    if default_seeds is not None and (
        not isinstance(default_seeds, list)
        or not default_seeds
        or not all(isinstance(seed, int) for seed in default_seeds)
    ):
        raise ValueError("Top-level 'seeds' must be a non-empty list of integers.")

    dataset_defs: list[DatasetSweepDefinition] = []
    resource_profile = _load_resource_profile(
        raw.get("resource_profile"),
        config_path=config_path,
    )
    default_dataset_values = defaults.get("dataset", {})
    default_training_values = defaults.get("training", {})
    default_model_values = defaults.get("model", {})
    if not isinstance(default_dataset_values, dict):
        raise ValueError("defaults.dataset must be a JSON object when present.")
    if not isinstance(default_training_values, dict):
        raise ValueError("defaults.training must be a JSON object when present.")
    if not isinstance(default_model_values, dict):
        raise ValueError("defaults.model must be a JSON object when present.")

    for raw_dataset in raw_datasets:
        if not isinstance(raw_dataset, dict):
            raise ValueError("Each dataset entry must be a JSON object.")
        dataset_name = raw_dataset.get("name")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("Each dataset entry must define a non-empty 'name'.")

        seeds = raw_dataset.get("seeds", default_seeds)
        if (
            not isinstance(seeds, list)
            or not seeds
            or not all(isinstance(seed, int) for seed in seeds)
        ):
            raise ValueError(
                f"Dataset '{dataset_name}' must define a non-empty integer 'seeds' list."
            )

        overrides = raw_dataset.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(
                f"Dataset '{dataset_name}' overrides must be a JSON object."
            )
        dataset_grid_values = raw_dataset.get("grid", {})
        if not isinstance(dataset_grid_values, dict):
            raise ValueError(f"Dataset '{dataset_name}' grid must be a JSON object.")

        merged_dataset_values = _merge_nested(
            default_dataset_values,
            overrides.get("dataset", {}),
        )
        merged_training_values = _merge_nested(
            default_training_values,
            overrides.get("training", {}),
        )
        merged_model_values = _merge_nested(
            default_model_values,
            overrides.get("model", {}),
        )
        merged_dataset_values["name"] = dataset_name

        dataset_config = _build_dataclass(DatasetConfig, merged_dataset_values)
        training_config = _build_dataclass(TrainingConfig, merged_training_values)
        model_config = _build_dataclass(ModelConfig, merged_model_values)

        effective_grid = _merge_nested(global_grid, dataset_grid_values)
        dataset_defs.append(
            DatasetSweepDefinition(
                seeds=tuple(seeds),
                dataset=dataset_config,
                training=training_config,
                model=model_config,
                dataset_grid=_normalize_grid(effective_grid.get("dataset")),
                training_grid=_normalize_grid(effective_grid.get("training")),
                model_grid=_normalize_grid(effective_grid.get("model")),
            )
        )

    return SweepDefinition(
        name=name,
        output_root=output_root,
        datasets=tuple(dataset_defs),
        resource_profile=resource_profile,
    )
