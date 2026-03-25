"""Deterministic trial planning and sharding for SLinOSS sweeps."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from itertools import product

from train_torch import derive_slinoss_run_seeds
from sweep.types import (
    DatasetConfig,
    SweepDefinition,
    SweepPlan,
    TrialGroup,
    TrialSpec,
)


def _stable_id(prefix: str, payload: object) -> str:
    digest = hashlib.sha1(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _expand_config_grid(base_config, axes: dict[str, tuple[object, ...]]):
    if not axes:
        return (base_config,)

    axis_names = tuple(sorted(axes))
    variants = []
    for values in product(*(axes[name] for name in axis_names)):
        config = base_config
        for name, value in zip(axis_names, values, strict=True):
            config = replace(config, **{name: value})
        variants.append(config)
    return tuple(variants)


def build_sweep_plan(definition: SweepDefinition) -> SweepPlan:
    pending_trials: list[TrialSpec] = []
    for dataset_def in sorted(definition.datasets, key=lambda item: item.dataset.name):
        dataset_variants = _expand_config_grid(
            dataset_def.dataset,
            dataset_def.dataset_grid,
        )
        training_variants = _expand_config_grid(
            dataset_def.training,
            dataset_def.training_grid,
        )
        model_variants = _expand_config_grid(
            dataset_def.model,
            dataset_def.model_grid,
        )

        trial_rows = []
        for dataset_cfg in dataset_variants:
            for training_cfg in training_variants:
                for model_cfg in model_variants:
                    family_id = _stable_id(
                        "family",
                        {
                            "dataset": dataset_cfg.to_dict(),
                            "training": training_cfg.to_dict(),
                            "model": model_cfg.to_dict(),
                        },
                    )
                    for seed in dataset_def.seeds:
                        run_seeds = derive_slinoss_run_seeds(seed)
                        group_id = _stable_id(
                            "group",
                            {
                                "dataset": dataset_cfg.to_dict(),
                                "dataset_seed": run_seeds.dataset_seed,
                            },
                        )
                        trial_id = f"{family_id}-seed-{seed}"
                        output_dir = os.path.join(
                            definition.output_root,
                            "trials",
                            dataset_cfg.name,
                            family_id,
                            f"seed-{seed}",
                        )
                        trial_rows.append(
                            (
                                dataset_cfg.name,
                                family_id,
                                seed,
                                TrialSpec(
                                    index=-1,
                                    trial_id=trial_id,
                                    family_id=family_id,
                                    group_id=group_id,
                                    seed=seed,
                                    dataset_seed=run_seeds.dataset_seed,
                                    model_seed=run_seeds.model_seed,
                                    shuffle_seed=run_seeds.shuffle_seed,
                                    dataset=dataset_cfg,
                                    training=training_cfg,
                                    model=model_cfg,
                                    output_dir=output_dir,
                                ),
                            )
                        )

        for index, (_, _, _, trial) in enumerate(
            sorted(trial_rows), start=len(pending_trials)
        ):
            pending_trials.append(replace(trial, index=index))

    groups_by_id: dict[str, list[TrialSpec]] = {}
    group_dataset: dict[str, DatasetConfig] = {}
    group_dataset_seed: dict[str, int] = {}
    for trial in pending_trials:
        groups_by_id.setdefault(trial.group_id, []).append(trial)
        group_dataset.setdefault(trial.group_id, trial.dataset)
        group_dataset_seed.setdefault(trial.group_id, trial.dataset_seed)

    grouped_rows = []
    for group_id, trials in groups_by_id.items():
        ordered_trials = tuple(
            sorted(
                trials,
                key=lambda trial: (trial.dataset.name, trial.seed, trial.family_id),
            )
        )
        grouped_rows.append(
            (
                group_dataset[group_id].name,
                group_dataset_seed[group_id],
                group_id,
                TrialGroup(
                    index=-1,
                    group_id=group_id,
                    dataset=group_dataset[group_id],
                    dataset_seed=group_dataset_seed[group_id],
                    trials=ordered_trials,
                ),
            )
        )

    groups = tuple(
        replace(group, index=index)
        for index, (_, _, _, group) in enumerate(sorted(grouped_rows))
    )

    return SweepPlan(
        name=definition.name,
        output_root=definition.output_root,
        trials=tuple(pending_trials),
        groups=groups,
    )


def parse_shard_spec(shard: str) -> tuple[int, int]:
    parts = shard.split("/", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid shard spec '{shard}'. Expected the form '1/4' or '2/8'."
        )
    raw_index, raw_total = parts
    index = int(raw_index)
    total = int(raw_total)
    if total <= 0:
        raise ValueError("Shard total must be positive.")
    if index <= 0 or index > total:
        raise ValueError("Shard index must be between 1 and the shard total.")
    return index - 1, total


def select_groups(
    plan: SweepPlan,
    *,
    shard: str | None = None,
    datasets: set[str] | None = None,
    max_groups: int | None = None,
    max_trials: int | None = None,
) -> tuple[TrialGroup, ...]:
    groups = list(plan.groups)
    if datasets:
        groups = [group for group in groups if group.dataset.name in datasets]
    if shard is not None:
        shard_index, shard_total = parse_shard_spec(shard)
        groups = [
            group
            for index, group in enumerate(groups)
            if index % shard_total == shard_index
        ]
    if max_groups is not None:
        groups = groups[:max_groups]
    if max_trials is None:
        return tuple(groups)

    limited_groups: list[TrialGroup] = []
    remaining = max_trials
    for group in groups:
        if remaining <= 0:
            break
        if len(group.trials) <= remaining:
            limited_groups.append(group)
            remaining -= len(group.trials)
            continue
        limited_groups.append(replace(group, trials=group.trials[:remaining]))
        break
    return tuple(limited_groups)


def iter_trials(groups: tuple[TrialGroup, ...]):
    for group in groups:
        yield from group.trials


def summarize_groups(groups: tuple[TrialGroup, ...]) -> dict[str, int]:
    return {
        "groups": len(groups),
        "trials": sum(len(group.trials) for group in groups),
    }
