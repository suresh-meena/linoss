from __future__ import annotations

import json

from sweep.config import load_sweep_definition
from sweep.planner import build_sweep_plan, select_groups


def _write_config(path, *, output_root: str) -> None:
    resource_profile_path = path.with_name("resources.json")
    resource_profile_path.write_text(
        json.dumps(
            {
                "default_tier": "rtx3050-6gb",
                "rules": [
                    {
                        "match": {
                            "dataset_name": "EigenWorms",
                            "d_model": 64,
                            "n_layers": 2,
                            "include_time": True,
                        },
                        "resource_tier": "ada6000",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "name": "unit-sweep",
        "output_root": output_root,
        "resource_profile": str(resource_profile_path),
        "defaults": {
            "dataset": {
                "data_dir": "data_dir",
                "use_presplit": False,
                "include_time": True,
                "T": 1.0,
            },
            "training": {
                "num_steps": 20,
                "print_steps": 5,
                "lr": 0.001,
                "batch_size": 2,
            },
            "model": {
                "d_model": 64,
                "n_layers": 2,
            },
        },
        "grid": {
            "training": {"lr": [0.001, 0.0003]},
            "model": {"d_state": [64, 128]},
        },
        "datasets": [
            {
                "name": "EigenWorms",
                "seeds": [111, 222],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_sweep_plan_and_shard_selection(tmp_path) -> None:
    config_path = tmp_path / "grid.json"
    output_root = tmp_path / "outputs"
    _write_config(config_path, output_root=str(output_root))

    definition = load_sweep_definition(config_path)
    plan = build_sweep_plan(definition)

    assert len(plan.trials) == 8
    assert len(plan.groups) == 2
    assert {group.dataset_seed for group in plan.groups}
    assert all(len(group.trials) == 4 for group in plan.groups)

    shard_groups = select_groups(plan, shard="1/2")
    assert len(shard_groups) == 1
    assert len(shard_groups[0].trials) == 4
    assert {trial.seed for trial in shard_groups[0].trials} == {111} or {
        trial.seed for trial in shard_groups[0].trials
    } == {222}

    tier_groups = select_groups(plan, resource_tiers={"ada6000"})
    assert len(tier_groups) == 2
    assert all(len(group.trials) == 1 for group in tier_groups)
    assert all(trial.resource_tier == "ada6000" for group in tier_groups for trial in group.trials)
