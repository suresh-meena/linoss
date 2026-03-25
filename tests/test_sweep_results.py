from __future__ import annotations

import json

from sweep.config import load_sweep_definition
from sweep.planner import build_sweep_plan
from sweep.results import reduce_plan_results, write_trial_record
from sweep.types import TrialRecord


def _write_config(path, *, output_root: str) -> None:
    payload = {
        "name": "unit-sweep",
        "output_root": output_root,
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


def test_reduce_plan_results_groups_by_family(tmp_path) -> None:
    config_path = tmp_path / "grid.json"
    output_root = tmp_path / "outputs"
    _write_config(config_path, output_root=str(output_root))

    plan = build_sweep_plan(load_sweep_definition(config_path))
    first_family = plan.trials[0].family_id
    second_family = plan.trials[-1].family_id

    first_family_trials = [
        trial for trial in plan.trials if trial.family_id == first_family
    ]
    second_family_trials = [
        trial for trial in plan.trials if trial.family_id == second_family
    ]

    for index, trial in enumerate(first_family_trials):
        write_trial_record(
            trial,
            TrialRecord(
                trial_id=trial.trial_id,
                family_id=trial.family_id,
                group_id=trial.group_id,
                status="success",
                dataset_name=trial.dataset.name,
                seed=trial.seed,
                device="cuda:0",
                output_dir=trial.output_dir,
                runner_id="runner",
                started_at="2026-03-26T00:00:00+00:00",
                finished_at="2026-03-26T00:00:10+00:00",
                duration_sec=10.0,
                completed_steps=20,
                best_validation_metric=0.8 + (0.1 * index),
                test_metric=0.7 + (0.1 * index),
            ),
        )
    write_trial_record(
        second_family_trials[0],
        TrialRecord(
            trial_id=second_family_trials[0].trial_id,
            family_id=second_family_trials[0].family_id,
            group_id=second_family_trials[0].group_id,
            status="success",
            dataset_name=second_family_trials[0].dataset.name,
            seed=second_family_trials[0].seed,
            device="cuda:0",
            output_dir=second_family_trials[0].output_dir,
            runner_id="runner",
            started_at="2026-03-26T00:00:00+00:00",
            finished_at="2026-03-26T00:00:10+00:00",
            duration_sec=10.0,
            completed_steps=20,
            best_validation_metric=0.5,
            test_metric=0.4,
        ),
    )

    summary = reduce_plan_results(plan)

    assert summary["status_counts"]["success"] == 3
    assert summary["status_counts"]["pending"] == 1
    assert summary["best_by_dataset"]["EigenWorms"]["family_id"] == first_family
