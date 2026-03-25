"""Manifest, result recording, and reduction helpers for SLinOSS sweeps."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from sweep.types import SweepDefinition, SweepPlan, TrialRecord, TrialSpec


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_manifest(
    *,
    definition: SweepDefinition,
    plan: SweepPlan,
    config_path: str,
) -> str:
    manifest_path = os.path.join(plan.output_root, "manifest.json")
    _ensure_parent(manifest_path)
    payload = {
        "name": plan.name,
        "config_path": config_path,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_root": plan.output_root,
        "trial_count": len(plan.trials),
        "group_count": len(plan.groups),
        "definition": definition.to_dict(),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def write_plan_jsonl(plan: SweepPlan) -> str:
    plan_path = os.path.join(plan.output_root, "plan.jsonl")
    _ensure_parent(plan_path)
    with open(plan_path, "w", encoding="utf-8") as handle:
        for trial in plan.trials:
            handle.write(json.dumps(trial.to_dict(), sort_keys=True))
            handle.write("\n")
    return plan_path


def runner_log_path(output_root: str, runner_id: str) -> str:
    return os.path.join(output_root, "results", f"{runner_id}.jsonl")


def trial_record_path(trial: TrialSpec) -> str:
    return os.path.join(trial.output_dir, "result.json")


def trial_spec_path(trial: TrialSpec) -> str:
    return os.path.join(trial.output_dir, "trial.json")


def load_trial_record(trial: TrialSpec) -> TrialRecord | None:
    path = trial_record_path(trial)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TrialRecord(**payload)


def write_trial_record(
    trial: TrialSpec,
    record: TrialRecord,
    *,
    log_path: str | None = None,
) -> None:
    os.makedirs(trial.output_dir, exist_ok=True)
    with open(trial_spec_path(trial), "w", encoding="utf-8") as handle:
        json.dump(trial.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(trial_record_path(trial), "w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    if log_path is not None:
        append_runner_record(log_path, record)


def append_runner_record(log_path: str, record: TrialRecord) -> None:
    _ensure_parent(log_path)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True))
        handle.write("\n")


def should_run_trial(
    trial: TrialSpec,
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> bool:
    if force:
        return True
    record = load_trial_record(trial)
    if record is None:
        return True
    if record.status == "failed":
        return retry_failed
    return False


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _safe_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(pstdev(values))


def reduce_plan_results(plan: SweepPlan) -> dict[str, Any]:
    records = {
        trial.trial_id: record
        for trial in plan.trials
        if (record := load_trial_record(trial)) is not None
    }
    status_counts = defaultdict(int)
    family_rows: list[dict[str, Any]] = []
    grouped_trials: dict[str, list[tuple[TrialSpec, TrialRecord | None]]] = defaultdict(
        list
    )
    for trial in plan.trials:
        record = records.get(trial.trial_id)
        if record is not None:
            status_counts[record.status] += 1
        else:
            status_counts["pending"] += 1
        grouped_trials[trial.family_id].append((trial, record))

    for family_id, entries in grouped_trials.items():
        representative = entries[0][0]
        success_records = [
            record
            for _, record in entries
            if record is not None and record.status == "success"
        ]
        validation_values = [
            record.best_validation_metric
            for record in success_records
            if record.best_validation_metric is not None
        ]
        test_values = [
            record.test_metric
            for record in success_records
            if record.test_metric is not None
        ]
        family_rows.append(
            {
                "dataset": representative.dataset.name,
                "family_id": family_id,
                "completed_trials": len(success_records),
                "planned_trials": len(entries),
                "mean_best_validation_metric": _safe_mean(validation_values),
                "std_best_validation_metric": _safe_stdev(validation_values),
                "mean_test_metric": _safe_mean(test_values),
                "std_test_metric": _safe_stdev(test_values),
                "training": representative.training.to_dict(),
                "model": representative.model.to_dict(),
            }
        )

    def _sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
        validation = row["mean_best_validation_metric"]
        test_metric = row["mean_test_metric"]
        return (
            float("-inf")
            if validation is None or math.isnan(validation)
            else validation,
            float("-inf")
            if test_metric is None or math.isnan(test_metric)
            else test_metric,
            row["family_id"],
        )

    leaderboard = sorted(family_rows, key=_sort_key, reverse=True)
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for row in leaderboard:
        best_by_dataset.setdefault(row["dataset"], row)

    return {
        "name": plan.name,
        "output_root": plan.output_root,
        "status_counts": dict(status_counts),
        "leaderboard": leaderboard,
        "best_by_dataset": best_by_dataset,
    }


def write_reduction_outputs(
    plan: SweepPlan, summary: dict[str, Any]
) -> tuple[str, str]:
    report_dir = os.path.join(plan.output_root, "reports")
    os.makedirs(report_dir, exist_ok=True)

    json_path = os.path.join(report_dir, "family_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_path = os.path.join(report_dir, "family_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "family_id",
                "completed_trials",
                "planned_trials",
                "mean_best_validation_metric",
                "std_best_validation_metric",
                "mean_test_metric",
                "std_test_metric",
                "training",
                "model",
            ],
        )
        writer.writeheader()
        for row in summary["leaderboard"]:
            writer.writerow(row)
    return json_path, csv_path
