"""Analyze SLinOSS sweep outputs and rank configs across seeds."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass

import numpy as np


SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


@dataclass
class RunRecord:
    dataset: str
    config_id: str
    seed: int
    test_metric: float
    best_val_metric: float
    last_val_metric: float
    eval_count: int
    run_dir: str


def _strip_seed_suffix(run_dir_name: str) -> str:
    return SEED_SUFFIX_RE.sub("", run_dir_name)


def _extract_seed(run_dir_name: str) -> int:
    marker = "_seed"
    idx = run_dir_name.rfind(marker)
    if idx == -1:
        raise ValueError(f"Run directory does not contain seed suffix: {run_dir_name}")
    return int(run_dir_name[idx + len(marker) :])


def _load_scalar(path: str) -> float:
    value = np.load(path)
    if np.isscalar(value):
        return float(value)
    arr = np.asarray(value).reshape(-1)
    return float(arr[-1])


def _load_vector(path: str) -> np.ndarray:
    value = np.load(path)
    return np.asarray(value).reshape(-1)


def collect_runs(outputs_root: str) -> list[RunRecord]:
    runs: list[RunRecord] = []
    if not os.path.isdir(outputs_root):
        return runs

    for dataset in sorted(os.listdir(outputs_root)):
        dataset_dir = os.path.join(outputs_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        for run_name in sorted(os.listdir(dataset_dir)):
            run_dir = os.path.join(dataset_dir, run_name)
            if not os.path.isdir(run_dir):
                continue

            test_metric_path = os.path.join(run_dir, "test_metric.npy")
            val_metric_path = os.path.join(run_dir, "all_val_metric.npy")
            if not os.path.isfile(test_metric_path) or not os.path.isfile(val_metric_path):
                continue

            val_metrics = _load_vector(val_metric_path)
            if val_metrics.size == 0:
                continue

            try:
                seed = _extract_seed(run_name)
            except ValueError:
                continue

            runs.append(
                RunRecord(
                    dataset=dataset,
                    config_id=_strip_seed_suffix(run_name),
                    seed=seed,
                    test_metric=_load_scalar(test_metric_path),
                    best_val_metric=float(np.max(val_metrics)),
                    last_val_metric=float(val_metrics[-1]),
                    eval_count=int(val_metrics.size - 1),
                    run_dir=run_dir,
                )
            )
    return runs


def summarize_runs(runs: list[RunRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in runs:
        key = (record.dataset, record.config_id)
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, object]] = []
    for (dataset, config_id), records in grouped.items():
        test_values = np.array([r.test_metric for r in records], dtype=float)
        best_val_values = np.array([r.best_val_metric for r in records], dtype=float)
        eval_counts = np.array([r.eval_count for r in records], dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "config_id": config_id,
                "num_seeds": int(len(records)),
                "mean_test": float(np.mean(test_values)),
                "std_test": float(np.std(test_values)),
                "mean_best_val": float(np.mean(best_val_values)),
                "std_best_val": float(np.std(best_val_values)),
                "mean_eval_count": float(np.mean(eval_counts)),
                "best_single_seed_test": float(np.max(test_values)),
            }
        )

    rows.sort(key=lambda row: (row["dataset"], -row["mean_test"], row["std_test"]))
    return rows


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: list[dict[str, object]], top_k: int) -> None:
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    for dataset, dataset_rows in by_dataset.items():
        print(f"\nDataset: {dataset}")
        for rank, row in enumerate(dataset_rows[:top_k], start=1):
            print(
                f"  #{rank} mean_test={row['mean_test']:.6f} +/- {row['std_test']:.6f}, "
                f"seeds={row['num_seeds']}, mean_best_val={row['mean_best_val']:.6f}"
            )
            print(f"     {row['config_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SLinOSS sweep outputs and rank configs by mean test score.",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="outputs/SLinOSS",
        help="Root directory containing per-dataset SLinOSS run folders.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top configs to print per dataset.",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="outputs/SLinOSS/sweep_summary.csv",
        help="Path to write full aggregated summary as CSV.",
    )
    args = parser.parse_args()

    runs = collect_runs(args.outputs_root)
    if not runs:
        print(f"No sweep runs found under: {args.outputs_root}")
        return

    rows = summarize_runs(runs)
    write_csv(args.csv_out, rows)
    print_top(rows, args.top_k)
    print(f"\nWrote summary CSV: {args.csv_out}")


if __name__ == "__main__":
    main()