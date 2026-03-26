"""CLI for the new SLinOSS sweep package."""

from __future__ import annotations

import argparse
import sys

from sweep.config import load_sweep_definition
from sweep.executor import (
    discover_devices,
    execute_groups,
    filter_groups_for_execution,
    make_runner_id,
)
from sweep.planner import build_sweep_plan, iter_trials, select_groups, summarize_groups
from sweep.results import (
    reduce_plan_results,
    write_manifest,
    write_plan_jsonl,
    write_reduction_outputs,
)


def _parse_devices(raw_devices: str | None) -> tuple[str, ...] | None:
    if raw_devices is None:
        return None
    devices = tuple(
        device.strip() for device in raw_devices.split(",") if device.strip()
    )
    if not devices:
        raise ValueError("Expected at least one CUDA device in --devices.")
    return devices


def _print_selected_trials(groups, *, preview: int) -> None:
    print(
        f"Selected {sum(len(group.trials) for group in groups)} trials "
        f"across {len(groups)} dataset-cache groups."
    )
    for trial in list(iter_trials(groups))[:preview]:
        print(
            "  "
            f"{trial.dataset.name} seed={trial.seed} family={trial.family_id} "
            f"lr={trial.training.lr} batch_size={trial.training.batch_size} "
            f"d_model={trial.model.d_model} n_layers={trial.model.n_layers} "
            f"d_state={trial.model.d_state} "
            f"tier={trial.resource_tier or 'unassigned'}"
        )


def _build_plan_and_materialize(config_path: str):
    definition = load_sweep_definition(config_path)
    plan = build_sweep_plan(definition)
    manifest_path = write_manifest(
        definition=definition,
        plan=plan,
        config_path=config_path,
    )
    plan_path = write_plan_jsonl(plan)
    return definition, plan, manifest_path, plan_path


def _selected_groups_from_args(plan, args):
    dataset_filter = set(args.dataset) if args.dataset else None
    resource_tiers = set(args.resource_tier) if args.resource_tier else None
    return select_groups(
        plan,
        shard=args.shard,
        datasets=dataset_filter,
        resource_tiers=resource_tiers,
        max_groups=args.max_groups,
        max_trials=args.max_trials,
    )


def _command_plan(args) -> int:
    _, plan, manifest_path, plan_path = _build_plan_and_materialize(args.config)
    groups = _selected_groups_from_args(plan, args)
    summary = summarize_groups(groups)
    print(f"Manifest: {manifest_path}")
    print(f"Plan: {plan_path}")
    print(
        f"Planned {len(plan.trials)} total trials in {len(plan.groups)} groups. "
        f"Selection contains {summary['trials']} trials in {summary['groups']} groups."
    )
    _print_selected_trials(groups, preview=args.preview)
    return 0


def _command_run(args) -> int:
    _, plan, manifest_path, plan_path = _build_plan_and_materialize(args.config)
    selected_groups = _selected_groups_from_args(plan, args)
    runnable_groups = filter_groups_for_execution(
        selected_groups,
        force=args.force,
        retry_failed=args.retry_failed,
    )

    selected_trials = sum(len(group.trials) for group in selected_groups)
    runnable_trials = sum(len(group.trials) for group in runnable_groups)
    skipped_trials = selected_trials - runnable_trials
    print(f"Manifest: {manifest_path}")
    print(f"Plan: {plan_path}")
    print(
        f"Selected {selected_trials} trials in {len(selected_groups)} groups. "
        f"Runnable now: {runnable_trials}. Skipped by existing results: {skipped_trials}."
    )
    _print_selected_trials(runnable_groups, preview=args.preview)
    if args.dry_run or runnable_trials == 0:
        return 0

    devices = discover_devices(_parse_devices(args.devices))
    runner_id = make_runner_id(name=plan.name, shard=args.shard)
    print(f"Runner: {runner_id}")
    print(f"Devices: {', '.join(devices)}")

    def progress(record, completed_trials: int, total_trials: int) -> None:
        metric = (
            f"best_val={record.best_validation_metric:.6f} test={record.test_metric:.6f}"
            if record.status == "success"
            and record.best_validation_metric is not None
            and record.test_metric is not None
            else f"error={record.error_type}"
        )
        print(
            f"[{completed_trials}/{total_trials}] {record.status.upper()} "
            f"{record.dataset_name} seed={record.seed} device={record.device} {metric}"
        )

    execute_groups(
        runnable_groups,
        output_root=plan.output_root,
        devices=devices,
        runner_id=runner_id,
        progress_callback=progress,
    )
    return 0


def _command_reduce(args) -> int:
    _, plan, _, _ = _build_plan_and_materialize(args.config)
    summary = reduce_plan_results(plan)
    json_path, csv_path = write_reduction_outputs(plan, summary)
    print(f"Reduction JSON: {json_path}")
    print(f"Reduction CSV: {csv_path}")
    print(f"Status counts: {summary['status_counts']}")
    for dataset, row in summary["best_by_dataset"].items():
        print(
            f"Best {dataset}: family={row['family_id']} "
            f"mean_val={row['mean_best_validation_metric']} "
            f"mean_test={row['mean_test_metric']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sweep",
        description="Deterministic, shardable SLinOSS hyperparameter sweeps.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_selection_flags(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--config", required=True, help="Path to sweep JSON config."
        )
        command_parser.add_argument(
            "--dataset",
            action="append",
            default=[],
            help="Optional dataset filter. Repeat for multiple datasets.",
        )
        command_parser.add_argument(
            "--resource-tier",
            action="append",
            default=[],
            help="Optional resource tier filter. Repeat for multiple tiers.",
        )
        command_parser.add_argument(
            "--shard",
            help="Optional shard selector in one-based form, for example 1/4 or 2/8.",
        )
        command_parser.add_argument(
            "--max-groups",
            type=int,
            help="Optional cap on selected dataset-cache groups.",
        )
        command_parser.add_argument(
            "--max-trials",
            type=int,
            help="Optional cap on selected trials after group selection.",
        )
        command_parser.add_argument(
            "--preview",
            type=int,
            default=10,
            help="Number of selected trials to print as a preview.",
        )

    plan_parser = subparsers.add_parser(
        "plan", help="Expand and inspect the sweep plan."
    )
    add_common_selection_flags(plan_parser)
    plan_parser.set_defaults(handler=_command_plan)

    run_parser = subparsers.add_parser("run", help="Execute selected trials.")
    add_common_selection_flags(run_parser)
    run_parser.add_argument(
        "--devices",
        help="Comma-separated CUDA devices, for example 'cuda:0,cuda:1'. "
        "Defaults to all visible CUDA devices.",
    )
    run_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun trials with an existing failed result record.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all selected trials, replacing existing output directories.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize the plan and print the selected work without executing training.",
    )
    run_parser.set_defaults(handler=_command_run)

    reduce_parser = subparsers.add_parser(
        "reduce",
        help="Aggregate per-trial results into family-level summaries.",
    )
    reduce_parser.add_argument(
        "--config", required=True, help="Path to sweep JSON config."
    )
    reduce_parser.set_defaults(handler=_command_reduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as error:  # pragma: no cover - exercised in CLI usage.
        print(f"error: {error}", file=sys.stderr)
        return 1
