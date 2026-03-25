"""Execution engine for the SLinOSS sweep package."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import socket
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable

import torch

from data_dir.torch_datasets import TorchDataset
from sweep.results import (
    append_runner_record,
    runner_log_path,
    should_run_trial,
    write_trial_record,
)
from sweep.types import TrialGroup, TrialRecord, TrialSpec
from train_torch import SLinOSSRunSeeds, create_slinoss_dataset, run_slinoss_training


LR_SCHEDULERS: dict[str, Callable[[float], float]] = {
    "identity": lambda lr: lr,
    "constant": lambda lr: lr,
}


def resolve_lr_scheduler(name: str) -> Callable[[float], float]:
    try:
        return LR_SCHEDULERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(LR_SCHEDULERS))
        raise ValueError(
            f"Unknown lr_scheduler '{name}'. Available schedulers: {available}."
        ) from exc


def discover_devices(
    explicit_devices: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if explicit_devices:
        return explicit_devices
    if not torch.cuda.is_available():
        raise RuntimeError("SLinOSS sweep execution requires at least one CUDA device.")
    return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))


def make_runner_id(*, name: str, shard: str | None = None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hostname = socket.gethostname().split(".", maxsplit=1)[0]
    shard_suffix = "all" if shard is None else shard.replace("/", "of")
    return f"{name}-{hostname}-{shard_suffix}-{timestamp}"


def filter_groups_for_execution(
    groups: tuple[TrialGroup, ...],
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> tuple[TrialGroup, ...]:
    runnable_groups: list[TrialGroup] = []
    for group in groups:
        runnable_trials = tuple(
            trial
            for trial in group.trials
            if should_run_trial(trial, force=force, retry_failed=retry_failed)
        )
        if runnable_trials:
            runnable_groups.append(replace(group, trials=runnable_trials))
    return tuple(runnable_groups)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_success_record(
    *,
    trial: TrialSpec,
    runner_id: str,
    device: str,
    started_at: str,
    finished_at: str,
    duration_sec: float,
    completed_steps: int,
    best_validation_metric: float,
    test_metric: float,
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial.trial_id,
        family_id=trial.family_id,
        group_id=trial.group_id,
        status="success",
        dataset_name=trial.dataset.name,
        seed=trial.seed,
        device=device,
        output_dir=trial.output_dir,
        runner_id=runner_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        completed_steps=completed_steps,
        best_validation_metric=best_validation_metric,
        test_metric=test_metric,
    )


def _make_failure_record(
    *,
    trial: TrialSpec,
    runner_id: str,
    device: str,
    started_at: str,
    finished_at: str,
    duration_sec: float,
    error: Exception,
) -> TrialRecord:
    message = f"{error}\n{traceback.format_exc()}".strip()
    return TrialRecord(
        trial_id=trial.trial_id,
        family_id=trial.family_id,
        group_id=trial.group_id,
        status="failed",
        dataset_name=trial.dataset.name,
        seed=trial.seed,
        device=device,
        output_dir=trial.output_dir,
        runner_id=runner_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        error_type=type(error).__name__,
        error_message=message,
    )


def _run_trial(
    *,
    trial: TrialSpec,
    device: str,
    runner_id: str,
    dataset_cache: dict[tuple[str, int], TorchDataset],
) -> TrialRecord:
    torch.cuda.set_device(torch.device(device))
    cache_key = (trial.dataset.name, trial.dataset_seed)
    dataset = dataset_cache.get(cache_key)
    if dataset is None:
        dataset = create_slinoss_dataset(
            data_dir=trial.dataset.data_dir,
            use_presplit=trial.dataset.use_presplit,
            dataset_name=trial.dataset.name,
            include_time=trial.dataset.include_time,
            T=trial.dataset.T,
            dataset_seed=trial.dataset_seed,
        )
        dataset_cache[cache_key] = dataset

    started_at = _now()
    started = datetime.now(timezone.utc)
    try:
        result = run_slinoss_training(
            seed=trial.seed,
            data_dir=trial.dataset.data_dir,
            use_presplit=trial.dataset.use_presplit,
            dataset_name=trial.dataset.name,
            include_time=trial.dataset.include_time,
            T=trial.dataset.T,
            model_args=trial.model.to_dict(),
            num_steps=trial.training.num_steps,
            print_steps=trial.training.print_steps,
            lr=trial.training.lr,
            lr_scheduler=resolve_lr_scheduler(trial.training.lr_scheduler),
            batch_size=trial.training.batch_size,
            output_dir=trial.output_dir,
            dataloader_workers=trial.training.dataloader_workers,
            overwrite_output_dir=os.path.isdir(trial.output_dir),
            allow_tf32=trial.training.allow_tf32,
            mixed_precision=trial.training.mixed_precision,
            check_numerics=trial.training.check_numerics,
            weight_decay=trial.training.weight_decay,
            grad_clip_norm=trial.training.grad_clip_norm,
            early_stopping_patience=trial.training.early_stopping_patience,
            min_steps_before_early_stop=trial.training.min_steps_before_early_stop,
            device=device,
            dataset=dataset,
            run_seeds=SLinOSSRunSeeds(
                dataset_seed=trial.dataset_seed,
                model_seed=trial.model_seed,
                shuffle_seed=trial.shuffle_seed,
            ),
            verbose=False,
            prompt_if_output_dir_exists=False,
        )
        finished = datetime.now(timezone.utc)
        duration_sec = (finished - started).total_seconds()
        record = _make_success_record(
            trial=trial,
            runner_id=runner_id,
            device=device,
            started_at=started_at,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_sec=duration_sec,
            completed_steps=result.summary.completed_steps,
            best_validation_metric=result.summary.best_validation_metric,
            test_metric=result.summary.test_metric,
        )
    except Exception as error:
        finished = datetime.now(timezone.utc)
        duration_sec = (finished - started).total_seconds()
        record = _make_failure_record(
            trial=trial,
            runner_id=runner_id,
            device=device,
            started_at=started_at,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_sec=duration_sec,
            error=error,
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    write_trial_record(trial, record)
    return record


def _worker_main(
    *,
    device: str,
    groups: tuple[TrialGroup, ...],
    result_queue,
    runner_id: str,
) -> None:
    dataset_cache: dict[tuple[str, int], TorchDataset] = {}
    try:
        for group in groups:
            for trial in group.trials:
                record = _run_trial(
                    trial=trial,
                    device=device,
                    runner_id=runner_id,
                    dataset_cache=dataset_cache,
                )
                result_queue.put(record.to_dict())
    finally:
        result_queue.put({"status": "__worker_done__", "device": device})


def execute_groups(
    groups: tuple[TrialGroup, ...],
    *,
    output_root: str,
    devices: tuple[str, ...],
    runner_id: str,
    progress_callback: Callable[[TrialRecord, int, int], None] | None = None,
) -> list[TrialRecord]:
    if not groups:
        return []

    device_groups: list[list[TrialGroup]] = [[] for _ in devices]
    for index, group in enumerate(groups):
        device_groups[index % len(devices)].append(group)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_worker_main,
            kwargs={
                "device": device,
                "groups": tuple(assigned_groups),
                "result_queue": result_queue,
                "runner_id": runner_id,
            },
        )
        for device, assigned_groups in zip(devices, device_groups, strict=True)
        if assigned_groups
    ]

    log_path = runner_log_path(output_root, runner_id)
    for worker in workers:
        worker.start()

    records: list[TrialRecord] = []
    completed_trials = 0
    total_trials = sum(len(group.trials) for group in groups)
    completed_workers = 0
    total_workers = len(workers)
    try:
        while completed_workers < total_workers:
            payload = result_queue.get()
            if payload.get("status") == "__worker_done__":
                completed_workers += 1
                continue
            record = TrialRecord(**payload)
            append_runner_record(log_path, record)
            records.append(record)
            completed_trials += 1
            if progress_callback is not None:
                progress_callback(record, completed_trials, total_trials)
    finally:
        for worker in workers:
            worker.join()

    return records
