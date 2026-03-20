"""Two-GPU wrapper for the SLinOSS hyperparameter sweep."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys

# Ensure repository root is on sys.path when running from sweep_slinoss folder directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sweep_slinoss.sweep_slinoss import DEFAULT_DATASETS, run_sweep


GPU0_DATASETS = {"EigenWorms", "SelfRegulationSCP1", "Heartbeat"}


def _split_datasets(datasets: list[str]) -> tuple[list[str], list[str]]:
    gpu0 = [dataset for dataset in datasets if dataset in GPU0_DATASETS]
    gpu1 = [dataset for dataset in datasets if dataset not in GPU0_DATASETS]
    return gpu0, gpu1


def _run_on_gpu(
    gpu_id: int,
    *,
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    skip_existing: bool,
    show_progress: bool,
    progress_position: int,
    tqdm_lock=None,
    progress_queue=None,
) -> None:
    try:
        # Isolate each worker to a single physical GPU so CUDA extensions that
        # default to local device 0 cannot accidentally collide on GPU 0.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch

        if not datasets:
            if progress_queue is not None:
                progress_queue.put({"type": "total", "value": 0, "worker": f"GPU {gpu_id}"})
            elif not show_progress:
                print(f"[GPU {gpu_id}] No datasets assigned. Worker exiting.")
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU execution was requested.")
        visible_count = torch.cuda.device_count()
        if visible_count < 1:
            raise RuntimeError(
                f"Requested GPU {gpu_id}, but no visible CUDA devices were detected in worker."
            )

        torch.cuda.set_device(0)
        active_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(active_device)
        if not show_progress and progress_queue is None:
            print(
                f"[GPU {gpu_id}] Using CUDA device cuda:{active_device} ({device_name}), "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            print(f"[GPU {gpu_id}] Running datasets: {datasets}")
        run_sweep(
            experiment_folder=experiment_folder,
            datasets=datasets,
            seeds_per_config=seeds_per_config,
            seeds=seeds,
            skip_existing=skip_existing,
            show_progress=show_progress,
            progress_desc=f"GPU {gpu_id}",
            progress_position=progress_position,
            tqdm_lock=tqdm_lock,
            progress_queue=progress_queue,
        )
    finally:
        if progress_queue is not None:
            progress_queue.put({"type": "done", "worker": f"GPU {gpu_id}"})


def _get_progress_message(progress_queue: mp.Queue, timeout: float | None):
    try:
        if timeout is None:
            return progress_queue.get()
        if timeout <= 0:
            return progress_queue.get_nowait()
        return progress_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def run_two_gpu_sweep(
    *,
    experiment_folder: str,
    datasets: list[str],
    seeds_per_config: int | None,
    seeds: list[int] | None,
    skip_existing: bool,
    gpu_ids: tuple[int, int],
    show_progress: bool,
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run a two-GPU sweep.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Two GPUs requested, but only {torch.cuda.device_count()} GPU(s) detected."
        )

    gpu0_datasets, gpu1_datasets = _split_datasets(datasets)
    if not show_progress:
        print(f"GPU {gpu_ids[0]} datasets: {gpu0_datasets}")
        print(f"GPU {gpu_ids[1]} datasets: {gpu1_datasets}")

    tqdm_lock = mp.RLock()
    progress_queue = mp.Queue() if show_progress else None

    workers = [
        mp.Process(
            target=_run_on_gpu,
            kwargs={
                "gpu_id": gpu_ids[0],
                "experiment_folder": experiment_folder,
                "datasets": gpu0_datasets,
                "seeds_per_config": seeds_per_config,
                "seeds": seeds,
                "skip_existing": skip_existing,
                "show_progress": show_progress,
                "progress_position": 0,
                "tqdm_lock": tqdm_lock,
                "progress_queue": progress_queue,
            },
        ),
        mp.Process(
            target=_run_on_gpu,
            kwargs={
                "gpu_id": gpu_ids[1],
                "experiment_folder": experiment_folder,
                "datasets": gpu1_datasets,
                "seeds_per_config": seeds_per_config,
                "seeds": seeds,
                "skip_existing": skip_existing,
                "show_progress": show_progress,
                "progress_position": 1,
                "tqdm_lock": tqdm_lock,
                "progress_queue": progress_queue,
            },
        ),
    ]

    for worker in workers:
        worker.start()

    if show_progress and progress_queue is not None:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if tqdm is not None:
            total_tasks = 0
            totals_received = 0
            pending_updates = 0
            completed_workers = 0
            expected_workers = len(workers)

            while totals_received + completed_workers < expected_workers:
                msg = _get_progress_message(progress_queue, timeout=0.1)
                if msg is None:
                    continue
                if msg["type"] == "total":
                    total_tasks += msg["value"]
                    totals_received += 1
                elif msg["type"] == "update":
                    pending_updates += 1
                elif msg["type"] == "done":
                    completed_workers += 1

            with tqdm(total=total_tasks, desc="Overall Sweep Progress", dynamic_ncols=True) as pbar:
                if pending_updates:
                    pbar.update(pending_updates)

                while completed_workers < expected_workers:
                    msg = _get_progress_message(progress_queue, timeout=0.1)
                    if msg is None:
                        continue
                    if msg["type"] == "total":
                        total_tasks += msg["value"]
                        pbar.total = total_tasks
                        pbar.refresh()
                    elif msg["type"] == "update":
                        pbar.update(1)
                    elif msg["type"] == "done":
                        completed_workers += 1

                while True:
                    msg = _get_progress_message(progress_queue, timeout=0)
                    if msg is None:
                        break
                    if msg["type"] == "total":
                        total_tasks += msg["value"]
                        pbar.total = total_tasks
                        pbar.refresh()
                    elif msg["type"] == "update":
                        pbar.update(1)
        else:
            # tqdm is missing but show_progress was true, just wait
            pass

    for worker in workers:
        worker.join()

    failed_workers = [worker.pid for worker in workers if worker.exitcode != 0]
    if failed_workers:
        raise RuntimeError(f"Sweep workers failed: {failed_workers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the SLinOSS sweep split across two NVIDIA GPUs.",
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="experiment_configs/repeats",
        help="Directory that contains the SLinOSS dataset config JSON files.",
    )
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to include before fixed GPU partitioning.",
    )
    parser.add_argument(
        "--seeds_per_config",
        type=int,
        default=None,
        help="Optional cap on number of seeds per hyperparameter combination.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated list of seed values to use for all configs. Overrides JSON seeds and seeds_per_config.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose output directory already exists.",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs=2,
        type=int,
        default=[0, 1],
        metavar=("GPU_A", "GPU_B"),
        help="Two GPU ids to use for the fixed dataset split.",
    )
    parser.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-GPU tqdm progress bars when tqdm is installed.",
    )
    args = parser.parse_args()

    user_seeds = None
    if args.seeds is not None:
        user_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    run_two_gpu_sweep(
        experiment_folder=args.experiment_folder,
        datasets=args.dataset_name,
        seeds_per_config=args.seeds_per_config,
        seeds=user_seeds,
        skip_existing=args.skip_existing,
        gpu_ids=(args.gpu_ids[0], args.gpu_ids[1]),
        show_progress=args.show_progress,
    )
