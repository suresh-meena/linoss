"""Torch training entrypoint for raw sequence classification models."""

from __future__ import annotations

import os
import shutil
import time
import hashlib
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, TextIO

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_dir.torch_datasets import TorchDataset, create_torch_dataset
from models.generate_torch_model import create_torch_model

# Prevent file descriptor leaks in sweeps
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class _RunLogger:
    path: str
    verbose: bool
    _handle: TextIO

    def _format(self, message: str) -> str:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        return f"[{timestamp}] {message.rstrip()}"

    def log(self, message: str) -> None:
        line = self._format(message)
        self._handle.write(line + "\n")
        self._handle.flush()
        if self.verbose:
            print(line, flush=True)

    def log_block(self, text: str) -> None:
        for line in text.rstrip("\n").splitlines():
            self.log(line)

    def exception(self, message: str) -> None:
        self.log(message)
        self.log_block(traceback.format_exc())

    def close(self) -> None:
        try:
            self._handle.flush()
        finally:
            self._handle.close()


@dataclass(frozen=True)
class TorchTrainingSummary:
    output_dir: str
    completed_steps: int
    best_validation_metric: float
    test_metric: float
    train_loss_history: tuple[float, ...]
    validation_metric_history: tuple[float, ...]
    elapsed_time_history: tuple[float, ...]


@dataclass(frozen=True)
class SLinOSSRunResult:
    model: torch.nn.Module
    output_dir: str
    summary: TorchTrainingSummary


def _create_run_logger(output_dir: str, *, verbose: bool) -> _RunLogger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")
    handle = open(log_path, "a", encoding="utf-8")
    logger = _RunLogger(path=log_path, verbose=verbose, _handle=handle)
    logger.log(f"Training log initialized at {log_path}")
    return logger


def _configure_cuda_training_runtime(
    *,
    allow_tf32: bool = False,
    logger: _RunLogger | None = None,
) -> None:
    if not torch.cuda.is_available():
        return

    if allow_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if logger is not None:
            logger.log(
                "Enabled CUDA TF32 matmul/cudnn execution for faster float32 training."
            )
        return

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if logger is not None:
        logger.log("Using strict FP32 matmul/cudnn execution (TF32 disabled).")


def _key_to_seed(key) -> int:
    values = np.asarray(key, dtype=np.uint32)
    if values.size < 2:
        seed = np.uint32(values[0])
    else:
        # Map a 64-bit key to a 32-bit seed accepted by numpy/torch.
        seed = np.uint32(values[0] ^ values[1])
    return int(seed)


def _prepare_output_dir(
    output_dir: str,
    *,
    overwrite: bool = False,
    auto_confirm: bool = False,
    prompt_if_exists: bool = True,
    verbose: bool = True,
) -> None:
    if os.path.isdir(output_dir):
        if overwrite or auto_confirm:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            if verbose:
                print(
                    f"Directory {output_dir} has been deleted and recreated.",
                    flush=True,
                )
            return

        if prompt_if_exists:
            user_input = input(
                f"Warning: Output directory {output_dir} already exists. "
                "Do you want to delete it? (yes/no): "
            )
            if user_input.lower() == "yes":
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                if verbose:
                    print(
                        f"Directory {output_dir} has been deleted and recreated.",
                        flush=True,
                    )
                return
            raise ValueError(f"Directory {output_dir} already exists. Exiting.")

        raise FileExistsError(
            f"Output directory {output_dir} already exists. "
            "Pass overwrite_output_dir=True to replace it."
        )

    os.makedirs(output_dir)
    if verbose:
        print(f"Directory {output_dir} has been created.", flush=True)


def _build_output_dir(
    *,
    seed: int,
    T: float,
    include_time: bool,
    num_steps: int,
    lr: float,
    model_name: str,
    stepsize: int,
    logsig_depth: int,
    model_args: dict,
) -> str:
    def _format_value(value) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _abbrev_key(key: str) -> str:
        aliases = {
            "d_model": "dM",
            "n_layers": "nL",
            "d_state": "dS",
            "expand": "x",
            "d_head": "dH",
            "d_conv": "dC",
            "chunk_size": "cS",
            "dropout": "do",
            "ffn_mult": "ffM",
            "dt_min": "dtMn",
            "dt_max": "dtMx",
            "dt_init_floor": "dtIF",
            "r_min": "rMn",
            "r_max": "rMx",
            "theta_bound": "thB",
            "k_max": "kMx",
            "eps": "eps",
        }
        return aliases.get(key, key)

    # Build a deterministic short output name that stays informative.
    base = f"T{_format_value(T)}_time{_format_value(include_time)}_steps{num_steps}_lr{_format_value(lr)}"
    if model_name in {"log_ncde", "nrde"}:
        base += f"_stepsize{stepsize}_depth{logsig_depth}"

    params = []
    for key, value in model_args.items():
        name = _format_value(value)
        if "(" in name:
            name = name.split("(", 1)[0]
        params.append(f"{_abbrev_key(key)}{name}")

    full_name = f"{model_name}_{base}_{'_'.join(params)}_seed{seed}"

    # Stay safely below common per-component filename limits.
    if len(full_name) > 240:
        digest = hashlib.sha256(full_name.encode("utf-8")).hexdigest()[:10]
        compact_prefix = [model_name, base, *params[:4], f"seed{seed}"]
        return f"{'_'.join(compact_prefix)}_{digest}"

    return full_name


def build_slinoss_output_dir(
    *,
    seed: int,
    dataset_name: str,
    output_parent_dir: str,
    T: float,
    include_time: bool,
    num_steps: int,
    lr: float,
    model_args: dict,
) -> str:
    run_dir_name = _build_output_dir(
        seed=seed,
        T=T,
        include_time=include_time,
        num_steps=num_steps,
        lr=lr,
        model_name="SLinOSS",
        stepsize=1,
        logsig_depth=1,
        model_args=model_args,
    )
    return os.path.join(
        output_parent_dir,
        "outputs",
        "SLinOSS",
        dataset_name,
        run_dir_name,
    )


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None = None,
    pin_memory: bool,
    num_workers: int | None = None,
) -> DataLoader:
    # Always use 0 workers for in-memory TensorDatasets to avoid IPC overhead
    # and CUDA/spawn instability in PyTorch multiprocessing.
    worker_count = 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        num_workers=worker_count,
        pin_memory=pin_memory,
    )


def _move_tensor_dataset_to_device(dataset, device: torch.device):
    if not hasattr(dataset, "tensors") or len(dataset.tensors) == 0:
        return dataset
    if all(t.device == device for t in dataset.tensors):
        return dataset
    return type(dataset)(
        *[t.to(device=device, non_blocking=True) for t in dataset.tensors]
    )


def _move_dataset_to_device(
    dataset: TorchDataset, device: torch.device
) -> TorchDataset:
    dataset.train = _move_tensor_dataset_to_device(dataset.train, device)
    dataset.val = _move_tensor_dataset_to_device(dataset.val, device)
    dataset.test = _move_tensor_dataset_to_device(dataset.test, device)
    return dataset


def _iter_tensor_batches(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None = None,
):
    if not hasattr(dataset, "tensors") or len(dataset.tensors) == 0:
        raise ValueError("Expected a TensorDataset-backed split for tensor batching.")

    tensors = dataset.tensors
    size = tensors[0].shape[0]
    limit = size if not drop_last else size - (size % batch_size)
    if limit <= 0:
        return

    if shuffle:
        indices = torch.randperm(size, generator=generator)
        if indices.device != tensors[0].device:
            indices = indices.to(device=tensors[0].device, non_blocking=True)
        for start in range(0, limit, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield tuple(
                torch.index_select(tensor, 0, batch_indices) for tensor in tensors
            )
        return

    for start in range(0, limit, batch_size):
        stop = min(start + batch_size, size)
        yield tuple(tensor[start:stop] for tensor in tensors)


def _infinite_tensor_batches(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None = None,
):
    while True:
        yield from _iter_tensor_batches(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )


def _infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _tensor_batches_factory(
    dataset,
    *,
    batch_size: int,
):
    def factory():
        return _iter_tensor_batches(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return factory


def _loader_batches_factory(loader: DataLoader):
    def factory():
        return loader

    return factory


def _close_iterator(iterator) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _first_nonfinite_index(tensor: torch.Tensor) -> tuple[int, ...] | None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return None
    bad = (~finite_mask).nonzero(as_tuple=False)
    return tuple(int(i) for i in bad[0].tolist())


@torch.no_grad()
def _evaluate_accuracy(
    model,
    batches,
    device: torch.device,
    *,
    mixed_precision: bool = False,
) -> float:
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.int64)
    total = 0
    use_amp = mixed_precision and device.type == "cuda"
    for x, lengths, labels in batches:
        if x.device != device:
            x = x.to(device=device, non_blocking=True)
        if lengths.device != device:
            lengths = lengths.to(device=device, non_blocking=True)
        if labels.device != device:
            labels = labels.to(device=device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            logits = model(x, lengths)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum()
        total += int(labels.shape[0])
    if total == 0:
        return 0.0
    return float(correct.float().div(total).item())


def _make_optimizer(
    model,
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
    logger: _RunLogger,
):
    if device.type == "cuda":
        try:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=True,
            )
            logger.log("Using fused Adam optimizer.")
            return optimizer
        except (TypeError, RuntimeError):
            logger.log("Fused Adam unavailable; falling back to standard Adam.")
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _save_metrics(
    output_dir: str,
    print_steps: int,
    all_train_metric: list[float],
    all_val_metric: list[float],
    all_time: list[float],
    test_metric: float,
) -> None:
    steps = np.arange(
        print_steps, (len(all_train_metric) + 1) * print_steps, print_steps
    )
    np.save(os.path.join(output_dir, "steps.npy"), steps)
    np.save(
        os.path.join(output_dir, "all_train_metric.npy"),
        np.asarray(all_train_metric),
    )
    np.save(
        os.path.join(output_dir, "all_val_metric.npy"),
        np.asarray(all_val_metric),
    )
    np.save(os.path.join(output_dir, "all_time.npy"), np.asarray(all_time))
    np.save(os.path.join(output_dir, "test_metric.npy"), np.asarray(test_metric))


def _set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_torch_training_summary(output_dir: str) -> TorchTrainingSummary:
    steps = np.load(os.path.join(output_dir, "steps.npy"))
    train_loss = np.load(os.path.join(output_dir, "all_train_metric.npy"))
    validation = np.load(os.path.join(output_dir, "all_val_metric.npy"))
    elapsed = np.load(os.path.join(output_dir, "all_time.npy"))
    test_metric = np.load(os.path.join(output_dir, "test_metric.npy"))

    completed_steps = int(steps[-1]) if steps.size else 0
    best_validation_metric = (
        float(np.max(validation)) if validation.size else float("-inf")
    )
    return TorchTrainingSummary(
        output_dir=output_dir,
        completed_steps=completed_steps,
        best_validation_metric=best_validation_metric,
        test_metric=float(np.asarray(test_metric).item()),
        train_loss_history=tuple(float(x) for x in np.asarray(train_loss).tolist()),
        validation_metric_history=tuple(
            float(x) for x in np.asarray(validation).tolist()
        ),
        elapsed_time_history=tuple(float(x) for x in np.asarray(elapsed).tolist()),
    )


def train_torch_model(
    dataset: TorchDataset,
    model,
    *,
    num_steps: int,
    print_steps: int,
    lr: float,
    lr_scheduler,
    batch_size: int,
    shuffle_seed: int,
    output_dir: str,
    device: torch.device,
    dataloader_workers: int | None = None,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    early_stopping_patience: int | None = 10,
    min_steps_before_early_stop: int | None = None,
    mixed_precision: bool = False,
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    logger: _RunLogger | None = None,
    check_numerics: bool = False,
):
    owns_logger = False
    if logger is None:
        logger = _create_run_logger(output_dir, verbose=verbose)
        owns_logger = True

    if batch_size > len(dataset.train):
        logger.log(
            "Invalid training batch size "
            f"{batch_size}; dataset only has {len(dataset.train)} samples."
        )
        if owns_logger:
            logger.close()
        raise ValueError("Batch size larger than training dataset size.")
    if dataloader_workers is not None and int(dataloader_workers) < 0:
        logger.log(
            "Invalid dataloader_workers value "
            f"{dataloader_workers}; expected a non-negative integer."
        )
        if owns_logger:
            logger.close()
        raise ValueError(
            f"dataloader_workers must be >= 0 when provided. Got {dataloader_workers}."
        )
    if weight_decay < 0:
        logger.log(f"Invalid weight_decay value {weight_decay}; expected >= 0.")
        if owns_logger:
            logger.close()
        raise ValueError("weight_decay must be >= 0.")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        logger.log(
            f"Invalid grad_clip_norm value {grad_clip_norm}; expected > 0 or None."
        )
        if owns_logger:
            logger.close()
        raise ValueError("grad_clip_norm must be > 0 when provided.")
    if early_stopping_patience is not None and early_stopping_patience < 0:
        logger.log(
            "Invalid early_stopping_patience value "
            f"{early_stopping_patience}; expected >= 0 or None."
        )
        if owns_logger:
            logger.close()
        raise ValueError("early_stopping_patience must be >= 0 when provided.")
    if min_steps_before_early_stop is not None and min_steps_before_early_stop < 0:
        logger.log(
            "Invalid min_steps_before_early_stop value "
            f"{min_steps_before_early_stop}; expected >= 0 or None."
        )
        if owns_logger:
            logger.close()
        raise ValueError("min_steps_before_early_stop must be >= 0 when provided.")

    dataset_is_on_gpu = False
    if hasattr(dataset.train, "tensors") and len(dataset.train.tensors) > 0:
        dataset_is_on_gpu = dataset.train.tensors[0].is_cuda

    if device.type == "cuda" and not dataset_is_on_gpu:
        logger.log(f"Moving dataset {dataset.name} onto {device} before training.")
        dataset = _move_dataset_to_device(dataset, device)
        dataset_is_on_gpu = True

    pin_memory = device.type == "cuda" and not dataset_is_on_gpu
    train_generator = torch.Generator()
    train_generator.manual_seed(shuffle_seed)
    use_amp = mixed_precision and device.type == "cuda"
    effective_lr = lr_scheduler(lr)
    use_tensor_batching = dataset_is_on_gpu
    early_stop_after = (
        print_steps
        if min_steps_before_early_stop is None
        else min_steps_before_early_stop
    )

    logger.log(
        "Starting training run: "
        f"output_dir={output_dir}, device={device}, batch_size={batch_size}, "
        f"num_steps={num_steps}, print_steps={print_steps}, lr={effective_lr}, "
        f"weight_decay={weight_decay}, grad_clip_norm={grad_clip_norm}, "
        f"early_stopping_patience={early_stopping_patience}, "
        f"min_steps_before_early_stop={early_stop_after}, "
        f"mixed_precision={mixed_precision}, dataset_on_gpu={dataset_is_on_gpu}, "
        f"tensor_batching={use_tensor_batching}, check_numerics={check_numerics}"
    )
    logger.log(
        "Dataset sizes: "
        f"train={len(dataset.train)}, val={len(dataset.val)}, test={len(dataset.test)}, "
        f"data_dim={dataset.data_dim}, label_dim={dataset.label_dim}"
    )

    train_loader = None
    val_loader = None
    test_loader = None
    if use_tensor_batching:
        train_batches = _infinite_tensor_batches(
            dataset.train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=train_generator,
        )
        val_batches_factory = _tensor_batches_factory(
            dataset.val,
            batch_size=batch_size,
        )
        test_batches_factory = _tensor_batches_factory(
            dataset.test,
            batch_size=batch_size,
        )
    else:
        train_loader = _make_loader(
            dataset.train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=train_generator,
            pin_memory=pin_memory,
            num_workers=dataloader_workers,
        )
        val_loader = _make_loader(
            dataset.val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=0,
        )
        test_loader = _make_loader(
            dataset.test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=0,
        )
        train_batches = _infinite_batches(train_loader)
        assert val_loader is not None
        assert test_loader is not None
        val_batches_factory = _loader_batches_factory(val_loader)
        test_batches_factory = _loader_batches_factory(test_loader)

    optimizer = _make_optimizer(
        model,
        lr=effective_lr,
        weight_decay=weight_decay,
        device=device,
        logger=logger,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    all_train_metric: list[float] = []
    all_val_metric = [0.0]
    best_val_metric = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_test_metric = 0.0
    no_val_improvement = 0
    all_time = []
    start = time.time()

    try:
        for step in range(num_steps):
            model.train()
            x, lengths, labels = next(train_batches)
            if x.device != device:
                x = x.to(device=device, non_blocking=True)
            if lengths.device != device:
                lengths = lengths.to(device=device, non_blocking=True)
            if labels.device != device:
                labels = labels.to(device=device, non_blocking=True)

            if check_numerics:
                bad_input = _first_nonfinite_index(x)
                if bad_input is not None:
                    raise FloatingPointError(
                        "Encountered non-finite input batch values "
                        f"at training step {step + 1}, tensor index {bad_input}."
                    )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(x, lengths)
                if check_numerics:
                    bad_logits = _first_nonfinite_index(logits)
                    if bad_logits is not None:
                        raise FloatingPointError(
                            "Encountered non-finite model logits "
                            f"at training step {step + 1}, tensor index {bad_logits}."
                        )

                loss = F.cross_entropy(logits, labels)
            if check_numerics and not bool(torch.isfinite(loss)):
                loss_value = float(loss.detach().item())
                raise FloatingPointError(
                    "Encountered non-finite loss "
                    f"at training step {step + 1}: loss={loss_value}."
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )
                else:
                    grad_norm = torch.zeros((), device=device, dtype=torch.float32)
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )
                else:
                    grad_norm = torch.zeros((), device=device, dtype=torch.float32)
            if (
                check_numerics
                and grad_clip_norm is not None
                and not bool(torch.isfinite(grad_norm))
            ):
                grad_norm_value = float(grad_norm.detach().item())
                raise FloatingPointError(
                    "Encountered non-finite gradient norm "
                    f"at training step {step + 1}: grad_norm={grad_norm_value}."
                )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            running_loss += loss.detach().to(dtype=running_loss.dtype)

            if (step + 1) % print_steps != 0:
                continue

            val_metric = _evaluate_accuracy(
                model,
                val_batches_factory(),
                device,
                mixed_precision=mixed_precision,
            )
            total_time = time.time() - start
            train_loss = float((running_loss / print_steps).item())
            logger.log(
                f"Step {step + 1}/{num_steps}: "
                f"train_loss={train_loss:.6f}, "
                f"validation_metric={val_metric:.6f}, "
                f"elapsed_sec={total_time:.2f}"
            )

            all_train_metric.append(train_loss)
            all_val_metric.append(val_metric)
            all_time.append(total_time)
            if progress_callback is not None:
                progress_callback(step + 1, num_steps)

            if (
                early_stopping_patience is not None
                and (step + 1) >= early_stop_after
                and best_val_metric != float("-inf")
            ):
                if val_metric <= best_val_metric:
                    no_val_improvement += 1
                    if no_val_improvement > early_stopping_patience:
                        logger.log(
                            "Early stopping triggered after "
                            f"{step + 1} steps due to no validation improvement."
                        )
                        break
                else:
                    no_val_improvement = 0

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                logger.log(f"New best validation metric: {best_val_metric:.6f}")

            running_loss.zero_()
            _save_metrics(
                output_dir,
                print_steps,
                all_train_metric,
                all_val_metric,
                all_time,
                best_test_metric,
            )
            start = time.time()

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        best_test_metric = _evaluate_accuracy(
            model,
            test_batches_factory(),
            device,
            mixed_precision=mixed_precision,
        )
        logger.log(f"Test metric: {best_test_metric:.6f}")
        _save_metrics(
            output_dir,
            print_steps,
            all_train_metric,
            all_val_metric,
            all_time,
            best_test_metric,
        )
        return model
    except Exception:
        logger.exception("Training failed with an unexpected error.")
        raise
    finally:
        _close_iterator(train_batches)
        del train_batches

        # Explicitly shutdown DataLoader workers to prevent file descriptor leaks.
        if (
            train_loader is not None
            and hasattr(train_loader, "_iterator")
            and train_loader._iterator is not None
        ):
            shutdown_workers = getattr(
                train_loader._iterator, "_shutdown_workers", None
            )
            if callable(shutdown_workers):
                shutdown_workers()

        del train_loader
        del val_loader
        del test_loader
        del optimizer
        del scaler
        if owns_logger:
            logger.close()


def create_dataset_model_and_train_torch(
    seed,
    data_dir,
    use_presplit,
    dataset_name,
    output_step,
    metric,
    include_time,
    T,
    model_name,
    stepsize,
    logsig_depth,
    linoss_discretization,
    model_args,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir="",
    id=None,
    dataloader_workers: int | None = None,
    overwrite_output_dir: bool = False,
    auto_confirm_output_dir: bool = False,
    allow_tf32: bool = False,
    mixed_precision: bool = False,
    check_numerics: bool = True,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    early_stopping_patience: int | None = 10,
    min_steps_before_early_stop: int | None = None,
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
):
    del output_step, stepsize, logsig_depth, linoss_discretization, id

    if model_name != "SLinOSS":
        raise ValueError(f"Unknown Torch training model: {model_name}")
    if metric != "accuracy":
        raise ValueError("SLinOSS Torch training currently supports accuracy only.")

    full_output_dir = build_slinoss_output_dir(
        seed=seed,
        dataset_name=dataset_name,
        output_parent_dir=output_parent_dir,
        T=T,
        include_time=include_time,
        num_steps=num_steps,
        lr=lr,
        model_args=model_args,
    )
    result = run_slinoss_training(
        seed=seed,
        data_dir=data_dir,
        use_presplit=use_presplit,
        dataset_name=dataset_name,
        include_time=include_time,
        T=T,
        model_args=model_args,
        num_steps=num_steps,
        print_steps=print_steps,
        lr=lr,
        lr_scheduler=lr_scheduler,
        batch_size=batch_size,
        output_dir=full_output_dir,
        dataloader_workers=dataloader_workers,
        overwrite_output_dir=overwrite_output_dir,
        auto_confirm_output_dir=auto_confirm_output_dir,
        allow_tf32=allow_tf32,
        mixed_precision=mixed_precision,
        check_numerics=check_numerics,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        early_stopping_patience=early_stopping_patience,
        min_steps_before_early_stop=min_steps_before_early_stop,
        verbose=verbose,
        progress_callback=progress_callback,
        prompt_if_output_dir_exists=True,
    )
    return result.model


def run_slinoss_training(
    *,
    seed: int,
    data_dir: str,
    use_presplit: bool,
    dataset_name: str,
    include_time: bool,
    T: float,
    model_args: dict,
    num_steps: int,
    print_steps: int,
    lr: float,
    lr_scheduler,
    batch_size: int,
    output_dir: str,
    dataloader_workers: int | None = None,
    overwrite_output_dir: bool = False,
    auto_confirm_output_dir: bool = False,
    allow_tf32: bool = False,
    mixed_precision: bool = False,
    check_numerics: bool = True,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    early_stopping_patience: int | None = 10,
    min_steps_before_early_stop: int | None = None,
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    prompt_if_output_dir_exists: bool = False,
) -> SLinOSSRunResult:
    from models.SLinOSS import ensure_slinoss_cuda_ready

    ensure_slinoss_cuda_ready()
    device = torch.device("cuda")

    _prepare_output_dir(
        output_dir,
        overwrite=overwrite_output_dir,
        auto_confirm=auto_confirm_output_dir,
        prompt_if_exists=prompt_if_output_dir_exists,
        verbose=verbose,
    )
    logger = _create_run_logger(output_dir, verbose=verbose)

    # Use native python/numpy/torch random to generate seeds instead of jax
    gen = torch.Generator().manual_seed(seed)
    datasetkey = int(torch.randint(0, 2**32, (1,), generator=gen).item())
    modelkey = int(torch.randint(0, 2**32, (1,), generator=gen).item())
    shufflekey = int(torch.randint(0, 2**32, (1,), generator=gen).item())

    try:
        logger.log(
            "Run configuration: "
            f"seed={seed}, dataset={dataset_name}, include_time={include_time}, "
            f"T={T}, num_steps={num_steps}, print_steps={print_steps}, "
            f"batch_size={batch_size}, allow_tf32={allow_tf32}, "
            f"mixed_precision={mixed_precision}, check_numerics={check_numerics}, "
            f"weight_decay={weight_decay}, grad_clip_norm={grad_clip_norm}, "
            f"early_stopping_patience={early_stopping_patience}, "
            f"min_steps_before_early_stop={min_steps_before_early_stop}, "
            f"model_args={model_args}"
        )
        logger.log(f"Creating dataset {dataset_name}")
        dataset = create_torch_dataset(
            data_dir,
            dataset_name,
            use_presplit,
            include_time,
            T,
            key=datasetkey,
        )
        logger.log(
            "Dataset created: "
            f"train={len(dataset.train)}, val={len(dataset.val)}, test={len(dataset.test)}"
        )

        logger.log("Creating model SLinOSS")
        _configure_cuda_training_runtime(allow_tf32=allow_tf32, logger=logger)
        _set_torch_seed(modelkey)
        model = create_torch_model(
            "SLinOSS",
            dataset.data_dim,
            dataset.label_dim,
            model_args=model_args,
        ).to(device)

        trained_model = train_torch_model(
            dataset,
            model,
            num_steps=num_steps,
            print_steps=print_steps,
            lr=lr,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            shuffle_seed=shufflekey,
            output_dir=output_dir,
            device=device,
            dataloader_workers=dataloader_workers,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            early_stopping_patience=early_stopping_patience,
            min_steps_before_early_stop=min_steps_before_early_stop,
            mixed_precision=mixed_precision,
            verbose=verbose,
            progress_callback=progress_callback,
            logger=logger,
            check_numerics=check_numerics,
        )
        summary = load_torch_training_summary(output_dir)
        logger.log(
            "Run finished: "
            f"completed_steps={summary.completed_steps}, "
            f"best_validation_metric={summary.best_validation_metric:.6f}, "
            f"test_metric={summary.test_metric:.6f}"
        )
        return SLinOSSRunResult(
            model=trained_model,
            output_dir=output_dir,
            summary=summary,
        )
    except Exception:
        logger.exception("Training setup or execution failed.")
        raise
    finally:
        logger.close()
