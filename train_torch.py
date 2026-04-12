"""Torch training entrypoint for raw sequence classification models."""

from __future__ import annotations

import os
import shutil
import time
import hashlib
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, TextIO

# Keep JAX/XLA off the GPU in the Torch SLinOSS path. The processed UEA pickles
# contain JAX arrays, so unpickling them can otherwise trigger JAX GPU runtime
# initialization and large non-PyTorch allocations in sweep workers.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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


@dataclass(frozen=True)
class SLinOSSRunSeeds:
    dataset_seed: int
    model_seed: int
    shuffle_seed: int


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
        indices = torch.randperm(
            size,
            generator=generator,
            device=tensors[0].device,
        )
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


def _iter_tensors(value: Any, *, path: str):
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensors(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, path=f"{path}[{index}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, path=f"{path}[{index}]")


def _first_nonfinite_index(tensor: torch.Tensor) -> tuple[int, ...] | None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return None
    bad = (~finite_mask).nonzero(as_tuple=False)
    return tuple(int(i) for i in bad[0].tolist())


def _summarize_nonfinite_tensor(
    tensor: torch.Tensor,
    *,
    path: str,
    bad_index: tuple[int, ...],
) -> str:
    detached = tensor.detach()
    if bad_index:
        bad_value = detached[bad_index].item()
    else:
        bad_value = detached.item()

    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    if detached.is_floating_point():
        nan_count = int(torch.isnan(detached).sum().item())
        posinf_count = int(torch.isposinf(detached).sum().item())
        neginf_count = int(torch.isneginf(detached).sum().item())

    return (
        f"{path}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"device={detached.device}, first_bad_index={bad_index}, "
        f"first_bad_value={bad_value!r}, nan_count={nan_count}, "
        f"posinf_count={posinf_count}, neginf_count={neginf_count}"
    )


def _describe_first_nonfinite_in_structure(value: Any, *, label: str) -> str | None:
    for path, tensor in _iter_tensors(value, path=label):
        bad_index = _first_nonfinite_index(tensor)
        if bad_index is not None:
            return _summarize_nonfinite_tensor(
                tensor,
                path=path,
                bad_index=bad_index,
            )
    return None


def _format_module_label(name: str, module: torch.nn.Module | None) -> str:
    module_name = name or "<root>"
    label = module_name
    if module is not None:
        label = f"{module_name} ({module.__class__.__name__})"
    location = _module_location_hint(module_name)
    if location is None:
        return label
    return f"{label} [{location}]"


def _module_location_hint(module_name: str) -> str | None:
    parts = module_name.split(".")
    if len(parts) < 2 or parts[0] != "blocks" or not parts[1].isdigit():
        return None

    layer_index = int(parts[1])
    component = "block"
    if len(parts) >= 3:
        component = parts[2]
    return f"layer={layer_index}, component={component}"


def _describe_nonfinite_named_tensors(
    named_tensors,
    *,
    kind: str,
    model: torch.nn.Module,
    max_items: int = 3,
) -> str | None:
    module_lookup = dict(model.named_modules())
    issues: list[str] = []
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        bad_index = _first_nonfinite_index(tensor)
        if bad_index is None:
            continue
        owner_name, _, _ = name.rpartition(".")
        owner_label = _format_module_label(owner_name, module_lookup.get(owner_name))
        issues.append(
            f"{kind} {name} in {owner_label}; "
            f"{_summarize_nonfinite_tensor(tensor, path='tensor', bad_index=bad_index)}"
        )
        if len(issues) >= max_items:
            break
    if not issues:
        return None
    return "; ".join(issues)


def _describe_nonfinite_parameters(
    model: torch.nn.Module,
    *,
    max_items: int = 3,
) -> str | None:
    return _describe_nonfinite_named_tensors(
        ((name, parameter) for name, parameter in model.named_parameters()),
        kind="parameter",
        model=model,
        max_items=max_items,
    )


def _describe_nonfinite_gradients(
    model: torch.nn.Module,
    *,
    max_items: int = 3,
) -> str | None:
    return _describe_nonfinite_named_tensors(
        (
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ),
        kind="gradient",
        model=model,
        max_items=max_items,
    )


class _ForwardNonfiniteTracker:
    def __init__(self, model: torch.nn.Module) -> None:
        self._first_summary: str | None = None
        self._last_entered_label: str | None = None
        self._module_stack: list[str] = []
        self._handles: list[Any] = []
        self._module_lookup = dict(model.named_modules())
        for name, module in model.named_modules():
            if name == "" or isinstance(module, torch.nn.ModuleList):
                continue
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self._handles.append(self._register_forward_hook(module, name))

    def _register_forward_hook(self, module: torch.nn.Module, name: str):
        hook = self._make_hook(name)
        try:
            return module.register_forward_hook(hook, always_call=True)
        except TypeError:
            # Older torch versions do not support always_call.
            return module.register_forward_hook(hook)

    def _make_pre_hook(self, name: str):
        def hook(module, inputs) -> None:
            del inputs
            self._module_stack.append(name)
            self._last_entered_label = _format_module_label(name, module)

        return hook

    def _remove_from_stack(self, name: str) -> None:
        if not self._module_stack:
            return
        if self._module_stack[-1] == name:
            self._module_stack.pop()
            return
        for index in range(len(self._module_stack) - 1, -1, -1):
            if self._module_stack[index] == name:
                del self._module_stack[index]
                return

    def _make_hook(self, name: str):
        def hook(module, inputs, output) -> None:
            del inputs
            try:
                if self._first_summary is not None:
                    return
                summary = _describe_first_nonfinite_in_structure(output, label="output")
                if summary is None:
                    return
                self._first_summary = (
                    "First module with non-finite forward output: "
                    f"{_format_module_label(name, module)}; {summary}"
                )
            finally:
                self._remove_from_stack(name)

        return hook

    def reset(self) -> None:
        self._first_summary = None
        self._last_entered_label = None
        self._module_stack.clear()

    def summary(self) -> str | None:
        return self._first_summary

    def last_entered_summary(self) -> str | None:
        if self._module_stack:
            name = self._module_stack[-1]
            module = self._module_lookup.get(name)
            return (
                "Last entered forward module before failure: "
                f"{_format_module_label(name, module)}"
            )
        if self._last_entered_label is None:
            return None
        return (
            "Last entered forward module before failure: "
            f"{self._last_entered_label}"
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _collect_forward_numerics_details(
    tracker: _ForwardNonfiniteTracker | None,
) -> list[str]:
    if tracker is None:
        return []

    details: list[str] = []
    last_entered = tracker.last_entered_summary()
    if last_entered is not None:
        details.append(last_entered)
    first_nonfinite = tracker.summary()
    if first_nonfinite is not None:
        details.append(first_nonfinite)
    return details


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
    for x, labels in batches:
        if x.device != device:
            x = x.to(device=device, non_blocking=True)
        if labels.device != device:
            labels = labels.to(device=device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            logits = model(x)
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
    has_complex_params = any(
        param.requires_grad and torch.is_complex(param)
        for param in model.parameters()
    )
    if device.type == "cuda":
        if has_complex_params:
            logger.log(
                "Detected complex-valued parameters; using standard Adam "
                "(fused Adam requires floating-point parameters only)."
            )
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
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


def derive_slinoss_run_seeds(seed: int) -> SLinOSSRunSeeds:
    gen = torch.Generator().manual_seed(seed)
    return SLinOSSRunSeeds(
        dataset_seed=int(torch.randint(0, 2**32, (1,), generator=gen).item()),
        model_seed=int(torch.randint(0, 2**32, (1,), generator=gen).item()),
        shuffle_seed=int(torch.randint(0, 2**32, (1,), generator=gen).item()),
    )


def create_slinoss_dataset(
    *,
    data_dir: str,
    use_presplit: bool,
    dataset_name: str,
    include_time: bool,
    T: float,
    dataset_seed: int,
    logger: _RunLogger | None = None,
) -> TorchDataset:
    if logger is not None:
        logger.log(f"Creating dataset {dataset_name}")
    dataset = create_torch_dataset(
        data_dir,
        dataset_name,
        use_presplit,
        include_time,
        T,
        key=dataset_seed,
    )
    if logger is not None:
        logger.log(
            "Dataset created: "
            f"train={len(dataset.train)}, val={len(dataset.val)}, test={len(dataset.test)}"
        )
    return dataset


def create_slinoss_model(
    *,
    dataset: TorchDataset,
    model_args: Mapping[str, Any],
    model_seed: int,
    device: torch.device,
    classification: bool = True,
    output_step: int = 1,
    allow_tf32: bool = False,
    logger: _RunLogger | None = None,
):
    if logger is not None:
        logger.log("Creating model SLinOSS")
    _configure_cuda_training_runtime(allow_tf32=allow_tf32, logger=logger)
    _set_torch_seed(model_seed)
    return create_torch_model(
        "SLinOSS",
        dataset.data_dim,
        dataset.label_dim,
        model_args=dict(model_args),
        classification=classification,
        output_step=output_step,
    ).to(device)


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
    train_generator = (
        torch.Generator(device=device.type)
        if dataset_is_on_gpu and device.type == "cuda"
        else torch.Generator()
    )
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
    scaler = torch.GradScaler("cuda", enabled=use_amp)

    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    all_train_metric: list[float] = []
    all_val_metric = [0.0]
    best_val_metric = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_test_metric = 0.0
    no_val_improvement = 0
    all_time = []
    start = time.time()
    forward_nonfinite_tracker = (
        _ForwardNonfiniteTracker(model) if check_numerics else None
    )

    try:
        for step in range(num_steps):
            model.train()
            x, labels = next(train_batches)
            if x.device != device:
                x = x.to(device=device, non_blocking=True)
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
            if forward_nonfinite_tracker is not None:
                forward_nonfinite_tracker.reset()
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                try:
                    logits = model(x)
                except Exception as exc:
                    if not check_numerics:
                        raise
                    details = _collect_forward_numerics_details(forward_nonfinite_tracker)
                    detail_suffix = ""
                    if details:
                        detail_suffix = " " + " ".join(details)
                    raise FloatingPointError(
                        "Model forward pass failed "
                        f"at training step {step + 1}.{detail_suffix}"
                    ) from exc
                if check_numerics:
                    bad_logits = _first_nonfinite_index(logits)
                    if bad_logits is not None:
                        details = _collect_forward_numerics_details(
                            forward_nonfinite_tracker
                        )
                        parameter_summary = _describe_nonfinite_parameters(model)
                        if parameter_summary is not None:
                            details.append(
                                "Non-finite parameter values already present: "
                                f"{parameter_summary}"
                            )
                        detail_suffix = ""
                        if details:
                            detail_suffix = " " + " ".join(details)
                        raise FloatingPointError(
                            "Encountered non-finite model logits "
                            f"at training step {step + 1}, tensor index {bad_logits}."
                            f"{detail_suffix}"
                        )

                loss = F.cross_entropy(logits, labels)
            if check_numerics and not bool(torch.isfinite(loss)):
                loss_value = float(loss.detach().item())
                details = _collect_forward_numerics_details(forward_nonfinite_tracker)
                parameter_summary = _describe_nonfinite_parameters(model)
                detail_suffix = ""
                if parameter_summary is not None:
                    details.append(
                        "Non-finite parameter values already present: "
                        f"{parameter_summary}"
                    )
                if details:
                    detail_suffix = " " + " ".join(details)
                raise FloatingPointError(
                    "Encountered non-finite loss "
                    f"at training step {step + 1}: loss={loss_value}."
                    f"{detail_suffix}"
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            gradient_summary = None
            if check_numerics:
                gradient_summary = _describe_nonfinite_gradients(model)
                if gradient_summary is not None:
                    details = _collect_forward_numerics_details(forward_nonfinite_tracker)
                    details.append(gradient_summary)
                    raise FloatingPointError(
                        "Encountered non-finite gradients "
                        f"at training step {step + 1}: {' '.join(details)}."
                    )
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
                details = _collect_forward_numerics_details(forward_nonfinite_tracker)
                if gradient_summary is not None:
                    details.append(gradient_summary)
                detail_suffix = ""
                if details:
                    detail_suffix = " " + " ".join(details)
                raise FloatingPointError(
                    "Encountered non-finite gradient norm "
                    f"at training step {step + 1}: grad_norm={grad_norm_value}."
                    f"{detail_suffix}"
                )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if check_numerics:
                parameter_summary = _describe_nonfinite_parameters(model)
                if parameter_summary is not None:
                    details = _collect_forward_numerics_details(forward_nonfinite_tracker)
                    details.append(parameter_summary)
                    raise FloatingPointError(
                        "Encountered non-finite parameter values immediately after "
                        f"optimizer step at training step {step + 1}: "
                        f"{' '.join(details)}."
                    )
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
        if forward_nonfinite_tracker is not None:
            forward_nonfinite_tracker.close()
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
    device: str | torch.device = "cuda",
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
):
    del stepsize, logsig_depth, linoss_discretization, id

    if model_name != "SLinOSS":
        raise ValueError(f"Unknown Torch training model: {model_name}")
    classification = metric == "accuracy"
    if not classification:
        raise ValueError(
            "SLinOSS Torch training currently supports accuracy only. "
            "The model now supports non-classification outputs, but the Torch "
            "trainer/data pipeline here has not been extended for regression yet."
        )

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
        classification=classification,
        output_step=output_step,
        device=device,
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
    classification: bool = True,
    output_step: int = 1,
    device: str | torch.device = "cuda",
    dataset: TorchDataset | None = None,
    run_seeds: SLinOSSRunSeeds | None = None,
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    prompt_if_output_dir_exists: bool = False,
) -> SLinOSSRunResult:
    from models.SLinOSS import ensure_slinoss_cuda_ready

    ensure_slinoss_cuda_ready()
    device = torch.device(device)
    run_seeds = derive_slinoss_run_seeds(seed) if run_seeds is None else run_seeds

    _prepare_output_dir(
        output_dir,
        overwrite=overwrite_output_dir,
        auto_confirm=auto_confirm_output_dir,
        prompt_if_exists=prompt_if_output_dir_exists,
        verbose=verbose,
    )
    logger = _create_run_logger(output_dir, verbose=verbose)

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
            f"classification={classification}, output_step={output_step}, "
            f"device={device}, "
            f"model_args={model_args}"
        )
        if dataset is None:
            dataset = create_slinoss_dataset(
                data_dir=data_dir,
                use_presplit=use_presplit,
                dataset_name=dataset_name,
                include_time=include_time,
                T=T,
                dataset_seed=run_seeds.dataset_seed,
                logger=logger,
            )
        else:
            logger.log(
                "Using caller-provided dataset: "
                f"train={len(dataset.train)}, val={len(dataset.val)}, test={len(dataset.test)}"
            )

        model = create_slinoss_model(
            dataset=dataset,
            model_args=model_args,
            model_seed=run_seeds.model_seed,
            device=device,
            classification=classification,
            output_step=output_step,
            allow_tf32=allow_tf32,
            logger=logger,
        )

        trained_model = train_torch_model(
            dataset,
            model,
            num_steps=num_steps,
            print_steps=print_steps,
            lr=lr,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            shuffle_seed=run_seeds.shuffle_seed,
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
