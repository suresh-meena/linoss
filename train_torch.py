"""Torch training entrypoint for raw sequence classification models."""

from __future__ import annotations

import os
import shutil
import time
import hashlib
import math
from typing import Callable

import jax.random as jr
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_dir.torch_datasets import TorchDataset, create_torch_dataset
from models.generate_torch_model import create_torch_model


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
    verbose: bool = True,
) -> None:
    if os.path.isdir(output_dir):
        if overwrite or auto_confirm:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            if verbose:
                print(f"Directory {output_dir} has been deleted and recreated.")
            return

        user_input = input(
            f"Warning: Output directory {output_dir} already exists. "
            "Do you want to delete it? (yes/no): "
        )
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            if verbose:
                print(f"Directory {output_dir} has been deleted and recreated.")
            return
        raise ValueError(f"Directory {output_dir} already exists. Exiting.")

    os.makedirs(output_dir)
    if verbose:
        print(f"Directory {output_dir} has been created.")


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
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        worker_count = int(
            os.environ.get("LINOSS_DATALOADER_WORKERS", min(4, max(1, cpu_count // 2)))
        )
    else:
        worker_count = int(num_workers)
    if worker_count <= 0:
        worker_count = 0

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        num_workers=worker_count,
        pin_memory=pin_memory,
    )
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        **loader_kwargs,
    )


def _infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _first_nonfinite_index(tensor: torch.Tensor) -> tuple[int, ...] | None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return None
    bad = (~finite_mask).nonzero(as_tuple=False)
    return tuple(int(i) for i in bad[0].tolist())


@torch.no_grad()
def _evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    use_amp = device.type == "cuda"
    for x, lengths, labels in loader:
        x = x.to(device=device, non_blocking=True)
        lengths = lengths.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x, lengths)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.shape[0])
    return float(correct / total)


def _save_metrics(
    output_dir: str,
    print_steps: int,
    all_train_metric: list[float],
    all_val_metric: list[float],
    all_time: list[float],
    test_metric: float,
) -> None:
    steps = np.arange(0, len(all_train_metric) * print_steps, print_steps)
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
    mixed_precision: bool = False,
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
):
    if batch_size > len(dataset.train):
        raise ValueError("Batch size larger than training dataset size.")

    pin_memory = device.type == "cuda"
    train_generator = torch.Generator()
    train_generator.manual_seed(shuffle_seed)
    use_amp = mixed_precision and device.type == "cuda"

    train_loader = _make_loader(
        dataset.train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        pin_memory=pin_memory,
        num_workers=None,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_scheduler(lr))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_batches = _infinite_batches(train_loader)

    running_loss = 0.0
    all_train_metric: list[float] = []
    all_val_metric = [0.0]
    best_val_metric = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_test_metric = 0.0
    no_val_improvement = 0
    all_time = []
    start = time.time()

    for step in range(num_steps):
        model.train()
        x, lengths, labels = next(train_batches)
        x = x.to(device=device, non_blocking=True)
        lengths = lengths.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        bad_input = _first_nonfinite_index(x)
        if bad_input is not None:
            raise FloatingPointError(
                "Encountered non-finite input batch values "
                f"at training step {step + 1}, tensor index {bad_input}."
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x, lengths)
            bad_logits = _first_nonfinite_index(logits)
            if bad_logits is not None:
                raise FloatingPointError(
                    "Encountered non-finite model logits "
                    f"at training step {step + 1}, tensor index {bad_logits}."
                )

            loss = F.cross_entropy(logits, labels)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                "Encountered non-finite loss "
                f"at training step {step + 1}: loss={loss_value}."
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_value = float(grad_norm)
        if not math.isfinite(grad_norm_value):
            raise FloatingPointError(
                "Encountered non-finite gradient norm "
                f"at training step {step + 1}: grad_norm={grad_norm_value}."
            )

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        running_loss += loss_value

        if (step + 1) % print_steps != 0:
            continue

        val_metric = _evaluate_accuracy(model, val_loader, device)
        total_time = time.time() - start
        if verbose:
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Validation metric: {val_metric}, Time: {total_time}"
            )

        all_train_metric.append(float("nan"))
        all_val_metric.append(val_metric)
        all_time.append(total_time)
        if progress_callback is not None:
            progress_callback(step + 1, num_steps)

        if step > 0:
            if val_metric <= best_val_metric:
                no_val_improvement += 1
                if no_val_improvement > 10:
                    break
            else:
                no_val_improvement = 0

        if val_metric >= best_val_metric:
            best_val_metric = val_metric
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        running_loss = 0.0
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
    best_test_metric = _evaluate_accuracy(model, test_loader, device)
    if verbose:
        print(f"Test metric: {best_test_metric}")
    _save_metrics(
        output_dir,
        print_steps,
        all_train_metric,
        all_val_metric,
        all_time,
        best_test_metric,
    )
    return model


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
    overwrite_output_dir: bool = False,
    auto_confirm_output_dir: bool = False,
    mixed_precision: bool = False,
    torch_compile: bool = False,
    torch_compile_mode: str = "reduce-overhead",
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
):
    del output_step, stepsize, logsig_depth, linoss_discretization, id

    if model_name != "SLinOSS":
        raise ValueError(f"Unknown Torch training model: {model_name}")
    if metric != "accuracy":
        raise ValueError("SLinOSS Torch training currently supports accuracy only.")

    from models.SLinOSS import ensure_slinoss_cuda_ready

    ensure_slinoss_cuda_ready()
    device = torch.device("cuda")

    output_parent_dir = os.path.join(output_parent_dir, "outputs", model_name, dataset_name)
    output_dir = _build_output_dir(
        seed=seed,
        T=T,
        include_time=include_time,
        num_steps=num_steps,
        lr=lr,
        model_name=model_name,
        stepsize=1,
        logsig_depth=1,
        model_args=model_args,
    )
    full_output_dir = os.path.join(output_parent_dir, output_dir)

    key = jr.PRNGKey(seed)
    datasetkey, modelkey, trainkey, _ = jr.split(key, 4)
    # Match the extra split performed inside the JAX dataloader path before shuffling batches.
    shufflekey, _ = jr.split(trainkey)

    if verbose:
        print(f"Creating dataset {dataset_name}")
    dataset = create_torch_dataset(
        data_dir,
        dataset_name,
        use_presplit,
        include_time,
        T,
        key=datasetkey,
    )

    if verbose:
        print(f"Creating model {model_name}")
    _set_torch_seed(_key_to_seed(modelkey))
    model = create_torch_model(
        model_name,
        dataset.data_dim,
        dataset.label_dim,
        model_args=model_args,
    ).to(device)

    if torch_compile:
        model = torch.compile(model, mode=torch_compile_mode, dynamic=True)

    _prepare_output_dir(
        full_output_dir,
        overwrite=overwrite_output_dir,
        auto_confirm=auto_confirm_output_dir,
        verbose=verbose,
    )
    return train_torch_model(
        dataset,
        model,
        num_steps=num_steps,
        print_steps=print_steps,
        lr=lr,
        lr_scheduler=lr_scheduler,
        batch_size=batch_size,
        shuffle_seed=_key_to_seed(shufflekey),
        output_dir=full_output_dir,
        device=device,
        mixed_precision=mixed_precision,
        verbose=verbose,
        progress_callback=progress_callback,
    )
