# Oscillatory State-Space Models (ICLR2025 Oral)

This  repository contains the official implementation for the paper [Oscillatory State-Space Models](https://openreview.net/pdf?id=GRMfXcAAFh) by [T. Konstantin Rusch](https://konstantinrusch.com/) and [Daniela Rus](https://www.csail.mit.edu/person/daniela-rus).

### ![NEW](https://img.shields.io/badge/NEW-red) For a complete JAX-based SSM library including LinOSS and other state-of-the-art SSM architectures, please check out our [Discretax library](https://github.com/camail-official/discretax).

--------------------
We propose Linear Oscillatory State-Space models (LinOSS) for efficiently learning on long sequences. Inspired by cortical dynamics of biological neural networks, we base our proposed LinOSS model on a system of forced harmonic oscillators. A stable discretization, integrated over time using fast associative parallel scans, yields the proposed state-space model. 

![linoss_animation](https://github.com/user-attachments/assets/9d034ddf-3fa8-48e8-9818-8c3217015135)

## Requirements

This repository is implemented in python 3.10 and uses Jax as their machine learning framework. This is an extension of [https://github.com/Benjamin-Walker/log-neural-cdes](https://github.com/Benjamin-Walker/log-neural-cdes).

### Environment

The code for preprocessing the datasets, training LinOSS, S5, LRU, NCDE, NRDE, and Log-NCDE uses the following packages:
- `jax` and `jaxlib` for automatic differentiation.
- `equinox` for constructing neural networks.
- `optax` for neural network optimisers.
- `diffrax` for differential equation solvers.
- `signax` for calculating the signature.
- `sktime` for handling time series data in ARFF format.
- `tqdm` for progress bars.
- `matplotlib` for plotting.
- `pre-commit` for code formatting.

```
conda create -n LinOSS python=3.10
conda activate LinOSS
conda install pre-commit=3.7.1 sktime=0.30.1 tqdm=4.66.4 matplotlib=3.8.4 -c conda-forge
# Substitue for correct Jax pip install: https://jax.readthedocs.io/en/latest/installation.html
pip install -U "jax[cuda12]" "jaxlib[cuda12]" equinox==0.11.4 optax==0.2.2 diffrax==0.5.1 signax==0.1.1
```

If running `data_dir/process_uea.py` throws this error: No module named 'packaging'
Then run: `pip install packaging`

After installing the requirements, run `pre-commit install` to install the pre-commit hooks.

---

## Data

The folder `data_dir` contains the scripts for downloading data, preprocessing the data, and creating dataloaders and 
datasets. Raw data should be downloaded into the `data_dir/raw` folder. Processed data should be saved into the `data_dir/processed`
folder in the following format: 
```
processed/{collection}/{dataset_name}/data.pkl, 
processed/{collection}/{dataset_name}/labels.pkl,
processed/{collection}/{dataset_name}/original_idxs.pkl (if the dataset has original data splits)
```
where `data.pkl` and `labels.pkl` are NumPy arrays with shape `(n_samples, n_timesteps, n_features)`
and `(n_samples,)` respectively. If the dataset had `original_idxs` then those should
be saved as NumPy integer arrays inside a tuple or list.

### The UEA Datasets

The UEA datasets are a collection of multivariate time series classification benchmarks. They can be downloaded by 
running `data_dir/download_uea.py` and preprocessed by running `data_dir/process_uea.py`.

The authoritative processed UEA format is now NumPy-backed pickle files. If you have an older processed tree that
was written with JAX arrays, rewrite it in place with:

```bash
python data_dir/process_uea.py --data-dir data_dir --rewrite-existing
```

If you want to regenerate the processed UEA tree from raw ARFF files instead, use:

```bash
python data_dir/process_uea.py --data-dir data_dir --overwrite
```

### The PPG-DaLiA Dataset

The PPG-DaLiA dataset is a multivariate time series regression dataset,
where the aim is to predict a person’s heart rate using data
collected from a wrist-worn device. The dataset can be downloaded from the 
<a href="https://archive.ics.uci.edu/dataset/495/ppg+dalia">UCI Machine Learning Repository</a>. The data should be 
unzipped and saved in the `data_dir/raw` folder in the following format `PPG_FieldStudy/S{i}/S{i}.pkl`. The data can be
preprocessed by running the `process_ppg.py` script.

---

## Experiments

The code for training and evaluating the models is contained in `train.py`. Experiments can be run using the `run_experiment.py` script. 
This script requires you to specify the names of the models you want to train, 
the names of the datasets you want to train on, and a directory which contains configuration files. By default,
it will run the LinOSS experiments. The configuration files should be organised as `config_dir/{model_name}/{dataset_name}.json` and contain the
following fields:
- `seeds`: A list of seeds to use for training.
- `data_dir`: The directory containing the data.
- `output_parent_dir`: The directory to save the output.
- `lr_scheduler`: A function which takes the learning rate and returns the new learning rate.
- `num_steps`: The number of steps to train for.
- `print_steps`: The number of steps between printing the loss.
- `batch_size`: The batch size.
- `metric`: The metric to use for evaluation.
- `classification`: Whether the task is a classification task.
- `linoss_discretization`: ONLY for LinoSS -- which discretization to use. Choices are ['IM','IMEX']
- `lr`: The initial learning rate.
- `time`: Whether to include time as a channel.
- Any further specific model parameters. 

See `experiment_configs/repeats` for examples.

## Sweeps

Large SLinOSS hyperparameter searches now live in `sweep/` and follow a deterministic `plan -> run -> reduce` flow. The production UEA sweep is checked in at `sweep/configs/slinoss_uea_grid.json`.

That config is now fully LinOSS-faithful on batch size:

- `EigenWorms`: `batch_size=4`
- all other UEA datasets: `batch_size=32`

The rest of the grid is:

- 6 datasets
- 3 model families:
  - `small`: `d_model=16`, `d_head=16`, `d_state=16`
  - `medium`: `d_model=64`, `d_head=32`, `d_state=16`
  - `large`: `d_model=128`, `d_head=64`, `d_state=64`
- 2 `include_time` settings: `false`, `true`
- 3 layer counts: `2`, `4`, `6`
- 3 learning rates: `1e-3`, `1e-4`, `1e-5`
- 5 seeds: `2345`, `3456`, `4567`, `5678`, `6789`
- total work: `1620` trials in `60` deterministic dataset-cache groups
- one `(dataset, include_time, seed)` group contains `27` trials before any tier filtering

The hardware split is not baked into the hyperparameter grid itself. It is expressed separately in `sweep/configs/slinoss_uea_grid.resources.json`, and the CLI can filter by that resource tier with `--resource-tier`.

`requirements.txt` now pins `slinoss v0.3.0`. The published Linux x86_64 CUDA wheels cover CPython `3.11`, `3.12`, and `3.13`, and that is the release this sweep stack is intended to run against.

### Hardware Tiers

The checked-in resource profile currently defines two execution tiers. On March
27, 2026, after rewriting the processed UEA pickles to NumPy and ensuring the
Torch path no longer unpickles JAX arrays, I reran the full `108` unique config
rows on the ADA box with the same conservative `5.5 GiB` per-process cap used
to approximate the RTX 3050 `6 GB` fleet. That recalibration tightened the
boundary from `18` to `37` ADA-only config rows. The earlier split was
optimistic.

| Resource tier | Intended hardware | Trials | Groups after filtering | Purpose |
| --- | --- | ---: | ---: | --- |
| `rtx3050-6gb` | RTX 3050 6 GB fleet | 1065 | 60 | Rows that stayed within the `5.5 GiB` safety cap under the NumPy-backed recalibration |
| `ada6000` | RTX ADA 6000 | 555 | 50 | Rows that exceeded the `5.5 GiB` safety cap after JAX was removed from the Torch data path |

The ADA-only rows are:

| Dataset | 3050-safe rows | ADA-only rows |
| --- | --- | --- |
| `EigenWorms` | `small` all layers, `medium` all layers | `large` all layers |
| `EthanolConcentration` | `small` all layers, `medium` all layers, `large` with `include_time=false`, `n_layers=2` | `large` with `include_time=false`, `n_layers=4/6`, and all `large` rows with `include_time=true` |
| `Heartbeat` | all families, all layers | none |
| `MotorImagery` | `small` all layers, `medium` `n_layers=2` | `medium` `n_layers=4/6`, `large` `n_layers=2/4/6` |
| `SelfRegulationSCP1` | `small` all layers, `medium` all layers | `large` all layers |
| `SelfRegulationSCP2` | `small` all layers, `medium` `n_layers=2` | `medium` `n_layers=4/6`, `large` all layers |

Counting both `include_time=false/true`, that is `37` ADA-only config rows or `555` trials.

This happened because the tiering rule is keyed off peak reserved CUDA memory
under a `5.5 GiB` safety cap. Removing JAX-backed pickles from the Torch data
path reduced one source of GPU pressure, but it also let the CUDA side reserve
larger pools or workspaces on some rows. The old split understated the true
memory footprint of the new sweep path.

### Remote Fleet Workflow

The intended operating model is still deliberately simple: no scheduler, no
worker coordinator, no master process. One control checkout manages many
machines through `scripts/remotectl.py`, and every remote worker still runs a
deterministic `python -m sweep run ...` shard locally.

The control checkout keeps its centralized state in the ignored
`.remote-state/` directory:

- `.remote-state/fleet-state.json`
  last-known smoke, setup, GPU, collect, and launch data per machine
- `.remote-state/sweeps/<name>/ledger.jsonl`
  append-only launch and collect events
- `.remote-state/sweeps/<name>/machines/<machine>/`
  per-machine collected result snapshots
- `.remote-state/sweeps/<name>/canonical/`
  merged result tree used to answer exact completion questions

That centralized cache is how an agent can later answer “what is complete,
what is still pending, and which trials are done?” without inventing a
scheduler or reverse-engineering random marker files.

1. Populate the root `.env` with machine groups and GPU metadata.

The remote helpers read:

- `KD_REMOTE_GROUP_3050=opb1,...,opb18`
- `KD_REMOTE_GROUP_ADA=ada1,ada2`
- `KD_REMOTE_<MACHINE>_GPU_CLASS=...`
- `KD_REMOTE_<MACHINE>_GPU_VRAM_GIB=...`
- `KD_REMOTE_<MACHINE>_GPU_COUNT=...`

Useful sanity checks:

```bash
./scripts/remote-list --groups
./scripts/remote-list --group 3050 --verbose
./scripts/remote-print-config --group 3050
```

2. Bootstrap each group from the control checkout.

`remote-smoke` now fails hard if the configured `WORKDIR` is missing. `remote-setup`
creates that workdir, syncs the repo, and builds the remote `.venv`.

```bash
./scripts/remote-smoke --group 3050
./scripts/remote-setup --group 3050 --check-path data_dir/processed/UEA

./scripts/remote-smoke --group ada
./scripts/remote-setup --group ada --check-path data_dir/processed/UEA
```

3. Materialize the plan once and keep the shard pools separate by tier.

```bash
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json --resource-tier rtx3050-6gb
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json --resource-tier ada6000
```

This should report:

- full plan: `1620` trials across `60` groups
- `rtx3050-6gb`: `1065` trials across `60` filtered groups
- `ada6000`: `555` trials across `50` filtered groups

`--shard` is applied after resource-tier filtering, so the 3050 and ADA pools
can be managed independently.

4. Choose targets politely before launching.

The policy is:

- do not take a GPU with an active compute process
- do not take a GPU whose free VRAM is below the intended tier requirement
- do not ignore active lease files unless you are deliberately preempting stale state

Examples:

```bash
./scripts/remote-gpu-status --group 3050 --require-idle --min-free-vram-gib 4.5
./scripts/remote-gpu-status --group ada --require-idle --min-free-vram-gib 16
```

5. Launch detached remote workers through `remote-sweep-run`.

This wrapper enforces the CuTe-safe single-visible-GPU pattern automatically:
it always runs one remote process with `CUDA_VISIBLE_DEVICES=<physical_gpu>` and
`--devices cuda:0` inside the process.

3050 example:

```bash
./scripts/remote-sweep-run \
  --machine opb7 \
  --gpu 0 \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier rtx3050-6gb \
  --shard 7/18 \
  --require-idle \
  --min-free-vram-gib 4.5 \
  --note "3050 shard 7/18"
```

ADA example:

```bash
./scripts/remote-sweep-run \
  --machine ada \
  --gpu 0 \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier ada6000 \
  --shard 1/2 \
  --require-idle \
  --min-free-vram-gib 16 \
  --note "ada shard 1/2"
```

If you want to inspect the exact SSH and remote launch command first:

```bash
./scripts/remote-sweep-run \
  --machine opb7 \
  --gpu 0 \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier rtx3050-6gb \
  --shard 7/18 \
  --dry-run
```

On multi-GPU machines, do not pass non-zero logical CUDA devices like
`--devices cuda:0,cuda:1` to one sweep process. The current CuTe / CUTLASS DSL
stack can still crash on logical devices such as `cuda:1` when multiple GPUs
are visible. The safe pattern is one process per physical GPU, one visible GPU
per process, and `--devices cuda:0` inside that process. `remote-sweep-run`
exists specifically to keep that rule automatic.

6. Collect results back into the centralized cache.

```bash
./scripts/remote-collect --group 3050 --sweep slinoss-uea-grid
./scripts/remote-collect --group ada --sweep slinoss-uea-grid
```

`remote-collect` stores each machine snapshot under
`.remote-state/sweeps/slinoss-uea-grid/machines/<machine>/` and also merges it
into `.remote-state/sweeps/slinoss-uea-grid/canonical/`.

7. Ask for exact completion status from the merged cache.

```bash
./scripts/remote-sweep-status --config sweep/configs/slinoss_uea_grid.json
./scripts/remote-sweep-status --config sweep/configs/slinoss_uea_grid.json --list pending
./scripts/remote-sweep-status --config sweep/configs/slinoss_uea_grid.json --list failed
```

This compares the deterministic plan against the canonical collected
`result.json` files, so it answers both “how much is done?” and “which specific
trials are still pending?”.

8. Resume or retry safely.

- Completed trials are skipped automatically.
- Use `--retry-failed` to rerun only failed trials.
- Use `--force` only when you intentionally want to replace existing outputs.

9. Reduce once the collected canonical tree is complete enough.

The remote workers already emit:

- `manifest.json`
- `plan.jsonl`
- `results/*.jsonl`
- `trials/**/result.json`
- `reports/family_summary.json`
- `reports/family_summary.csv`

The centralized cache mirrors those files under
`.remote-state/sweeps/<name>/canonical/`. The existing `python -m sweep reduce`
command still reduces from `outputs/sweeps/<name>/`, so first mirror the
canonical cache into that local output root:

```bash
rsync -a .remote-state/sweeps/slinoss-uea-grid/canonical/ outputs/sweeps/slinoss-uea-grid/
python -m sweep reduce --config sweep/configs/slinoss_uea_grid.json
```

### Probe-Based Placement Estimates

The March 27, 2026 placement rerun is now the authoritative boundary for the
checked-in resource profile. It used the new NumPy-backed processed UEA data,
strict FP32, `torch.compile` disabled, mixed precision disabled, `5` warmup
steps, and `20` timed training steps per configuration. Every one of the `108`
calibration rows completed cleanly, and every one loaded `ndarray` pickles on
the Torch path.

The earlier wall-clock budget table is intentionally not carried forward here.
Those numbers were tied to the old `18`-row ADA split, and the new placement
rerun was executed on the ADA box rather than the slower 3050 fleet. Leaving
the old table in place would be more misleading than helpful.

Use this per-dataset split for placement planning:

| Dataset | 3050 trials | ADA trials | Placement note |
| --- | ---: | ---: | --- |
| `EigenWorms` | 180 | 90 | all `large` rows move to ADA |
| `EthanolConcentration` | 195 | 75 | all `large` rows with `include_time=true`, plus `large` `include_time=false` at `n_layers=4/6`, move to ADA |
| `Heartbeat` | 270 | 0 | entire grid stays on 3050 |
| `MotorImagery` | 120 | 150 | `medium` `n_layers=4/6` and all `large` rows move to ADA |
| `SelfRegulationSCP1` | 180 | 90 | all `large` rows move to ADA |
| `SelfRegulationSCP2` | 120 | 150 | `medium` `n_layers=4/6` and all `large` rows move to ADA |
| Total | 1065 | 555 | `37` ADA-only row families out of `108` |

The practical implication is:

- `19` row families moved from the old 3050 tier to ADA after the NumPy-backed recalibration.
- `12` of those `19` rows went above `6 GiB` peak reserved memory outright, so this is not just a cap-margin artifact.
- If you need fresh wall-clock budgets, rerun the timing pass on the actual 3050 fleet or on a deliberately conservative proxy before depending on the old GPU-hour estimates.

---

## Reproducing the Results

The configuration files for all the experiments with fixed hyperparameters can be found in the `experiment_configs` folder and
`run_experiment.py` is currently configured to run the repeat experiments on the UEA datasets.
The `outputs` folder contains a zip file of the output files from the UEA, and PPG experiments. 

---

# Citation
If you found our work useful in your research, please cite our paper at:
```bibtex
@inproceedings{rusch2025linoss,
  title={Oscillatory State-Space Models},
  author={Rusch, T Konstantin and Rus, Daniela},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
(Also consider starring the project on GitHub.)
