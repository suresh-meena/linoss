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
where data.pkl and labels.pkl are jnp.arrays with shape (n_samples, n_timesteps, n_features) 
and (n_samples, n_classes) respectively. If the dataset had original_idxs then those should
be saved as a list of jnp.arrays with shape [(n_train,), (n_val,), (n_test,)].

### The UEA Datasets

The UEA datasets are a collection of multivariate time series classification benchmarks. They can be downloaded by 
running `data_dir/download_uea.py` and preprocessed by running `data_dir/process_uea.py`.

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

`requirements.txt` now pins `slinoss v0.2.0`, which includes the upstream eval / `torch.no_grad()` fix from issue `#6`. That is the release this sweep stack is intended to run against.

### Hardware Tiers

The checked-in resource profile currently defines two execution tiers. On March
26, 2026, I reran the full `108` unique config rows on the remote 3090 box with
a conservative `5.5 GiB` per-process cap to emulate the actual RTX 3050 `6 GB`
fleet. That recalibration reproduced the same `18` ADA-only rows as the earlier
draft, so the split below is now confirmed for `6 GB` cards rather than carried
over from the old `8 GB` planning assumption.

| Resource tier | Intended hardware | Trials | Groups after filtering | Purpose |
| --- | --- | ---: | ---: | --- |
| `rtx3050-6gb` | RTX 3050 6 GB fleet | 1350 | 60 | Default tier for everything that probed cleanly under the LinOSS-faithful batch sizes |
| `ada6000` | RTX ADA 6000 | 270 | 40 | Deep-family rows that OOMed on the probe box under the LinOSS-faithful batch sizes |

The ADA-only rows are:

| Dataset | 3050-safe rows | ADA-only rows |
| --- | --- | --- |
| `EigenWorms` | `small` all layers, `medium` all layers, `large` `n_layers=2` | `large` `n_layers=4/6` |
| `EthanolConcentration` | `small` all layers, `medium` all layers, `large` `n_layers=2` | `large` `n_layers=4/6` |
| `Heartbeat` | all families, all layers | none |
| `MotorImagery` | `small` all layers, `medium` `n_layers=2/4` | `medium` `n_layers=6`, `large` `n_layers=2/4/6` |
| `SelfRegulationSCP1` | all families, all layers | none |
| `SelfRegulationSCP2` | `small` all layers, `medium` all layers, `large` `n_layers=2/4` | `large` `n_layers=6` |

Counting both `include_time=false/true`, that is `18` ADA-only config rows or `270` trials.

### Manual Multi-Machine Workflow

The intended operating model is deliberately simple: no master process, no distributed scheduler, no worker coordinator. Each machine claims one or more deterministic shards and runs them locally.

1. Prepare each machine:

```bash
git clone <repo-url>
cd linoss
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional bootstrap for Guix-managed machines:

```bash
guix shell -m manifest.scm
```

Each machine also needs the processed datasets under `data_dir/processed/UEA/...`.

2. Materialize the plan once:

```bash
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json --resource-tier rtx3050-6gb
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json --resource-tier ada6000
```

This should report:

- full plan: `1620` trials across `60` groups
- `rtx3050-6gb`: `1350` trials across `60` filtered groups
- `ada6000`: `270` trials across `40` filtered groups

3. Keep a tiny shared ledger outside the repo.

The minimal useful columns are:

- `machine`
- `gpu`
- `resource_tier`
- `shard`
- `started_at`
- `finished_at`
- `status`
- `notes`

4. Run the 3050 fleet and ADA machines as separate shard pools.

`--shard` is applied after dataset and resource-tier filtering, so the 3050 and ADA pools can be managed independently.

Example: first 3050 machine:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier rtx3050-6gb \
  --devices cuda:0 \
  --shard 1/16
```

Second 3050 machine:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier rtx3050-6gb \
  --devices cuda:0 \
  --shard 2/16
```

First ADA machine:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier ada6000 \
  --devices cuda:0 \
  --shard 1/2
```

Second ADA machine:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier ada6000 \
  --devices cuda:0 \
  --shard 2/2
```

If a machine has multiple GPUs, either pass them together:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier rtx3050-6gb \
  --devices cuda:0,cuda:1 \
  --shard 3/16
```

or run one process per GPU with different shards.

Useful selection variants:

```bash
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json --dataset MotorImagery
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json --dataset MotorImagery --resource-tier ada6000
python -m sweep run --config sweep/configs/slinoss_uea_grid.json --devices cuda:0 --dataset MotorImagery --resource-tier ada6000 --shard 1/2
```

5. Resume or retry safely.

- Completed trials are skipped automatically.
- Use `--retry-failed` to rerun only failed trials.
- Use `--force` only when you intentionally want to replace existing outputs.

6. Aggregate results after the shards finish.

Rsync each machine's `outputs/sweeps/slinoss-uea-grid/` tree back into one canonical copy, then reduce:

```bash
python -m sweep reduce --config sweep/configs/slinoss_uea_grid.json
```

The important artifacts are:

- `manifest.json`
- `plan.jsonl`
- `results/*.jsonl`
- `trials/**/result.json`
- `reports/family_summary.json`
- `reports/family_summary.csv`

### Probe-Based Budget And Placement Estimates

The timing probes were run with the new SLinOSS path in strict FP32, with
`torch.compile` disabled, mixed precision disabled, `5` warmup steps, and `20`
timed training steps per configuration. The timing numbers below use the raw
probe timings as conservative planning inputs. They are not direct RTX 3050
benchmarks, but they were measured on a remote 3090 box that, on a matched
calibration case, was slower than the local 3060 run for this code path.

The same March 26, 2026 rerun also reapplied a `5.5 GiB` cap to the full
`108`-row calibration table to mimic the real RTX 3050 `6 GB` cards. That pass
confirmed that the `6 GB` fleet uses the same placement boundary as the earlier
draft, so the trial counts and budget tables below are unchanged apart from the
updated tier name.

The ADA-only rows also OOMed on the 24 GB probe box under the exact LinOSS-faithful batch sizes. Their runtime budget is therefore an upper bound derived from smaller-batch safe probes for the same rows. In practice, the ADA 6000 should be strictly better than this bound.

Per-dataset split and budget:

| Dataset | 3050 trials | ADA trials | 3050 expected GPU-h | 3050 conservative GPU-h | ADA expected upper GPU-h | ADA conservative upper GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EigenWorms | 210 | 60 | 62.0 | 82.7 | 20.0 | 26.6 |
| EthanolConcentration | 210 | 60 | 45.9 | 61.2 | 15.4 | 20.6 |
| Heartbeat | 270 | 0 | 45.2 | 60.3 | 0.0 | 0.0 |
| MotorImagery | 150 | 120 | 36.1 | 48.2 | 43.6 | 58.2 |
| SelfRegulationSCP1 | 270 | 0 | 67.4 | 89.9 | 0.0 | 0.0 |
| SelfRegulationSCP2 | 240 | 30 | 55.4 | 73.8 | 11.8 | 15.7 |
| Total | 1350 | 270 | 312.1 | 416.1 | 90.8 | 121.1 |

Interpretation:

- `Expected GPU-h` assumes average stopping around `15k` steps.
- `Conservative GPU-h` assumes every trial runs the full `20k` steps.
- ADA values are upper bounds, not direct measurements on an ADA 6000.

Observed memory placement signal from the probe set:

| Dataset | Placement note |
| --- | --- |
| `EigenWorms` | `large` at `batch_size=4` needs ADA once `n_layers >= 4` |
| `EthanolConcentration` | `large` at `batch_size=32` needs ADA once `n_layers >= 4` |
| `Heartbeat` | entire LinOSS-faithful grid stays on the 3050 tier |
| `MotorImagery` | `medium` `n_layers=6` and all `large` rows need ADA |
| `SelfRegulationSCP1` | entire LinOSS-faithful grid stays on the 3050 tier |
| `SelfRegulationSCP2` | `large` `n_layers=6` needs ADA |

Concrete wall-clock examples if you run both tiers at the same time:

| Hardware allocation | 3050 tier wall-clock h expected / conservative | ADA tier wall-clock h expected upper / conservative upper | Whole sweep wall-clock h expected upper / conservative upper |
| --- | ---: | ---: | ---: |
| `4 x RTX 3050` + `2 x ADA 6000` | `78.0 / 104.0` | `45.4 / 60.6` | `78.0 / 104.0` |
| `8 x RTX 3050` + `1 x ADA 6000` | `39.0 / 52.0` | `90.8 / 121.1` | `90.8 / 121.1` |
| `8 x RTX 3050` + `2 x ADA 6000` | `39.0 / 52.0` | `45.4 / 60.6` | `45.4 / 60.6` |
| `16 x RTX 3050` + `2 x ADA 6000` | `19.5 / 26.0` | `45.4 / 60.6` | `45.4 / 60.6` |

For practical planning, the cleanest baseline is:

- budget `312.1` GPU-hours for the 3050 tier
- budget at most `90.8` additional GPU-hours for the ADA tier
- keep `537.2` GPU-hours in mind as the strict combined worst-case upper bound if everything runs to `20k` steps and the ADA-only rows behave no better than the conservative proxy

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
