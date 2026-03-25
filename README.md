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

Large SLinOSS hyperparameter searches now live in `sweep/` and follow a deterministic `plan -> run -> reduce` flow. The production UEA grid is checked in at `sweep/configs/slinoss_uea_grid.json`.

That config expands to:

- 6 datasets
- 3 model families:
  - `small`: `d_model=16`, `d_head=16`, `d_state=16`
  - `medium`: `d_model=64`, `d_head=32`, `d_state=16`
  - `large`: `d_model=128`, `d_head=64`, `d_state=64`
- 2 `include_time` settings: `false`, `true`
- 3 layer counts: `2`, `4`, `6`
- 3 learning rates: `1e-3`, `1e-4`, `1e-5`
- 5 seeds: `2345`, `3456`, `4567`, `5678`, `6789`
- total work: `1620` trials in `60` dataset-cache groups
- one `(dataset, include_time, seed)` group contains `27` trials

Before running a long production sweep, update `requirements.txt` to the first `slinoss` release that includes the upstream eval / `torch.no_grad()` fix from issue `#6`. The sweep code is ready now; the currently pinned `v0.1.1` wheel is not the release I would use for a multi-day production run.

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

Optional bootstrap for machines that use Guix:

```bash
guix shell -m manifest.scm
```

Each machine also needs the processed datasets under `data_dir/processed/UEA/...`.

2. Materialize the plan once:

```bash
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json
```

This should report `1620` trials and `60` groups.

3. Claim shards manually.

Keep a very small shared ledger outside the repo with columns like:

- `machine`
- `gpu`
- `shard`
- `started_at`
- `finished_at`
- `status`
- `notes`

For one-GPU RTX 3060 machines, one shard per machine is the clean default:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --devices cuda:0 \
  --shard 1/8
```

On the next machine:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --devices cuda:0 \
  --shard 2/8
```

And so on.

If a machine has multiple GPUs, either pass them together:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --devices cuda:0,cuda:1 \
  --shard 3/8
```

or run one process per GPU with different shards. Because sharding happens over deterministic groups, machines do not need to know about each other.

Useful selection variants:

```bash
python -m sweep plan --config sweep/configs/slinoss_uea_grid.json --dataset MotorImagery
python -m sweep run --config sweep/configs/slinoss_uea_grid.json --devices cuda:0 --dataset MotorImagery --shard 1/2
```

4. Resume or retry safely.

- Completed trials are skipped automatically.
- Use `--retry-failed` to rerun only failed trials.
- Use `--force` only when you intentionally want to replace existing outputs.

5. Aggregate results after the shards finish.

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

### Probe-Based Budget And VRAM

The tables below were measured with the new SLinOSS path in strict FP32, with `torch.compile` disabled, mixed precision disabled, `5` warmup steps, and `20` timed training steps per configuration. The initial batch search was then corrected with targeted reruns for the three deep-family cases that still needed smaller batch sizes.

One matched calibration case (`Heartbeat`, `small`, `batch_size=2`, `include_time=true`, `n_layers=2`) ran at `26.46 sec / 1000 steps` on the remote RTX 3090 probe box and `8.83 sec / 1000 steps` on the local RTX 3060. In other words, the remote box is currently slower than the local 3060 for this exact code path. The timing tables below therefore use the raw probe timings as conservative planning numbers for 3060-class machines, rather than scaling them down aggressively.

Safe batch sizes and measured peak reserved VRAM for the full `20`-step probe:

| Dataset | Shape | Small batch / peak GiB | Medium batch / peak GiB | Large batch / peak GiB |
| --- | --- | ---: | ---: | ---: |
| EigenWorms | `236 x 17984 x 6` | 8 / 3.27 | 2 / 2.50 | 1 / 4.53 |
| EthanolConcentration | `524 x 1751 x 3` | 32 / 1.32 | 32 / 3.81 | 8 / 2.80 |
| Heartbeat | `409 x 405 x 61` | 32 / 1.08 | 32 / 1.68 | 32 / 3.39 |
| MotorImagery | `378 x 3000 x 64` | 32 / 2.45 | 16 / 3.58 | 8 / 5.29 |
| SelfRegulationSCP1 | `561 x 896 x 6` | 32 / 0.65 | 32 / 1.98 | 16 / 2.66 |
| SelfRegulationSCP2 | `380 x 1152 x 7` | 32 / 1.34 | 32 / 2.99 | 16 / 4.09 |

Estimated full-budget runtime per trial at `20k` steps, in minutes. Each cell is `include_time=false / include_time=true`.

| Dataset | Family | Batch | Peak GiB | 20k min/trial n=2 (off/on) | 20k min/trial n=4 (off/on) | 20k min/trial n=6 (off/on) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EigenWorms | small | 8 | 4.44 | 11.2 / 10.7 | 21.4 / 20.9 | 31.3 / 31.1 |
| EigenWorms | medium | 2 | 2.50 | 9.0 / 9.2 | 16.0 / 15.9 | 23.5 / 23.4 |
| EigenWorms | large | 1 | 4.53 | 11.0 / 11.0 | 21.3 / 21.5 | 31.7 / 31.7 |
| EthanolConcentration | small | 32 | 2.22 | 4.4 / 4.4 | 8.3 / 8.3 | 12.2 / 12.3 |
| EthanolConcentration | medium | 32 | 4.56 | 11.0 / 10.8 | 21.1 / 21.5 | 31.3 / 31.5 |
| EthanolConcentration | large | 8 | 2.80 | 10.7 / 10.2 | 17.0 / 16.9 | 24.1 / 24.4 |
| Heartbeat | small | 32 | 1.41 | 9.1 / 9.0 | 13.3 / 13.6 | 14.0 / 13.5 |
| Heartbeat | medium | 32 | 1.88 | 8.2 / 8.9 | 13.0 / 13.4 | 13.7 / 14.3 |
| Heartbeat | large | 32 | 3.39 | 10.4 / 10.2 | 15.8 / 16.1 | 22.3 / 22.6 |
| MotorImagery | small | 32 | 3.21 | 6.8 / 8.8 | 13.3 / 14.2 | 20.5 / 19.7 |
| MotorImagery | medium | 16 | 4.21 | 9.5 / 10.2 | 18.5 / 18.9 | 27.4 / 27.7 |
| MotorImagery | large | 8 | 5.29 | 15.3 / 15.1 | 29.8 / 29.9 | 43.8 / 43.7 |
| SelfRegulationSCP1 | small | 32 | 1.56 | 8.5 / 8.5 | 13.5 / 13.4 | 14.4 / 12.7 |
| SelfRegulationSCP1 | medium | 32 | 2.85 | 9.4 / 9.4 | 14.0 / 14.2 | 17.1 / 17.2 |
| SelfRegulationSCP1 | large | 16 | 2.66 | 8.2 / 9.1 | 17.2 / 16.8 | 23.9 / 24.1 |
| SelfRegulationSCP2 | small | 32 | 1.76 | 8.6 / 8.7 | 13.3 / 13.1 | 13.3 / 13.6 |
| SelfRegulationSCP2 | medium | 32 | 3.30 | 10.0 / 10.1 | 15.0 / 15.0 | 21.2 / 21.0 |
| SelfRegulationSCP2 | large | 16 | 4.09 | 10.8 / 10.7 | 21.0 / 21.6 | 31.2 / 31.7 |

Sweep budget by dataset. The three runtime columns mean:

- `Expected GPU-h (15k avg)`: reasonable planning number if average early stopping lands around `15k` steps
- `Conservative GPU-h (20k)`: every trial runs the full `20k` steps
- `Early-stop floor GPU-h (12k)`: optimistic lower bound if every trial stops at the earliest plausible `12k`-step point

| Dataset | Trials | Groups | Expected GPU-h (15k avg) | Conservative GPU-h (20k) | Early-stop floor GPU-h (12k) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EigenWorms | 270 | 10 | 66.0 | 88.0 | 52.8 |
| EthanolConcentration | 270 | 10 | 52.5 | 70.0 | 42.0 |
| Heartbeat | 270 | 10 | 45.2 | 60.3 | 36.2 |
| MotorImagery | 270 | 10 | 70.0 | 93.3 | 56.0 |
| SelfRegulationSCP1 | 270 | 10 | 47.2 | 62.9 | 37.7 |
| SelfRegulationSCP2 | 270 | 10 | 54.4 | 72.5 | 43.5 |
| Total | 1620 | 60 | 335.3 | 447.0 | 268.2 |

Expected wall-clock if you spread the sweep across multiple RTX 3060 GPUs:

| 3060 GPUs in parallel | Expected wall-clock h (15k avg) | Conservative wall-clock h (20k) | Early-stop floor h (12k) |
| ---: | ---: | ---: | ---: |
| 1 | 335.3 | 447.0 | 268.2 |
| 2 | 167.6 | 223.5 | 134.1 |
| 4 | 83.8 | 111.8 | 67.1 |
| 8 | 41.9 | 55.9 | 33.5 |

For practical planning, I would budget around `335 GPU-hours` for the full sweep, and keep `447 GPU-hours` in mind as the strict worst-case cap if the entire grid runs to `20k` steps.

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
