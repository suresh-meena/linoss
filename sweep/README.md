# SLinOSS Sweep

This package replaces the old ad hoc sweep code with a deterministic plan -> execute -> reduce flow.

Core ideas:

- `plan`: expand an immutable grid into concrete `TrialSpec`s and dataset-cache `TrialGroup`s
- `run`: map selected groups onto one persistent worker process per GPU
- `reduce`: aggregate per-trial results into family-level summaries across seeds

The grouping key is `(dataset config, dataset seed)`, so trials that share the same randomized split naturally reuse a cached dataset inside each worker.

## Layout

- `config.py`: parse the sweep JSON into typed definitions
- `planner.py`: deterministic grid expansion, stable IDs, sharding, and grouping
- `executor.py`: persistent CUDA workers with dataset caching
- `results.py`: manifest writing, per-trial result records, and family-level reduction
- `cli.py`: `python -m sweep`

## Config Shape

```json
{
  "name": "slinoss-uea-grid",
  "output_root": "outputs/sweeps/slinoss-uea-grid",
  "resource_profile": "slinoss_uea_grid.resources.json",
  "defaults": {
    "dataset": {
      "data_dir": "data_dir",
      "use_presplit": false,
      "include_time": true,
      "T": 1.0
    },
    "training": {
      "num_steps": 20000,
      "print_steps": 1000,
      "lr": 0.001,
      "lr_scheduler": "identity",
      "batch_size": 2,
      "check_numerics": true
    },
    "model": {
      "d_model": 128,
      "n_layers": 2,
      "d_state": 64
    }
  },
  "grid": {
    "training": {
      "lr": [0.001, 0.0003]
    },
    "model": {
      "d_state": [64, 128],
      "n_layers": [2, 4]
    }
  },
  "datasets": [
    {
      "name": "EigenWorms",
      "seeds": [2345, 3456, 4567]
    }
  ]
}
```

`grid` axes can be defined at the top level and overridden per dataset under `datasets[].grid`.

If you want to keep one hyperparameter grid but route specific rows to different hardware classes, add a `resource_profile`. The profile can set a `default_tier` and then override selected rows with rule-based matches.

## Commands

Preview the full plan:

```bash
python -m sweep plan --config sweep/configs/slinoss_uea_grid.example.json
```

Preview only the trials assigned to one hardware tier:

```bash
python -m sweep plan \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier rtx3050-6gb
```

On multi-GPU machines, do not pass logical devices like
`--devices cuda:0,cuda:1` to one process right now. The current CuTe /
CUTLASS DSL stack is reliable when each process sees exactly one GPU and uses
`--devices cuda:0`.

Run one quarter of the work on one physical GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.example.json \
  --devices cuda:0 \
  --shard 1/4
```

Run the second quarter on another physical GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.example.json \
  --devices cuda:0 \
  --shard 2/4
```

Run only the ADA-tagged rows:

```bash
python -m sweep run \
  --config sweep/configs/slinoss_uea_grid.json \
  --resource-tier ada6000 \
  --devices cuda:0 \
  --shard 1/2
```

Reduce finished trials into a family leaderboard:

```bash
python -m sweep reduce --config sweep/configs/slinoss_uea_grid.example.json
```

## Output Structure

For a sweep rooted at `outputs/sweeps/slinoss-uea-grid`, the package writes:

- `manifest.json`: config snapshot and plan metadata
- `plan.jsonl`: one immutable `TrialSpec` per line
- `results/<runner>.jsonl`: append-only runner log
- `trials/<dataset>/family-<hash>/seed-<seed>/trial.json`
- `trials/<dataset>/family-<hash>/seed-<seed>/result.json`
- `trials/<dataset>/family-<hash>/seed-<seed>/training.log`
- `reports/family_summary.json`
- `reports/family_summary.csv`

The primary control plane is the deterministic plan plus explicit per-trial result records. There is no hidden scheduler state machine or inference from random marker files.
