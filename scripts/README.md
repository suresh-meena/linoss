# Remote Scripts

This directory contains the local control-plane wrappers for running SLinOSS
sweeps across physically separate machines without introducing a scheduler.
The intended model is still manual and explicit: one control checkout, many
remote machines, deterministic shards, polite leases, and a centralized local
result cache.

## Files

- `manifest.scm`
  Optional Guix manifest for the command-line tools the remote helpers use
- `guix-run`
  Optional wrapper to run a command inside the Guix shell described by
  `manifest.scm`
- `remote-list`
  List configured machines, optionally filtered to one group
- `remote-print-config`
  Print resolved machine metadata without secrets
- `remote-shell`
  Open an interactive shell or run one command on a selected machine
- `remote-for-each`
  Run the same non-interactive command across a machine group
- `remote-rsync`
  Sync the repo to or from one machine or a whole group
- `remote-smoke`
  Check connectivity, workdir, Python, and `nvidia-smi`
- `remote-gpu-status`
  Inspect GPU occupancy, free VRAM, and polite lease state
- `remote-lease`
  Acquire, inspect, or release a remote GPU lease file
- `remote-setup`
  Create the remote workdir, sync the repo, and build the remote `.venv`
- `remote-sweep-run`
  Launch one detached sweep worker with the CuTe-safe single-visible-GPU pattern
- `remote-collect`
  Pull one sweep's result artifacts into the local centralized cache
- `remote-sweep-status`
  Compare the deterministic sweep plan against the centralized collected results

## Local Configuration

Copy `.env.example` to `.env` and fill in the machines you want to manage.

The scripts read:

- `KD_REMOTE_MACHINES`
- `KD_REMOTE_DEFAULT_MACHINE`
- optional `KD_REMOTE_GROUP_<GROUP>`
- optional `KD_REMOTE_STATE_DIR`
- optional `KD_REMOTE_LEASE_ROOT`
- optional `KD_REMOTE_TOOLCHAIN=auto|system|guix`
- per-machine fields such as:
  `KD_REMOTE_<MACHINE>_HOST`
  `KD_REMOTE_<MACHINE>_USER`
  `KD_REMOTE_<MACHINE>_PORT`
  `KD_REMOTE_<MACHINE>_WORKDIR`
  `KD_REMOTE_<MACHINE>_AUTH`
  `KD_REMOTE_<MACHINE>_SSH_KEY`
  `KD_REMOTE_<MACHINE>_PASSWORD`
  `KD_REMOTE_<MACHINE>_GPU_CLASS`
  `KD_REMOTE_<MACHINE>_GPU_VRAM_GIB`
  `KD_REMOTE_<MACHINE>_GPU_COUNT`
  optional `KD_REMOTE_<MACHINE>_LEASE_ROOT`

Machine and group names are normalized for the env key by upper-casing and
replacing `-` or `.` with `_`.

`WORKDIR` is optional for generic shell access, but strongly recommended. It is
required for `remote-smoke`, `remote-setup`, `remote-collect`, and
`remote-sweep-run`.

## Local Tooling Requirements

Guix is optional. The remote helpers now support two local toolchain modes:

- `system`
  Use `ssh`, `rsync`, and, for password-auth machines, `sshpass` directly from
  `PATH`
- `guix`
  Run those tools through `./scripts/guix-run`

By default the scripts use `KD_REMOTE_TOOLCHAIN=auto`, which prefers the direct
system tools when they are available and falls back to Guix otherwise.

So a user without Guix can still use the remote helpers as long as they have:

- `ssh`
- `rsync`
- `sshpass` when any configured machine uses `AUTH=password`

If you want to force one mode explicitly:

```bash
export KD_REMOTE_TOOLCHAIN=system
export KD_REMOTE_TOOLCHAIN=guix
```

## Centralized Local State

By default the control checkout keeps its local machine cache under
`.remote-state/`, which is ignored by git.

Important files:

- `.remote-state/fleet-state.json`
  last-known smoke, GPU, setup, collect, and launch information per machine
- `.remote-state/sweeps/<name>/ledger.jsonl`
  append-only launch and collect events
- `.remote-state/sweeps/<name>/machines/<machine>/`
  per-machine collected result snapshots
- `.remote-state/sweeps/<name>/canonical/`
  merged result tree used by `remote-sweep-status`

This is the mechanism that lets an agent answer questions like “how much of the
sweep is complete?” without inventing a scheduler.

## Operating Pattern

The polite workflow is:

1. verify the target machines
2. inspect GPU availability and respect active leases
3. claim a GPU only when it is idle and has enough free VRAM for the intended tier
4. launch one detached sweep worker per physical GPU, always with
   `CUDA_VISIBLE_DEVICES=<gpu>` and `--devices cuda:0`
5. periodically collect results into `.remote-state`
6. ask `remote-sweep-status` for exact completion state

The CuTe / CUTLASS DSL stack is still not safe for non-zero logical devices in
one process with multiple GPUs visible, so `remote-sweep-run` deliberately
enforces the single-visible-GPU model.

## Examples

List machines and groups:

```bash
./scripts/remote-list
./scripts/remote-list --groups
./scripts/remote-list --group 3050 --verbose
```

Inspect machine metadata:

```bash
./scripts/remote-print-config --machine ampere
./scripts/remote-print-config --group 3050
```

Check a group and bootstrap it:

```bash
./scripts/remote-smoke --group 3050
./scripts/remote-setup --group 3050 --check-path data_dir/processed/UEA
```

Find polite targets for a 3050-tier job:

```bash
./scripts/remote-gpu-status --group 3050 --require-idle --min-free-vram-gib 4.5
```

Run the same command across all currently-eligible 3050 machines:

```bash
./scripts/remote-for-each \
  --group 3050 \
  --require-idle \
  --min-free-vram-gib 4.5 \
  -- \
  hostname
```

Acquire or inspect a lease:

```bash
./scripts/remote-lease show --machine opb7 --gpu 0
./scripts/remote-lease acquire --machine opb7 --gpu 0 --ttl-hours 12 --note "slinoss shard 7/18"
./scripts/remote-lease release --machine opb7 --gpu 0
```

Launch one detached sweep shard:

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

Collect and summarize progress:

```bash
./scripts/remote-collect --group 3050 --sweep slinoss-uea-grid
./scripts/remote-collect --group ada --sweep slinoss-uea-grid
./scripts/remote-sweep-status --config sweep/configs/slinoss_uea_grid.json
./scripts/remote-sweep-status --config sweep/configs/slinoss_uea_grid.json --list pending
```

Inspect resolved commands without executing them:

```bash
./scripts/remote-shell --machine ampere --dry-run -- hostname
./scripts/remote-rsync --group 3050 --dry-run
./scripts/remote-sweep-run --machine opb7 --gpu 0 --config sweep/configs/slinoss_uea_grid.json --dry-run
```
