# AGENTS

## Working Rules

- Use conventional commits.
- Keep commits narrow and reviewable.
- Prefer minimal, typed, testable changes over broad rewrites.
- Do not mix remote-infra churn with unrelated modeling or training changes.

## Remote Control

- This repo ships remote experiment helpers under `scripts/`.
- Machine definitions live in a root `.env` file that must remain untracked.
- Prefer `AUTH=key` with `SSH_KEY` set. Keep `PASSWORD` populated as a fallback
  while public-key access is being rolled out or repaired.
- Important local state is kept under the ignored `.remote-state/` directory.
- Groups and metadata belong in `.env`, not in agent memory. Use:
  - `KD_REMOTE_GROUP_<GROUP>=...`
  - `KD_REMOTE_<MACHINE>_GPU_CLASS=...`
  - `KD_REMOTE_<MACHINE>_GPU_VRAM_GIB=...`
  - `KD_REMOTE_<MACHINE>_GPU_COUNT=...`
- Primary commands:
  - `./scripts/remote-list`
  - `./scripts/remote-print-config --machine <name>`
  - `./scripts/remote-smoke --group <group>`
  - `./scripts/remote-gpu-status --group <group> --require-idle --min-free-vram-gib <gib>`
  - `./scripts/remote-setup --group <group>`
  - `./scripts/remote-sweep-run --machine <name> --gpu <idx> ...`
  - `./scripts/remote-collect --group <group> --sweep <name>`
  - `./scripts/remote-sweep-status --config <config>`
- The scripts use a repo-local `.remote-known-hosts` file for non-interactive
  access. Manual aliases like `ssh ampere` or `ssh volta` are managed through
  the user's `~/.ssh/config`.
- Standard remote workflow:
  1. run `./scripts/remote-smoke --group <group>`
  2. bootstrap with `./scripts/remote-setup --group <group>`
  3. find polite targets with `./scripts/remote-gpu-status --group <group> --require-idle --min-free-vram-gib <gib>`
  4. launch detached jobs with `./scripts/remote-sweep-run`
  5. collect with `./scripts/remote-collect`
  6. inspect completion with `./scripts/remote-sweep-status`
- Stay polite:
  1. do not take a GPU with an active compute process
  2. do not take a GPU whose free VRAM is below the expected tier requirement
  3. do not ignore active leases unless explicitly justified
- For SLinOSS sweeps, never run one process with multiple visible GPUs and
  never pass `--devices cuda:0,cuda:1`. Use one visible physical GPU per
  process and `--devices cuda:0` inside that process.
- Keep this tooling lean. The goal is reliable remote experiment control, not a
  new orchestration layer.
