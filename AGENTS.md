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
- Primary commands:
  - `./scripts/remote-list`
  - `./scripts/remote-print-config --machine <name>`
  - `./scripts/remote-shell --machine <name>`
  - `./scripts/remote-rsync --machine <name>`
  - `./scripts/remote-smoke --machine <name>`
- The scripts use a repo-local `.remote-known-hosts` file for non-interactive
  access. Manual aliases like `ssh ampere` or `ssh volta` are managed through
  the user's `~/.ssh/config`.
- Standard remote workflow:
  1. run `./scripts/remote-smoke --machine <name>`
  2. sync the repo with `./scripts/remote-rsync --machine <name>`
  3. launch the experiment through `./scripts/remote-shell --machine <name> -- ...`
- Keep this tooling lean. The goal is reliable remote experiment control, not a
  new orchestration layer.
