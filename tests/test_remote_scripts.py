from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "guix-run",
    ROOT / "scripts" / "remote-list",
    ROOT / "scripts" / "remote-print-config",
    ROOT / "scripts" / "remote-shell",
    ROOT / "scripts" / "remote-for-each",
    ROOT / "scripts" / "remote-rsync",
    ROOT / "scripts" / "remote-smoke",
    ROOT / "scripts" / "remote-gpu-status",
    ROOT / "scripts" / "remote-lease",
    ROOT / "scripts" / "remote-setup",
    ROOT / "scripts" / "remote-collect",
    ROOT / "scripts" / "remote-sweep-run",
    ROOT / "scripts" / "remote-sweep-status",
]


def _write_remote_env(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        textwrap.dedent(
            """
            KD_REMOTE_MACHINES=dgx1,scratch-box
            KD_REMOTE_DEFAULT_MACHINE=dgx1
            KD_REMOTE_GROUP_3050=scratch-box
            KD_REMOTE_GROUP_LAB=dgx1,scratch-box
            KD_REMOTE_STATE_DIR=/tmp/test-state

            KD_REMOTE_DGX1_HOST=dgx.example.edu
            KD_REMOTE_DGX1_USER=alice
            KD_REMOTE_DGX1_PORT=2222
            KD_REMOTE_DGX1_WORKDIR=/srv/kdrifting
            KD_REMOTE_DGX1_AUTH=key
            KD_REMOTE_DGX1_SSH_KEY=/tmp/test-key
            KD_REMOTE_DGX1_PASSWORD=fallback-secret
            KD_REMOTE_DGX1_GPU_CLASS=rtx3090
            KD_REMOTE_DGX1_GPU_VRAM_GIB=24
            KD_REMOTE_DGX1_GPU_COUNT=2

            KD_REMOTE_SCRATCH_BOX_HOST=10.0.0.5
            KD_REMOTE_SCRATCH_BOX_USER=bob
            KD_REMOTE_SCRATCH_BOX_WORKDIR=/scratch/kdrifting
            KD_REMOTE_SCRATCH_BOX_AUTH=password
            KD_REMOTE_SCRATCH_BOX_PASSWORD=super-secret
            KD_REMOTE_SCRATCH_BOX_GPU_CLASS=rtx3050-6gb
            KD_REMOTE_SCRATCH_BOX_GPU_VRAM_GIB=6
            KD_REMOTE_SCRATCH_BOX_GPU_COUNT=1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return env_file


def _run_script(
    script_name: str,
    *args: str,
    env_file: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["KD_REMOTE_ENV_FILE"] = str(env_file)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(ROOT / "scripts" / script_name), *args],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
    )


def _write_fake_tools(tmp_path: Path, *names: str) -> Path:
    tool_dir = tmp_path / "fake-tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = tool_dir / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    return tool_dir


def _write_sweep_fixture(tmp_path: Path, state_dir: Path) -> Path:
    config_path = tmp_path / "toy_sweep.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "toy-sweep",
                "output_root": "outputs/sweeps/toy-sweep",
                "defaults": {
                    "dataset": {"data_dir": "data_dir"},
                    "training": {"batch_size": 4},
                    "model": {"d_model": 16, "n_layers": 2},
                },
                "datasets": [{"name": "ToyDataset", "seeds": [1234, 2345]}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical = state_dir / "sweeps" / "toy-sweep" / "canonical"
    (canonical / "trials" / "ToyDataset" / "family-aaa" / "seed-1234").mkdir(
        parents=True, exist_ok=True
    )
    (canonical / "trials" / "ToyDataset" / "family-bbb" / "seed-2345").mkdir(
        parents=True, exist_ok=True
    )
    (canonical / "plan.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trial_id": "family-aaa-seed-1234",
                        "seed": 1234,
                        "dataset": {"name": "ToyDataset"},
                        "output_dir": "outputs/sweeps/toy-sweep/trials/ToyDataset/family-aaa/seed-1234",
                        "resource_tier": "rtx3050-6gb",
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "trial_id": "family-bbb-seed-2345",
                        "seed": 2345,
                        "dataset": {"name": "ToyDataset"},
                        "output_dir": "outputs/sweeps/toy-sweep/trials/ToyDataset/family-bbb/seed-2345",
                        "resource_tier": "ada6000",
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (canonical / "manifest.json").write_text(
        json.dumps(
            {"output_root": "outputs/sweeps/toy-sweep"}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    (
        canonical / "trials" / "ToyDataset" / "family-aaa" / "seed-1234" / "result.json"
    ).write_text(
        json.dumps({"status": "success"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def test_remote_scripts_have_valid_shell_syntax() -> None:
    for script in SCRIPTS:
        subprocess.run(["sh", "-n", str(script)], check=True, cwd=ROOT)


def test_remote_list_reads_groups_and_machine_filters(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script("remote-list", env_file=env_file)
    assert result.stdout.strip().splitlines() == ["dgx1", "scratch-box"]

    groups = _run_script("remote-list", "--groups", env_file=env_file)
    assert groups.stdout.strip().splitlines() == ["3050", "lab"]

    filtered = _run_script("remote-list", "--group", "3050", env_file=env_file)
    assert filtered.stdout.strip().splitlines() == ["scratch-box"]


def test_remote_print_config_resolves_group_and_metadata(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-print-config", "--machine", "scratch-box", env_file=env_file
    )
    assert "machine=scratch-box" in result.stdout
    assert "host=10.0.0.5" in result.stdout
    assert "user=bob" in result.stdout
    assert "workdir=/scratch/kdrifting" in result.stdout
    assert "auth=password" in result.stdout
    assert "has_password=true" in result.stdout
    assert "groups=3050,lab" in result.stdout
    assert "gpu_class=rtx3050-6gb" in result.stdout
    assert "gpu_vram_gib=6" in result.stdout
    assert "super-secret" not in result.stdout


def test_remote_shell_dry_run_uses_resolved_machine_settings(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-shell",
        "--machine",
        "dgx1",
        "--dry-run",
        "--",
        "python",
        "-V",
        env_file=env_file,
    )
    assert "SSHPASS=<redacted>" in result.stdout
    assert "sshpass" in result.stdout
    assert "ssh" in result.stdout
    assert "/tmp/test-key" in result.stdout
    assert "alice@dgx.example.edu" in result.stdout
    assert "PreferredAuthentications=publickey,password" in result.stdout
    assert "cd /srv/kdrifting" in result.stdout
    assert "python -V" in result.stdout
    assert "fallback-secret" not in result.stdout


def test_remote_for_each_dry_run_filters_group(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-for-each",
        "--group",
        "3050",
        "--dry-run",
        "--",
        "hostname",
        env_file=env_file,
    )
    assert "10.0.0.5" in result.stdout
    assert "hostname" in result.stdout
    assert "dgx.example.edu" not in result.stdout


def test_remote_rsync_dry_run_redacts_password_auth(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-rsync",
        "--machine",
        "scratch-box",
        "--dry-run",
        env_file=env_file,
    )
    assert "SSHPASS=<redacted>" in result.stdout
    assert "sshpass" in result.stdout
    assert "rsync" in result.stdout
    assert "bob@10.0.0.5:/scratch/kdrifting/" in result.stdout
    assert "super-secret" not in result.stdout


def test_remote_shell_dry_run_supports_system_toolchain_without_guix(
    tmp_path: Path,
) -> None:
    env_file = _write_remote_env(tmp_path)
    fake_tools = _write_fake_tools(tmp_path, "ssh", "sshpass")
    result = _run_script(
        "remote-shell",
        "--machine",
        "scratch-box",
        "--dry-run",
        "--",
        "hostname",
        env_file=env_file,
        extra_env={
            "KD_REMOTE_TOOLCHAIN": "system",
            "PATH": f"{fake_tools}{os.pathsep}{os.environ['PATH']}",
        },
    )
    assert "sshpass" in result.stdout
    assert "ssh " in result.stdout
    assert "scripts/guix-run" not in result.stdout


def test_remote_rsync_dry_run_supports_system_toolchain_without_guix(
    tmp_path: Path,
) -> None:
    env_file = _write_remote_env(tmp_path)
    fake_tools = _write_fake_tools(tmp_path, "ssh", "sshpass", "rsync")
    result = _run_script(
        "remote-rsync",
        "--machine",
        "scratch-box",
        "--dry-run",
        env_file=env_file,
        extra_env={
            "KD_REMOTE_TOOLCHAIN": "system",
            "PATH": f"{fake_tools}{os.pathsep}{os.environ['PATH']}",
        },
    )
    assert "rsync" in result.stdout
    assert "sshpass" in result.stdout
    assert "scripts/guix-run" not in result.stdout


def test_remote_smoke_dry_run_uses_machine_workdir_and_strict_cd(
    tmp_path: Path,
) -> None:
    env_file = _write_remote_env(tmp_path)
    result = _run_script(
        "remote-smoke", "--machine", "dgx1", "--dry-run", env_file=env_file
    )
    assert "alice@dgx.example.edu" in result.stdout
    assert "set -eu" in result.stdout
    assert "cd /srv/kdrifting" in result.stdout
    assert "nvidia-smi" in result.stdout


def test_remote_setup_and_collect_dry_run_include_expected_commands(
    tmp_path: Path,
) -> None:
    env_file = _write_remote_env(tmp_path)
    setup = _run_script(
        "remote-setup",
        "--machine",
        "scratch-box",
        "--check-path",
        "data_dir/processed/UEA",
        "--dry-run",
        env_file=env_file,
    )
    assert "mkdir -p /scratch/kdrifting" in setup.stdout
    assert ".venv/bin/pip install -r requirements.txt" in setup.stdout
    assert "test -e data_dir/processed/UEA" in setup.stdout

    collect = _run_script(
        "remote-collect",
        "--machine",
        "scratch-box",
        "--sweep",
        "slinoss-uea-grid",
        "--dry-run",
        env_file=env_file,
    )
    assert "results/***" in collect.stdout
    assert "plan.jsonl" in collect.stdout
    assert "result.json" in collect.stdout
    assert (
        "bob@10.0.0.5:/scratch/kdrifting/outputs/sweeps/slinoss-uea-grid/"
        in collect.stdout
    )


def test_remote_sweep_run_dry_run_enforces_single_visible_gpu_pattern(
    tmp_path: Path,
) -> None:
    env_file = _write_remote_env(tmp_path)
    config_path = tmp_path / "toy_sweep.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "toy-sweep",
                "datasets": [{"name": "ToyDataset", "seeds": [1234]}],
                "defaults": {
                    "dataset": {"data_dir": "data_dir"},
                    "training": {"batch_size": 4},
                    "model": {"d_model": 16, "n_layers": 2},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run_script(
        "remote-sweep-run",
        "--machine",
        "scratch-box",
        "--gpu",
        "0",
        "--config",
        str(config_path),
        "--resource-tier",
        "rtx3050-6gb",
        "--shard",
        "1/18",
        "--dry-run",
        env_file=env_file,
    )
    assert "CUDA_VISIBLE_DEVICES" in result.stdout
    assert "--devices" in result.stdout
    assert "cuda:0" in result.stdout
    assert "--resource-tier" in result.stdout
    assert "rtx3050-6gb" in result.stdout
    assert "--shard" in result.stdout
    assert "1/18" in result.stdout


def test_remote_sweep_status_reads_collected_canonical_cache(tmp_path: Path) -> None:
    env_file = _write_remote_env(tmp_path)
    state_dir = tmp_path / "state"
    config_path = _write_sweep_fixture(tmp_path, state_dir)
    result = _run_script(
        "remote-sweep-status",
        "--config",
        str(config_path),
        env_file=env_file,
        extra_env={"KD_REMOTE_STATE_DIR": str(state_dir)},
    )
    assert "sweep=toy-sweep" in result.stdout
    assert "total=2 success=1 failed=0 pending=1" in result.stdout
    assert "dataset=ToyDataset total=2 success=1 failed=0 pending=1" in result.stdout
