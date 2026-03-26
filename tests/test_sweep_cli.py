from __future__ import annotations

import json
import subprocess
import sys


def test_sweep_plan_cli_smoke(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "grid.json"
    resource_profile_path = tmp_path / "resources.json"
    resource_profile_path.write_text(
        json.dumps(
            {
                "default_tier": "rtx3050-6gb",
                "rules": [
                    {
                        "match": {
                            "dataset_name": "EigenWorms",
                            "d_model": 64,
                            "n_layers": 2,
                            "include_time": True,
                        },
                        "resource_tier": "ada6000",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "name": "unit-sweep",
        "output_root": str(output_root),
        "resource_profile": str(resource_profile_path),
        "defaults": {
            "dataset": {
                "data_dir": "data_dir",
                "use_presplit": False,
                "include_time": True,
                "T": 1.0,
            },
            "training": {
                "num_steps": 20,
                "print_steps": 5,
                "lr": 0.001,
                "batch_size": 2,
            },
            "model": {
                "d_model": 64,
                "n_layers": 2,
            },
        },
        "datasets": [
            {
                "name": "EigenWorms",
                "seeds": [111],
            }
        ],
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sweep",
            "plan",
            "--config",
            str(config_path),
            "--preview",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "manifest.json").exists()
    assert (output_root / "plan.jsonl").exists()

    tier_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sweep",
            "plan",
            "--config",
            str(config_path),
            "--resource-tier",
            "ada6000",
            "--preview",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert tier_result.returncode == 0, tier_result.stderr
    assert "Selected 1 trials across 1 dataset-cache groups." in tier_result.stdout
