"""Persistent GPU worker for SLinOSS sweep task groups."""

from __future__ import annotations

import json
import os
import sys
import traceback

# Torch-only worker: prevent accidental JAX GPU preallocation if JAX is imported
# transitively by a dependency stack.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sweep_slinoss.dataset_worker import run_task_group


def _emit(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _build_logger(log_path: str):
    def _log(message: str) -> None:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message.rstrip("\n"))
            log_file.write("\n")

    return _log


def main() -> int:
    _emit({"status": "ready"})

    while True:
        line = sys.stdin.readline()
        if not line:
            return 0

        command = json.loads(line)
        command_type = command.get("type")

        if command_type == "shutdown":
            _emit({"status": "bye"})
            return 0

        if command_type != "run":
            _emit({"status": "error", "error_message": f"Unknown command type: {command_type}"})
            continue

        payload_path = str(command["payload_path"])
        log_path = str(command["log_path"])

        try:
            with open(payload_path, "r", encoding="utf-8") as payload_file:
                payload = json.load(payload_file)
            result = run_task_group(payload, logger=_build_logger(log_path))
            _emit({"status": "ok", **result})
        except Exception as exc:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[Worker] Group failed: {exc}\n")
                log_file.write(traceback.format_exc())
                if not traceback.format_exc().endswith("\n"):
                    log_file.write("\n")
            _emit(
                {
                    "status": "error",
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=25),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
