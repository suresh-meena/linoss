from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import os
import shlex
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / ".env"
GUIX_RUN = ROOT / "scripts" / "guix-run"
KNOWN_HOSTS_FILE = ROOT / ".remote-known-hosts"
DEFAULT_STATE_DIR = ROOT / ".remote-state"
DEFAULT_LEASE_ROOT = "/tmp/linoss-agent-leases"
DEFAULT_FANOUT_PARALLELISM = 4
VALID_TOOLCHAIN_MODES = {"auto", "system", "guix"}


class RemoteConfigError(RuntimeError):
    pass


AuthMode = Literal["key", "password"]


@dataclass(frozen=True)
class RemoteMachine:
    name: str
    host: str
    user: str
    port: int
    workdir: str | None
    auth: AuthMode
    ssh_key: str | None
    password: str | None
    gpu_class: str | None
    gpu_vram_gib: float | None
    gpu_count: int | None
    groups: tuple[str, ...]
    lease_root: str

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


@dataclass(frozen=True)
class CommandResult:
    machine: RemoteMachine
    returncode: int
    stdout: str = ""
    stderr: str = ""
    skipped_reason: str | None = None


@dataclass(frozen=True)
class SweepPaths:
    name: str
    output_root: str
    local_root: Path
    local_machine_root: Path
    local_canonical_root: Path
    ledger_path: Path


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat(timespec="seconds")


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RemoteConfigError(f"missing remote env file: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            raise RemoteConfigError(
                f"invalid env line {line_number} in {path}: {raw_line!r}"
            )
        key = key.strip()
        parsed = shlex.split(value, posix=True)
        if len(parsed) > 1:
            raise RemoteConfigError(
                "env value for "
                f"{key} on line {line_number} must parse to one token, got {len(parsed)}"
            )
        values[key] = parsed[0] if parsed else ""
    return values


def load_remote_env(path: Path) -> dict[str, str]:
    values = parse_env_file(path)
    for key, value in os.environ.items():
        if key.startswith("KD_REMOTE_"):
            values[key] = value
    return values


def normalize_name(name: str) -> str:
    return name.upper().replace("-", "_").replace(".", "_")


def display_group_name(env_key_suffix: str) -> str:
    return env_key_suffix.lower().replace("_", "-")


def configured_machine_names(env: dict[str, str]) -> list[str]:
    configured = env.get("KD_REMOTE_MACHINES", "")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]

    return [
        key[len("KD_REMOTE_") : -len("_HOST")].lower()
        for key in sorted(env)
        if key.startswith("KD_REMOTE_") and key.endswith("_HOST")
    ]


def configured_group_names(env: dict[str, str]) -> list[str]:
    groups = []
    for key in sorted(env):
        if key.startswith("KD_REMOTE_GROUP_"):
            groups.append(display_group_name(key[len("KD_REMOTE_GROUP_") :]))
    return groups


def group_machine_names(env: dict[str, str], group: str) -> list[str]:
    key = f"KD_REMOTE_GROUP_{normalize_name(group)}"
    raw = env.get(key)
    if raw is None:
        configured = ", ".join(configured_group_names(env)) or "<none>"
        raise RemoteConfigError(
            f"unknown machine group {group!r}; configured groups: {configured}"
        )
    names = [item.strip() for item in raw.split(",") if item.strip()]
    known = set(configured_machine_names(env))
    unknown = [name for name in names if name not in known]
    if unknown:
        raise RemoteConfigError(
            f"group {group!r} references unknown machines: {', '.join(sorted(unknown))}"
        )
    return names


def resolve_machine_name(env: dict[str, str], requested: str | None) -> str:
    if requested:
        return requested
    if env.get("KD_REMOTE_MACHINE"):
        return env["KD_REMOTE_MACHINE"]
    if env.get("KD_REMOTE_DEFAULT_MACHINE"):
        return env["KD_REMOTE_DEFAULT_MACHINE"]

    names = configured_machine_names(env)
    if len(names) == 1:
        return names[0]

    configured = ", ".join(names) if names else "<none>"
    raise RemoteConfigError(
        f"select a machine with --machine; configured machines: {configured}"
    )


def machine_field(env: dict[str, str], machine: str, field: str) -> str | None:
    return env.get(f"KD_REMOTE_{normalize_name(machine)}_{field}")


def require_machine_field(env: dict[str, str], machine: str, field: str) -> str:
    value = machine_field(env, machine, field)
    if value is None or value == "":
        raise RemoteConfigError(
            f"missing required field {field} for machine {machine!r}"
        )
    return value


def machine_groups(env: dict[str, str], machine: str) -> tuple[str, ...]:
    groups = [
        group
        for group in configured_group_names(env)
        if machine in group_machine_names(env, group)
    ]
    return tuple(sorted(groups))


def _parse_optional_int(value: str | None, *, field: str, machine: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RemoteConfigError(
            f"invalid integer {value!r} for field {field} on {machine!r}"
        ) from exc


def _parse_optional_float(
    value: str | None, *, field: str, machine: str
) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RemoteConfigError(
            f"invalid float {value!r} for field {field} on {machine!r}"
        ) from exc


def resolve_machine(env: dict[str, str], requested: str | None) -> RemoteMachine:
    name = resolve_machine_name(env, requested)
    host = require_machine_field(env, name, "HOST")
    user = require_machine_field(env, name, "USER")
    workdir = machine_field(env, name, "WORKDIR") or None
    port_text = machine_field(env, name, "PORT") or "22"
    auth_text = (machine_field(env, name, "AUTH") or "key").lower()
    ssh_key = machine_field(env, name, "SSH_KEY") or None
    password = machine_field(env, name, "PASSWORD") or None
    gpu_class = machine_field(env, name, "GPU_CLASS") or None
    gpu_vram_gib = _parse_optional_float(
        machine_field(env, name, "GPU_VRAM_GIB"),
        field="GPU_VRAM_GIB",
        machine=name,
    )
    gpu_count = _parse_optional_int(
        machine_field(env, name, "GPU_COUNT"),
        field="GPU_COUNT",
        machine=name,
    )
    lease_root = (
        machine_field(env, name, "LEASE_ROOT")
        or env.get("KD_REMOTE_LEASE_ROOT")
        or DEFAULT_LEASE_ROOT
    )

    try:
        port = int(port_text)
    except ValueError as exc:
        raise RemoteConfigError(
            f"invalid port {port_text!r} for machine {name!r}"
        ) from exc

    if ssh_key is not None:
        ssh_key = str(Path(ssh_key).expanduser())

    if auth_text == "key":
        auth: AuthMode = "key"
    elif auth_text == "password":
        auth = "password"
        if not password:
            raise RemoteConfigError(
                f"missing PASSWORD for password-auth machine {name!r}"
            )
    else:
        raise RemoteConfigError(
            f"unsupported auth mode {auth_text!r} for machine {name!r}"
        )

    return RemoteMachine(
        name=name,
        host=host,
        user=user,
        port=port,
        workdir=workdir,
        auth=auth,
        ssh_key=ssh_key,
        password=password,
        gpu_class=gpu_class,
        gpu_vram_gib=gpu_vram_gib,
        gpu_count=gpu_count,
        groups=machine_groups(env, name),
        lease_root=lease_root,
    )


def resolve_selected_machines(
    env: dict[str, str],
    *,
    machine: str | None,
    group: str | None,
    allow_default_single: bool,
) -> list[RemoteMachine]:
    if machine and group:
        raise RemoteConfigError("pass either --machine or --group, not both")
    if group:
        return [resolve_machine(env, name) for name in group_machine_names(env, group)]
    if machine or allow_default_single:
        return [resolve_machine(env, machine)]
    raise RemoteConfigError("select targets with --machine or --group")


def prefixed_env(machine: RemoteMachine) -> dict[str, str]:
    env = os.environ.copy()
    if uses_sshpass(machine):
        assert machine.password is not None
        env["SSHPASS"] = machine.password
    return env


def quoted_command(parts: list[str]) -> str:
    return shlex.join(parts)


def uses_sshpass(machine: RemoteMachine) -> bool:
    return machine.password is not None


def toolchain_mode() -> str:
    mode = os.environ.get("KD_REMOTE_TOOLCHAIN", "auto").strip().lower()
    if mode not in VALID_TOOLCHAIN_MODES:
        allowed = ", ".join(sorted(VALID_TOOLCHAIN_MODES))
        raise RemoteConfigError(
            f"invalid KD_REMOTE_TOOLCHAIN={mode!r}; expected one of: {allowed}"
        )
    return mode


def _have_system_tools(*tools: str) -> bool:
    return all(shutil.which(tool) for tool in tools)


def _have_guix_toolchain() -> bool:
    return GUIX_RUN.is_file() and shutil.which("guix") is not None


def _toolchain_error(*tools: str) -> RemoteConfigError:
    needed = ", ".join(tools)
    return RemoteConfigError(
        "missing required remote helper tools: "
        f"{needed}. Install them directly on PATH or install Guix and use "
        f"{GUIX_RUN}."
    )


def command_prefix(*tools: str) -> list[str]:
    mode = toolchain_mode()
    if mode == "system":
        if not _have_system_tools(*tools):
            raise _toolchain_error(*tools)
        return []
    if mode == "guix":
        if not _have_guix_toolchain():
            raise RemoteConfigError(
                f"KD_REMOTE_TOOLCHAIN=guix requires `guix` on PATH and {GUIX_RUN}"
            )
        return [str(GUIX_RUN)]

    if _have_system_tools(*tools):
        return []
    if _have_guix_toolchain():
        return [str(GUIX_RUN)]
    raise _toolchain_error(*tools)


def ssh_auth_options(machine: RemoteMachine) -> list[str]:
    options: list[str] = []
    if machine.auth == "password":
        return [
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
        ]
    if machine.ssh_key:
        options += ["-i", machine.ssh_key]
    if machine.password is not None:
        options += ["-o", "PreferredAuthentications=publickey,password"]
    else:
        options += ["-o", "BatchMode=yes"]
    return options


def ssh_command(
    machine: RemoteMachine,
    *,
    allocate_tty: bool,
    remote_command: str | None,
) -> list[str]:
    required_tools = ["ssh"]
    if uses_sshpass(machine):
        required_tools.append("sshpass")
    command: list[str] = command_prefix(*required_tools)
    if uses_sshpass(machine):
        command += ["sshpass", "-e"]
    command.append("ssh")
    if allocate_tty:
        command.append("-t")
    command += [
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "VisualHostKey=no",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_FILE}",
        "-p",
        str(machine.port),
    ]
    command += ssh_auth_options(machine)
    command.append(machine.target)
    if remote_command is not None:
        command.append(f"bash -lc {shlex.quote(remote_command)}")
    return command


def rsync_ssh_transport(machine: RemoteMachine) -> str:
    parts: list[str] = []
    required_tools = ["ssh"]
    if uses_sshpass(machine):
        required_tools.append("sshpass")
    parts += command_prefix(*required_tools)
    if uses_sshpass(machine):
        parts += ["sshpass", "-e"]
    parts += [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "VisualHostKey=no",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_FILE}",
        "-p",
        str(machine.port),
    ]
    parts += ssh_auth_options(machine)
    return quoted_command(parts)


def render_command(command: list[str], *, machine: RemoteMachine) -> str:
    prefix = "SSHPASS=<redacted> " if uses_sshpass(machine) else ""
    return prefix + quoted_command(command)


def run_command(command: list[str], *, machine: RemoteMachine, dry_run: bool) -> int:
    if dry_run:
        print(render_command(command, machine=machine))
        return 0
    completed = subprocess.run(command, env=prefixed_env(machine), check=False)
    return completed.returncode


def run_command_capture(
    command: list[str],
    *,
    machine: RemoteMachine,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=render_command(command, machine=machine) + "\n",
            stderr="",
        )
    return subprocess.run(
        command,
        env=prefixed_env(machine),
        check=False,
        capture_output=True,
        text=True,
    )


def build_remote_script(
    *commands: str, workdir: str | None, require_workdir: bool
) -> str:
    script_lines = ["set -eu"]
    if workdir:
        if require_workdir:
            script_lines.append(f"cd {shlex.quote(workdir)}")
        else:
            script_lines.append(
                f"if [ -d {shlex.quote(workdir)} ]; then cd {shlex.quote(workdir)}; fi"
            )
    elif require_workdir:
        raise RemoteConfigError("this command requires a configured WORKDIR")
    script_lines.extend(commands)
    return "\n".join(script_lines)


def remote_command_for_shell(
    *,
    command: list[str],
    workdir: str | None,
) -> tuple[bool, str | None]:
    if command:
        joined = quoted_command(command)
        return False, build_remote_script(
            joined, workdir=workdir, require_workdir=workdir is not None
        )
    if workdir:
        return True, build_remote_script(
            "exec ${SHELL:-/bin/sh} -l",
            workdir=workdir,
            require_workdir=True,
        )
    return True, None


def state_dir(env: dict[str, str]) -> Path:
    raw = env.get("KD_REMOTE_STATE_DIR")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_STATE_DIR


def sweep_paths(env: dict[str, str], sweep_name: str) -> SweepPaths:
    root = state_dir(env) / "sweeps" / sweep_name
    return SweepPaths(
        name=sweep_name,
        output_root="",
        local_root=root,
        local_machine_root=root / "machines",
        local_canonical_root=root / "canonical",
        ledger_path=root / "ledger.jsonl",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def load_state_cache(env: dict[str, str]) -> dict[str, Any]:
    path = state_dir(env) / "fleet-state.json"
    payload = load_json(path)
    if payload is None:
        return {"machines": {}}
    return payload


def save_state_cache(env: dict[str, str], payload: dict[str, Any]) -> None:
    path = state_dir(env) / "fleet-state.json"
    write_json(path, payload)


def update_machine_state(
    env: dict[str, str], machine: RemoteMachine, patch: dict[str, Any]
) -> None:
    payload = load_state_cache(env)
    machines = payload.setdefault("machines", {})
    current = dict(machines.get(machine.name, {}))
    current.update(patch)
    current["host"] = machine.host
    current["updated_at"] = iso_now()
    machines[machine.name] = current
    save_state_cache(env, payload)


def group_membership_text(machine: RemoteMachine) -> str:
    return ",".join(machine.groups)


def machine_summary(machine: RemoteMachine) -> str:
    parts = [
        f"name={machine.name}",
        f"host={machine.host}",
        f"user={machine.user}",
        f"port={machine.port}",
    ]
    if machine.workdir:
        parts.append(f"workdir={machine.workdir}")
    if machine.gpu_class:
        parts.append(f"gpu_class={machine.gpu_class}")
    if machine.gpu_vram_gib is not None:
        parts.append(f"gpu_vram_gib={machine.gpu_vram_gib:g}")
    if machine.gpu_count is not None:
        parts.append(f"gpu_count={machine.gpu_count}")
    if machine.groups:
        parts.append(f"groups={group_membership_text(machine)}")
    return " ".join(parts)


def fanout(
    machines: list[RemoteMachine],
    *,
    max_parallel: int,
    work,
) -> list[CommandResult]:
    if not machines:
        return []
    max_workers = max(1, min(max_parallel, len(machines)))
    if max_workers == 1:
        return [work(machine) for machine in machines]

    results: list[CommandResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_machine = {
            executor.submit(work, machine): machine for machine in machines
        }
        for future in concurrent.futures.as_completed(future_to_machine):
            results.append(future.result())
    results.sort(key=lambda item: item.machine.name)
    return results


def print_command_results(results: list[CommandResult]) -> int:
    overall = 0
    for result in results:
        header = f"== {result.machine.name} ({result.machine.host}) =="
        print(header)
        if result.skipped_reason is not None:
            print(f"skipped: {result.skipped_reason}")
            print()
            continue
        stdout = result.stdout.rstrip()
        stderr = result.stderr.rstrip()
        if stdout:
            print(stdout)
        if stderr:
            if stdout:
                print()
            print("[stderr]")
            print(stderr)
        if not stdout and not stderr:
            print("(no output)")
        print()
        if result.returncode != 0:
            overall = result.returncode
    return overall


def _csv_rows(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text.strip()))
    return [[column.strip() for column in row] for row in reader if row]


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def lease_path(machine: RemoteMachine, gpu: int) -> str:
    return f"{machine.lease_root.rstrip('/')}/gpu-{gpu}.json"


def _lease_is_active(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    expires = _parse_iso8601(str(payload.get("expires_at", "")))
    if expires is None:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > now_utc()


def _probe_gpu_status_machine(
    machine: RemoteMachine,
    *,
    dry_run: bool,
) -> CommandResult:
    payload = {
        "lease_root": machine.lease_root,
        "workdir": machine.workdir,
    }
    script = textwrap.dedent(
        f"""
        import csv
        import io
        import json
        import os
        import pathlib
        import subprocess
        import sys

        payload = json.loads({json.dumps(payload, sort_keys=True)!r})

        def run(args):
            return subprocess.run(args, check=False, capture_output=True, text=True)

        gpu_query = run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        if gpu_query.returncode != 0:
            print(
                json.dumps(
                    {{
                        "ok": False,
                        "error": gpu_query.stderr.strip() or gpu_query.stdout.strip() or "nvidia-smi query failed",
                    }},
                    sort_keys=True,
                )
            )
            raise SystemExit(0)

        app_query = run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )

        gpu_rows = list(csv.reader(io.StringIO(gpu_query.stdout)))
        app_rows = (
            list(csv.reader(io.StringIO(app_query.stdout)))
            if app_query.returncode == 0 and app_query.stdout.strip()
            else []
        )

        process_meta = {{}}
        for row in app_rows:
            if len(row) < 4:
                continue
            pid_text = row[1].strip()
            if not pid_text or pid_text.upper() == "N/A":
                continue
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            ps_result = run(["ps", "-p", str(pid), "-o", "user=", "-o", "args="])
            if ps_result.returncode == 0 and ps_result.stdout.strip():
                ps_line = ps_result.stdout.strip().splitlines()[0]
                user, _, command = ps_line.partition(" ")
                process_meta[pid] = {{
                    "user": user.strip(),
                    "command": command.strip(),
                }}

        apps_by_uuid: dict[str, list[dict[str, object]]] = {{}}
        for row in app_rows:
            if len(row) < 4:
                continue
            gpu_uuid, pid_text, process_name, used_memory = [column.strip() for column in row[:4]]
            if not gpu_uuid or gpu_uuid.upper() == "N/A":
                continue
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            used_gpu_memory_mib = None
            if used_memory and used_memory.upper() != "N/A":
                try:
                    used_gpu_memory_mib = int(float(used_memory))
                except ValueError:
                    used_gpu_memory_mib = None
            meta = process_meta.get(pid, {{}})
            apps_by_uuid.setdefault(gpu_uuid, []).append(
                {{
                    "pid": pid,
                    "process_name": process_name,
                    "used_gpu_memory_mib": used_gpu_memory_mib,
                    "user": meta.get("user"),
                    "command": meta.get("command"),
                }}
            )

        gpus = []
        lease_root = pathlib.Path(payload["lease_root"])
        for row in gpu_rows:
            if len(row) < 6:
                continue
            index_text, uuid, name, total_text, used_text, util_text = [column.strip() for column in row[:6]]
            try:
                index = int(index_text)
            except ValueError:
                continue
            total_mib = int(float(total_text))
            used_mib = int(float(used_text))
            util_percent = int(float(util_text)) if util_text and util_text.upper() != "N/A" else None
            lease_file = lease_root / f"gpu-{{index}}.json"
            lease = None
            if lease_file.is_file():
                try:
                    lease = json.loads(lease_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    lease = {{"invalid": True}}
            gpus.append(
                {{
                    "index": index,
                    "uuid": uuid,
                    "name": name,
                    "total_mib": total_mib,
                    "used_mib": used_mib,
                    "free_mib": total_mib - used_mib,
                    "utilization_gpu_percent": util_percent,
                    "compute_apps": apps_by_uuid.get(uuid, []),
                    "lease": lease,
                    "lease_path": str(lease_file),
                }}
            )

        payload = {{
            "ok": True,
            "host": os.uname().nodename,
            "user": os.environ.get("USER", ""),
            "cwd": os.getcwd(),
            "gpus": gpus,
            "workdir": payload.get("workdir"),
        }}
        print(json.dumps(payload, sort_keys=True))
        """
    ).strip()
    remote_command = build_remote_script(
        f"python3 - <<'PY'\n{script}\nPY",
        workdir=machine.workdir,
        require_workdir=False,
    )
    completed = run_command_capture(
        ssh_command(machine, allocate_tty=False, remote_command=remote_command),
        machine=machine,
        dry_run=dry_run,
    )
    return CommandResult(
        machine=machine,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _extract_status_payload(result: CommandResult) -> dict[str, Any]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RemoteConfigError(
            f"no GPU status payload returned from {result.machine.name}"
        )
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RemoteConfigError(
            f"failed to parse GPU status payload from {result.machine.name}: {lines[-1]!r}"
        ) from exc


def gpu_matches_policy(
    gpu: dict[str, Any],
    *,
    require_idle: bool,
    min_free_vram_gib: float | None,
    ignore_leases: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if require_idle and gpu.get("compute_apps"):
        reasons.append("active_compute")
    if min_free_vram_gib is not None:
        free_gib = float(gpu.get("free_mib", 0)) / 1024.0
        if free_gib + 1e-9 < min_free_vram_gib:
            reasons.append(f"free_vram<{min_free_vram_gib:g}GiB")
    if not ignore_leases and _lease_is_active(gpu.get("lease")):
        reasons.append("leased")
    return (not reasons), reasons


def probe_gpu_status(
    env: dict[str, str],
    machines: list[RemoteMachine],
    *,
    max_parallel: int,
    dry_run: bool,
) -> list[tuple[RemoteMachine, dict[str, Any] | None, CommandResult]]:
    results = fanout(
        machines,
        max_parallel=max_parallel,
        work=lambda machine: _probe_gpu_status_machine(machine, dry_run=dry_run),
    )
    parsed: list[tuple[RemoteMachine, dict[str, Any] | None, CommandResult]] = []
    for result in results:
        payload = None
        if result.returncode == 0 and not dry_run:
            payload = _extract_status_payload(result)
            update_machine_state(
                env,
                result.machine,
                {
                    "last_gpu_status_at": iso_now(),
                    "gpu_status": payload,
                },
            )
        parsed.append((result.machine, payload, result))
    return parsed


def _load_sweep_metadata(config_path: str) -> tuple[str, str]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise RemoteConfigError(f"missing sweep name in config: {config_path}")
    output_root = payload.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        output_root = os.path.join("outputs", "sweeps", name)
    return name, output_root


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def _choose_gpu(payload: dict[str, Any], gpu_index: int) -> dict[str, Any]:
    for gpu in payload.get("gpus", []):
        if int(gpu.get("index", -1)) == gpu_index:
            return gpu
    raise RemoteConfigError(f"remote host does not expose gpu index {gpu_index}")


def _relative_trial_result_path(trial: dict[str, Any], output_root: str) -> Path:
    output_dir = PurePosixPath(str(trial["output_dir"]))
    root = PurePosixPath(output_root)
    try:
        relative_dir = output_dir.relative_to(root)
    except ValueError as exc:
        raise RemoteConfigError(
            f"trial output_dir {output_dir} is not under output_root {output_root}"
        ) from exc
    return Path(relative_dir) / "result.json"


def _load_plan_trials(
    config_path: str, canonical_root: Path
) -> tuple[str, list[dict[str, Any]]]:
    plan_path = canonical_root / "plan.jsonl"
    output_root: str | None = None
    if plan_path.is_file():
        trials: list[dict[str, Any]] = []
        with plan_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trials.append(json.loads(line))
        if trials:
            first_output_dir = PurePosixPath(str(trials[0]["output_dir"]))
            try:
                output_root = str(first_output_dir.parent.parent.parent.parent)
            except Exception:
                output_root = None
        manifest = load_json(canonical_root / "manifest.json")
        if manifest and isinstance(manifest.get("output_root"), str):
            output_root = str(manifest["output_root"])
        if output_root is not None:
            return output_root, trials

    try:
        from sweep.config import load_sweep_definition
        from sweep.planner import build_sweep_plan
    except Exception:
        venv_python = ROOT / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            raise RemoteConfigError(
                "sweep plan metadata is missing from the collected cache and the local "
                "repo virtualenv is unavailable for regenerating it"
            )
        helper = textwrap.dedent(
            """
            import json
            import sys

            from sweep.config import load_sweep_definition
            from sweep.planner import build_sweep_plan

            config_path = sys.argv[1]
            plan = build_sweep_plan(load_sweep_definition(config_path))
            print(
                json.dumps(
                    {
                        "output_root": plan.output_root,
                        "trials": [trial.to_dict() for trial in plan.trials],
                    },
                    sort_keys=True,
                )
            )
            """
        ).strip()
        completed = subprocess.run(
            [str(venv_python), "-c", helper, config_path],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RemoteConfigError(
                "sweep plan metadata is missing from the collected cache and the local "
                "repo virtualenv failed to regenerate it"
            )
        payload = json.loads(completed.stdout)
        return str(payload["output_root"]), list(payload["trials"])

    definition = load_sweep_definition(config_path)
    plan = build_sweep_plan(definition)
    return plan.output_root, [trial.to_dict() for trial in plan.trials]


def _merge_stage_into_canonical(stage_root: Path, canonical_root: Path) -> None:
    canonical_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", f"{stage_root}/", f"{canonical_root}/"],
        check=True,
        cwd=ROOT,
    )


def _collect_include_args(include_logs: bool) -> list[str]:
    includes = [
        "--prune-empty-dirs",
        "--include",
        "*/",
        "--include",
        "manifest.json",
        "--include",
        "plan.jsonl",
        "--include",
        "results/***",
        "--include",
        "reports/***",
        "--include",
        "trial.json",
        "--include",
        "result.json",
    ]
    if include_logs:
        includes += ["--include", "training.log"]
    includes += ["--exclude", "*"]
    return includes


def _create_lease_payload(
    machine: RemoteMachine,
    *,
    gpu: int,
    owner: str,
    ttl_hours: float,
    note: str | None,
) -> dict[str, Any]:
    created_at = now_utc()
    expires_at = created_at + timedelta(hours=ttl_hours)
    return {
        "machine": machine.name,
        "gpu": gpu,
        "owner": owner,
        "note": note or "",
        "created_at": created_at.isoformat(timespec="seconds"),
        "expires_at": expires_at.isoformat(timespec="seconds"),
    }


def _remote_lease_command(
    machine: RemoteMachine,
    *,
    action: str,
    gpu: int,
    payload: dict[str, Any] | None,
    force: bool,
) -> str:
    script_payload = {
        "action": action,
        "gpu": gpu,
        "lease_path": lease_path(machine, gpu),
        "payload": payload,
        "force": force,
    }
    script = textwrap.dedent(
        f"""
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        payload = json.loads({json.dumps(script_payload, sort_keys=True)!r})
        lease_path = Path(payload["lease_path"])
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        action = payload["action"]
        current = None
        if lease_path.is_file():
            try:
                current = json.loads(lease_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {{"invalid": True}}

        def is_active(value):
            if not isinstance(value, dict):
                return False
            expires_at = value.get("expires_at")
            if not expires_at:
                return False
            try:
                parsed = datetime.fromisoformat(expires_at)
            except ValueError:
                return False
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed > datetime.now(timezone.utc)

        if action == "show":
            print(json.dumps({{"lease_path": str(lease_path), "lease": current}}, sort_keys=True))
            raise SystemExit(0)
        if action == "release":
            if lease_path.exists():
                lease_path.unlink()
            print(json.dumps({{"released": True, "lease_path": str(lease_path)}}, sort_keys=True))
            raise SystemExit(0)

        if action != "acquire":
            raise SystemExit(2)

        if is_active(current) and not payload["force"]:
            print(
                json.dumps(
                    {{
                        "acquired": False,
                        "lease_path": str(lease_path),
                        "lease": current,
                    }},
                    sort_keys=True,
                )
            )
            raise SystemExit(3)

        new_payload = payload["payload"]
        lease_path.write_text(json.dumps(new_payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        print(json.dumps({{"acquired": True, "lease_path": str(lease_path), "lease": new_payload}}, sort_keys=True))
        """
    ).strip()
    return build_remote_script(
        f"python3 - <<'PY'\n{script}\nPY",
        workdir=machine.workdir,
        require_workdir=False,
    )


def command_list(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    if args.groups:
        for group in configured_group_names(env):
            print(group)
        return 0

    names = configured_machine_names(env)
    if args.group:
        names = group_machine_names(env, args.group)
    machines = [resolve_machine(env, name) for name in names]
    for machine in machines:
        if args.verbose:
            print(machine_summary(machine))
        else:
            print(machine.name)
    return 0


def command_print_config(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )
    for index, machine in enumerate(machines):
        if index:
            print()
        print(f"env_file={args.env_file}")
        print(f"machine={machine.name}")
        print(f"host={machine.host}")
        print(f"user={machine.user}")
        print(f"port={machine.port}")
        print(f"workdir={machine.workdir or ''}")
        print(f"auth={machine.auth}")
        print(f"ssh_key={machine.ssh_key or ''}")
        print(f"has_password={'true' if machine.password else 'false'}")
        print(f"groups={group_membership_text(machine)}")
        print(f"gpu_class={machine.gpu_class or ''}")
        print(
            f"gpu_vram_gib={'' if machine.gpu_vram_gib is None else f'{machine.gpu_vram_gib:g}'}"
        )
        print(f"gpu_count={'' if machine.gpu_count is None else machine.gpu_count}")
        print(f"lease_root={machine.lease_root}")
    return 0


def command_shell(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machine = resolve_machine(env, args.machine)
    workdir = None if args.no_workdir else (args.cwd or machine.workdir)
    allocate_tty, remote_command = remote_command_for_shell(
        command=args.command,
        workdir=workdir,
    )
    return run_command(
        ssh_command(machine, allocate_tty=allocate_tty, remote_command=remote_command),
        machine=machine,
        dry_run=args.dry_run,
    )


def upload_excludes() -> list[str]:
    return [
        ".env",
        ".env.local",
        ".git/",
        ".mypy_cache/",
        ".nox/",
        ".pytest_cache/",
        ".pyright/",
        ".ruff_cache/",
        ".venv/",
        ".remote-state/",
        "__pycache__/",
        "build/",
        "dist/",
        "log/",
        "remote-downloads/",
        "runs/",
        "*.pyc",
    ]


def ensure_remote_directory(machine: RemoteMachine, path: str, *, dry_run: bool) -> int:
    remote_command = build_remote_script(
        f"mkdir -p {shlex.quote(path)}",
        workdir=None,
        require_workdir=False,
    )
    return run_command(
        ssh_command(
            machine,
            allocate_tty=False,
            remote_command=remote_command,
        ),
        machine=machine,
        dry_run=dry_run,
    )


def _build_rsync_command(
    machine: RemoteMachine,
    *,
    direction: str,
    source: str,
    dest: str,
    delete: bool,
    extra_args: list[str] | None = None,
) -> list[str]:
    command: list[str] = command_prefix("rsync") + ["rsync", "-az"]
    if delete:
        command.append("--delete")
    if extra_args:
        command += extra_args
    if direction == "upload":
        for pattern in upload_excludes():
            command += ["--exclude", pattern]
    command += ["-e", rsync_ssh_transport(machine)]
    if direction == "upload":
        command += [source, f"{machine.target}:{dest}"]
    else:
        command += [f"{machine.target}:{source}", dest]
    return command


def _rsync_one(machine: RemoteMachine, args: argparse.Namespace) -> CommandResult:
    mkdir_stdout = ""
    if args.direction == "upload":
        source = args.source or f"{ROOT}/"
        if args.dest is not None:
            dest = args.dest
        elif machine.workdir is not None:
            dest = f"{machine.workdir}/"
        else:
            raise RemoteConfigError(
                f"machine {machine.name!r} has no WORKDIR; pass --dest explicitly for upload"
            )
        status = ensure_remote_directory(machine, dest, dry_run=args.dry_run)
        if status != 0:
            return CommandResult(machine=machine, returncode=status)
    else:
        if args.source is not None:
            source = args.source
        elif machine.workdir is not None:
            source = f"{machine.workdir}/"
        else:
            raise RemoteConfigError(
                f"machine {machine.name!r} has no WORKDIR; pass --source explicitly for download"
            )
        dest = args.dest or str(ROOT / "remote-downloads" / machine.name) + "/"
        mkdir_command = ["mkdir", "-p", dest]
        if args.dry_run:
            mkdir_stdout = quoted_command(mkdir_command) + "\n"
        else:
            subprocess.run(mkdir_command, check=True)
            mkdir_stdout = ""

    if args.direction == "upload":
        command = _build_rsync_command(
            machine,
            direction="upload",
            source=source,
            dest=dest,
            delete=args.delete,
        )
    else:
        command = _build_rsync_command(
            machine,
            direction="download",
            source=source,
            dest=dest,
            delete=args.delete,
        )

    completed = run_command_capture(command, machine=machine, dry_run=args.dry_run)
    stdout = completed.stdout
    if args.direction == "download" and args.dry_run:
        stdout = mkdir_stdout + stdout
    elif args.direction == "download":
        stdout = mkdir_stdout + stdout

    if completed.returncode == 0:
        update_machine_state(
            load_remote_env(args.env_file),
            machine,
            {"last_sync_at": iso_now(), "last_sync_direction": args.direction},
        )
    return CommandResult(
        machine=machine,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=completed.stderr,
    )


def command_rsync(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )
    results = fanout(
        machines,
        max_parallel=args.max_parallel,
        work=lambda machine: _rsync_one(machine, args),
    )
    return print_command_results(results)


def _smoke_one(machine: RemoteMachine, *, dry_run: bool) -> CommandResult:
    smoke = textwrap.dedent(
        """
        echo "host=$(hostname)"
        echo "pwd=$(pwd)"
        echo "user=$(whoami)"
        if command -v python3 >/dev/null 2>&1; then python3 --version; fi
        if command -v nvidia-smi >/dev/null 2>&1; then
          nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        fi
        """
    ).strip()
    remote_command = build_remote_script(
        smoke,
        workdir=machine.workdir,
        require_workdir=machine.workdir is not None,
    )
    completed = run_command_capture(
        ssh_command(machine, allocate_tty=False, remote_command=remote_command),
        machine=machine,
        dry_run=dry_run,
    )
    return CommandResult(
        machine=machine,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def command_smoke(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )
    results = fanout(
        machines,
        max_parallel=args.max_parallel,
        work=lambda machine: _smoke_one(machine, dry_run=args.dry_run),
    )
    if not args.dry_run:
        for result in results:
            if result.returncode == 0:
                update_machine_state(
                    env,
                    result.machine,
                    {"last_smoke_at": iso_now(), "last_smoke_ok": True},
                )
    return print_command_results(results)


def _filter_machines_by_gpu_policy(
    env: dict[str, str],
    machines: list[RemoteMachine],
    *,
    gpu: int,
    require_idle: bool,
    min_free_vram_gib: float | None,
    ignore_leases: bool,
    max_parallel: int,
    dry_run: bool,
) -> tuple[list[RemoteMachine], list[CommandResult]]:
    if not require_idle and min_free_vram_gib is None and ignore_leases:
        return machines, []
    if not require_idle and min_free_vram_gib is None:
        return machines, []

    probes = probe_gpu_status(env, machines, max_parallel=max_parallel, dry_run=dry_run)
    selected: list[RemoteMachine] = []
    skipped: list[CommandResult] = []
    for machine, payload, raw in probes:
        if raw.returncode != 0 or payload is None:
            skipped.append(raw)
            continue
        if not payload.get("ok", False):
            skipped.append(
                CommandResult(
                    machine=machine,
                    returncode=1,
                    stdout=raw.stdout,
                    stderr=payload.get("error", ""),
                )
            )
            continue
        try:
            gpu_payload = _choose_gpu(payload, gpu)
        except RemoteConfigError as exc:
            skipped.append(
                CommandResult(machine=machine, returncode=1, stderr=str(exc))
            )
            continue
        allowed, reasons = gpu_matches_policy(
            gpu_payload,
            require_idle=require_idle,
            min_free_vram_gib=min_free_vram_gib,
            ignore_leases=ignore_leases,
        )
        if allowed:
            selected.append(machine)
            continue
        skipped.append(
            CommandResult(
                machine=machine,
                returncode=0,
                skipped_reason=", ".join(reasons),
            )
        )
    return selected, skipped


def command_gpu_status(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )
    probe_rows = probe_gpu_status(
        env,
        machines,
        max_parallel=args.max_parallel,
        dry_run=args.dry_run,
    )
    output_rows: list[dict[str, Any]] = []
    errors: list[CommandResult] = []
    for machine, payload, raw in probe_rows:
        if raw.returncode != 0:
            errors.append(raw)
            continue
        if args.dry_run:
            errors.append(raw)
            continue
        assert payload is not None
        if not payload.get("ok", False):
            errors.append(
                CommandResult(
                    machine=machine, returncode=1, stderr=str(payload.get("error", ""))
                )
            )
            continue
        for gpu in payload.get("gpus", []):
            allowed, reasons = gpu_matches_policy(
                gpu,
                require_idle=args.require_idle,
                min_free_vram_gib=args.min_free_vram_gib,
                ignore_leases=args.ignore_leases,
            )
            row = {
                "machine": machine.name,
                "host": machine.host,
                "gpu": gpu["index"],
                "gpu_name": gpu["name"],
                "total_gib": round(float(gpu["total_mib"]) / 1024.0, 2),
                "used_gib": round(float(gpu["used_mib"]) / 1024.0, 2),
                "free_gib": round(float(gpu["free_mib"]) / 1024.0, 2),
                "utilization_gpu_percent": gpu.get("utilization_gpu_percent"),
                "active_compute": bool(gpu.get("compute_apps")),
                "active_lease": _lease_is_active(gpu.get("lease")),
                "eligible": allowed,
                "reasons": reasons,
                "compute_apps": gpu.get("compute_apps", []),
                "lease": gpu.get("lease"),
            }
            if (
                (
                    args.require_idle
                    or args.min_free_vram_gib is not None
                    or not args.ignore_leases
                )
                and not args.all
                and not allowed
            ):
                continue
            output_rows.append(row)

    if args.json:
        print(json.dumps(output_rows, indent=2, sort_keys=True))
    else:
        for row in output_rows:
            reasons = ",".join(row["reasons"]) if row["reasons"] else "-"
            print(
                " ".join(
                    [
                        f"machine={row['machine']}",
                        f"gpu={row['gpu']}",
                        f"name={row['gpu_name']}",
                        f"free_gib={row['free_gib']:.2f}/{row['total_gib']:.2f}",
                        f"util={row['utilization_gpu_percent']}",
                        f"active_compute={'yes' if row['active_compute'] else 'no'}",
                        f"lease={'yes' if row['active_lease'] else 'no'}",
                        f"eligible={'yes' if row['eligible'] else 'no'}",
                        f"reasons={reasons}",
                    ]
                )
            )
    if errors:
        return print_command_results(errors)
    return 0


def command_for_each(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=False,
    )
    machines, skipped = _filter_machines_by_gpu_policy(
        env,
        machines,
        gpu=args.gpu,
        require_idle=args.require_idle,
        min_free_vram_gib=args.min_free_vram_gib,
        ignore_leases=args.ignore_leases,
        max_parallel=args.max_parallel,
        dry_run=args.dry_run,
    )
    if not args.command:
        raise RemoteConfigError("remote-for-each requires a command after --")

    def work(machine: RemoteMachine) -> CommandResult:
        workdir = None if args.no_workdir else (args.cwd or machine.workdir)
        allocate_tty, remote_command = remote_command_for_shell(
            command=args.command,
            workdir=workdir,
        )
        completed = run_command_capture(
            ssh_command(
                machine, allocate_tty=allocate_tty, remote_command=remote_command
            ),
            machine=machine,
            dry_run=args.dry_run,
        )
        return CommandResult(
            machine=machine,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    results = fanout(machines, max_parallel=args.max_parallel, work=work)
    return print_command_results(skipped + results)


def command_lease(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )
    owner = args.owner or os.environ.get("USER", "unknown")
    payload = None
    if args.action == "acquire":
        payload = _create_lease_payload(
            resolve_machine(env, machines[0].name),
            gpu=args.gpu,
            owner=owner,
            ttl_hours=args.ttl_hours,
            note=args.note,
        )

    def work(machine: RemoteMachine) -> CommandResult:
        machine_payload = payload
        if args.action == "acquire":
            machine_payload = _create_lease_payload(
                machine,
                gpu=args.gpu,
                owner=owner,
                ttl_hours=args.ttl_hours,
                note=args.note,
            )
        remote_command = _remote_lease_command(
            machine,
            action=args.action,
            gpu=args.gpu,
            payload=machine_payload,
            force=args.force,
        )
        completed = run_command_capture(
            ssh_command(machine, allocate_tty=False, remote_command=remote_command),
            machine=machine,
            dry_run=args.dry_run,
        )
        return CommandResult(
            machine=machine,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    results = fanout(machines, max_parallel=args.max_parallel, work=work)
    return print_command_results(results)


def command_setup(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )

    def work(machine: RemoteMachine) -> CommandResult:
        stdout_parts: list[str] = []
        if not args.skip_sync:
            if machine.workdir is None:
                return CommandResult(
                    machine=machine,
                    returncode=2,
                    stderr="setup requires a configured WORKDIR",
                )
            mkdir_status = ensure_remote_directory(
                machine, machine.workdir, dry_run=args.dry_run
            )
            if mkdir_status != 0:
                return CommandResult(machine=machine, returncode=mkdir_status)
            rsync_result = _rsync_one(
                machine,
                argparse.Namespace(
                    env_file=args.env_file,
                    direction="upload",
                    source=args.source,
                    dest=args.dest,
                    delete=args.delete,
                    dry_run=args.dry_run,
                    machine=machine.name,
                    group=None,
                    max_parallel=1,
                ),
            )
            if rsync_result.returncode != 0:
                return rsync_result
            stdout_parts.append(rsync_result.stdout.rstrip())

        check_lines = []
        for path in args.check_path:
            check_lines.append(f"test -e {shlex.quote(path)}")
        install_lines = []
        if not args.skip_install:
            install_lines += [
                f"{shlex.quote(args.python)} -m venv .venv",
                ".venv/bin/python -m pip install -U pip",
                ".venv/bin/pip install -r requirements.txt",
            ]
        install_lines += [
            "python3 --version",
            "if command -v nvidia-smi >/dev/null 2>&1; then "
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; "
            "else echo 'nvidia-smi missing' >&2; exit 1; fi",
        ]
        remote_command = build_remote_script(
            *install_lines,
            *check_lines,
            workdir=machine.workdir,
            require_workdir=True,
        )
        completed = run_command_capture(
            ssh_command(machine, allocate_tty=False, remote_command=remote_command),
            machine=machine,
            dry_run=args.dry_run,
        )
        stdout_parts.append(completed.stdout.rstrip())
        if completed.returncode == 0 and not args.dry_run:
            update_machine_state(
                env,
                machine,
                {"last_setup_at": iso_now(), "last_setup_ok": True},
            )
        return CommandResult(
            machine=machine,
            returncode=completed.returncode,
            stdout="\n".join(part for part in stdout_parts if part),
            stderr=completed.stderr,
        )

    results = fanout(machines, max_parallel=args.max_parallel, work=work)
    return print_command_results(results)


def command_collect(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )
    sweep_name = args.sweep
    paths = sweep_paths(env, sweep_name)
    paths.local_machine_root.mkdir(parents=True, exist_ok=True)
    paths.local_canonical_root.mkdir(parents=True, exist_ok=True)

    def work(machine: RemoteMachine) -> CommandResult:
        if machine.workdir is None:
            return CommandResult(
                machine=machine, returncode=2, stderr="collect requires WORKDIR"
            )
        source = f"{machine.workdir}/outputs/sweeps/{sweep_name}/"
        dest = paths.local_machine_root / machine.name
        dest.mkdir(parents=True, exist_ok=True)
        command = _build_rsync_command(
            machine,
            direction="download",
            source=source,
            dest=str(dest) + "/",
            delete=False,
            extra_args=_collect_include_args(args.include_logs),
        )
        completed = run_command_capture(command, machine=machine, dry_run=args.dry_run)
        return CommandResult(
            machine=machine,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    results = fanout(machines, max_parallel=args.max_parallel, work=work)
    if not args.dry_run:
        for result in results:
            if result.returncode != 0:
                continue
            stage_root = paths.local_machine_root / result.machine.name
            _merge_stage_into_canonical(stage_root, paths.local_canonical_root)
            ledger_entry = {
                "event": "collect",
                "timestamp": iso_now(),
                "machine": result.machine.name,
                "sweep": sweep_name,
                "include_logs": args.include_logs,
                "stage_root": str(stage_root),
                "canonical_root": str(paths.local_canonical_root),
            }
            append_jsonl(paths.ledger_path, ledger_entry)
            update_machine_state(
                env,
                result.machine,
                {"last_collect_at": iso_now(), "last_collected_sweep": sweep_name},
            )
    return print_command_results(results)


def command_sweep_run(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machine = resolve_machine(env, args.machine)
    if machine.workdir is None:
        raise RemoteConfigError("remote-sweep-run requires a configured WORKDIR")

    probes = probe_gpu_status(
        env,
        [machine],
        max_parallel=1,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        _, payload, raw = probes[0]
        if raw.returncode != 0 or payload is None or not payload.get("ok", False):
            return print_command_results([raw])
        gpu_payload = _choose_gpu(payload, args.gpu)
        allowed, reasons = gpu_matches_policy(
            gpu_payload,
            require_idle=args.require_idle,
            min_free_vram_gib=args.min_free_vram_gib,
            ignore_leases=args.ignore_leases,
        )
        if not allowed:
            return print_command_results(
                [
                    CommandResult(
                        machine=machine,
                        returncode=0,
                        skipped_reason=", ".join(reasons),
                    )
                ]
            )

    sweep_name, _ = _load_sweep_metadata(args.config)
    paths = sweep_paths(env, sweep_name)
    job_id = (
        f"{sweep_name}-"
        f"{machine.name}-"
        f"gpu{args.gpu}-"
        f"{args.shard.replace('/', 'of') if args.shard else 'all'}-"
        f"{now_utc().strftime('%Y%m%dT%H%M%SZ')}"
    )
    lease_payload = _create_lease_payload(
        machine,
        gpu=args.gpu,
        owner=args.owner or os.environ.get("USER", "unknown"),
        ttl_hours=args.ttl_hours,
        note=args.note,
    )
    remote_payload = {
        "job_id": job_id,
        "workdir": machine.workdir,
        "gpu": args.gpu,
        "lease_path": lease_path(machine, args.gpu),
        "lease_payload": lease_payload,
        "force": args.force_lease,
        "log_subdir": args.log_subdir,
        "sweep_name": sweep_name,
        "argv": [
            ".venv/bin/python",
            "-m",
            "sweep",
            "run",
            "--config",
            args.config,
            "--devices",
            "cuda:0",
        ],
    }
    if args.resource_tier:
        remote_payload["argv"] += ["--resource-tier", args.resource_tier]
    if args.shard:
        remote_payload["argv"] += ["--shard", args.shard]
    for dataset in args.dataset:
        remote_payload["argv"] += ["--dataset", dataset]
    if args.max_groups is not None:
        remote_payload["argv"] += ["--max-groups", str(args.max_groups)]
    if args.max_trials is not None:
        remote_payload["argv"] += ["--max-trials", str(args.max_trials)]
    if args.retry_failed:
        remote_payload["argv"].append("--retry-failed")
    if args.force:
        remote_payload["argv"].append("--force")

    remote_script = textwrap.dedent(
        f"""
        import json
        import os
        import shlex
        import subprocess
        from datetime import datetime, timezone
        from pathlib import Path

        payload = json.loads({json.dumps(remote_payload, sort_keys=True)!r})

        workdir = Path(payload["workdir"])
        if not workdir.is_dir():
            print(json.dumps({{"ok": False, "error": f"missing workdir: {{workdir}}"}}))
            raise SystemExit(2)
        if not (workdir / ".venv" / "bin" / "python").is_file():
            print(json.dumps({{"ok": False, "error": f"missing virtualenv at {{workdir / '.venv'}}"}}))
            raise SystemExit(2)

        lease_path = Path(payload["lease_path"])
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        current = None
        if lease_path.is_file():
            try:
                current = json.loads(lease_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {{"invalid": True}}

        def lease_active(entry):
            if not isinstance(entry, dict):
                return False
            expires_at = entry.get("expires_at")
            if not expires_at:
                return False
            try:
                parsed = datetime.fromisoformat(expires_at)
            except ValueError:
                return False
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed > datetime.now(timezone.utc)

        if lease_active(current) and not payload["force"]:
            print(json.dumps({{"ok": False, "error": "active lease present", "lease": current}}, sort_keys=True))
            raise SystemExit(3)

        log_dir = workdir / "log" / payload["log_subdir"] / payload["sweep_name"]
        log_dir.mkdir(parents=True, exist_ok=True)
        runner_dir = workdir / ".remote-jobs" / payload["sweep_name"]
        runner_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{{payload['job_id']}}.log"
        launcher_path = runner_dir / f"{{payload['job_id']}}.sh"
        done_path = runner_dir / f"{{payload['job_id']}}.done.json"

        lease_text = json.dumps(payload["lease_payload"], indent=2, sort_keys=True)
        argv_text = shlex.join(payload["argv"])
        launcher = "\\n".join(
            [
                "#!/usr/bin/env bash",
                "set -eu",
                f"cleanup() {{ rm -f {{shlex.quote(str(lease_path))}}; }}",
                "trap cleanup EXIT",
                f"cd {{shlex.quote(str(workdir))}}",
                f"cat > {{shlex.quote(str(lease_path))}} <<'JSON'",
                lease_text,
                "JSON",
                f"exec env CUDA_VISIBLE_DEVICES={{payload['gpu']}} " + argv_text,
            ]
        )
        launcher_path.write_text(launcher + "\\n", encoding="utf-8")
        launcher_path.chmod(0o755)

        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                ["bash", str(launcher_path)],
                cwd=str(workdir),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        done_path.write_text(
            json.dumps(
                {{
                    "job_id": payload["job_id"],
                    "pid": process.pid,
                    "log_path": str(log_path),
                    "launcher_path": str(launcher_path),
                    "lease_path": str(lease_path),
                    "launched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }},
                indent=2,
                sort_keys=True,
            )
            + "\\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {{
                    "ok": True,
                    "job_id": payload["job_id"],
                    "pid": process.pid,
                    "log_path": str(log_path),
                    "launcher_path": str(launcher_path),
                    "lease_path": str(lease_path),
                    "done_path": str(done_path),
                    "argv": payload["argv"],
                }},
                sort_keys=True,
            )
        )
        """
    ).strip()

    remote_command = build_remote_script(
        f"python3 - <<'PY'\n{remote_script}\nPY",
        workdir=machine.workdir,
        require_workdir=True,
    )
    completed = run_command_capture(
        ssh_command(machine, allocate_tty=False, remote_command=remote_command),
        machine=machine,
        dry_run=args.dry_run,
    )
    result = CommandResult(
        machine=machine,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if args.dry_run:
        return print_command_results([result])
    if completed.returncode != 0:
        return print_command_results([result])
    payload = _extract_status_payload(result)
    if not payload.get("ok", False):
        return print_command_results(
            [
                CommandResult(
                    machine=machine,
                    returncode=1,
                    stdout=result.stdout,
                    stderr=str(payload.get("error", "")),
                )
            ]
        )
    paths.local_root.mkdir(parents=True, exist_ok=True)
    ledger_entry = {
        "event": "launch",
        "timestamp": iso_now(),
        "machine": machine.name,
        "host": machine.host,
        "gpu": args.gpu,
        "config": args.config,
        "sweep": sweep_name,
        "resource_tier": args.resource_tier,
        "shard": args.shard,
        "datasets": args.dataset,
        "job_id": payload["job_id"],
        "pid": payload["pid"],
        "remote_log_path": payload["log_path"],
        "lease_path": payload["lease_path"],
        "note": args.note or "",
        "commit": _git_head(),
    }
    append_jsonl(paths.ledger_path, ledger_entry)
    update_machine_state(
        env,
        machine,
        {
            "last_launch_at": iso_now(),
            "last_launch_sweep": sweep_name,
            "last_launch_job_id": payload["job_id"],
            "last_launch_gpu": args.gpu,
            "last_launch_log_path": payload["log_path"],
        },
    )
    print(json.dumps(ledger_entry, indent=2, sort_keys=True))
    return 0


def command_sweep_status(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    sweep_name, output_root = _load_sweep_metadata(args.config)
    paths = sweep_paths(env, sweep_name)
    canonical_root = paths.local_canonical_root
    canonical_root.mkdir(parents=True, exist_ok=True)
    plan_output_root, trials = _load_plan_trials(args.config, canonical_root)
    if not plan_output_root:
        plan_output_root = output_root

    summary = {
        "success": 0,
        "failed": 0,
        "pending": 0,
        "total": 0,
    }
    dataset_summary: dict[str, dict[str, int]] = {}
    detailed_rows: list[dict[str, Any]] = []

    for trial in trials:
        dataset_name = str(trial["dataset"]["name"])
        record_path = canonical_root / _relative_trial_result_path(
            trial, plan_output_root
        )
        status = "pending"
        record_payload = None
        if record_path.is_file():
            record_payload = load_json(record_path)
            if record_payload is not None:
                status = str(record_payload.get("status", "pending"))
        summary[status] += 1
        summary["total"] += 1
        dataset_row = dataset_summary.setdefault(
            dataset_name,
            {"success": 0, "failed": 0, "pending": 0, "total": 0},
        )
        dataset_row[status] += 1
        dataset_row["total"] += 1
        detailed_rows.append(
            {
                "trial_id": trial["trial_id"],
                "dataset": dataset_name,
                "seed": trial["seed"],
                "resource_tier": trial.get("resource_tier"),
                "status": status,
                "record_path": str(record_path),
            }
        )

    if args.list:
        selected_rows = [
            row
            for row in detailed_rows
            if args.list == "all" or row["status"] == args.list
        ]
        if args.json:
            print(json.dumps(selected_rows, indent=2, sort_keys=True))
            return 0
        for row in selected_rows:
            print(
                " ".join(
                    [
                        f"trial_id={row['trial_id']}",
                        f"dataset={row['dataset']}",
                        f"seed={row['seed']}",
                        f"resource_tier={row['resource_tier']}",
                        f"status={row['status']}",
                    ]
                )
            )
        return 0

    machine_counts: dict[str, int] = {}
    if paths.local_machine_root.is_dir():
        for machine_dir in sorted(paths.local_machine_root.iterdir()):
            if not machine_dir.is_dir():
                continue
            count = sum(1 for _ in machine_dir.rglob("result.json"))
            machine_counts[machine_dir.name] = count

    payload = {
        "sweep": sweep_name,
        "canonical_root": str(canonical_root),
        "ledger_path": str(paths.ledger_path),
        "summary": summary,
        "datasets": dataset_summary,
        "machine_result_counts": machine_counts,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"sweep={sweep_name}")
    print(f"canonical_root={canonical_root}")
    print(
        " ".join(
            [
                f"total={summary['total']}",
                f"success={summary['success']}",
                f"failed={summary['failed']}",
                f"pending={summary['pending']}",
            ]
        )
    )
    print()
    print("by_dataset:")
    for dataset_name in sorted(dataset_summary):
        row = dataset_summary[dataset_name]
        print(
            " ".join(
                [
                    f"dataset={dataset_name}",
                    f"total={row['total']}",
                    f"success={row['success']}",
                    f"failed={row['failed']}",
                    f"pending={row['pending']}",
                ]
            )
        )
    if machine_counts:
        print()
        print("by_machine_cache:")
        for machine_name, count in sorted(machine_counts.items()):
            print(f"machine={machine_name} cached_results={count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remotectl")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.environ.get("KD_REMOTE_ENV_FILE", DEFAULT_ENV_FILE)),
        help="Path to the machine env file",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    list_parser = subparsers.add_parser("list", help="List configured remote machines")
    list_parser.add_argument(
        "--group", help="Filter listed machines to one configured group"
    )
    list_parser.add_argument(
        "--groups", action="store_true", help="List configured group names"
    )
    list_parser.add_argument(
        "--verbose", action="store_true", help="Show machine metadata"
    )
    list_parser.set_defaults(handler=command_list)

    config_parser = subparsers.add_parser(
        "print-config", help="Print resolved machine configuration"
    )
    config_parser.add_argument("--machine", help="Machine name from the env file")
    config_parser.add_argument(
        "--group", help="Print configs for all machines in a group"
    )
    config_parser.set_defaults(handler=command_print_config)

    shell_parser = subparsers.add_parser(
        "shell", help="Open a remote shell or run a command"
    )
    shell_parser.add_argument("--machine", help="Machine name from the env file")
    shell_parser.add_argument("--cwd", help="Override the remote working directory")
    shell_parser.add_argument(
        "--no-workdir",
        action="store_true",
        help="Do not cd into the configured workdir",
    )
    shell_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command instead of executing it",
    )
    shell_parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to run remotely"
    )
    shell_parser.set_defaults(handler=command_shell)

    for_each_parser = subparsers.add_parser(
        "for-each", help="Run one command across multiple machines"
    )
    for_each_parser.add_argument("--machine", help="Run on exactly one machine")
    for_each_parser.add_argument(
        "--group", help="Run across a configured machine group"
    )
    for_each_parser.add_argument("--cwd", help="Override the remote working directory")
    for_each_parser.add_argument(
        "--no-workdir",
        action="store_true",
        help="Do not cd into the configured workdir",
    )
    for_each_parser.add_argument(
        "--gpu", type=int, default=0, help="Physical GPU index to policy-check"
    )
    for_each_parser.add_argument(
        "--require-idle",
        action="store_true",
        help="Skip targets with active compute jobs",
    )
    for_each_parser.add_argument(
        "--min-free-vram-gib",
        type=float,
        help="Skip targets below this free VRAM threshold",
    )
    for_each_parser.add_argument(
        "--ignore-leases", action="store_true", help="Ignore active remote lease files"
    )
    for_each_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    for_each_parser.add_argument("--dry-run", action="store_true")
    for_each_parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to run after --"
    )
    for_each_parser.set_defaults(handler=command_for_each)

    rsync_parser = subparsers.add_parser(
        "rsync", help="Sync files to or from remote machines"
    )
    rsync_parser.add_argument("--machine", help="Machine name from the env file")
    rsync_parser.add_argument("--group", help="Operate on every machine in a group")
    rsync_parser.add_argument(
        "--direction", choices=["upload", "download"], default="upload"
    )
    rsync_parser.add_argument("--source", help="Override the sync source path")
    rsync_parser.add_argument("--dest", help="Override the sync destination path")
    rsync_parser.add_argument(
        "--delete", action="store_true", help="Pass --delete to rsync"
    )
    rsync_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    rsync_parser.add_argument("--dry-run", action="store_true")
    rsync_parser.set_defaults(handler=command_rsync)

    smoke_parser = subparsers.add_parser("smoke", help="Run a remote sanity check")
    smoke_parser.add_argument("--machine", help="Machine name from the env file")
    smoke_parser.add_argument("--group", help="Run smoke checks across a group")
    smoke_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    smoke_parser.add_argument("--dry-run", action="store_true")
    smoke_parser.set_defaults(handler=command_smoke)

    gpu_status_parser = subparsers.add_parser(
        "gpu-status", help="Inspect GPU occupancy, free VRAM, and lease state"
    )
    gpu_status_parser.add_argument("--machine", help="Machine name from the env file")
    gpu_status_parser.add_argument("--group", help="Run across a configured group")
    gpu_status_parser.add_argument(
        "--require-idle",
        action="store_true",
        help="Only show GPUs without active compute jobs",
    )
    gpu_status_parser.add_argument(
        "--min-free-vram-gib",
        type=float,
        help="Only show GPUs with at least this much free VRAM",
    )
    gpu_status_parser.add_argument(
        "--ignore-leases",
        action="store_true",
        help="Ignore active remote leases in policy evaluation",
    )
    gpu_status_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all GPUs even when policy filters are active",
    )
    gpu_status_parser.add_argument(
        "--json", action="store_true", help="Emit machine/GPU status as JSON"
    )
    gpu_status_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    gpu_status_parser.add_argument("--dry-run", action="store_true")
    gpu_status_parser.set_defaults(handler=command_gpu_status)

    lease_parser = subparsers.add_parser(
        "lease", help="Acquire, inspect, or release a polite remote GPU lease"
    )
    lease_parser.add_argument("action", choices=["show", "acquire", "release"])
    lease_parser.add_argument("--machine", help="Machine name from the env file")
    lease_parser.add_argument("--group", help="Operate across a configured group")
    lease_parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index")
    lease_parser.add_argument(
        "--owner", help="Lease owner label; defaults to local USER"
    )
    lease_parser.add_argument("--note", help="Optional free-form note")
    lease_parser.add_argument(
        "--ttl-hours", type=float, default=8.0, help="Lease duration for acquire"
    )
    lease_parser.add_argument(
        "--force", action="store_true", help="Replace an active lease"
    )
    lease_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    lease_parser.add_argument("--dry-run", action="store_true")
    lease_parser.set_defaults(handler=command_lease)

    setup_parser = subparsers.add_parser(
        "setup", help="Create workdirs, sync the repo, and build remote virtualenvs"
    )
    setup_parser.add_argument("--machine", help="Machine name from the env file")
    setup_parser.add_argument("--group", help="Operate across a configured group")
    setup_parser.add_argument(
        "--python",
        default="python3",
        help="Remote Python executable to use for venv creation",
    )
    setup_parser.add_argument(
        "--source", help="Local rsync upload source; defaults to the repo root"
    )
    setup_parser.add_argument(
        "--dest", help="Remote destination override; defaults to WORKDIR"
    )
    setup_parser.add_argument(
        "--delete", action="store_true", help="Pass --delete to the upload rsync"
    )
    setup_parser.add_argument(
        "--skip-sync", action="store_true", help="Skip the repo upload step"
    )
    setup_parser.add_argument(
        "--skip-install", action="store_true", help="Skip virtualenv/pip installation"
    )
    setup_parser.add_argument(
        "--check-path",
        action="append",
        default=[],
        help="Remote path, relative to WORKDIR, that must exist after setup",
    )
    setup_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    setup_parser.add_argument("--dry-run", action="store_true")
    setup_parser.set_defaults(handler=command_setup)

    collect_parser = subparsers.add_parser(
        "collect", help="Pull one sweep's result artifacts into the local fleet cache"
    )
    collect_parser.add_argument("--machine", help="Machine name from the env file")
    collect_parser.add_argument("--group", help="Operate across a configured group")
    collect_parser.add_argument(
        "--sweep", required=True, help="Sweep name, for example slinoss-uea-grid"
    )
    collect_parser.add_argument(
        "--include-logs", action="store_true", help="Also download training.log files"
    )
    collect_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    collect_parser.add_argument("--dry-run", action="store_true")
    collect_parser.set_defaults(handler=command_collect)

    sweep_run_parser = subparsers.add_parser(
        "sweep-run", help="Launch a detached remote sweep worker on one machine/GPU"
    )
    sweep_run_parser.add_argument(
        "--machine", required=True, help="Machine name from the env file"
    )
    sweep_run_parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU index to expose via CUDA_VISIBLE_DEVICES",
    )
    sweep_run_parser.add_argument(
        "--config",
        required=True,
        help="Sweep config path relative to the repo root on the remote machine",
    )
    sweep_run_parser.add_argument(
        "--resource-tier", help="Filter the remote run to one resource tier"
    )
    sweep_run_parser.add_argument("--shard", help="Shard spec like 1/18")
    sweep_run_parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Restrict the remote run to one or more datasets",
    )
    sweep_run_parser.add_argument("--max-groups", type=int)
    sweep_run_parser.add_argument("--max-trials", type=int)
    sweep_run_parser.add_argument("--retry-failed", action="store_true")
    sweep_run_parser.add_argument("--force", action="store_true")
    sweep_run_parser.add_argument(
        "--force-lease",
        action="store_true",
        help="Replace any active polite lease on the selected GPU",
    )
    sweep_run_parser.add_argument(
        "--require-idle",
        action="store_true",
        help="Refuse to launch if the selected GPU has active compute jobs",
    )
    sweep_run_parser.add_argument(
        "--min-free-vram-gib",
        type=float,
        help="Refuse to launch if the selected GPU has less free VRAM",
    )
    sweep_run_parser.add_argument(
        "--ignore-leases",
        action="store_true",
        help="Ignore active polite leases during readiness checks",
    )
    sweep_run_parser.add_argument(
        "--owner", help="Lease owner label; defaults to local USER"
    )
    sweep_run_parser.add_argument(
        "--note", help="Optional free-form note for the local ledger"
    )
    sweep_run_parser.add_argument(
        "--ttl-hours",
        type=float,
        default=24.0,
        help="Lease duration written alongside the launched job",
    )
    sweep_run_parser.add_argument(
        "--log-subdir",
        default="remote-sweeps",
        help="Remote log subdirectory under log/",
    )
    sweep_run_parser.add_argument("--dry-run", action="store_true")
    sweep_run_parser.set_defaults(handler=command_sweep_run)

    sweep_status_parser = subparsers.add_parser(
        "sweep-status",
        help="Summarize collected sweep results against the deterministic plan",
    )
    sweep_status_parser.add_argument(
        "--config", required=True, help="Sweep config path"
    )
    sweep_status_parser.add_argument(
        "--json", action="store_true", help="Emit the status summary as JSON"
    )
    sweep_status_parser.add_argument(
        "--list",
        choices=["pending", "success", "failed", "all"],
        help="List specific trial rows instead of the summary",
    )
    sweep_status_parser.set_defaults(handler=command_sweep_status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = getattr(args, "command", None)
    if isinstance(command, list) and command and command[0] == "--":
        args.command = command[1:]
    try:
        return int(args.handler(args))
    except RemoteConfigError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "error: interrupted\n")


if __name__ == "__main__":
    raise SystemExit(main())
