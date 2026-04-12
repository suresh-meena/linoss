from __future__ import annotations

import argparse
import concurrent.futures
import csv
import errno
import io
import json
import os
import shlex
import shutil
import subprocess
import textwrap
import time
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
# Keep the control socket path short enough for Unix domain socket limits.
DEFAULT_SSH_CONTROL_DIR = Path("/tmp/linoss-ssh-control")
DEFAULT_FANOUT_PARALLELISM = 4
DEFAULT_SSH_CONNECT_TIMEOUT_SEC = 10
DEFAULT_SSH_SERVER_ALIVE_INTERVAL_SEC = 15
DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX = 3
DEFAULT_PROBE_TIMEOUT_SEC = 15.0
DEFAULT_MIN_FREE_DISK_GIB = 5.0
DEFAULT_LEASE_LOCK_WAIT_SEC = 5.0
DEFAULT_LEASE_LOCK_STALE_SEC = 300.0
DEFAULT_STATE_LOCK_WAIT_SEC = 10.0
DEFAULT_STATE_LOCK_STALE_SEC = 300.0
VALID_TOOLCHAIN_MODES = {"auto", "system", "guix"}
VALID_STATE_LOCK_MODES = {"auto", "mkdir", "flock"}


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


def _parse_env_float(
    env: dict[str, str], key: str, *, default: float, minimum: float = 0.0
) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RemoteConfigError(f"invalid float for {key}: {raw!r}") from exc
    if value < minimum:
        raise RemoteConfigError(f"{key} must be >= {minimum:g}, got {value:g}")
    return value


def _parse_env_bool(env: dict[str, str], key: str, *, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RemoteConfigError(f"invalid boolean for {key}: {raw!r}")


def probe_timeout_seconds(env: dict[str, str]) -> float:
    return _parse_env_float(
        env,
        "KD_REMOTE_PROBE_TIMEOUT_SEC",
        default=DEFAULT_PROBE_TIMEOUT_SEC,
        minimum=1.0,
    )


def min_free_disk_gib(env: dict[str, str]) -> float:
    return _parse_env_float(
        env,
        "KD_REMOTE_MIN_FREE_DISK_GIB",
        default=DEFAULT_MIN_FREE_DISK_GIB,
        minimum=0.0,
    )


def state_lock_wait_seconds(env: dict[str, str]) -> float:
    return _parse_env_float(
        env,
        "KD_REMOTE_STATE_LOCK_WAIT_SEC",
        default=DEFAULT_STATE_LOCK_WAIT_SEC,
        minimum=0.1,
    )


def state_lock_stale_seconds(env: dict[str, str]) -> float:
    return _parse_env_float(
        env,
        "KD_REMOTE_STATE_LOCK_STALE_SEC",
        default=DEFAULT_STATE_LOCK_STALE_SEC,
        minimum=1.0,
    )


def state_lock_mode(env: dict[str, str]) -> str:
    mode = env.get("KD_REMOTE_STATE_LOCK_MODE", "auto").strip().lower()
    if mode not in VALID_STATE_LOCK_MODES:
        allowed = ", ".join(sorted(VALID_STATE_LOCK_MODES))
        raise RemoteConfigError(
            f"invalid KD_REMOTE_STATE_LOCK_MODE={mode!r}; expected one of: {allowed}"
        )
    return mode


def launcher_enable_pytorch_allocator(env: dict[str, str]) -> bool:
    return _parse_env_bool(
        env,
        "KD_REMOTE_ENABLE_PYTORCH_ALLOC_CONF",
        default=True,
    )


def launcher_cleanup_shm(env: dict[str, str]) -> bool:
    return _parse_env_bool(
        env,
        "KD_REMOTE_CLEAN_DEV_SHM",
        default=False,
    )


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
    default_lease_root = (
        f"{workdir.rstrip('/')}/.remote-jobs/leases"
        if workdir is not None
        else DEFAULT_LEASE_ROOT
    )
    lease_root = (
        machine_field(env, name, "LEASE_ROOT")
        or env.get("KD_REMOTE_LEASE_ROOT")
        or default_lease_root
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


def command_prefix(*tools: str, allow_missing: bool = False) -> list[str]:
    mode = toolchain_mode()
    if mode == "system":
        if allow_missing or _have_system_tools(*tools):
            return []
        raise _toolchain_error(*tools)
    if mode == "guix":
        if _have_guix_toolchain() or allow_missing:
            return [str(GUIX_RUN)]
        raise RemoteConfigError(
            f"KD_REMOTE_TOOLCHAIN=guix requires `guix` on PATH and {GUIX_RUN}"
        )

    if _have_system_tools(*tools):
        return []
    if _have_guix_toolchain():
        return [str(GUIX_RUN)]
    if allow_missing:
        return []
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
    allow_missing_tools: bool = False,
) -> list[str]:
    DEFAULT_SSH_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    required_tools = ["ssh"]
    if uses_sshpass(machine):
        required_tools.append("sshpass")
    command: list[str] = command_prefix(
        *required_tools,
        allow_missing=allow_missing_tools,
    )
    if uses_sshpass(machine):
        command += ["sshpass", "-e"]
    command.append("ssh")
    if allocate_tty:
        command.append("-t")
    command += [
        "-F",
        "/dev/null",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "VisualHostKey=no",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_FILE}",
        "-o",
        f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_SEC}",
        "-o",
        f"ServerAliveInterval={DEFAULT_SSH_SERVER_ALIVE_INTERVAL_SEC}",
        "-o",
        f"ServerAliveCountMax={DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX}",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={DEFAULT_SSH_CONTROL_DIR / 'ssh-%C.sock'}",
        "-o",
        "ControlPersist=5m",
        "-p",
        str(machine.port),
    ]
    command += ssh_auth_options(machine)
    command.append(machine.target)
    if remote_command is not None:
        command.append(f"bash -lc {shlex.quote(remote_command)}")
    return command


def rsync_ssh_transport(
    machine: RemoteMachine,
    *,
    allow_missing_tools: bool = False,
) -> str:
    DEFAULT_SSH_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    required_tools = ["ssh"]
    if uses_sshpass(machine):
        required_tools.append("sshpass")
    parts += command_prefix(*required_tools, allow_missing=allow_missing_tools)
    if uses_sshpass(machine):
        parts += ["sshpass", "-e"]
    parts += [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "VisualHostKey=no",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_FILE}",
        "-o",
        f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_SEC}",
        "-o",
        f"ServerAliveInterval={DEFAULT_SSH_SERVER_ALIVE_INTERVAL_SEC}",
        "-o",
        f"ServerAliveCountMax={DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX}",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={DEFAULT_SSH_CONTROL_DIR / 'ssh-%C.sock'}",
        "-o",
        "ControlPersist=5m",
        "-p",
        str(machine.port),
    ]
    parts += ssh_auth_options(machine)
    return quoted_command(parts)


def render_command(command: list[str], *, machine: RemoteMachine) -> str:
    prefix = "SSHPASS=<redacted> " if uses_sshpass(machine) else ""
    return prefix + quoted_command(command)


def run_command(
    command: list[str],
    *,
    machine: RemoteMachine,
    dry_run: bool,
    timeout_seconds: float | None = None,
) -> int:
    if dry_run:
        print(render_command(command, machine=machine))
        return 0
    try:
        completed = subprocess.run(
            command,
            env=prefixed_env(machine),
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return 124
    return completed.returncode


def run_command_capture(
    command: list[str],
    *,
    machine: RemoteMachine,
    dry_run: bool,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=render_command(command, machine=machine) + "\n",
            stderr="",
        )
    try:
        return subprocess.run(
            command,
            env=prefixed_env(machine),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "").rstrip()
        timeout_text = (
            f"command timed out after {timeout_seconds:g}s"
            if timeout_seconds is not None
            else "command timed out"
        )
        if stderr:
            stderr = f"{stderr}\n{timeout_text}"
        else:
            stderr = timeout_text
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
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


def load_json_safe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except json.JSONDecodeError as exc:
        return None, f"{path}: invalid json ({exc})"


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


def _load_json_from_text(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if not stripped:
        return {"machines": {}}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RemoteConfigError(f"invalid fleet-state.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RemoteConfigError("fleet-state.json must contain a JSON object")
    return payload


def _fleet_state_path(env: dict[str, str]) -> Path:
    return state_dir(env) / "fleet-state.json"


def _fleet_state_lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _acquire_mkdir_lock(path: Path, *, wait_seconds: float, stale_seconds: float) -> None:
    lock_path = _fleet_state_lock_path(path)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            os.mkdir(lock_path)
            return
        except FileExistsError:
            stale = 0.0
            try:
                stale = time.time() - lock_path.stat().st_mtime
            except OSError:
                stale = 0.0
            if stale > stale_seconds:
                try:
                    os.rmdir(lock_path)
                    continue
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise RemoteConfigError(
                    f"timed out waiting for state lock {lock_path} after {wait_seconds:g}s"
                )
            time.sleep(0.05)


def _release_mkdir_lock(path: Path) -> None:
    lock_path = _fleet_state_lock_path(path)
    try:
        os.rmdir(lock_path)
    except OSError:
        pass


def _flock_supported(exc: OSError) -> bool:
    return exc.errno not in {
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
        errno.EINVAL,
    }


def _load_state_cache_locked(handle) -> dict[str, Any]:
    handle.seek(0)
    payload = _load_json_from_text(handle.read())
    payload.setdefault("machines", {})
    return payload


def _write_state_cache_locked(handle, payload: dict[str, Any]) -> None:
    payload.setdefault("machines", {})
    handle.seek(0)
    handle.truncate()
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def load_state_cache(env: dict[str, str]) -> dict[str, Any]:
    path = _fleet_state_path(env)
    ensure_parent(path)
    mode = state_lock_mode(env)
    if mode == "mkdir":
        _acquire_mkdir_lock(
            path,
            wait_seconds=state_lock_wait_seconds(env),
            stale_seconds=state_lock_stale_seconds(env),
        )
        try:
            with path.open("a+", encoding="utf-8") as handle:
                return _load_state_cache_locked(handle)
        finally:
            _release_mkdir_lock(path)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return _load_state_cache_locked(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            if mode == "flock" or _flock_supported(exc):
                raise
    _acquire_mkdir_lock(
        path,
        wait_seconds=state_lock_wait_seconds(env),
        stale_seconds=state_lock_stale_seconds(env),
    )
    try:
        with path.open("a+", encoding="utf-8") as handle:
            return _load_state_cache_locked(handle)
    finally:
        _release_mkdir_lock(path)


def save_state_cache(env: dict[str, str], payload: dict[str, Any]) -> None:
    path = _fleet_state_path(env)
    ensure_parent(path)
    mode = state_lock_mode(env)
    if mode == "mkdir":
        _acquire_mkdir_lock(
            path,
            wait_seconds=state_lock_wait_seconds(env),
            stale_seconds=state_lock_stale_seconds(env),
        )
        try:
            with path.open("a+", encoding="utf-8") as handle:
                _write_state_cache_locked(handle, payload)
            return
        finally:
            _release_mkdir_lock(path)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                _write_state_cache_locked(handle, payload)
                return
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            if mode == "flock" or _flock_supported(exc):
                raise
    _acquire_mkdir_lock(
        path,
        wait_seconds=state_lock_wait_seconds(env),
        stale_seconds=state_lock_stale_seconds(env),
    )
    try:
        with path.open("a+", encoding="utf-8") as handle:
            _write_state_cache_locked(handle, payload)
    finally:
        _release_mkdir_lock(path)


def update_machine_state(
    env: dict[str, str], machine: RemoteMachine, patch: dict[str, Any]
) -> None:
    path = _fleet_state_path(env)
    ensure_parent(path)
    mode = state_lock_mode(env)

    def _write_with_handle(handle) -> None:
        payload = _load_state_cache_locked(handle)
        machines = payload.setdefault("machines", {})
        current = dict(machines.get(machine.name, {}))
        current.update(patch)
        current["host"] = machine.host
        current["updated_at"] = iso_now()
        machines[machine.name] = current
        _write_state_cache_locked(handle, payload)

    if mode == "mkdir":
        _acquire_mkdir_lock(
            path,
            wait_seconds=state_lock_wait_seconds(env),
            stale_seconds=state_lock_stale_seconds(env),
        )
        try:
            with path.open("a+", encoding="utf-8") as handle:
                _write_with_handle(handle)
            return
        finally:
            _release_mkdir_lock(path)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                _write_with_handle(handle)
                return
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            if mode == "flock" or _flock_supported(exc):
                raise
    _acquire_mkdir_lock(
        path,
        wait_seconds=state_lock_wait_seconds(env),
        stale_seconds=state_lock_stale_seconds(env),
    )
    try:
        with path.open("a+", encoding="utf-8") as handle:
            _write_with_handle(handle)
    finally:
        _release_mkdir_lock(path)


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
    timeout_seconds: float,
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
        import shutil
        import subprocess
        import sys

        payload = json.loads({json.dumps(payload, sort_keys=True)!r})

        def run(args):
            if shutil.which("timeout") is not None and args and args[0] == "nvidia-smi":
                args = ["timeout", "5", *args]
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
        workdir = payload.get("workdir")
        disk_info = {{"path": workdir, "ok": False, "free_kib": None, "free_gib": None}}
        if workdir:
            df_query = run(["df", "-k", str(workdir)])
            if df_query.returncode == 0 and df_query.stdout.strip():
                lines = [line for line in df_query.stdout.splitlines() if line.strip()]
                if len(lines) >= 2:
                    columns = lines[-1].split()
                    if len(columns) >= 4:
                        try:
                            free_kib = int(columns[3])
                            disk_info = {{
                                "path": workdir,
                                "ok": True,
                                "free_kib": free_kib,
                                "free_gib": round(free_kib / (1024.0 * 1024.0), 3),
                            }}
                        except ValueError:
                            disk_info["error"] = "failed to parse df output"
                    else:
                        disk_info["error"] = "unexpected df output shape"
                else:
                    disk_info["error"] = "df output missing data rows"
            else:
                disk_info["error"] = df_query.stderr.strip() or df_query.stdout.strip() or "df failed"
        payload["disk"] = disk_info
        print(json.dumps(payload, sort_keys=True))
        """
    ).strip()
    remote_command = build_remote_script(
        f"python3 - <<'PY'\n{script}\nPY",
        workdir=machine.workdir,
        require_workdir=False,
    )
    completed = run_command_capture(
        ssh_command(
            machine,
            allocate_tty=False,
            remote_command=remote_command,
            allow_missing_tools=dry_run,
        ),
        machine=machine,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
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
    machine_disk: dict[str, Any] | None,
    min_free_disk_gib: float,
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
    if machine_disk and machine_disk.get("ok"):
        try:
            free_gib = float(machine_disk.get("free_gib", 0.0))
        except (TypeError, ValueError):
            free_gib = 0.0
        if free_gib + 1e-9 < min_free_disk_gib:
            reasons.append(f"disk_free<{min_free_disk_gib:g}GiB")
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
        work=lambda machine: _probe_gpu_status_machine(
            machine,
            dry_run=dry_run,
            timeout_seconds=probe_timeout_seconds(env),
        ),
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


def _extract_json_payload(result: CommandResult, *, context: str) -> dict[str, Any]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RemoteConfigError(f"no payload returned for {context} on {result.machine.name}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RemoteConfigError(
            f"failed to parse payload for {context} on {result.machine.name}: {lines[-1]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RemoteConfigError(
            f"unexpected payload type for {context} on {result.machine.name}: {type(payload).__name__}"
        )
    return payload


def _known_launch_job_ids(ledger_path: Path, *, sweep_name: str) -> set[str]:
    if not ledger_path.is_file():
        return set()
    known: set[str] = set()
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("sweep") != sweep_name:
                continue
            if payload.get("event") not in {"launch", "launch_reconciled"}:
                continue
            job_id = payload.get("job_id")
            if isinstance(job_id, str) and job_id:
                known.add(job_id)
    return known


def _query_remote_done_jobs(
    machine: RemoteMachine, *, sweep_name: str, dry_run: bool, timeout_seconds: float
) -> CommandResult:
    if machine.workdir is None:
        return CommandResult(
            machine=machine,
            returncode=2,
            stderr="query-remote-done-jobs requires WORKDIR",
        )
    payload = {"sweep_name": sweep_name}
    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path

        payload = json.loads({json.dumps(payload, sort_keys=True)!r})
        jobs_dir = Path(".remote-jobs") / payload["sweep_name"]
        jobs = []
        if jobs_dir.is_dir():
            for done_path in sorted(jobs_dir.glob("*.done.json")):
                item = {{
                    "done_path": str(done_path),
                    "job_id": done_path.stem.replace(".done", ""),
                }}
                try:
                    parsed = json.loads(done_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        item.update(parsed)
                    else:
                        item["invalid"] = True
                        item["error"] = "payload is not a json object"
                except json.JSONDecodeError:
                    item["invalid"] = True
                    item["error"] = "invalid json"
                jobs.append(item)
        print(json.dumps({{"ok": True, "jobs": jobs}}, sort_keys=True))
        """
    ).strip()
    remote_command = build_remote_script(
        f"python3 - <<'PY'\n{script}\nPY",
        workdir=machine.workdir,
        require_workdir=True,
    )
    completed = run_command_capture(
        ssh_command(
            machine,
            allocate_tty=False,
            remote_command=remote_command,
            allow_missing_tools=dry_run,
        ),
        machine=machine,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
    )
    return CommandResult(
        machine=machine,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _audit_remote_jobs(
    machine: RemoteMachine,
    *,
    sweep_name: str,
    kill_orphans: bool,
    dry_run: bool,
    timeout_seconds: float,
) -> CommandResult:
    if machine.workdir is None:
        return CommandResult(
            machine=machine,
            returncode=2,
            stderr="remote-audit requires WORKDIR",
        )
    payload = {
        "sweep_name": sweep_name,
        "workdir": machine.workdir,
        "kill_orphans": kill_orphans,
    }
    script = textwrap.dedent(
        f"""
        import json
        import os
        import signal
        from pathlib import Path

        payload = json.loads({json.dumps(payload, sort_keys=True)!r})
        sweep_name = payload["sweep_name"]
        workdir = Path(payload["workdir"]).resolve()
        jobs_dir = workdir / ".remote-jobs" / sweep_name
        kill_orphans = bool(payload.get("kill_orphans", False))

        def pid_alive(pid, launcher_path=None):
            try:
                os.kill(pid, 0)
                cmdline = Path(f"/proc/{{pid}}/cmdline").read_bytes().replace(b"\\x00", b" ").lower()
            except OSError:
                return False
            except FileNotFoundError:
                return False
            if launcher_path:
                launcher_marker = str(launcher_path).encode("utf-8", "ignore").lower()
                if launcher_marker and launcher_marker in cmdline:
                    return True
            if b"python" in cmdline and b"sweep" in cmdline:
                return True
            return False

        launches = []
        active_launch_pids = set()
        stale_leases_cleared = 0
        if jobs_dir.is_dir():
            for done_path in sorted(jobs_dir.glob("*.done.json")):
                row = {{"done_path": str(done_path), "job_id": done_path.stem.replace(".done", "")}}
                try:
                    parsed = json.loads(done_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    row["invalid"] = True
                    row["error"] = "invalid json"
                    launches.append(row)
                    continue
                if isinstance(parsed, dict):
                    row.update(parsed)
                else:
                    row["invalid"] = True
                    row["error"] = "payload is not a json object"
                    launches.append(row)
                    continue
                pid_value = row.get("pid")
                pid = None
                if isinstance(pid_value, int):
                    pid = pid_value
                elif isinstance(pid_value, str) and pid_value.isdigit():
                    pid = int(pid_value)
                alive = bool(pid and pid > 0 and pid_alive(pid, row.get("launcher_path")))
                row["pid_alive"] = alive
                if alive and pid is not None:
                    active_launch_pids.add(pid)
                if (not alive) and isinstance(row.get("lease_path"), str):
                    lease_path = Path(row["lease_path"])
                    if lease_path.exists():
                        try:
                            lease_path.unlink()
                            stale_leases_cleared += 1
                        except OSError:
                            pass
                launches.append(row)

        dead_launches = [
            {{
                "job_id": row.get("job_id"),
                "pid": row.get("pid"),
                "done_path": row.get("done_path"),
                "log_path": row.get("log_path"),
                "lease_path": row.get("lease_path"),
            }}
            for row in launches
            if not row.get("pid_alive", False)
        ]

        killed_orphans = []
        if kill_orphans:
            for proc_dir in Path("/proc").iterdir():
                if not proc_dir.name.isdigit():
                    continue
                pid = int(proc_dir.name)
                if pid in active_launch_pids:
                    continue
                status_path = proc_dir / "status"
                try:
                    status_text = status_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                ppid = None
                for line in status_text.splitlines():
                    if line.startswith("PPid:"):
                        try:
                            ppid = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            ppid = None
                        break
                if ppid != 1:
                    continue
                cwd_path = proc_dir / "cwd"
                cmdline_path = proc_dir / "cmdline"
                try:
                    cwd = Path(os.readlink(cwd_path)).resolve()
                    cmdline_raw = cmdline_path.read_bytes().replace(b"\\x00", b" ").decode("utf-8", errors="replace")
                except OSError:
                    continue
                if not str(cwd).startswith(str(workdir)):
                    continue
                cmdline_lower = cmdline_raw.lower()
                if ("python" not in cmdline_lower) and ("sweep" not in cmdline_lower):
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed_orphans.append({{"pid": pid, "cwd": str(cwd), "cmdline": cmdline_raw}})
                except OSError:
                    continue

        print(
            json.dumps(
                {{
                    "ok": True,
                    "sweep": sweep_name,
                    "jobs_total": len(launches),
                    "active_launches": sum(1 for row in launches if row.get("pid_alive", False)),
                    "dead_launches": dead_launches,
                    "stale_leases_cleared": stale_leases_cleared,
                    "killed_orphans": killed_orphans,
                }},
                sort_keys=True,
            )
        )
        """
    ).strip()
    remote_command = build_remote_script(
        f"python3 - <<'PY'\n{script}\nPY",
        workdir=machine.workdir,
        require_workdir=True,
    )
    completed = run_command_capture(
        ssh_command(
            machine,
            allocate_tty=False,
            remote_command=remote_command,
            allow_missing_tools=dry_run,
        ),
        machine=machine,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
    )
    return CommandResult(
        machine=machine,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


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
        import os
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        payload = json.loads({json.dumps(script_payload, sort_keys=True)!r})
        lease_path = Path(payload["lease_path"])
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(lease_path) + ".lock")
        deadline = time.monotonic() + {DEFAULT_LEASE_LOCK_WAIT_SEC}
        action = payload["action"]

        while True:
            try:
                os.mkdir(lock_path)
                break
            except FileExistsError:
                stale_seconds = 0.0
                try:
                    stale_seconds = time.time() - lock_path.stat().st_mtime
                except OSError:
                    stale_seconds = 0.0
                if stale_seconds > {DEFAULT_LEASE_LOCK_STALE_SEC}:
                    try:
                        os.rmdir(lock_path)
                        continue
                    except OSError:
                        pass
                if time.monotonic() >= deadline:
                    print(json.dumps({{"acquired": False, "error": "lease lock timeout", "lease_path": str(lease_path)}}, sort_keys=True))
                    raise SystemExit(4)
                time.sleep(0.05)

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

        try:
            current = None
            if lease_path.is_file():
                try:
                    current = json.loads(lease_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    current = {{"invalid": True}}

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
        finally:
            try:
                os.rmdir(lock_path)
            except OSError:
                pass
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
        ssh_command(
            machine,
            allocate_tty=allocate_tty,
            remote_command=remote_command,
            allow_missing_tools=args.dry_run,
        ),
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
            allow_missing_tools=dry_run,
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
    allow_missing_tools: bool,
    extra_args: list[str] | None = None,
) -> list[str]:
    command: list[str] = command_prefix("rsync", allow_missing=allow_missing_tools) + [
        "rsync",
        "-az",
        "--compress-level=1",
    ]
    if delete:
        command.append("--delete")
    if extra_args:
        command += extra_args
    if direction == "upload":
        for pattern in upload_excludes():
            command += ["--exclude", pattern]
    command += [
        "-e",
        rsync_ssh_transport(machine, allow_missing_tools=allow_missing_tools),
    ]
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
            allow_missing_tools=args.dry_run,
        )
    else:
        command = _build_rsync_command(
            machine,
            direction="download",
            source=source,
            dest=dest,
            delete=args.delete,
            allow_missing_tools=args.dry_run,
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


def _smoke_one(
    machine: RemoteMachine, *, dry_run: bool, timeout_seconds: float
) -> CommandResult:
    smoke = textwrap.dedent(
        """
        echo "host=$(hostname)"
        echo "pwd=$(pwd)"
        echo "user=$(whoami)"
        if command -v python3 >/dev/null 2>&1; then python3 --version; fi
        if command -v nvidia-smi >/dev/null 2>&1; then
          if command -v timeout >/dev/null 2>&1; then
            timeout 5 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
          else
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
          fi
        fi
        """
    ).strip()
    remote_command = build_remote_script(
        smoke,
        workdir=machine.workdir,
        require_workdir=machine.workdir is not None,
    )
    completed = run_command_capture(
        ssh_command(
            machine,
            allocate_tty=False,
            remote_command=remote_command,
            allow_missing_tools=dry_run,
        ),
        machine=machine,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
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
        work=lambda machine: _smoke_one(
            machine,
            dry_run=args.dry_run,
            timeout_seconds=probe_timeout_seconds(env),
        ),
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
            machine_disk=payload.get("disk"),
            min_free_disk_gib=min_free_disk_gib(env),
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
                machine_disk=payload.get("disk"),
                min_free_disk_gib=min_free_disk_gib(env),
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
                "disk": payload.get("disk"),
            }
            if (
                (
                    args.require_idle
                    or args.min_free_vram_gib is not None
                    or not args.ignore_leases
                    or min_free_disk_gib(env) > 0.0
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
                machine,
                allocate_tty=allocate_tty,
                remote_command=remote_command,
                allow_missing_tools=args.dry_run,
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
            ssh_command(
                machine,
                allocate_tty=False,
                remote_command=remote_command,
                allow_missing_tools=args.dry_run,
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
                f'SETUP_PYTHON={shlex.quote(args.python)}',
                'if ! command -v "$SETUP_PYTHON" >/dev/null 2>&1; then',
                '  MINIFORGE_ROOT="$HOME/miniforge3"',
                '  MINIFORGE_PY="$MINIFORGE_ROOT/bin/python"',
                '  if [ ! -x "$MINIFORGE_PY" ]; then',
                '    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"',
                '    MINIFORGE_INSTALLER="/tmp/miniforge-installer.sh"',
                '    if command -v curl >/dev/null 2>&1; then',
                '      curl -fsSL "$MINIFORGE_URL" -o "$MINIFORGE_INSTALLER"',
                '    elif command -v wget >/dev/null 2>&1; then',
                '      wget -qO "$MINIFORGE_INSTALLER" "$MINIFORGE_URL"',
                "    else",
                '      echo "missing requested python and no curl/wget available to bootstrap Miniforge" >&2',
                "      exit 1",
                "    fi",
                '    bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_ROOT"',
                "  fi",
                '  SETUP_PYTHON="$MINIFORGE_PY"',
                "fi",
                'if [ ! -x .venv/bin/python ]; then',
                "  rm -rf .venv",
                '  "$SETUP_PYTHON" -m venv .venv',
                "  .venv/bin/python -m pip install -U pip",
                "fi",
                ".venv/bin/python -m pip install -r requirements.txt",
            ]
        install_lines += [
            "if [ -d .remote-jobs ]; then find .remote-jobs -type f -mtime +7 -delete; fi",
            ".venv/bin/python --version",
            "if command -v nvidia-smi >/dev/null 2>&1; then "
            "if command -v timeout >/dev/null 2>&1; then "
            "timeout 5 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; "
            "else "
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; "
            "fi; "
            "else echo 'nvidia-smi missing' >&2; exit 1; fi",
        ]
        remote_command = build_remote_script(
            *install_lines,
            *check_lines,
            workdir=machine.workdir,
            require_workdir=True,
        )
        completed = run_command_capture(
            ssh_command(
                machine,
                allocate_tty=False,
                remote_command=remote_command,
                allow_missing_tools=args.dry_run,
            ),
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
    known_job_ids = _known_launch_job_ids(paths.ledger_path, sweep_name=sweep_name)

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
            allow_missing_tools=args.dry_run,
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
            reconcile_result = _query_remote_done_jobs(
                result.machine,
                sweep_name=sweep_name,
                dry_run=False,
                timeout_seconds=probe_timeout_seconds(env),
            )
            reconciled = 0
            if reconcile_result.returncode == 0:
                try:
                    payload = _extract_json_payload(
                        reconcile_result,
                        context="remote done job query",
                    )
                except RemoteConfigError:
                    payload = {}
                for row in payload.get("jobs", []):
                    if not isinstance(row, dict):
                        continue
                    job_id = row.get("job_id")
                    if not isinstance(job_id, str) or not job_id:
                        continue
                    if job_id in known_job_ids:
                        continue
                    reconciled_entry = {
                        "event": "launch_reconciled",
                        "timestamp": iso_now(),
                        "machine": result.machine.name,
                        "host": result.machine.host,
                        "sweep": sweep_name,
                        "job_id": job_id,
                        "pid": row.get("pid"),
                        "remote_log_path": row.get("log_path"),
                        "launcher_path": row.get("launcher_path"),
                        "lease_path": row.get("lease_path"),
                        "source_done_path": row.get("done_path"),
                    }
                    append_jsonl(paths.ledger_path, reconciled_entry)
                    known_job_ids.add(job_id)
                    reconciled += 1
            update_machine_state(
                env,
                result.machine,
                {
                    "last_collect_at": iso_now(),
                    "last_collected_sweep": sweep_name,
                    "last_reconciled_launches": reconciled,
                },
            )
    return print_command_results(results)


def command_audit(args: argparse.Namespace) -> int:
    env = load_remote_env(args.env_file)
    machines = resolve_selected_machines(
        env,
        machine=args.machine,
        group=args.group,
        allow_default_single=True,
    )
    paths = sweep_paths(env, args.sweep)

    def work(machine: RemoteMachine) -> CommandResult:
        return _audit_remote_jobs(
            machine,
            sweep_name=args.sweep,
            kill_orphans=args.kill_orphans,
            dry_run=args.dry_run,
            timeout_seconds=probe_timeout_seconds(env),
        )

    results = fanout(machines, max_parallel=args.max_parallel, work=work)
    if args.dry_run:
        return print_command_results(results)

    output_rows: list[dict[str, Any]] = []
    errors: list[CommandResult] = []
    for result in results:
        if result.returncode != 0:
            errors.append(result)
            continue
        try:
            payload = _extract_json_payload(result, context="remote audit")
        except RemoteConfigError as exc:
            errors.append(
                CommandResult(
                    machine=result.machine,
                    returncode=1,
                    stdout=result.stdout,
                    stderr=str(exc),
                )
            )
            continue
        if not payload.get("ok", False):
            errors.append(
                CommandResult(
                    machine=result.machine,
                    returncode=1,
                    stdout=result.stdout,
                    stderr=str(payload.get("error", "")),
                )
            )
            continue
        dead_launches = payload.get("dead_launches", [])
        killed_orphans = payload.get("killed_orphans", [])
        row = {
            "machine": result.machine.name,
            "host": result.machine.host,
            "sweep": args.sweep,
            "jobs_total": payload.get("jobs_total", 0),
            "active_launches": payload.get("active_launches", 0),
            "dead_launches": dead_launches,
            "dead_launch_count": len(dead_launches)
            if isinstance(dead_launches, list)
            else 0,
            "stale_leases_cleared": payload.get("stale_leases_cleared", 0),
            "killed_orphans": killed_orphans,
            "killed_orphans_count": len(killed_orphans)
            if isinstance(killed_orphans, list)
            else 0,
        }
        output_rows.append(row)

        if isinstance(dead_launches, list):
            for launch in dead_launches:
                if not isinstance(launch, dict):
                    continue
                append_jsonl(
                    paths.ledger_path,
                    {
                        "event": "audit_dead_launch",
                        "timestamp": iso_now(),
                        "machine": result.machine.name,
                        "host": result.machine.host,
                        "sweep": args.sweep,
                        "job_id": launch.get("job_id"),
                        "pid": launch.get("pid"),
                        "remote_log_path": launch.get("log_path"),
                        "lease_path": launch.get("lease_path"),
                        "source_done_path": launch.get("done_path"),
                    },
                )
        if isinstance(killed_orphans, list):
            for orphan in killed_orphans:
                if not isinstance(orphan, dict):
                    continue
                append_jsonl(
                    paths.ledger_path,
                    {
                        "event": "audit_killed_orphan",
                        "timestamp": iso_now(),
                        "machine": result.machine.name,
                        "host": result.machine.host,
                        "sweep": args.sweep,
                        "pid": orphan.get("pid"),
                        "cwd": orphan.get("cwd"),
                        "cmdline": orphan.get("cmdline"),
                    },
                )
        update_machine_state(
            env,
            result.machine,
            {
                "last_audit_at": iso_now(),
                "last_audit_sweep": args.sweep,
                "last_audit_dead_launches": row["dead_launch_count"],
                "last_audit_orphans_killed": row["killed_orphans_count"],
            },
        )

    if args.json:
        print(json.dumps(output_rows, indent=2, sort_keys=True))
    else:
        for row in output_rows:
            print(
                " ".join(
                    [
                        f"machine={row['machine']}",
                        f"sweep={row['sweep']}",
                        f"jobs_total={row['jobs_total']}",
                        f"active_launches={row['active_launches']}",
                        f"dead_launches={row['dead_launch_count']}",
                        f"stale_leases_cleared={row['stale_leases_cleared']}",
                        f"killed_orphans={row['killed_orphans_count']}",
                    ]
                )
            )
    if errors:
        return print_command_results(errors)
    return 0


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
            machine_disk=payload.get("disk"),
            min_free_disk_gib=min_free_disk_gib(env),
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

    cleanup_shm = launcher_cleanup_shm(env)
    use_pytorch_allocator = launcher_enable_pytorch_allocator(env)

    remote_script = textwrap.dedent(
        f"""
        import json
        import os
        import shlex
        import subprocess
        import time
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
        lock_path = Path(str(lease_path) + ".lock")
        deadline = time.monotonic() + {DEFAULT_LEASE_LOCK_WAIT_SEC}

        while True:
            try:
                os.mkdir(lock_path)
                break
            except FileExistsError:
                stale_seconds = 0.0
                try:
                    stale_seconds = time.time() - lock_path.stat().st_mtime
                except OSError:
                    stale_seconds = 0.0
                if stale_seconds > {DEFAULT_LEASE_LOCK_STALE_SEC}:
                    try:
                        os.rmdir(lock_path)
                        continue
                    except OSError:
                        pass
                if time.monotonic() >= deadline:
                    print(json.dumps({{"ok": False, "error": "lease lock timeout", "lease_path": str(lease_path)}}, sort_keys=True))
                    raise SystemExit(4)
                time.sleep(0.05)

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

        try:
            current = None
            if lease_path.is_file():
                try:
                    current = json.loads(lease_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    current = {{"invalid": True}}

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
            cleanup_lines = [f"rm -f {shlex.quote(str(lease_path))}"]
            if {cleanup_shm!r}:
                cleanup_lines.append(
                    "find /dev/shm -maxdepth 1 -user \\\"$(id -u)\\\" "
                    "\\\\( -name 'torch_*' -o -name 'pymp-*' \\\\) "
                    "-exec rm -rf {{}} + 2>/dev/null || true"
                )
            cleanup_body = " ; ".join(cleanup_lines)
            launcher_lines = [
                "#!/usr/bin/env bash",
                "set -eu",
                "cleanup() {{ " + cleanup_body + "; }}",
                "trap cleanup EXIT",
                f"cd {{shlex.quote(str(workdir))}}",
                f"cat > {{shlex.quote(str(lease_path))}} <<'JSON'",
                lease_text,
                "JSON",
            ]
            if {use_pytorch_allocator!r}:
                launcher_lines.append(
                    "export PYTORCH_ALLOC_CONF="
                    '"${{PYTORCH_ALLOC_CONF:-expandable_segments:True}}"'
                )
            launcher_lines.append(
                f"exec env CUDA_VISIBLE_DEVICES={{payload['gpu']}} " + argv_text
            )
            launcher = "\\n".join(launcher_lines)
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
        finally:
            try:
                os.rmdir(lock_path)
            except OSError:
                pass
        """
    ).strip()

    remote_command = build_remote_script(
        f"python3 - <<'PY'\n{remote_script}\nPY",
        workdir=machine.workdir,
        require_workdir=True,
    )
    completed = run_command_capture(
        ssh_command(
            machine,
            allocate_tty=False,
            remote_command=remote_command,
            allow_missing_tools=args.dry_run,
        ),
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
        status_error = None
        if record_path.is_file():
            record_payload, status_error = load_json_safe(record_path)
            if record_payload is not None:
                status = str(record_payload.get("status", "pending"))
            elif status_error is not None:
                status = "failed"
        if status not in {"success", "failed", "pending"}:
            status = "failed"
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
                "status_error": status_error,
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

    audit_parser = subparsers.add_parser(
        "audit",
        help=(
            "Audit remote sweep launcher state, clear stale leases for dead launches, "
            "and optionally kill orphaned worker processes"
        ),
    )
    audit_parser.add_argument("--machine", help="Machine name from the env file")
    audit_parser.add_argument("--group", help="Operate across a configured group")
    audit_parser.add_argument(
        "--sweep", required=True, help="Sweep name, for example slinoss-uea-grid"
    )
    audit_parser.add_argument(
        "--kill-orphans",
        action="store_true",
        help="Kill orphaned Python sweep processes under WORKDIR (PPID=1)",
    )
    audit_parser.add_argument("--json", action="store_true")
    audit_parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_FANOUT_PARALLELISM
    )
    audit_parser.add_argument("--dry-run", action="store_true")
    audit_parser.set_defaults(handler=command_audit)

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
