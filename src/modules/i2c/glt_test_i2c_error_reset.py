# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tracefs proof for the Tegra I2C ACPI device_reset() fallback."""
# pylint: disable=missing-function-docstring,too-many-branches,too-many-return-statements
# pylint: disable=too-many-statements,broad-exception-caught

import errno
import fcntl
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils

try:
    from .glt_test_i2c import (
        _IPMI_TEGRA_TOPOLOGY_ERROR,
        _IPMI_TEGRA_TOPOLOGY_MATCH,
        _IPMI_TEGRA_TOPOLOGY_NO_MATCH,
        _dev_unbind_bind,
        _ipmi_over_tegra_i2c_acpi_topology,
        _run_ipmitool_mc_info,
        _tegra_i2c_devices,
        _tegra_i2c_kernel_error_messages,
    )
except ImportError:
    from modules.i2c.glt_test_i2c import (
        _IPMI_TEGRA_TOPOLOGY_ERROR,
        _IPMI_TEGRA_TOPOLOGY_MATCH,
        _IPMI_TEGRA_TOPOLOGY_NO_MATCH,
        _dev_unbind_bind,
        _ipmi_over_tegra_i2c_acpi_topology,
        _run_ipmitool_mc_info,
        _tegra_i2c_devices,
        _tegra_i2c_kernel_error_messages,
    )

_STATUS_FAIL, _STATUS_OK, _STATUS_SKIP, _STATUS_WARN = "fail", "ok", "skip", "warn"

_LOCK_PATH = Path("/run/lock/greenlit-tegra-i2c-acpi-reset.lock")
_MARKER_PATH = Path("/run/lock/greenlit-tegra-i2c-acpi-reset.json")
_MARKER_VERSION = "greenlit-tegra-i2c-acpi-reset-v1"
_TRACE_GROUP = _TRACE_INSTANCE = "greenlit_tegra_i2c_acpi_reset"
_EVENT_INIT_ENTRY = "greenlit_tegra_i2c_init_entry"
_EVENT_INIT_RETURN = "greenlit_tegra_i2c_init_return"
_EVENT_RESET_ENTRY = "greenlit_device_reset_entry"
_EVENT_RESET_RETURN = "greenlit_device_reset_return"
_EVENT_WRITEL_ENTRY = "greenlit_i2c_writel_entry"
_TRACE_EVENTS = (_EVENT_INIT_ENTRY, _EVENT_INIT_RETURN, _EVENT_RESET_ENTRY, _EVENT_RESET_RETURN, _EVENT_WRITEL_ENTRY)
_DEVICE_RESET_KPROBE_COMMANDS = (
    f"p:{_TRACE_GROUP}/{_EVENT_INIT_ENTRY} tegra_i2c_init i2c=%x0:x64",
    f"r:{_TRACE_GROUP}/{_EVENT_INIT_RETURN} tegra_i2c_init retval=$retval:s32",
    f"p:{_TRACE_GROUP}/{_EVENT_RESET_ENTRY} __device_reset dev=%x0:x64",
    f"r:{_TRACE_GROUP}/{_EVENT_RESET_RETURN} __device_reset retval=$retval:s32",
    f"p:{_TRACE_GROUP}/{_EVENT_WRITEL_ENTRY} i2c_writel i2c=%x0:x64 value=%x1:x64 reg=%x2:x64",
)
_DELETE_KPROBE_COMMANDS = tuple(f"-:{_TRACE_GROUP}/{event}" for event in _TRACE_EVENTS)
_DEVICE_RE = re.compile(r"\ANVDA0301:[0-9A-Fa-f]{2}\Z")
_TOPOLOGY_DEVICE_RE = re.compile(r"\ANVDA[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}\Z")
_SITE_HEX = r"0x[0-9A-Fa-f]+"
_CALLER_SYMBOL, _MOD = r"[A-Za-z0-9_.$]+", r"(?: \[i2c_tegra\])?"
_VAL = r"(-?(?:0x[0-9A-Fa-f]+|[0-9]+))"
_WARNING_START_RE = re.compile(r"\b(?:WARNING|WARN(?:_ON)?)\b")
_INIT_STACK_RE = re.compile(rf"\s*\??\s*tegra_i2c_init\+{_SITE_HEX}/{_SITE_HEX}(?:\s+\[[^]]+\])?\s*\Z")
_FIELD_GRAMMARS = {
    _EVENT_INIT_ENTRY: (re.compile(rf"\A\(tegra_i2c_init\+{_SITE_HEX}/{_SITE_HEX}{_MOD}\) i2c={_VAL}\Z"), ("i2c",)),
    _EVENT_INIT_RETURN: (
        re.compile(rf"\A\({_CALLER_SYMBOL}\+{_SITE_HEX}/{_SITE_HEX}{_MOD} <- tegra_i2c_init\) retval={_VAL}\Z"),
        ("retval",),
    ),
    _EVENT_RESET_ENTRY: (re.compile(rf"\A\(__device_reset\+{_SITE_HEX}/{_SITE_HEX}\) dev={_VAL}\Z"), ("dev",)),
    _EVENT_RESET_RETURN: (
        re.compile(rf"\A\({_CALLER_SYMBOL}\+{_SITE_HEX}/{_SITE_HEX}{_MOD} <- __device_reset\) retval={_VAL}\Z"),
        ("retval",),
    ),
    _EVENT_WRITEL_ENTRY: (
        re.compile(
            rf"\A\(i2c_writel\+{_SITE_HEX}/{_SITE_HEX}{_MOD}\) "
            rf"i2c={_VAL} value={_VAL} reg={_VAL}\Z"
        ),
        ("i2c", "value", "reg"),
    ),
}
_RESET_RETURN, _RESET_ASSERT_VALUE, _RESET_DEASSERT_VALUE, _RESET_REGISTER = -2, 1, 0, 0x0A8
_TRACE_STAT_KEYS = ("overrun", "commit overrun", "dropped events")


@dataclass
class _Cycle:
    i2c: int
    reset_open: bool = False
    reset_retval: int | None = None
    reset_writes: list[int] = field(default_factory=list)


def _inconclusive(reason: str, status: str = _STATUS_WARN) -> tuple[str, str]:
    return (status, reason + "; trace proof is non-conclusive.")


def _event_dir(trace_root: Path, event: str) -> Path:
    return trace_root / "events" / _TRACE_GROUP / event


def _instance_dir(trace_root: Path) -> Path:
    return trace_root / "instances" / _TRACE_INSTANCE


def _instance_event_dir(trace_root: Path, event: str) -> Path:
    return _instance_dir(trace_root) / "events" / _TRACE_GROUP / event


def _write_text(path: Path, value: str, mode: str = "w") -> None:
    opener = (lambda path, _flags: os.open(path, os.O_WRONLY | os.O_CLOEXEC)) if mode == "a" else None
    with open(path, "w", encoding="ascii", opener=opener) as handle:
        handle.write(value)


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _tracefs_root() -> Path | None:
    for trace_root in (Path("/sys/kernel/tracing"), Path("/sys/kernel/debug/tracing")):
        try:
            kprobe = (trace_root / "kprobe_events").stat().st_mode
            instances = (trace_root / "instances").stat().st_mode
        except FileNotFoundError:
            continue
        if kprobe & 0o170000 != 0o100000:
            raise RuntimeError(f"{trace_root / 'kprobe_events'} is not a regular file")
        if instances & 0o170000 != 0o040000:
            raise RuntimeError(f"{trace_root / 'instances'} is not a directory")
        return trace_root
    return None


def _valid_device_name(device: str | None) -> bool:
    return isinstance(device, str) and _DEVICE_RE.fullmatch(device) is not None


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _fd_authenticity_error(file_fd: int, path: Path) -> str | None:
    try:
        metadata = os.fstat(file_fd)
    except OSError as exc:
        return f"cannot stat {path}: {exc}"
    mode = metadata.st_mode & 0o7777
    if metadata.st_mode & 0o170000 != 0o100000:
        return f"{path} is not a regular file"
    if metadata.st_uid != os.geteuid():
        return f"{path} is not owned by the current effective UID"
    if mode != 0o600:
        return f"{path} mode is {mode:04o}, expected 0600"
    return None


def _marker_payload(device: str) -> dict:
    return {
        "version": _MARKER_VERSION,
        "boot_id": _boot_id(),
        "selected_device": device,
        "trace_instance": _TRACE_INSTANCE,
        "trace_group": _TRACE_GROUP,
        "trace_events": list(_TRACE_EVENTS),
    }


def _fsync_parent(path: Path) -> None:
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _write_marker(device: str) -> None:
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{_MARKER_PATH.name}.", suffix=".tmp", dir=_MARKER_PATH.parent
        )
        temp_path = Path(temp_name)
        os.fchmod(temp_fd, 0o600)
        with os.fdopen(temp_fd, "w", encoding="utf-8") as marker_file:
            temp_fd = None
            json.dump(_marker_payload(device), marker_file, sort_keys=True)
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(temp_path, _MARKER_PATH)
        temp_path = None
        _fsync_parent(_MARKER_PATH)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def _read_valid_marker() -> tuple[dict | None, str]:
    try:
        open_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        marker_fd = os.open(_MARKER_PATH, open_flags)
    except FileNotFoundError:
        return (None, "")
    except OSError as exc:
        return (None, f"cannot safely open fixed cleanup marker: {exc}")
    auth_error = _fd_authenticity_error(marker_fd, _MARKER_PATH)
    if auth_error:
        os.close(marker_fd)
        return (None, "unsafe fixed cleanup marker: " + auth_error)
    try:
        with os.fdopen(marker_fd, "r", encoding="utf-8") as marker_file:
            marker = json.load(marker_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return (None, f"cannot read fixed cleanup marker: {exc}")
    expected = {
        "version": _MARKER_VERSION,
        "boot_id": _boot_id(),
        "trace_instance": _TRACE_INSTANCE,
        "trace_group": _TRACE_GROUP,
        "trace_events": list(_TRACE_EVENTS),
    }
    if not isinstance(marker, dict):
        return (None, "fixed cleanup marker is not a JSON object")
    for key, value in expected.items():
        if marker.get(key) != value:
            return (None, f"fixed cleanup marker has unexpected {key}")
    if not _valid_device_name(marker.get("selected_device")):
        return (None, "fixed cleanup marker has an invalid selected device")
    return (marker, "")


def _fixed_artifacts(trace_root: Path) -> list[str]:
    artifacts = [str(_instance_dir(trace_root))] if _path_present(_instance_dir(trace_root)) else []
    artifacts.extend(
        f"{_TRACE_GROUP}/{event}" for event in _TRACE_EVENTS if _path_present(_event_dir(trace_root, event))
    )
    return artifacts


def _restore_binding(device: str) -> list[str]:
    if not _valid_device_name(device):
        return [f"Invalid device token for bind restoration: {device!r}"]
    if device in _tegra_i2c_devices(prefix="NVDA"):
        return []
    errors = []
    polls, sleep_seconds = 5, 0.2
    for attempt in range(polls):
        try:
            _write_text(Path("/sys/bus/platform/drivers/tegra-i2c/bind"), device)
        except OSError as exc:
            errors.append(f"write tegra-i2c bind for {device}: {exc}")
        if device in _tegra_i2c_devices(prefix="NVDA"):
            return []
        if attempt + 1 < polls:
            time.sleep(sleep_seconds)
    return errors or [f"{device} is still absent from the tegra-i2c driver after bind restore"]


def _stop_instance_tracing(trace_root: Path) -> str | None:
    try:
        if _path_present(_instance_dir(trace_root)):
            _write_text(_instance_dir(trace_root) / "tracing_on", "0")
    except OSError as exc:
        return f"Cannot stop private trace instance: {exc}"
    return None


def _cleanup_owned_tracefs(trace_root: Path, device: str) -> list[str]:
    errors = _restore_binding(device)
    stop_error = _stop_instance_tracing(trace_root)
    if stop_error:
        errors.append(stop_error)
    for event in _TRACE_EVENTS:
        try:
            if _path_present(_instance_event_dir(trace_root, event)):
                _write_text(_instance_event_dir(trace_root, event) / "enable", "0")
        except OSError as exc:
            errors.append(f"Cannot disable private event {_TRACE_GROUP}/{event}: {exc}")
    try:
        if _path_present(_instance_dir(trace_root)):
            _instance_dir(trace_root).rmdir()
    except OSError as exc:
        errors.append(f"Cannot remove private trace instance {_instance_dir(trace_root)}: {exc}")
    for command, event in zip(_DELETE_KPROBE_COMMANDS, _TRACE_EVENTS):
        try:
            if _path_present(_event_dir(trace_root, event)):
                _write_text(trace_root / "kprobe_events", command, "a")
        except OSError as exc:
            errors.append(f"Cannot delete kprobe event {_TRACE_GROUP}/{event}: {exc}")
    remaining = _fixed_artifacts(trace_root)
    if remaining:
        errors.append("Owned tracefs artifacts remain: " + ", ".join(remaining))
    if not errors:
        try:
            _MARKER_PATH.unlink()
            _fsync_parent(_MARKER_PATH)
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"Cannot remove fixed cleanup marker: {exc}")
    return errors


def _recover_or_refuse_foreign_artifacts(trace_root: Path) -> tuple[str, str]:
    marker, marker_error = _read_valid_marker()
    if marker_error:
        return (_STATUS_FAIL, f"{marker_error}; refusing to touch fixed tracefs artifacts without same-boot ownership.")
    if marker is not None:
        errors = _cleanup_owned_tracefs(trace_root, marker["selected_device"])
        return (_STATUS_FAIL, "Stale cleanup failed: " + "; ".join(errors)) if errors else (_STATUS_OK, "")
    artifacts = _fixed_artifacts(trace_root)
    if artifacts:
        reason = "Fixed tracefs artifacts already exist without a valid same-boot marker: "
        return (_STATUS_FAIL, reason + ", ".join(artifacts))
    return (_STATUS_OK, "")


def _acquire_lock() -> tuple[int | None, tuple[str, str]]:
    try:
        lock_fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        return (None, (_STATUS_SKIP, f"Cannot open tracefs lifecycle lock: {exc}"))
    auth_error = _fd_authenticity_error(lock_fd, _LOCK_PATH)
    if auth_error:
        os.close(lock_fd)
        return (None, (_STATUS_FAIL, "Unsafe tracefs lifecycle lock: " + auth_error))
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(lock_fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return (None, (_STATUS_SKIP, "Another tracefs lifecycle owner is active."))
        return (None, (_STATUS_FAIL, f"Cannot acquire tracefs lifecycle lock: {exc}"))
    return (lock_fd, (_STATUS_OK, ""))


def _release_lock(lock_fd: int | None) -> str | None:
    if lock_fd is None:
        return None
    errors = []
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError as exc:
        errors.append(f"Cannot unlock tracefs lifecycle lock: {exc}")
    try:
        os.close(lock_fd)
    except OSError as exc:
        errors.append(f"Cannot close tracefs lifecycle lock: {exc}")
    return "; ".join(errors) if errors else None


def _select_device(logger) -> tuple[tuple[str, str], str | None]:
    if not Path("/sys/firmware/acpi").is_dir():
        return ((_STATUS_SKIP, "System is not ACPI-booted."), None)
    status, reason, _path = _ipmi_over_tegra_i2c_acpi_topology(logger)
    if status == _IPMI_TEGRA_TOPOLOGY_ERROR:
        return ((_STATUS_FAIL, reason), None)
    if status == _IPMI_TEGRA_TOPOLOGY_NO_MATCH:
        return ((_STATUS_SKIP, reason), None)
    if status != _IPMI_TEGRA_TOPOLOGY_MATCH:
        return ((_STATUS_FAIL, f"Unexpected Tegra I2C topology result: {reason}"), None)
    if not _valid_device_name(reason):
        if isinstance(reason, str) and _TOPOLOGY_DEVICE_RE.fullmatch(reason):
            return (_inconclusive(
                f"{reason} is not a supported Grace/Tegra194 ACPI I2C controller", _STATUS_SKIP), None)
        return ((_STATUS_FAIL, f"Unexpected Tegra I2C topology result: {reason}"), None)
    if reason not in _tegra_i2c_devices(prefix="NVDA"):
        return ((_STATUS_FAIL, f"{reason} is not bound to tegra-i2c."), None)
    return ((_STATUS_OK, ""), reason)


def _setup_tracefs(trace_root: Path) -> tuple[str, str]:
    try:
        for command in _DEVICE_RESET_KPROBE_COMMANDS:
            _write_text(trace_root / "kprobe_events", command, "a")
        instance = _instance_dir(trace_root)
        instance.mkdir()
        _write_text(instance / "tracing_on", "0")
        _write_text(instance / "trace", "")
        for event in _TRACE_EVENTS:
            event_dir = _instance_event_dir(trace_root, event)
            if not event_dir.is_dir():
                raise OSError(f"trace event {_TRACE_GROUP}/{event} is unavailable in the private instance")
            _write_text(event_dir / "filter", f"common_pid == {os.getpid()}")
            _write_text(event_dir / "enable", "1")
        _write_text(instance / "tracing_on", "1")
    except OSError as exc:
        return _inconclusive(f"Cannot configure tracefs proof: {exc}", _STATUS_SKIP)
    return (_STATUS_OK, "")


def _parse_int(value: str) -> int:
    parsed = int(value, 0)
    return parsed - 2 ** 64 if parsed >= 2 ** 63 else parsed


def _trace_event_name(line: str) -> str | None:
    return next((event for event in _TRACE_EVENTS if f": {event}:" in line), None)


def _trace_pid(line: str) -> int | None:
    matches = re.findall(r"-(\d+)\s+\[\d+\]", line)
    return int(matches[-1]) if matches else None


def _trace_fields(line: str, event: str) -> dict[str, int] | None:
    _prefix, separator, body = line.partition(f": {event}:")
    if not separator:
        return None
    pattern, names = _FIELD_GRAMMARS[event]
    match = pattern.fullmatch(body.strip())
    if match is None:
        return None
    return {name: _parse_int(value) for name, value in zip(names, match.groups())}


def _parse_trace_lifecycle(trace_text: str) -> tuple[str, str]:
    active, proof_pid, proof_i2c, last_completed_reset_retval = {}, None, None, None
    acpi_reset_complete = False
    relevant = complete = 0
    for line in trace_text.splitlines():
        event = _trace_event_name(line)
        if event is None:
            continue
        pid = _trace_pid(line)
        if pid is None:
            return _inconclusive("trace row lacks a parseable common_pid")
        if proof_pid is None:
            proof_pid = pid
        elif pid != proof_pid:
            return _inconclusive("trace contains relevant events from multiple PIDs")
        fields = _trace_fields(line, event)
        if fields is None:
            return _inconclusive(f"{event} trace row has malformed fields")
        relevant += 1
        if event == _EVENT_INIT_ENTRY:
            if active:
                return _inconclusive("nested tegra_i2c_init lifecycle observed")
            if proof_i2c is not None and fields["i2c"] != proof_i2c:
                return _inconclusive("trace contains tegra_i2c_init lifecycles for multiple I2C controllers")
            active[pid] = _Cycle(proof_i2c := fields["i2c"])
        elif event == _EVENT_RESET_ENTRY:
            cycle = active.get(pid)
            if cycle is None:
                return _inconclusive("__device_reset entry has no enclosing tegra_i2c_init")
            if cycle.reset_open or cycle.reset_retval is not None:
                return _inconclusive("nested or repeated __device_reset lifecycle observed")
            cycle.reset_open = True
        elif event == _EVENT_RESET_RETURN:
            cycle = active.get(pid)
            if cycle is None or not cycle.reset_open:
                return _inconclusive("__device_reset return has no matching entry")
            cycle.reset_open, cycle.reset_retval = False, fields.get("retval")
            if cycle.reset_retval not in (_RESET_RETURN, 0):
                return (_STATUS_FAIL, f"__device_reset returned {cycle.reset_retval}, expected 0 or -ENOENT (-2).")
        elif event == _EVENT_WRITEL_ENTRY:
            cycle = active.get(pid)
            if cycle is None:
                if fields["reg"] != _RESET_REGISTER:
                    continue
                if acpi_reset_complete and last_completed_reset_retval == 0:
                    return (_STATUS_FAIL, "reset-register i2c_writel after ACPI _RST contradicted no-fallback proof.")
                return _inconclusive("i2c_writel has no enclosing tegra_i2c_init")
            if cycle.reset_retval is None:
                return (_STATUS_FAIL, "i2c_writel occurred before __device_reset returned.")
            if fields.get("i2c") != cycle.i2c:
                return _inconclusive("i2c_writel controller does not match tegra_i2c_init")
            value, register = fields.get("value"), fields.get("reg")
            if register != _RESET_REGISTER:
                continue
            if cycle.reset_retval == 0:
                return (_STATUS_FAIL, "reset-register write after ACPI _RST contradicted the no-fallback path.")
            if len(cycle.reset_writes) >= 2:
                return (_STATUS_FAIL, "unexpected reset-register i2c_writel after the assert/deassert pair.")
            if not cycle.reset_writes:
                if value != _RESET_ASSERT_VALUE:
                    return (_STATUS_FAIL, f"first reset-register write value was {value}, expected 1.")
            elif value != _RESET_DEASSERT_VALUE:
                return (_STATUS_FAIL, f"second reset-register write value was {value}, expected 0.")
            cycle.reset_writes.append(value)
        else:
            cycle = active.pop(pid, None)
            if cycle is None:
                return _inconclusive("tegra_i2c_init return has no matching entry")
            if cycle.reset_open:
                return _inconclusive("__device_reset return is missing before tegra_i2c_init returns")
            if fields["retval"] != 0:
                return (_STATUS_FAIL, f"tegra_i2c_init returned {fields.get('retval')}, expected 0.")
            if cycle.reset_retval is None:
                return (_STATUS_FAIL, "tegra_i2c_init completed without an attributable __device_reset return.")
            last_completed_reset_retval = cycle.reset_retval
            if cycle.reset_retval == 0:
                acpi_reset_complete = True
                continue
            if cycle.reset_writes != [_RESET_ASSERT_VALUE, _RESET_DEASSERT_VALUE]:
                return (_STATUS_FAIL, "635bf3c8853359a987c5c909d424df92a0d3016a "
                        "ACPI-without-_RST fallback discriminator: missing reset assert/deassert pair.")
            complete += 1
    if active:
        return _inconclusive("trace ended before tegra_i2c_init returned")
    if complete:
        return (_STATUS_OK, "")
    if acpi_reset_complete:
        return (_STATUS_SKIP, "ACPI _RST handled reset; -ENOENT fallback was not exercised.")
    if relevant == 0:
        return (_STATUS_FAIL, "No relevant tracefs events were observed after probe setup and rebind.")
    return (_STATUS_FAIL, "No complete tegra_i2c_init reset lifecycle was observed.")


def _trace_stats_overrun_or_dropped(trace_root: Path) -> list[str]:
    try:
        stats_files = sorted((_instance_dir(trace_root) / "per_cpu").glob("cpu*/stats"))
    except OSError as exc:
        return [f"Cannot enumerate private trace instance stats: {exc}"]
    if not stats_files:
        return ["Private trace instance per-CPU stats are unavailable"]
    problems = []
    for stats_file in stats_files:
        try:
            stats_text = stats_file.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"Cannot read {stats_file}: {exc}")
            continue
        seen = {key: 0 for key in _TRACE_STAT_KEYS}
        for line in stats_text.splitlines():
            stripped = line.strip()
            key, sep, raw_value = stripped.partition(":")
            key = key.strip()
            if key not in _TRACE_STAT_KEYS:
                continue
            if not sep:
                problems.append(f"{stats_file}: {key} missing ':'")
                continue
            seen[key] += 1
            if seen[key] > 1:
                problems.append(f"{stats_file}: duplicate {key}")
            value_text = raw_value.strip()
            if not value_text:
                problems.append(f"{stats_file}: empty {key} value")
                continue
            if re.fullmatch(r"[0-9]+", value_text) is None:
                problems.append(f"{stats_file}: malformed {key} value {value_text}")
                continue
            value = int(value_text)
            if value > 0:
                problems.append(f"{stats_file}: {stripped}")
        for key, count in seen.items():
            if count == 0:
                problems.append(f"{stats_file}: missing {key}")
    return problems


def _read_private_trace(trace_root: Path) -> tuple[tuple[str, str], str]:
    try:
        return ((_STATUS_OK, ""), (_instance_dir(trace_root) / "trace").read_text(encoding="utf-8"))
    except OSError as exc:
        return (_inconclusive(f"Cannot read private trace buffer: {exc}"), "")


def _tegra_i2c_init_warning(kernel_log: str) -> str | None:
    block = []
    for line in kernel_log.splitlines():
        starts_warning = _WARNING_START_RE.search(line)
        if block and (starts_warning or line.strip().startswith("---[ end trace")):
            if any(_INIT_STACK_RE.fullmatch(item) for item in block):
                return block[0].strip()
            block = []
        if starts_warning or block:
            block.append(line)
    return block[0].strip() if block and any(_INIT_STACK_RE.fullmatch(item) for item in block) else None


def _run_tracefs_proof(trace_root: Path, context: dict, logger) -> tuple[str, str]:
    recovery = _recover_or_refuse_foreign_artifacts(trace_root)
    if recovery[0] != _STATUS_OK:
        return recovery
    selected, device = _select_device(logger)
    if selected[0] != _STATUS_OK:
        return selected
    context["selected_device"] = device
    healthy, reason = _run_ipmitool_mc_info(logger)
    if not healthy:
        return (_STATUS_FAIL, reason)
    context["marker_active"] = True
    try:
        _write_marker(device)
    except OSError as exc:
        return (_STATUS_FAIL, f"Cannot write same-boot tracefs cleanup marker: {exc}")
    setup = _setup_tracefs(trace_root)
    if setup[0] != _STATUS_OK:
        return setup
    try:
        checkpoint = glt_utils.capture_kernel_log_checkpoint()
    except RuntimeError as exc:
        return (_STATUS_FAIL, f"Failed to capture kernel log checkpoint before rebind: {exc}")
    rebound, rebind_reason = _dev_unbind_bind(device)
    mutation_failure = f"Controlled unbind/bind cycle on {device} failed: {rebind_reason}" if not rebound else None
    if mutation_failure is None and device not in _tegra_i2c_devices(prefix="NVDA"):
        mutation_failure = f"{device} is not bound to tegra-i2c after the controlled rebind."
    if mutation_failure is not None and (restore_errors := _restore_binding(device)):
        return (_STATUS_FAIL, "Binding restore failed after controlled rebind: " + "; ".join(restore_errors))
    stop_error = _stop_instance_tracing(trace_root)
    pending_stop = _inconclusive(stop_error) if stop_error else None
    healthy, reason = _run_ipmitool_mc_info(logger)
    if not healthy:
        return (_STATUS_FAIL, reason)
    try:
        kernel_log = glt_utils.get_kernel_log(checkpoint)
    except RuntimeError as exc:
        return (_STATUS_FAIL, f"Failed to retrieve scoped kernel log after rebind: {exc}")
    kernel_errors = [message for message in _tegra_i2c_kernel_error_messages(device) if message in kernel_log]
    if kernel_errors:
        return (_STATUS_FAIL, "Kernel reported Tegra I2C errors after rebind: " + "; ".join(kernel_errors))
    if warning := _tegra_i2c_init_warning(kernel_log):
        return (_STATUS_FAIL, f"Kernel reported tegra_i2c_init warning after rebind: {warning}")
    if mutation_failure is not None or pending_stop is not None:
        return (_STATUS_FAIL, mutation_failure) if mutation_failure is not None else pending_stop
    stat_errors = _trace_stats_overrun_or_dropped(trace_root)
    if stat_errors:
        return _inconclusive("Private trace instance lost events: " + "; ".join(stat_errors))
    trace_status, trace_text = _read_private_trace(trace_root)
    return trace_status if trace_status[0] != _STATUS_OK else _parse_trace_lifecycle(trace_text)


def _cleanup_after_outcome(context: dict) -> list[str]:
    if not context["marker_active"]:
        return []
    try:
        return _cleanup_owned_tracefs(context["trace_root"], context["selected_device"])
    except Exception as exc:
        return [f"Cleanup raised after Tegra I2C ACPI reset proof: {exc}"]


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra_i2c_acpi_error_reset(_, logger):
    """
    Verify ACPI-backed Tegra I2C rebind calls device_reset() cleanly.

    PATCH: https://git.kernel.org/torvalds/c/56344e241c543f17e8102fa13466ad5c3e7dc9ff
    PATCH: https://git.kernel.org/torvalds/c/635bf3c8853359a987c5c909d424df92a0d3016a
    """
    lock_fd = None
    context = {"trace_root": None, "selected_device": None, "marker_active": False}
    outcome = (_STATUS_SKIP, "tracefs is not mounted; device_reset proof is non-conclusive.")
    active_base_exception = None
    try:
        context["trace_root"] = trace_root = _tracefs_root()
        if trace_root is not None:
            lock_fd, lock_outcome = _acquire_lock()
            outcome = _run_tracefs_proof(trace_root, context, logger) if lock_outcome[0] == _STATUS_OK else lock_outcome
    except Exception as exc:
        outcome = (_STATUS_FAIL, f"Unexpected error during Tegra I2C ACPI reset proof: {exc}")
    except BaseException as exc:
        active_base_exception = exc

    cleanup_errors = _cleanup_after_outcome(context)
    lock_error = _release_lock(lock_fd)
    if lock_error:
        cleanup_errors.append(lock_error)
    if active_base_exception is not None:
        if cleanup_errors:
            logger.error("Cleanup failed while preserving original exception: %s", "; ".join(cleanup_errors))
            if hasattr(active_base_exception, "add_note"):
                active_base_exception.add_note("; ".join(cleanup_errors))
        raise active_base_exception
    if cleanup_errors:
        detail = "Cleanup failed after Tegra I2C ACPI reset proof: " + "; ".join(cleanup_errors)
        if outcome[0] != _STATUS_OK or outcome[1]:
            detail += f". Original outcome: {outcome[0]}: {outcome[1]}"
        GLTCore.GLTTestResult.fail(detail)
    if outcome[0] == _STATUS_WARN:
        GLTCore.GLTTestResult.warn(outcome[1])
    if outcome[0] == _STATUS_SKIP:
        GLTCore.GLTTestResult.skip(outcome[1])
    if outcome[0] == _STATUS_FAIL:
        GLTCore.GLTTestResult.fail(outcome[1])
