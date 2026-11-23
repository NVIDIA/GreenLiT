# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Read-only Linux per-CPU tick and scheduler activity observations."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import re


_CPU_STAT_PATH = Path("/proc/stat")
_SCHEDSTAT_PATH = Path("/proc/schedstat")
_SUPPORTED_SCHEDSTAT_VERSIONS = frozenset({15, 16, 17})
_TIMER_LIST_PATH = Path("/proc/timer_list")


@dataclass(frozen=True)
class CpuActivity:
    """Per-CPU tick and scheduler activity counters."""

    busy_ticks: int
    idle_ticks: int
    runtime_ns: int
    timeslices: int


@dataclass(frozen=True)
class TickState:
    """Per-CPU NOHZ tick-scheduler snapshot from /proc/timer_list."""

    last_tick_ns: int
    nohz_enabled: bool
    tick_stopped: bool


class ObserverUnavailable(RuntimeError):
    """Required scheduler activity accounting is absent or unsupported."""


def _read_cpu_schedstat(cpus, schedstat_path):
    """Read high-resolution runtime and timeslice counts for requested CPUs.

    ``FileNotFoundError`` indicates ``CONFIG_SCHEDSTATS`` is absent and raises
    ``ObserverUnavailable``. Non-absence I/O errors such as ``PermissionError``
    propagate as ``RuntimeError`` so callers fail closed.
    """
    try:
        schedstat_lines = Path(schedstat_path).read_text(
            encoding="ascii"
        ).splitlines()
    except FileNotFoundError as err:
        raise ObserverUnavailable(
            f"Cannot read per-CPU scheduler activity from {schedstat_path}: {err}"
        ) from err
    except OSError as err:
        raise RuntimeError(
            f"Cannot read per-CPU scheduler activity from {schedstat_path}: {err}"
        ) from err
    except UnicodeError as err:
        raise RuntimeError(
            f"Cannot decode per-CPU scheduler activity from {schedstat_path}: {err}"
        ) from err

    version_line = schedstat_lines[0] if schedstat_lines else ""
    version_match = re.fullmatch(r"version ([0-9]{1,10})", version_line)
    if version_match is None:
        raise RuntimeError(f"Missing schedstat version in {schedstat_path}")
    schedstat_version = int(version_match.group(1))
    if schedstat_version not in _SUPPORTED_SCHEDSTAT_VERSIONS:
        raise ObserverUnavailable(
            f"Unsupported schedstat version {schedstat_version} in {schedstat_path}"
        )

    activity = {}
    for line in schedstat_lines[1:]:
        fields = line.split()
        if (
            not fields
            or not fields[0].startswith("cpu")
            or not fields[0][3:].isdigit()
        ):
            continue
        cpu = int(fields[0][3:])
        if cpu not in cpus:
            continue
        if cpu in activity or len(fields) < 10:
            raise RuntimeError(
                f"Malformed CPU scheduler activity row in {schedstat_path}: {line!r}"
            )
        try:
            runtime_ns = int(fields[7])
            timeslices = int(fields[9])
        except ValueError as err:
            raise RuntimeError(
                f"Malformed CPU scheduler activity row in {schedstat_path}: {line!r}"
            ) from err
        if runtime_ns < 0 or timeslices < 0:
            raise RuntimeError(
                f"Negative CPU scheduler activity counter in {schedstat_path}: "
                f"{line!r}"
            )
        activity[cpu] = (runtime_ns, timeslices)

    missing_cpus = cpus - activity.keys()
    if missing_cpus:
        raise RuntimeError(
            f"Missing CPU scheduler activity rows in {schedstat_path}: "
            f"{sorted(missing_cpus)}"
        )
    return activity


def _read_cpu_ticks(cpus, stat_path):
    """Read busy and idle ticks for requested CPUs from /proc/stat."""
    tick_activity = {}
    try:
        stat_lines = Path(stat_path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as err:
        raise RuntimeError(
            f"Cannot read per-CPU activity from {stat_path}: {err}"
        ) from err

    for line in stat_lines:
        fields = line.split()
        if not fields:
            continue
        cpu_text = fields[0][3:] if fields[0].startswith("cpu") else ""
        if not cpu_text.isdigit():
            continue
        cpu = int(cpu_text)
        if cpu not in cpus:
            continue
        if cpu in tick_activity or len(fields) < 9:
            raise RuntimeError(
                f"Malformed CPU activity row in {stat_path}: {line!r}"
            )
        try:
            counters = [int(value) for value in fields[1:9]]
        except ValueError as err:
            raise RuntimeError(
                f"Malformed CPU activity row in {stat_path}: {line!r}"
            ) from err
        if any(value < 0 for value in counters):
            raise RuntimeError(
                f"Negative CPU activity counter in {stat_path}: {line!r}"
            )

        idle = counters[3]
        iowait = counters[4]
        tick_activity[cpu] = (sum(counters) - idle - iowait, idle)

    missing_cpus = cpus - tick_activity.keys()
    if missing_cpus:
        raise RuntimeError(
            f"Missing CPU activity rows in {stat_path}: {sorted(missing_cpus)}"
        )
    return tick_activity


def activity_delta(
    before: Mapping[int, CpuActivity],
    after: Mapping[int, CpuActivity],
    cpu: int,
) -> CpuActivity:
    """Return monotonic activity deltas for one CPU."""
    try:
        before_cpu = before[cpu]
        after_cpu = after[cpu]
    except KeyError as err:
        raise RuntimeError(f"Missing CPU {cpu} from activity snapshot") from err
    before_values = (
        before_cpu.busy_ticks,
        before_cpu.idle_ticks,
        before_cpu.runtime_ns,
        before_cpu.timeslices,
    )
    after_values = (
        after_cpu.busy_ticks,
        after_cpu.idle_ticks,
        after_cpu.runtime_ns,
        after_cpu.timeslices,
    )
    if any(
        after_value < before_value
        for before_value, after_value in zip(before_values, after_values)
    ):
        raise RuntimeError(f"CPU {cpu} activity counters moved backwards")
    return CpuActivity(
        busy_ticks=after_cpu.busy_ticks - before_cpu.busy_ticks,
        idle_ticks=after_cpu.idle_ticks - before_cpu.idle_ticks,
        runtime_ns=after_cpu.runtime_ns - before_cpu.runtime_ns,
        timeslices=after_cpu.timeslices - before_cpu.timeslices,
    )


def cpu_stayed_quiet(
    before: Mapping[int, CpuActivity],
    after: Mapping[int, CpuActivity],
    cpu: int,
) -> bool:
    """Return whether a CPU accrued idle time without task activity."""
    delta = activity_delta(before, after, cpu)
    return (
        delta.runtime_ns == 0
        and delta.timeslices == 0
        and delta.idle_ticks > 0
    )


def read_cpu_activity(
    cpus: Iterable[int],
    stat_path: str | PathLike[str] = _CPU_STAT_PATH,
    schedstat_path: str | PathLike[str] = _SCHEDSTAT_PATH,
) -> dict[int, CpuActivity]:
    """Read tick and high-resolution activity for requested logical CPUs."""
    requested_cpus = set(cpus)
    sched_activity = _read_cpu_schedstat(requested_cpus, schedstat_path)
    tick_activity = _read_cpu_ticks(requested_cpus, stat_path)
    return {
        cpu: CpuActivity(
            busy_ticks=tick_activity[cpu][0],
            idle_ticks=tick_activity[cpu][1],
            runtime_ns=sched_activity[cpu][0],
            timeslices=sched_activity[cpu][1],
        )
        for cpu in requested_cpus
    }


def read_sched_domain_cpus(
    schedstat_path: str | PathLike[str] = _SCHEDSTAT_PATH,
) -> set[int]:
    """Return logical CPUs attached to at least one scheduler domain.

    In ``/proc/schedstat``, ``show_schedstat()`` emits ``domain<N>`` lines
    below ``cpu<M>`` if and only if ``rq->sd`` is non-NULL.
    """
    try:
        lines = Path(schedstat_path).read_text(encoding="ascii").splitlines()
    except FileNotFoundError as err:
        raise ObserverUnavailable(
            f"Cannot read scheduler domains from {schedstat_path}: {err}"
        ) from err
    except (OSError, UnicodeError) as err:
        raise RuntimeError(
            f"Cannot read scheduler domains from {schedstat_path}: {err}"
        ) from err

    version_line = lines[0] if lines else ""
    version_match = re.fullmatch(r"version ([0-9]{1,10})", version_line)
    if version_match is None:
        raise RuntimeError(f"Missing schedstat version in {schedstat_path}")
    version = int(version_match.group(1))
    if version not in _SUPPORTED_SCHEDSTAT_VERSIONS:
        raise ObserverUnavailable(
            f"Unsupported schedstat version {version} in {schedstat_path}"
        )

    domain_cpus = set()
    current_cpu = None
    saw_cpu = False
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        head = fields[0]
        if head.startswith("cpu") and head[3:].isdigit():
            current_cpu = int(head[3:])
            saw_cpu = True
        elif (
            current_cpu is not None
            and head.startswith("domain")
            and head[6:].isdigit()
        ):
            domain_cpus.add(current_cpu)

    if not saw_cpu:
        raise RuntimeError(f"Missing per-CPU entries in {schedstat_path}")
    return domain_cpus


def read_tick_states(
    cpus: Iterable[int],
    timer_list_path: str | PathLike[str] = _TIMER_LIST_PATH,
) -> dict[int, TickState]:
    """Read per-CPU NOHZ ``tick_sched`` state from ``/proc/timer_list``."""
    requested = set(cpus)
    try:
        lines = Path(timer_list_path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as err:
        raise ObserverUnavailable(
            f"Cannot read NOHZ tick state from {timer_list_path}: {err}"
        ) from err
    except (OSError, UnicodeError) as err:
        raise RuntimeError(
            f"Cannot read NOHZ tick state from {timer_list_path}: {err}"
        ) from err

    states = {}
    current_cpu = None
    fields = {}

    def flush():
        if current_cpu is None or current_cpu not in requested:
            return
        has_nohz = ".nohz" in fields or ".nohz_mode" in fields
        if not has_nohz or not {".tick_stopped", ".last_tick"} <= fields.keys():
            raise ObserverUnavailable(
                f"NOHZ tick_sched fields are absent for CPU {current_cpu} in "
                f"{timer_list_path}"
            )
        if ".nohz_mode" in fields:
            nohz_enabled = fields[".nohz_mode"] in {1, 2}
        else:
            nohz_enabled = fields[".nohz"] == 1
        states[current_cpu] = TickState(
            last_tick_ns=fields[".last_tick"],
            nohz_enabled=nohz_enabled,
            tick_stopped=fields[".tick_stopped"] == 1,
        )

    for line in lines:
        if line.startswith("cpu: "):
            flush()
            cpu_token = line[5:].strip()
            if not cpu_token.isdigit():
                raise RuntimeError(
                    f"Malformed CPU header in {timer_list_path}: {line!r}"
                )
            current_cpu = int(cpu_token)
            fields = {}
            continue
        if current_cpu is None or current_cpu not in requested:
            continue
        if not line.startswith("  ."):
            continue
        key, sep, raw_value = line.strip().partition(":")
        if not sep:
            continue
        key = key.strip()
        if key in {".nohz", ".nohz_mode", ".tick_stopped", ".last_tick"}:
            value_token = raw_value.strip().split()[0]
            try:
                fields[key] = int(value_token)
            except ValueError as err:
                raise RuntimeError(
                    f"Malformed {key} value in {timer_list_path}: {line!r}"
                ) from err

    flush()
    missing = requested - states.keys()
    if missing:
        raise RuntimeError(
            f"Missing CPU sections in {timer_list_path}: {sorted(missing)}"
        )
    return states
