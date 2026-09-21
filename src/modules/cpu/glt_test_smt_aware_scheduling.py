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

"""Test Vera SMT-aware asymmetric-capacity wakeup placement."""
# pylint: disable=too-many-lines

import ctypes
import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from common import glt_core as GLTCore
from common.cpu_topology import parse_cpu_mask
from common.nv_device import NvDev


CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
NODE_SYSFS_ROOT = Path("/sys/devices/system/node")
CPU_STAT_PATH = Path("/proc/stat")
SCHEDSTAT_PATH = Path("/proc/schedstat")
SUPPORTED_SCHEDSTAT_VERSIONS = frozenset({15, 16, 17})

FIX_COMMIT = "25a32e400a14"
FIX_SUBJECT = "sched/fair: Prefer fully-idle SMT cores in asym-capacity idle selection"
FIX_COMMIT_URL = (
    "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/"
    "commit/?id=25a32e400a14009601c0a727643057f5515152df"
)

MAX_CANDIDATE_CORES = 32
PROBE_ATTEMPTS_PER_SIBLING = 5
PROBE_MAX_TRIALS_PER_SIBLING = 15
PROBE_SETTLE_SECONDS = 0.10
PROBE_TIMEOUT_SECONDS = 3.0
PROBE_LINE_MAX_BYTES = 64
WAKEE_SCRIPT_PATH = Path("/usr/share/greenlit/bin/smt_aware_wakee.py")


@dataclass(frozen=True)
class _SmtCore:
    """One eligible two-thread SMT core."""

    capacity: int
    node: int
    siblings: tuple[int, int]


@dataclass(frozen=True)
class _ProbeLayout:
    """CPU layout used by the wakeup placement probe."""

    candidates: tuple[_SmtCore, ...]
    source: _SmtCore

    @property
    def candidate_cpus(self):
        """Return one eligible logical CPU from every candidate core."""
        return tuple(core.siblings[0] for core in self.candidates)


@dataclass(frozen=True)
class _TrialLayout:
    """Qualified placement CPUs plus a core reserved to prime the idle hint."""

    candidates: tuple[_SmtCore, ...]
    source: _SmtCore
    hint_seed_core: _SmtCore
    hint_seed_cpu: int

    @property
    def candidate_cpus(self):
        """Return the allowed wakee thread from every placement core."""
        return tuple(core.siblings[0] for core in self.candidates)


@dataclass(frozen=True)
class _CpuActivity:
    """Per-CPU activity counters used to reject contaminated trials."""

    busy_ticks: int
    idle_ticks: int
    runtime_ns: int
    timeslices: int


@dataclass(frozen=True)
class _WakeupTrial:
    """One uncontaminated wakeup placement observation."""

    candidate_cpus: tuple[int, ...]
    destination_cpu: int
    hint_seed_cpu: int


class _TrialContended(RuntimeError):
    """The selected CPUs did not remain quiet enough for a kernel verdict."""


class _ObserverUnavailable(RuntimeError):
    """Required scheduler activity accounting is unsupported or unavailable."""


class _ProbeInconclusive(RuntimeError):
    """The environment did not yield enough controlled trials for a verdict."""


def _candidate_layout_for_node(cores):
    """Use one node's local maximum-capacity core as the probe source."""
    capacities = {core.capacity for core in cores}
    if len(capacities) < 2:
        return None

    maximum_capacity = max(capacities)
    source_cores = [core for core in cores if core.capacity == maximum_capacity]
    source = min(source_cores, key=lambda core: (0 in core.siblings, core.siblings))
    candidates = [
        core for core in cores
        if core.capacity < source.capacity and 0 not in core.siblings
    ]
    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda core: (-core.capacity, core.siblings))
    return _ProbeLayout(
        candidates=tuple(candidates[:MAX_CANDIDATE_CORES]),
        source=source,
    )


def _get_current_cpu():
    """Return the current logical CPU through libc sched_getcpu()."""
    try:
        sched_getcpu = ctypes.CDLL(None, use_errno=True).sched_getcpu
    except AttributeError as err:
        raise RuntimeError("libc does not provide sched_getcpu()") from err

    sched_getcpu.argtypes = []
    sched_getcpu.restype = ctypes.c_int
    cpu = sched_getcpu()
    if cpu < 0:
        errno_value = ctypes.get_errno()
        raise RuntimeError(
            f"sched_getcpu() failed: {os.strerror(errno_value)}"
        )
    return cpu


def _placement_result(layout, results):
    """Return the GreenLiT verdict and reason for completed wakeup results."""
    candidate_cpus = set(layout.candidate_cpus)
    incomplete = {
        previous_cpu: results.get(previous_cpu, ())
        for previous_cpu in layout.source.siblings
        if len(results.get(previous_cpu, ())) != PROBE_ATTEMPTS_PER_SIBLING
    }
    if incomplete:
        details = "; ".join(
            f"previous CPU {previous_cpu} -> {list(destinations)}"
            for previous_cpu, destinations in sorted(incomplete.items())
        )
        return (
            GLTCore.GLTTestResult.FAIL,
            "Controlled wakeup probe returned incomplete uncontaminated "
            f"results ({details}); no kernel placement verdict is possible.",
        )

    mixed = {
        previous_cpu: destinations
        for previous_cpu, destinations in results.items()
        if candidate_cpus.intersection(destinations)
        and not set(destinations).issubset(candidate_cpus)
    }
    if mixed:
        details = "; ".join(
            f"previous CPU {previous_cpu} -> {list(destinations)}"
            for previous_cpu, destinations in sorted(mixed.items())
        )
        return (
            GLTCore.GLTTestResult.WARN,
            "Controlled wakeups produced mixed uncontaminated placements "
            f"({details}); the scheduler behavior is not consistent enough "
            "to attribute to the target patch.",
        )

    missing = {}
    for previous_cpu in layout.source.siblings:
        destinations = results.get(previous_cpu, ())
        if not candidate_cpus.intersection(destinations):
            missing[previous_cpu] = destinations
    if not missing:
        return GLTCore.GLTTestResult.PASS, ""

    details = "; ".join(
        f"previous CPU {previous_cpu} -> {list(destinations)}"
        for previous_cpu, destinations in sorted(missing.items())
    )
    return (
        GLTCore.GLTTestResult.FAIL,
        "Controlled wakeups did not move away from a logical CPU whose SMT "
        f"sibling was running ({details}); at least one lower-capacity "
        "candidate core was verified fully idle in each clean trial. "
        f"Configured candidate CPUs were {sorted(candidate_cpus)} and the "
        f"source capacity was {layout.source.capacity}. The observed behavior "
        f"is inconsistent with the targeted runtime behavior from {FIX_COMMIT} "
        f"({FIX_SUBJECT}) or its idle-core hint prerequisite. Review the "
        f"mainline commit: {FIX_COMMIT_URL}. This test does not "
        "exercise the complete patch series, and this result does not by itself "
        "prove that a particular source commit is absent.",
    )


def _process_state(pid, proc_root=Path("/proc")):
    """Read a process state from /proc/<pid>/stat."""
    stat_path = Path(proc_root) / str(pid) / "stat"
    try:
        stat_text = stat_path.read_text(encoding="ascii")
    except OSError as err:
        raise RuntimeError(f"Cannot read wakee state from {stat_path}: {err}") from err

    command_end = stat_text.rfind(")")
    if command_end < 0 or len(stat_text) <= command_end + 2:
        raise RuntimeError(f"Cannot parse wakee state from {stat_path}")
    return stat_text[command_end + 2]


def _read_cpu_ticks(cpus, stat_path):
    """Read busy and idle ticks for requested CPUs from /proc/stat."""
    tick_activity = {}
    try:
        stat_lines = Path(stat_path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as err:
        raise RuntimeError(f"Cannot read per-CPU activity from {stat_path}: {err}") from err

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
            raise RuntimeError(f"Malformed CPU activity row in {stat_path}: {line!r}")
        try:
            counters = [int(value) for value in fields[1:9]]
        except ValueError as err:
            raise RuntimeError(
                f"Malformed CPU activity row in {stat_path}: {line!r}"
            ) from err
        if any(value < 0 for value in counters):
            raise RuntimeError(f"Negative CPU activity counter in {stat_path}: {line!r}")

        idle = counters[3]
        iowait = counters[4]
        tick_activity[cpu] = (sum(counters) - idle - iowait, idle)

    missing_cpus = cpus - tick_activity.keys()
    if missing_cpus:
        raise RuntimeError(
            f"Missing CPU activity rows in {stat_path}: {sorted(missing_cpus)}"
        )
    return tick_activity


def _read_cpu_schedstat(cpus, schedstat_path):
    """Read high-resolution runtime and timeslice counts for requested CPUs."""
    try:
        schedstat_lines = Path(schedstat_path).read_text(
            encoding="ascii"
        ).splitlines()
    except (FileNotFoundError, PermissionError) as err:
        raise _ObserverUnavailable(
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
    if schedstat_version not in SUPPORTED_SCHEDSTAT_VERSIONS:
        raise _ObserverUnavailable(
            f"Unsupported schedstat version {schedstat_version} in {schedstat_path}"
        )

    activity = {}
    for line in schedstat_lines[1:]:
        fields = line.split()
        if (
                not fields
                or not fields[0].startswith("cpu")
                or not fields[0][3:].isdigit()):
            continue
        cpu_text = fields[0][3:]
        cpu = int(cpu_text)
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


def _read_cpu_activity(
        cpus, stat_path=CPU_STAT_PATH, schedstat_path=SCHEDSTAT_PATH):
    """Read tick and high-resolution activity for requested logical CPUs."""
    requested_cpus = set(cpus)
    sched_activity = _read_cpu_schedstat(requested_cpus, schedstat_path)
    tick_activity = _read_cpu_ticks(requested_cpus, stat_path)
    return {
        cpu: _CpuActivity(
            busy_ticks=tick_activity[cpu][0],
            idle_ticks=tick_activity[cpu][1],
            runtime_ns=sched_activity[cpu][0],
            timeslices=sched_activity[cpu][1],
        )
        for cpu in requested_cpus
    }


def _activity_delta(before, after, cpu):
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
    if any(after_value < before_value for before_value, after_value in zip(
            before_values, after_values)):
        raise RuntimeError(f"CPU {cpu} activity counters moved backwards")
    return _CpuActivity(
        busy_ticks=after_cpu.busy_ticks - before_cpu.busy_ticks,
        idle_ticks=after_cpu.idle_ticks - before_cpu.idle_ticks,
        runtime_ns=after_cpu.runtime_ns - before_cpu.runtime_ns,
        timeslices=after_cpu.timeslices - before_cpu.timeslices,
    )


def _cpu_stayed_quiet(before, after, cpu):
    """Return whether a CPU accrued idle time without task activity."""
    delta = _activity_delta(before, after, cpu)
    return (
        delta.runtime_ns == 0
        and delta.timeslices == 0
        and delta.idle_ticks > 0
    )


def _core_stayed_quiet(before, after, core):
    """Return whether both logical CPUs of an SMT core remained idle."""
    return all(
        _cpu_stayed_quiet(before, after, cpu) for cpu in core.siblings
    )


def _busy_wait(duration):
    """Keep the pinned waker running for one activity-observation window."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        pass


def _layout_cpus(layout):
    """Return every logical CPU whose state participates in the probe."""
    return {
        *layout.source.siblings,
        *(cpu for core in layout.candidates for cpu in core.siblings),
    }


def _qualify_trial_layout(layout, busy_cpu, previous_cpu):
    """Reserve a quiet hint-seed core and return quiet placement candidates."""
    observed_cpus = _layout_cpus(layout)
    activity_before = _read_cpu_activity(observed_cpus)
    _busy_wait(PROBE_SETTLE_SECONDS)
    activity_after = _read_cpu_activity(observed_cpus)

    busy_delta = _activity_delta(activity_before, activity_after, busy_cpu)
    if (
            busy_delta.busy_ticks == 0
            and busy_delta.runtime_ns == 0
            and busy_delta.timeslices == 0):
        raise RuntimeError(
            f"CPU activity accounting did not observe the waker on CPU {busy_cpu}"
        )
    if not _cpu_stayed_quiet(activity_before, activity_after, previous_cpu):
        raise _TrialContended(
            f"previous CPU {previous_cpu} did not remain idle before wakeup"
        )

    quiet_candidates = tuple(
        core for core in layout.candidates
        if _core_stayed_quiet(activity_before, activity_after, core)
    )
    if len(quiet_candidates) < 2:
        raise _TrialContended(
            "Fewer than two candidate SMT cores remained fully idle before wakeup"
        )

    hint_seed_core = quiet_candidates[-1]
    placement_candidates = quiet_candidates[:-1]
    return (
        _TrialLayout(
            candidates=placement_candidates,
            source=layout.source,
            hint_seed_core=hint_seed_core,
            hint_seed_cpu=hint_seed_core.siblings[0],
        ),
        activity_before,
    )


def _prime_idle_core_hint(busy_cpu, previous_cpu, seed_core):
    """Force one candidate core through busy-to-idle, then resume the waker.

    Moving away from busy_cpu lets its runqueue select the idle task. With
    previous_cpu already qualified idle, set_next_task_idle() runs
    update_idle_core() for the source domain. The seed core is excluded from
    wakee placement.
    """
    seed_cpu = seed_core.siblings[0]
    seed_sibling = seed_core.siblings[1]
    observed_cpus = {busy_cpu, previous_cpu, *seed_core.siblings}
    activity_before = _read_cpu_activity(observed_cpus)
    try:
        os.sched_setaffinity(0, {seed_cpu})
        if _get_current_cpu() != seed_cpu:
            raise RuntimeError(f"Waker did not reach idle-hint seed CPU {seed_cpu}")

        _busy_wait(PROBE_SETTLE_SECONDS)
        activity_after = _read_cpu_activity(observed_cpus)
        busy_delta = _activity_delta(activity_before, activity_after, busy_cpu)
        if busy_delta.idle_ticks == 0:
            raise _TrialContended(
                f"source CPU {busy_cpu} did not enter controlled scheduler idle"
            )
        if not _cpu_stayed_quiet(
                activity_before, activity_after, previous_cpu):
            raise _TrialContended(
                f"previous CPU {previous_cpu} was active during idle-hint priming"
            )
        if not _cpu_stayed_quiet(
                activity_before, activity_after, seed_sibling):
            raise _TrialContended(
                f"idle-hint seed CPU {seed_cpu} had activity on sibling "
                f"{seed_sibling} during priming"
            )
    finally:
        os.sched_setaffinity(0, {busy_cpu})
        if _get_current_cpu() != busy_cpu:
            raise RuntimeError(f"Waker did not return to busy CPU {busy_cpu}")


def _validate_trial_activity(
        layout, previous_cpu, destination_cpu, activity_before):
    """Reject a result if competing activity could have changed its verdict."""
    activity_after = _read_cpu_activity(activity_before)

    seed_sibling = next(
        cpu for cpu in layout.hint_seed_core.siblings
        if cpu != layout.hint_seed_cpu
    )
    if not _cpu_stayed_quiet(activity_before, activity_after, seed_sibling):
        raise _TrialContended(
            f"idle-hint seed CPU {layout.hint_seed_cpu} had activity on "
            f"sibling {seed_sibling}"
        )

    if destination_cpu == previous_cpu:
        if not any(
                _core_stayed_quiet(activity_before, activity_after, core)
                for core in layout.candidates):
            raise _TrialContended(
                "No candidate SMT core remained fully idle through wakeup"
            )
        return

    destination_cores = [
        core for core in layout.candidates
        if destination_cpu in core.siblings
    ]
    if len(destination_cores) != 1:
        raise RuntimeError(
            f"Cannot map destination CPU {destination_cpu} to one candidate core"
        )
    if not _cpu_stayed_quiet(activity_before, activity_after, previous_cpu):
        raise _TrialContended(
            f"previous CPU {previous_cpu} had competing activity during wakeup"
        )

    destination_core = destination_cores[0]
    other_sibling = next(
        cpu for cpu in destination_core.siblings if cpu != destination_cpu
    )
    if not _cpu_stayed_quiet(activity_before, activity_after, other_sibling):
        raise _TrialContended(
            f"candidate CPU {destination_cpu} had activity on sibling {other_sibling}"
        )


def _read_cpu_mask(path, allow_empty=False):
    """Read a Linux CPU-list sysfs file."""
    try:
        cpus = parse_cpu_mask(Path(path).read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError) as err:
        raise ValueError(f"Cannot read CPU mask from {path}: {err}") from err
    if not cpus and not allow_empty:
        raise ValueError(f"CPU mask is empty in {path}")
    return cpus


def _read_node_masks(node_root, online_cpus):
    """Return NUMA-node CPU masks, falling back to a single system mask."""
    node_masks = {}
    for node_path in sorted(Path(node_root).glob("node[0-9]*")):
        node_id_text = node_path.name.removeprefix("node")
        if not node_id_text.isdigit():
            continue
        node_cpus = (
            _read_cpu_mask(node_path / "cpulist", allow_empty=True) & online_cpus
        )
        if node_cpus:
            node_masks[int(node_id_text)] = node_cpus

    if not node_masks:
        node_masks[0] = set(online_cpus)
    return node_masks


def _read_positive_int(path):
    """Read a positive integer from a sysfs file."""
    try:
        value = int(Path(path).read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as err:
        raise ValueError(f"Cannot read a positive integer from {path}: {err}") from err
    if value <= 0:
        raise ValueError(f"Expected a positive integer in {path}, got {value}")
    return value


def _read_smt_cores(cpu_root, usable_cpus, node_masks):
    """Read complete, affinity-usable SMT2 cores and their capacities."""
    cores = []
    seen = set()

    for cpu in sorted(usable_cpus):
        topology_path = cpu_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        siblings = _read_cpu_mask(topology_path)
        if cpu not in siblings:
            raise ValueError(
                f"SMT topology {topology_path} does not contain CPU {cpu}: "
                f"{sorted(siblings)}"
            )
        if len(siblings) != 2 or not siblings.issubset(usable_cpus):
            continue

        sibling_tuple = tuple(sorted(siblings))
        if sibling_tuple in seen:
            continue

        for sibling in sibling_tuple:
            sibling_path = (
                cpu_root / f"cpu{sibling}" / "topology" / "thread_siblings_list"
            )
            sibling_mask = _read_cpu_mask(sibling_path)
            if sibling_mask != siblings:
                raise ValueError(
                    f"SMT topology {sibling_path} is not reciprocal for siblings "
                    f"{list(sibling_tuple)}: {sorted(sibling_mask)}"
                )
        seen.add(sibling_tuple)

        capacities = {
            _read_positive_int(cpu_root / f"cpu{sibling}" / "cpu_capacity")
            for sibling in siblings
        }
        if len(capacities) != 1:
            raise ValueError(
                f"SMT siblings {list(sibling_tuple)} expose different CPU capacities: "
                f"{sorted(capacities)}"
            )

        matching_nodes = [
            node for node, node_cpus in node_masks.items()
            if siblings.issubset(node_cpus)
        ]
        if len(matching_nodes) != 1:
            raise ValueError(
                f"Cannot place SMT siblings {list(sibling_tuple)} in one NUMA node"
            )

        cores.append(
            _SmtCore(
                capacity=capacities.pop(),
                node=matching_nodes[0],
                siblings=sibling_tuple,
            )
        )

    return cores


def _readline_with_timeout(stream, timeout, description):
    """Read one UTF-8 line from a pipe with a bounded wait."""
    deadline = time.monotonic() + timeout
    line = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Timed out waiting for wakee {description}")
        readable, _, _ = select.select([stream.fileno()], [], [], remaining)
        if not readable:
            raise RuntimeError(f"Timed out waiting for wakee {description}")

        byte = os.read(stream.fileno(), 1)
        if not byte:
            raise RuntimeError(
                f"Wakee closed its output before reporting {description}"
            )
        if byte == b"\n":
            break
        line.extend(byte)
        if len(line) > PROBE_LINE_MAX_BYTES:
            raise RuntimeError(f"Wakee returned an oversized {description}")
    try:
        return line.decode("ascii").strip()
    except UnicodeError as err:
        raise RuntimeError(f"Wakee returned non-ASCII {description}") from err


def _select_probe_layout(  # pylint: disable=too-many-return-statements
        cpu_root=CPU_SYSFS_ROOT, node_root=NODE_SYSFS_ROOT, affinity=None):
    """Return the layout-selection verdict, layout, and reason."""
    cpu_root = Path(cpu_root)
    smt_active_path = cpu_root / "smt" / "active"
    try:
        smt_active = smt_active_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as err:
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot read SMT state from {smt_active_path}: {err}",
        )

    if smt_active == "0":
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "SMT is not active; the SMT-aware scheduler path is not exercisable.",
        )
    if smt_active != "1":
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot interpret SMT active value {smt_active!r} from {smt_active_path}",
        )

    try:
        online_cpus = _read_cpu_mask(cpu_root / "online")
    except ValueError as err:
        return GLTCore.GLTTestResult.FAIL, None, str(err)
    if affinity is None:
        try:
            affinity = os.sched_getaffinity(0)
        except OSError as err:
            return (
                GLTCore.GLTTestResult.FAIL,
                None,
                f"Cannot read the GreenLiT process CPU affinity: {err}",
            )
    usable_cpus = online_cpus & set(affinity)

    try:
        node_masks = _read_node_masks(node_root, online_cpus)
        cores = _read_smt_cores(cpu_root, usable_cpus, node_masks)
    except ValueError as err:
        return GLTCore.GLTTestResult.FAIL, None, str(err)
    if len(cores) < 2:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "Fewer than two complete online SMT2 cores are available in the "
            "current process CPU affinity.",
        )

    layouts = []
    for node in sorted(node_masks):
        layout = _candidate_layout_for_node(
            [core for core in cores if core.node == node]
        )
        if layout is not None:
            layouts.append(layout)

    if not layouts:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "No NUMA node has complete affinity-usable SMT2 cores with distinct "
            "CPU capacity values and at least two lower-capacity candidate cores.",
        )

    layout = max(layouts, key=lambda item: (len(item.candidates), -item.source.node))
    return GLTCore.GLTTestResult.PASS, layout, ""


def _stop_wakee(process):
    """Close and reap a wakee process, terminating it only when necessary."""
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()

    if process.poll() is not None:
        process.wait()
        return

    process.terminate()
    try:
        process.wait(timeout=PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=PROBE_TIMEOUT_SECONDS)


def _wait_until_sleeping(pid, timeout=PROBE_TIMEOUT_SECONDS):
    """Wait until the wakee is blocked on its input pipe."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _process_state(pid)
        if state == "S":
            return
        if state in {"X", "Z"}:
            raise RuntimeError(f"Wakee exited before blocking (state {state})")
        time.sleep(0.01)
    raise RuntimeError("Timed out waiting for wakee to block on its input pipe")


def _cleanup_wakeup_trial(process, original_affinity):
    """Restore the caller affinity and reap the child, or raise RuntimeError."""
    cleanup_errors = []
    try:
        os.sched_setaffinity(0, original_affinity)
    except OSError as err:
        cleanup_errors.append(f"restore GreenLiT process affinity: {err}")

    if process is not None:
        try:
            _stop_wakee(process)
        except (OSError, subprocess.SubprocessError) as err:
            cleanup_errors.append(f"stop wakee process: {err}")

    if cleanup_errors:
        raise RuntimeError(
            "Failed to clean up controlled scheduler wakeup: "
            + "; ".join(cleanup_errors)
        )


def _run_wakeup_trial(layout, busy_cpu, previous_cpu):
    """Wake one task beside a busy sibling and return its selected CPU."""
    try:
        original_affinity = os.sched_getaffinity(0)
    except OSError as err:
        raise RuntimeError(
            f"Cannot read the GreenLiT process CPU affinity: {err}"
        ) from err
    if not WAKEE_SCRIPT_PATH.is_file():
        raise RuntimeError(
            f"Installed SMT-aware wakee script is missing: {WAKEE_SCRIPT_PATH}"
        )
    process = None

    try:
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, str(WAKEE_SCRIPT_PATH), str(previous_cpu)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stdin is None:
            raise RuntimeError("Failed to create wakee communication pipes")

        ready = _readline_with_timeout(
            process.stdout, PROBE_TIMEOUT_SECONDS, "readiness"
        )
        if ready != "READY":
            raise RuntimeError(f"Unexpected wakee readiness marker: {ready!r}")

        _wait_until_sleeping(process.pid)
        os.sched_setaffinity(0, {busy_cpu})
        if _get_current_cpu() != busy_cpu:
            raise RuntimeError(f"Waker did not reach requested busy CPU {busy_cpu}")

        trial_layout, activity_before = _qualify_trial_layout(
            layout, busy_cpu, previous_cpu
        )
        wakee_affinity = {previous_cpu, *trial_layout.candidate_cpus}
        os.sched_setaffinity(process.pid, wakee_affinity)
        if os.sched_getaffinity(process.pid) != wakee_affinity:
            raise RuntimeError("Kernel did not apply the requested wakee CPU affinity")

        _prime_idle_core_hint(
            busy_cpu, previous_cpu, trial_layout.hint_seed_core
        )
        process.stdin.write(b"W")
        process.stdin.flush()

        destination_text = _readline_with_timeout(
            process.stdout, PROBE_TIMEOUT_SECONDS, "destination CPU"
        )
        try:
            destination_cpu = int(destination_text)
        except ValueError as err:
            raise RuntimeError(
                f"Wakee returned invalid destination CPU {destination_text!r}"
            ) from err

        if destination_cpu not in wakee_affinity:
            raise RuntimeError(
                f"Wakee ran on CPU {destination_cpu} outside requested affinity "
                f"{sorted(wakee_affinity)}"
            )
        _validate_trial_activity(
            trial_layout, previous_cpu, destination_cpu, activity_before
        )
        return _WakeupTrial(
            candidate_cpus=trial_layout.candidate_cpus,
            destination_cpu=destination_cpu,
            hint_seed_cpu=trial_layout.hint_seed_cpu,
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise RuntimeError(f"Unable to execute controlled scheduler wakeup: {err}") from err
    finally:
        _cleanup_wakeup_trial(process, original_affinity)


def _run_wakeup_probe(layout, logger):
    """Collect consistent uncontaminated trials in both sibling directions."""
    results = {}
    first_cpu, second_cpu = layout.source.siblings

    for busy_cpu, previous_cpu in (
            (first_cpu, second_cpu), (second_cpu, first_cpu)):
        destinations = []
        for trial in range(1, PROBE_MAX_TRIALS_PER_SIBLING + 1):
            try:
                observation = _run_wakeup_trial(
                    layout, busy_cpu, previous_cpu
                )
            except _TrialContended as err:
                logger.info(
                    "Discarding contended wakeup trial %d/%d: busy_cpu=%d "
                    "previous_cpu=%d reason=%s",
                    trial,
                    PROBE_MAX_TRIALS_PER_SIBLING,
                    busy_cpu,
                    previous_cpu,
                    err,
                )
                continue

            destinations.append(observation.destination_cpu)
            logger.info(
                "SMT-aware clean wakeup %d/%d (trial %d/%d): "
                "busy_cpu=%d previous_cpu=%d "
                "idle_hint_seed_cpu=%d destination_cpu=%d candidate_cpus=%s",
                len(destinations),
                PROBE_ATTEMPTS_PER_SIBLING,
                trial,
                PROBE_MAX_TRIALS_PER_SIBLING,
                busy_cpu,
                previous_cpu,
                observation.hint_seed_cpu,
                observation.destination_cpu,
                list(observation.candidate_cpus),
            )
            if len(destinations) == PROBE_ATTEMPTS_PER_SIBLING:
                break
        if len(destinations) != PROBE_ATTEMPTS_PER_SIBLING:
            raise _ProbeInconclusive(
                f"Collected only {len(destinations)}/"
                f"{PROBE_ATTEMPTS_PER_SIBLING} uncontaminated wakeups for "
                f"busy CPU {busy_cpu} and previous CPU {previous_cpu} after "
                f"{PROBE_MAX_TRIALS_PER_SIBLING} trials"
            )
        results[previous_cpu] = tuple(destinations)

    return results


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_smt_aware_asym_capacity_wakeup(_, logger):
    """Verify Vera wakeups prefer fully idle SMT cores over busy siblings.

    JIRA: DGX-17050
    PATCH: 25a32e400a14 sched/fair: Prefer fully-idle SMT cores in
           asym-capacity idle selection
    MAINLINE: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=25a32e400a14

    SERIES COVERAGE:
      - c9d93a73ce87: NOHZ RCU-lock cleanup with no intended functional change;
        this placement test does not cover it.
      - fdfe5a8cd873: scopes sched_domain_shared to the asymmetric-capacity
        domain; it is required for correctly scoped idle-core hints. The probe
        primes that hint with a controlled busy-to-idle core transition but
        does not independently distinguish this prerequisite.
      - 25a32e400a14: the targeted fully-idle-core wakeup behavior is directly
        covered by the controlled placement oracle.
      - bf6aa722198d: changes load-balancing-time misfit pulls rather than
        synchronous wakeup placement; it needs a separate reproducer with a
        positive load-balance control.
      - 61ea17a63719: bounds SIS_UTIL idle-capacity scans and can change scan cost
        without changing the selected CPU; this oracle does not force its
        no-idle-core, exhausted-budget condition.
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not a Vera CPU; the SMT-aware scheduling test is Vera-specific."
        )

    layout_verdict, layout, layout_reason = _select_probe_layout()
    if layout_verdict == GLTCore.GLTTestResult.SKIP:
        GLTCore.GLTTestResult.skip(layout_reason)
    if layout_verdict == GLTCore.GLTTestResult.FAIL:
        GLTCore.GLTTestResult.fail(
            f"Cannot inspect Vera SMT/capacity topology: {layout_reason}"
        )
    if layout_verdict != GLTCore.GLTTestResult.PASS or layout is None:
        GLTCore.GLTTestResult.fail(
            f"Unexpected topology-selection verdict: {layout_verdict}"
        )

    logger.info(
        "Vera SMT-aware wakeup layout: node=%d source_core=%s "
        "source_capacity=%d candidate_cpus=%s candidate_capacities=%s",
        layout.source.node,
        list(layout.source.siblings),
        layout.source.capacity,
        list(layout.candidate_cpus),
        [core.capacity for core in layout.candidates],
    )

    try:
        _read_cpu_activity(_layout_cpus(layout))
        results = _run_wakeup_probe(layout, logger)
    except _ObserverUnavailable as err:
        GLTCore.GLTTestResult.skip(
            f"Required scheduler activity observer is unavailable: {err}"
        )
    except _ProbeInconclusive as err:
        GLTCore.GLTTestResult.skip(
            f"Controlled scheduler wakeup probe was inconclusive: {err}"
        )
    except RuntimeError as err:
        GLTCore.GLTTestResult.fail(f"Controlled scheduler wakeup probe failed: {err}")

    placement_verdict, placement_reason = _placement_result(layout, results)
    if placement_verdict == GLTCore.GLTTestResult.WARN:
        GLTCore.GLTTestResult.warn(placement_reason)
    if placement_verdict == GLTCore.GLTTestResult.FAIL:
        GLTCore.GLTTestResult.fail(placement_reason)
    if placement_verdict != GLTCore.GLTTestResult.PASS:
        GLTCore.GLTTestResult.fail(
            f"Unexpected placement verdict: {placement_verdict}"
        )

    logger.info(
        "PASS: Vera wakeups selected fully idle SMT cores in both sibling directions: %s",
        results,
    )
