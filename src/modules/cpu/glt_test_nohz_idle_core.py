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

"""Verify NOHZ idle load balancing prefers fully idle SMT cores."""

from dataclasses import dataclass
import gzip
import os
from pathlib import Path
import platform
import re
import select
import signal
import subprocess
import sys
import time

from common import glt_core as GLTCore
from common.cpu.activity import (
    ObserverUnavailable,
    activity_delta,
    cpu_stayed_quiet,
    read_cpu_activity,
    read_sched_domain_cpus,
    read_tick_states,
)
from common.cpu_topology import parse_cpu_mask
from scripts import check_kernel_config


BOOT_CONFIG_ROOT = Path("/boot")
CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
CSD_NOHZ_LINE_RE = re.compile(
    r"\[(?P<src>\d+)\]\s+\S+\s+\S+:\s+csd_queue_cpu:\s+"
    r"cpu=(?P<dst>\d+)\s+callsite=\S+\s+func=nohz_csd_func\b"
)
CSD_QUEUE_REL = Path("events/csd/csd_queue_cpu")
KSTRTOBOOL_FALSE = frozenset({"0", "f", "false", "n", "no", "off"})
MAX_TRIALS = 15
MIN_CONCLUSIVE_TRIALS = 3
PRECONDITION_WINDOW_SECONDS = 0.10
PROBE_POLL_SECONDS = 0.005
PROBE_SETTLE_SECONDS = 0.05
PROBE_WINDOW_SECONDS = 0.15
PROC_CMDLINE_PATH = Path("/proc/cmdline")
PROC_CONFIG_GZ_PATH = Path("/proc/config.gz")
PROC_SCHEDSTAT_PATH = Path("/proc/schedstat")
REQUIRED_KCONFIGS = ("CONFIG_NO_HZ_COMMON", "CONFIG_KALLSYMS")
SPINNER_TIMEOUT_SECONDS = 3.0
TRACE_BUFFER_SIZE_KB = 4096
TRACING_ROOTS = (
    Path("/sys/kernel/tracing"),
    Path("/sys/kernel/debug/tracing"),
)


@dataclass(frozen=True)
class _NohzSmtLayout:
    """Lowest housekeeping SMT core (idle thread + trigger sibling) and idle cores."""

    first_idle_cpu: int
    trigger_cpu: int
    partial_core: tuple[int, ...]
    idle_cores: tuple[tuple[int, ...], ...]

    @property
    def observed_cpus(self):
        """Return every CPU whose idle or busy state participates in the trial."""
        return {
            self.first_idle_cpu,
            self.trigger_cpu,
            *(cpu for core in self.idle_cores for cpu in core),
        }


@dataclass(frozen=True)
class _TrialResult:
    """Classified outcome of one controlled NOHZ ILB first-kick observation."""

    conclusive_target: int | None = None
    failed_partial_cpu: int | None = None
    failed_partial_core: tuple[int, ...] | None = None
    observer_gap: bool = False
    reason: str = ""


def _check_observer_kconfig(
    boot_root=BOOT_CONFIG_ROOT,
    proc_config_gz=PROC_CONFIG_GZ_PATH,
    release=None,
):
    """Validate required observer Kconfig options, failing closed if unreadable."""
    kernel_release = platform.release() if release is None else release
    config_path = boot_root / f"config-{kernel_release}"
    config = None
    config_label = str(config_path)
    try:
        config = check_kernel_config.parse_kconfig(str(config_path))
    except FileNotFoundError:
        try:
            with gzip.open(proc_config_gz, "rt", encoding="latin-1") as stream:
                config = {}
                for line in stream:
                    match = re.match(r"(CONFIG_\w+)=(.*)", line.strip())
                    if match:
                        config[match.group(1)] = match.group(2).strip()
            config_label = str(proc_config_gz)
        except FileNotFoundError:
            return (
                GLTCore.GLTTestResult.FAIL,
                f"Cannot read running kernel config from {config_path} or "
                f"{proc_config_gz}: neither file exists.",
            )
        except (OSError, EOFError, ValueError, gzip.BadGzipFile) as error:
            return (
                GLTCore.GLTTestResult.FAIL,
                f"Cannot parse running kernel config {proc_config_gz}: {error}",
            )
    except (OSError, ValueError) as error:
        return (
            GLTCore.GLTTestResult.FAIL,
            f"Cannot parse running kernel config {config_path}: {error}",
        )

    missing = [
        f"{option}=y"
        for option in REQUIRED_KCONFIGS
        if config.get(option) != "y"
    ]
    if missing:
        return (
            GLTCore.GLTTestResult.SKIP,
            f"NOHZ ILB trace observer requires {', '.join(missing)} in {config_label}.",
        )
    return GLTCore.GLTTestResult.PASS, ""


def _classify_kick_targets(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    layout, pre_idle_cores, after, post, after_tick, post_tick, targets
):
    """Classify the first kick target against pre-window and post-probe state."""
    if not targets:
        return _TrialResult(
            observer_gap=True,
            reason="captured no nohz_csd_func kicks.",
        )

    kicked_cpus = set(targets)
    first_target = targets[0]
    trigger_busy = _cpu_stayed_busy(after, post, layout.trigger_cpu)

    if first_target in layout.partial_core:
        witness_idle = any(
            _core_stayed_task_idle(after, post, core)
            and all(
                _cpu_tick_stayed_stopped(after_tick, post_tick, cpu)
                for cpu in core
            )
            for core in pre_idle_cores
        )
        if trigger_busy and witness_idle:
            return _TrialResult(
                failed_partial_cpu=first_target,
                failed_partial_core=layout.partial_core,
            )
        reasons = []
        if not trigger_busy:
            reasons.append(
                f"trigger sibling CPU {layout.trigger_cpu} did not stay "
                "continuously busy across the probe."
            )
        if not witness_idle:
            reasons.append(
                "no pre-window-proven SMT core remained NOHZ tick-stopped "
                "and task-idle across the probe."
            )
        return _TrialResult(
            reason=(
                f"first kick targeted partially busy core CPU {first_target}, "
                f"but witness state was not sustained: {' '.join(reasons)}"
            ),
        )

    target_cores = [core for core in pre_idle_cores if first_target in core]
    if not target_cores:
        return _TrialResult(
            reason=(
                f"first kick target CPU {first_target} was not on a core "
                "proven fully idle in the pre-window."
            ),
        )

    post_reasons = []
    if not trigger_busy:
        post_reasons.append(
            f"trigger sibling CPU {layout.trigger_cpu} did not stay "
            "continuously busy across the probe."
        )
    first_idle_tick_ok = (
        layout.first_idle_cpu in kicked_cpus
        or (
            activity_delta(after, post, layout.first_idle_cpu).busy_ticks == 0
            and _cpu_tick_stayed_stopped(
                after_tick, post_tick, layout.first_idle_cpu
            )
        )
    )
    if not _cpu_stayed_task_idle(after, post, layout.first_idle_cpu) or not first_idle_tick_ok:
        post_reasons.append(
            f"partial-core idle thread CPU {layout.first_idle_cpu} had competing "
            "activity or exited NOHZ tick-stopped idle across the probe."
        )
    target_core = target_cores[0]
    if not all(
        _cpu_stayed_task_idle(after, post, cpu)
        for cpu in target_core
        if cpu != first_target
    ):
        post_reasons.append(
            f"sibling of target CPU {first_target} woke a task across the probe."
        )
    if post_reasons:
        return _TrialResult(reason=" ".join(post_reasons))

    return _TrialResult(conclusive_target=first_target)


def _cleanup_trial(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    procs,
    gate_fds,
    original_affinity,
    affinity_changed,
    instance_dir,
    instance_created,
    event_enabled,
):
    """Restore caller affinity, reap children, and remove the tracefs instance."""
    cleanup_errors = []
    for fd in gate_fds:
        if fd is None:
            continue
        try:
            os.close(fd)
        except OSError as error:
            cleanup_errors.append(f"close spinner gate fd: {error}")

    cleanup_errors.extend(_stop_spinners(procs))

    if affinity_changed and original_affinity is not None:
        try:
            os.sched_setaffinity(0, original_affinity)
        except OSError as error:
            cleanup_errors.append(
                f"restore GreenLiT process affinity: {error}"
            )

    if event_enabled:
        for rel in (Path("tracing_on"), CSD_QUEUE_REL / "enable"):
            try:
                (instance_dir / rel).write_text("0\n", encoding="utf-8")
            except OSError as error:
                cleanup_errors.append(f"disable {instance_dir / rel}: {error}")

    if instance_created:
        try:
            instance_dir.rmdir()
        except OSError as error:
            cleanup_errors.append(
                f"remove tracefs instance {instance_dir}: {error}"
            )

    if cleanup_errors:
        raise RuntimeError(
            "Failed to clean up NOHZ ILB trial: " + "; ".join(cleanup_errors)
        )


def _core_stayed_task_idle(before, after, core):
    """Return whether every logical CPU in an SMT core ran zero tasks."""
    return all(_cpu_stayed_task_idle(before, after, cpu) for cpu in core)


def _cpu_stayed_busy(before, after, cpu):
    """Return whether a pinned spinner kept a CPU continuously out of idle."""
    delta = activity_delta(before, after, cpu)
    return delta.idle_ticks == 0 and (
        delta.busy_ticks > 0 or delta.runtime_ns > 0 or delta.timeslices > 0
    )


def _cpu_tick_stayed_stopped(before_tick, after_tick, cpu):
    """Return whether a CPU's NOHZ tick stayed continuously stopped."""
    tick_start = before_tick.get(cpu)
    tick_end = after_tick.get(cpu)
    if tick_start is None or tick_end is None:
        return False
    return (
        tick_start.nohz_enabled
        and tick_end.nohz_enabled
        and tick_start.tick_stopped
        and tick_end.tick_stopped
        and tick_end.last_tick_ns == tick_start.last_tick_ns
    )


def _cpu_stayed_nohz_eligible(
    before_act, after_act, before_tick, after_tick, cpu
):
    """Return whether a CPU stayed idle with its NOHZ tick continuously stopped."""
    return (
        cpu_stayed_quiet(before_act, after_act, cpu)
        and activity_delta(before_act, after_act, cpu).busy_ticks == 0
        and _cpu_tick_stayed_stopped(before_tick, after_tick, cpu)
    )


def _cpu_stayed_task_idle(before, after, cpu):
    """Return whether a CPU scheduled zero task runtime or timeslices."""
    delta = activity_delta(before, after, cpu)
    return delta.runtime_ns == 0 and delta.timeslices == 0


def _evaluate_precondition(
    layout, before_act, after_act, before_tick, after_tick
):
    """Return whether the pre-window established a discriminating NOHZ state."""
    idle_thread_ready = _cpu_stayed_nohz_eligible(
        before_act, after_act, before_tick, after_tick, layout.first_idle_cpu
    )
    trigger_busy = _cpu_stayed_busy(before_act, after_act, layout.trigger_cpu)
    pre_idle_cores = tuple(
        core
        for core in layout.idle_cores
        if all(
            _cpu_stayed_nohz_eligible(
                before_act, after_act, before_tick, after_tick, cpu
            )
            for cpu in core
        )
    )
    reasons = []
    if not idle_thread_ready:
        reasons.append(
            f"partial-core idle thread CPU {layout.first_idle_cpu} was not "
            "continuously NOHZ tick-stopped and idle."
        )
    if not trigger_busy:
        reasons.append(
            f"trigger sibling CPU {layout.trigger_cpu} did not stay "
            "continuously busy across the pre-window."
        )
    if not pre_idle_cores:
        reasons.append(
            "no candidate SMT core stayed fully NOHZ tick-stopped and idle "
            "across the pre-window."
        )
    precondition_ok = (
        idle_thread_ready and trigger_busy and bool(pre_idle_cores)
    )
    return precondition_ok, pre_idle_cores, " ".join(reasons)


def _format_tracing_cpumask(cpus):
    """Format a CPU set as comma-separated 32-bit hex bitmap words."""
    mask = 0
    for cpu in cpus:
        if cpu < 0:
            raise ValueError(f"Negative CPU ID in tracing cpumask: {cpu}")
        mask |= 1 << cpu
    if mask == 0:
        raise ValueError("Tracing cpumask must not be empty")
    words = []
    while mask:
        words.append(mask & 0xFFFFFFFF)
        mask >>= 32
    words.reverse()
    return ",".join([f"{words[0]:x}", *(f"{word:08x}" for word in words[1:])])


def _is_nohz_disabled_on_cmdline(cmdline_text):
    """Return whether the effective last ``nohz=`` parameter disables NOHZ."""
    disabled = False
    for token in cmdline_text.split():
        if not token.startswith("nohz="):
            continue
        value = token.partition("=")[2].strip().lower()
        disabled = value in KSTRTOBOOL_FALSE
    return disabled


def _parse_nohz_kick_targets(trace_text, trigger_cpu):
    """Return destination CPUs for nohz_csd_func queued from trigger_cpu."""
    targets = []
    for line in trace_text.splitlines():
        match = CSD_NOHZ_LINE_RE.search(line)
        if match is None:
            continue
        if int(match.group("src")) == trigger_cpu:
            targets.append(int(match.group("dst")))
    return targets


def _parse_optional_cpu_mask(path):
    """Read an optional sysfs CPU list, propagating non-absence I/O errors."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not raw or raw == "(null)":
        return set()
    return parse_cpu_mask(raw)


def _run_nohz_ilb_trial(layout, tracing_root):
    """Run one controlled NOHZ ILB probe and judge the first kick target."""
    try:
        original_affinity = os.sched_getaffinity(0)
    except OSError as error:
        raise RuntimeError(
            f"Cannot read process CPU affinity: {error}"
        ) from error

    instance_dir = tracing_root / "instances" / f"glt_nohz_ilb_{os.getpid()}"
    tracing_on_path = instance_dir / "tracing_on"
    cpumask_path = instance_dir / "tracing_cpumask"
    buffer_size_path = instance_dir / "buffer_size_kb"
    enable_path = instance_dir / CSD_QUEUE_REL / "enable"
    trace_path = instance_dir / "trace"
    stats_path = (
        instance_dir / "per_cpu" / f"cpu{layout.trigger_cpu}" / "stats"
    )

    procs = []
    gate_read_fd = None
    gate_write_fd = None
    affinity_changed = False
    instance_created = False
    event_enabled = False

    try:
        gate_read_fd, gate_write_fd = os.pipe()
        instance_dir.mkdir(exist_ok=False)
        instance_created = True

        tracing_on_path.write_text("0\n", encoding="utf-8")
        cpumask_path.write_text(
            f"{_format_tracing_cpumask({layout.trigger_cpu})}\n",
            encoding="utf-8",
        )
        applied_mask = int(
            cpumask_path.read_text(encoding="utf-8").strip().replace(",", ""),
            16,
        )
        if applied_mask != (1 << layout.trigger_cpu):
            raise RuntimeError(
                f"Kernel did not apply tracing_cpumask for CPU {layout.trigger_cpu}"
            )
        buffer_size_path.write_text(
            f"{TRACE_BUFFER_SIZE_KB}\n", encoding="utf-8"
        )
        enable_path.write_text("1\n", encoding="utf-8")
        event_enabled = True
        trace_path.write_text("\n", encoding="utf-8")

        os.sched_setaffinity(0, {layout.trigger_cpu})
        affinity_changed = True

        procs.append(_spawn_spinner(layout.trigger_cpu))
        procs.append(
            _spawn_spinner(layout.trigger_cpu, gate_read_fd=gate_read_fd)
        )
        os.close(gate_read_fd)
        gate_read_fd = None

        time.sleep(PROBE_SETTLE_SECONDS)
        if any(proc.poll() is not None for proc in procs):
            raise RuntimeError("Spinner exited before pre-window")

        tick_cpus = {
            layout.first_idle_cpu,
            *(cpu for core in layout.idle_cores for cpu in core),
        }
        before_tick = read_tick_states(tick_cpus)
        before_act = read_cpu_activity(layout.observed_cpus)
        time.sleep(PRECONDITION_WINDOW_SECONDS)
        if any(proc.poll() is not None for proc in procs):
            raise RuntimeError("Spinner exited during pre-window")
        after_act = read_cpu_activity(layout.observed_cpus)
        after_tick = read_tick_states(tick_cpus)

        precondition_ok, pre_idle_cores, reason = _evaluate_precondition(
            layout, before_act, after_act, before_tick, after_tick
        )
        if not precondition_ok:
            return _TrialResult(reason=reason)

        trace_path.write_text("\n", encoding="utf-8")
        tracing_on_path.write_text("1\n", encoding="utf-8")
        try:
            os.close(gate_write_fd)
            gate_write_fd = None
            deadline = time.monotonic() + PROBE_WINDOW_SECONDS
            while time.monotonic() < deadline:
                time.sleep(PROBE_POLL_SECONDS)
                probe_trace = trace_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                if _parse_nohz_kick_targets(probe_trace, layout.trigger_cpu):
                    break
        finally:
            tracing_on_path.write_text("0\n", encoding="utf-8")

        if any(proc.poll() is not None for proc in procs):
            raise RuntimeError("Spinner exited during probe window")

        post_act = read_cpu_activity(layout.observed_cpus)
        post_tick = read_tick_states(tick_cpus)
        stats_text = stats_path.read_text(encoding="utf-8")
        trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
        if _trace_lost_events(stats_text, trace_text):
            return _TrialResult(
                observer_gap=True,
                reason="trace buffer reported lost or overrun events.",
            )

        targets = _parse_nohz_kick_targets(trace_text, layout.trigger_cpu)
        return _classify_kick_targets(
            layout,
            pre_idle_cores,
            after_act,
            post_act,
            after_tick,
            post_tick,
            targets,
        )
    finally:
        _cleanup_trial(
            procs,
            (gate_read_fd, gate_write_fd),
            original_affinity,
            affinity_changed,
            instance_dir,
            instance_created,
            event_enabled,
        )


def _select_nohz_smt_layout(  # pylint: disable=too-many-branches,too-many-return-statements
    cpu_root=CPU_SYSFS_ROOT,
    cmdline_path=PROC_CMDLINE_PATH,
    schedstat_path=PROC_SCHEDSTAT_PATH,
    affinity=None,
):
    """Select the lowest housekeeping SMT core and candidate idle cores."""
    smt_active_path = cpu_root / "smt" / "active"
    try:
        smt_active = smt_active_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            f"{smt_active_path} is absent; SMT is not supported.",
        )
    except (OSError, UnicodeError) as error:
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot read {smt_active_path}: {error}",
        )
    if smt_active == "0":
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            f"SMT is not active ({smt_active_path}={smt_active!r}).",
        )
    if smt_active != "1":
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot interpret SMT active value {smt_active!r} from {smt_active_path}.",
        )

    try:
        cmdline_text = cmdline_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot read {cmdline_path}: {error}",
        )
    if _is_nohz_disabled_on_cmdline(cmdline_text):
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "NOHZ idle tick stopping is disabled on the kernel command line.",
        )

    nohz_full_path = cpu_root / "nohz_full"
    try:
        nohz_full_cpus = _parse_optional_cpu_mask(nohz_full_path)
    except (OSError, UnicodeError, ValueError) as error:
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot read {nohz_full_path}: {error}",
        )
    if nohz_full_cpus:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "nohz_full is active; tick_nohz_full_enabled() keeps the timekeeping "
            "CPU ticking and excludes nohz_full CPUs from NOHZ ILB selection.",
        )

    online_path = cpu_root / "online"
    try:
        online_cpus = parse_cpu_mask(online_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot read online CPU mask from {online_path}: {error}",
        )
    if not online_cpus:
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Online CPU mask is empty in {online_path}.",
        )

    isolated_path = cpu_root / "isolated"
    housekeeping_path = cpu_root / "housekeeping"
    try:
        isolated_cpus = _parse_optional_cpu_mask(isolated_path) or set()
        housekeeping_cpus = _parse_optional_cpu_mask(housekeeping_path)
    except (OSError, UnicodeError, ValueError) as error:
        return (
            GLTCore.GLTTestResult.FAIL,
            None,
            f"Cannot read CPU isolation/housekeeping masks: {error}",
        )

    try:
        domain_cpus = read_sched_domain_cpus(schedstat_path)
    except ObserverUnavailable as error:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            f"Required scheduler activity observer is unavailable: {error}",
        )
    except RuntimeError as error:
        return GLTCore.GLTTestResult.FAIL, None, str(error)

    hk_online_cpus = (online_cpus - isolated_cpus) & domain_cpus
    if housekeeping_cpus:
        hk_online_cpus &= housekeeping_cpus
    if not hk_online_cpus:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "No online load-balanced housekeeping CPUs are available for NOHZ ILB testing.",
        )

    if affinity is None:
        try:
            affinity = os.sched_getaffinity(0)
        except OSError as error:
            return (
                GLTCore.GLTTestResult.FAIL,
                None,
                f"Cannot read process CPU affinity: {error}",
            )

    usable_cpus = sorted(hk_online_cpus & set(affinity))
    if len(usable_cpus) < 4:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "At least 4 usable load-balanced housekeeping CPUs (online, "
            "schedulable, attached to a sched_domain, and not in nohz_full or "
            "isolated) are required for NOHZ SMT testing.",
        )

    usable_set = set(usable_cpus)
    seen_cores = []
    seen_members = set()
    for cpu in usable_cpus:
        if cpu in seen_members:
            continue
        siblings_path = (
            cpu_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        )
        try:
            all_siblings = (
                parse_cpu_mask(siblings_path.read_text(encoding="utf-8"))
                & online_cpus
            )
        except (OSError, UnicodeError, ValueError) as error:
            return (
                GLTCore.GLTTestResult.FAIL,
                None,
                f"Cannot read {siblings_path}: {error}",
            )
        if cpu not in all_siblings:
            return (
                GLTCore.GLTTestResult.FAIL,
                None,
                f"SMT topology {siblings_path} does not contain CPU {cpu}.",
            )
        seen_members.update(all_siblings)
        if len(all_siblings) >= 2 and all_siblings <= usable_set:
            seen_cores.append(tuple(sorted(all_siblings)))

    seen_cores.sort()
    lowest_hk_cpu = min(hk_online_cpus)
    if len(seen_cores) < 2 or seen_cores[0][0] != lowest_hk_cpu:
        return (
            GLTCore.GLTTestResult.SKIP,
            None,
            "At least 2 complete usable housekeeping SMT cores starting at the "
            "lowest load-balanced housekeeping CPU are required to separate "
            "the partially busy core and a fully idle target core.",
        )

    partial_core = seen_cores[0]
    idle_cores = tuple(seen_cores[1:])
    return (
        GLTCore.GLTTestResult.PASS,
        _NohzSmtLayout(
            first_idle_cpu=partial_core[0],
            trigger_cpu=partial_core[1],
            partial_core=partial_core,
            idle_cores=idle_cores,
        ),
        "",
    )


def _spawn_spinner(cpu, gate_read_fd=None):
    """Start a tight CPU-bound Python child pinned to ``cpu``."""
    gate_snippet = (
        f"os.read({int(gate_read_fd)}, 1)\n"
        f"os.close({int(gate_read_fd)})\n"
        if gate_read_fd is not None
        else ""
    )
    code = (
        "import os, signal\n"
        "signal.signal(signal.SIGTERM, lambda *_: os._exit(0))\n"
        f"os.sched_setaffinity(0, {{{int(cpu)}}})\n"
        "os.write(1, b'READY\\n')\n"
        f"{gate_snippet}"
        "while True:\n"
        "    pass\n"
    )
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(gate_read_fd,) if gate_read_fd is not None else (),
    )
    try:
        if proc.stdout is None:
            raise RuntimeError("Failed to create spinner stdout pipe")
        readable, _, _ = select.select(
            [proc.stdout.fileno()], [], [], SPINNER_TIMEOUT_SECONDS
        )
        if not readable:
            raise RuntimeError(
                f"Timed out waiting for spinner readiness on CPU {cpu}"
            )
        ready = os.read(proc.stdout.fileno(), 16)
        if ready != b"READY\n":
            stderr_bytes = b""
            if proc.stderr is not None and proc.poll() is not None:
                stderr_bytes = os.read(proc.stderr.fileno(), 512)
            detail = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Failed to start spinner on CPU {cpu}: {detail or ready!r}"
            )
    except BaseException:
        _stop_spinners([proc])
        raise
    return proc


def _stop_spinners(procs):
    """Terminate all spinner subprocesses and return any cleanup errors."""
    errors = []
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
    for proc in procs:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        try:
            proc.wait(timeout=SPINNER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=SPINNER_TIMEOUT_SECONDS)
            except (subprocess.TimeoutExpired, OSError) as error:
                errors.append(f"reap spinner pid {proc.pid}: {error}")
    return errors


def _trace_lost_events(stats_text, trace_text):
    """Return whether per-CPU trace stats or trace output report lost events."""
    if "[LOST" in trace_text:
        return True
    seen_keys = set()
    for line in stats_text.splitlines():
        key, sep, raw_value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key in {"overrun", "commit overrun", "dropped events"}:
            value = int(raw_value.strip())
            if value < 0:
                raise ValueError(f"Negative trace stat {key!r}: {value}")
            seen_keys.add(key)
            if value > 0:
                return True
    if not {"overrun", "commit overrun"} <= seen_keys:
        raise ValueError("Missing overrun counters in per-CPU trace stats")
    return False


def _tracing_root(candidates=TRACING_ROOTS):
    """Return the mounted tracefs root that exposes csd_queue_cpu, or None."""
    for root in candidates:
        if (
            (root / CSD_QUEUE_REL / "enable").is_file()
            and (root / "trace").is_file()
            and (root / "tracing_cpumask").is_file()
            and (root / "buffer_size_kb").is_file()
            and (root / "instances").is_dir()
        ):
            return root
    return None


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nohz_ilb_prefers_fully_idle_smt_core(  # pylint: disable=too-many-branches
    _, logger
):
    """Verify NOHZ idle load balancing prefers a fully idle SMT core.

    JIRA: DGX-18182
    MAINLINE PROVENANCE: https://git.kernel.org/torvalds/c/293f9611ae73
      (sched/fair: Prefer fully idle cores for NOHZ balancing; CONFIG_SCHED_SMT=n
      builds also pair with 815c5cb76a3e). This behavioral test requires active
      SMT, skips when SMT is disabled or unsupported, and does not establish
      source-commit presence or absence on the running kernel.
    """
    kconfig_verdict, kconfig_reason = _check_observer_kconfig()
    if kconfig_verdict == GLTCore.GLTTestResult.SKIP:
        GLTCore.GLTTestResult.skip(kconfig_reason)
    if kconfig_verdict != GLTCore.GLTTestResult.PASS:
        GLTCore.GLTTestResult.fail(kconfig_reason)

    verdict, layout, reason = _select_nohz_smt_layout()
    if verdict == GLTCore.GLTTestResult.SKIP:
        GLTCore.GLTTestResult.skip(reason)
    if verdict != GLTCore.GLTTestResult.PASS or layout is None:
        GLTCore.GLTTestResult.fail(
            reason or "Unexpected NOHZ SMT layout verdict."
        )

    tracing_root = _tracing_root()
    if tracing_root is None:
        GLTCore.GLTTestResult.skip(
            "tracefs instances or event csd/csd_queue_cpu is unavailable."
        )

    try:
        read_cpu_activity(layout.observed_cpus)
        read_tick_states(
            {
                layout.first_idle_cpu,
                *(cpu for core in layout.idle_cores for cpu in core),
            }
        )
    except ObserverUnavailable as error:
        GLTCore.GLTTestResult.skip(
            f"Required scheduler or NOHZ tick observer is unavailable: {error}"
        )
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"NOHZ ILB observer check failed: {error}"
        )

    logger.info(
        "NOHZ SMT layout: partial_core=%s (idle=%d, trigger=%d), idle_cores=%d",
        list(layout.partial_core),
        layout.first_idle_cpu,
        layout.trigger_cpu,
        len(layout.idle_cores),
    )

    conclusive_targets = []
    observer_gaps = 0
    last_reason = ""
    for trial in range(1, MAX_TRIALS + 1):
        try:
            observation = _run_nohz_ilb_trial(layout, tracing_root)
        except ObserverUnavailable as error:
            GLTCore.GLTTestResult.skip(
                f"Required scheduler or NOHZ tick observer is unavailable: {error}"
            )
        except (OSError, ValueError, RuntimeError) as error:
            GLTCore.GLTTestResult.fail(f"NOHZ ILB probe failed: {error}")

        if observation.failed_partial_cpu is not None:
            selected_core = list(
                observation.failed_partial_core or layout.partial_core
            )
            GLTCore.GLTTestResult.fail(
                f"find_new_ilb() selected CPU {observation.failed_partial_cpu} "
                f"on partially busy SMT core {selected_core} on the first kick "
                f"from CPU {layout.trigger_cpu} while another housekeeping SMT "
                "core remained NOHZ tick-stopped and task-idle. Expected NOHZ "
                "idle load balancing to "
                "prefer a fully idle SMT core over an idle sibling of an "
                "active SMT core (mainline provenance: 293f9611ae73 "
                "'sched/fair: Prefer fully idle cores for NOHZ balancing'; "
                "this behavioral result alone does not establish source-commit "
                "presence)."
            )
        if observation.conclusive_target is None:
            if observation.observer_gap:
                observer_gaps += 1
            last_reason = observation.reason
            logger.info(
                "Trial %d/%d discarded: %s",
                trial,
                MAX_TRIALS,
                last_reason,
            )
            continue

        conclusive_targets.append(observation.conclusive_target)
        logger.info(
            "Trial %d/%d: first nohz_csd_func kick from CPU %d -> CPU %d "
            "(%d/%d conclusive)",
            trial,
            MAX_TRIALS,
            layout.trigger_cpu,
            observation.conclusive_target,
            len(conclusive_targets),
            MIN_CONCLUSIVE_TRIALS,
        )
        if len(conclusive_targets) >= MIN_CONCLUSIVE_TRIALS:
            break

    if len(conclusive_targets) < MIN_CONCLUSIVE_TRIALS:
        summary = (
            f"Collected only {len(conclusive_targets)}/{MIN_CONCLUSIVE_TRIALS} "
            f"conclusive first-kick trials from CPU {layout.trigger_cpu} after "
            f"{MAX_TRIALS} attempts. Last non-conclusive reason: {last_reason}"
        )
        if observer_gaps > 0:
            GLTCore.GLTTestResult.warn(summary)
        GLTCore.GLTTestResult.skip(summary)

    logger.info(
        "PASS: first nohz_csd_func kicks across %d clean trials from CPU %d "
        "avoided partially busy SMT core %s (selected CPUs: %s).",
        len(conclusive_targets),
        layout.trigger_cpu,
        list(layout.partial_core),
        conclusive_targets,
    )
