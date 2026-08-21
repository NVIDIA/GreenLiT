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

"""Test NVIDIA Olympus PMCCNTR_EL0 handling."""

import re
import subprocess
import sys
from pathlib import Path

from common import glt_core as GLTCore
from common import perf_utils
from common.cpu_topology import parse_cpu_mask
from common.nv_device import NvDev


CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
NMI_WATCHDOG_PATH = Path("/proc/sys/kernel/nmi_watchdog")

FIX_COMMIT = "49f2413f9b55"
FIX_SUBJECT = "perf/arm_pmu: Skip PMCCNTR_EL0 on NVIDIA Olympus"

IDLE_CYCLE_MAX = 100_000_000
MAX_DELTA_RATIO = 0.10
PROBE_SECONDS = 1
PROBE_TIMEOUT_SECONDS = 10


def _singleton_topology_reason(cpu_sysfs_root):
    """Return why online CPU sibling topology is unsupported, else empty."""
    cpu_sysfs_root = Path(cpu_sysfs_root)
    online_path = cpu_sysfs_root / "online"
    try:
        online_cpus = parse_cpu_mask(online_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(
            f"Cannot validate online CPU topology from {online_path}: {err}"
        ) from err

    if not online_cpus:
        raise ValueError(
            f"Cannot validate SMT topology because {online_path} is empty."
        )

    for cpu_id in sorted(online_cpus):
        siblings_path = (
            cpu_sysfs_root
            / f"cpu{cpu_id}"
            / "topology"
            / "thread_siblings_list"
        )
        try:
            siblings = parse_cpu_mask(
                siblings_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as err:
            raise ValueError(
                f"Cannot validate SMT topology from {siblings_path}: {err}"
            ) from err

        if cpu_id not in siblings:
            raise ValueError(
                f"Cannot validate SMT topology because {siblings_path} "
                f"does not include CPU {cpu_id}: {sorted(siblings)}."
            )

        if siblings != {cpu_id}:
            return (
                f"CPU {cpu_id} exposes SMT siblings {sorted(siblings)} in "
                f"{siblings_path}; this test requires singleton firmware "
                "topology."
            )

    return ""


def _firmware_single_thread_reason(cpu_sysfs_root=CPU_SYSFS_ROOT):
    """Return why firmware-disabled SMT topology is unsupported, else empty.

    The generic ARM PMU SMT workaround checks firmware topology even when a
    sibling is offline. Requiring the kernel's ``notsupported`` state prevents
    ``nosmt`` from masking a missing Olympus-specific fix.
    """
    cpu_sysfs_root = Path(cpu_sysfs_root)
    smt_control_path = cpu_sysfs_root / "smt" / "control"
    try:
        smt_control = smt_control_path.read_text(encoding="utf-8").strip()
    except OSError as err:
        raise ValueError(
            "Cannot confirm that SMT is disabled in firmware because "
            f"{smt_control_path} is unreadable: {err}"
        ) from err

    if smt_control == "notsupported":
        return _singleton_topology_reason(cpu_sysfs_root)

    # off/forceoff disable threads in Linux but preserve firmware SMT topology;
    # notimplemented cannot establish that firmware exposed singleton cores.
    if smt_control in {"forceoff", "notimplemented", "off", "on"}:
        return (
            "This test requires SMT disabled in UEFI so Linux reports "
            f"'{smt_control_path}=notsupported'; current value is "
            f"{smt_control!r}. The nosmt kernel parameter or offline SMT "
            "siblings are not sufficient because the generic SMT "
            "workaround already avoids PMCCNTR_EL0."
        )

    raise ValueError(
        f"Cannot interpret SMT control value {smt_control!r} from "
        f"{smt_control_path}."
    )


def _nmi_watchdog_reason(path=NMI_WATCHDOG_PATH):
    """Return why PMU watchdog state is unsupported, else empty."""
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError as err:
        raise ValueError(
            f"Cannot confirm PMU hard-lockup watchdog state from {path}: {err}"
        ) from err

    if value == "0":
        return ""
    if value == "1":
        return (
            f"PMU hard-lockup watchdog is enabled ({path}=1); its pinned "
            "CPU_CYCLES event can occupy PMCCNTR_EL0 and mask the regression."
        )
    raise ValueError(
        f"Cannot interpret PMU hard-lockup watchdog value {value!r} from {path}."
    )


def _runtime_precondition_reason():
    """Return the first unsupported runtime precondition reason, else empty."""
    topology_reason = _firmware_single_thread_reason()
    if topology_reason:
        return topology_reason
    return _nmi_watchdog_reason()


def _parse_perf_stat_cycle_rows(stderr):
    """Return ordered cycle counts for each CPU from perf stat CSV output."""
    counts = {}

    for line in stderr.splitlines():
        parts = [part.strip() for part in line.strip().split(",")]
        if len(parts) < 4:
            continue

        cpu_match = re.fullmatch(r"(?:CPU)?(?P<cpu>[0-9]+)", parts[0])
        if cpu_match is None:
            continue
        event_name = parts[3].split(":", 1)[0]
        if event_name not in {"cycles", "cpu-cycles", "cpu_cycles"}:
            continue

        cpu_id = int(cpu_match.group("cpu"))
        count_field = parts[1]
        if count_field in {"<not counted>", "<not supported>"}:
            raise ValueError(
                f"CPU {cpu_id} cycle counter is unavailable: {count_field}"
            )

        try:
            count = int(count_field)
        except ValueError as err:
            raise ValueError(
                f"CPU {cpu_id} has an invalid cycle count {count_field!r}"
            ) from err
        if count < 0:
            raise ValueError(f"CPU {cpu_id} has a negative cycle count: {count}")

        counts.setdefault(cpu_id, []).append(count)

    return counts


def _parse_perf_stat_repeated_cycles(stderr, expected_count=2):
    """Parse exactly ``expected_count`` ordered cycle counts for every CPU."""
    counts = _parse_perf_stat_cycle_rows(stderr)
    stderr_tail = stderr.strip()[-600:] or "<empty>"

    if not counts:
        raise ValueError(
            "No per-CPU cycle counts found in perf stat output; "
            f"stderr tail: {stderr_tail}"
        )

    for cpu_id, cpu_counts in sorted(counts.items()):
        if len(cpu_counts) != expected_count:
            raise ValueError(
                f"CPU {cpu_id} reported {len(cpu_counts)} cycle counts; "
                f"expected {expected_count}; stderr tail: {stderr_tail}"
            )

    return counts


def _probe_olympus_pmccntr(logger, perf_cmd):
    """Measure two grouped system-wide CPU_CYCLES events for one second."""
    command = [
        perf_cmd,
        "stat",
        "-x,",
        "-a",
        "-A",
        "-e",
        "{cycles,cycles}:e",
        "--",
        sys.executable,
        "-c",
        f"import time; time.sleep({PROBE_SECONDS})",
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as err:
        raise ValueError(
            f"perf stat timed out after {PROBE_TIMEOUT_SECONDS} seconds"
        ) from err
    except subprocess.CalledProcessError as err:
        stderr = err.stderr or ""
        logger.debug(
            "Olympus PMCCNTR perf stat returncode=%d stderr=%s",
            err.returncode,
            stderr.strip()[-1200:],
        )
        raise ValueError(
            f"perf stat exited with status {err.returncode}; "
            f"stderr tail: {stderr.strip()[-600:] or '<empty>'}"
        ) from err
    except OSError as err:
        raise ValueError(f"Unable to execute perf stat: {err}") from err

    stderr = result.stderr or ""
    logger.debug(
        "Olympus PMCCNTR perf stat returncode=%d stderr=%s",
        result.returncode,
        stderr.strip()[-1200:],
    )

    return _parse_perf_stat_repeated_cycles(stderr)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_olympus_pmccntr_wfi_handling(_, logger):
    """Verify Olympus CPU_CYCLES events do not use PMCCNTR_EL0 in WFI/WFE.

    PATCH: 49f2413f9b55 perf/arm_pmu: Skip PMCCNTR_EL0 on NVIDIA Olympus
    PATCH: https://lore.kernel.org/all/20260608234135.1856911-1-bwicaksono@nvidia.com/
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not a Vera CPU; the Olympus PMCCNTR test is Vera-specific."
        )

    try:
        precondition_reason = _runtime_precondition_reason()
    except ValueError as err:
        GLTCore.GLTTestResult.fail(str(err))
    if precondition_reason:
        GLTCore.GLTTestResult.skip(precondition_reason)

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    try:
        cycle_counts = _probe_olympus_pmccntr(logger, perf_cmd)
    except ValueError as err:
        GLTCore.GLTTestResult.fail(f"Olympus PMCCNTR_EL0 probe failed: {err}")

    cycle_results = []
    for cpu_id, (first_cycles, second_cycles) in sorted(cycle_counts.items()):
        maximum_cycles = max(first_cycles, second_cycles)
        relative_delta = (
            abs(first_cycles - second_cycles) / maximum_cycles
            if maximum_cycles else 0.0
        )
        logger.debug(
            "Olympus PMCCNTR probe: cpu=%d first=%d second=%d delta=%.4f",
            cpu_id,
            first_cycles,
            second_cycles,
            relative_delta,
        )

        if first_cycles == 0 and second_cycles == 0:
            continue
        cycle_results.append(
            (cpu_id, first_cycles, second_cycles, relative_delta)
        )

    pmccntr_divergent = [
        result for result in cycle_results
        if result[1] > result[2] and result[3] > MAX_DELTA_RATIO
    ]
    if pmccntr_divergent:
        cpu_id, first_cycles, second_cycles, relative_delta = max(
            pmccntr_divergent, key=lambda result: result[3]
        )
        GLTCore.GLTTestResult.fail(
            f"CPU {cpu_id} grouped CPU_CYCLES counts showed the Olympus "
            f"PMCCNTR signature: first={first_cycles}, "
            f"second={second_cycles}, relative delta={relative_delta:.4f}, "
            f"maximum={MAX_DELTA_RATIO}. PMCCNTR_EL0 may have been used for "
            "the first event while the PE was in WFI/WFE; consider checking "
            f"whether the kernel is missing {FIX_COMMIT} ({FIX_SUBJECT})."
        )

    idle_results = [
        result for result in cycle_results if result[2] <= IDLE_CYCLE_MAX
    ]
    if not idle_results:
        GLTCore.GLTTestResult.skip(
            "No CPU had a positive programmable CPU_CYCLES count at or below "
            f"{IDLE_CYCLE_MAX}; cannot confirm Olympus PMCCNTR_EL0 behavior "
            "while a PE enters WFI/WFE."
        )

    reverse_divergent = [
        result for result in idle_results
        if result[1] < result[2] and result[3] > MAX_DELTA_RATIO
    ]
    if reverse_divergent:
        cpu_id, first_cycles, second_cycles, relative_delta = max(
            reverse_divergent, key=lambda result: result[3]
        )
        GLTCore.GLTTestResult.warn(
            f"CPU {cpu_id} grouped CPU_CYCLES counts diverged in the "
            f"unexpected direction: first={first_cycles}, "
            f"second={second_cycles}, relative delta={relative_delta:.4f}, "
            f"maximum={MAX_DELTA_RATIO}. The first event did not exceed the "
            "programmable-counter reference, so the Olympus PMCCNTR_EL0 "
            "regression state is inconclusive."
        )

    logger.info(
        "Olympus CPU_CYCLES counters matched on %d sufficiently idle CPUs",
        len(idle_results),
    )
