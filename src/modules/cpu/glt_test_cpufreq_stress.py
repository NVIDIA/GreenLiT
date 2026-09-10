# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Python script to test CPU frequency under stress.
"""
# pylint: disable=too-many-lines
import errno
import glob
import os
import platform
import re
import signal
import subprocess
import sys
import time

from common import glt_core as GLTCore
from common import glt_utils
from common.cpu_topology import parse_cpu_mask as _parse_cpu_mask
from common.nv_device import NvDev

# =============================================================================
# Configuration Constants
# =============================================================================

# Duration (in seconds) for stress-ng to run CPU stress workload.
# Set high enough to leave ample margin after the stabilization wait for
# sysfs reads on large systems (e.g. 144 CPUs).
STRESS_NG_TIMEOUT_SECS = 20

# Time (in seconds) to wait for CPU frequencies to stabilize after stress starts.
# This duration allows CPU frequencies to reach steady state under load, accounting
# for typical ARM CPU frequency scaling ramp-up behavior and thermal settling.
FREQ_STABILIZATION_WAIT_SECS = 8

# Maximum acceptable CPU frequency variance (percentage). If the variance between
# min and max frequencies exceeds this threshold, the test fails. High variance
# may indicate thermal throttling or power management issues.
# The 20% threshold is an empirically derived tolerance for ARM CPPC frequency
# scaling: under a uniform all-core stress workload with the performance governor,
# CPUs on the same die typically converge within ~5-10% of each other. The 20%
# ceiling provides headroom for NUMA topology and thermal differences while still
# catching anomalies such as stuck governors or power-capping.
MAX_FREQ_VARIANCE_PERCENT = 20

# Minimum frequency (in MHz) to consider a CPU as "high-performance".
# CPUs running above this threshold are counted in the results summary.
HIGH_PERF_CPU_FREQ_MHZ = 1000

# Minimum CPU frequency as a percentage of the CPU's rated max under stress.
# Applies to platforms using the cppc_cpufreq driver.
SCALING_FREQ_MIN_PERCENT = 70
CPPC_PERF_LIMITED_CLEAR_MASK = "0x3"
CPPC_PERF_LIMITED_SUPPORTED_MASK = 0x3
CPPC_QOS_UPDATE_SETTLE_SECS = 1
CPPC_QOS_UPDATE_TIMEOUT_SECS = 5
CPPC_QOS_UPDATE_POLL_SECS = 0.1
CPPC_PERF_LIMITED_ABI_PATCH = "https://git.kernel.org/torvalds/c/13c45a26635f"
CPPC_BOOST_QOS_PATCH = "https://git.kernel.org/torvalds/c/6e39ba4e5a82"
CPPC_BOOST_QOS_PATCHES = (
    "https://git.kernel.org/torvalds/c/04aa9d0726cc",
    CPPC_BOOST_QOS_PATCH,
    "https://git.kernel.org/torvalds/c/9266b4da051a",
)
CPPC_BOOST_QOS_HINT = (
    f"This symptom matches the fix in {CPPC_BOOST_QOS_PATCH}. Check whether "
    "the complete CPPC boost QoS series is applied, including its QoS lifetime "
    "fix (upstream order):\n" + "\n".join(CPPC_BOOST_QOS_PATCHES)
)
CPPC_PERF_CONTROLS_PATCHES = (
    "https://git.kernel.org/torvalds/c/658fa7b1c47a",
    "https://git.kernel.org/torvalds/c/b3e45fb2db9d",
    "https://git.kernel.org/torvalds/c/38428a680026",
    "https://git.kernel.org/torvalds/c/24ad4c6c136b",
    "https://git.kernel.org/torvalds/c/ea3db45ae476",
    CPPC_PERF_LIMITED_ABI_PATCH,
    "https://git.kernel.org/torvalds/c/856250ba2e81",
)
CPU_SYSFS_ROOT = "/sys/devices/system/cpu"
CPU_HOTPLUG_OFFLINE_SETTLE_SECS = 2
CPU_HOTPLUG_ONLINE_TIMEOUT_SECS = 15
CPU_HOTPLUG_ONLINE_POLL_SECS = 1
CPU_HOTPLUG_POLICY_TIMEOUT_SECS = 15
CPU_HOTPLUG_POLICY_POLL_SECS = 1

CPUFREQ_ONLINE_GET_FAILED_MSG = "cpufreq: cpufreq_online: ->get() failed"

REQUIRED_PATCHES = (
    "https://lore.kernel.org/all/173982791748.4020779.2848639862581042284.b4-ty@arm.com/",
    "https://lore.kernel.org/all/20250131155842.3839098-1-beata.michalska@arm.com/",
    "https://lore.kernel.org/all/20240827154818.1195849-1-ionela.voinescu@arm.com/",
    "https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git/commit/"
    "?id=20711efa91e8ba44149f5e2ed1cf81e5355650e5",
)


def _cpu_cpufreq_glob(filename):
    """Return glob results for a cpufreq sysfs file across all CPUs."""
    return glob.glob(f"/sys/devices/system/cpu/cpu[0-9]*/cpufreq/{filename}")


def _get_cpufreq_policy_dir(cpu_num, cpu_root=CPU_SYSFS_ROOT, path_exists=os.path.isdir):
    """Return the cpufreq policy directory for a CPU."""
    policy_dir = os.path.join(cpu_root, "cpufreq", f"policy{cpu_num}")
    if path_exists(policy_dir):
        return policy_dir

    cpu_dir = os.path.join(cpu_root, f"cpu{cpu_num}", "cpufreq")
    if path_exists(cpu_dir):
        return cpu_dir

    return policy_dir


def _is_single_cpu_cppc_policy(
        cpu_num, cpu_root=CPU_SYSFS_ROOT, path_exists=os.path.isdir,
        read_file=glt_utils.read_sysfs_value):
    """Return whether CPU hotplug can prove policy teardown/recreation."""
    policy_dir = _get_cpufreq_policy_dir(cpu_num, cpu_root, path_exists)
    if not path_exists(policy_dir):
        return False, f"cpufreq policy directory missing for CPU{cpu_num}: {policy_dir}"
    try:
        affected_cpus = _parse_cpu_mask(read_file(os.path.join(policy_dir, "affected_cpus")))
        related_cpus = _parse_cpu_mask(read_file(os.path.join(policy_dir, "related_cpus")))
    except (IOError, OSError, ValueError, KeyError) as e:
        return False, f"Cannot inspect {policy_dir}: {e}"

    if affected_cpus != {cpu_num} or related_cpus != {cpu_num}:
        return False, (
            f"{policy_dir} is shared (affected={sorted(affected_cpus)}, "
            f"related={sorted(related_cpus)}); cannot prove policy recreation."
        )
    return True, ""


def _select_hotplug_cpu(
        online_cpus, cpu_root=CPU_SYSFS_ROOT, is_writable=os.access,
        path_exists=os.path.isdir, read_file=glt_utils.read_sysfs_value):
    """Return a safe non-CPU0 online CPU with a writable hotplug control."""
    last_reason = "No non-CPU0 online CPU has a writable hotplug control."
    for cpu_num in sorted(online_cpus):
        if cpu_num == 0:
            continue
        online_path = os.path.join(cpu_root, f"cpu{cpu_num}", "online")
        if not is_writable(online_path, os.W_OK):
            continue
        ok, reason = _is_single_cpu_cppc_policy(
            cpu_num, cpu_root=cpu_root, path_exists=path_exists, read_file=read_file
        )
        if ok:
            return cpu_num, ""
        last_reason = reason
    return None, last_reason


def _validate_cppc_policy_for_cpu(
        cpu_num, cpu_root=CPU_SYSFS_ROOT, path_exists=os.path.isdir):
    """Validate a CPU has a recreated cppc_cpufreq policy with positive frequencies."""
    policy_dir = _get_cpufreq_policy_dir(cpu_num, cpu_root, path_exists)
    if not path_exists(policy_dir):
        return False, f"cpufreq policy directory missing for CPU{cpu_num}: {policy_dir}"

    try:
        affected_cpus = _parse_cpu_mask(
            glt_utils.read_sysfs_value(os.path.join(policy_dir, "affected_cpus"))
        )
        related_cpus = _parse_cpu_mask(
            glt_utils.read_sysfs_value(os.path.join(policy_dir, "related_cpus"))
        )
        if affected_cpus != {cpu_num}:
            return False, f"{policy_dir}/affected_cpus is not CPU{cpu_num}: {sorted(affected_cpus)}"
        if related_cpus != {cpu_num}:
            return False, f"{policy_dir}/related_cpus is not CPU{cpu_num}: {sorted(related_cpus)}"

        for filename in ("scaling_cur_freq", "cpuinfo_max_freq"):
            path = os.path.join(policy_dir, filename)
            value = int(glt_utils.read_sysfs_value(path))
            if value <= 0:
                return False, f"{path} is not positive: {value}"
    except (IOError, OSError, ValueError, KeyError) as e:
        return False, f"Cannot validate {policy_dir} for CPU{cpu_num}: {e}"

    return True, ""


def _wait_for_cppc_policy_for_cpu(
        cpu_num, timeout_secs=CPU_HOTPLUG_POLICY_TIMEOUT_SECS,
        poll_secs=CPU_HOTPLUG_POLICY_POLL_SECS):
    """Wait until a CPU has a valid recreated cppc_cpufreq policy."""
    deadline = time.monotonic() + timeout_secs
    last_reason = ""
    while True:
        ok, reason = _validate_cppc_policy_for_cpu(cpu_num)
        if ok:
            return True, ""
        last_reason = reason
        if time.monotonic() >= deadline:
            return False, last_reason
        time.sleep(poll_secs)


def _verify_cppc_policy_absent_for_cpu(
        cpu_num, cpu_root=CPU_SYSFS_ROOT, path_exists=os.path.isdir,
        read_file=glt_utils.read_sysfs_value):
    """Return whether a CPU's cpufreq policy is absent or unreadable while offline."""
    policy_dir = _get_cpufreq_policy_dir(cpu_num, cpu_root, path_exists)
    if not path_exists(policy_dir):
        return True, ""
    try:
        read_file(os.path.join(policy_dir, "scaling_cur_freq"))
    except (IOError, OSError, ValueError, KeyError):
        return True, ""
    return False, (
        f"cpufreq policy still readable for offline CPU{cpu_num}: {policy_dir}"
    )


def _is_cppc_not_applicable(reason):
    """Return whether a cppc prerequisite failure means this test should skip."""
    return reason.startswith("cpufreq sysfs not available") or \
        reason.startswith("cppc_cpufreq not active")


def _is_cpu_online(cpu_num, logger=None, unreadable_is_online=True):
    """Return whether a CPU is online, treating absent cpu0 online files as online."""
    online_path = os.path.join(CPU_SYSFS_ROOT, f"cpu{cpu_num}", "online")
    if not os.path.exists(online_path):
        return True
    try:
        with open(online_path, 'r', encoding='utf-8') as f:
            return f.read().strip() != "0"
    except (IOError, OSError):
        if logger:
            logger.warning("Cannot read online state for cpu%d", cpu_num)
        return unreadable_is_online


def _wait_for_cpu_online(
        cpu_num, timeout_secs=CPU_HOTPLUG_ONLINE_TIMEOUT_SECS,
        poll_secs=CPU_HOTPLUG_ONLINE_POLL_SECS, logger=None):
    """Wait until a CPU reports online after hotplug."""
    deadline = time.monotonic() + timeout_secs
    while True:
        if _is_cpu_online(cpu_num, logger=logger, unreadable_is_online=False):
            return True, ""
        if time.monotonic() >= deadline:
            return False, f"CPU{cpu_num} did not come back online after hotplug."
        time.sleep(poll_secs)


def _read_sysfs_int_value(path):
    """Read and validate a decimal or base-prefixed integer sysfs value."""
    try:
        value = glt_utils.read_sysfs_value(path)
    except (IOError, OSError) as e:
        message = f"cannot read {path}: {e}"
        if e.errno is None:
            raise OSError(message) from e
        raise OSError(e.errno, message) from e
    try:
        int(value, 0)
    except ValueError as e:
        raise ValueError(f"{path} is not an integer: {value}") from e
    return value


def _get_first_cppc_cpufreq_dir():
    """Return the first cpu cpufreq directory using cppc_cpufreq."""
    driver_paths = sorted(_cpu_cpufreq_glob("scaling_driver"))
    if not driver_paths:
        return None, "cpufreq sysfs not available; cannot read scaling_driver."

    non_cppc_drivers = []
    for driver_path in driver_paths:
        try:
            scaling_driver = glt_utils.read_sysfs_value(driver_path)
        except (IOError, OSError) as e:
            return None, f"Cannot read {driver_path}: {e}"
        if scaling_driver == "cppc_cpufreq":
            return os.path.dirname(driver_path), None
        non_cppc_drivers.append(scaling_driver)

    drivers = ", ".join(sorted(set(non_cppc_drivers))) or "unknown"
    return None, f"cppc_cpufreq not active (current driver: {drivers})."


def _get_cppc_boost_qos_test_caps(scaling_min, nominal_max):
    """Choose two distinct scaling caps between the minimum and nominal max."""
    span = nominal_max - scaling_min
    if scaling_min <= 0 or span < 3:
        return None

    first_cap = scaling_min + span // 3
    second_cap = scaling_min + (2 * span) // 3
    if not scaling_min < first_cap < second_cap < nominal_max:
        return None
    return first_cap, second_cap


def _get_cppc_ospm_nominal_test_freq(scaling_min, nominal_max):
    """Choose an OSPM nominal frequency that leaves boost headroom."""
    span = nominal_max - scaling_min
    if scaling_min <= 0 or span < 4:
        return None

    ospm_nominal = scaling_min + (3 * span) // 4
    if not scaling_min < ospm_nominal < nominal_max:
        return None
    return ospm_nominal


def _wait_for_cppc_scaling_max(
        path, expected, deadline, poll_secs=CPPC_QOS_UPDATE_POLL_SECS):
    """Wait for the expected cap, returning the last observation on timeout."""
    observed = int(_read_sysfs_int_value(path), 0)
    while observed != expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_secs, remaining))
        observed = int(_read_sysfs_int_value(path), 0)
    return observed


def _write_cppc_scaling_max_and_wait(
        path, requested,
        timeout_secs=CPPC_QOS_UPDATE_TIMEOUT_SECS,
        poll_secs=CPPC_QOS_UPDATE_POLL_SECS):
    """Write a cap and wait for that value, not just an intermediate change."""
    requested = int(requested)
    glt_utils.write_sysfs_value(path, requested)
    return _wait_for_cppc_scaling_max(
        path, requested, time.monotonic() + timeout_secs, poll_secs
    )


def _validate_cppc_sysfs_helpers(logger):
    """Validate Vera CPPC sysfs files added or exercised by DGX-16332."""
    cpufreq_dir, reason = _get_first_cppc_cpufreq_dir()
    if cpufreq_dir is None:
        return False, reason

    scaling_min_path = os.path.join(cpufreq_dir, "scaling_min_freq")
    scaling_max_path = os.path.join(cpufreq_dir, "scaling_max_freq")
    perf_limited_path = os.path.join(cpufreq_dir, "perf_limited")
    original_min = original_max = None

    try:
        cpuinfo_min = _read_sysfs_int_value(
            os.path.join(cpufreq_dir, "cpuinfo_min_freq")
        )
        cpuinfo_max = _read_sysfs_int_value(
            os.path.join(cpufreq_dir, "cpuinfo_max_freq")
        )
        original_min = _read_sysfs_int_value(scaling_min_path)
        original_max = _read_sysfs_int_value(scaling_max_path)
        perf_limited_before = _read_sysfs_int_value(perf_limited_path)

        logger.info("CPPC policy path: %s", cpufreq_dir)
        logger.info("cpuinfo_min_freq=%s cpuinfo_max_freq=%s", cpuinfo_min, cpuinfo_max)
        logger.info("scaling_min_freq=%s scaling_max_freq=%s", original_min, original_max)
        logger.info("perf_limited before clear=%s", perf_limited_before)

        glt_utils.write_sysfs_value(scaling_max_path, original_max)
        glt_utils.write_sysfs_value(scaling_min_path, original_min)

        current_min = _read_sysfs_int_value(scaling_min_path)
        current_max = _read_sysfs_int_value(scaling_max_path)
        if current_min != original_min:
            return False, (
                f"{scaling_min_path} did not round-trip "
                f"({current_min} != {original_min})."
            )
        if current_max != original_max:
            return False, (
                f"{scaling_max_path} did not round-trip "
                f"({current_max} != {original_max})."
            )

        glt_utils.write_sysfs_value(
            perf_limited_path, CPPC_PERF_LIMITED_CLEAR_MASK
        )
        perf_limited_after = _read_sysfs_int_value(perf_limited_path)
        logger.info("perf_limited after clear=%s", perf_limited_after)
    except (IOError, OSError, ValueError) as e:
        return False, f"CPPC sysfs validation failed: {e}"
    finally:
        if original_max is not None:
            try:
                glt_utils.write_sysfs_value(scaling_max_path, original_max)
            except (IOError, OSError):
                logger.warning("Failed to restore %s", scaling_max_path)
        if original_min is not None:
            try:
                glt_utils.write_sysfs_value(scaling_min_path, original_min)
            except (IOError, OSError):
                logger.warning("Failed to restore %s", scaling_min_path)

    return True, None


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cppc_boost_frequency_qos(_, logger):  # pylint: disable=too-many-branches
    """
    Verify CPPC boost and scaling-max requests remain independent.

    Requires cppc_cpufreq on all policies and usable boost headroom. When
    firmware provides no static headroom, use the optional OSPM nominal
    control only if the policy exposes it and supports boost transitions.

    PATCH: https://git.kernel.org/torvalds/c/6e39ba4e5a82
    RECOMMENDED UPSTREAM SERIES (only PATCH above is discriminated):
        https://git.kernel.org/torvalds/c/04aa9d0726cc
        https://git.kernel.org/torvalds/c/6e39ba4e5a82
        https://git.kernel.org/torvalds/c/9266b4da051a
    OSPM ABI: https://lore.kernel.org/lkml/20260807214837.863209-3-sumitg@nvidia.com/
    OSPM BOOST: https://lore.kernel.org/lkml/20260807214837.863209-4-sumitg@nvidia.com/
    """
    ok, reason = _validate_cppc_prerequisites()
    if not ok:
        if _is_cppc_not_applicable(reason):
            GLTCore.GLTTestResult.skip(reason)
        GLTCore.GLTTestResult.fail(reason)

    cpufreq_dir, reason = _get_first_cppc_cpufreq_dir()
    if cpufreq_dir is None:
        if _is_cppc_not_applicable(reason):
            GLTCore.GLTTestResult.skip(reason)
        GLTCore.GLTTestResult.fail(reason)

    boost_path = os.path.join(CPU_SYSFS_ROOT, "cpufreq", "boost")
    cpuinfo_max_path = os.path.join(cpufreq_dir, "cpuinfo_max_freq")
    scaling_min_path = os.path.join(cpufreq_dir, "scaling_min_freq")
    scaling_max_path = os.path.join(cpufreq_dir, "scaling_max_freq")
    ospm_nominal_path = os.path.join(cpufreq_dir, "ospm_nominal_freq")

    if not os.path.isfile(boost_path):
        GLTCore.GLTTestResult.skip(
            f"{boost_path} is unavailable; cpufreq does not expose global "
            "boost control."
        )

    try:
        original_boost = int(_read_sysfs_int_value(boost_path), 0)
    except (IOError, OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot read the original CPPC boost state: {error}"
        )
    if original_boost not in (0, 1):
        GLTCore.GLTTestResult.fail(
            f"{boost_path} has invalid state {original_boost}; expected 0 or 1."
        )

    policy_dirs = sorted(glob.glob(
        os.path.join(CPU_SYSFS_ROOT, "cpufreq", "policy[0-9]*")
    ))
    if not policy_dirs:
        GLTCore.GLTTestResult.skip(
            "Cannot enumerate cpufreq policies to preserve all scaling-max caps "
            "before changing global boost."
        )
    policy_boost_paths = sorted(glob.glob(
        os.path.join(CPU_SYSFS_ROOT, "cpufreq", "policy[0-9]*", "boost")
    ))
    cppc_caps_dir = os.path.join(os.path.dirname(cpufreq_dir), "acpi_cppc")
    highest_perf_path = os.path.join(cppc_caps_dir, "highest_perf")
    nominal_perf_path = os.path.join(cppc_caps_dir, "nominal_perf")
    original_policy_boosts = {}
    use_ospm_nominal = False
    ospm_restore_required = False
    boost_restore_required = False
    scaling_max_restore_required = False
    ospm_boost_transition_supported = False
    original_ospm_nominal = None
    ospm_test_freq = None
    original_cpuinfo_max = None
    try:
        for policy_boost_path in policy_boost_paths:
            policy_boost = int(_read_sysfs_int_value(policy_boost_path), 0)
            if policy_boost not in (0, 1):
                GLTCore.GLTTestResult.fail(
                    f"{policy_boost_path} has invalid state {policy_boost}; "
                    "expected 0 or 1."
                )
            original_policy_boosts[policy_boost_path] = policy_boost

        unsynchronized = [
            path for path, state in original_policy_boosts.items()
            if state != original_boost
        ]
        if unsynchronized:
            GLTCore.GLTTestResult.skip(
                "Global and per-policy boost states are not synchronized; "
                "cannot safely round-trip the system-wide boost setting: "
                + ", ".join(unsynchronized)
            )

        original_scaling_max = int(_read_sysfs_int_value(scaling_max_path), 0)
        # Global boost also changes unselected policies. On affected kernels
        # those transitions overwrite their userspace QoS requests as well.
        # Enumerate policies independently: older kernels expose global boost
        # without per-policy boost files. Read every cap before the first write.
        original_scaling_maxes = {scaling_max_path: original_scaling_max}
        for policy_dir in policy_dirs:
            policy_max_path = os.path.join(policy_dir, "scaling_max_freq")
            if os.path.realpath(policy_max_path) != \
                    os.path.realpath(scaling_max_path):
                original_scaling_maxes[policy_max_path] = int(
                    _read_sysfs_int_value(policy_max_path), 0
                )
        scaling_min = int(_read_sysfs_int_value(scaling_min_path), 0)
        original_cpuinfo_max = int(_read_sysfs_int_value(cpuinfo_max_path), 0)

        if os.path.isfile(highest_perf_path) and \
                os.path.isfile(nominal_perf_path):
            highest_perf = int(_read_sysfs_int_value(highest_perf_path), 0)
            nominal_perf = int(_read_sysfs_int_value(nominal_perf_path), 0)
            if highest_perf < nominal_perf:
                GLTCore.GLTTestResult.skip(
                    "Firmware reports invalid CPPC boost limits: "
                    f"highest_perf={highest_perf}, nominal_perf={nominal_perf}."
                )
            if highest_perf == nominal_perf:
                if not os.path.isfile(ospm_nominal_path):
                    GLTCore.GLTTestResult.skip(
                        "Firmware does not expose static CPPC boost headroom "
                        "and the policy has no OSPM nominal-frequency control: "
                        f"highest_perf={highest_perf}, "
                        f"nominal_perf={nominal_perf}."
                    )

                # The public CPPC v4 OSPM series deliberately exposes this as
                # a write-only ABI. It also registers boost support up front
                # and makes set_boost() use the software-tracked nominal value.
                use_ospm_nominal = True
    except (IOError, OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot read the original CPPC frequency limits: {error}"
        )

    restore_errors = []
    try:
        if use_ospm_nominal:
            # cpuinfo.max_freq exposes the effective OSPM nominal while boost
            # is disabled. Use it to preserve a pre-existing value without
            # attempting to read the write-only ospm_nominal_freq attribute.
            # This cannot recover a raw firmware request or unset sentinel;
            # frequency-conversion mismatches are checked during cleanup.
            if original_boost:
                boost_restore_required = True
                scaling_max_restore_required = True
                glt_utils.write_sysfs_value(boost_path, "0")
                time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
                disabled_policy_boosts = {
                    path: int(_read_sysfs_int_value(path), 0)
                    for path in policy_boost_paths
                }
                if int(_read_sysfs_int_value(boost_path), 0) != 0 or \
                        any(disabled_policy_boosts.values()):
                    GLTCore.GLTTestResult.skip(
                        "Cannot synchronize disabled CPPC boost state to "
                        "preserve the effective OSPM nominal frequency before "
                        "testing."
                    )
            original_ospm_nominal = int(
                _read_sysfs_int_value(cpuinfo_max_path), 0
            )
            ospm_test_freq = _get_cppc_ospm_nominal_test_freq(
                scaling_min, original_ospm_nominal
            )
            if ospm_test_freq is None:
                GLTCore.GLTTestResult.skip(
                    "The CPPC v4 policy has no room to lower its OSPM "
                    "nominal frequency while staying above scaling_min: "
                    f"scaling_min={scaling_min}, "
                    f"nominal_max={original_ospm_nominal}."
                )

            try:
                glt_utils.write_sysfs_value(
                    ospm_nominal_path, ospm_test_freq
                )
            except OSError as error:
                if error.errno not in (errno.EOPNOTSUPP, errno.ENODEV):
                    # A failed sysfs write can still have partially changed
                    # firmware state, so conservatively attempt restoration.
                    ospm_restore_required = True
                    raise
                GLTCore.GLTTestResult.skip(
                    "The platform does not support programming its OSPM "
                    f"nominal frequency at {ospm_nominal_path}: {error}"
                )
            else:
                ospm_restore_required = True

            # Prove the OSPM support series registered boost for every policy
            # before installing either QoS cap. The transition also makes
            # earlier downstream revisions recalculate cpuinfo.max_freq.
            scaling_max_restore_required = True
            boost_restore_required = True
            try:
                glt_utils.write_sysfs_value(boost_path, "1")
            except OSError as error:
                if error.errno not in (
                        errno.EINVAL, errno.EOPNOTSUPP, errno.ENODEV):
                    raise
                current_boost = int(
                    _read_sysfs_int_value(boost_path), 0
                )
                current_policy_boosts = {
                    path: int(_read_sysfs_int_value(path), 0)
                    for path in policy_boost_paths
                }
                if current_boost != 0 or any(
                        current_policy_boosts.values()):
                    raise
                # A confirmed unsupported no-op needs no compensating
                # write, unless fixture setup disabled original boost.
                boost_restore_required = bool(original_boost)
                scaling_max_restore_required = bool(original_boost)
                GLTCore.GLTTestResult.skip(
                    "The OSPM nominal-frequency control did not provide "
                    "global CPPC boost support: "
                    f"{error}"
                )
            else:
                ospm_boost_transition_supported = True
                time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
                fixture_boosted = int(
                    _read_sysfs_int_value(cpuinfo_max_path), 0
                )
                enabled_policy_boosts = {
                    path: int(_read_sysfs_int_value(path), 0)
                    for path in policy_boost_paths
                }
                fixture_boost = int(
                    _read_sysfs_int_value(boost_path), 0
                )
                glt_utils.write_sysfs_value(boost_path, "0")
                time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
                disabled_policy_boosts = {
                    path: int(_read_sysfs_int_value(path), 0)
                    for path in policy_boost_paths
                }
                if fixture_boost != 1 or \
                        any(state != 1 for state in
                            enabled_policy_boosts.values()) or \
                        fixture_boosted <= ospm_test_freq or \
                        int(_read_sysfs_int_value(boost_path), 0) != 0 or \
                        any(disabled_policy_boosts.values()):
                    GLTCore.GLTTestResult.skip(
                        "The OSPM nominal-frequency control did not create "
                        "a synchronized, distinct CPPC boost range."
                    )
        elif original_boost:
            boost_restore_required = True
            scaling_max_restore_required = True
            glt_utils.write_sysfs_value(boost_path, "0")
            time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
        boost_disabled = int(_read_sysfs_int_value(boost_path), 0)
        nominal_max = int(_read_sysfs_int_value(cpuinfo_max_path), 0)
        logger.info(
            "CPPC boost QoS policy %s: scaling_min=%d nominal_max=%d",
            cpufreq_dir, scaling_min, nominal_max
        )
        if boost_disabled != 0:
            GLTCore.GLTTestResult.fail(f"{boost_path} remained enabled after writing 0.")
        if use_ospm_nominal and not \
                scaling_min < nominal_max < original_cpuinfo_max:
            GLTCore.GLTTestResult.skip(
                "Lowering the OSPM nominal frequency did not create a "
                "distinct CPPC boost range: "
                f"scaling_min={scaling_min}, nominal_max={nominal_max}, "
                f"platform_max={original_cpuinfo_max}."
            )

        caps = _get_cppc_boost_qos_test_caps(scaling_min, nominal_max)
        if caps is None:
            GLTCore.GLTTestResult.skip(
                "The CPPC policy has no room for two distinct scaling-max "
                f"constraints between {scaling_min} and {nominal_max}."
            )

        first_cap, second_cap = caps
        scaling_max_restore_required = True
        first_accepted_cap = _write_cppc_scaling_max_and_wait(
            scaling_max_path, first_cap
        )
        # A clamped or still-pending value does not prove this request
        # took effect. Do not use it as a boost-overwrite baseline.
        if first_accepted_cap != first_cap:
            GLTCore.GLTTestResult.skip(
                f"Cannot establish the requested scaling-max constraint "
                f"at {scaling_max_path} within "
                f"{CPPC_QOS_UPDATE_TIMEOUT_SECS}s: requested {first_cap}, "
                f"observed {first_accepted_cap}."
            )

        boost_restore_required = True
        glt_utils.write_sysfs_value(boost_path, "1")
        time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
        boost_enabled = int(_read_sysfs_int_value(boost_path), 0)
        boosted_max = int(_read_sysfs_int_value(cpuinfo_max_path), 0)
        observed_cap = int(_read_sysfs_int_value(scaling_max_path), 0)
        logger.info(
            "After boost enable: boost=%d cpuinfo_max=%d scaling_max=%d",
            boost_enabled, boosted_max, observed_cap
        )
        if boost_enabled != 1:
            GLTCore.GLTTestResult.fail(f"{boost_path} remained disabled after writing 1.")
        if boosted_max <= nominal_max:
            GLTCore.GLTTestResult.skip(
                "The CPPC policy does not expose a distinct boost frequency "
                f"({boosted_max} <= nominal {nominal_max})."
            )
        if observed_cap != first_accepted_cap:
            GLTCore.GLTTestResult.fail(
                "Enabling CPPC boost overwrote the scaling-max QoS constraint "
                f"at {scaling_max_path}: expected {first_accepted_cap}, "
                f"observed {observed_cap}.\n{CPPC_BOOST_QOS_HINT}"
            )

        second_accepted_cap = _write_cppc_scaling_max_and_wait(
            scaling_max_path, second_cap
        )
        if second_accepted_cap != second_cap:
            GLTCore.GLTTestResult.skip(
                "Cannot establish the second requested scaling-max "
                f"constraint at {scaling_max_path} within "
                f"{CPPC_QOS_UPDATE_TIMEOUT_SECS}s: requested {second_cap}, "
                f"observed {second_accepted_cap}."
            )

        glt_utils.write_sysfs_value(boost_path, "0")
        time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
        boost_disabled = int(_read_sysfs_int_value(boost_path), 0)
        restored_nominal_max = int(_read_sysfs_int_value(cpuinfo_max_path), 0)
        observed_cap = int(_read_sysfs_int_value(scaling_max_path), 0)
        logger.info(
            "After boost disable: boost=%d cpuinfo_max=%d scaling_max=%d",
            boost_disabled, restored_nominal_max, observed_cap
        )
        if boost_disabled != 0:
            GLTCore.GLTTestResult.fail(f"{boost_path} remained enabled after writing 0.")
        if restored_nominal_max != nominal_max:
            GLTCore.GLTTestResult.fail(
                "Disabling CPPC boost did not restore the nominal maximum "
                f"frequency: expected {nominal_max}, observed "
                f"{restored_nominal_max}."
            )
        if observed_cap != second_accepted_cap:
            GLTCore.GLTTestResult.fail(
                "Disabling CPPC boost overwrote the scaling-max QoS "
                f"constraint at {scaling_max_path}: expected "
                f"{second_accepted_cap}, observed {observed_cap}.\n"
                f"{CPPC_BOOST_QOS_HINT}"
            )
    except (IOError, OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(
            f"CPPC boost-frequency QoS validation failed: {error}"
        )
    finally:
        # Preserve the propagating diagnostic if cleanup also fails.
        original_error = sys.exc_info()[1]
        if ospm_restore_required:
            try:
                glt_utils.write_sysfs_value(
                    ospm_nominal_path, original_ospm_nominal
                )
            except (IOError, OSError, ValueError) as error:
                restore_errors.append(
                    f"cannot restore {ospm_nominal_path}: {error}"
                )

        try:
            # A global write re-synchronizes every policy even if the global
            # value already matches. With a restored OSPM nominal value, a
            # 1 -> 0 transition also refreshes the disabled policy maximum.
            if ospm_restore_required and ospm_boost_transition_supported:
                glt_utils.write_sysfs_value(boost_path, "1")
                glt_utils.write_sysfs_value(boost_path, "0")
                time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
                restored_nominal = int(_read_sysfs_int_value(cpuinfo_max_path), 0)
                if restored_nominal != original_ospm_nominal:
                    restore_errors.append(
                        f"effective OSPM nominal restored to {restored_nominal}, "
                        f"expected {original_ospm_nominal}"
                    )
        except (IOError, OSError, ValueError) as error:
            restore_errors.append(f"cannot verify restored OSPM nominal: {error}")

        try:
            if boost_restore_required or \
                    (ospm_restore_required and original_boost):
                glt_utils.write_sysfs_value(boost_path, original_boost)
            if boost_restore_required or ospm_restore_required:
                time.sleep(CPPC_QOS_UPDATE_SETTLE_SECS)
            restored_boost = int(_read_sysfs_int_value(boost_path), 0)
            if restored_boost != original_boost:
                restore_errors.append(
                    f"{boost_path} restored to {restored_boost}, expected "
                    f"{original_boost}"
                )
            if ospm_restore_required:
                restored_cpuinfo_max = int(
                    _read_sysfs_int_value(cpuinfo_max_path), 0
                )
                if restored_cpuinfo_max != original_cpuinfo_max:
                    restore_errors.append(
                        f"{cpuinfo_max_path} restored to "
                        f"{restored_cpuinfo_max}, expected "
                        f"{original_cpuinfo_max}"
                    )
        except (IOError, OSError, ValueError) as error:
            restore_errors.append(f"cannot restore {boost_path}: {error}")

        if scaling_max_restore_required:
            # Two shared wait budgets: first allow peers to converge without
            # writes, then verify all issued restores. Neither grows with the
            # policy count. The bounds cover polling, not blocking sysfs I/O.
            peer_restore_deadline = time.monotonic() + CPPC_QOS_UPDATE_TIMEOUT_SECS
            pending_restores = {}
            for path, original_max in original_scaling_maxes.items():
                try:
                    # Do not rewrite unchanged peer requests: sysfs reports
                    # the effective limit, not other clients' hidden requests.
                    if path != scaling_max_path and _wait_for_cppc_scaling_max(
                            path, original_max, peer_restore_deadline) == original_max:
                        continue
                    # Queue every necessary restore before waiting for any.
                    glt_utils.write_sysfs_value(path, original_max)
                    pending_restores[path] = original_max
                except (IOError, OSError, ValueError) as error:
                    restore_errors.append(
                        f"cannot restore {path}: {error}"
                    )

            # A fresh phase budget gives even the final restore time to apply;
            # the passive convergence deadline may already have expired.
            restore_deadline = time.monotonic() + CPPC_QOS_UPDATE_TIMEOUT_SECS
            for path, original_max in pending_restores.items():
                try:
                    restored_scaling_max = _wait_for_cppc_scaling_max(
                        path, original_max, restore_deadline
                    )
                    if restored_scaling_max != original_max:
                        restore_errors.append(
                            f"{path} restored to {restored_scaling_max}, "
                            f"expected {original_max}"
                        )
                except (IOError, OSError, ValueError) as error:
                    restore_errors.append(
                        f"cannot restore {path}: {error}"
                    )

        for policy_boost_path, original_policy_boost in \
                original_policy_boosts.items():
            try:
                restored_policy_boost = int(
                    _read_sysfs_int_value(policy_boost_path), 0
                )
                if restored_policy_boost != original_policy_boost:
                    restore_errors.append(
                        f"{policy_boost_path} restored to "
                        f"{restored_policy_boost}, expected "
                        f"{original_policy_boost}"
                    )
            except (IOError, OSError, ValueError) as error:
                restore_errors.append(
                    f"cannot verify {policy_boost_path}: {error}"
                )

        if restore_errors:
            restore_reason = "Failed to restore CPPC policy state: " + "; ".join(restore_errors)
            reason = str(original_error) if original_error is not None else None
            if isinstance(original_error, GLTCore.GLTTestSkip):
                # The CLI summary keeps one line; lead with why SKIP became FAIL.
                reason = f"{restore_reason}\nOriginal skip: {reason}"
            else:
                reason = f"{reason}\nAdditionally, {restore_reason}" if reason else restore_reason
            GLTCore.GLTTestResult.fail(reason)

    logger.info("PASS: CPPC boost-frequency QoS constraints remained independent")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_cppc_sysfs_helpers(_, logger):
    """
    Validate Vera CPPC sysfs helper coverage.

    JIRA: DGX-16332
    PATCH: https://git.kernel.org/torvalds/c/13c45a26635f
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    ok, reason = _validate_cppc_sysfs_helpers(logger)
    if not ok:
        GLTCore.GLTTestResult.fail(reason)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cppc_perf_controls(_, logger):
    """
    Verify CPPC perf_limited control is readable and accepts a clear mask.

    ABI PATCH: https://git.kernel.org/torvalds/c/13c45a26635f
    RECOMMENDED UPSTREAM SERIES:
        https://git.kernel.org/torvalds/c/658fa7b1c47a
        https://git.kernel.org/torvalds/c/b3e45fb2db9d
        https://git.kernel.org/torvalds/c/38428a680026
        https://git.kernel.org/torvalds/c/24ad4c6c136b
        https://git.kernel.org/torvalds/c/ea3db45ae476
        https://git.kernel.org/torvalds/c/13c45a26635f
        https://git.kernel.org/torvalds/c/856250ba2e81
    """
    ok, reason = _validate_cppc_prerequisites()
    cpufreq_dir = None
    if ok:
        cpufreq_dir, reason = _get_first_cppc_cpufreq_dir()
    if not ok or cpufreq_dir is None:
        if _is_cppc_not_applicable(reason):
            GLTCore.GLTTestResult.skip(reason)
        GLTCore.GLTTestResult.fail(reason)

    perf_limited_path = os.path.join(cpufreq_dir, "perf_limited")
    patches_msg = "\n".join(
        f"  - {patch}" for patch in CPPC_PERF_CONTROLS_PATCHES
    )
    try:
        before_value = glt_utils.read_sysfs_value(perf_limited_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            "CPPC perf_limited sysfs interface is unavailable. This interface is "
            f"introduced by:\n  - {CPPC_PERF_LIMITED_ABI_PATCH}\n"
            "To keep the CPPC performance-control implementation aligned with "
            f"upstream, apply the complete upstream patch series in order:\n{patches_msg}"
        )
    except (IOError, OSError) as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot validate CPPC perf_limited control: {error}"
        )

    if before_value == "<unsupported>":
        GLTCore.GLTTestResult.skip(
            "CPPC perf_limited is unsupported by platform firmware."
        )

    try:
        before = int(before_value, 0)
        glt_utils.write_sysfs_value(perf_limited_path, CPPC_PERF_LIMITED_CLEAR_MASK)
        after = int(glt_utils.read_sysfs_value(perf_limited_path), 0)
    except (IOError, OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot validate CPPC perf_limited control: {error}"
        )

    logger.info(
        "CPPC perf_limited at %s: before=%#x after=%#x", perf_limited_path, before, after
    )
    if after & CPPC_PERF_LIMITED_SUPPORTED_MASK:
        GLTCore.GLTTestResult.warn(
            "CPPC perf_limited status bits were set again after clear; "
            "the platform may have reported a concurrent performance limitation."
        )


def _get_cpu_max_freq_khz():
    """
    Return the highest cpuinfo_max_freq across all CPUs in kHz, or None if unavailable.
    """
    max_freq = None
    for path in _cpu_cpufreq_glob("cpuinfo_max_freq"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                freq = int(f.read().strip())
                if max_freq is None or freq > max_freq:
                    max_freq = freq
        except (IOError, OSError, ValueError):
            continue
    return max_freq


def _get_cpu_scaling_cur_freqs(logger=None):
    """
    Read scaling_cur_freq for all CPUs.

    Returns:
        Dict mapping cpu_num (int) to frequency in kHz, or empty dict if unavailable.
    """
    freqs = {}
    for path in sorted(_cpu_cpufreq_glob("scaling_cur_freq")):
        match = re.search(r'cpu(\d+)', path)
        if not match:
            continue
        cpu_num = int(match.group(1))
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    freq = int(content)
                    if freq > 0:
                        freqs[cpu_num] = freq
        except (IOError, OSError, ValueError):
            if logger:
                logger.warning("Cannot read scaling_cur_freq for cpu%d", cpu_num)
    return freqs


def _get_cpus_with_invalid_scaling_cur_freq(logger=None):
    """
    Return CPUs that expose cpufreq but do not report positive scaling_cur_freq.
    """
    cpufreq_cpus = set()
    for path in _cpu_cpufreq_glob("scaling_driver"):
        match = re.search(r'cpu(\d+)', path)
        if not match:
            continue
        cpu_num = int(match.group(1))
        if _is_cpu_online(cpu_num, logger=logger):
            cpufreq_cpus.add(cpu_num)

    freqs = _get_cpu_scaling_cur_freqs(logger=logger)
    return sorted(cpufreq_cpus - set(freqs))


def _is_amu_extension_enabled():
    """
    Check if CONFIG_ARM64_AMU_EXTN is enabled in the kernel.

    Returns:
        True if enabled, False if explicitly not enabled, None if the
        kernel config file cannot be read (e.g. embedded targets where
        /boot/config-$(uname -r) is unavailable).
    """
    config_path = "/boot/config-" + platform.uname().release
    try:
        with open(config_path, "r", encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line.startswith("CONFIG_ARM64_AMU_EXTN="):
                    return line.split("=", 1)[1] == "y"
        return False
    except (IOError, OSError, FileNotFoundError):
        return None


def _get_cpu_frequencies(cpu_list=None, logger=None):
    """
    Get CPU frequencies from cpuinfo_avg_freq interface.

    Args:
        cpu_list: Optional list of CPU numbers to filter. If None, returns all CPUs.
        logger: Optional logger for reporting anomalies.

    Returns:
        List of CPU frequencies in kHz, or empty list if interface unavailable.
    """
    paths = _cpu_cpufreq_glob("cpuinfo_avg_freq")
    if not paths:
        return []

    frequencies = []
    for path in sorted(paths):
        match = re.search(r'cpu(\d+)', path)
        if not match:
            continue
        cpu_num = int(match.group(1))

        if cpu_list is not None and cpu_num not in cpu_list:
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    continue
                freq = int(content)
                if freq <= 0:
                    if logger:
                        logger.warning("CPU%d reported non-positive frequency %d kHz "
                                       "- possible hardware issue", cpu_num, freq)
                    continue
                frequencies.append(freq)
        except (IOError, OSError, ValueError, TypeError):
            continue

    return frequencies


def _start_stress_ng_background():
    """
    Start stress-ng in background to stress all CPUs.

    Returns:
        subprocess.Popen: Process handle for the stress-ng process.
    """
    cmd = ["stress-ng", "--cpu", "0", "--cpu-method", "all",
           "--metrics-brief", "--temp-path", "/tmp",
           "--timeout", str(STRESS_NG_TIMEOUT_SECS)]
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    return process


def _terminate_stress_ng(process):
    """
    Terminate a stress-ng process and its children via process group.

    Args:
        process: subprocess.Popen handle to terminate.

    Returns:
        bool: True if process terminated successfully, False otherwise.
    """
    if process is None:
        return False
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
        process.wait(timeout=5)
        return True
    except (subprocess.TimeoutExpired, OSError):
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return False


def _cleanup_stress_ng(process):
    """Kill a specific stress-ng process group as a safety net."""
    if process is None or process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _get_all_cpu_governors():
    """Get current governor for all CPUs."""
    governors = {}
    for cpu_dir in _cpu_cpufreq_glob("scaling_governor"):
        try:
            cpu_match = re.search(r'cpu(\d+)', cpu_dir)
            if not cpu_match:
                continue
            cpu_num = int(cpu_match.group(1))
            with open(cpu_dir, 'r', encoding='utf-8') as f:
                governors[cpu_num] = f.read().strip()
        except (IOError, OSError):
            continue
    return governors


def _set_all_cpu_governors(target_governor, logger=None):
    """Set governor for all CPUs."""
    success_count = 0
    total_count = 0

    for cpu_dir in _cpu_cpufreq_glob("scaling_governor"):
        try:
            total_count += 1
            with open(cpu_dir, 'w', encoding='utf-8') as f:
                f.write(target_governor)
            success_count += 1
        except (IOError, OSError):
            continue

    if logger:
        logger.info("Set %d/%d CPUs to '%s' governor",
                   success_count, total_count, target_governor)

    return success_count, total_count


def _restore_cpu_governors(original_governors, logger=None):
    """Restore original governor settings for all CPUs."""
    success_count = 0
    for cpu_num, governor in original_governors.items():
        cpu_dir = f"/sys/devices/system/cpu/cpu{cpu_num}/cpufreq/scaling_governor"
        try:
            with open(cpu_dir, 'w', encoding='utf-8') as f:
                f.write(governor)
            success_count += 1
        except (IOError, OSError):
            continue

    if logger:
        logger.info("Restored %d/%d CPUs to original governors",
                   success_count, len(original_governors))


def _check_stress_ng_alive(process):
    """
    Check if stress-ng is still running.

    Returns:
        (True, None) if alive, (False, reason) if exited prematurely.
    """
    if process.poll() is None:
        return True, None
    stderr_output = ""
    try:
        _, stderr = process.communicate(timeout=2)
        stderr_output = stderr.decode(errors='replace').strip()
    except Exception:  # pylint: disable=broad-except
        pass
    reason = f"stress-ng exited prematurely with code {process.returncode}"
    if stderr_output:
        reason += f": {stderr_output}"
    return False, reason


def _validate_cppc_prerequisites(require_stress_ng=False):
    """
    Validate cppc_cpufreq prerequisites on the target system.

    Returns:
        (True, None) if all checks pass,
        (False, reason) on failure.
    """
    driver_paths = sorted(_cpu_cpufreq_glob("scaling_driver"))
    if not driver_paths:
        return False, "cpufreq sysfs not available; cannot read scaling_driver."
    non_cppc_drivers = []
    try:
        for driver_path in driver_paths:
            with open(driver_path, 'r', encoding='utf-8') as f:
                scaling_driver = f.read().strip()
            if scaling_driver != "cppc_cpufreq":
                non_cppc_drivers.append(f"{driver_path}: {scaling_driver}")
    except (IOError, OSError) as error:
        return False, f"Cannot read {driver_path}: {error}"

    if non_cppc_drivers:
        return False, ("cppc_cpufreq not active for all cpufreq policies "
                       f"({', '.join(non_cppc_drivers)}).")

    if require_stress_ng and not glt_utils.is_cli_available("stress-ng"):
        return False, ("stress-ng is not available. Please follow the user guide "
                       "to install it on your distribution or build it from source: "
                       "https://github.com/ColinIanKing/stress-ng")

    return True, None


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cppc_cpufreq_online_get_did_not_fail(_, logger):
    """
    Verify cppc_cpufreq did not fail CPU policy creation in the current boot.

    JIRA: DGX-10543
    PATCH: https://git.kernel.org/torvalds/c/c47195631960
    """
    ok, reason = _validate_cppc_prerequisites()
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    try:
        online_get_failed = glt_utils.check_kernel_log(
            CPUFREQ_ONLINE_GET_FAILED_MSG
        )
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    if online_get_failed:
        GLTCore.GLTTestResult.fail(
            f"Found cppc_cpufreq online failure in current boot log: "
            f"{CPUFREQ_ONLINE_GET_FAILED_MSG}. Apply patch "
            "c47195631960 (cppc_cpufreq: Use desired perf if feedback ctrs are 0 or unchanged)."
        )

    invalid_cpus = _get_cpus_with_invalid_scaling_cur_freq(logger=logger)
    if invalid_cpus:
        GLTCore.GLTTestResult.fail(
            "CPUs expose cppc_cpufreq but do not report positive scaling_cur_freq: "
            f"{invalid_cpus}. Apply patch c47195631960 "
            "(cppc_cpufreq: Use desired perf if feedback ctrs are 0 or unchanged)."
        )

    logger.info("PASS: cppc_cpufreq online get check completed successfully")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_cppc_cpufreq_policy_after_cpu_hotplug(_, logger):
    """
    Verify cppc_cpufreq policy is recreated after CPU hotplug.

    JIRA: DGX-17150
    PATCH: https://git.kernel.org/torvalds/c/c47195631960
    """
    if platform.machine() not in ("aarch64", "arm64"):
        GLTCore.GLTTestResult.skip("cppc_cpufreq hotplug policy check is ARM64-only.")

    ok, reason = _validate_cppc_prerequisites()
    if not ok:
        if _is_cppc_not_applicable(reason):
            GLTCore.GLTTestResult.skip(reason)
        GLTCore.GLTTestResult.fail(reason)

    try:
        online_cpus = _parse_cpu_mask(
            glt_utils.read_sysfs_value(os.path.join(CPU_SYSFS_ROOT, "online"))
        )
    except (IOError, OSError, ValueError) as e:
        GLTCore.GLTTestResult.fail(f"Cannot read online CPU topology: {e}")

    cpu_num, reason = _select_hotplug_cpu(online_cpus)
    if cpu_num is None:
        GLTCore.GLTTestResult.skip(reason)

    ok, reason = _validate_cppc_policy_for_cpu(cpu_num)
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    online_path = os.path.join(CPU_SYSFS_ROOT, f"cpu{cpu_num}", "online")
    try:
        checkpoint = glt_utils.capture_kernel_log_checkpoint()
    except RuntimeError as e:
        GLTCore.GLTTestResult.fail(
            f"Cannot capture kernel log checkpoint before hotplug: {e}"
        )

    logger.info("Cycling CPU%d offline/online for cppc_cpufreq policy validation", cpu_num)
    restore_reason = ""
    failure_reason = ""
    try:
        glt_utils.write_sysfs_value(online_path, "0")
        time.sleep(CPU_HOTPLUG_OFFLINE_SETTLE_SECS)
        if _is_cpu_online(cpu_num, logger=logger):
            GLTCore.GLTTestResult.fail(f"CPU{cpu_num} remained online after hotplug offline.")

        ok, reason = _verify_cppc_policy_absent_for_cpu(cpu_num)
        if not ok:
            GLTCore.GLTTestResult.fail(reason)

        glt_utils.write_sysfs_value(online_path, "1")
        ok, reason = _wait_for_cpu_online(cpu_num, logger=logger)
        if not ok:
            GLTCore.GLTTestResult.fail(reason)

        ok, reason = _wait_for_cppc_policy_for_cpu(cpu_num)
        if not ok:
            GLTCore.GLTTestResult.fail(reason)

        try:
            online_get_failed = glt_utils.check_kernel_log(
                CPUFREQ_ONLINE_GET_FAILED_MSG,
                checkpoint=checkpoint,
            )
        except RuntimeError as e:
            GLTCore.GLTTestResult.fail(f"Cannot inspect kernel log after hotplug: {e}")
        if online_get_failed:
            GLTCore.GLTTestResult.fail(
                f"Found cppc_cpufreq online failure after CPU hotplug: "
                f"{CPUFREQ_ONLINE_GET_FAILED_MSG}. Apply patch "
                "c47195631960 (cppc_cpufreq: Use desired perf if feedback ctrs are 0 or unchanged)."
            )
    except (IOError, OSError) as e:
        failure_reason = f"Cannot hotplug CPU{cpu_num}: {e}"
    except GLTCore.GLTTestFail as e:
        failure_reason = str(e)
    finally:
        try:
            glt_utils.write_sysfs_value(online_path, "1")
        except (IOError, OSError) as e:
            restore_reason = f"Failed to restore CPU{cpu_num} online: {e}"
            logger.warning(restore_reason)

    if failure_reason and restore_reason:
        GLTCore.GLTTestResult.fail(f"{failure_reason}; additionally, {restore_reason}")
    if failure_reason:
        GLTCore.GLTTestResult.fail(failure_reason)
    if restore_reason:
        GLTCore.GLTTestResult.fail(restore_reason)

    logger.info("PASS: CPU%d cppc_cpufreq policy recreated after hotplug", cpu_num)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_cpu_frequency_under_stress(_, logger):
    """
    Test CPU frequency behavior under stress load.
    Runs stress-ng and validates CPU frequencies are reasonable.

    JIRA: DGX-11735, DGX-11722, DGX-11725, DGX-11737, DGX-11728
    PATCH: https://lore.kernel.org/all/173982791748.4020779.2848639862581042284.b4-ty@arm.com/
           https://lore.kernel.org/all/20250131155842.3839098-1-beata.michalska@arm.com/
           https://lore.kernel.org/all/20240827154818.1195849-1-ionela.voinescu@arm.com/
           https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git/commit/?id=20711efa91e8ba44149f5e2ed1cf81e5355650e5
    """
    patches_msg = "Required patches:\n" + "\n".join(f"  - {p}" for p in REQUIRED_PATCHES)

    ok, reason = _validate_cppc_prerequisites(require_stress_ng=True)
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    amu_status = _is_amu_extension_enabled()
    if amu_status is None:
        GLTCore.GLTTestResult.skip(
            "Cannot determine AMU extension status: kernel config file not readable.")
    elif not amu_status:
        GLTCore.GLTTestResult.fail("CONFIG_ARM64_AMU_EXTN not enabled in kernel. "
                                   "This is required for cpuinfo_avg_freq support.")

    logger.info("Saving current CPU governor settings...")
    original_governors = _get_all_cpu_governors()
    if not original_governors:
        GLTCore.GLTTestResult.fail(
            "Cannot read CPU scaling_governor sysfs. "
            "This may happen if the test is not run as root "
            "or the cpufreq driver does not expose a governor interface.")

    performance_cpus = [cpu for cpu, gov in original_governors.items() if gov == 'performance']
    logger.info("Current governor state: %d/%d CPUs on performance governor",
               len(performance_cpus), len(original_governors))

    if len(performance_cpus) == len(original_governors):
        logger.info("All CPUs already on performance governor, skipping governor change")
    else:
        logger.info("Setting all CPUs to performance governor for testing...")
        success_count, _ = _set_all_cpu_governors('performance', logger)
        if success_count == 0:
            GLTCore.GLTTestResult.skip("Cannot set CPU governors to performance mode")

    stress_process = None
    try:
        baseline_frequencies = _get_cpu_frequencies(logger=logger)
        if not baseline_frequencies:
            GLTCore.GLTTestResult.fail(
                f"Cannot read baseline CPU frequencies. {patches_msg}")

        logger.info("Using cpuinfo_avg_freq interface - found %d CPUs with frequency data",
                   len(baseline_frequencies))

        baseline_min = min(baseline_frequencies) / 1000
        baseline_max = max(baseline_frequencies) / 1000
        baseline_avg = sum(baseline_frequencies) / len(baseline_frequencies) / 1000

        logger.info("Baseline frequencies: min=%.0f MHz, max=%.0f MHz, avg=%.0f MHz",
                   baseline_min, baseline_max, baseline_avg)

        logger.info("Starting stress-ng on all available CPUs...")
        stress_process = _start_stress_ng_background()

        try:
            logger.info("Waiting for frequencies to stabilize under stress...")
            time.sleep(FREQ_STABILIZATION_WAIT_SECS)

            alive, reason = _check_stress_ng_alive(stress_process)
            if not alive:
                GLTCore.GLTTestResult.fail(reason)

            logger.info("Measuring CPU frequencies under load...")
            post_frequencies = _get_cpu_frequencies(logger=logger)
            if not post_frequencies:
                GLTCore.GLTTestResult.fail(
                    f"Cannot read CPU frequencies under stress. {patches_msg}")
        finally:
            logger.info("Stopping stress-ng...")
            if not _terminate_stress_ng(stress_process):
                logger.warning("Failed to cleanly terminate stress-ng")

        min_freq = min(post_frequencies) / 1000
        max_freq = max(post_frequencies) / 1000
        avg_freq = sum(post_frequencies) / len(post_frequencies) / 1000
        freq_variance = ((max_freq - min_freq) / avg_freq) * 100 if avg_freq > 0 else 0

        high_freq_cpus = [f for f in post_frequencies if f / 1000 > HIGH_PERF_CPU_FREQ_MHZ]

        logger.info("Stress test results:")
        logger.info("  Frequency range: %.0f-%.0f MHz", min_freq, max_freq)
        logger.info("  Average frequency: %.0f MHz", avg_freq)
        logger.info("  Frequency variance: %.1f%%", freq_variance)
        logger.info("  High-performance CPUs (>%dMHz): %d/%d",
                   HIGH_PERF_CPU_FREQ_MHZ, len(high_freq_cpus), len(post_frequencies))

        freq_change = ((avg_freq - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
        logger.info("  Baseline -> stress avg change: %+.1f%%", freq_change)

        if freq_variance > MAX_FREQ_VARIANCE_PERCENT:
            GLTCore.GLTTestResult.fail(
                reason=f"CPU frequency variance too high: {freq_variance:.1f}% "
                       f"> {MAX_FREQ_VARIANCE_PERCENT}% "
                       f"(check thermal/power management). {patches_msg}"
            )

        logger.info("PASS: CPU frequency stress test completed successfully")

    finally:
        _cleanup_stress_ng(stress_process)
        logger.info("Restoring original CPU governor settings...")
        _restore_cpu_governors(original_governors, logger)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cpu_scaling_frequency(_, logger):
    """
    Verify each CPU scales to at least SCALING_FREQ_MIN_PERCENT of its rated
    max frequency under stress load on cppc_cpufreq platforms.

    Sets the performance governor, runs a stress workload, reads
    scaling_cur_freq per CPU, and fails if any CPU stays below the threshold.

    Assumes homogeneous cppc_cpufreq ARM platforms where all cores
    share the same rated max frequency.

    JIRA: DGX-11768
    PATCH: https://lore.kernel.org/all/OS3P286MB2490EB027398DDB852BE2169B1C02@OS3P286MB2490.JPNP286.PROD.OUTLOOK.COM/
    """
    ok, reason = _validate_cppc_prerequisites(require_stress_ng=True)
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    max_freq_khz = _get_cpu_max_freq_khz()
    if max_freq_khz is None:
        GLTCore.GLTTestResult.fail(
            "Cannot read cpuinfo_max_freq from sysfs. "
            "Ensure the cppc_cpufreq module is loaded.")

    min_expected_khz = max_freq_khz * SCALING_FREQ_MIN_PERCENT // 100
    logger.info("CPU max freq: %d MHz, minimum expected under stress: %d MHz (%d%%)",
                max_freq_khz // 1000, min_expected_khz // 1000, SCALING_FREQ_MIN_PERCENT)

    stress_process = None
    original_governors = {}
    try:
        original_governors = _get_all_cpu_governors()
        if not original_governors:
            GLTCore.GLTTestResult.fail(
                "Cannot read CPU scaling_governor sysfs. "
                "This may happen if the test is not run as root "
                "or the cpufreq driver does not expose a governor interface.")

        performance_cpus = [cpu for cpu, gov in original_governors.items() if gov == 'performance']
        logger.info("Current governor state: %d/%d CPUs on performance governor",
                    len(performance_cpus), len(original_governors))

        if len(performance_cpus) < len(original_governors):
            logger.info("Setting all CPUs to performance governor for testing...")
            success_count, _ = _set_all_cpu_governors('performance', logger)
            if success_count == 0:
                GLTCore.GLTTestResult.skip("Cannot set CPU governors to performance mode.")
        logger.info("Starting stress-ng on all CPUs...")

        stress_process = _start_stress_ng_background()

        try:
            logger.info("Waiting %ds for frequencies to stabilize...", FREQ_STABILIZATION_WAIT_SECS)
            time.sleep(FREQ_STABILIZATION_WAIT_SECS)

            alive, reason = _check_stress_ng_alive(stress_process)
            if not alive:
                GLTCore.GLTTestResult.fail(reason)

            logger.info("Reading per-CPU scaling_cur_freq under load...")
            freqs = _get_cpu_scaling_cur_freqs(logger=logger)
            if not freqs:
                GLTCore.GLTTestResult.fail(
                    "Cannot read scaling_cur_freq for any CPU under stress.")
        finally:
            if not _terminate_stress_ng(stress_process):
                logger.warning("Failed to cleanly terminate stress-ng")

        failing_cpus = {
            cpu_num: freq_khz
            for cpu_num, freq_khz in freqs.items()
            if freq_khz < min_expected_khz
        }

        for cpu_num, freq_khz in sorted(failing_cpus.items()):
            logger.error("CPU%d: %d MHz < %d MHz (%d%% of max)",
                         cpu_num, freq_khz // 1000, min_expected_khz // 1000,
                         SCALING_FREQ_MIN_PERCENT)

        if failing_cpus:
            GLTCore.GLTTestResult.fail(
                f"{len(failing_cpus)}/{len(freqs)} CPUs below "
                f"{SCALING_FREQ_MIN_PERCENT}% of max frequency ({max_freq_khz // 1000} MHz). "
                "Check thermal/power capping or apply patch: "
                "https://lore.kernel.org/all/"
                "OS3P286MB2490EB027398DDB852BE2169B1C02@OS3P286MB2490.JPNP286.PROD.OUTLOOK.COM/")

        min_freq = min(freqs.values()) // 1000
        max_freq = max(freqs.values()) // 1000
        logger.info("PASS: All %d CPUs reached >= %d%% of max frequency under stress. "
                    "Frequency range: %d-%d MHz",
                    len(freqs), SCALING_FREQ_MIN_PERCENT, min_freq, max_freq)

    finally:
        _cleanup_stress_ng(stress_process)
        _restore_cpu_governors(original_governors, logger)
