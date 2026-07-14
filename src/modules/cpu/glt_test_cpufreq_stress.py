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
import glob
import os
import platform
import re
import signal
import shutil
import subprocess
import time

from common import glt_core as GLTCore
from common import glt_utils
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
CPU_SYSFS_ROOT = "/sys/devices/system/cpu"
CPU_HOTPLUG_OFFLINE_SETTLE_SECS = 2
CPU_HOTPLUG_ONLINE_SETTLE_SECS = 3
CPU_HOTPLUG_POLICY_TIMEOUT_SECS = 15
CPU_HOTPLUG_POLICY_POLL_SECS = 1
KERNEL_LOG_COMMAND_TIMEOUT_SECS = 30

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


def _parse_cpu_mask(mask_text):
    """Parse a Linux CPU mask such as '0-3,8' into a set of CPU numbers."""
    cpus = set()
    mask_text = mask_text.strip()
    if not mask_text:
        return cpus

    for part in mask_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            cpus.add(int(part))
            continue

        start_text, end_text = part.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid CPU range: {part}")
        cpus.update(range(start, end + 1))
    return cpus


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
        driver = read_file(os.path.join(policy_dir, "scaling_driver")).strip()
        affected_cpus = _parse_cpu_mask(read_file(os.path.join(policy_dir, "affected_cpus")))
        related_cpus = _parse_cpu_mask(read_file(os.path.join(policy_dir, "related_cpus")))
    except (IOError, OSError, ValueError, KeyError) as e:
        return False, f"Cannot inspect {policy_dir}: {e}"

    if driver != "cppc_cpufreq":
        return False, f"{policy_dir} uses {driver}, expected cppc_cpufreq"
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


def _read_positive_policy_int(policy_dir, filename, read_file):
    path = os.path.join(policy_dir, filename)
    value = read_file(path).strip()
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{path} is not positive: {parsed}")
    return parsed


def _validate_cppc_policy_for_cpu(
        cpu_num, cpu_root=CPU_SYSFS_ROOT, path_exists=os.path.isdir,
        read_file=glt_utils.read_sysfs_value):
    """Validate a CPU has a recreated cppc_cpufreq policy with positive frequencies."""
    policy_dir = _get_cpufreq_policy_dir(cpu_num, cpu_root, path_exists)
    if not path_exists(policy_dir):
        return False, f"cpufreq policy directory missing for CPU{cpu_num}: {policy_dir}"

    try:
        driver = read_file(os.path.join(policy_dir, "scaling_driver")).strip()
        if driver != "cppc_cpufreq":
            return False, f"{policy_dir} uses {driver}, expected cppc_cpufreq"

        affected_cpus = _parse_cpu_mask(read_file(os.path.join(policy_dir, "affected_cpus")))
        related_cpus = _parse_cpu_mask(read_file(os.path.join(policy_dir, "related_cpus")))
        if affected_cpus != {cpu_num}:
            return False, f"{policy_dir}/affected_cpus is not CPU{cpu_num}: {sorted(affected_cpus)}"
        if related_cpus != {cpu_num}:
            return False, f"{policy_dir}/related_cpus is not CPU{cpu_num}: {sorted(related_cpus)}"

        _read_positive_policy_int(policy_dir, "scaling_cur_freq", read_file)
        _read_positive_policy_int(policy_dir, "cpuinfo_max_freq", read_file)
    except (IOError, OSError, ValueError, KeyError) as e:
        return False, f"Cannot validate {policy_dir} for CPU{cpu_num}: {e}"

    return True, ""


def _read_current_boot_kernel_log(preferred_source=None):
    """Return (source, text) for the current boot kernel log."""
    sources = [preferred_source] if preferred_source else ["journalctl", "dmesg"]

    for source in sources:
        if source == "journalctl":
            journalctl_path = shutil.which("journalctl")
            if journalctl_path:
                try:
                    result = subprocess.run(
                        [journalctl_path, "-b0", "-k"],
                        capture_output=True, text=True, check=False,
                        timeout=KERNEL_LOG_COMMAND_TIMEOUT_SECS)
                except subprocess.TimeoutExpired:
                    continue
                if result.returncode == 0:
                    return source, result.stdout
        if source == "dmesg":
            dmesg_path = shutil.which("dmesg")
            if dmesg_path:
                try:
                    result = subprocess.run(
                        [dmesg_path], capture_output=True, text=True, check=False,
                        timeout=KERNEL_LOG_COMMAND_TIMEOUT_SECS)
                except subprocess.TimeoutExpired:
                    continue
                if result.returncode == 0:
                    return source, result.stdout

    raise OSError("Cannot read current boot kernel log via journalctl or dmesg.")


def _count_kernel_log_occurrences(msg, preferred_source=None):
    """Return (source, count) for current-boot kernel log occurrences of msg."""
    source, output = _read_current_boot_kernel_log(preferred_source)
    return source, output.count(msg)


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


def _read_sysfs_int_value(path):
    """Read and validate a decimal or base-prefixed integer sysfs value."""
    try:
        value = glt_utils.read_sysfs_value(path)
    except (IOError, OSError) as e:
        raise OSError(f"cannot read {path}: {e}") from e
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
@GLTCore.GLTPlan("sanity")
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
    except (IOError, OSError):
        return False, "cpufreq sysfs not available; cannot read scaling_driver."

    if non_cppc_drivers:
        return False, ("cppc_cpufreq not active for all cpufreq policies "
                       f"({', '.join(non_cppc_drivers)}).")

    if require_stress_ng and not glt_utils.is_cli_available("stress-ng"):
        return False, ("stress-ng is not available. Please follow the user guide "
                       "to install it on your distribution or build it from source: "
                       "https://github.com/ColinIanKing/stress-ng")

    return True, None


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
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

    if glt_utils.check_kernel_log(CPUFREQ_ONLINE_GET_FAILED_MSG):
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
        log_source, online_get_failures_before = _count_kernel_log_occurrences(
            CPUFREQ_ONLINE_GET_FAILED_MSG
        )
    except OSError as e:
        GLTCore.GLTTestResult.fail(f"Cannot inspect kernel log before hotplug: {e}")

    logger.info("Cycling CPU%d offline/online for cppc_cpufreq policy validation", cpu_num)
    restore_reason = ""
    failure_reason = ""
    try:
        glt_utils.write_sysfs_value(online_path, "0")
        time.sleep(CPU_HOTPLUG_OFFLINE_SETTLE_SECS)
        if _is_cpu_online(cpu_num, logger=logger):
            GLTCore.GLTTestResult.fail(f"CPU{cpu_num} remained online after hotplug offline.")

        glt_utils.write_sysfs_value(online_path, "1")
        time.sleep(CPU_HOTPLUG_ONLINE_SETTLE_SECS)
        if not _is_cpu_online(cpu_num, logger=logger, unreadable_is_online=False):
            GLTCore.GLTTestResult.fail(f"CPU{cpu_num} did not come back online after hotplug.")

        ok, reason = _wait_for_cppc_policy_for_cpu(cpu_num)
        if not ok:
            GLTCore.GLTTestResult.fail(reason)

        try:
            _, online_get_failures_after = _count_kernel_log_occurrences(
                CPUFREQ_ONLINE_GET_FAILED_MSG, preferred_source=log_source
            )
        except OSError as e:
            GLTCore.GLTTestResult.fail(f"Cannot inspect kernel log after hotplug: {e}")
        if online_get_failures_after > online_get_failures_before:
            GLTCore.GLTTestResult.fail(
                f"Found cppc_cpufreq online failure after CPU hotplug: "
                f"{CPUFREQ_ONLINE_GET_FAILED_MSG}. Apply patch "
                "c47195631960 (cppc_cpufreq: Use desired perf if feedback ctrs are 0 or unchanged)."
            )
        if online_get_failures_before:
            logger.warning(
                "Current boot log already contained %d occurrence(s) of %s before CPU hotplug.",
                online_get_failures_before,
                CPUFREQ_ONLINE_GET_FAILED_MSG,
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
