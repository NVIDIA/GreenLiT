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
import glob
import os
import platform
import re
import signal
import subprocess
import time

from common import glt_core as GLTCore
from common import glt_utils

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


def _validate_cppc_stress_prerequisites():
    """
    Validate common prerequisites for CPU frequency stress tests.

    Returns:
        (True, None) if all checks pass,
        (False, reason) on failure.
    """
    driver_paths = sorted(_cpu_cpufreq_glob("scaling_driver"))
    if not driver_paths:
        return False, "cpufreq sysfs not available; cannot read scaling_driver."
    try:
        with open(driver_paths[0], 'r', encoding='utf-8') as f:
            scaling_driver = f.read().strip()
        if scaling_driver != "cppc_cpufreq":
            return False, (f"cppc_cpufreq not active "
                           f"(current driver: {scaling_driver}).")
    except (IOError, OSError):
        return False, "cpufreq sysfs not available; cannot read scaling_driver."

    if not glt_utils.is_cli_available("stress-ng"):
        return False, ("stress-ng is not available. Please follow the user guide "
                       "to install it on your distribution or build it from source: "
                       "https://github.com/ColinIanKing/stress-ng")

    return True, None


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

    ok, reason = _validate_cppc_stress_prerequisites()
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
    ok, reason = _validate_cppc_stress_prerequisites()
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
