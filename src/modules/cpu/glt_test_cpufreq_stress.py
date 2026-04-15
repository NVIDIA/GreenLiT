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

REQUIRED_PATCHES = (
    "https://lore.kernel.org/all/173982791748.4020779.2848639862581042284.b4-ty@arm.com/",
    "https://lore.kernel.org/all/20250131155842.3839098-1-beata.michalska@arm.com/",
    "https://lore.kernel.org/all/20240827154818.1195849-1-ionela.voinescu@arm.com/",
    "https://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git/commit/"
    "?id=20711efa91e8ba44149f5e2ed1cf81e5355650e5",
)


def is_amu_extension_enabled():
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


def get_cpu_frequencies(cpu_list=None, logger=None):
    """
    Get CPU frequencies from cpuinfo_avg_freq interface.

    Args:
        cpu_list: Optional list of CPU numbers to filter. If None, returns all CPUs.
        logger: Optional logger for reporting anomalies.

    Returns:
        List of CPU frequencies in kHz, or empty list if interface unavailable.
    """
    paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_avg_freq")
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


def start_stress_ng_background():
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


def terminate_stress_ng(process):
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


def cleanup_stress_ng(process):
    """Kill a specific stress-ng process group as a safety net."""
    if process is None or process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass


def get_all_cpu_governors():
    """Get current governor for all CPUs."""
    governors = {}
    for cpu_dir in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            cpu_match = re.search(r'cpu(\d+)', cpu_dir)
            if cpu_match:
                cpu_num = int(cpu_match.group(1))
                with open(cpu_dir, 'r', encoding='utf-8') as f:
                    governors[cpu_num] = f.read().strip()
        except (IOError, OSError):
            continue
    return governors


def set_all_cpu_governors(target_governor, logger=None):
    """Set governor for all CPUs."""
    success_count = 0
    total_count = 0

    for cpu_dir in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
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


def restore_cpu_governors(original_governors, logger=None):
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
    """Fail the test if stress-ng exited prematurely, including its stderr."""
    if process.poll() is None:
        return
    stderr_output = ""
    try:
        _, stderr = process.communicate(timeout=2)
        stderr_output = stderr.decode(errors='replace').strip()
    except Exception:  # pylint: disable=broad-except
        pass
    reason = f"stress-ng exited prematurely with code {process.returncode}"
    if stderr_output:
        reason += f": {stderr_output}"
    GLTCore.GLTTestResult.fail(reason=reason)


def _validate_stress_test_prerequisites():
    """Validate system prerequisites for CPU frequency stress test."""
    if not glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/"):
        GLTCore.GLTTestResult.fail(
            "CPU frequency scaling sysfs not available. "
            "Ensure the cppc_cpufreq module is loaded (modprobe cppc_cpufreq).")

    if not glt_utils.is_cli_available("stress-ng"):
        GLTCore.GLTTestResult.fail(
            "stress-ng is not available. Please follow the user guide to install "
            "it on your distribution or build it from source: "
            "https://github.com/ColinIanKing/stress-ng")

    amu_status = is_amu_extension_enabled()
    if amu_status is None:
        GLTCore.GLTTestResult.skip(
            "Cannot determine AMU extension status: kernel config file not readable.")
    elif not amu_status:
        GLTCore.GLTTestResult.fail("CONFIG_ARM64_AMU_EXTN not enabled in kernel. "
                                 "This is required for cpuinfo_avg_freq support.")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
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

    _validate_stress_test_prerequisites()

    logger.info("Saving current CPU governor settings...")
    original_governors = get_all_cpu_governors()
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
        success_count, _ = set_all_cpu_governors('performance', logger)
        if success_count == 0:
            GLTCore.GLTTestResult.skip("Cannot set CPU governors to performance mode")

    stress_process = None
    try:
        baseline_frequencies = get_cpu_frequencies(logger=logger)
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
        stress_process = start_stress_ng_background()

        try:
            logger.info("Waiting for frequencies to stabilize under stress...")
            time.sleep(FREQ_STABILIZATION_WAIT_SECS)

            _check_stress_ng_alive(stress_process)

            logger.info("Measuring CPU frequencies under load...")
            post_frequencies = get_cpu_frequencies(logger=logger)
            if not post_frequencies:
                GLTCore.GLTTestResult.fail(
                    f"Cannot read CPU frequencies under stress. {patches_msg}")
        finally:
            logger.info("Stopping stress-ng...")
            if not terminate_stress_ng(stress_process):
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
        cleanup_stress_ng(stress_process)
        logger.info("Restoring original CPU governor settings...")
        restore_cpu_governors(original_governors, logger)
