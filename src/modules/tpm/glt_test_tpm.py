# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test tpm.
"""
import os
import platform
import subprocess
import time
from common import glt_core as GLTCore
from common import glt_utils
from scripts import check_kernel_config

def _stop_stress_processes(process1, process2):
    """Terminate the two stress processes; kill them if they do not exit in time."""
    for proc in (process1, process2):
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_tpm(_, logger):
    """Run TPM tests"""
    logger.info("PASS.")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra_quad_spi(_, logger):  # pylint: disable=too-many-branches
    """
    Test tegra-quad-spi driver under stress conditions.
    Run two stress-ng workloads in parallel to trigger the driver:
      - stress-ng --hrtimers 0 --taskset 0 --timeout 3m
      - taskset -c 0 chrt -f 50 stress-ng --cpu 1 --cpu-load 100 --timeout 3m
    Check whether the driver reports interrupt timeout errors.

    JIRA: DGX-15527, DGX-15528, DGX-15529
    PATCH: https://lore.kernel.org/linux-tegra/20260116-tegra_xfer-v1-0-02d96c790619@debian.org/T/#u
    """

    if not os.path.exists("/dev/tpm0") or not os.path.exists("/dev/tpmrm0"):
        GLTCore.GLTTestResult.skip("SKIPPED: Missing TPM devices (/dev/tpm0 or /dev/tpmrm0 not exist)")

    # Check that /dev/tpm0 is on the tegra QSPI controller (NVDA1513) via udev DEVPATH
    try:
        udev = subprocess.run(
            ["udevadm", "info", "/dev/tpm0"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"udevadm info /dev/tpm0 failed unexpectedly: {e}")
    devpath = ""
    for line in udev.stdout.splitlines():
        if "DEVPATH=" in line:
            devpath = line.split("DEVPATH=", 1)[-1].strip()
            break
    if "NVDA1513" not in devpath:
        GLTCore.GLTTestResult.skip(
            "SKIPPED: missing QSPI device"
        )

    if not glt_utils.is_cli_available("stress-ng"):
        GLTCore.GLTTestResult.skip(
            "SKIPPED: stress-ng is not installed or not usable (required for this test)"
        )

    # Monitor kernel logs while both processes run
    poll_interval = 5  # Check every 5 seconds
    max_runtime = 210  # Maximum 210 seconds
    elapsed_time = 0
    error_msg = "QSPI interrupt timeout, but transfer complete"
    stress_process_1 = None
    stress_process_2 = None

    try:
        # stress-ng hrtimers on CPU 0, 3 minutes
        stress_process_1 = subprocess.Popen(  # pylint: disable=consider-using-with
            ["stress-ng", "--hrtimers", "0", "--taskset", "0", "--timeout", "3m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"stress-ng (hrtimers) started with PID: {stress_process_1.pid}")

        # high-priority stress-ng CPU load on CPU 0, 3 minutes
        stress_process_2 = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                "taskset", "-c", "0",
                "chrt", "-f", "50",
                "stress-ng", "--cpu", "1", "--cpu-load", "100", "--timeout", "3m",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"stress-ng (cpu) started with PID: {stress_process_2.pid}")

        while elapsed_time < max_runtime:
            # Check kernel log for the error message
            if glt_utils.check_kernel_log(error_msg):
                logger.error(f"Error detected in dmesg: {error_msg}")
                logger.info("Terminating stress processes")
                _stop_stress_processes(stress_process_1, stress_process_2)
                GLTCore.GLTTestResult.fail(
                    f"FAIL: Found tegra-qspi interrupt timeout error in dmesg: '{error_msg}'. "
                    "Under load the driver can time out waiting for the completion interrupt "
                    "(transfer may already be done). Consider backporting commit 9b7977f to fix it. "
                    "Refer to https://lore.kernel.org/linux-tegra/20260116-tegra_xfer-v1-0-02d96c790619"
                    "@debian.org/T/#u for details."
                )

            time.sleep(poll_interval)
            elapsed_time += poll_interval

        logger.warning(f"Maximum runtime ({max_runtime}s) reached, terminating stress processes")
        _stop_stress_processes(stress_process_1, stress_process_2)

    except (OSError, ValueError) as e:
        _stop_stress_processes(stress_process_1, stress_process_2)
        GLTCore.GLTTestResult.fail(f"FAIL: Error during test execution: {e}")

    # Final check of kernel log after both processes complete
    if glt_utils.check_kernel_log(error_msg):
        GLTCore.GLTTestResult.fail(
            f"FAIL: Found tegra-qspi interrupt timeout error in dmesg: '{error_msg}'. "
            "Under load the driver can time out waiting for the completion interrupt "
            "(transfer may already be done). Consider backporting commit 9b7977f to fix it. "
            "Refer to https://lore.kernel.org/linux-tegra/20260116-tegra_xfer-v1-0-02d96c790619"
            "@debian.org/T/#u for details."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra_qspi_xfer_kasan_oob(_, __):
    """
    Checking KASAN slab-out-of-bounds splat in the Tegra QSPI handle_cpu_based_xfer path
    by reading current-boot kernel log.

    JIRA: DGX-8750
    PATCH: https://github.com/torvalds/linux/commit/195aba96
    """

    if not os.path.exists("/dev/tpm0") or not os.path.exists("/dev/tpmrm0"):
        GLTCore.GLTTestResult.skip("Missing TPM devices (/dev/tpm0 or /dev/tpmrm0).")

    if not glt_utils.is_cli_available("udevadm"):
        GLTCore.GLTTestResult.skip("udevadm is not available.")

    try:
        udev = subprocess.run(
            ["udevadm", "info", "/dev/tpm0"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"udevadm info /dev/tpm0 failed unexpectedly: {e}")
    devpath = ""
    for line in udev.stdout.splitlines():
        if "DEVPATH=" in line:
            devpath = line.split("DEVPATH=", 1)[-1].strip()
            break
    if "NVDA1513" not in devpath:
        GLTCore.GLTTestResult.skip("TPM is not on the Tegra QSPI controller (NVDA1513).")

    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(f"Cannot read kernel config at {kconfig_path}.")

    if kconfig.get("CONFIG_TCG_TIS_SPI") not in ("y", "m"):
        GLTCore.GLTTestResult.skip("CONFIG_TCG_TIS_SPI is not enabled.")
    if kconfig.get("CONFIG_KASAN") != "y":
        GLTCore.GLTTestResult.skip("CONFIG_KASAN is not enabled.")

    splat = "KASAN: slab-out-of-bounds in handle_cpu_based_xfer"
    if glt_utils.check_kernel_log(splat):
        GLTCore.GLTTestResult.fail(
            f"Found KASAN slab-out-of-bounds splat '{splat}' in kernel log. "
            "Consider backporting commit 195aba9."
        )
