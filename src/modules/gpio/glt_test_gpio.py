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
Python script to test GPIO.
"""

import re
import subprocess
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra241_gpio_device(_, logger):
    """Test if could detect tegra 241 gpio devices.

    PATCH: https://git.kernel.org/torvalds/c/d1056b771ddbe119f733fd9
    """

    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("The cpu is not Grace.")

    # Check if can find the relation between NVDA0508 and gpiochip0
    target_msg = "DEVPATH=/devices/platform/NVDA0508:00/gpiochip0"
    result = subprocess.run("udevadm info -n /dev/gpiochip0", capture_output=True, shell=True, text=True, check=False)

    if target_msg not in result.stdout:
        GLTCore.GLTTestResult.fail("Cannot find NVDA0508:00 on gpiochip0.")

    # Check if can find tegra241-gpio 74 GPIOs from gpiodetect
    if glt_utils.is_cli_available("gpiodetect"):
        result = subprocess.run("sudo gpiodetect", capture_output=True, shell=True, text=True, check=False)

        if "gpiochip0 [tegra241-gpio] (74 lines)" not in result.stdout:
            GLTCore.GLTTestResult.fail("Can't detect tegra241-gpio 74 GPIOs. " +
                                       "Include kernel commit d1056b771ddb to fix it.")
    else:
        GLTCore.GLTTestResult.fail("Please install 'gpiodetect' to perform the test.")

    # Check if NVDA0508 has exactly 2 interrupts registered
    result = subprocess.run(
        "grep NVDA0508 /proc/interrupts | awk '{print $1, $(NF-3), $(NF-2), $(NF-1), $NF}'",
        capture_output=True, shell=True, text=True, check=False
    )
    grep_output = result.stdout.strip()
    interrupt_count = len(grep_output.splitlines()) if grep_output else 0

    if interrupt_count != 2:
        logger.error("Expected 2 NVDA0508 interrupts, got %d", interrupt_count)
        logger.error("grep NVDA0508 /proc/interrupts output:\n%s", grep_output if grep_output else "(empty)")
        GLTCore.GLTTestResult.fail(
            f"Expected 2 NVDA0508 interrupts in /proc/interrupts, but found {interrupt_count}. "
            "Include kernel commit d1056b771ddb to fix it."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra410_gpio_device(_, logger):
    """Test if could detect tegra410 gpio devices.

    PATCH: https://git.kernel.org/torvalds/c/9631a10083d8
    """

    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The cpu is not Vera.")

    if not glt_utils.is_cli_available("gpiodetect"):
        GLTCore.GLTTestResult.fail("Please install 'gpiodetect' to perform the test.")

    result = subprocess.run("gpiodetect", capture_output=True,
                            shell=True, text=True, check=False)
    gpiodetect_output = result.stdout

    logger.info("gpiodetect output:\n%s", gpiodetect_output)

    # Check for tegra410 GPIO controllers
    if "gpiochip0 [tegra410-gpio-compute] (30 lines)" not in gpiodetect_output or \
       "gpiochip1 [tegra410-gpio-system] (94 lines)" not in gpiodetect_output:
        GLTCore.GLTTestResult.fail(
            "Can't detect tegra410 GPIO controllers. "
            "Include kernel commit 9631a10083d8 to fix it."
        )


@GLTCore.GLTRequires.must_fail("gpio", "test_tegra241_gpio_device")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra241_gpio_break_graceful_shutdown(_, __):
    """Test if can break graceful shutdown.

    PATCH: https://git.kernel.org/torvalds/c/d806f474a9a7993648a2c70
    """
    check_kernel_msg = "tegra186-gpio NVDA0508:00: Failed to request GPIO for pin 0x0046, err -22"
    try:
        shutdown_error_found = glt_utils.check_kernel_log(check_kernel_msg)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    if shutdown_error_found:
        GLTCore.GLTTestResult.fail("Can't break graceful shutdown. " +
                                   "Include kernel commit d806f474a9a7 to fix it.")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_tegra410_gpio_name_collision(_, logger):
    """
    Test if there exists name collision between Compute GPIO and System GPIO on tegra410.

    JIRA: DGX-15534
    PATCH: https://git.kernel.org/torvalds/c/67f9b828d4e5e
    """

    # For now, only Vera is associated with Tegra410.
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The cpu is not Vera (tegra410).")

    # Check if tegra410 GPIO device is present or not
    if not glt_utils.is_cli_available("gpiodetect"):
        GLTCore.GLTTestResult.skip("Please install 'gpiodetect' to perform the test.")
    try:
        result = subprocess.run(
            "gpiodetect",
            capture_output=True, shell=True, text=True, check=True,
        )
        if "tegra410" not in result.stdout:
            GLTCore.GLTTestResult.fail("Tegra410 GPIO device not detected via gpiodetect.")
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to find tegra410 GPIO device via gpiodetect: %s", e)
        GLTCore.GLTTestResult.fail("Failed to find tegra410 GPIO device via gpiodetect.")


    collision_msg = "Detected name collision for GPIO name"
    try:
        collision_found = glt_utils.check_kernel_log(collision_msg)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    if collision_found:
        GLTCore.GLTTestResult.fail(
            "Found GPIO name collision in kernel log. "
            "Backport commit 67f9b828d4e5e to fix it, "
            "if you are using multi-socket system, also "
            "backport commit 2c299030c6."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_tegra410_gpio_numa_prefix(_, logger):
    """Verifies tegra410 GPIO line names are NUMA-prefixed on multi-socket Vera.

    JIRA: DGX-16336
    PATCH: https://git.kernel.org/torvalds/c/2c299030c6813eaa9ef95773c64d65c50fa706ac
    """

    line_name_re = re.compile(
        r"^\d+-(?:COMPUTE|SYSTEM)-P[A-Z]+\.[0-9a-fA-F]{2}$"
    )
    gpioinfo_line_re = re.compile(r'^\s*line\s+\d+:\s+"(?P<name>[^"]*)"')
    gpiodetect_re = re.compile(
        r'^(?P<chip>\S+)\s+\[(?P<label>[^\]]+)\]\s+\((?P<count>\d+)\s+lines\)'
    )

    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The test is only applicable to Vera CPUs.")

    socket_count = NvDev.get_cpu_socket_count()
    if socket_count == -1:
        GLTCore.GLTTestResult.fail(
            "Unable to determine CPU socket count on a Vera system."
            " Check that 'lscpu' is installed and runnable, and that its"
            " output contains a 'Socket(s):' line, or the CPU topology is"
            " reported incorrectly by the firmware, ACPI, etc."
        )
    elif socket_count < 2:
        GLTCore.GLTTestResult.skip(
            f"Single-socket system ({socket_count} socket); "
            "patch effect is only observable on multi-socket Vera."
        )

    if not glt_utils.is_cli_available("gpioinfo"):
        GLTCore.GLTTestResult.skip(
            "Please install 'gpiod' to perform the test."
        )

    # Discover tegra410 controllers by label
    try:
        detect = glt_utils.run_command(
            ["gpiodetect"],
            "gpiodetect failed to enumerate GPIO controllers",
        )
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(str(error))

    chips = []
    for line in detect.stdout.splitlines():
        m = gpiodetect_re.match(line)
        if m and m.group("label").startswith("tegra410-gpio-"):
            chips.append(m.group("chip"))

    if not chips:
        logger.warning("gpiodetect stdout:\n%s", detect.stdout)
        GLTCore.GLTTestResult.fail(
            "No tegra410 GPIO controllers detected on a Vera system; "
            "expected tegra410-gpio-compute and tegra410-gpio-system."
        )

    # Pull every line name from every tegra410 chip via gpioinfo.
    gpio_names = []
    for chip_name in chips:
        try:
            info = glt_utils.run_command(
                ["gpioinfo", chip_name],
                f"Failed to enumerate GPIO line names on {chip_name}",
            )
        except RuntimeError as error:
            GLTCore.GLTTestResult.fail(str(error))
        gpio_names.extend(
            mm.group("name")
            for mm in (gpioinfo_line_re.match(ln) for ln in info.stdout.splitlines())
            if mm is not None and mm.group("name")
        )

    if not gpio_names:
        GLTCore.GLTTestResult.fail(
            "gpioinfo produced no parseable GPIO line names."
        )

    for name in gpio_names:
        if not line_name_re.match(name):
            GLTCore.GLTTestResult.fail(
                f"Tegra410 GPIO line names don't follow the expected "
                f"'<numa_node>-<COMPUTE|SYSTEM>-P<bank>.<pin_hex>' shape "
                f"(e.g. '{name}'). "
                "Consider to backport commit 2c299030."
            )
