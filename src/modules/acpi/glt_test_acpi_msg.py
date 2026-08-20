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
Python script to test ACPI messages.
"""

import os
import platform
import re

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_acpi_hmat_msg(_, __):
    """
    Test if ACPI HMAT info messages can be found in dmesg.

    JIRA: DGX-8205
    PATCH: https://git.kernel.org/torvalds/linux/c/c8e88de1b44e v6.14+
    """
    acpi_hmat_info_msg = "acpi/hmat:"

    try:
        hmat_message_found = glt_utils.check_kernel_log(acpi_hmat_info_msg)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    if hmat_message_found:
        GLTCore.GLTTestResult.fail(f"Found '{acpi_hmat_info_msg}' in kernel message. " +
                                   "Include kernel commit c8e88de1b44e to fix it.")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_no_legacy_tegra_sku_log_on_acpi(_, logger):
    """
    Verify that on ACPI-booted systems that Tegra would not
    register SoC device in sysfs nor does it print legacy SKU
    info in kernel logs.

    JIRA: DGX-15535
    PATCH: https://git.kernel.org/torvalds/linux/c/c87f820bc
    """

    if not os.path.isdir("/sys/firmware/acpi"):
        GLTCore.GLTTestResult.skip(
            "Non-ACPI boot (no /sys/firmware/acpi); test not applicable."
        )

    # Check if tegra-fuse driver is present
    platform_drivers_path = "/sys/bus/platform/drivers"
    if os.path.isdir(platform_drivers_path):
        drivers = os.listdir(platform_drivers_path)
        if "tegra-fuse" not in drivers:
            GLTCore.GLTTestResult.skip(
                "tegra-fuse driver not found in /sys/bus/platform/drivers; test not applicable."
            )
    else:
        GLTCore.GLTTestResult.skip(
            "Platform drivers path not found; test not applicable."
        )

    try:
        kernel_log = glt_utils.get_kernel_log()
    except RuntimeError as error:
        logger.error(f"Failed to collect kernel logs: {error}")
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    if re.search(r"tegra-fuse.*error.*-EINVAL.*Unsupported SoC", kernel_log):
        GLTCore.GLTTestResult.skip(
            "tegra-fuse driver loaded but failed to initialize (Unsupported SoC); test not applicable."
        )

    if "Tegra Revision:" in kernel_log:
        GLTCore.GLTTestResult.fail(
            "Legacy Tegra SKU info found in kernel log (Tegra Revision:). Backport commit c87f820 to fix it."
        )

    # Only soc0 should exists under /sys/devices
    sys_devices = "/sys/devices"
    if os.path.isdir(sys_devices):
        soc_entries = [
            name for name in os.listdir(sys_devices)
            if name.startswith("soc")
        ]
        if set(soc_entries) != {"soc0"}:
            GLTCore.GLTTestResult.fail(
                f"Under {sys_devices} only soc0 should exist, found: {sorted(soc_entries)}. "
                "Backport commit c87f820 to fix it."
            )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra_fuse_probe(_, __):
    """
    Verify tegra-fuse driver probes successfully on Grace (Tegra241).

    JIRA: DGX-10631
    PATCH: https://github.com/torvalds/linux/commit/8402074
    """
    if not os.path.isdir("/sys/firmware/acpi"):
        GLTCore.GLTTestResult.skip("Non-ACPI boot; test not applicable.")

    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("Not a Grace CPU; test not applicable.")

    if not os.path.isdir("/sys/bus/platform/devices/NVDA200F:00"):
        GLTCore.GLTTestResult.skip("NVDA200F:00 device not found; tegra-fuse not applicable.")

    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
        if kconfig.get("CONFIG_ARCH_TEGRA_241_SOC") != "y":
            GLTCore.GLTTestResult.fail(
                "CONFIG_ARCH_TEGRA_241_SOC is not enabled in kernel config."
                " Enable CONFIG_ARCH_TEGRA_241_SOC=y and rebuild the kernel to fix it."
            )
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"Cannot read kernel config at {kconfig_path};"
            " cannot verify CONFIG_ARCH_TEGRA_241_SOC."
        )

    # Check driver bound to the ACPI device
    if not os.path.exists("/sys/bus/platform/drivers/tegra-fuse/NVDA200F:00"):
        GLTCore.GLTTestResult.fail(
            "tegra-fuse driver did not bind to NVDA200F:00."
            " Consider backporting commit 8402074 to fix it."
        )

    try:
        probe_error_found = glt_utils.check_kernel_log(
            "tegra-fuse NVDA200F:00: error -EINVAL: Unsupported SoC"
        )
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    if probe_error_found:
        GLTCore.GLTTestResult.fail(
            "tegra-fuse probe error found in kernel log."
            " Consider backporting commit 8402074 to fix it."
        )
