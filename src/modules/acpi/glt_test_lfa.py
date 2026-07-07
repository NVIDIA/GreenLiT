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

"""
Python script to test ARM Live Firmware Activation (LFA) availability.

LFA (Live Firmware Activation) allows firmware updates to be applied without
requiring a full system reboot. This test validates that the LFA driver is
properly initialized and the sysfs interface is available.
"""
import os
import platform
import re
from common import glt_core as GLTCore
from common import glt_utils
from common import nv_device
from scripts import check_kernel_config

nvdev = nv_device.NvDev()


# Path to the LFA sysfs directory
LFA_SYSFS_PATH = "/sys/firmware/lfa"

# Expected LFA dmesg pattern (simple string for grep matching)
LFA_DMESG_PATTERN = "Arm LFA: Live Firmware Activation"

# Expected sysfs attributes for each LFA component
LFA_EXPECTED_ATTRS = {
    "activate",
    "activation_capable",
    "activation_pending",
    "cancel",
    "pending_version",
}


def get_lfa_components(logger):
    """
    Get LFA component directories (UUID-named directories).

    Args:
        logger: Logger object for logging

    Returns:
        list: List of LFA component directory paths
    """
    if not os.path.exists(LFA_SYSFS_PATH):
        return []

    components = []
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

    try:
        for entry in os.listdir(LFA_SYSFS_PATH):
            entry_path = os.path.join(LFA_SYSFS_PATH, entry)
            if os.path.isdir(entry_path) and uuid_pattern.match(entry):
                components.append(entry_path)
                logger.debug("Found LFA component: %s", entry)
    except (OSError, PermissionError) as e:
        logger.warning("Error accessing LFA sysfs directory: %s", e)
        return []

    return components


def check_lfa_component_attributes(component_path, logger):
    """
    Check if an LFA component has the expected attributes.

    Args:
        component_path: Path to the LFA component directory
        logger: Logger object for logging

    Returns:
        tuple: (bool, set, set) - (all present, found attributes, missing attributes)
    """
    found_attrs = set()
    try:
        for entry in os.listdir(component_path):
            if entry in LFA_EXPECTED_ATTRS:
                found_attrs.add(entry)
                logger.debug("Found attribute %s in %s", entry, component_path)
    except (OSError, PermissionError) as e:
        logger.warning("Error accessing LFA component directory %s: %s", component_path, e)
        return False, set(), LFA_EXPECTED_ATTRS

    missing_attrs = LFA_EXPECTED_ATTRS - found_attrs
    return not missing_attrs, found_attrs, missing_attrs


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_lfa_availability(_, logger):
    """
    Test if ARM Live Firmware Activation (LFA) is available.

    This test checks:
    1. LFA driver initialization messages in dmesg
    2. LFA sysfs interface at /sys/firmware/lfa
    3. LFA component directories and their attributes

    LFA allows firmware updates without full system reboot.
    This test is only valid for Vera CPU platforms.

    Kernel config: Requires CONFIG_ARM_LFA=y/m.

    JIRA: DGX-15670
    PATCH: N/A
    """
    # LFA is only available on Vera CPU platforms
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("LFA is only available on Vera CPU platforms")

    # Check if CONFIG_ARM_LFA is enabled in the running kernel
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
        if kconfig.get("CONFIG_ARM_LFA") not in ("y", "m"):
            GLTCore.GLTTestResult.fail(
                "CONFIG_ARM_LFA is not enabled in kernel config"
            )
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {kconfig_path}, "
            "cannot verify CONFIG_ARM_LFA"
        )

    # Check if LFA sysfs directory exists
    if not os.path.exists(LFA_SYSFS_PATH):
        GLTCore.GLTTestResult.skip(
            f"LFA sysfs interface not available ({LFA_SYSFS_PATH} does not exist). "
            "LFA may not be supported on this platform."
        )

    # Check dmesg for LFA detection message
    logger.info("Checking kernel log for LFA driver messages...")
    try:
        lfa_message_found = glt_utils.check_kernel_log(LFA_DMESG_PATTERN)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    if lfa_message_found:
        logger.info("Found LFA dmesg pattern: %s", LFA_DMESG_PATTERN)
        dmesg_ok = True
    else:
        logger.warning("LFA dmesg pattern not found: %s (kernel log may have rotated)",
                       LFA_DMESG_PATTERN)
        dmesg_ok = False

    # Get and validate LFA components
    logger.info("Checking LFA components...")
    components = get_lfa_components(logger)
    if not components:
        GLTCore.GLTTestResult.fail(
            "No LFA component directories found under /sys/firmware/lfa. "
            "LFA may not be properly configured."
        )

    logger.info("Found %d LFA component(s)", len(components))

    # Check attributes for each component, collecting all failures
    failed_components = []
    for component in components:
        component_name = os.path.basename(component)
        attrs_ok, found_attrs, missing_attrs = check_lfa_component_attributes(component, logger)
        logger.info("Component %s has attributes: %s", component_name, found_attrs)

        if not attrs_ok:
            failed_components.append((component_name, missing_attrs))
            logger.warning("Component %s is missing attributes: %s",
                           component_name, missing_attrs)

    if failed_components:
        details = "; ".join(
            f"{name} missing: {attrs}" for name, attrs in failed_components
        )
        GLTCore.GLTTestResult.fail(
            f"LFA components with missing required attributes: {details}"
        )

    # Log summary
    logger.info("LFA availability test summary:")
    logger.info("  - Sysfs path: %s", LFA_SYSFS_PATH)
    logger.info("  - Components found: %d", len(components))
    logger.info("  - dmesg check: %s", "PASS" if dmesg_ok else "WARN (not critical)")

    logger.info("PASS: LFA is available and properly configured")
