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
Python script to verify ARM FF-A transport availability.
"""

import os
import platform
import subprocess

from common import glt_core as GLTCore
from common import glt_utils
from scripts import check_kernel_config


ARM_FFA_DMESG_PATTERN = "ARM FF-A:"
# Stable alias used across distros to query the FF-A module.
# The real module name varies (e.g. 'arm_ffa' on some kernels,
# 'ffa_module' on RHEL 10 / Ubuntu 24.04).
ARM_FFA_MODULE_ALIAS = "arm-ffa"
ARM_FFA_CONFIG = "CONFIG_ARM_FFA_TRANSPORT"
ARM_FFA_SMCCC_CONFIG = "CONFIG_ARM_FFA_SMCCC"
ARM_FFA_SYSFS_DEVICES_PATH = "/sys/bus/arm_ffa/devices"
ARM_FFA_RXTX_MAP_FAILURE_PATTERN = "failed to register FFA RxTx buffers"
ARM_FFA_REQUIRED_PARTITION_ATTRS = ("partition_id", "uuid", "modalias")


def _query_ffa_modinfo():
    """
    Resolve the FF-A module's real name and load type via the 'arm-ffa' alias.

    Returns:
        (module_name, is_builtin) — module_name is None if modinfo failed;
        is_builtin is True when modinfo reports 'filename: (builtin)'.
    """
    result = subprocess.run(
        ["modinfo", ARM_FFA_MODULE_ALIAS],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, shell=False,
    )
    if result.returncode != 0:
        return None, False

    module_name = None
    is_builtin = False
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "name":
            module_name = value
        elif key == "filename":
            is_builtin = value == "(builtin)"
    return module_name, is_builtin


def _validate_ffa_partition_attrs(partitions):
    """Verify each discovered FF-A partition exposes its expected sysfs data."""
    for partition in partitions:
        partition_path = os.path.join(ARM_FFA_SYSFS_DEVICES_PATH, partition)
        for attr_name in ARM_FFA_REQUIRED_PARTITION_ATTRS:
            attr_path = os.path.join(partition_path, attr_name)
            try:
                with open(attr_path, "r", encoding="utf-8") as attr_file:
                    attr_value = attr_file.read().strip()
            except OSError as err:
                return False, f"Failed to read {attr_path}: {err}"

            if not attr_value:
                return False, f"ARM FF-A partition {partition} has empty {attr_name}"

    return True, ""


def _validate_ffa_runtime():
    """Return discovered partitions and any FF-A runtime failure reason."""
    if glt_utils.check_kernel_log(ARM_FFA_RXTX_MAP_FAILURE_PATTERN):
        return [], (
            "ARM FF-A RXTX_MAP registration failed. Verify the kernel includes "
            "'firmware: arm_ffa: Respect firmware advertised RX/TX buffer size "
            "limits'."
        )

    if not os.path.isdir(ARM_FFA_SYSFS_DEVICES_PATH):
        return [], (
            f"ARM FF-A bus not registered: {ARM_FFA_SYSFS_DEVICES_PATH} missing. "
            "Probe likely failed; check dmesg for ARM FF-A errors."
        )

    try:
        partitions = sorted(os.listdir(ARM_FFA_SYSFS_DEVICES_PATH))
    except OSError as err:
        return [], (
            f"Failed to list ARM FF-A partitions in "
            f"{ARM_FFA_SYSFS_DEVICES_PATH}: {err}"
        )

    if not partitions:
        return [], (
            f"ARM FF-A bus registered but no partitions were discovered in "
            f"{ARM_FFA_SYSFS_DEVICES_PATH}. Firmware may not be exposing any "
            "FF-A endpoints, or probe failed after bus registration."
        )

    valid, reason = _validate_ffa_partition_attrs(partitions)
    if not valid:
        return [], reason

    return partitions, ""


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_ffa_availability(_, logger):
    """
    Verify ARM FF-A transport is externally usable without internal diags.

    This test validates:
    1. CONFIG_ARM_FFA_TRANSPORT is enabled as built-in or module
    2. CONFIG_ARM_FFA_SMCCC is enabled
    3. The FF-A module is discoverable via its 'arm-ffa' alias (modinfo)
    4. If modular, the resolved module is loadable and visible in lsmod
    5. RXTX_MAP did not report the firmware buffer-size regression
    6. The arm_ffa sysfs bus exposes usable partition attributes
    7. The running system shows FF-A initialization in the kernel log

    PATCH: https://patch.msgid.link/20260602-b4-ffa-rxtx-map-fixes-v2-1-7cb06508da84@nvidia.com
    """
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {kconfig_path}, "
            f"cannot verify {ARM_FFA_CONFIG} and {ARM_FFA_SMCCC_CONFIG}"
        )

    transport = kconfig.get(ARM_FFA_CONFIG)
    smccc = kconfig.get(ARM_FFA_SMCCC_CONFIG)

    if smccc != "y":
        GLTCore.GLTTestResult.fail(
            f"{ARM_FFA_SMCCC_CONFIG} must be enabled (=y); found {smccc!r}"
        )

    if transport not in ("y", "m"):
        GLTCore.GLTTestResult.fail(
            f"{ARM_FFA_CONFIG} must be enabled as built-in or module; found {transport!r}"
        )

    module_name, is_builtin = _query_ffa_modinfo()
    if module_name is None:
        GLTCore.GLTTestResult.fail(
            f"modinfo could not resolve the '{ARM_FFA_MODULE_ALIAS}' alias; "
            "FF-A module info is unavailable for this kernel."
        )
    logger.info("ARM FF-A module resolved: name=%s, builtin=%s",
                module_name, is_builtin)

    if transport == "y" or is_builtin:
        logger.info("ARM FF-A transport mode: builtin")
    else:
        logger.info("ARM FF-A transport mode: module")

        result = subprocess.run(
            ["modprobe", module_name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "modprobe failed"
            GLTCore.GLTTestResult.fail(
                f"Failed to load {module_name}: {detail}"
            )

        try:
            result = subprocess.run(
                ["lsmod"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
        except subprocess.CalledProcessError as err:
            GLTCore.GLTTestResult.fail(f"Failed to inspect loaded modules: {err}")

        if not any(
            line.split()[0] == module_name
            for line in result.stdout.splitlines()[1:]
        ):
            GLTCore.GLTTestResult.fail(
                f"{module_name} did not appear in lsmod after modprobe"
            )

    partitions, reason = _validate_ffa_runtime()
    if reason:
        GLTCore.GLTTestResult.fail(reason)

    logger.info("ARM FF-A bus registered with %d partition(s): %s",
                len(partitions), ", ".join(partitions))

    if not glt_utils.check_kernel_log(ARM_FFA_DMESG_PATTERN):
        logger.warning(
            "ARM FF-A initialization message not found in kernel log: %s "
            "(kernel log may have rotated)",
            ARM_FFA_DMESG_PATTERN,
        )
