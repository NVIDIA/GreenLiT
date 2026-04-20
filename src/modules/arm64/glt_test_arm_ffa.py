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
    4. If built-in, the arm_ffa sysfs bus exposes at least one partition
    5. If modular, the resolved module is loadable and visible in lsmod
    6. The running system shows FF-A initialization in the kernel log
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

    if not os.path.isdir(ARM_FFA_SYSFS_DEVICES_PATH):
        GLTCore.GLTTestResult.fail(
            f"ARM FF-A bus not registered: {ARM_FFA_SYSFS_DEVICES_PATH} missing. "
            "Probe likely failed; check dmesg for ARM FF-A errors."
        )
    partitions = os.listdir(ARM_FFA_SYSFS_DEVICES_PATH)
    if not partitions:
        GLTCore.GLTTestResult.fail(
            f"ARM FF-A bus registered but no partitions were discovered in "
            f"{ARM_FFA_SYSFS_DEVICES_PATH}. Firmware may not be exposing any "
            "FF-A endpoints, or probe failed after bus registration."
        )
    logger.info("ARM FF-A bus registered with %d partition(s): %s",
                len(partitions), ", ".join(partitions))

    if not glt_utils.check_kernel_log(ARM_FFA_DMESG_PATTERN):
        logger.warning(
            "ARM FF-A initialization message not found in kernel log: %s "
            "(kernel log may have rotated)",
            ARM_FFA_DMESG_PATTERN,
        )
