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
import re
import subprocess
import uuid
from pathlib import Path

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
ARM_FFA_SECURE_PARTITION_ID_FLAG = 1 << 15
ARM_FFA_SYSFS_DEVICES_PATH = "/sys/bus/arm_ffa/devices"
ARM_FFA_RXTX_MAP_FAILURE_PATTERN = "failed to register FFA RxTx buffers"
ARM_FFA_REQUIRED_PARTITION_ATTRS = ("partition_id", "uuid", "modalias")


def _collect_ffa_partitions(devices_path):
    """Collect coherent FF-A partition IDs and UUIDs from a sysfs path."""
    devices_path = Path(devices_path)
    try:
        device_paths = sorted(path for path in devices_path.iterdir() if path.is_dir())
    except OSError as error:
        return [], f"Failed to inspect {devices_path}: {error}"

    partitions = []
    for device_path in device_paths:
        partition, error = _read_ffa_partition(device_path)
        if error:
            return [], error
        partitions.append(partition)

    return partitions, ""


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


def _validate_ffa_runtime(checkpoint=None):  # pylint: disable=too-many-return-statements
    """Return discovered partitions and any FF-A runtime failure reason."""
    try:
        rxtx_map_failed = glt_utils.check_kernel_log(
            ARM_FFA_RXTX_MAP_FAILURE_PATTERN,
            checkpoint=checkpoint,
        )
    except RuntimeError as error:
        return [], f"Failed to retrieve ARM FF-A kernel log: {error}"

    if rxtx_map_failed:
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


def _read_ffa_partition(device_path):
    """Read and validate one FF-A sysfs partition record."""
    partition_id_pattern = re.compile(r"0x[0-9a-f]{4}")
    attributes = {}
    for attribute in ARM_FFA_REQUIRED_PARTITION_ATTRS:
        attribute_path = device_path / attribute
        try:
            attributes[attribute] = attribute_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as error:
            return None, f"Failed to read {attribute_path}: {error}"

    partition_id = attributes["partition_id"]
    if not partition_id_pattern.fullmatch(partition_id):
        return None, f"Invalid FF-A partition ID {partition_id!r} in {device_path.name}"

    partition_uuid = attributes["uuid"]
    try:
        parsed_uuid = uuid.UUID(partition_uuid)
    except ValueError:
        return None, f"Invalid FF-A partition UUID {partition_uuid!r} in {device_path.name}"
    canonical_uuid = str(parsed_uuid)

    expected_modalias = f"arm_ffa:{partition_id[2:]}:{canonical_uuid}"
    if attributes["modalias"] != expected_modalias:
        return None, (
            f"FF-A modalias mismatch in {device_path.name}: expected "
            f"{expected_modalias!r}, found {attributes['modalias']!r}"
        )

    return (partition_id, canonical_uuid), ""


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
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

    checkpoint = None
    if transport == "y" or is_builtin:
        logger.info("ARM FF-A transport mode: builtin")
    else:
        logger.info("ARM FF-A transport mode: module")

        try:
            checkpoint = glt_utils.capture_kernel_log_checkpoint()
        except RuntimeError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to capture ARM FF-A kernel log checkpoint: {error}"
            )
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
            result = glt_utils.run_command(
                ["lsmod"],
                "Failed to inspect loaded modules",
            )
        except RuntimeError as error:
            GLTCore.GLTTestResult.fail(str(error))

        if not any(
            line.split()[0] == module_name
            for line in result.stdout.splitlines()[1:]
        ):
            GLTCore.GLTTestResult.fail(
                f"{module_name} did not appear in lsmod after modprobe"
            )

    partitions, reason = _validate_ffa_runtime(checkpoint)
    if reason:
        GLTCore.GLTTestResult.fail(reason)

    logger.info("ARM FF-A bus registered with %d partition(s): %s",
                len(partitions), ", ".join(partitions))

    try:
        has_ffa_log = glt_utils.check_kernel_log(
            ARM_FFA_DMESG_PATTERN,
            checkpoint=checkpoint,
        )
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve ARM FF-A kernel log: {error}"
        )

    if not has_ffa_log:
        logger.warning(
            "ARM FF-A initialization message not found in kernel log: %s "
            "(kernel log may have rotated)",
            ARM_FFA_DMESG_PATTERN,
        )


@GLTCore.GLTRequires.must_pass("arm64", "test_arm_ffa_availability")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_ffa_partition_info_descriptors(_, logger):
    """
    Detect the known Vera duplicate FF-A descriptor-stride signature.

    This targeted regression check does not prove patch presence for every
    firmware descriptor layout.

    PATCH: https://git.kernel.org/torvalds/c/01b9cae70616
    """
    fix_guidance = (
        "apply or backport upstream commit "
        "01b9cae70616 (firmware: arm_ffa: Honor partition info descriptor size)"
    )
    partitions, error = _collect_ffa_partitions(ARM_FFA_SYSFS_DEVICES_PATH)
    if error:
        GLTCore.GLTTestResult.fail(
            "Unable to validate ARM FF-A partition descriptors: "
            f"{error}"
        )

    logger.info(
        "Observed ARM FF-A partitions: %s",
        ", ".join(
            f"{partition_id}={partition_uuid}"
            for partition_id, partition_uuid in partitions
        ),
    )

    secure_partitions = [
        partition
        for partition in partitions
        if int(partition[0], 16) & ARM_FFA_SECURE_PARTITION_ID_FLAG
    ]
    seen_partitions = set()
    duplicate_partitions = set()
    for partition in secure_partitions:
        if partition in seen_partitions:
            duplicate_partitions.add(partition)
        seen_partitions.add(partition)

    if duplicate_partitions:
        GLTCore.GLTTestResult.fail(
            "Duplicate secure FF-A partition records found: "
            f"{sorted(duplicate_partitions)}; {fix_guidance}"
        )
