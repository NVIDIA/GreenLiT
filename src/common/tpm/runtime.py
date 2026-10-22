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

"""Ordinary TPM runtime identity and response helpers, without test registration."""

import re
import uuid
from pathlib import Path

import yaml

from common import glt_utils
from common.tpm import ARM_FFA_TPM_UUID
from scripts import check_kernel_config


def find_ffa_tpm_endpoint(partition_id, devices_path):
    """Require the ACPI-advertised endpoint to bind the architected TPM driver."""
    for endpoint in sorted(Path(devices_path).iterdir()):
        endpoint_uuid = uuid.UUID(glt_utils.read_sysfs_value(endpoint / "uuid"))
        if endpoint_uuid != ARM_FFA_TPM_UUID:
            continue
        identifier = glt_utils.read_sysfs_value(endpoint / "partition_id")
        if not re.fullmatch(r"0x[0-9a-fA-F]{4}", identifier):
            raise ValueError(f"Invalid FF-A partition ID {identifier!r} in {endpoint.name}")
        if int(identifier, 16) != partition_id:
            continue
        driver = (endpoint / "driver").resolve(strict=True).name
        if driver != "ffa-crb":
            raise ValueError(f"TPM FF-A endpoint {endpoint.name} binds {driver!r}, expected 'ffa-crb'")
        return endpoint
    raise ValueError(
        f"No architected TPM FF-A endpoint matches ACPI partition ID 0x{partition_id:04x}"
    )


def find_tpm_resource_managers(tpm_class, device_directory):
    """Enumerate resource managers for ACPI TPM2 CRB devices.

    Callers first validate TPM2 method 15 and the expected bound FF-A endpoint.
    The supported driver model uses the first global TPM2 table and a singleton
    ffa-crb transport. Its ACPI TPM and FF-A bus devices have separate parents;
    no direct sysfs ancestry link joins them. Callers must reject an ambiguous
    resource-manager set rather than infer a per-device association.
    """
    resources = []
    for tpm in sorted(Path(tpm_class).iterdir()):
        if not re.fullmatch(r"tpm[0-9]+", tpm.name):
            continue
        device = tpm / "device"
        hid_path = device / "hid"
        if not hid_path.exists():
            hid_path = device / "firmware_node" / "hid"
        try:
            hardware_id = glt_utils.read_sysfs_value(hid_path)
        except FileNotFoundError:
            continue
        if hardware_id != "MSFT0101":
            continue
        major = glt_utils.read_sysfs_value(tpm / "tpm_version_major")
        if major != "2":
            raise ValueError(f"ACPI TPM {tpm.name} reports version {major!r}, expected TPM2")
        driver = (device / "driver").resolve(strict=True).name
        if driver != "tpm_crb":
            raise ValueError(f"ACPI TPM {tpm.name} binds {driver!r}, expected 'tpm_crb'")
        resource = Path(device_directory) / f"tpmrm{tpm.name[3:]}"
        if not resource.is_char_device():
            raise ValueError(f"TPM2 resource manager character device {resource} is missing")
        resources.append(resource)
    if not resources:
        raise ValueError("ACPI advertises TPM CRB over FF-A but no MSFT0101 TPM2 CRB device exists")
    return resources


def load_modular_tpm_drivers(logger, config_path, timeout):
    """Load configured modules before checking binding; retain runtime-only fallback."""
    try:
        config = check_kernel_config.parse_kconfig(config_path)
    except OSError as error:
        logger.info("Cannot read %s (%s); validating already-bound TPM drivers", config_path, error)
        return
    for option, module in (
        ("CONFIG_TCG_ARM_CRB_FFA", "tpm_crb_ffa"),
        ("CONFIG_TCG_CRB", "tpm_crb"),
    ):
        if config.get(option) != "m":
            continue
        if not glt_utils.is_cli_available("modprobe"):
            raise RuntimeError(f"modprobe is required to load configured TPM module {module}")
        glt_utils.run_command(
            ["modprobe", module],
            f"Failed to load TPM module {module} within {timeout} seconds",
            timeout=timeout,
        )


def validate_tpm_properties(output):
    """Require an actual TPM2 family response from the explicit device TCTI."""
    try:
        properties = yaml.safe_load(output)
    except yaml.YAMLError as error:
        raise ValueError(f"tpm2_getcap returned malformed YAML: {error}") from error
    if not isinstance(properties, dict):
        raise ValueError("tpm2_getcap did not return a properties mapping")
    family = properties.get("TPM2_PT_FAMILY_INDICATOR")
    if not isinstance(family, dict) or family.get("value") != "2.0":
        raise ValueError("tpm2_getcap did not report TPM2_PT_FAMILY_INDICATOR value '2.0'")
    if family.get("raw") != 0x322E3000:
        raise ValueError("tpm2_getcap TPM2 family raw value is inconsistent with '2.0'")
