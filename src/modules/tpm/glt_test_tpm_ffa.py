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

"""Validate the architected TPM CRB command path over Arm FF-A."""

import platform
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.tpm import runtime as tpm_runtime
from common.tpm.acpi import read_tpm2_ffa_partition

TPM2_ACPI_TABLE = Path("/sys/firmware/acpi/tables/TPM2")
FFA_DEVICES = Path("/sys/bus/arm_ffa/devices")
TPM_CLASS = Path("/sys/class/tpm")
DEVICE_DIRECTORY = Path("/dev")
TPM_COMMAND_TIMEOUT = 30
PATCH_HINT = (
    "Consider checking whether the kernel is missing 9afb9eaa9f7e "
    "(tpm_crb: Add support for the ARM FF-A start method) or eb93f0734ef1 "
    "(tpm_crb: ffa_tpm: Implement driver compliant to CRB over FF-A)."
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tpm_crb_ffa_command_path(_, logger):
    """
    Validate a read-only TPM2 capability command through the CRB FF-A start method.

    Require arm64, a valid TPM2 ACPI table advertising start method 15, its TPM
    service UUID/partition bound to ffa-crb, and exactly one eligible MSFT0101
    TPM2 device bound to tpm_crb with a resource manager character device.
    Run the bounded command
    `tpm2_getcap -T device:/dev/tpmrmN properties-fixed` and validate TPM2 family.
    Configured modular TPM drivers may be loaded; no TPM state is changed.

    SKIP on other architectures, absent TPM2 tables, valid non-FF-A start methods,
    multiple eligible resource managers, or missing tpm2_getcap after binding
    checks. Advertised but malformed/broken
    FF-A TPM interfaces and command errors/timeouts FAIL. This covers the TPM CRB
    FF-A command path; it does not validate FF-A v1.2 messages or notifications,
    or all patches in a broader FF-A backport series.

    PATCH: https://git.kernel.org/torvalds/c/9afb9eaa9f7e
    PATCH: https://git.kernel.org/torvalds/c/eb93f0734ef1
    """
    if platform.machine().lower() not in ("aarch64", "arm64"):
        GLTCore.GLTTestResult.skip("TPM CRB over Arm FF-A requires arm64")
    try:
        partition_id = read_tpm2_ffa_partition(TPM2_ACPI_TABLE)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip("Firmware does not expose a TPM2 ACPI table")
    except (OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(f"Cannot validate advertised TPM2 ACPI table: {error}")
    if partition_id is None:
        GLTCore.GLTTestResult.skip("TPM2 ACPI start method is not CRB over Arm FF-A")
    logger.info("TPM2 ACPI advertises FF-A partition 0x%04x", partition_id)
    try:
        tpm_runtime.load_modular_tpm_drivers(
            logger, f"/boot/config-{platform.release()}", TPM_COMMAND_TIMEOUT,
        )
        endpoint = tpm_runtime.find_ffa_tpm_endpoint(partition_id, FFA_DEVICES)
        resources = tpm_runtime.find_tpm_resource_managers(TPM_CLASS, DEVICE_DIRECTORY)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        GLTCore.GLTTestResult.fail(f"Advertised TPM CRB FF-A path is unavailable: {error}. {PATCH_HINT}")
    logger.info("TPM FF-A endpoint %s binds ffa-crb", endpoint.name)
    if len(resources) > 1:
        GLTCore.GLTTestResult.skip(
            "Cannot uniquely identify one ACPI TPM2 CRB resource manager for the FF-A command path"
        )
    if not glt_utils.is_cli_available("tpm2_getcap"):
        GLTCore.GLTTestResult.skip("Install tpm2-tools: tpm2_getcap is required for the TPM command probe")
    resource = resources[0]
    try:
        result = glt_utils.run_command(
            ["tpm2_getcap", "-T", f"device:{resource}", "properties-fixed"],
            f"TPM2 capability query on {resource} failed or exceeded {TPM_COMMAND_TIMEOUT} seconds",
            timeout=TPM_COMMAND_TIMEOUT,
        )
        tpm_runtime.validate_tpm_properties(result.stdout)
    except (RuntimeError, ValueError) as error:
        GLTCore.GLTTestResult.fail(f"TPM CRB FF-A command failed: {error}. {PATCH_HINT}")
    logger.info("TPM2 properties-fixed completed via %s (family 2.0)", resource)
