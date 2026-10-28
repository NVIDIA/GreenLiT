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
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.tpm import ARM_FFA_TPM_UUID
from common.tpm.acpi import read_tpm2_ffa_partition
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
ARM_FFA_DMA_MODULE = "glt_ffa_dma_runtime"
ARM_FFA_DMA_HEADERS_ROOT = Path("/lib/modules")
ARM_FFA_DMA_RESULT = Path("/proc/glt_ffa_dma_runtime")
ARM_FFA_DMA_MODULE_STATE = Path("/sys/module/glt_ffa_dma_runtime")
ARM_FFA_DMA_PAYLOAD = Path(__file__).resolve().parent / "ffa_dma_runtime"
ARM_FFA_TPM2_ACPI_TABLE = Path("/sys/firmware/acpi/tables/TPM2")
ARM_FFA_DISCOVERY_GUIDANCE = (
    "Check FF-A initialization, Linux device enumeration, and firmware discovery data."
)
ARM_FFA_BIG_ENDIAN_HINT = (
    "On this big-endian DUT, also check descriptor conversion fixes f94ebb724067 "
    "(__ffa_partition_info_get) and 7bc0f589c81d (__ffa_partition_info_regs_get)."
)
ARM_FFA_DMA_PATCH_HINT = (
    "Consider checking whether the running kernel is missing cc0aac7ca17e "
    "(firmware: arm_ffa: Set dma_mask for ffa devices)."
)


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



def _ffa_dma_candidates():
    """Prefer an unbound host device and exclude the architected TPM service."""
    devices = Path(ARM_FFA_SYSFS_DEVICES_PATH)
    if not devices.is_dir():
        return []
    candidates = []
    for device in devices.iterdir():
        if not device.is_dir() or not re.fullmatch(r"arm-ffa-[0-9]+", device.name):
            continue
        if (device / "driver").exists() or (device / "driver").is_symlink():
            continue
        partition, error = _read_ffa_partition(device)
        if error:
            raise RuntimeError(error)
        identifier, service = partition
        service_uuid = uuid.UUID(service)
        if service_uuid == ARM_FFA_TPM_UUID:
            continue
        host = service_uuid.int == 0 and not int(identifier, 16) & ARM_FFA_SECURE_PARTITION_ID_FLAG
        candidates.append((not host, device.name))
    return [name for _, name in sorted(candidates)]


def _parse_ffa_dma_result(output, run_id, release, device):  # pylint: disable=too-many-branches
    """Derive the verdict from fresh raw DMA evidence and verified restoration."""
    values = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise RuntimeError("Malformed or duplicate FF-A DMA result field")
        values[key] = value
    expected = {
        "schema_version": "2", "evidence_kind": "running_kernel_dma", "run_id": run_id,
        "kernel_release": release, "device": device,
    }
    fields = set(expected) | {
        "stage", "mask_present", "mask_set_rc", "allocation_bytes", "data_checked", "state_restored",
    }
    if set(values) != fields:
        raise RuntimeError("FF-A DMA result fields are missing or unexpected")
    for key, value in expected.items():
        if values[key] != value:
            raise RuntimeError(f"FF-A DMA result has missing or mismatched {key}")
    for key in ("mask_present", "data_checked", "state_restored"):
        if values[key] not in ("0", "1"):
            raise RuntimeError(f"FF-A DMA result has invalid {key}")
    for key, pattern in (("mask_set_rc", r"-?[0-9]+"), ("allocation_bytes", r"[0-9]+")):
        if not re.fullmatch(pattern, values[key]):
            raise RuntimeError(f"FF-A DMA result has invalid {key}")
    if values["state_restored"] != "1":
        raise RuntimeError("FF-A DMA probe did not restore the original device DMA state")
    stage = values["stage"]
    if stage in ("device_missing", "device_bound", "tpm_device"):
        if any(values[key] != "0" for key in ("mask_present", "mask_set_rc", "allocation_bytes", "data_checked")):
            raise RuntimeError("FF-A DMA SKIP contains unexpected DMA execution evidence")
        return "SKIP", f"Selected FF-A device became unavailable or ineligible: {stage}"
    if stage == "dma_mask_missing":
        if any(values[key] != "0" for key in ("mask_present", "mask_set_rc", "allocation_bytes", "data_checked")):
            raise RuntimeError("FF-A DMA missing-mask failure contains contradictory execution evidence")
        return "FAIL", (
            f"Running-kernel FF-A DMA check failed at {stage}; "
            f"mask_set_rc={values['mask_set_rc']}. {ARM_FFA_DMA_PATCH_HINT}"
        )
    if stage in ("dma_mask_set", "dma_allocation", "dma_buffer_check"):
        allocation = os.sysconf("SC_PAGE_SIZE") if stage == "dma_buffer_check" else 0
        mask_failed = int(values["mask_set_rc"]) != 0
        if (values["mask_present"] != "1" or values["data_checked"] != "0" or
                int(values["allocation_bytes"]) != allocation or mask_failed != (stage == "dma_mask_set")):
            raise RuntimeError(f"FF-A DMA {stage} contains contradictory execution evidence")
        return "FAIL", (
            f"Running-kernel FF-A DMA check failed at {stage}; mask_set_rc={values['mask_set_rc']}. "
            "Inspect device DMA configuration, allocation resources, and probe diagnostics."
        )
    if stage != "complete":
        raise RuntimeError("FF-A DMA result has invalid stage")
    if (values["mask_present"] != "1" or values["mask_set_rc"] != "0" or
            values["data_checked"] != "1" or
            int(values["allocation_bytes"]) != os.sysconf("SC_PAGE_SIZE")):
        raise RuntimeError("FF-A DMA PASS lacks successful mask setup and one-page allocation/readback")
    return "PASS", f"Coherent DMA allocation, CPU readback, free, and state restoration completed on {device}"


def _run_ffa_dma_probe(logger):  # pylint: disable=too-many-return-statements,too-many-branches
    """Build a private header-only probe and always unload it before returning."""
    release = platform.release()
    build_headers = ARM_FFA_DMA_HEADERS_ROOT / release / "build"
    if not build_headers.is_dir() or not (build_headers / "Makefile").is_file():
        return "SKIP", f"Matching installed kernel headers are unavailable: {build_headers}"
    for command in ("make", "cc", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return "SKIP", f"{command} is required to build and run the FF-A DMA probe"
    reason = glt_utils.module_unload_unavailable_reason()
    if reason:
        return "SKIP", reason

    if ARM_FFA_DMA_MODULE_STATE.exists() or ARM_FFA_DMA_RESULT.exists():
        return "SKIP", "An FF-A DMA probe module or result already exists; refusing to replace it"
    candidates = _ffa_dma_candidates()
    if not candidates:
        return "SKIP", "No existing unbound non-TPM FF-A device is available for coherent DMA validation"
    for name in ("Makefile", ARM_FFA_DMA_MODULE + ".c"):
        if not (ARM_FFA_DMA_PAYLOAD / name).is_file():
            return "FAIL", f"Packaged FF-A DMA probe payload is missing: {name}"
    status, reason = "FAIL", "FF-A DMA probe did not execute"
    # TODO: Unify temporary module build/load/unload with the Buffers probe in a shared helper.
    with tempfile.TemporaryDirectory(prefix="glt-ffa-dma-") as temporary:
        directory = Path(temporary)
        for name in ("Makefile", ARM_FFA_DMA_MODULE + ".c"):
            shutil.copy2(ARM_FFA_DMA_PAYLOAD / name, directory / name)
        loaded = False
        load_attempted = False
        run_id = None
        device = candidates[0]
        try:
            header_release = glt_utils.run_command(
                ["make", "-s", "-C", str(build_headers), f"M={directory}", "kernelrelease"],
                "Cannot determine the installed kernel-header release", timeout=30,
            ).stdout.strip()
            if header_release != release:
                return "SKIP", f"Installed kernel headers identify {header_release!r}, expected {release!r}"
            glt_utils.run_command(
                ["make", "-s", "-C", str(build_headers), f"M={directory}", "modules"],
                "Cannot build the running-kernel FF-A DMA probe using installed headers", timeout=180,
            )
            module = directory / (ARM_FFA_DMA_MODULE + ".ko")
            if not glt_utils.sign_kernel_module(str(module), logger):
                return "SKIP", "Cannot sign the FF-A DMA probe for this host's Secure Boot configuration"
            run_id = uuid.uuid4().hex
            load_attempted = True
            glt_utils.run_command(
                ["insmod", str(module), f"device_name={device}", f"run_id={run_id}"],
                "Cannot load the running-kernel FF-A DMA probe", timeout=30,
            )
            loaded = True
            output = glt_utils.read_sysfs_value(ARM_FFA_DMA_RESULT)
            status, reason = _parse_ffa_dma_result(output, run_id, release, device)
            logger.info("FF-A DMA runtime evidence: %s", output.strip().replace("\n", "; "))
        except (OSError, RuntimeError, UnicodeError) as error:
            status, reason = "FAIL", str(error)
        finally:
            # A timed-out loader can finish module initialization before it is
            # reaped. Only reclaim such a module when its full result proves
            # that this invocation owns it; never remove another invocation.
            if not loaded and load_attempted and ARM_FFA_DMA_RESULT.exists():
                try:
                    _parse_ffa_dma_result(
                        glt_utils.read_sysfs_value(ARM_FFA_DMA_RESULT), run_id, release, device,
                    )
                    loaded = True
                except (OSError, RuntimeError, UnicodeError) as error:
                    reason += f"; cannot establish ownership for cleanup: {error}"
            if loaded:
                try:
                    glt_utils.run_command(
                        ["rmmod", ARM_FFA_DMA_MODULE],
                        "Cannot unload the owned FF-A DMA probe module", timeout=30,
                    )
                    if ARM_FFA_DMA_MODULE_STATE.exists() or ARM_FFA_DMA_RESULT.exists():
                        raise RuntimeError("Owned FF-A DMA probe resources remain after unloading")
                except (OSError, RuntimeError) as error:
                    status, reason = "FAIL", f"{reason}; cleanup failed: {error}"
    return status, reason


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


def _validate_ffa_firmware_identity(partitions):
    """Compare Linux discovery against the independent ACPI TPM2 FF-A identity.

    This does not compare two Linux renderings of the same descriptor. ACPI
    method 15 independently supplies the expected partition ID; the architected
    TPM service defines its UUID. An unavailable or invalid optional TPM2 table
    leaves only generic record checks, without independently verified identity.
    """
    try:
        identifier = read_tpm2_ffa_partition(ARM_FFA_TPM2_ACPI_TABLE)
    except FileNotFoundError:
        return "No TPM2 ACPI table; checked descriptor consistency only, without an independent identity witness"
    except (OSError, ValueError) as error:
        return (
            f"Optional TPM2 ACPI identity unavailable ({error}); checked descriptor consistency only, "
            "without an independent identity witness"
        )
    if identifier is None:
        return (
            "TPM2 ACPI uses another start method; checked descriptor consistency only, "
            "without an FF-A identity witness"
        )
    expected = (f"0x{identifier:04x}", str(ARM_FFA_TPM_UUID))
    if expected not in partitions:
        raise ValueError(
            "ACPI TPM2 advertises FF-A partition "
            f"{expected[0]} with architected TPM UUID {expected[1]}, "
            "but Linux discovery exposes no matching ID/UUID pair"
        )
    return f"Validated independent ACPI TPM2 FF-A identity {expected[0]}={expected[1]}"


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
    # Transport unload/reload cleanup (46dcd68aacca) is outside this
    # functional availability check.
    #
    # TODO: Cover FFA_MSG_SEND2 for 02c19d84c7c5.
    # Runtime coverage is deferred pending a supported firmware receiver
    # and documented indirect-message protocol. Availability and direct-
    # message checks do not validate this SEND2 path.
    #
    # TODO: Cover virtual-instance notifications for 9472fe20d396.
    # Run GreenLiT inside an FF-A-enabled guest with a nonzero FF-A endpoint
    # ID and a controlled VM/hypervisor notification source; verify delivery
    # through the guest's running kernel and safe callback cleanup.
    # A generic VM boot does not exercise this path.
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
def test_arm_ffa_coherent_dma(_, logger):
    """Allocate coherent DMA through an existing unbound running-kernel FF-A device.

    Requires make/cc/insmod/rmmod, matching installed kernel headers, and readable
    /boot/config-<running-release> confirming CONFIG_MODULES=y and
    CONFIG_MODULE_UNLOAD=y. No full kernel source tree is required. Prefer the unbound host device and
    exclude the TPM service and all bound devices. A temporary GPL module uses
    the public FF-A bus and DMA APIs, preserves existing nonzero DMA limits,
    and uses a temporary 64-bit mask only when both mask values are unset.
    It checks CPU readback, frees one DMA page, and restores the original DMA
    state. It does not bind a client, call firmware, replace transport operations,
    or test a peripheral DMA transfer. Build products and the module are cleaned up.

    Missing prerequisites or eligible devices SKIP. A missing DMA mask or
    failed DMA operation on an eligible device FAIL; malformed evidence and
    cleanup failures FAIL. Module loading may taint the kernel as out-of-tree.

    PATCH: https://git.kernel.org/torvalds/c/cc0aac7ca17e
    """
    if platform.machine().lower() not in ("arm64", "aarch64"):
        GLTCore.GLTTestResult.skip("FF-A coherent DMA validation requires arm64")
    try:
        status, reason = _run_ffa_dma_probe(logger)
    except (OSError, RuntimeError) as error:
        GLTCore.GLTTestResult.fail(f"Cannot validate running-kernel FF-A DMA: {error}")
    if status == "FAIL":
        GLTCore.GLTTestResult.fail(reason)
    if status == "SKIP":
        GLTCore.GLTTestResult.skip(reason)
    logger.info("%s", reason)


@GLTCore.GLTRequires.must_pass("arm64", "test_arm_ffa_availability")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_ffa_partition_info_descriptors(_, logger):
    """
    Validate FF-A descriptor consistency and independently advertised identity.

    Use the same case on little- or big-endian DUTs. Check sysfs record format,
    aliases and duplicate secure records. When a valid TPM2 ACPI table
    advertises FF-A start method 15, require its partition ID paired with the
    architected TPM UUID in Linux discovery; a missing/mismatched pair FAILs.
    An absent, unreadable, malformed or non-FF-A TPM2 table supplies no independent
    witness: retain the consistency check and log its narrower scope. This
    fallback does not prove correct descriptor decoding or TPM functionality.

    PASS covers observed current-DUT discovery. It does not prove the complete
    firmware inventory, execution-context/property fields, both discovery
    routes, register-response pagination, or another target's endian support.
    Little-endian runs do not exercise big-endian descriptor conversion behavior.

    Upstream origin:
    PATCH: https://git.kernel.org/torvalds/c/01b9cae70616
    Equivalent stable backport (linux-nvidia-6.18-next): 0da72a88dbba
    Related big-endian descriptor conversion fixes:
    https://git.kernel.org/torvalds/c/f94ebb724067
    https://git.kernel.org/torvalds/c/7bc0f589c81d
    """
    fix_guidance = (
        "apply or backport upstream commit "
        "01b9cae70616 (firmware: arm_ffa: Honor partition info descriptor size), "
        "or its equivalent stable backport 0da72a88dbba (linux-nvidia-6.18-next)"
    )
    partitions, error = _collect_ffa_partitions(ARM_FFA_SYSFS_DEVICES_PATH)
    if error or not partitions:
        GLTCore.GLTTestResult.fail(
            "Unable to validate ARM FF-A partition descriptors: "
            f"{error or 'No FF-A partition records were discovered'}. "
            f"{ARM_FFA_DISCOVERY_GUIDANCE}"
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

    try:
        identity_evidence = _validate_ffa_firmware_identity(partitions)
    except (OSError, ValueError) as identity_error:
        guidance = (
            f"{ARM_FFA_DISCOVERY_GUIDANCE} If this partition exposes multiple service UUIDs, "
            "also check the multiple-UUID admission fix be61da938576."
        )
        if sys.byteorder == "big":
            guidance += f" {ARM_FFA_BIG_ENDIAN_HINT}"
        GLTCore.GLTTestResult.fail(
            f"Unable to validate firmware-advertised FF-A identity: {identity_error}. {guidance}"
        )
    if sys.byteorder == "little":
        identity_evidence += "; big-endian descriptor conversion behavior was not exercised"
    logger.info(
        "Observed ARM FF-A partitions: %s; %s",
        ", ".join(
            f"{partition_id}={partition_uuid}"
            for partition_id, partition_uuid in partitions
        ),
        identity_evidence,
    )
