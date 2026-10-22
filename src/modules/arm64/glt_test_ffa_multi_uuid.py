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

"""Exercise two public UUID services on the same real FF-A partition."""

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import tempfile
import uuid

from common import glt_core as GLTCore
from common import glt_utils
from common.tpm import runtime as tpm_runtime
from common.tpm.acpi import read_tpm2_ffa_partition

TPM2_ACPI_TABLE = Path("/sys/firmware/acpi/tables/TPM2")
FFA_DEVICES = Path("/sys/bus/arm_ffa/devices")
TPM_CLASS = Path("/sys/class/tpm")
DEVICE_DIRECTORY = Path("/dev")
HEADERS_ROOT = Path("/lib/modules")
BUFFERS_UUID = uuid.UUID("e257054d-2092-4e1e-b40c-2ef1f68e4b0a")
PROBE_MODULE = "glt_ffa_stmm_runtime"
PROBE_PAYLOAD = Path(__file__).resolve().parent / "ffa_stmm_runtime"
PROBE_RESULT = Path("/proc/glt_ffa_stmm_runtime")
PROBE_MODULE_STATE = Path("/sys/module/glt_ffa_stmm_runtime")
COMMAND_TIMEOUT = 30
RUNTIME_GUIDANCE = "Check Linux FF-A driver/API support, the firmware service protocol, and kernel logs."
RESULT_FIELDS = frozenset((
    "schema_version", "evidence_kind", "run_id", "kernel_release", "device", "uuid", "partition_id",
    "properties", "mode_32bit", "driver_bound", "req2_supported", "api_available", "query_attempted",
    "api_rc", "response_received", "service_status", "base_nonzero", "buffer_size", "minimum_size", "stage",
))


@dataclass(frozen=True)
class _Profile:
    """The actual exposed public services selected before any command."""

    partition_id: int
    tpm_endpoint: Path
    resource: Path
    buffers_endpoint: Path


def _discover_profile(partition_id):
    """Validate advertised TPM first; only an absent compatible Buffers profile skips."""
    tpm_endpoint = tpm_runtime.find_ffa_tpm_endpoint(partition_id, FFA_DEVICES)
    resources = tpm_runtime.find_tpm_resource_managers(TPM_CLASS, DEVICE_DIRECTORY)
    if len(resources) > 1:
        return None, "Cannot uniquely identify one ACPI TPM2 CRB resource manager for this profile"
    buffers = []
    for endpoint in sorted(FFA_DEVICES.iterdir()):
        service = uuid.UUID(glt_utils.read_sysfs_value(endpoint / "uuid"))
        if service != BUFFERS_UUID:
            continue
        identifier = glt_utils.read_sysfs_value(endpoint / "partition_id")
        if not re.fullmatch(r"0x[0-9a-fA-F]{4}", identifier):
            raise ValueError(f"Invalid public Buffers partition ID {identifier!r}")
        if int(identifier, 16) != partition_id:
            continue
        if not re.fullmatch(r"arm-ffa-[0-9]+", endpoint.name):
            raise ValueError("Public Buffers service has an invalid bus device name")
        if (endpoint / "driver").exists() or (endpoint / "driver").is_symlink():
            return None, "The public Buffers service is bound to another client"
        buffers.append(endpoint)
    if len(buffers) != 1:
        return None, (
            "No unique unbound public Buffers service shares the advertised TPM partition; "
            "multi-UUID admission is inconclusive. Linux enumeration cannot distinguish firmware absence "
            "from Linux admission loss without a bounded independent firmware inventory"
        )
    return _Profile(partition_id, tpm_endpoint, resources[0], buffers[0]), ""


def _parse_probe_result(output, token, release, profile):  # pylint: disable=too-many-branches
    """Require fresh, typed, address-free evidence from the actual kernel query."""
    values = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise RuntimeError("Malformed or duplicate FF-A Buffers result field")
        values[key] = value
    if set(values) != RESULT_FIELDS:
        raise RuntimeError("FF-A Buffers result fields are missing or unexpected")
    for key, expected in {
            "schema_version": "2", "evidence_kind": "running_kernel_ffa_multi_uuid", "run_id": token,
            "kernel_release": release, "device": profile.buffers_endpoint.name, "minimum_size": "25"}.items():
        if values[key] != expected:
            raise RuntimeError(f"FF-A Buffers result has mismatched {key}")
    for key in ("mode_32bit", "driver_bound", "req2_supported", "api_available", "query_attempted",
                "response_received", "base_nonzero"):
        if values[key] not in ("0", "1"):
            raise RuntimeError(f"FF-A Buffers result has invalid {key}")
    for key, pattern in (("properties", r"0x[0-9a-fA-F]{8}"), ("partition_id", r"0x[0-9a-fA-F]{4}"),
                         ("api_rc", r"-?[0-9]+"), ("service_status", r"0x[0-9a-fA-F]{1,16}"),
                         ("buffer_size", r"[0-9]+")):
        if not re.fullmatch(pattern, values[key]):
            raise RuntimeError(f"FF-A Buffers result has invalid {key}")
    try:
        service = uuid.UUID(values["uuid"])
    except ValueError as error:
        raise RuntimeError("FF-A Buffers result has invalid UUID") from error
    identity_matches = service == BUFFERS_UUID and int(values["partition_id"], 16) == profile.partition_id
    stage = values["stage"]
    if stage in (
            "device_missing", "device_identity_changed", "device_bound", "req2_unsupported"):
        if any(values[key] != "0" for key in ("query_attempted", "api_rc", "response_received",
                                               "base_nonzero", "buffer_size")) or int(values["service_status"], 16):
            raise RuntimeError("FF-A Buffers SKIP contains unexpected query execution")
        if stage not in ("device_missing", "device_identity_changed") and not identity_matches:
            raise RuntimeError("FF-A Buffers SKIP has an inconsistent service identity")
        return "SKIP", (
            f"The compatible public Buffers profile is unavailable: {stage}; messaging coverage is inconclusive"
        )
    if not identity_matches:
        raise RuntimeError("FF-A Buffers response does not match the selected UUID/partition")
    eligible = {"mode_32bit": "0", "driver_bound": "0", "req2_supported": "1"}
    if (any(values[key] != expected for key, expected in eligible.items())
            or not int(values["properties"], 16) & (1 << 9)):
        raise RuntimeError("FF-A Buffers query stage lacks advertised REQ2 eligibility")
    if stage in ("api_missing", "api_error", "service_error"):
        expected = {
            "api_available": "0" if stage == "api_missing" else "1",
            "query_attempted": "0" if stage == "api_missing" else "1",
            "response_received": "1" if stage == "service_error" else "0",
            "base_nonzero": "0", "buffer_size": "0",
        }
        if (any(values[key] != value for key, value in expected.items())
                or (int(values["api_rc"]) != 0) != (stage == "api_error")
                or (int(values["service_status"], 16) != 0) != (stage == "service_error")):
            raise RuntimeError("FF-A Buffers failure stage has inconsistent query evidence")
        return "FAIL", (
            f"Public Buffers request failed at {stage}: API {values['api_rc']}, "
            f"service status {values['service_status']}. {RUNTIME_GUIDANCE}"
        )
    if stage != "complete":
        raise RuntimeError("FF-A Buffers result has an invalid stage")
    required = {"mode_32bit": "0", "driver_bound": "0", "req2_supported": "1", "api_available": "1",
                "query_attempted": "1", "api_rc": "0", "response_received": "1"}
    if any(values[key] != expected for key, expected in required.items()) or int(values["service_status"], 16):
        raise RuntimeError("FF-A Buffers complete stage lacks a successful eligible request and response")
    if values["base_nonzero"] != "1" or int(values["buffer_size"]) < 25:
        return "FAIL", f"Public Buffers request returned invalid_metadata. {RUNTIME_GUIDANCE}"
    return "PASS", "The real public Buffers UUID returned valid metadata through DIRECT_REQ2"


def _probe_prerequisite_reason():
    """Require installed headers, tools, and confirmed kernel module unloading."""
    headers = HEADERS_ROOT / platform.release() / "build"
    if not headers.is_dir() or not (headers / "Makefile").is_file():
        return f"Matching installed kernel headers are unavailable: {headers}"
    for command in ("tpm2_getcap", "make", "cc", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return f"{command} is required for the TPM/Buffers messaging sequence"
    reason = glt_utils.module_unload_unavailable_reason()
    if reason:
        return reason

    if PROBE_MODULE_STATE.exists() or PROBE_RESULT.exists():
        return "A Buffers probe module or result already exists; refusing to replace it"
    return ""


def _query_tpm(resource, phase, logger):
    """Use the legitimate resource-manager client and validate its actual response."""
    result = glt_utils.run_command(
        ["tpm2_getcap", "-T", f"device:{resource}", "properties-fixed"],
        f"TPM capability query {phase} failed or exceeded {COMMAND_TIMEOUT} seconds", timeout=COMMAND_TIMEOUT,
    )
    tpm_runtime.validate_tpm_properties(result.stdout)
    logger.info("TPM2 properties-fixed %s completed via %s (family 2.0)", phase, resource)


def _run_buffers_probe(profile, logger):  # pylint: disable=too-many-branches
    """Build/load one documented request and reclaim only this invocation's module."""
    # TODO: Share the DMA/StMM build, load and unload lifecycle in a common helper.
    release = platform.release()
    headers = HEADERS_ROOT / release / "build"
    status, reason = "FAIL", "Public Buffers query did not execute"
    token = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="glt-ffa-stmm-") as temporary:
        directory = Path(temporary)
        for name in ("Makefile", PROBE_MODULE + ".c"):
            shutil.copy2(PROBE_PAYLOAD / name, directory / name)
        actual_release = glt_utils.run_command(
            ["make", "-s", "-C", str(headers), f"M={directory}", "kernelrelease"],
            "Cannot determine the installed kernel-header release", timeout=COMMAND_TIMEOUT,
        ).stdout.strip()
        if actual_release != release:
            return "SKIP", f"Installed headers identify {actual_release!r}, expected {release!r}"
        loaded = False
        attempted = False
        try:
            glt_utils.run_command(
                ["make", "-s", "-C", str(headers), f"M={directory}", "modules"],
                "Cannot build the public Buffers probe using installed kernel headers", timeout=180,
            )
            module = directory / (PROBE_MODULE + ".ko")
            if not glt_utils.sign_kernel_module(str(module), logger):
                return "SKIP", "Cannot sign the public Buffers probe for this host's Secure Boot configuration"
            attempted = True
            glt_utils.run_command(
                ["insmod", str(module), f"device_name={profile.buffers_endpoint.name}",
                 f"partition_id={profile.partition_id}", f"run_id={token}"],
                "Cannot load the public Buffers runtime probe", timeout=COMMAND_TIMEOUT,
            )
            loaded = True
            output = glt_utils.read_sysfs_value(PROBE_RESULT)
            status, reason = _parse_probe_result(output, token, release, profile)
            logger.info("FF-A multi-UUID Buffers evidence: %s", output.strip().replace("\n", "; "))
        except (OSError, RuntimeError, UnicodeError) as error:
            status, reason = "FAIL", str(error)
        finally:
            if attempted and not loaded and PROBE_RESULT.exists():
                try:
                    _parse_probe_result(glt_utils.read_sysfs_value(PROBE_RESULT), token, release, profile)
                    loaded = True
                except (OSError, RuntimeError, UnicodeError) as error:
                    reason += f"; cannot establish module ownership for cleanup: {error}"
            if loaded:
                try:
                    glt_utils.run_command(
                        ["rmmod", PROBE_MODULE], "Cannot unload the owned Buffers probe", timeout=COMMAND_TIMEOUT,
                    )
                    if PROBE_MODULE_STATE.exists() or PROBE_RESULT.exists():
                        raise RuntimeError("Owned Buffers probe resources remain after unloading")
                except (OSError, RuntimeError) as error:
                    status, reason = "FAIL", f"{reason}; cleanup failed: {error}"
    return status, reason


def _tpm_identity(profile, descriptor):
    """Check retained real-character-device identity and the existing kernel bindings."""
    endpoint = tpm_runtime.find_ffa_tpm_endpoint(profile.partition_id, FFA_DEVICES)
    resources = tpm_runtime.find_tpm_resource_managers(TPM_CLASS, DEVICE_DIRECTORY)
    if endpoint.resolve(strict=True) != profile.tpm_endpoint.resolve(strict=True) or resources != [profile.resource]:
        raise RuntimeError("Selected TPM endpoint or resource-manager identity changed")
    retained = os.fstat(descriptor)
    current = profile.resource.stat()
    retained_identity = (retained.st_dev, retained.st_ino, retained.st_rdev)
    current_identity = (current.st_dev, current.st_ino, current.st_rdev)
    if not stat.S_ISCHR(retained.st_mode) or not stat.S_ISCHR(current.st_mode) or current_identity != retained_identity:
        raise RuntimeError("Selected TPM resource manager is no longer the retained real character device")
    return (endpoint.resolve(strict=True), (endpoint / "driver").resolve(strict=True), retained_identity)


def _exercise_profile(profile, logger):
    """Retain a legitimate TPM client while exercising TPM A1 / Buffers B / TPM A2."""
    descriptor = os.open(profile.resource, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        original = _tpm_identity(profile, descriptor)
        _query_tpm(profile.resource, "A1", logger)
        try:
            status, reason = _run_buffers_probe(profile, logger)
        except (OSError, RuntimeError) as error:
            status, reason = "FAIL", str(error)
        try:
            if _tpm_identity(profile, descriptor) != original:
                raise RuntimeError("Selected TPM binding/identity changed during the Buffers query")
            _query_tpm(profile.resource, "A2", logger)
            if _tpm_identity(profile, descriptor) != original:
                raise RuntimeError("Selected TPM binding/identity changed during the second TPM query")
        except (OSError, RuntimeError, ValueError) as error:
            return "FAIL", f"{reason}; TPM recovery query failed: {error}"
        return status, reason
    finally:
        os.close(descriptor)


@GLTCore.GLTRequires.must_pass("arm64", "test_arm_ffa_availability")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_ffa_multi_uuid_messaging(_, logger):
    """Check TPM / public Buffers / TPM messaging on one real FF-A partition.

    Require ACPI method15, its genuine bound TPM2 CRB resource manager, and the
    public unbound StMM Buffers UUID on the same partition. Run two read-only
    TPM2 capability commands around one documented GET_MM_NS_BUFFER_FID request
    through the actual existing kernel REQ2 API. Retain the real TPM client,
    check identities, never access the returned buffer or publish its address,
    and always unload the temporary Buffers module. Require matching installed
    headers and readable /boot/config-<running-release> confirming CONFIG_MODULES=y
    and CONFIG_MODULE_UNLOAD=y; no full kernel source is needed. Prerequisites/profile absence
    SKIP, while broken advertised TPM integration, an eligible request failure,
    malformed evidence or cleanup failure FAIL. This is independent messaging
    coverage, not SRI callback/notification or generic SEND2 coverage.

    Missing or ambiguous Linux Buffers devices are inconclusive for admission:
    sysfs cannot distinguish genuine firmware absence from Linux admission loss.
    TODO: Qualify be61da938576 admission coverage with a bounded independent
    firmware inventory, including TPM-first enumeration. The recorded negative
    control does not qualify all UUID orderings or SRI callback behavior.

    Protocol: https://github.com/NVIDIA/edk2-nvidia/tree/e1d1a17337c6331c71ba12aeb4928dbddaf0add6
    Silicon/NVIDIA/Library/StMmDxeGetBuffersLib/StMmDxeGetBuffersLib.c
    """
    if platform.machine().lower() not in ("aarch64", "arm64"):
        GLTCore.GLTTestResult.skip("The multi-UUID FF-A messaging profile requires arm64")
    try:
        partition_id = read_tpm2_ffa_partition(TPM2_ACPI_TABLE)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip("The multi-UUID profile has no firmware TPM2 ACPI table")
    except (OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(f"Cannot validate the multi-UUID profile ACPI advertisement: {error}")
    if partition_id is None:
        GLTCore.GLTTestResult.skip("ACPI uses another TPM start method; no CRB-over-FF-A messaging profile")
    try:
        tpm_runtime.load_modular_tpm_drivers(
            logger, f"/boot/config-{platform.release()}", COMMAND_TIMEOUT,
        )
        profile, reason = _discover_profile(partition_id)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        GLTCore.GLTTestResult.fail(f"Advertised TPM/public FF-A identity is broken: {error}. {RUNTIME_GUIDANCE}")
    if profile is None:
        GLTCore.GLTTestResult.skip(reason)
    reason = _probe_prerequisite_reason()
    if reason:
        GLTCore.GLTTestResult.skip(reason)
    try:
        status, reason = _exercise_profile(profile, logger)
    except (OSError, RuntimeError, ValueError) as error:
        GLTCore.GLTTestResult.fail(f"Cannot validate the TPM/Buffers messaging sequence: {error}. {RUNTIME_GUIDANCE}")
    if status == "FAIL":
        GLTCore.GLTTestResult.fail(reason)
    if status == "SKIP":
        GLTCore.GLTTestResult.skip(reason)
    logger.info("TPM A1 / Buffers B / TPM A2 succeeded on partition 0x%04x", profile.partition_id)
