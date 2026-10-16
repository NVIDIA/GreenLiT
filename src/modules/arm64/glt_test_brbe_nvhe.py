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

"""Validate BRBE host isolation while an arm64 nVHE guest is running."""

# Keep the nVHE lifecycle and result schema together with the temporary guard.
# pylint: disable=too-many-lines

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path

from common import virt as glt_virt
from common import glt_core as GLTCore
from common import glt_utils
from common import perf_utils
from common.cpu_topology import parse_cpu_mask
from common.nv_device import NvDev
from scripts import check_kernel_config


BRBE_NVHE_BUILD_TIMEOUT_SECONDS = 300
BRBE_NVHE_HARNESS_TIMEOUT_SECONDS = 300
BRBE_NVHE_ITERATIONS = 128
BRBE_NVHE_GUEST_BRANCHES = 20_000_000
BRBE_NVHE_SAMPLE_PERIOD = 10_000_000
BRBE_NVHE_KERNEL_PARAMETERS = (
    "kvm-arm.mode=nvhe",
    "arm64_sw.hvhe=0",
    "id_aa64mmfr1.vh=0",
)
BRBE_NVHE_LOG_MESSAGE = "Hyp nVHE mode initialized successfully"

BRBE_NVHE_OWNERSHIP_TOKEN = "BRBENvheParamOwned"
BRBE_NVHE_RESTORE_REBOOT_TOKEN = "BRBENvheRestoreReboot"
GLT_REBOOT_TOKEN = "TriggerReboot"

CPU_ONLINE_PATH = Path("/sys/devices/system/cpu/online")
RUNNING_KERNEL_NOTES_PATH = Path("/sys/kernel/notes")
HARNESS_SOURCE_DIR = Path(__file__).resolve().parent / "brbe_nvhe"
HARNESS_MAKEFILE_PATH = HARNESS_SOURCE_DIR / "Makefile"
KVM_MICROVM_SOURCE_DIR = (
    Path(glt_virt.__file__).resolve().parent / "kvm_microvm"
)
KVM_MICROVM_REQUIRED_FILES = (
    KVM_MICROVM_SOURCE_DIR / "glt_arm64_kvm.c",
    KVM_MICROVM_SOURCE_DIR / "glt_arm64_kvm.h",
)
RESULT_REQUIRED_TYPES = {
    "schema_version": int,
    "verdict": str,
    "exit_code": int,
    "cpu": int,
    "iterations": int,
    "iterations_completed": int,
    "guest_branches": int,
    "sample_period": int,
    "max_pre_guest_count": int,
    "guest_samples": int,
    "guest_windows_verified": int,
    "min_guest_instruction_delta": int,
    "nvhe_cmdline": bool,
    "samples": int,
    "entries": int,
    "guest_range_start": str,
    "guest_range_end": str,
    "guest_hits": int,
    "host_pre_hits": int,
    "host_post_hits": int,
    "guest_brbe_id": int,
    "guest_brbcr_undef": bool,
    "guest_unexpected_exception": bool,
    "lost_records": int,
    "throttle_records": int,
    "context_switches": int,
    "host_pre_range": str,
    "host_post_range": str,
    "reason": str,
}
HARNESS_EXIT_VERDICTS = {
    0: "fixed",
    1: "contaminated",
    2: "inconclusive",
}
HEX_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]+")
ADDRESS_RANGE_PATTERN = re.compile(
    r"(0x[0-9a-fA-F]+)-(0x[0-9a-fA-F]+)"
)


def _clear_nvhe_progress(sd, test_name: str) -> None:
    """Clear the three progress tokens owned by this test."""
    for token in (
        BRBE_NVHE_OWNERSHIP_TOKEN,
        BRBE_NVHE_RESTORE_REBOOT_TOKEN,
    ):
        sd.progress_delete_time(test_name, token)


def _combine_reasons(primary: str, secondary: str) -> str:
    if not primary:
        return secondary
    if not secondary:
        return primary
    return f"{primary}; additionally, {secondary}"


def _check_nvhe_parameters(test_name: str, sd) -> tuple[dict, str]:
    """Read the active, configured, and owned state of every nVHE parameter."""
    statuses = {}
    failures = []
    for parameter in BRBE_NVHE_KERNEL_PARAMETERS:
        status = glt_utils.check_kernel_parameter(
            parameter,
            test_case_name=test_name,
            session_dir=sd.log_dir,
        )
        statuses[parameter] = status
        if not status.ok:
            failures.append(f"{parameter}: {status.reason}")

    # Treat a user-owned underscore spelling as the canonical mode.
    # If GLT owns the canonical token, check it exactly: a user alias must not
    # hide failed setup or leftover GLT-owned configuration during cleanup.
    ownership = _get_progress_value(sd, test_name, BRBE_NVHE_OWNERSHIP_TOKEN)
    owned_parameters = (
        ownership.get("added_parameters", BRBE_NVHE_KERNEL_PARAMETERS)
        if sd.progress_check(test_name, BRBE_NVHE_OWNERSHIP_TOKEN) else ()
    )
    mode_parameter = "kvm-arm.mode=nvhe"
    mode_status = statuses[mode_parameter]
    if mode_status.ok and mode_parameter not in owned_parameters:
        alias_status = glt_utils.check_kernel_parameter(
            "kvm_arm.mode=nvhe", test_case_name=test_name, session_dir=sd.log_dir
        )
        if not alias_status.ok:
            failures.append(f"kvm_arm.mode=nvhe: {alias_status.reason}")
        else:
            mode_status.active = mode_status.active or alias_status.active
            mode_status.configured = mode_status.configured or alias_status.configured
            mode_status.reboot_required = mode_status.active != mode_status.configured

    return statuses, "; ".join(failures)


def _format_parameter_states(statuses: dict) -> str:
    return ", ".join(
        f"{parameter}(active={status.active}, configured={status.configured}, "
        f"owned={status.added_by_glt})"
        for parameter, status in statuses.items()
    )


# pylint: disable=too-many-return-statements
def _get_active_parameter_conflict() -> tuple[str, bool]:
    """Return a reason and whether conflicting user values should be skipped."""
    try:
        cmdline = Path("/proc/cmdline").read_text(encoding="utf-8")
    except OSError as error:
        return (
            f"Unable to inspect /proc/cmdline for nVHE conflicts: {error}",
            False,
        )

    desired_values = {
        "kvm-arm.mode": "nvhe",
        "kvm_arm.mode": "nvhe",
        "arm64_sw.hvhe": "0",
        "id_aa64mmfr1.vh": "0",
    }
    conflicting = []
    for token in cmdline.split():
        key, separator, value = token.partition("=")
        if separator and key in desired_values and value != desired_values[key]:
            conflicting.append(token)
    if conflicting:
        return (
            "Current boot has conflicting user-selected nVHE parameters: "
            + ", ".join(sorted(conflicting)),
            True,
        )
    return "", False


def _get_nvhe_log_failure() -> str:
    """Require the exact nVHE initialization evidence from this boot."""
    try:
        kernel_log = glt_utils.get_kernel_log()
    except RuntimeError as error:
        return f"Unable to inspect the current-boot kernel log: {error}"
    if BRBE_NVHE_LOG_MESSAGE not in kernel_log:
        return (
            "Current-boot kernel log does not contain the required nVHE "
            f"message: {BRBE_NVHE_LOG_MESSAGE!r}"
        )
    return ""


def _get_progress_value(sd, test_name: str, key: str):
    for entry in getattr(sd, "progress", []):
        if test_name in entry:
            return entry[test_name].get(key)
    return None


def _get_running_kernel_identity(
    notes_path=RUNNING_KERNEL_NOTES_PATH,
) -> tuple[dict, str]:
    """Return an identity for the running image, including its ELF notes."""
    digest = hashlib.sha256()
    try:
        with Path(notes_path).open("rb") as notes_file:
            for chunk in iter(lambda: notes_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        return {}, (
            f"Unable to fingerprint the running kernel through {notes_path}: "
            f"{error}"
        )

    return {
        "release": platform.release(),
        "notes_sha256": digest.hexdigest(),
    }, ""


def _verify_owned_kernel(test_name: str, sd, phase: str) -> str:
    """Require the exact running image recorded before the setup reboot."""
    expected = _get_progress_value(
        sd, test_name, BRBE_NVHE_OWNERSHIP_TOKEN)
    if isinstance(expected, dict):
        expected = expected.copy()
        added_parameters = expected.pop(
            "added_parameters", list(BRBE_NVHE_KERNEL_PARAMETERS))
        if (
            not isinstance(added_parameters, list) or not added_parameters
            or any(
                not isinstance(parameter, str)
                or parameter not in BRBE_NVHE_KERNEL_PARAMETERS
                for parameter in added_parameters
            )
            or len(set(added_parameters)) != len(added_parameters)
        ):
            return "Owned nVHE parameters have an invalid saved parameter list"

    if not isinstance(expected, dict) or set(expected) != {
        "release", "notes_sha256",
    } or not all(isinstance(value, str) and value for value in expected.values()):
        return "Owned nVHE parameters have no valid saved kernel identity"

    current, reason = _get_running_kernel_identity()
    if reason:
        return reason
    if current != expected:
        return (
            f"nVHE {phase} changed kernels: expected release "
            f"{expected['release']} with notes SHA-256 "
            f"{expected['notes_sha256']}, running {current['release']} with "
            f"notes SHA-256 {current['notes_sha256']}"
        )
    return ""


def _remove_files(paths: tuple[Path, ...]) -> str:
    errors = []
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            errors.append(f"unable to remove {path}: {error}")
    return "; ".join(errors)


def _remove_owned_nvhe_parameters(
    test_name: str, sd, *, parameters=BRBE_NVHE_KERNEL_PARAMETERS
) -> tuple[bool, str]:
    """Remove every parameter owned by this test's shared transaction."""
    failures = []
    for parameter in parameters:
        ok, reason = glt_utils.remove_kernel_parameter(
            parameter,
            test_case_name=test_name,
            session_dir=sd.log_dir,
        )
        if not ok:
            failures.append(f"{parameter}: {reason}")
    if failures:
        return False, "Failed to remove GLT-owned nVHE parameters: " + "; ".join(failures)
    return True, ""


def _setup_failure_with_cleanup(reason: str, test_name: str, sd) -> str:
    cleanup_ok, cleanup_reason = _remove_owned_nvhe_parameters(test_name, sd)
    if cleanup_ok:
        return reason
    return _combine_reasons(reason, cleanup_reason)


# pylint: disable=too-many-branches,too-many-return-statements,too-many-statements
def _ensure_nvhe_environment(
    test_name: str, sd, logger
) -> tuple[bool, str, bool]:
    """Ensure an active nVHE environment while preserving user parameters."""
    # Return (ready, reason, skip); the decorated caller sets PASS/FAIL/SKIP.
    # active describes this boot; configured describes future boot settings.
    statuses, check_reason = _check_nvhe_parameters(test_name, sd)
    if check_reason:
        return False, _setup_failure_with_cleanup(
            f"Unable to inspect nVHE parameters: {check_reason}", test_name, sd
        ), False

    owns_parameters = sd.progress_check(test_name, BRBE_NVHE_OWNERSHIP_TOKEN)
    has_owned_metadata = any(
        status.added_by_glt for status in statuses.values())
    # Do not adopt GLT changes whose saved ownership checkpoint is missing.
    if has_owned_metadata and not owns_parameters:
        return False, (
            "Found orphan GreenLiT nVHE transaction metadata without the "
            "ownership progress token; leaving it unchanged: "
            f"{_format_parameter_states(statuses)}"
        ), False

    conflict_reason, conflict_is_user_state = _get_active_parameter_conflict()
    # User conflicts are SKIP; conflicts in our owned setup are FAIL.
    if conflict_reason:
        if owns_parameters or not conflict_is_user_state:
            conflict_reason = _setup_failure_with_cleanup(
                conflict_reason, test_name, sd)
            return False, conflict_reason, False
        return False, conflict_reason, True

    all_active = all(status.active for status in statuses.values())

    # Resume GLT's setup boot and remove only our persistent settings.
    # The live parameters remain active until the post-hook reboot.
    if owns_parameters:
        identity_reason = _verify_owned_kernel(
            test_name, sd, "setup reboot")
        if identity_reason:
            return False, _setup_failure_with_cleanup(
                identity_reason, test_name, sd), False
        added_parameters = _get_progress_value(
            sd, test_name, BRBE_NVHE_OWNERSHIP_TOKEN
        ).get("added_parameters", BRBE_NVHE_KERNEL_PARAMETERS)
        if not all_active:
            reason = (
                "nVHE setup reboot did not activate the complete parameter set: "
                f"{_format_parameter_states(statuses)}"
            )
            return False, _setup_failure_with_cleanup(reason, test_name, sd), False

        cleanup_ok, cleanup_reason = _remove_owned_nvhe_parameters(test_name, sd)
        if not cleanup_ok:
            return False, cleanup_reason, False
        statuses, check_reason = _check_nvhe_parameters(test_name, sd)
        if check_reason:
            return False, _setup_failure_with_cleanup(
                f"Unable to verify nVHE setup cleanup: {check_reason}",
                test_name,
                sd,
            ), False
        # All parameters must stay active; only our additions must lose saved state.
        if not all(status.active for status in statuses.values()) or any(
            statuses[parameter].configured or statuses[parameter].added_by_glt
            for parameter in added_parameters
        ):
            reason = (
                "nVHE setup cleanup did not preserve only active state: "
                f"{_format_parameter_states(statuses)}"
            )
            return False, _setup_failure_with_cleanup(reason, test_name, sd), False
        return True, "", False

    # Already active without GLT ownership: use the environment as-is.
    if all_active:
        return True, "", False
    # A setup reboot would lose active-only user parameters; do not make
    # them persistent on the user's behalf.
    if any(
        status.active and not status.configured
        for status in statuses.values()
    ):
        return False, (
            "User nVHE parameters are active but not persistent; cannot "
            "preserve them across a setup reboot: "
            f"{_format_parameter_states(statuses)}"
        ), True
    # Keep user configuration untouched; append only missing settings.
    added_parameters = [
        parameter for parameter, status in statuses.items()
        if not status.configured
    ]
    # Everything is configured by the user, but some settings are not active.
    # Leave their pending boot changes alone instead of rebooting for them.
    if not added_parameters:
        return False, (
            "User nVHE configuration is pending; leaving it unchanged: "
            f"{_format_parameter_states(statuses)}"
        ), True
    # A resumed run without ownership has lost setup state; do not set up again.
    if sd.progress_check(test_name, GLT_REBOOT_TOKEN):
        return False, (
            "System rebooted without an nVHE ownership token or active parameters"
        ), False

    kernel_identity, identity_reason = _get_running_kernel_identity()
    if identity_reason:
        return False, identity_reason, False

    for parameter in added_parameters:
        logger.info("Appending %s for nVHE validation", parameter)
        ok, reason = glt_utils.append_kernel_parameter(
            parameter,
            test_case_name=test_name,
            session_dir=sd.log_dir,
        )
        if not ok:
            failure = f"Failed to append {parameter}: {reason}"
            return False, _setup_failure_with_cleanup(
                failure, test_name, sd), False
        _, check_reason = _check_nvhe_parameters(test_name, sd)
        if check_reason:
            failure = f"Unable to verify nVHE setup: {check_reason}"
            return False, _setup_failure_with_cleanup(
                failure, test_name, sd), False

    statuses, check_reason = _check_nvhe_parameters(test_name, sd)
    if check_reason:
        return False, _setup_failure_with_cleanup(
            f"Unable to verify complete nVHE setup: {check_reason}",
            test_name,
            sd,
        ), False
    # Every setting must exist for the next boot, including user-owned ones.
    # Only the settings added by this run must carry our transaction ownership.
    if not (
        all(status.configured for status in statuses.values())
        and all(statuses[parameter].added_by_glt for parameter in added_parameters)
    ):
        reason = (
            "GreenLiT did not acquire the complete nVHE parameter set: "
            f"{_format_parameter_states(statuses)}"
        )
        return False, _setup_failure_with_cleanup(reason, test_name, sd), False

    kernel_identity = {**kernel_identity, "added_parameters": added_parameters}
    # This ownership token also pins both reboot phases to this running image.
    sd.progress_update(
        test_name, BRBE_NVHE_OWNERSHIP_TOKEN, kernel_identity)

    all_active = all(status.active for status in statuses.values())
    any_active = any(statuses[parameter].active for parameter in added_parameters)
    # If one of our additions is already active, the whole set must be active.
    if any_active and not all_active:
        reason = (
            "nVHE parameters became only partially active during setup: "
            f"{_format_parameter_states(statuses)}"
        )
        return False, _setup_failure_with_cleanup(reason, test_name, sd), False
    # Already active after setup: clear our saved settings without rebooting yet.
    if all_active:
        cleanup_ok, cleanup_reason = _remove_owned_nvhe_parameters(test_name, sd)
        if not cleanup_ok:
            return False, cleanup_reason, False
        statuses, check_reason = _check_nvhe_parameters(test_name, sd)
        if check_reason:
            return False, _setup_failure_with_cleanup(
                f"Unable to verify active nVHE setup cleanup: {check_reason}",
                test_name,
                sd,
            ), False
        # Cleanup must preserve live settings and drop only our persistent state.
        if not all(status.active for status in statuses.values()) or any(
            statuses[parameter].configured or statuses[parameter].added_by_glt
            for parameter in added_parameters
        ):
            reason = (
                "Active nVHE setup cleanup did not remove persistent state: "
                f"{_format_parameter_states(statuses)}"
            )
            return False, _setup_failure_with_cleanup(reason, test_name, sd), False
        return True, "", False
    # Otherwise every added setting must report that a setup reboot is needed.
    if not all(statuses[parameter].reboot_required for parameter in added_parameters):
        reason = (
            "Complete nVHE configuration did not consistently require a reboot: "
            f"{_format_parameter_states(statuses)}"
        )
        return False, _setup_failure_with_cleanup(reason, test_name, sd), False

    glt_utils.reboot(test_name, sd)

    failure = "Reboot was triggered but the process did not exit"
    return False, _setup_failure_with_cleanup(failure, test_name, sd), False


def _restore_nvhe_boot(sd, attr, logger):
    """Best-effort restore of GLT-owned boot parameters after saving the verdict."""
    test_name = attr.name
    if not sd.progress_check(test_name, BRBE_NVHE_OWNERSHIP_TOKEN):
        return

    # Known cleanup errors must not rewrite the verdict or abort the suite.
    # Keep ownership on failure for diagnosis; completed hooks are not retried.
    identity_reason = _verify_owned_kernel(test_name, sd, "post-hook cleanup")
    if identity_reason:
        logger.error(
            "BRBE nVHE post-hook cleanup skipped: %s; ownership retained. "
            "Check boot parameters before trusting subsequent results.",
            identity_reason,
        )
        return

    ownership = _get_progress_value(sd, test_name, BRBE_NVHE_OWNERSHIP_TOKEN)
    added_parameters = ownership.get(
        "added_parameters", BRBE_NVHE_KERNEL_PARAMETERS
    )
    cleanup_ok, cleanup_reason = _remove_owned_nvhe_parameters(
        test_name, sd, parameters=added_parameters
    )
    statuses, check_reason = _check_nvhe_parameters(test_name, sd)
    if check_reason or not cleanup_ok:
        logger.error(
            "BRBE nVHE post-hook cleanup failed: %s; no restore reboot. "
            "Check boot parameters before trusting subsequent results.",
            _combine_reasons(cleanup_reason, check_reason),
        )
        return
    if any(
        statuses[parameter].configured or statuses[parameter].added_by_glt
        for parameter in added_parameters
    ):
        logger.error(
            "Some GLT-added nVHE parameters remain configured after post-hook "
            "cleanup: %s; no restore reboot. Check the recorded ownership "
            "before trusting subsequent results.",
            _format_parameter_states(statuses),
        )
        return
    reboot_needed = any(
        statuses[parameter].active for parameter in added_parameters
    )
    _clear_nvhe_progress(sd, test_name)
    if not reboot_needed:
        return

    logger.info("[post-hook] Rebooting after removing GLT-added nVHE parameters")
    sd.progress_record_time(test_name, BRBE_NVHE_RESTORE_REBOOT_TOKEN)
    try:
        glt_utils.reboot(test_name, sd)
    except OSError as error:
        reboot_reason = f"Restore reboot failed: {error}"
    else:
        reboot_reason = "Restore reboot was triggered but the process did not exit"

    # The normal reboot exits the process. Restore diagnostic ownership only
    # when it fails or unexpectedly returns; the current boot may still be nVHE.
    sd.progress_update(test_name, BRBE_NVHE_OWNERSHIP_TOKEN, ownership)
    logger.error(
        "%s; temporary nVHE settings may remain active. Reboot manually "
        "before trusting subsequent results.",
        reboot_reason,
    )


def _select_test_cpu() -> tuple[int, str]:
    try:
        online_cpus = parse_cpu_mask(
            CPU_ONLINE_PATH.read_text(encoding="utf-8"))
        allowed_cpus = set(os.sched_getaffinity(0))
    except (OSError, ValueError) as error:
        return -1, f"Unable to select an online CPU: {error}"

    usable_cpus = sorted(online_cpus & allowed_cpus)
    if not usable_cpus:
        return -1, "No online CPU is available in the GreenLiT process affinity"
    return usable_cpus[-1], ""


def _validate_hex_range(
    start_value: str,
    end_value: str,
    start_field: str,
    end_field: str,
    label: str,
) -> str:
    """Validate one non-empty, increasing hexadecimal address range."""
    for field, value in ((start_field, start_value), (end_field, end_value)):
        if not HEX_ADDRESS_PATTERN.fullmatch(value):
            return f"BRBE nVHE result field {field!r} is not hexadecimal"
    if int(start_value, 16) >= int(end_value, 16):
        return f"BRBE nVHE {label} address range is empty or reversed"
    return ""


# pylint: disable=too-many-branches,too-many-return-statements
def _validate_result_json(
    result_path: Path, expected_cpu: int
) -> tuple[dict, str]:
    try:
        with result_path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
    except (OSError, json.JSONDecodeError) as error:
        return {}, f"Unable to read BRBE nVHE result JSON {result_path}: {error}"
    if not isinstance(result, dict):
        return {}, "BRBE nVHE result JSON must contain an object"

    for key, expected_type in RESULT_REQUIRED_TYPES.items():
        if key not in result:
            return {}, f"BRBE nVHE result JSON is missing {key!r}"
        if type(result[key]) is not expected_type:  # pylint: disable=unidiomatic-typecheck
            return {}, (
                f"BRBE nVHE result field {key!r} must be "
                f"{expected_type.__name__}"
            )

    unknown_fields = sorted(set(result).difference(RESULT_REQUIRED_TYPES))
    if unknown_fields:
        return {}, (
            "BRBE nVHE result JSON contains unknown fields: "
            + ", ".join(repr(field) for field in unknown_fields)
        )

    expected_values = {
        "schema_version": 3,
        "iterations": BRBE_NVHE_ITERATIONS,
        "guest_branches": BRBE_NVHE_GUEST_BRANCHES,
        "sample_period": BRBE_NVHE_SAMPLE_PERIOD,
    }
    for key, expected_value in expected_values.items():
        if result[key] != expected_value:
            return {}, (
                f"Harness reported {key}={result[key]}, expected {expected_value}"
            )

    completed = result["iterations_completed"]
    if completed < 0 or completed > BRBE_NVHE_ITERATIONS:
        return {}, (
            f"Harness reported iterations_completed={completed}, expected a "
            f"value from 0 through {BRBE_NVHE_ITERATIONS}"
        )
    if result["exit_code"] == 0 and completed != BRBE_NVHE_ITERATIONS:
        return {}, (
            f"Harness reported iterations_completed={completed}, expected "
            f"{BRBE_NVHE_ITERATIONS} for fixed evidence"
        )
    if result["exit_code"] == 1 and completed == 0:
        return {}, (
            "Harness reported contamination without completing a guest iteration"
        )
    if result["cpu"] != expected_cpu:
        return {}, (
            f"Harness reported CPU {result['cpu']}, expected {expected_cpu}"
        )
    if result["nvhe_cmdline"] is not True:
        return {}, (
            "Harness did not confirm the required nVHE kernel command line"
        )

    count_fields = (
        "max_pre_guest_count", "samples", "entries", "guest_hits",
        "host_pre_hits", "host_post_hits", "lost_records",
        "throttle_records", "context_switches", "guest_samples",
        "guest_windows_verified", "min_guest_instruction_delta",
    )
    for key in count_fields:
        if result[key] < 0:
            return {}, f"BRBE nVHE result field {key!r} cannot be negative"

    pre_guest_count_limit = result["sample_period"] // 4
    if result["exit_code"] == 0 and result["max_pre_guest_count"] >= pre_guest_count_limit:
        return {}, (
            "Harness reported max_pre_guest_count="
            f"{result['max_pre_guest_count']}, expected less than "
            f"{pre_guest_count_limit}"
        )

    range_reason = _validate_hex_range(
        result["guest_range_start"], result["guest_range_end"],
        "guest_range_start", "guest_range_end", "guest",
    )
    if range_reason:
        return {}, range_reason
    for field, label in (
        ("host_pre_range", "host pre-control"),
        ("host_post_range", "host post-control"),
    ):
        match = ADDRESS_RANGE_PATTERN.fullmatch(result[field])
        if not match:
            return {}, (
                f"BRBE nVHE result field {field!r} must contain a "
                "hexadecimal start-end range"
            )
        range_reason = _validate_hex_range(
            match.group(1), match.group(2),
            f"{field}_start", f"{field}_end", label,
        )
        if range_reason:
            return {}, range_reason

    if any(result[key] > result["entries"] for key in (
        "guest_hits", "host_pre_hits", "host_post_hits"
    )):
        return {}, "BRBE nVHE hit count exceeds the decoded entry count"
    if result["guest_samples"] > result["samples"]:
        return {}, "Guest-IP sample count exceeds the perf sample count"
    if result["guest_windows_verified"] > result["guest_samples"]:
        return {}, "Verified guest-window count exceeds guest-IP samples"
    if result["guest_windows_verified"] > completed:
        return {}, "Verified guest-window count exceeds completed iterations"
    if result["guest_windows_verified"] > result["entries"]:
        return {}, "Verified guest-window count exceeds branch entries"
    return result, ""


def _classify_harness_result(result: dict, returncode: int) -> tuple[str, bool]:
    """Return (reason, should_skip); ambiguous evidence never passes."""
    expected_verdict = HARNESS_EXIT_VERDICTS.get(result["exit_code"])
    if expected_verdict is None:
        return (
            "BRBE nVHE harness JSON contains unsupported exit_code "
            f"{result['exit_code']}"
        ), False
    if returncode != result["exit_code"]:
        return (
            f"BRBE nVHE harness process exit status {returncode} does not "
            f"match JSON exit_code {result['exit_code']}"
        ), False
    if result["verdict"] != expected_verdict:
        return (
            f"BRBE nVHE harness JSON verdict {result['verdict']!r} does not "
            f"match exit_code {result['exit_code']}; expected "
            f"{expected_verdict!r}"
        ), False

    # Positive contamination evidence cannot be downgraded by missing controls.
    if result["guest_hits"] > 0:
        return _combine_reasons(
            f"BRBE nVHE host isolation failed: decoded {result['guest_hits']} "
            "guest-range branch entries. Consider checking whether the kernel "
            "is missing d7567e9b9ba5.",
            result["reason"],
        ), False
    if result["exit_code"] == 1:
        return (
            f"BRBE nVHE harness classified the result as {result['verdict']!r} "
            f"with exit code {result['exit_code']} despite non-contaminated "
            "branch evidence."
        ), False
    if (
        result["guest_brbe_id"] != 0
        or result["guest_brbcr_undef"] is not True
        or result["guest_unexpected_exception"]
    ):
        return (
            "Guest BRBE isolation prerequisites are invalid: "
            f"guest_brbe_id={result['guest_brbe_id']}, "
            f"guest_brbcr_undef={result['guest_brbcr_undef']}, "
            f"guest_unexpected_exception={result['guest_unexpected_exception']}. "
            "Host isolation could not be determined."
        ), True

    # Exit 2 is observer failure; a claimed fixed result with missing controls
    # is instead inconsistent harness output and remains a test failure.
    observer_reason = ""
    if result["max_pre_guest_count"] >= BRBE_NVHE_SAMPLE_PERIOD // 4:
        observer_reason = (
            "BRBE nVHE result is inconclusive because the pre-guest sampling "
            "window was not isolated: max_pre_guest_count="
            f"{result['max_pre_guest_count']}"
        )
    elif result["lost_records"] or result["throttle_records"]:
        observer_reason = (
            "BRBE nVHE result is inconclusive because perf evidence was "
            f"incomplete: lost_records={result['lost_records']}, "
            f"throttle_records={result['throttle_records']}."
        )
    elif not result["samples"] or not result["entries"]:
        observer_reason = (
            "BRBE nVHE result is inconclusive because perf produced no usable "
            f"evidence: samples={result['samples']}, entries={result['entries']}."
        )
    elif not result["host_pre_hits"] or not result["host_post_hits"]:
        observer_reason = (
            "BRBE nVHE result is inconclusive because host control branches "
            "were not observed before and after guest execution: "
            f"host_pre_hits={result['host_pre_hits']}, "
            f"host_post_hits={result['host_post_hits']}."
        )
    elif (
        result["guest_windows_verified"] != result["iterations_completed"]
        or result["guest_samples"] != result["iterations_completed"]
        or result["min_guest_instruction_delta"]
        < result["sample_period"] - result["max_pre_guest_count"]
    ):
        observer_reason = (
            "BRBE nVHE result lacks verified guest-window sampling: "
            f"verified={result['guest_windows_verified']}, "
            f"guest_samples={result['guest_samples']}, "
            f"completed={result['iterations_completed']}, "
            f"min_instruction_delta={result['min_guest_instruction_delta']}"
        )
    if observer_reason:
        if result["exit_code"] == 2:
            return observer_reason, True
        return f"Inconsistent fixed BRBE harness result: {observer_reason}", False
    if result["exit_code"] == 2:
        return _combine_reasons(
            "BRBE nVHE harness classified the result as 'inconclusive' with "
            "exit code 2; host isolation could not be determined.",
            result["reason"],
        ), True
    if result["reason"]:
        return (
            "BRBE nVHE harness reported a failure reason despite otherwise "
            "passing evidence."
        ), False
    return "", False


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches,too-many-statements
def test_brbe_nvhe_host_isolation(sd, logger):
    """Ensure nVHE guest branches do not contaminate host BRBE records.

    PATCH: https://git.kernel.org/torvalds/c/d7567e9b9ba5
    """
    test_name = test_brbe_nvhe_host_isolation.__name__
    if sd.progress_check(test_name, GLT_REBOOT_TOKEN):
        glt_utils.systemd_enable(enable=False)


    prerequisite_failure = ""
    if not glt_utils.is_cli_available("make"):
        prerequisite_failure = (
            "Please install 'make' to build the BRBE nVHE harness"
        )
    elif not HARNESS_MAKEFILE_PATH.is_file():
        prerequisite_failure = (
            f"BRBE nVHE harness Makefile is missing: {HARNESS_MAKEFILE_PATH}"
        )
    else:
        missing_sources = [
            str(path) for path in KVM_MICROVM_REQUIRED_FILES
            if not path.is_file()
        ]
        if missing_sources:
            prerequisite_failure = (
                "BRBE nVHE common KVM source is missing: "
                + ", ".join(missing_sources)
            )
    if prerequisite_failure:
        GLTCore.GLTTestResult.fail(prerequisite_failure)

    # TODO(DGX-18215): Once the CLI can include prerequisites across plans, restore
    # must_pass("arm64", "test_brbe_branch_stack") and remove these duplicate
    # Vera/kernel/PMU checks. Until then, reject unsupported environments before
    # changing boot parameters. CPU selection and harness controls must remain.
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not an NVIDIA Vera CPU; BRBE nVHE validation is Vera-specific."
        )
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except (OSError, ValueError) as error:
        GLTCore.GLTTestResult.skip(
            f"Unable to inspect kernel config {kconfig_path}: {error}"
        )
    if kconfig.get("CONFIG_ARM64_BRBE") != "y":
        GLTCore.GLTTestResult.skip(
            "CONFIG_ARM64_BRBE is not enabled; guest isolation cannot be observed"
        )
    try:
        perf_utils.read_brbe_capabilities()
    except ValueError as error:
        GLTCore.GLTTestResult.skip(str(error))

    # Unsupported environments must SKIP before CPU inspection can fail.
    cpu, cpu_reason = _select_test_cpu()
    if cpu_reason:
        GLTCore.GLTTestResult.fail(cpu_reason)

    ready, preparation_reason, should_skip = _ensure_nvhe_environment(
        test_name, sd, logger)
    if should_skip:
        GLTCore.GLTTestResult.skip(preparation_reason)
    if not ready:
        GLTCore.GLTTestResult.fail(preparation_reason)

    harness_path = Path(sd.log_dir) / "glt_brbe_nvhe_harness"
    result_path = Path(sd.log_dir) / "brbe_nvhe_result.json"
    entries_path = Path(sd.log_dir) / "brbe_nvhe_entries.txt"
    failure_reason = _get_nvhe_log_failure()
    skip_reason = ""
    returncode = -1
    try:
        os.makedirs(sd.log_dir, exist_ok=True)
        if not failure_reason:
            failure_reason = _remove_files(
                (harness_path, result_path, entries_path))
        if not failure_reason:
            build_command = [
                "make", "-C", str(HARNESS_SOURCE_DIR),
                f"OUTPUT={harness_path}",
                f"KVM_MICROVM_DIR={KVM_MICROVM_SOURCE_DIR}",
            ]
            logger.info("Building BRBE nVHE harness at %s", harness_path)
            try:
                build_result = glt_utils.run_command(
                    build_command,
                    "BRBE nVHE harness build failed",
                    timeout=BRBE_NVHE_BUILD_TIMEOUT_SECONDS,
                )
            except RuntimeError as error:
                failure_reason = str(error)
            else:
                if not harness_path.is_file() or not os.access(
                    harness_path, os.X_OK
                ):
                    failure_reason = (
                        "BRBE nVHE harness build did not create "
                        f"{harness_path}"
                    )
                elif build_result.stdout:
                    logger.debug(
                        "BRBE nVHE harness build output:\n%s",
                        build_result.stdout.rstrip(),
                    )
        if not failure_reason:
            harness_command = [
                str(harness_path),
                "--cpu", str(cpu),
                "--iterations", str(BRBE_NVHE_ITERATIONS),
                "--guest-branches", str(BRBE_NVHE_GUEST_BRANCHES),
                "--sample-period", str(BRBE_NVHE_SAMPLE_PERIOD),
                "--result-json", str(result_path),
                "--entries", str(entries_path),
            ]
            logger.info("Running BRBE nVHE harness on CPU %d", cpu)
            try:
                harness_result = subprocess.run(
                    harness_command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=BRBE_NVHE_HARNESS_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                failure_reason = (
                    "BRBE nVHE harness timed out after "
                    f"{error.timeout} seconds"
                )
            except OSError as error:
                failure_reason = (
                    f"Failed to execute BRBE nVHE harness: {error}"
                )
            else:
                returncode = harness_result.returncode
                if harness_result.stdout:
                    logger.debug(
                        "BRBE nVHE harness stdout:\n%s",
                        harness_result.stdout.rstrip(),
                    )
                if harness_result.stderr:
                    logger.debug(
                        "BRBE nVHE harness stderr:\n%s",
                        harness_result.stderr.rstrip(),
                    )

        if not failure_reason:
            result, failure_reason = _validate_result_json(result_path, cpu)
            if not failure_reason:
                logger.info(
                    "BRBE nVHE evidence: samples=%d entries=%d guest_hits=%d "
                    "max_pre_guest_count=%d host_pre_hits=%d "
                    "host_post_hits=%d guest_range=%s-%s",
                    result["samples"], result["entries"], result["guest_hits"],
                    result["max_pre_guest_count"], result["host_pre_hits"],
                    result["host_post_hits"],
                    result["guest_range_start"], result["guest_range_end"],
                )
                classification_reason, should_skip = _classify_harness_result(
                        result, returncode)
                if should_skip:
                    skip_reason = classification_reason
                else:
                    failure_reason = classification_reason
    except OSError as error:
        failure_reason = _combine_reasons(
            failure_reason, f"BRBE nVHE runtime setup failed: {error}")
    except Exception as error:  # pylint: disable=broad-exception-caught
        failure_reason = _combine_reasons(
            failure_reason,
            f"BRBE nVHE validation raised an unexpected exception: {error!r}",
        )
    finally:
        cleanup_reason = _remove_files((harness_path,))
    failure_reason = _combine_reasons(failure_reason, cleanup_reason)

    if failure_reason:
        GLTCore.GLTTestResult.fail(failure_reason)
    if skip_reason:
        GLTCore.GLTTestResult.skip(skip_reason)

    logger.info("BRBE nVHE host-isolation validation passed on CPU %d", cpu)


test_brbe_nvhe_host_isolation.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_restore_nvhe_boot,
    postskip=_restore_nvhe_boot,
    postafail=_restore_nvhe_boot,
    postfail=_restore_nvhe_boot,
    postwarn=_restore_nvhe_boot,
)
