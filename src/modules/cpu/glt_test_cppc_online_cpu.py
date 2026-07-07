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

"""ACPI CPPC online-CPU validation for DGX-16331."""

import glob
import os
import platform

from common import glt_core as GLTCore
from common import glt_utils


CPU_SYSFS_ROOT = "/sys/devices/system/cpu"
CPPC_MAXCPUS_PARAM_MARKER = "CPPCMaxcpusParamAdded"
CPPC_MAXCPUS_PARAM_VALUE = "CPPCMaxcpusParam"
CPPC_RESTORE_REBOOT_MARKER = "CPPCRestoreReboot"
CPPC_RESTORE_RESULT_REASON = "CPPCRestoreResultReason"
GLT_REBOOT_MARKER = "TriggerReboot"
MIN_REBOOT_CPUS = 1


def _parse_cpu_mask(mask_text):
    """Parse a Linux CPU mask such as '0-3,8' into a set of CPU numbers."""
    cpus = set()
    mask_text = mask_text.strip()
    if not mask_text:
        return cpus

    for part in mask_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            cpus.add(int(part))
            continue

        start_text, end_text = part.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid CPU range: {part}")
        cpus.update(range(start, end + 1))

    return cpus


def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_cpu_mask(name):
    return _parse_cpu_mask(_read_file(os.path.join(CPU_SYSFS_ROOT, name)))


def _has_partial_online_topology(present_cpus, online_cpus):
    """Return True when some present CPUs are not currently online."""
    return bool(present_cpus - online_cpus)


def _recommended_maxcpus(present_cpus):
    """Choose a conservative maxcpus value that leaves offline CPUs at boot."""
    if len(present_cpus) <= 1:
        return MIN_REBOOT_CPUS
    return max(MIN_REBOOT_CPUS, len(present_cpus) // 2)


def _has_boot_time_cpu_limit(cmdline):
    tokens = cmdline.split()
    return any(token.startswith("maxcpus=") or token.startswith("nr_cpus=") for token in tokens)


def _get_progress_value(sd, test_name, key):
    for entry in getattr(sd, "progress", []):
        if test_name in entry:
            return entry[test_name].get(key)
    return None


def _clear_maxcpus_progress(sd, test_name):
    sd.progress_delete_time(test_name, CPPC_MAXCPUS_PARAM_MARKER)
    sd.progress_delete_time(test_name, CPPC_MAXCPUS_PARAM_VALUE)


def _get_maxcpus_parameter(sd, test_name):
    parameter = _get_progress_value(sd, test_name, CPPC_MAXCPUS_PARAM_VALUE)
    if parameter:
        return parameter
    return ""


def _finish_restore_reboot_if_needed(test_name, sd):
    if not sd.progress_check(test_name, CPPC_RESTORE_REBOOT_MARKER):
        return False, ""

    parameter = _get_maxcpus_parameter(sd, test_name)
    if not parameter:
        sd.progress_delete_time(test_name, CPPC_RESTORE_REBOOT_MARKER)
        return True, (
            "Restore reboot marker exists, but no GreenLiT-owned maxcpus parameter "
            "was recorded."
        )

    status = glt_utils.check_kernel_parameter(
        parameter,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if status.active:
        return True, (
            f"System rebooted after removing {parameter}, but the parameter is still active."
        )
    if status.configured or status.added_by_glt:
        return True, (
            f"System rebooted after removing {parameter}, but the parameter is still configured."
        )

    prior_reason = _get_progress_value(sd, test_name, CPPC_RESTORE_RESULT_REASON) or ""
    sd.progress_delete_time(test_name, CPPC_RESTORE_REBOOT_MARKER)
    sd.progress_delete_time(test_name, CPPC_RESTORE_RESULT_REASON)
    sd.progress_delete_time(test_name, CPPC_MAXCPUS_PARAM_MARKER)
    sd.progress_delete_time(test_name, CPPC_MAXCPUS_PARAM_VALUE)
    return True, prior_reason


# pylint: disable=too-many-return-statements
def _prepare_maxcpus_boot_parameter(test_name, sd, logger, present_cpus, online_cpus):
    """Append maxcpus= and reboot when current boot has every present CPU online."""
    if _has_partial_online_topology(present_cpus, online_cpus):
        return True, ""

    maxcpus = _recommended_maxcpus(present_cpus)
    parameter = f"maxcpus={maxcpus}"
    status = glt_utils.check_kernel_parameter(
        parameter,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )

    if sd.progress_check(test_name, GLT_REBOOT_MARKER):
        failure_reason = (
            f"System rebooted after configuring {parameter}, but all present CPUs "
            "are still online."
        )
        restore_ok, restore_reason = _restore_maxcpus_boot_parameter(
            test_name, sd, logger, failure_reason
        )
        if restore_ok:
            return False, failure_reason
        return False, f"{failure_reason}; additionally, {restore_reason}"

    if status.active:
        return False, (
            f"{parameter} is active, but all present CPUs are still online; "
            "cannot create DGX-16331 never-onlined CPU topology."
        )

    logger.info("Appending %s for DGX-16331 CPPC online-CPU validation.", parameter)
    ok, reason = glt_utils.append_kernel_parameter(
        parameter,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if not ok:
        cleanup_ok, cleanup_reason = _remove_maxcpus_boot_parameter(test_name, sd, parameter)
        if cleanup_ok:
            return False, f"Failed to append kernel parameter: {reason}"
        return False, (
            f"Failed to append kernel parameter: {reason}; additionally, {cleanup_reason}"
        )

    status = glt_utils.check_kernel_parameter(
        parameter,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    sd.progress_record_time(test_name, CPPC_MAXCPUS_PARAM_MARKER)
    sd.progress_update(test_name, CPPC_MAXCPUS_PARAM_VALUE, parameter)

    if not status.reboot_required:
        cleanup_ok, cleanup_reason = _remove_maxcpus_boot_parameter(test_name, sd, parameter)
        if cleanup_ok:
            _clear_maxcpus_progress(sd, test_name)
            return False, (
                f"{parameter} was configured but no reboot was required; "
                "cannot create DGX-16331 never-onlined CPU topology."
            )
        return False, (
            f"{parameter} was configured but no reboot was required; additionally, "
            f"{cleanup_reason}"
        )

    glt_utils.reboot(test_name, sd)

    cleanup_ok, cleanup_reason = _remove_maxcpus_boot_parameter(test_name, sd, parameter)
    if not cleanup_ok:
        return False, f"Reboot did not exit, and {cleanup_reason}"
    return False, "Reboot was triggered but the process did not exit."


def _remove_maxcpus_boot_parameter(test_name, sd, parameter=None):
    parameter = parameter or _get_maxcpus_parameter(sd, test_name)
    if not parameter:
        return True, ""

    ok, reason = glt_utils.remove_kernel_parameter(
        parameter,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if not ok:
        return False, f"Failed to remove kernel command-line configuration: {reason}"
    return True, ""


def _restore_maxcpus_boot_parameter(test_name, sd, logger, result_reason=""):
    """Remove GreenLiT-owned maxcpus= configuration and reboot if still active."""
    parameter = _get_maxcpus_parameter(sd, test_name)
    if not parameter:
        return True, ""

    if result_reason:
        sd.progress_update(test_name, CPPC_RESTORE_RESULT_REASON, result_reason)

    ok, reason = _remove_maxcpus_boot_parameter(test_name, sd, parameter)
    if not ok:
        return False, reason

    status = glt_utils.check_kernel_parameter(
        parameter,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if not status.active:
        _clear_maxcpus_progress(sd, test_name)
        sd.progress_delete_time(test_name, CPPC_RESTORE_RESULT_REASON)
        return True, ""

    logger.info("Rebooting to restore kernel command-line state after CPPC validation.")
    sd.progress_record_time(test_name, CPPC_RESTORE_REBOOT_MARKER)
    glt_utils.reboot(test_name, sd)
    return False, "Restore reboot was triggered but the process did not exit."


def _read_int_file(path, read_file=_read_file):
    value = read_file(path).strip()
    if not value:
        raise ValueError(f"{path} is empty")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{path} is not positive: {parsed}")
    return parsed


def _policy_cpu_mask(policy_dir, filename, read_file):
    return _parse_cpu_mask(read_file(os.path.join(policy_dir, filename)))


def _validate_policy(policy_dir, online_cpus, read_file):
    driver = read_file(os.path.join(policy_dir, "scaling_driver")).strip()
    if driver != "cppc_cpufreq":
        return set(), f"{policy_dir} uses {driver}, expected cppc_cpufreq"

    related = _policy_cpu_mask(policy_dir, "related_cpus", read_file)
    affected = _policy_cpu_mask(policy_dir, "affected_cpus", read_file)
    if not related:
        return set(), f"{policy_dir}/related_cpus is empty"
    if not affected:
        return set(), f"{policy_dir}/affected_cpus is empty"
    if not affected.issubset(online_cpus):
        return set(), (
            f"{policy_dir}/affected_cpus includes offline CPUs: "
            f"{sorted(affected - online_cpus)}"
        )
    if not related.issubset(online_cpus):
        return set(), (
            f"{policy_dir}/related_cpus includes offline CPUs: "
            f"{sorted(related - online_cpus)}"
        )

    _read_int_file(os.path.join(policy_dir, "scaling_cur_freq"), read_file)
    _read_int_file(os.path.join(policy_dir, "cpuinfo_max_freq"), read_file)
    return affected, ""


def _validate_cppc_observability(online_cpus, policy_dirs=None, read_file=_read_file):
    """Validate CPPC sysfs evidence on every online CPU."""
    for cpu in sorted(online_cpus):
        try:
            _read_int_file(
                os.path.join(CPU_SYSFS_ROOT, f"cpu{cpu}", "acpi_cppc", "highest_perf"),
                read_file,
            )
        except (OSError, ValueError, KeyError) as e:
            return False, f"Cannot read positive acpi_cppc/highest_perf for CPU{cpu}: {e}"

    if policy_dirs is None:
        policy_dirs = sorted(glob.glob(os.path.join(CPU_SYSFS_ROOT, "cpufreq", "policy[0-9]*")))
        if not policy_dirs:
            policy_dirs = sorted(glob.glob(os.path.join(CPU_SYSFS_ROOT, "cpu[0-9]*", "cpufreq")))
    if not policy_dirs:
        return False, "cpufreq sysfs policy directories are not available."

    covered_cpus = set()
    for policy_dir in policy_dirs:
        try:
            affected, reason = _validate_policy(policy_dir, online_cpus, read_file)
        except (OSError, ValueError, KeyError) as e:
            return False, f"Cannot validate {policy_dir}: {e}"
        if reason:
            return False, reason
        covered_cpus.update(affected)

    for cpu in sorted(online_cpus):
        if cpu not in covered_cpus:
            return False, f"No cppc_cpufreq policy covers online CPU{cpu}"

    return True, ""


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-return-statements,too-many-branches
def test_cppc_online_cpu_coverage(sd, logger):
    """
    Validate ACPI CPPC on ARM64 systems booted with never-onlined CPUs.

    JIRA: DGX-16331
    PATCHES:
        https://git.kernel.org/torvalds/c/6dd3b8a709a1
        https://git.kernel.org/torvalds/c/8821c8e80a65
        https://git.kernel.org/torvalds/c/0fce75870666
        https://git.kernel.org/torvalds/c/56eb0c0ed345
    """
    test_name = test_cppc_online_cpu_coverage.__name__

    if sd.progress_check(test_name, GLT_REBOOT_MARKER):
        glt_utils.systemd_enable(enable=False)

    restore_complete, restore_reason = _finish_restore_reboot_if_needed(test_name, sd)
    if restore_complete:
        if restore_reason:
            GLTCore.GLTTestResult.fail(restore_reason)
        return

    if platform.machine() not in ("aarch64", "arm64"):
        GLTCore.GLTTestResult.skip("CPPC online-CPU coverage is ARM64-only.")

    try:
        present_cpus = _read_cpu_mask("present")
        online_cpus = _read_cpu_mask("online")
        cmdline = _read_file("/proc/cmdline").strip()
    except (OSError, ValueError) as e:
        GLTCore.GLTTestResult.fail(f"Cannot read CPU topology: {e}")

    if not online_cpus:
        GLTCore.GLTTestResult.fail("No online CPUs reported by sysfs.")
    if not present_cpus:
        GLTCore.GLTTestResult.fail("No present CPUs reported by sysfs.")

    ok, reason = _prepare_maxcpus_boot_parameter(
        test_name, sd, logger,
        present_cpus=present_cpus,
        online_cpus=online_cpus,
    )
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    if not _has_partial_online_topology(present_cpus, online_cpus):
        GLTCore.GLTTestResult.skip(
            "All present CPUs are online; DGX-16331 bug condition is not exercisable."
        )

    if not _has_boot_time_cpu_limit(cmdline):
        logger.warning(
            "CPU topology is partial, but /proc/cmdline lacks maxcpus=/nr_cpus= evidence; "
            "result is weaker because runtime-offlined CPUs cannot be ruled out."
        )

    ok, reason = _validate_cppc_observability(online_cpus)
    if not ok:
        restore_ok, restore_reason = _restore_maxcpus_boot_parameter(
            test_name, sd, logger, reason
        )
        if restore_ok:
            GLTCore.GLTTestResult.fail(reason)
        GLTCore.GLTTestResult.fail(f"{reason}; additionally, {restore_reason}")

    failure_reason = ""
    for signature in ("cppc_allow_fast_switch", "cppc_perf_ctrs_in_pcc"):
        if glt_utils.check_kernel_log(signature):
            failure_reason = f"Kernel log contains CPPC failure signature: {signature}"
            break
    if failure_reason:
        restore_ok, restore_reason = _restore_maxcpus_boot_parameter(
            test_name, sd, logger, failure_reason
        )
        if restore_ok:
            GLTCore.GLTTestResult.fail(failure_reason)
        GLTCore.GLTTestResult.fail(f"{failure_reason}; additionally, {restore_reason}")

    restore_ok, restore_reason = _restore_maxcpus_boot_parameter(test_name, sd, logger)
    if not restore_ok:
        GLTCore.GLTTestResult.fail(restore_reason)
