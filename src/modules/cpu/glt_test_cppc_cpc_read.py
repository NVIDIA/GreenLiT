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

"""Check CPPC cpc_read error propagation through sysfs."""

import errno
import glob
import os
import platform
import subprocess

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config

CPPC_CPC_READ_PATCH = "https://git.kernel.org/torvalds/c/0cc24977224a"
CPPC_CPC_READ_PROBE_MODULE = "cppc_cpc_read_probe"
CPPC_CPC_READ_PROBE_PROC_PATH = "/proc/greenlit_cppc_cpc_read"
CPPC_CPC_READ_PROBE_SYSFS_PATH = (
    f"/sys/module/{CPPC_CPC_READ_PROBE_MODULE}"
)
CPPC_CPC_READ_PROBE_ARM_COMMAND = "arm"
CPPC_CPC_READ_PROBE_DISARM_COMMAND = "disarm"
CPPC_CPC_READ_REQUIRED_CONFIGS = (
    "CONFIG_KPROBES",
    "CONFIG_KRETPROBES",
    "CONFIG_MODULE_UNLOAD",
)
CPPC_CPC_READ_SYMBOL = "cpc_read"
CPPC_CPC_READ_COMMAND_TIMEOUT_SECS = 120
CPPC_CPC_READ_SYSFS_GETTERS = {
    "highest_perf": "cppc_get_perf_caps",
    "feedback_ctrs": "cppc_get_perf_ctrs",
}
CPU_SYSFS_ROOT = "/sys/devices/system/cpu"


def _is_cppc_cpc_read_probe_present(path_exists=os.path.exists):
    """Return whether either userspace path exposes a loaded probe."""
    return (path_exists(CPPC_CPC_READ_PROBE_PROC_PATH) or
            path_exists(CPPC_CPC_READ_PROBE_SYSFS_PATH))


def _cleanup_cppc_cpc_read_probe(
        module_loaded, logger, probe_may_be_owned):
    """Unload the probe and verify that its userspace paths disappear."""
    cleanup_errors = []
    probe_present = module_loaded or (
        probe_may_be_owned and _is_cppc_cpc_read_probe_present()
    )
    unload_error = None
    if module_loaded:
        try:
            glt_utils.write_sysfs_value(
                CPPC_CPC_READ_PROBE_PROC_PATH,
                CPPC_CPC_READ_PROBE_DISARM_COMMAND,
            )
        except OSError as error:
            logger.warning(
                "Failed to disarm the cpc_read probe during cleanup: %s", error
            )
    if probe_present:
        try:
            glt_utils.run_command(
                ["rmmod", CPPC_CPC_READ_PROBE_MODULE],
                f"Failed to unload {CPPC_CPC_READ_PROBE_MODULE}",
                timeout=CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
            )
        except RuntimeError as error:
            unload_error = str(error)

    probe_remains = probe_present and _is_cppc_cpc_read_probe_present()
    if probe_remains:
        if unload_error:
            cleanup_errors.append(unload_error)
        cleanup_errors.append(
            f"{CPPC_CPC_READ_PROBE_MODULE} remains loaded; "
            "the cpc_read kretprobe may still be active"
        )
    elif unload_error:
        logger.warning(
            "Probe paths disappeared despite the rmmod error: %s", unload_error
        )

    for error in cleanup_errors:
        logger.error(error)
    return cleanup_errors


def _find_cppc_cpc_read_sysfs_oracles(path_isfile=os.path.isfile):
    """Find one CPU with the most readable affected sysfs getters."""
    cppc_dirs = sorted(glob.glob(os.path.join(
        CPU_SYSFS_ROOT, "cpu[0-9]*", "acpi_cppc"
    )))
    read_errors = []
    best_candidate = None
    best_candidate_size = 0
    for cppc_dir in cppc_dirs:
        paths = {}
        baselines = {}
        candidate_errors = []
        for filename in CPPC_CPC_READ_SYSFS_GETTERS:
            path = os.path.join(cppc_dir, filename)
            if not path_isfile(path):
                continue
            try:
                value = glt_utils.read_sysfs_value(path)
            except OSError as error:
                detail = f"{path}: {error}"
                read_errors.append(detail)
                candidate_errors.append(detail)
                continue
            if not value:
                detail = f"{path}: empty value"
                read_errors.append(detail)
                candidate_errors.append(detail)
                continue
            paths[filename] = path
            baselines[filename] = value

        if not paths:
            continue
        cpu_name = os.path.basename(os.path.dirname(cppc_dir))
        candidate_reason = None
        if candidate_errors:
            candidate_reason = (
                "Some CPPC sysfs oracles on the selected CPU are unavailable: "
                + "; ".join(candidate_errors)
            )
        candidate = (cpu_name, paths, baselines, candidate_reason)
        if len(paths) == len(CPPC_CPC_READ_SYSFS_GETTERS):
            return candidate
        if len(paths) > best_candidate_size:
            best_candidate = candidate
            best_candidate_size = len(paths)

    if best_candidate is not None:
        return best_candidate

    if read_errors:
        examples = "; ".join(read_errors[:3])
        remaining = len(read_errors) - 3
        if remaining > 0:
            examples += f"; and {remaining} more"
        return None, None, None, (
            "No CPU has either readable highest_perf or feedback_ctrs sysfs "
            f"attributes: {examples}"
        )
    return None, None, None, (
        "No CPU exposes either acpi_cppc/highest_perf or "
        "acpi_cppc/feedback_ctrs."
    )


def _is_kernel_symbol_available(
        symbol, kallsyms_path="/proc/kallsyms", opener=open):
    """Return whether an exact symbol is visible in /proc/kallsyms."""
    with opener(kallsyms_path, encoding="utf-8", errors="ignore") as kallsyms:
        return any(
            len(fields) >= 3 and fields[2] == symbol
            for fields in (line.split() for line in kallsyms)
        )


def _parse_cppc_cpc_read_probe_status(value):
    """Parse the injection and missed-instance counters from procfs."""
    fields = value.split()
    if len(fields) != 2:
        raise ValueError("expected two counters")
    injections, missed_instances = (int(field) for field in fields)
    if injections < 0 or missed_instances < 0:
        raise ValueError("observer counters must be non-negative")
    return injections, missed_instances


def _cppc_cpc_read_zero_injection_reason(path, getter, missed_instances):
    """Explain whether zero injections came with observer saturation."""
    if missed_instances:
        return (
            "The cpc_read kretprobe ran out of instances system-wide "
            f"{missed_instances} times while reading {path}; the patch result "
            "is inconclusive."
        )
    return (
        f"Reading {path} did not complete a successful, injectable "
        f"cpc_read() in {getter}; the patch cannot be checked on this platform."
    )


def _classify_cppc_cpc_read_observation(filename, path, observation, logger):
    """Classify one completed baseline/injection sysfs observation."""
    getter = CPPC_CPC_READ_SYSFS_GETTERS[filename]
    injected_value = observation["injected_value"]
    read_error = observation["read_error"]
    status_error = observation["status_error"]
    injections = observation["injections"]
    missed_instances = observation["missed_instances"]
    if status_error is not None:
        result = GLTCore.GLTTestResult.WARN, (
            f"Cannot read the cpc_read observer status for {path}: "
            f"{status_error}; the patch result is inconclusive."
        )
    elif injections == 0:
        result = (
            GLTCore.GLTTestResult.WARN,
            _cppc_cpc_read_zero_injection_reason(
                path, getter, missed_instances
            ),
        )
    elif injections != 1:
        result = GLTCore.GLTTestResult.WARN, (
            f"The one-shot cpc_read probe reported {injections} injections "
            f"while reading {path}; the patch result is inconclusive."
        )
    elif read_error is None:
        logger.info("Injected %s value: %s", filename, injected_value)
        result = GLTCore.GLTTestResult.FAIL, (
            f"{path} still succeeded after cpc_read() returned -EIO; "
            f"{getter} did not propagate the error. Apply {CPPC_CPC_READ_PATCH}."
        )
    elif read_error.errno is None:
        result = GLTCore.GLTTestResult.WARN, (
            f"{path} raised an error without errno after cpc_read injection: "
            f"{read_error}; the patch result is inconclusive."
        )
    elif read_error.errno != errno.EIO:
        result = GLTCore.GLTTestResult.WARN, (
            f"{path} returned errno {read_error.errno} after an injected "
            f"cpc_read() -EIO; expected errno {errno.EIO}: {read_error}. "
            "The patch result is inconclusive."
        )
    else:
        result = GLTCore.GLTTestResult.PASS, None
    return result


def _run_cppc_cpc_read_sysfs_oracle(filename, path, logger):
    """Inject one cpc_read error and check the synchronous sysfs result."""
    read_error = None
    status_error = None
    injected_value = None
    injections = None
    missed_instances = None
    try:
        # Keep this write in-process: the probe records the caller's TGID.
        glt_utils.write_sysfs_value(
            CPPC_CPC_READ_PROBE_PROC_PATH,
            CPPC_CPC_READ_PROBE_ARM_COMMAND,
        )
    except OSError as error:
        return GLTCore.GLTTestResult.SKIP, (
            f"Cannot arm the cpc_read observer for {path}: {error}; "
            "the patch cannot be checked."
        )
    try:
        try:
            injected_value = glt_utils.read_sysfs_value(path)
        except OSError as error:
            read_error = error
        try:
            injections, missed_instances = _parse_cppc_cpc_read_probe_status(
                glt_utils.read_sysfs_value(CPPC_CPC_READ_PROBE_PROC_PATH)
            )
        except (OSError, ValueError) as error:
            status_error = error
    finally:
        try:
            glt_utils.write_sysfs_value(
                CPPC_CPC_READ_PROBE_PROC_PATH,
                CPPC_CPC_READ_PROBE_DISARM_COMMAND,
            )
        except OSError as error:
            logger.warning("Failed to disarm the cpc_read probe: %s", error)

    observation = {
        "injected_value": injected_value,
        "read_error": read_error,
        "status_error": status_error,
        "injections": injections,
        "missed_instances": missed_instances,
    }
    return _classify_cppc_cpc_read_observation(
        filename, path, observation, logger
    )


def _load_cppc_cpc_read_probe(logger):
    """Build, sign, and load the probe from its module directory."""
    try:
        glt_utils.run_command(
            ["make"], "Failed to build CPPC cpc_read probe",
            timeout=CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
        )
    except RuntimeError as error:
        return False, f"Cannot build the cpc_read observer: {error}", False

    if not glt_utils.sign_kernel_module(
            f"{CPPC_CPC_READ_PROBE_MODULE}.ko", logger):
        return False, (
            f"Cannot sign {CPPC_CPC_READ_PROBE_MODULE}.ko for Secure Boot. "
            "See docs/user-guides/secure-boot.md for setup."
        ), False

    try:
        # run_command() intentionally hides the original subprocess exception
        # needed to distinguish definite failure from an uncertain timeout.
        subprocess.run(
            ["insmod", f"{CPPC_CPC_READ_PROBE_MODULE}.ko"],
            check=True,
            capture_output=True,
            env={**os.environ, "LC_ALL": "C"},
            text=True,
            timeout=CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        return False, (
            f"Cannot load the cpc_read observer (insmod exit "
            f"{error.returncode}){suffix}"
        ), False
    except (subprocess.TimeoutExpired, UnicodeDecodeError) as error:
        return False, (
            f"Cannot load the cpc_read observer: {error}. The load outcome is "
            "uncertain; cleanup will remove the probe only if its paths appeared."
        ), True
    except OSError as error:
        return False, f"Cannot execute insmod: {error}", False

    return True, None, True


def _run_cppc_cpc_read_probe(module_dir, paths, logger):
    """Build, load, exercise, and always clean up the cpc_read probe."""
    results = []
    module_loaded = False
    probe_may_be_owned = False
    build_context_entered = False
    failure_stage = "module build context"
    try:
        with glt_utils.chdir_managed(module_dir):
            build_context_entered = True
            failure_stage = "observer"
            module_loaded, load_reason, probe_may_be_owned = (
                _load_cppc_cpc_read_probe(logger)
            )
            if not module_loaded:
                results.append((
                    GLTCore.GLTTestResult.SKIP,
                    None,
                    load_reason,
                ))
            else:
                for filename, path in paths.items():
                    verdict, reason = _run_cppc_cpc_read_sysfs_oracle(
                        filename, path, logger
                    )
                    results.append((verdict, filename, reason))
                    if verdict is GLTCore.GLTTestResult.PASS:
                        logger.info(
                            "%s propagated an injected cpc_read() -EIO from %s",
                            CPPC_CPC_READ_SYSFS_GETTERS[filename],
                            path,
                        )
    except OSError as error:
        results.append((
            (GLTCore.GLTTestResult.WARN if module_loaded
             else GLTCore.GLTTestResult.SKIP),
            None,
            f"CPPC cpc_read {failure_stage} failed: {error}",
        ))
    finally:
        cleanup_errors = _cleanup_cppc_cpc_read_probe(
            module_loaded, logger, probe_may_be_owned
        )
        if build_context_entered:
            try:
                with glt_utils.chdir_managed(module_dir):
                    glt_utils.run_command(
                        ["make", "clean"],
                        "Failed to clean CPPC cpc_read probe build artifacts",
                        timeout=CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
                    )
            except (OSError, RuntimeError) as error:
                cleanup_error = (
                    "Cannot clean the cpc_read observer build artifacts: "
                    f"{error}"
                )
                cleanup_errors.append(cleanup_error)
                logger.error(cleanup_error)
    return results, cleanup_errors


def _aggregate_cppc_cpc_read_results(results, cleanup_errors, logger):
    """Apply cleanup, FAIL, PASS, WARN, then all-SKIP precedence."""
    failure_reasons = [
        reason for verdict, _filename, reason in results
        if verdict is GLTCore.GLTTestResult.FAIL
    ]
    propagated_oracles = [
        filename for verdict, filename, _reason in results
        if verdict is GLTCore.GLTTestResult.PASS
    ]
    warning_reasons = [
        reason for verdict, _filename, reason in results
        if verdict is GLTCore.GLTTestResult.WARN
    ]
    skip_reasons = [
        reason for verdict, _filename, reason in results
        if verdict is GLTCore.GLTTestResult.SKIP
    ]
    for verdict, _filename, reason in results:
        if verdict is GLTCore.GLTTestResult.WARN:
            logger.warning("Inconclusive CPPC sysfs oracle: %s", reason)
        elif verdict is GLTCore.GLTTestResult.SKIP:
            logger.info("Unavailable CPPC sysfs oracle: %s", reason)

    pass_reason = (
        "CPPC sysfs getter(s) " + ", ".join(propagated_oracles)
        + " propagated an injected cpc_read() -EIO"
    )
    if cleanup_errors:
        if failure_reasons:
            prior_reason = "; ".join(failure_reasons)
        elif propagated_oracles:
            prior_reason = pass_reason
        else:
            prior_reason = "; ".join(warning_reasons + skip_reasons)
        cleanup_reason = "; ".join(cleanup_errors)
        if prior_reason:
            cleanup_reason = f"{prior_reason}; additionally, {cleanup_reason}"
        return GLTCore.GLTTestResult.FAIL, cleanup_reason
    if failure_reasons:
        return GLTCore.GLTTestResult.FAIL, "; ".join(failure_reasons)
    if propagated_oracles:
        return GLTCore.GLTTestResult.PASS, pass_reason
    if warning_reasons:
        return GLTCore.GLTTestResult.WARN, "; ".join(warning_reasons)
    return GLTCore.GLTTestResult.SKIP, "; ".join(skip_reasons)


def _check_cppc_cpc_read_kernel_config(kernel_release):
    """Return why the running kernel configuration cannot support the probe."""
    kconfig_path = "/boot/config-" + kernel_release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except (OSError, ValueError) as error:
        return GLTCore.GLTTestResult.SKIP, (
            f"Cannot inspect the running kernel configuration: {error}"
        )
    missing_configs = [
        config for config in CPPC_CPC_READ_REQUIRED_CONFIGS
        if kconfig.get(config) not in ("y", "m")
    ]
    if missing_configs:
        return GLTCore.GLTTestResult.SKIP, (
            "cpc_read error injection requires kernel configs: "
            f"{', '.join(missing_configs)}"
        )
    return None, None


def _check_cppc_cpc_read_observer_prerequisites(kernel_release):
    """Return why the userspace observer cannot run on this kernel."""
    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return GLTCore.GLTTestResult.SKIP, (
                f"The '{command}' command is required for cpc_read error injection."
            )

    kernel_build_dir = os.path.join("/lib/modules", kernel_release, "build")
    if not os.path.isdir(kernel_build_dir):
        return GLTCore.GLTTestResult.SKIP, (
            f"Matching running-kernel headers are unavailable: {kernel_build_dir}"
        )

    try:
        symbol_available = _is_kernel_symbol_available(CPPC_CPC_READ_SYMBOL)
    except OSError as error:
        return GLTCore.GLTTestResult.SKIP, (
            f"Cannot inspect /proc/kallsyms for {CPPC_CPC_READ_SYMBOL}: {error}"
        )
    if not symbol_available:
        return GLTCore.GLTTestResult.SKIP, (
            f"Kernel symbol {CPPC_CPC_READ_SYMBOL} is not probe-visible."
        )

    if _is_cppc_cpc_read_probe_present():
        return GLTCore.GLTTestResult.SKIP, (
            f"{CPPC_CPC_READ_PROBE_MODULE} is already loaded; refusing to "
            "reuse an unknown probe instance. Remove the stale or unknown "
            "module with "
            f"'rmmod {CPPC_CPC_READ_PROBE_MODULE}' before retrying."
        )
    return None, None


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_cppc_cpc_read_error_propagation(_, logger):
    """
    Check for consistent cpc_read() error propagation through CPPC sysfs.

    JIRA: DGX-16335
    PATCH: https://git.kernel.org/torvalds/c/0cc24977224a
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    kernel_release = platform.uname().release
    verdict, reason = _check_cppc_cpc_read_kernel_config(kernel_release)
    if verdict is GLTCore.GLTTestResult.SKIP:
        GLTCore.GLTTestResult.skip(reason)

    module_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CPPC_CPC_READ_PROBE_MODULE,
    )
    if not os.path.isdir(module_dir):
        GLTCore.GLTTestResult.fail(
            f"Packaged cpc_read probe sources are unavailable: {module_dir}. "
            "The GreenLiT installation is incomplete."
        )

    verdict, reason = _check_cppc_cpc_read_observer_prerequisites(
        kernel_release
    )
    if verdict is GLTCore.GLTTestResult.SKIP:
        GLTCore.GLTTestResult.skip(reason)

    cpu_name, paths, baselines, availability_reason = (
        _find_cppc_cpc_read_sysfs_oracles()
    )
    if paths is None:
        GLTCore.GLTTestResult.skip(availability_reason)
    if availability_reason:
        logger.info(
            "CPPC sysfs oracle availability: %s", availability_reason
        )
    for filename, value in baselines.items():
        logger.info("%s %s baseline: %s", cpu_name, filename, value)

    results, cleanup_errors = _run_cppc_cpc_read_probe(
        module_dir, paths, logger
    )
    verdict, reason = _aggregate_cppc_cpc_read_results(
        results, cleanup_errors, logger
    )
    if verdict is GLTCore.GLTTestResult.FAIL:
        GLTCore.GLTTestResult.fail(reason)
    if verdict is GLTCore.GLTTestResult.WARN:
        GLTCore.GLTTestResult.warn(reason)
    if verdict is GLTCore.GLTTestResult.SKIP:
        GLTCore.GLTTestResult.skip(reason)
    logger.info("PASS: %s", reason)
