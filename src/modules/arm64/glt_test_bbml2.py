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

"""Validate the running kernel BBML2 noabort capability on Vera."""

import platform
import re
import zlib
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.cpu_topology import parse_cpu_mask
from common.nv_device import NvDev

BBML2_MODULE = "bbml2_capability_probe"
BBML2_MODULE_PATH = Path(__file__).resolve().parent / "bbml2_capability"
BBML2_MODULE_STATE_PATH = Path("/sys/module") / BBML2_MODULE
CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
BBML2_LOG_RE = re.compile(
    r"^glt_bbml2_capability: compiled=(?P<compiled>[01]) enabled=(?P<enabled>[01])$",
    re.MULTILINE,
)


def _read_kernel_config_text():
    try:
        with open("/proc/config.gz", "rb") as config_file:
            return zlib.decompress(
                config_file.read(), 16 + zlib.MAX_WBITS
            ).decode()
    except OSError:
        with open(f"/boot/config-{platform.release()}", encoding="utf-8") as config_file:
            return config_file.read()


def _non_olympus_present_cpus(cpu_root=CPU_SYSFS_ROOT):
    cpu_root = Path(cpu_root)
    present_path = cpu_root / "present"
    present_cpus = parse_cpu_mask(present_path.read_text(encoding="utf-8"))
    if not present_cpus:
        raise ValueError(f"No present CPUs found in {present_path}")

    non_olympus = []
    for cpu_id in sorted(present_cpus):
        midr_path = cpu_root / f"cpu{cpu_id}" / "regs" / "identification" / "midr_el1"
        midr = int(midr_path.read_text(encoding="utf-8").strip(), 0)
        implementer = (midr >> 24) & 0xff
        part = (midr >> 4) & 0xfff
        if implementer != 0x4e or part != 0x010:
            non_olympus.append(cpu_id)
    return non_olympus


def _parse_probe_log(kernel_log):
    result_lines = [
        line for line in kernel_log.splitlines()
        if line.startswith("glt_bbml2_capability:")
    ]
    if len(result_lines) != 1:
        raise ValueError(
            f"Expected one BBML2 capability result, found {len(result_lines)}"
        )
    match = BBML2_LOG_RE.fullmatch(result_lines[0])
    if match is None:
        raise ValueError("BBML2 capability result is malformed")
    compiled = int(match.group("compiled"))
    enabled = int(match.group("enabled"))
    if not compiled and enabled:
        raise ValueError("BBML2 capability cannot be enabled when support was not compiled")
    return compiled, enabled


def _evaluate_probe_result(compiled, enabled):
    if not compiled:
        return (
            "skip",
            "Running kernel headers do not provide ARM64_HAS_BBML2_NOABORT; "
            "the early capability patch series is required",
        )
    if not enabled:
        return (
            "fail",
            "All present CPUs are Olympus but the running kernel has not "
            "enabled BBML2 noabort; check the early all-CPU capability series",
        )
    return "pass", "BBML2 noabort capability is enabled on all-Olympus Vera"


def _run_bbml2_probe(logger, cpu_root=CPU_SYSFS_ROOT, module_dir=BBML2_MODULE_PATH,
                     module_state_path=BBML2_MODULE_STATE_PATH,
                     headers_root=Path("/lib/modules"), kernel_release=None):
    if kernel_release is None:
        kernel_release = platform.release()
    module_dir = Path(module_dir)
    module_state_path = Path(module_state_path)

    try:
        config_text = _read_kernel_config_text()
    except OSError as err:
        return "skip", f"Cannot verify module cleanup support in kernel config: {err}"
    missing_configs = [
        name for name in ("CONFIG_MODULES=y", "CONFIG_MODULE_UNLOAD=y")
        if name not in config_text
    ]
    if missing_configs:
        return (
            "skip",
            "BBML2 probe requires loadable and unloadable modules; missing "
            + ", ".join(missing_configs),
        )

    try:
        non_olympus = _non_olympus_present_cpus(cpu_root)
    except (OSError, ValueError) as err:
        return "skip", f"Cannot inspect present CPU MIDRs: {err}"
    if non_olympus:
        return (
            "skip",
            f"Mixed present-CPU MIDR topology includes non-Olympus CPUs {non_olympus}; "
            "an all-Olympus early capability result is not expected",
        )

    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return "skip", f"{command} is required to run the BBML2 capability probe"
    headers = Path(headers_root) / kernel_release / "build"
    if not headers.is_dir():
        return "skip", f"Matching kernel headers are unavailable: {headers}"
    if not module_dir.is_dir():
        return "skip", f"BBML2 probe sources are missing: {module_dir}"
    if module_state_path.exists():
        return "skip", f"{BBML2_MODULE} is already loaded; refusing an unknown instance"

    loaded_by_test = False
    outcome = "pass"
    reason = ""
    cleanup_errors = []

    with glt_utils.chdir_managed(module_dir):
        try:
            try:
                glt_utils.run_command(["make"], "Cannot build BBML2 capability probe", timeout=120)
            except RuntimeError as err:
                outcome = "skip"
                reason = str(err)

            if outcome == "pass" and not glt_utils.sign_kernel_module(
                f"{BBML2_MODULE}.ko", logger
            ):
                outcome = "fail"
                reason = f"Cannot sign {BBML2_MODULE}.ko for Secure Boot"

            checkpoint = None
            if outcome == "pass":
                try:
                    checkpoint = glt_utils.capture_kernel_log_checkpoint()
                except RuntimeError as err:
                    outcome = "skip"
                    reason = str(err)

            if outcome == "pass":
                try:
                    glt_utils.run_command(
                        ["insmod", f"{BBML2_MODULE}.ko"],
                        "Cannot load BBML2 capability probe",
                        timeout=30,
                    )
                    loaded_by_test = True
                except RuntimeError as err:
                    if module_state_path.exists():
                        outcome = "fail"
                        reason = (
                            f"{err}; {BBML2_MODULE} exists after a failed insmod, "
                            "but ownership is unknown. Refusing to unload it; "
                            "manual cleanup is required"
                        )
                    else:
                        outcome = "skip"
                        reason = str(err)

            if outcome == "pass":
                try:
                    kernel_log = glt_utils.get_kernel_log(checkpoint)
                    compiled, enabled = _parse_probe_log(kernel_log)
                    outcome, reason = _evaluate_probe_result(compiled, enabled)
                except (RuntimeError, ValueError) as err:
                    outcome = "fail"
                    reason = f"Invalid BBML2 capability result: {err}"
        finally:
            if loaded_by_test:
                try:
                    glt_utils.run_command(
                        ["rmmod", BBML2_MODULE],
                        "Failed to unload BBML2 capability probe",
                        timeout=30,
                    )
                except RuntimeError as err:
                    cleanup_errors.append(str(err))
            try:
                glt_utils.run_command(
                    ["make", "clean"],
                    "Failed to clean BBML2 capability probe",
                    timeout=120,
                )
            except RuntimeError as err:
                cleanup_errors.append(str(err))

    if cleanup_errors:
        cleanup_reason = "; ".join(cleanup_errors)
        if outcome == "fail":
            reason = f"{reason}; cleanup also failed: {cleanup_reason}"
        else:
            outcome = "fail"
            reason = f"BBML2 probe cleanup failed: {cleanup_reason}"
    return outcome, reason


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_bbml2_capability(_, logger):
    """Check runtime BBML2 noabort capability (upstream 5aa4b625762e and cc80537caaa7)."""
    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("BBML2 capability validation is arm64-specific")
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("BBML2 capability validation is Vera-specific")

    outcome, reason = _run_bbml2_probe(logger)
    if outcome == "fail":
        GLTCore.GLTTestResult.fail(reason)
    if outcome == "skip":
        GLTCore.GLTTestResult.skip(reason)
    logger.info("%s", reason)
