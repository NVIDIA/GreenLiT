# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test PMU.
"""
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from packaging import version

from common import glt_core as GLTCore
from common import glt_utils
from common import perf_utils
from common.nv_device import NvDev
from scripts import check_kernel_config

NVIDIA_PMU_SYSFS_PATH = "/sys/bus/event_source/devices"

KNOWN_PATCH_SERIES = {
    "t410_prep": "T410 preparatory arm_cspmu patches "
    "(reset_ev_filter, PMPIDR, revision-id matching, PMEVFILTR2)",
    "t410_pmu": "Tegra410 PMU support patches "
    "(UCF, PCIE, PCIE-TGT, and perf vendor events)",
}

VERA_REQUIRED_CS_PMU_FAMILIES = (
    "nvidia_ucf_pmu",
    "nvidia_pcie_pmu",
    "nvidia_pcie_tgt_pmu",
)

GRACE_PMU_FAMILIES = (
    "nvidia_cnvlink_pmu",
    "nvidia_nvlink_c2c0_pmu",
    "nvidia_nvlink_c2c1_pmu",
    "nvidia_pcie_pmu",
    "nvidia_scf_pmu",
)

GENERIC_FALLBACK_PMU_FAMILIES = (
    "nvidia_uncore_pmu",
)


def _normalize_nvidia_pmu_name(pmu_name):
    """Return the PMU family name without instance-specific suffixes."""
    normalized_name = pmu_name.lower()
    if not normalized_name.startswith("nvidia_"):
        return pmu_name
    match = re.match(r"(nvidia_.*?_pmu)(?:_.*)?$", normalized_name)
    if match:
        return match.group(1)
    return normalized_name


def _get_pmu_families(pmu_names):
    """Return normalized NVIDIA PMU families from device or perf unit names."""
    return {
        _normalize_nvidia_pmu_name(pmu_name)
        for pmu_name in pmu_names
        if isinstance(pmu_name, str)
        and pmu_name.lower().startswith("nvidia_")
        and re.search(r"_pmu(?:_|$)", pmu_name.lower())
    }


def _format_patch_guidance():
    """Format concise guidance for missing T410 PMU runtime support."""
    patch_lines = [
        f"{name}: {description}"
        for name, description in KNOWN_PATCH_SERIES.items()
    ]
    return (
        "Consider checking whether the kernel is missing the T410 patch "
        "groups from data/kernel-patches/vera.txt with check-kernel-patches: "
        + "; ".join(patch_lines)
    )


def _parse_nvidia_perf_units(perf_list_json):
    """Extract normalized NVIDIA PMU families from perf list JSON entries."""
    return _get_pmu_families(
        item.get("Unit", "")
        for item in perf_list_json
        if isinstance(item, dict)
    )


def _get_nvidia_perf_list(logger):
    """Get NVIDIA-specific perf event unit families."""
    try:
        cmd_output = subprocess.run(
            ["perf", "list", "-j"],
            check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, OSError) as e:
        logger.error("Error executing command: %s", e)
        return set()

    try:
        perf_list_json = json.loads(cmd_output.stdout)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse perf list JSON output: %s", e)
        return set()

    return _parse_nvidia_perf_units(perf_list_json)


def _get_nvidia_pmu_devices(pmu_path=NVIDIA_PMU_SYSFS_PATH):
    """Get sorted NVIDIA PMU device names from sysfs."""
    path = Path(pmu_path)
    if not path.is_dir():
        return []

    try:
        return sorted(
            device.name
            for device in path.iterdir()
            if device.name.lower().startswith("nvidia_")
            and re.search(r"_pmu(?:_|$)", device.name.lower())
        )
    except OSError:
        return []


def _is_fallback_pmu(pmu_name):
    """Return True for generic NVIDIA CoreSight PMU fallback devices."""
    return _normalize_nvidia_pmu_name(pmu_name) in GENERIC_FALLBACK_PMU_FAMILIES


def _classify_pmu_devices(pmu_devices):
    """Classify PMU devices into named and generic fallback devices."""
    named = []
    fallback = []
    for pmu_device in pmu_devices:
        if _is_fallback_pmu(pmu_device):
            fallback.append(pmu_device)
        else:
            named.append(pmu_device)
    return named, fallback


def _missing_pmu_families(observed_families, required_families):
    """Return required PMU families absent from observed families."""
    return sorted(set(required_families) - set(observed_families))


def _validate_common_nvidia_pmus(logger, nvevent_from_dut, pmu_devices,
                                 guidance=""):
    """Validate NVIDIA PMU visibility that applies to all NVIDIA CPUs."""
    logger.debug("Supported NVIDIA PMU families from perf: %s",
                 sorted(nvevent_from_dut))
    if not nvevent_from_dut:
        return f"No NVIDIA perf events found. {guidance}".strip()

    logger.debug("Registered NVIDIA PMU devices: %s", pmu_devices)
    if not pmu_devices:
        return f"No NVIDIA PMU devices found in sysfs. {guidance}".strip()

    named_pmus, fallback_pmus = _classify_pmu_devices(pmu_devices)
    logger.debug("Named NVIDIA PMU devices: %s", named_pmus)
    logger.debug("Fallback NVIDIA PMU devices: %s", fallback_pmus)
    if fallback_pmus and not named_pmus:
        return (
            f"Only generic fallback NVIDIA PMU devices found: {fallback_pmus}. "
            f"{guidance}"
        ).strip()

    logger.debug("Observed NVIDIA PMU families from sysfs: %s",
                 sorted(_get_pmu_families(pmu_devices)))
    return None


def validate_grace_pmus(logger, nvevent_from_dut, pmu_devices):
    """Validate Grace-specific NVIDIA PMU expectations."""
    failure_reason = _validate_common_nvidia_pmus(
        logger, nvevent_from_dut, pmu_devices)
    if failure_reason:
        return failure_reason

    pmu_families = _get_pmu_families(pmu_devices)
    expected_families = sorted(GRACE_PMU_FAMILIES)
    detected_perf_families = sorted(nvevent_from_dut)
    detected_sysfs_families = sorted(pmu_families)
    missing_perf_families = _missing_pmu_families(
        nvevent_from_dut, GRACE_PMU_FAMILIES)
    missing_sysfs_families = _missing_pmu_families(
        pmu_families, GRACE_PMU_FAMILIES)
    if missing_perf_families or missing_sysfs_families:
        return (
            "Missing expected Grace NVIDIA PMU families. "
            f"expected: {expected_families}; "
            f"detected perf: {detected_perf_families}; "
            f"detected sysfs: {detected_sysfs_families}; "
            f"missing perf: {missing_perf_families}; "
            f"missing sysfs: {missing_sysfs_families}"
        )

    allowed_families = set(GRACE_PMU_FAMILIES) | set(
        GENERIC_FALLBACK_PMU_FAMILIES)
    extra_perf_families = sorted(set(nvevent_from_dut) - allowed_families)
    extra_sysfs_families = sorted(pmu_families - allowed_families)
    if extra_perf_families or extra_sysfs_families:
        return (
            "Found unexpected Grace NVIDIA PMU families. "
            f"expected: {expected_families}; "
            f"detected perf: {detected_perf_families}; "
            f"detected sysfs: {detected_sysfs_families}; "
            f"extra perf: {extra_perf_families}; "
            f"extra sysfs: {extra_sysfs_families}"
        )
    return None


def validate_vera_pmus(logger, nvevent_from_dut, pmu_devices):
    """Validate Vera-specific NVIDIA PMU expectations."""
    patch_guidance = _format_patch_guidance()
    failure_reason = _validate_common_nvidia_pmus(
        logger, nvevent_from_dut, pmu_devices, patch_guidance)
    if failure_reason:
        return failure_reason

    missing_sysfs_families = _missing_pmu_families(
        _get_pmu_families(pmu_devices), VERA_REQUIRED_CS_PMU_FAMILIES)
    if missing_sysfs_families:
        return (
            "Missing required Vera T410 CoreSight PMU families from sysfs: "
            f"{', '.join(missing_sysfs_families)}. {patch_guidance}"
        )

    missing_perf_families = _missing_pmu_families(
        nvevent_from_dut, VERA_REQUIRED_CS_PMU_FAMILIES)
    if missing_perf_families:
        return (
            "Missing required Vera T410 CoreSight PMU families from perf: "
            f"{', '.join(missing_perf_families)}. {patch_guidance}"
        )
    return None


@GLTCore.GLTRequires.must_pass("hwtracing", "test_perf_coresight")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_perf_events(_, logger):
    """Check whether NVIDIA perf PMU events match platform expectations."""
    # FIXME: DGX-10447 perf seems not enable coresight on ubuntu.
    nvdev = NvDev()
    is_grace = nvdev.is_cpu_type("grace")
    is_vera = nvdev.is_cpu_type("vera")
    if not is_grace and not is_vera:
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    # Coresight is targeted to RHEL95 for now. DGX-7793 and DGX-7787
    if "Red Hat" in NvDev.get_release_info().get("NAME") and \
            version.parse(NvDev.get_release_info().get("VERSION_ID")) < \
            version.parse("9.5"):
        GLTCore.GLTTestResult.skip("RHEL supports coresight if version >= 9.5.")

    nvevent_from_dut = _get_nvidia_perf_list(logger)
    pmu_devices = _get_nvidia_pmu_devices()
    failure_reason = None
    if is_grace:
        failure_reason = validate_grace_pmus(
            logger, nvevent_from_dut, pmu_devices)
    elif is_vera:
        failure_reason = validate_vera_pmus(
            logger, nvevent_from_dut, pmu_devices)

    if failure_reason:
        GLTCore.GLTTestResult.fail(failure_reason)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_all_pmu_events(_, logger):
    """Perform perf all pmu events check"""
    # FIXME: DGX-10165 pylint: disable=fixme
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    if not glt_utils.is_cli_available("perf"):
        GLTCore.GLTTestResult.fail("Please install 'perf' to peform the test.")

    result = perf_utils.run_perf_test_candidates(
        ["perf all PMU test", "Parsing of all PMU events from sysfs"]
    )
    logger.debug(
        "perf PMU test status=%s test_name=%s reason=%s",
        result.status.value, result.test_name, result.reason)
    if result.status != perf_utils.PerfTestStatus.PASS:
        GLTCore.GLTTestResult.fail(f"perf PMU test failed: {result.reason}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
def test_perf_cspmu_rejects_foreign_events(_, logger):
    """Verify perf rejects CSPMU events from other PMUs (related to commit 15c7ef73)."""
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    if not glt_utils.is_cli_available("perf"):
        GLTCore.GLTTestResult.fail("Please install 'perf' to perform the test.")

    try:
        # Test if the kernel properly rejects PMU events meant for other PMUs (commit 15c7ef73).
        cmd_output = subprocess.run(
            "perf record -q -o /dev/null -e cycles:Pu -a -- sleep 0.1",
            shell=True, executable="/bin/bash", check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug("perf record completed: %s", cmd_output)
    except subprocess.CalledProcessError as e:
        stderr_line = " ".join((e.stderr or "").splitlines())
        GLTCore.GLTTestResult.fail(
            f"perf command failed with return code {e.returncode}: {stderr_line}. "
            "This may indicate that commit 15c7ef73 is missing; consider backporting it."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_spe_pmu(sd, logger):
    """Verify ARM SPE PMU support."""
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
        if kconfig.get("CONFIG_ARM_SPE_PMU") not in ("y", "m"):
            GLTCore.GLTTestResult.fail("CONFIG_ARM_SPE_PMU is not enabled in kernel config")
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {kconfig_path}, cannot verify CONFIG_ARM_SPE_PMU"
        )

    if not os.path.exists("/sys/bus/event_source/devices/arm_spe_0/"):
        GLTCore.GLTTestResult.fail(
            "/sys/bus/event_source/devices/arm_spe_0/ does not exist. "
            "SPE requires OS and firmware support, please check both."
        )

    if not glt_utils.is_cli_available("perf"):
        GLTCore.GLTTestResult.fail("Please install 'perf' to perform the test.")

    perf_data_path = os.path.join(sd.log_dir, "test_arm_spe_pmu.perf.data")

    try:
        subprocess.run(
            ["perf", "record", "-e", "arm_spe//", "-o", perf_data_path, "--",
             "dd", "if=/dev/zero", "of=/dev/null", "bs=1M", "count=500"],
            check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr_out = e.stderr.decode(errors="replace") if e.stderr else ""
        GLTCore.GLTTestResult.fail(f"perf record failed with arm_spe event: {e} - {stderr_out}")

    try:
        cmd_output = subprocess.run(
            ["perf", "report", "-i", perf_data_path, "--stdio"],
            check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logger.debug(cmd_output.stdout)
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"perf report failed with {perf_data_path}: {e} - {e.stdout}")

    # Since we are using `perf record -e arm_spe//` instead of `perf mem`,
    # the output format changes. We verify that the SPE event was actually sampled.
    expected_strings = [
        "Samples:", "l1d-miss", "l1d-access", "llc-miss", "llc-access",
        "tlb-access", "memory", "branch"
    ]
    missing = [s for s in expected_strings if s not in cmd_output.stdout]
    if missing:
        GLTCore.GLTTestResult.fail(
            f"Missing expected strings {missing} in perf report with {perf_data_path}"
        )
