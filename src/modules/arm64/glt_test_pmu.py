# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from packaging import version

from common import glt_core as GLTCore
from common import glt_utils
from common import perf_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


def _get_nvidia_perf_list(logger):
    """Get nvidia specific perf events."""
    try:
        cmd_output = subprocess.run(
            "perf list -j",
            shell=True, executable='/bin/bash', check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error("Error executing command: %s", e)
        return []

    perf_list_json = json.loads(cmd_output.stdout)
    events = set(
        re.sub(r'\d$', '', item["Unit"])
        for item in perf_list_json
        if "Unit" in item and item["Unit"].startswith("nvidia")
    )
    return {s.strip() for s in events}


@GLTCore.GLTRequires.must_pass("hwtracing", "test_perf_coresight")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_perf_events(_, logger):
    """Check whether having nvidia expects perf events."""
    # FIXME: DGX-10447 perf seems not enable coresight on ubuntu.
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    # Coresight is targeted to RHEL95 for now. DGX-7793 and DGX-7787
    if "Red Hat" in NvDev.get_release_info().get("NAME") and \
            version.parse(NvDev.get_release_info().get("VERSION_ID")) < \
            version.parse("9.5"):
        GLTCore.GLTTestResult.skip("RHEL supports coresight if version >= 9.5.")

    nvevent_list = [
        "nvidia_cnvlink_pmu_",
        "nvidia_nvlink_c2c0_pmu_",
        "nvidia_nvlink_c2c1_pmu_",
        "nvidia_pcie_pmu_",
        "nvidia_scf_pmu_",
    ]

    nvevent_from_dut = _get_nvidia_perf_list(logger)
    logger.debug("Supported nvevents: %s", nvevent_from_dut)
    if not nvevent_from_dut:
        GLTCore.GLTTestResult.fail("Not found nvidia perf events.")
    if not set(nvevent_from_dut).issubset(nvevent_list):
        GLTCore.GLTTestResult.fail(f"Not found expected nvidia events {nvevent_list}")


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
