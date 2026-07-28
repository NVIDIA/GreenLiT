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
Python script to test ARM coresight.
"""
from packaging import version
from common import glt_core as GLTCore
from common import perf_utils
from common.nv_device import NvDev

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_perf_coresight(_, logger):
    """Check ARM coresight tracing are available"""
    # FIXME: DGX-10447 perf seems not enable coresight on ubuntu.
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    # Coresight is targeted to RHEL95 for now. DGX-7793 and DGX-7787
    if "Red Hat" in NvDev.get_release_info().get("NAME") and \
            version.parse(NvDev.get_release_info().get("VERSION_ID")) < \
            version.parse("9.5"):
        GLTCore.GLTTestResult.skip("RHEL supports coresight with " +
                                   "version >= 9.5.")

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    result = perf_utils.run_perf_test_candidates(
        ["arm coresight"], perf_cmd=perf_cmd)
    logger.debug(
        "perf Coresight test status=%s test_name=%s reason=%s",
        result.status.value, result.test_name, result.reason)
    if result.status == perf_utils.PerfTestStatus.PASS:
        return

    # Some perf builds expose CoreSight through cs_etm/sysfs but do not
    # install or list the CoreSight shell selftest. In that case, avoid a
    # false hardware failure and accept this lightweight availability check.
    if result.status == perf_utils.PerfTestStatus.NO_MATCH and \
            perf_utils.has_coresight_fallback_evidence():
        logger.info("No perf Coresight selftest found; using cs_etm/sysfs fallback.")
        return

    GLTCore.GLTTestResult.fail(f"perf Coresight test failed: {result.reason}")
