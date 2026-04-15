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
import subprocess
from packaging import version
from common import glt_core as GLTCore
from common import glt_utils
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

    if not glt_utils.is_cli_available("perf"):
        GLTCore.GLTTestResult.fail("Please install 'perf' to perform the test.")

    try:
        cmd_output = subprocess.run(
            "perf test -vv 'arm coresight'",
            shell=True, executable='/bin/bash', check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug(cmd_output)
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail("Error executing command: %s", e)

    if not cmd_output.stderr.strip().endswith(": Ok"):
        GLTCore.GLTTestResult.fail(f"perf test 'arm coresight' didn't get PASS with: {cmd_output}")
