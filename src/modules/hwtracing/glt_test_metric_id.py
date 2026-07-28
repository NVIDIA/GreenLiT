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
Python script to test perf metric ID support.
"""
# pylint: disable=duplicate-code
import shlex
import subprocess
from common import glt_core as GLTCore
from common import perf_utils

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_perf_metric_id(_, logger):
    """
    JIRA: DGX-11476
    Check whether perf supports metric IDs.
    """

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    try:
        cmd_output = subprocess.run(
            f"sudo {shlex.quote(perf_cmd)} stat -v -a -M ipc ls /",
            shell=True, executable='/bin/bash', check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug(f"stdout: {cmd_output.stdout}\nstderr: {cmd_output.stderr}")
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"Error executing command: {e}")

    # Both stderr and stdout might contain the output we need
    full_output = cmd_output.stdout + cmd_output.stderr
    expected_output = '{INST_RETIRED/metric-id=INST_RETIRED/,CPU_CYCLES/metric-id=CPU_CYCLES/}:W'

    if expected_output not in full_output:
        GLTCore.GLTTestResult.fail("Metric IDs not found in perf output. " +
                                   "Include kernel commit 4473949074c3 and a24d9d9dc096 to fix it.")
