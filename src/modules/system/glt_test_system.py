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

"""
System state test cases.
"""

import subprocess
from common import glt_core as GLTCore
from common import glt_utils


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_system_state_is_running(_, __):
    """
    Test if the overall system state is running.
    """
    if not glt_utils.is_cli_available("systemctl"):
        GLTCore.GLTTestResult.skip("systemctl is not available on this system.")

    if not glt_utils.is_systemd_running():
        GLTCore.GLTTestResult.skip("System is not booted with systemd.")

    try:
        result = subprocess.run(
            ["systemctl", "is-system-running"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        state = result.stdout.strip()

        if state != "running":
            failed_check = subprocess.run(
                ["systemctl", "--failed"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )
            error_msg = f"System state is '{state}', expected 'running'.\nFailed units:\n{failed_check.stdout}"
            GLTCore.GLTTestResult.fail(error_msg)
    except subprocess.TimeoutExpired as e:
        GLTCore.GLTTestResult.fail(f"Command '{e.cmd}' timed out after {e.timeout} seconds")
