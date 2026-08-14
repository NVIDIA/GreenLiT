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
Python script to test ast gpu driver.
"""
import subprocess
import time
from common import glt_core as GLTCore
from common import glt_utils

def is_ast_device_present(vdc):
    """
    See if ASPEED VGA device is present
    """
    p = subprocess.run(f"lspci -d {vdc} | wc -c", shell=True, check=False)
    return p.returncode == 0 and p.stdout != '0'


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_ast_gpu_driver(sd, _):
    """
    Try to reload ast gpu driver and detect whether has CPU lockup.

    JIRA: DGX-8261
    PATCH: https://git.kernel.org/torvalds/c/4327a6137ed4 # v6.4+
    """

    for tests in sd.progress:
        if test_ast_gpu_driver.__name__ not in tests:
            continue
        if "REISUB" in tests[test_ast_gpu_driver.__name__]:
            GLTCore.GLTTestResult.fail("Removing ast module caused CPU lockup. " +
                                       "Include kernel commit 4327a6137ed4 to fix it.")

    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.fail("lspci is unavailable on the system.")

    ast_device_vdcs = ['1a03:2000:300', '1a03:2001:300']
    if not any(is_ast_device_present(vdc) for vdc in ast_device_vdcs):
        GLTCore.GLTTestResult.skip("ASPEED VGA device isn't present")

    with subprocess.Popen(['sudo', 'modprobe', '-r', 'ast']) as process:
        if process.poll() is None:
            time.sleep(5)
            if process.poll() is None:
                ok, reason = glt_utils.reboot_with_reisub(
                    test_ast_gpu_driver.__name__, sd
                )
                if not ok:
                    GLTCore.GLTTestResult.fail(
                        "AST module operation timed out. Include kernel commit "
                        "4327a6137ed4 to fix it. Failed to trigger the recovery "
                        f"reboot: {reason}"
                    )

    with subprocess.Popen(['sudo', 'modprobe', 'ast']) as process:
        if process.poll() is None:
            time.sleep(5)
            if process.poll() is None:
                ok, reason = glt_utils.reboot_with_reisub(
                    test_ast_gpu_driver.__name__, sd
                )
                if not ok:
                    GLTCore.GLTTestResult.fail(
                        "AST module operation timed out. Include kernel commit "
                        "4327a6137ed4 to fix it. Failed to trigger the recovery "
                        f"reboot: {reason}"
                    )
