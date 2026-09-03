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
Python script to test framebuffer.
"""

import re
import subprocess
from common import glt_core as GLTCore
from common import glt_utils


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_fb_efi_addresses(_, __):
    """Check the AST framebuffer addresses reported by EFI"""
    try:
        cmd_output = subprocess.run("lspci -d 1a03:2000 | awk '{print $1}'",
                                    shell=True, executable="/bin/bash", check=True, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, PermissionError, FileNotFoundError, OSError) as e:
        GLTCore.GLTTestResult.fail(f"Command excuting failed: {e}")

    bus = list(cmd_output.stdout.splitlines())
    if not bus:
        GLTCore.GLTTestResult.skip("No AST VGA device found.")

    try:
        kernel_log = glt_utils.get_kernel_log()
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(f"Failed to collect kernel logs: {error}")

    events = [
        line for line in kernel_log.splitlines() if "framebuffer at" in line
    ]
    if not events:
        GLTCore.GLTTestResult.skip("Not found framebuffer address.")

    address = None
    for s in events:
        match = re.search(r"0x([0-9a-fA-F]+),", s.strip())
        if match:
            address = match.group(1)
            break
    if address is None:
        GLTCore.GLTTestResult.skip(f"Not found valid framebuffer address from {events}")

    found = False
    for b in bus:
        pattern = rf"{address}.*{b}"

        with open("/proc/iomem", "r", encoding="utf-8") as f:
            for line in f:
                if re.search(pattern, line):
                    found = True

    if not found:
        GLTCore.GLTTestResult.fail(f"{address} can not be found on {bus}.\n"
                                   "Please update SBIOS for newer EFI GOP driver.")
