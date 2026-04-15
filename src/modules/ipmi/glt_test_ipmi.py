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
Python script to test ipmi.
"""

import re
import subprocess
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
def test_bmc_manufacturer_name(_, __):
    """Check BMC manufacturer name"""

    # Check if platform supports BMC
    nvdev = NvDev()
    if not nvdev.plat_support_bmc():
        GLTCore.GLTTestResult.skip("Platform does not support BMC")

    if not glt_utils.is_cli_available("bmc-info"):
        GLTCore.GLTTestResult.fail("Please install 'freeipmi' to perform the test")

    try:
        cmd_output = subprocess.run("bmc-info --get-device-id",
                                    shell=True, executable="/bin/bash", text=True, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"bmc-info execute fail: {e}")

    match = re.search(r"Manufacturer ID\s*:\s*(.*\s\(\d+\))", cmd_output.stdout)

    if not match:
        GLTCore.GLTTestResult.fail("Manufacturer ID doesn't match in bmc-info output. "
                                   "Please fix the BMC firmware")
