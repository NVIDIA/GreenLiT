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
Python script to test I2C.
"""
import os
import subprocess
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev


def is_NVDA0301_present(logger): # pylint: disable=invalid-name
    """
    Check if could find i2c device NVDA0301 from stdout of
    'udevadm info -n /dev/ipmi0'
    """
    # Use shell=False to prevent command injection (CWE-78)
    result = subprocess.run(["udevadm", "info", "-n", "/dev/ipmi0"],
                            capture_output=True, shell=False, text=True, check=False)
    if result.returncode != 0:
        logger.error(result.stdout)
        return False

    return "NVDA0301" in result.stdout and "IPI0001" in result.stdout


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_i2c_and_ipmi_device(_, logger):
    """Test if could detect i2c and ipmi device.

    kernel patch:
    https://git.kernel.org/torvalds/c/5bf71889ad9a4d39b7665c105a005c5a33d730ba
    """

    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("System is not using a Grace CPU.")

    if not is_NVDA0301_present(logger):
        GLTCore.GLTTestResult.fail("NVDA0301 or IPI0001 device not found in /dev/ipmi0 device node. " +
                                   "Include kernel commit 5bf71889ad9 to fix it.")

    # Check if could find correct ipmi device state by ipmitool
    if glt_utils.is_cli_available("ipmitool"):
        # Use shell=False to prevent command injection (CWE-78)
        result = subprocess.run(["ipmitool", "mc", "info"],
                                capture_output=True, shell=False, text=True, check=False)
        if "Device Available          : yes" not in result.stdout:
            logger.error(result.stdout)
            GLTCore.GLTTestResult.fail("Invalid IPMI device state detected by ipmitool. "
                                       "Include kernel commit 5bf71889ad9 to fix it")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_ipmi_i2c_pec(_, logger):
    """
    Test if I2C PEC is working.

    JIRA: DGX-8249
    PATCH: https://git.kernel.org/torvalds/c/9f855779a387 v6.4+
    """

    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("System is not using a Grace CPU.")

    if not is_NVDA0301_present(logger):
        GLTCore.GLTTestResult.fail("NVDA0301 or IPI0001 device not found in /dev/ipmi0 device node. "
                                   "Include kernel commit 5bf71889ad9 to fix it.")

    if not glt_utils.is_cli_available("ipmitool"):
        GLTCore.GLTTestResult.fail("Please install 'ipmitool' to perform the test.")

    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.join(module_dir, "kretprobes")
    with glt_utils.chdir_managed(module_dir):
        try:
            # Use shell=False to prevent command injection (CWE-78)
            subprocess.run(["make"], check=True, shell=False)
            if not glt_utils.sign_kernel_module("kretprobe.ko", logger):
                GLTCore.GLTTestResult.fail(
                    "Cannot sign kretprobe.ko for Secure Boot. "
                    "See docs/user-guides/secure-boot.md for setup.")
            subprocess.run(["insmod", "kretprobe.ko"], check=True, shell=False)
            subprocess.run(["ipmitool", "mc", "info"],
                           capture_output=True, shell=False, text=True, check=False)
            result = subprocess.run(["cat", "/proc/greenlit"],
                                    capture_output=True, shell=False, text=True, check=False)
            subprocess.run(["rmmod", "kretprobe"], check=True, shell=False)
            subprocess.run(["make", "clean"], check=True, shell=False)
            if result.returncode != 0 or "0" in result.stdout:
                GLTCore.GLTTestResult.fail("IPMI PEC validation failed. "
                                           "Include kernel commit 9f855779a387 to fix it.")

        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(f"Error executing command: {e}")
