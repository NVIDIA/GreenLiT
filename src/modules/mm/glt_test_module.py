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
Python script to test module allocation.
"""
import os
import platform
import subprocess
from packaging import version
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_module_allocation(_, logger) -> bool:
    """
    Test if there's enough region for module allocation

    DGX-8285
    PATCH: https://git.kernel.org/torvalds/c/3e35d303ab7d # v6.5+
    TODO: Modify kernel command line to disable KASLR
    """

    uname = platform.release().split('.')

    # Kernels shipped by RHEL and SLES don't have the commit, so
    # only allow Ubuntu to run this test for now.

    if NvDev.get_release_info().get("NAME") != "Ubuntu" or \
        version.parse(f"{uname[0]}.{uname[1]}") < version.parse("6.5"):
        GLTCore.GLTTestResult.skip("Test requires Ubuntu with kernel version 6.5 or later.")

    if platform.machine() == "aarch64":
        # shell=True required for pipe and awk; command is hardcoded with no user input
        output = subprocess.run(
                r"""sudo journalctl -b -k | awk \
                    'match($0, ": ([[:digit:]]+).* non-PLT", cap) {print cap[1]}'""",
                capture_output=True, shell=True, text=True, check=True)

        if not output.stdout:
            GLTCore.GLTTestResult.fail("Non-PLT pages for module not found. " +
                                       "Include kernel commit 3e35d303ab7d to fix it.")

        non_plt_page_count = int(output.stdout)

        # shell=True required for pipe and awk; command is hardcoded with no user input
        output = subprocess.run(
                r"""sudo journalctl -b -k | awk \
                    'match($0, ": ([[:digit:]]+).* PLT", cap) {print cap[1]}'""",
                capture_output=True, shell=True, text=True, check=True)

        if not output.stdout:
            GLTCore.GLTTestResult.fail("PLT pages for module not found. " +
                                       "Include kernel commit 3e35d303ab7d to fix it.")

        plt_page_count = int(output.stdout)

        # Non-PLT (i.e. in direct jump range) pages are
        # allocated when KASLR is disabled.
        logger.info("KASLR is " + "disabled" if
                                      non_plt_page_count else "enabled")

        # Use shell=False to prevent command injection (CWE-78)
        output = subprocess.run(["getconf", "PAGE_SIZE"], capture_output=True, shell=False, text=True, check=True)

        page_size = int(output.stdout) if output.stdout else 64 << 10

        # Total size for module allocation shoule be approximately 2G, it can be
        # slightly lower. But at least the region will be bigger than 1G.
        if (non_plt_page_count + plt_page_count) * page_size < 1 << 30:
            GLTCore.GLTTestResult.fail("Module allocation region is less than 1GB. " +
                                       "Include kernel commit 3e35d303ab7d to fix it.")

    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.join(module_dir, "bingo")
    with glt_utils.chdir_managed(module_dir):
        try:
            # Use shell=False to prevent command injection (CWE-78)
            subprocess.run(["make"], check=True, shell=False)
            if not glt_utils.sign_kernel_module("bingo.ko", logger):
                GLTCore.GLTTestResult.fail(
                    "Cannot sign bingo.ko for Secure Boot. "
                    "See docs/user-guides/secure-boot.md for setup.")
            subprocess.run(["sudo", "insmod", "bingo.ko"], check=True, shell=False)
            subprocess.run(["sudo", "rmmod", "bingo"], check=True, shell=False)
            subprocess.run(["make", "clean"], check=True, shell=False)
        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(f"Error executing command: {e}")
