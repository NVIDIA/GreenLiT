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
Python script to check NVIDIA GIC errata.
"""

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_gic_errata(_, logger):
    """
    Perform the GIC errata check.

    JIRA: DGX-8064
    PATCH: https://git.kernel.org/torvalds/linux/c/35727af2b15d
    """
    # Skip if not NVIDIA Grace CPU
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("SKIP: This is not NVIDIA Grace CPU.")

    # Skip if number of processor sockets is less than or equal 2
    socket_count = NvDev.get_cpu_socket_count()
    if socket_count == -1:
        GLTCore.GLTTestResult.skip("SKIP: Unable to determine CPU socket count.")
    if socket_count <= 2:
        GLTCore.GLTTestResult.skip(f"SKIP: CPU socket num: {socket_count} less than or equal 2.")

    # Search for the NVIDIA errata T241-FABRIC-4 workaround message
    target_msg = "GIC: enabling workaround for GICv3: NVIDIA erratum T241-FABRIC-4"

    # Use the check_kernel_log helper function to look for the message
    if glt_utils.check_kernel_log(target_msg):
        logger.info(f"Found required GIC workaround message: {target_msg}")
    else:
        patch_url = "https://git.kernel.org/torvalds/linux/c/35727af2b15d"
        error_msg = f"ERROR: The GIC errata workaround is not enabled. Please apply kernel patch: {patch_url}"
        GLTCore.GLTTestResult.fail(error_msg)
