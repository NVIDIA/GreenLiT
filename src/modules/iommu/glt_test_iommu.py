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
Python script to test IOMMU.
"""
import glob
import os
import platform
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_iommu_id_mapping(_, __):
    """Test if could detect iommu id mapping logs in dmesg."""

    # This test is specific to ARM64 SMMU
    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("Not ARM64, skip the test")

    id_mapping_error_msg = "ID mapping index overflows valid mappings"

    if glt_utils.check_kernel_log(id_mapping_error_msg):
        GLTCore.GLTTestResult.fail(f'Find iommu error message {id_mapping_error_msg}. ' +
                                   'Include kernel commit 05da178ce0aa to fix it.')


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cmdqv_alignment(_, __):
    """Test Tegra241 CMDQV queue alignment fix.

    JIRA: DGX-10648
    Patch: https://lore.kernel.org/all/20241111030226.1940737-1-nicolinc@nvidia.com/
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("Not a Grace CPU, CMDQV is Grace-specific")

    page_size = os.sysconf("SC_PAGE_SIZE")
    if page_size != 4096:
        GLTCore.GLTTestResult.skip(
            f"Page size is {page_size}, not 4KB."
            " The alignment bug only triggered on 4KB page kernels.")

    if not glob.glob("/sys/devices/platform/NVDA200C:*"):
        GLTCore.GLTTestResult.skip("No NVDA200C CMDQV devices found in sysfs")

    if glt_utils.check_kernel_log("arm_smmu_init_one_queue"):
        GLTCore.GLTTestResult.fail(
            "WARNING at arm_smmu_init_one_queue detected in kernel log."
            " The Tegra241 CMDQV max_n_shift alignment fix is missing."
            " Consider backporting commit a379971 to fix it.")
