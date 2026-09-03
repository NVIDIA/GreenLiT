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
Python script to verify Arm CCA kernel readiness.
"""

import os
import platform

from common import glt_core as GLTCore
from common.nv_device import NvDev
from scripts import check_kernel_config


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_cca_kernel_support(_, logger):
    """Verify Vera host kernel readiness for Arm CCA Realm execution."""
    config_kvm = "CONFIG_KVM"
    dev_kvm = "/dev/kvm"
    kvm_cap_arm_rmi_name = "KVM_CAP_ARM_RMI"
    sysfs_kvm_cap_arm_rmi = "/sys/module/kvm/parameters/kvm_cap_arm_rmi"

    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    page_size = os.sysconf("SC_PAGE_SIZE")
    if page_size != 4096:
        GLTCore.GLTTestResult.fail(
            f"Arm CCA requires a 4KB host kernel page size; found {page_size} bytes."
        )

    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {kconfig_path}; cannot verify {config_kvm}."
        )

    if kconfig.get(config_kvm) != "y":
        GLTCore.GLTTestResult.fail(
            f"{config_kvm} must be enabled as built-in (=y); found {kconfig.get(config_kvm)!r}."
        )

    if not os.path.exists(dev_kvm):
        GLTCore.GLTTestResult.fail(f"{dev_kvm} is missing.")

    try:
        with open(sysfs_kvm_cap_arm_rmi, encoding="ascii") as f:
            cap_num = int(f.read().strip())
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"{sysfs_kvm_cap_arm_rmi} is missing; kernel may lack {kvm_cap_arm_rmi_name} support."
        )
    except (OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to read {sysfs_kvm_cap_arm_rmi}: {error}."
        )
    else:
        if cap_num == 0:
            GLTCore.GLTTestResult.fail(
                f"{sysfs_kvm_cap_arm_rmi} reports zero; {kvm_cap_arm_rmi_name} is not available. "
                "Ensure the required CCA/Realm Guest kernel patches are backported; see "
                "https://docs.nvidia.com/dccpu/patches-config-guide/platform-software-patches-config-addendum_vera.html"
            )
        logger.info("%s = %d", sysfs_kvm_cap_arm_rmi, cap_num)
