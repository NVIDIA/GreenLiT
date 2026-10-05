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

"""Verify Arm architected timer kernel configuration on Vera."""

import platform

from common import glt_core as GLTCore
from common.nv_device import NvDev
from scripts import check_kernel_config


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_arch_timer_evtstream_kernel_config(_, logger):
    """
    Verify that the Vera kernel config enables the Arm arch timer event stream.

    CONFIG_ARM_ARCH_TIMER_EVTSTREAM=y lets the architected timer generate
    periodic events that wake CPUs from WFE. This checks the exact-release
    config artifact only, not whether the event stream is active at runtime.
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    config_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(config_path)
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot read exact-release kernel config {config_path}: {error}"
        )

    config_name = "CONFIG_ARM_ARCH_TIMER_EVTSTREAM"
    value = kconfig.get(config_name)
    if value != "y":
        GLTCore.GLTTestResult.fail(
            f"{config_name}={value!r} (expected 'y'). "
            "Enable the Arm arch timer event stream in the kernel config."
        )

    logger.info("Exact-release kernel config has %s=y", config_name)
