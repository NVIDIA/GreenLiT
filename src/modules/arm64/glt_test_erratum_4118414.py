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
Python script to test ARM64 erratum 4118414 mitigation.
"""

import platform

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


ERRATUM_CONFIG = "CONFIG_ARM64_ERRATUM_4118414"
REPEAT_TLBI_CONFIG = "CONFIG_ARM64_WORKAROUND_REPEAT_TLBI"
REQUIRED_CONFIGS = (ERRATUM_CONFIG, REPEAT_TLBI_CONFIG)
RUNTIME_MSG = "CPU features: detected: Broken broadcast TLBI completion"

PATCH_CONTEXT = (
    "consider checking whether the kernel is missing ARM64_ERRATUM_4118414 "
    "mitigation commits ba18ed542804 and 1caf7bd4a13b, or config enablement "
    "commit 5a9e6b0fe3f2."
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm64_erratum_4118414(_, logger):
    """
    Verify ARM64 erratum 4118414 TLBI mitigation config and runtime detection.

    PATCH: ba18ed542804, 1caf7bd4a13b, 5a9e6b0fe3f2
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    config_path = "/boot/config-" + platform.uname().release
    logger.info("Checking kernel config: %s", config_path)
    try:
        kconfig = check_kernel_config.parse_kconfig(config_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {config_path}; {PATCH_CONTEXT}"
        )

    missing_configs = [symbol for symbol in REQUIRED_CONFIGS if kconfig.get(symbol) != "y"]
    if missing_configs:
        missing = ", ".join(f"{symbol}=y" for symbol in missing_configs)
        GLTCore.GLTTestResult.fail(f"Missing required kernel configs: {missing}; {PATCH_CONTEXT}")

    logger.info("Required erratum configs are enabled: %s", ", ".join(REQUIRED_CONFIGS))

    try:
        runtime_msg_found = glt_utils.check_kernel_log(RUNTIME_MSG)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve ARM64 erratum 4118414 kernel log: {error}"
        )

    if not runtime_msg_found:
        GLTCore.GLTTestResult.fail(
            f"Runtime erratum print not found: '{RUNTIME_MSG}'; {PATCH_CONTEXT}"
        )

    logger.info("Runtime erratum print found: %s", RUNTIME_MSG)
