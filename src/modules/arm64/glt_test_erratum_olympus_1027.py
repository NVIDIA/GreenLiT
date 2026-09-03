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
Python script to test NVIDIA Olympus T410-OLY-1027 erratum mitigation.
"""

import platform

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


ERRATUM_CONFIG = "CONFIG_NVIDIA_OLYMPUS_1027_ERRATUM"
RUNTIME_MSG = "NVIDIA Olympus device store/load ordering erratum"

# Queued in arm64 for-next; update to the mainline commit once the fix lands.
FIX_COMMIT = "12aab25ca56ee9ab3051f7832402702d0c1953eb"
FIX_REFERENCE = (
    "https://git.kernel.org/pub/scm/linux/kernel/git/arm64/linux.git/"
    f"commit/?id={FIX_COMMIT}"
)

PATCH_CONTEXT = (
    "consider checking whether the kernel carries arm64 for-next commit "
    f"{FIX_COMMIT} or an equivalent downstream backport ({FIX_REFERENCE}). "
    "This test checks the T410-OLY-1027 config and runtime cpucap marker; "
    "replace this reference with the mainline commit once the fix lands."
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_olympus_1027_erratum(_, logger):
    """
    Verify NVIDIA Olympus T410-OLY-1027 config and runtime cpucap markers.

    PATCH: arm64 for-next commit 12aab25ca56e; see FIX_REFERENCE.
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not an NVIDIA Vera CPU with Olympus cores.")

    config_path = "/boot/config-" + platform.uname().release
    logger.info("Checking kernel config: %s", config_path)
    try:
        kconfig = check_kernel_config.parse_kconfig(config_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {config_path}; {PATCH_CONTEXT}"
        )

    if kconfig.get(ERRATUM_CONFIG) != "y":
        GLTCore.GLTTestResult.fail(
            f"Missing required kernel config: {ERRATUM_CONFIG}=y; {PATCH_CONTEXT}"
        )

    logger.info("Required erratum config is enabled: %s", ERRATUM_CONFIG)

    try:
        runtime_msg_found = glt_utils.check_kernel_log(RUNTIME_MSG)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve Olympus 1027 erratum kernel log: {error}"
        )

    if not runtime_msg_found:
        GLTCore.GLTTestResult.fail(
            f"Runtime erratum print not found: '{RUNTIME_MSG}'; {PATCH_CONTEXT}"
        )

    logger.info("Runtime erratum print found: %s", RUNTIME_MSG)
