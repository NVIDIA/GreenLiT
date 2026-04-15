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

"""Python script to test Vera CPU model name via lscpu."""

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev

EXPECTED_MODEL_NAME = "Olympus"


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_lscpu_model_name(_, logger):
    """
    Test if the Vera CPU model name is Olympus.

    PATCH: https://github.com/util-linux/util-linux/commit/9087774
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    model_name = glt_utils.get_cpu_model_name(logger)
    if model_name is None:
        GLTCore.GLTTestResult.fail("Failed to get CPU model name from lscpu.")

    logger.info("lscpu Model name: %s", model_name)

    if EXPECTED_MODEL_NAME != model_name:
        GLTCore.GLTTestResult.fail(
            f"CPU model name mismatch. Expected '{EXPECTED_MODEL_NAME}' in model name, "
            f"but got '{model_name}'. "
            "Please backport the patch to util-linux: "
            "https://github.com/util-linux/util-linux/commit/9087774"
        )
