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
NVIDIA GPU test cases.
"""

import re
import subprocess
from common import glt_core as GLTCore
from common import glt_utils

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_persistence_mode(_, logger):
    """
    Test if NVIDIA GPU Persistence Mode is enabled.
    """
    if not glt_utils.is_cli_available("nvidia-smi"):
        GLTCore.GLTTestResult.skip("nvidia-smi is not available on this system.")

    try:
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )

        if result.returncode != 0:
            GLTCore.GLTTestResult.fail(f"nvidia-smi failed with exit code {result.returncode}:\n{result.stderr}")

        output = result.stdout
        logger.debug("nvidia-smi output:\n%s", output)

        # Dynamically find the number of attached GPUs using regex
        # This handles any number of spaces/tabs between the key and value
        attached_match = re.search(r"Attached GPUs\s*:\s*(\d+)", output)
        if not attached_match:
            GLTCore.GLTTestResult.skip("Could not determine the number of attached GPUs.")

        attached_gpus = int(attached_match.group(1))
        if attached_gpus == 0:
            GLTCore.GLTTestResult.skip("No NVIDIA GPUs attached to the system.")

        # Dynamically find all Persistence Mode entries for all GPUs
        # Ensure we only match the top-level "Persistence Mode" and not nested ones or other keys containing it
        persistence_modes = re.findall(r"^\s{4}Persistence Mode\s*:\s*(\w+)", output, re.MULTILINE)

        if not persistence_modes:
            GLTCore.GLTTestResult.fail("Could not determine Persistence Mode from nvidia-smi output.")

        # Ensure we found exactly as many Persistence Mode entries as Attached GPUs
        if len(persistence_modes) != attached_gpus:
            GLTCore.GLTTestResult.fail(
                f"Found {len(persistence_modes)} Persistence Mode entries, but expected {attached_gpus}."
            )

        # Count how many GPUs do not have it Enabled
        disabled_count = sum(1 for mode in persistence_modes if mode.lower() != "enabled")

        if disabled_count > 0:
            GLTCore.GLTTestResult.fail(
                f"Persistence Mode is not enabled on {disabled_count} out of {len(persistence_modes)} GPU(s)."
            )

    except subprocess.TimeoutExpired as e:
        GLTCore.GLTTestResult.fail(f"Command '{e.cmd}' timed out after {e.timeout} seconds")
