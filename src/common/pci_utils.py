# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Utilities for inspecting PCI device capabilities via lspci.
"""

import re
import subprocess

# Extended Capability names as printed by lspci.
# Add new names here as tests require them.
EXT_CAP_ACS = "Access Control Services"

_BDF_RE = re.compile(r"^([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s")


def get_bdfs_with_ext_cap(cap_name: str) -> list[str]:
    """Find all PCI devices that advertise a given extended capability.

    Runs ``lspci -Dvv`` and parses the output for capability entries
    matching *cap_name*.

    Args:
        cap_name: Capability name as printed by lspci
                  (e.g. "Access Control Services").

    Returns:
        List of domain-prefixed BDF strings (e.g. "0000:00:1f.0")
        that have the capability, or empty list on failure.
    """
    try:
        result = subprocess.run(
            ["lspci", "-Dvv"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    matched_bdfs = []
    current_bdf = None

    for line in result.stdout.splitlines():
        bdf_match = _BDF_RE.match(line)
        if bdf_match:
            current_bdf = bdf_match.group(1)
            continue

        if current_bdf and cap_name in line:
            matched_bdfs.append(current_bdf)
            current_bdf = None

    return matched_bdfs
