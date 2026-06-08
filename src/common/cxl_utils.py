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
CXL-specific discovery helpers (devices, port DVSEC, regions).

This module is intentionally independent of the GreenLiT result classes
(GLTTestResult); callers are responsible for translating return values
into skip/fail/pass outcomes.
"""

import glob
import os
import re
import subprocess


# PCI class code reported by CXL Type-3 devices ("CXL Memory Device").
_CXL_DEVICE_PCI_CLASS = "0502"
# DVSEC Vendor ID assigned to the CXL Consortium.
_CXL_DVSEC_VENDOR = 0x1E98
# DVSEC ID for "PCIe DVSEC for CXL Ports".
_CXL_PORT_EXT_DVSEC_ID = 0x0003
# Lines look like:
#   Capabilities: [468 v1] Designated Vendor-Specific: Vendor=1e98 ID=0003 Rev=2 Len=14: CXL
_DVSEC_LINE_RE = re.compile(
    r"Capabilities:\s*\[([0-9a-fA-F]+)\s+v\d+\]\s+"
    r"Designated Vendor-Specific:\s+"
    r"Vendor=([0-9a-fA-F]+)\s+ID=([0-9a-fA-F]+)",
    re.IGNORECASE,
)


def find_first_cxl_bdf() -> str | None:
    """Return the BDF of the first PCI device.

    Returns None if no CXL device is found, lspci is missing, or lspci exits
    non-zero.
    """
    try:
        result = subprocess.run(
            ["lspci", "-d", f"::{_CXL_DEVICE_PCI_CLASS}", "-D"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        return line.split()[0]
    return None


def find_cxl_port_dvsec_base(bdf: str) -> int | None:
    """Find the absolute config-space offset of the CXL Port Extensions DVSEC
    on `bdf` by parsing `lspci -vv` output.

    Returns None if lspci is missing or fails, or no DVSEC line on `bdf`
    matches the CXL Port Extensions vendor (0x1E98) and DVSEC ID (0x0003) pair.
    """
    try:
        result = subprocess.run(
            ["lspci", "-vv", "-s", bdf],
            capture_output=True, text=True, check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        m = _DVSEC_LINE_RE.search(line)
        if not m:
            continue
        if (int(m.group(2), 16) == _CXL_DVSEC_VENDOR
                and int(m.group(3), 16) == _CXL_PORT_EXT_DVSEC_ID):
            return int(m.group(1), 16)
    return None


def any_cxl_region_committed(sysfs_root: str = "/") -> bool:
    """Return True if any CXL region directory under `sysfs_root` is committed."""
    pattern = os.path.join(
        sysfs_root, "sys", "bus", "cxl", "devices", "region*",
    )
    for region_dir in glob.glob(pattern):
        commit_path = os.path.join(region_dir, "commit")
        try:
            with open(commit_path, "r", encoding="utf-8") as f:
                value = f.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            # Unreadable commit state -> conservatively assume committed so a
            # caller guarding a destructive op never proceeds when unsure.
            return True
        if value == "1":
            return True
    return False
