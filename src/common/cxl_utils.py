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
# DVSEC ID for "PCIe DVSEC for CXL Devices" (Type-1/2/3 endpoints).
_CXL_DEVICE_DVSEC_ID = 0x0000
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
# Lines emitted by lspci's Register Locator DVSEC decoder look like:
#   Block1: BIR: bar0, ID: component registers, offset: 0000000000ec0000
_REGLOC_BLOCK_RE = re.compile(
    r"Block\d+:\s+BIR:\s+bar(\d+),\s+ID:\s+([^,]+?),\s+offset:\s+([0-9a-fA-F]+)",
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


def find_cxl_device_dvsec_base(bdf: str) -> int | None:
    """
    Find the absolute config-space offset of the CXL Device DVSEC
    (Vendor=0x1E98, ID=0x0000) on `bdf` by parsing `lspci -vv` output.

    Mirror of find_cxl_port_dvsec_base() but for the CXL Device DVSEC
    (ID=0x0000) instead of the CXL Port Extensions DVSEC (ID=0x0003).

    Returns None if lspci is missing or fails, or no DVSEC line on `bdf`
    matches the CXL Device DVSEC vendor/ID pair.
    """
    try:
        result = subprocess.run(
            ["lspci", "-vv", "-s", bdf],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None

    for line in result.stdout.splitlines():
        m = _DVSEC_LINE_RE.search(line)
        if not m:
            continue
        if (int(m.group(2), 16) == _CXL_DVSEC_VENDOR
                and int(m.group(3), 16) == _CXL_DEVICE_DVSEC_ID):
            return int(m.group(1), 16)
    return None


def find_first_cxl_device_dvsec_bdf() -> str | None:
    """
    Return the BDF of the first PCI device that exposes a CXL Device DVSEC
    (Vendor=0x1E98, ID=0x0000), or None if none found.

    Discovery rationale: cxl-cli (`cxl list -vvv`) only enumerates Type-3
    memdev objects (and CXL buses/ports/decoders). Type-2 accelerators -
    NVIDIA GPUs, FPGAs etc. - bind to vendor drivers, not cxl_pci/cxl_mem,
    and never appear under /sys/bus/cxl. Walk PCI via lspci instead: Device
    DVSEC presence is the durable, type-agnostic signal.

    Also: find_first_cxl_bdf() matches PCI class 0x0502 (CXL Memory Device)
    which excludes Type-2 by definition.
    """
    try:
        result = subprocess.run(
            ["lspci", "-D"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        bdf = line.split()[0]
        if find_cxl_device_dvsec_base(bdf) is not None:
            return bdf
    return None


def find_cxl_component_regs(bdf: str) -> tuple[int, int] | None:
    """
    Return (bar_index, offset_within_bar) of the CXL component registers
    block on `bdf` by parsing the Register Locator DVSEC (Vendor=0x1E98,
    ID=0x0008) as decoded by `lspci -vv`. Returns None if the DVSEC is
    absent or unparseable.

    Each Block-N entry in the Register Locator DVSEC is rendered by lspci as
    e.g. "Block1: BIR: bar0, ID: component registers, offset: 0000000000ec0000".
    We scan for such Block lines whose ID is "component registers" and return
    the first match.
    """
    try:
        result = subprocess.run(
            ["lspci", "-vv", "-s", bdf],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None

    for line in result.stdout.splitlines():
        m = _REGLOC_BLOCK_RE.search(line)
        if not m:
            continue
        if m.group(2).strip().lower() != "component registers":
            continue
        return int(m.group(1)), int(m.group(3), 16)
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
