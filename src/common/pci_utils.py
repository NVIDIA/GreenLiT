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

import os
import re
import subprocess
from typing import Optional

# Extended Capability names as printed by lspci.
EXT_CAP_ACS = "Access Control Services"

# Known PCI vendor / device IDs.
ASPEED_VENDOR_ID = "1a03"
ASPEED_AST1150_DEVICE_ID = "1150"

# PCI class prefix for USB host controllers (UHCI/OHCI/EHCI/xHCI = 0x0c03xx).
PCI_CLASS_USB_HOST = "0x0c03"

_BDF_RE = re.compile(r"^([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s")
_BDF_BARE_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")


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


def get_bdfs_by_pci_id(vendor_id: str, device_id: Optional[str] = None) -> list[str]:
    """Find PCI devices matching a vendor (and optionally device) ID.

    Args:
        vendor_id: Four-digit hexadecimal PCI vendor ID.
        device_id: Optional four-digit hexadecimal PCI device ID. When
            omitted, all devices from the given vendor are returned.

    Returns:
        List of domain-prefixed BDF strings for matching devices, or empty
        list when lspci is unavailable, fails, or finds no matches.
    """
    pci_id = f"{vendor_id.lower()}:{device_id.lower() if device_id else ''}"
    try:
        result = subprocess.run(
            ["lspci", "-Dnn", "-d", pci_id],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    return [
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := _BDF_RE.match(line))
    ]


def get_pci_children(parent_bdf: str) -> list[str]:
    """Return BDFs of PCI devices directly behind ``parent_bdf``.

    Args:
        parent_bdf: Domain-prefixed BDF of a PCI bridge.

    Returns:
        List of child BDFs, or empty list when ``parent_bdf`` is absent
        from sysfs.
    """
    parent_path = f"/sys/bus/pci/devices/{parent_bdf}"
    try:
        entries = os.listdir(parent_path)
    except OSError:
        return []
    return [entry for entry in entries if _BDF_BARE_RE.match(entry)]


def get_pci_class(bdf: str) -> Optional[str]:
    """Return the PCI class for a device, as a hex string from sysfs.

    Args:
        bdf: Domain-prefixed BDF of the device.

    Returns:
        Hex string like ``"0x0c0330"`` (24 bits: class, subclass, prog-if),
        or ``None`` if the class file cannot be read.
    """
    class_file = f"/sys/bus/pci/devices/{bdf}/class"
    try:
        with open(class_file, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None
