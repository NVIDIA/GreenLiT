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
import json
import os
import re
import subprocess

from common import glt_utils
from common import pci_utils


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


def read_cxl_device_dvsec_cap_bits(
    bdf: str, dvsec_off: int, sysfs_root: str = "/",
) -> dict[str, bool] | None:
    """
    Read the CXL Device DVSEC CAP register on `bdf` (CXL r3.2 §8.1.3.1)
    via <sysfs_root>/sys/bus/pci/devices/<bdf>/config and return its decoded
    bit flags (cache_capable, mem_capable, cache_wbi_capable, rst_capable).

    `dvsec_off` is the absolute config-space offset of the CXL Device DVSEC,
    as returned by find_cxl_device_dvsec_base().

    Returns None if the config-space read fails (device gone, permission
    denied, short read).
    """
    # Bit positions per CXL r3.2 §8.1.3.1.
    cap_reg_off = 0xA
    cache_capable_bit = 1 << 0
    mem_capable_bit = 1 << 2
    cache_wbi_capable_bit = 1 << 6
    rst_capable_bit = 1 << 7

    raw = pci_utils.read_config_word(
        bdf, dvsec_off + cap_reg_off, sysfs_root=sysfs_root,
    )
    if raw is None:
        return None
    return {
        "cache_capable":     bool(raw & cache_capable_bit),
        "mem_capable":       bool(raw & mem_capable_bit),
        "cache_wbi_capable": bool(raw & cache_wbi_capable_bit),
        "rst_capable":       bool(raw & rst_capable_bit),
    }


def read_cxl_device_dvsec_status2_bits(
    bdf: str, dvsec_off: int, sysfs_root: str = "/",
) -> dict[str, bool] | None:
    """
    Read the CXL Device DVSEC STATUS2 register on `bdf` (CXL r3.2 §8.1.3.1)
    via <sysfs_root>/sys/bus/pci/devices/<bdf>/config and return its decoded
    bit flags (cache_inv, rst_done, rst_err).

    `dvsec_off` is the absolute config-space offset of the CXL Device DVSEC,
    as returned by find_cxl_device_dvsec_base().

    Returns None if the config-space read fails (device gone, permission
    denied, short read).
    """
    # Bit positions per CXL r3.2 §8.1.3.1.
    status2_reg_off = 0x12
    cache_inv_bit = 1 << 0
    rst_done_bit = 1 << 1
    rst_err_bit = 1 << 2

    raw = pci_utils.read_config_word(
        bdf, dvsec_off + status2_reg_off, sysfs_root=sysfs_root,
    )
    if raw is None:
        return None
    return {
        "cache_inv": bool(raw & cache_inv_bit),
        "rst_done":  bool(raw & rst_done_bit),
        "rst_err":   bool(raw & rst_err_bit),
    }


def read_cxl_port_unmask_sbr(
    bdf: str, dvsec_off: int, sysfs_root: str = "/",
) -> bool | None:
    """Return whether Unmask SBR is set in the CXL Port Control register.

    `dvsec_off` is the CXL Port Extensions DVSEC base from
    find_cxl_port_dvsec_base(). True means Unmask SBR is 1 (SBR can take
    effect); False means it is 0 (SBR is masked). None if the config-space
    read fails.
    """
    # CXL r3.1 §8.1.5.2 Port Control Extensions; Unmask SBR is bit 0.
    port_ctl_off = 0x0C
    unmask_sbr_bit = 0x0001

    raw = pci_utils.read_config_word(
        bdf, dvsec_off + port_ctl_off, sysfs_root=sysfs_root,
    )
    if raw is None:
        return None
    return bool(raw & unmask_sbr_bit)


def read_cxl_flexbus_port_cxl_mode(
    bdf: str, dvsec_off: int, sysfs_root: str = "/",
) -> bool | None:
    """Return whether Flex Bus Port Status reports CXL.cache or CXL.mem.

    Mirrors pcie_is_cxl() / set_pcie_cxl(): the kernel sets pci_dev->is_cxl
    when either status bit is set. `dvsec_off` is the Flex Bus Port DVSEC
    base (DVSEC ID 0x0007). None if the config-space read fails.
    """
    # Flex Bus Port Status (CXL r3.1 / r4.0 §8.1.8).
    status_off = 0x0E
    cache_status_bit = 1 << 0
    mem_status_bit = 1 << 2

    raw = pci_utils.read_config_word(
        bdf, dvsec_off + status_off, sysfs_root=sysfs_root,
    )
    if raw is None:
        return None
    return bool(raw & (cache_status_bit | mem_status_bit))


def is_cxl_sbr_masked_port(bdf: str, sysfs_root: str = "/") -> bool | None:
    """Return whether `bdf` matches pcie_is_cxl() && cxl_sbr_masked().

    True when the port is in CXL mode and Unmask SBR is 0, so the PCI core
    must hide the ``bus`` reset method from downstream devices. False when
    the condition is conclusively absent (no CXL Port / Flex Bus DVSEC, link
    in PCIe mode, or Unmask SBR is 1). None when lspci or config space
    cannot be read, so the caller cannot tell.
    """
    try:
        result = glt_utils.run_command(
            ["lspci", "-vv", "-s", bdf],
            f"Failed to read CXL DVSECs on {bdf}",
        )
    except RuntimeError:
        return None

    flexbus_port_dvsec_id = 0x0007
    port_off = None
    flex_off = None
    for line in result.stdout.splitlines():
        match = _DVSEC_LINE_RE.search(line)
        if not match:
            continue
        if int(match.group(2), 16) != _CXL_DVSEC_VENDOR:
            continue
        dvsec_id = int(match.group(3), 16)
        offset = int(match.group(1), 16)
        if dvsec_id == _CXL_PORT_EXT_DVSEC_ID and port_off is None:
            port_off = offset
        elif dvsec_id == flexbus_port_dvsec_id and flex_off is None:
            flex_off = offset
        if port_off is not None and flex_off is not None:
            break
    if port_off is None or flex_off is None:
        # pciutils still exits 0 when a capability cannot be decoded.
        if "<access denied>" in result.stdout:
            return None
        return False
    in_cxl_mode = read_cxl_flexbus_port_cxl_mode(bdf, flex_off, sysfs_root)
    if not in_cxl_mode:
        return in_cxl_mode
    unmask_sbr = read_cxl_port_unmask_sbr(bdf, port_off, sysfs_root)
    if unmask_sbr is None:
        return None
    return not unmask_sbr


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


def find_committed_zero_size_decoders(sysfs_root: str = "/") -> list[str]:
    """Return names of decoders committed and LOCKed with size 0.

    A size-0 decoder on its own is just an idle slot; locked == 1 marks the
    slot BIOS committed and locked, which is the case the kernel must
    enumerate. Decoders whose size/locked attributes cannot be read are skipped
    so they are never mistaken for the target.
    """
    pattern = os.path.join(
        sysfs_root, "sys", "bus", "cxl", "devices", "decoder*",
    )
    names = []
    for decoder_dir in sorted(glob.glob(pattern)):
        try:
            with open(os.path.join(decoder_dir, "size"),
                      "r", encoding="utf-8") as f:
                size = int(f.read().strip(), 0)
            with open(os.path.join(decoder_dir, "locked"),
                      "r", encoding="utf-8") as f:
                locked = int(f.read().strip())
        except (OSError, ValueError):
            continue
        if size == 0 and locked == 1:
            names.append(os.path.basename(decoder_dir))
    return names


def cxl_listed_decoder_names() -> set[str] | None:
    """Return the set of decoder names reported by `cxl list`.

    Uses `-i` so a committed but region-less (idle-looking) decoder is still
    listed. The output is walked recursively so the result is independent of
    how a given cxl-cli version nests decoder objects.
    """
    try:
        result = subprocess.run(
            ["cxl", "list", "-D", "-i"],
            capture_output=True, text=True, check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    names: set[str] = set()

    def _collect(node):
        if isinstance(node, dict):
            name = node.get("decoder")
            if isinstance(name, str):
                names.add(name)
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(data)
    return names
