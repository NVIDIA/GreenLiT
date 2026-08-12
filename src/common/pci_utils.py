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
Utilities for inspecting PCI device capabilities via lspci and sysfs.
"""

import os
import re
import subprocess
from typing import Optional

from common import glt_utils

# Extended Capability names as printed by lspci.
EXT_CAP_ACS = "Access Control Services"
EXT_CAP_ATS = "Address Translation Service (ATS)"

# Known PCI vendor / device IDs.
ASPEED_VENDOR_ID = "1a03"
ASPEED_AST1150_DEVICE_ID = "1150"

# PCI class prefix for USB host controllers (UHCI/OHCI/EHCI/xHCI = 0x0c03xx).
PCI_CLASS_USB_HOST = "0x0c03"

_BDF_RE = re.compile(r"^([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s")
_BDF_BARE_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")

# Stricter BDF form (lowercase hex, single-digit function) for callers that
# splice untrusted input into a sysfs path and must reject path-traversal
# payloads such as "../../etc/passwd". Prefer _BDF_BARE_RE for general
# matching; use this variant when path safety matters.
BDF_STRICT_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


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


def default_route_ifaces() -> set:
    """Return the interface names holding an IPv4 or IPv6 default route.

    Reads /proc/net/route (IPv4) and /proc/net/ipv6_route (IPv6). Returns
    an empty set when the readable table(s) contain no default route.

    Raises:
        OSError: if neither route table is readable. Callers that gate
        destructive PCI operations must treat this as fail-closed — an
        empty set must not be confused with "no management NIC".
    """
    ifaces = set()
    any_readable = False
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as f:
            lines = f.readlines()
        any_readable = True
    except OSError:
        lines = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "00000000":
            ifaces.add(fields[0])

    try:
        with open("/proc/net/ipv6_route", "r", encoding="utf-8") as f:
            lines6 = f.readlines()
        any_readable = True
    except OSError:
        lines6 = []
    for line in lines6:
        fields = line.split()
        # All-zero destination with prefix length 00 is the IPv6 default
        # route; the interface name is the last field.
        if len(fields) >= 10 and fields[0] == "0" * 32 and fields[1] == "00":
            ifaces.add(fields[-1])

    if not any_readable:
        raise OSError(
            "neither /proc/net/route nor /proc/net/ipv6_route is readable")
    return ifaces


def _iface_pci_bdf(iface):
    """Return the PCI BDF backing a network interface, or None if not PCI."""
    # Domain-prefixed PCI BDF sysfs path component (e.g. "0000:01:00.0").
    bdf_component_re = re.compile(
        r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
    base = os.path.basename(
        os.path.realpath(f"/sys/class/net/{iface}/device"))
    return base if bdf_component_re.match(base) else None


def iface_pci_bdfs(iface, _seen=None) -> set:
    """Return the PCI BDF(s) of the physical NIC(s) backing `iface`.

    A directly PCI-backed interface maps to its own BDF; bond/bridge/VLAN
    interfaces are descended through /sys/class/net/<if>/lower_* links and
    bonding/slaves until PCI-backed interfaces are found.

    Args:
        iface: Network interface name (e.g. "eth0", "bond0").
        _seen: Internal recursion guard; callers must not pass it.

    Returns:
        set: Lower-cased domain-prefixed BDF strings, or an empty set if no
        physical PCI NIC backs the interface.
    """
    if _seen is None:
        _seen = set()
    if iface in _seen:
        return set()
    _seen.add(iface)

    direct = _iface_pci_bdf(iface)
    if direct is not None:
        return {direct.lower()}

    lowers = []
    try:
        entries = os.listdir(f"/sys/class/net/{iface}")
    except (FileNotFoundError, NotADirectoryError):
        entries = []
    for entry in entries:
        if entry.startswith("lower_"):
            lowers.append(entry[len("lower_"):])
    try:
        lowers.extend(glt_utils.read_sysfs_value(
            f"/sys/class/net/{iface}/bonding/slaves").split())
    except PermissionError:
        raise
    except OSError:
        pass

    bdfs = set()
    for lower in lowers:
        bdfs |= iface_pci_bdfs(lower, _seen)
    return bdfs


def mounted_block_devices() -> set:
    """Return base names of block devices backing mounted filesystems.

    Parses /proc/mounts and resolves each /dev/* mount source through
    symlinks (e.g. /dev/mapper/vg-root -> dm-0) to its kernel device name.

    Returns:
        set: Block device base names (e.g. {"nvme0n1p2", "dm-0"}), or an
        empty set when /proc/mounts is readable but has no /dev/* mounts.

    Raises:
        OSError: if /proc/mounts is unreadable. Callers that gate
        destructive PCI operations must treat this as fail-closed — an
        empty set must not be confused with "no mounted storage".
    """
    with open("/proc/mounts", "r", encoding="utf-8") as f:
        lines = f.readlines()
    devs = set()
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("/dev/"):
            continue
        devs.add(os.path.basename(os.path.realpath(fields[0])))
    return devs


def block_device_physical_leaves(dev, _seen=None) -> set:
    """Return the physical leaf block devices backing `dev`.

    Recursively walks /sys/class/block/<dev>/slaves so stacked devices
    (LVM/dm, md RAID, dm-crypt — e.g. /dev/mapper/vg-root -> /dev/dm-0)
    resolve down to their real backing disks. A device with no slaves is
    its own leaf.

    Args:
        dev: Block device base name (e.g. "dm-0", "nvme0n1p2").
        _seen: Internal recursion guard; callers must not pass it.

    Returns:
        set: Leaf block device base names backing `dev`.
    """
    if _seen is None:
        _seen = set()
    if dev in _seen:
        return set()
    _seen.add(dev)
    try:
        slaves = os.listdir(f"/sys/class/block/{dev}/slaves")
    except (FileNotFoundError, NotADirectoryError):
        slaves = []
    if not slaves:
        return {dev}
    leaves = set()
    for slave in slaves:
        leaves |= block_device_physical_leaves(slave, _seen)
    return leaves


def _pci_path_bdfs(sysfs_path: str) -> set:
    """Collect domain-prefixed PCI BDF path components from a sysfs path."""
    return {
        part.lower()
        for part in sysfs_path.split(os.sep)
        if _BDF_BARE_RE.match(part)
    }


def unsafe_pci_bdfs() -> set:
    """BDFs that must never be unbound for destructive PCI rebinds.

    Computes:

    - For each default-route interface: the PCI BDF(s) from
      ``iface_pci_bdfs``; plus, when ``/sys/class/net/<iface>/device``
      exists, every PCI BDF component on that sysfs path (ancestors of a
      directly PCI-backed iface). Bond/bridge/VLAN tops without a
      ``device`` link contribute only the lower physical BDFs from
      ``iface_pci_bdfs`` (no ancestor walk of each slave).
    - For each mounted block device: the **leaf** PCI BDF behind the
      physical backing disk only (bridge/root-port ancestors are not
      included).

    Raises:
        OSError: if default-route tables or /proc/mounts are unreadable,
        or if sysfs resolution hits a non-absence error (e.g.
        ``PermissionError``). Callers that gate destructive PCI
        operations must treat this as fail-closed — an empty set must
        not be confused with "nothing to protect".
    """
    unsafe = set()
    for iface in default_route_ifaces():
        unsafe |= {b.lower() for b in iface_pci_bdfs(iface)}
        device_link = f"/sys/class/net/{iface}/device"
        if os.path.exists(device_link):
            unsafe |= _pci_path_bdfs(os.path.realpath(device_link))
    for dev in mounted_block_devices():
        for leaf in block_device_physical_leaves(dev):
            resolved = os.path.realpath(f"/sys/class/block/{leaf}")
            bdfs = [b.lower() for b in resolved.split(os.sep)
                    if _BDF_BARE_RE.match(b)]
            if bdfs:
                unsafe.add(bdfs[-1])
    return unsafe
def sysfs_path_to_pci_bdf(path: str) -> Optional[str]:
    """Walk a sysfs path upward and return the first valid PCI BDF segment.

    Args:
        path: A sysfs path (e.g. a resolved device symlink target).

    Returns:
        The first path segment, walking from the leaf upward, that is a
        well-formed BDF (per ``BDF_STRICT_RE``) present under
        ``/sys/bus/pci/devices``, or ``None`` if no such segment exists.
    """
    while path and path != "/":
        base = os.path.basename(path)
        if BDF_STRICT_RE.fullmatch(base) and os.path.exists(f"/sys/bus/pci/devices/{base}"):
            return base
        path = os.path.dirname(path)
    return None


def bdf_in_translated_domain(bdf: str) -> bool:
    """Return whether ``bdf`` sits in a translating DMA or DMA-FQ IOMMU group.

    Args:
        bdf: Domain-prefixed BDF of the device.

    Returns:
        ``True`` when the device's ``iommu_group/type`` sysfs attribute is
        ``DMA`` or ``DMA-FQ``, ``False`` otherwise (including when ``bdf`` is
        malformed, or the device or its IOMMU group is not present).
    """
    if not BDF_STRICT_RE.fullmatch(bdf):
        return False
    group_link = f"/sys/bus/pci/devices/{bdf}/iommu_group"
    if not os.path.exists(group_link):
        return False
    try:
        group = os.path.realpath(group_link)
        with open(os.path.join(group, "type"), encoding="utf-8") as handle:
            group_type = handle.read().strip()
    except OSError:
        return False
    return group_type in ("DMA", "DMA-FQ")


def pci_bdfs_backing_mount(mountpoint: str = "/") -> set:
    """Return the PCI BDFs of the physical devices backing ``mountpoint``.

    Resolves the mount source through ``mounted_block_devices`` and
    ``block_device_physical_leaves``, so stacked devices (LVM/dm, MD) reduce to
    the physical disks underneath, then maps each leaf to the PCI device it
    hangs off.

    Args:
        mountpoint: Mount point to resolve, e.g. ``"/"``.

    Returns:
        set: Domain-prefixed BDFs, empty when ``mountpoint`` is not backed by a
        ``/dev/*`` source or has no PCI device beneath it. An empty set does
        not distinguish "nothing found" from "could not look"; callers gating
        destructive operations must check readability themselves rather than
        treating this as fail-closed.

    Raises:
        OSError: if /proc/mounts is unreadable.
    """
    mounted = mounted_block_devices()
    source_dev = None
    with open("/proc/mounts", "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if (len(fields) >= 2 and fields[1] == mountpoint
                    and fields[0].startswith("/dev/")):
                source_dev = os.path.basename(os.path.realpath(fields[0]))
                break
    if source_dev is None or source_dev not in mounted:
        return set()

    bdfs = set()
    for leaf in block_device_physical_leaves(source_dev):
        bdf = sysfs_path_to_pci_bdf(
            os.path.realpath(f"/sys/class/block/{leaf}/device")
        )
        if bdf:
            bdfs.add(bdf)
    return bdfs
