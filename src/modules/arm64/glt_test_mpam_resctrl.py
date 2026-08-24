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

"""ARM64 MPAM and resctrl validation tests."""
# pylint: disable=too-many-lines

from contextlib import contextmanager
import glob
import mmap
import os
import platform
import re
import signal
import struct
import subprocess
import threading
import time

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


PROC_IOMEM = "/proc/iomem"
RESCTRL_FS_TYPE = "resctrl"
RESCTRL_MOUNT_POINT = "/sys/fs/resctrl"
PROC_FILESYSTEMS = "/proc/filesystems"
PROC_MOUNTS = "/proc/mounts"
MPAM_MON_FEATURES = "mon_features"
MPAM_MBM_TOTAL_BYTES = "mbm_total_bytes"
MPAM_PLATFORM_CAPABILITIES = {
    "grace": {
        "expected_dirs": ("L3", "L3_MON", "MB", "MB_HLIM"),
        "expected_files": ("last_cmd_status",),
        "expected_mb_mon_features": (),
    },
    "vera": {
        "expected_dirs": ("L3", "L3_MON", "MB", "MB_HLIM", "MB_MON"),
        "expected_files": ("last_cmd_status",),
        "expected_mb_mon_features": (MPAM_MBM_TOTAL_BYTES,),
    },
}


def _resctrl_fs_supported() -> bool:
    """Return True if the kernel registers the resctrl filesystem type."""
    try:
        with open(PROC_FILESYSTEMS, "r", encoding="utf-8") as f:
            return any(line.split()[-1] == RESCTRL_FS_TYPE
                       for line in f if line.strip())
    except OSError:
        return False


def _resctrl_mount_points():
    """Return every mount point currently using the resctrl filesystem."""
    with open(PROC_MOUNTS, "r", encoding="utf-8") as f:
        return [
            fields[1] for line in f
            if len(fields := line.split()) >= 3
            and fields[2] == RESCTRL_FS_TYPE
        ]


def _resctrl_is_mounted() -> bool:
    """Return True if resctrl is mounted at RESCTRL_MOUNT_POINT."""
    try:
        return RESCTRL_MOUNT_POINT in _resctrl_mount_points()
    except OSError:
        return False


@contextmanager
def _managed_resctrl_mount(logger):
    """Yield resctrl mount state and clean up only a test-owned mount.

    Yields (mount_succeeded, missing_cli). missing_cli is the first of
    mount/umount that is not on PATH when a mount is required, else None.
    Callers skip when missing_cli is set and fail when mount_succeeded is
    False.
    """
    mounted_by_test = False
    missing_cli = None
    mount_succeeded = _resctrl_is_mounted()

    if not mount_succeeded:
        missing_cli = next(
            (cli for cli in ("mount", "umount")
             if not glt_utils.is_cli_available(cli)),
            None,
        )
        if missing_cli is None:
            try:
                glt_utils.run_command(
                    ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE,
                     RESCTRL_MOUNT_POINT],
                    "Failed to mount resctrl.",
                )
            except RuntimeError:
                pass
            else:
                mounted_by_test = True
                mount_succeeded = _resctrl_is_mounted()

    try:
        yield mount_succeeded, missing_cli
    finally:
        if mounted_by_test:
            try:
                glt_utils.run_command(
                    ["umount", RESCTRL_MOUNT_POINT],
                    "Failed to unmount resctrl.",
                )
            except RuntimeError:
                logger.warning("Failed to unmount resctrl.")


def _cpus_of_node(node) -> list:
    """Return the list of online CPU ids owned by a NUMA node.

    Parses /sys/devices/system/node/node<N>/cpulist (e.g. "0-15,88-103").
    Returns an empty list if the node has no CPUs or cannot be read.
    """
    cpus = []
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist",
                  "r", encoding="utf-8") as f:
            spec = f.read().strip()
    except OSError:
        return cpus
    for part in spec.split(","):
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = part.split("-")
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(part))
        except ValueError:
            continue
    return cpus


def _sum_l3_occupancy(occupancy_files) -> int:
    """Return total llc_occupancy summed across the given mon_L3 domains."""
    total = 0
    for occ in occupancy_files:
        with open(occ, "r", encoding="utf-8") as f:
            total += int(f.read().strip())
    return total


def _mpam_msc_regions():
    """Return MPAM MSC MMIO regions from /proc/iomem."""
    msc_iomem_re = re.compile(
        r"^\s*(?P<start>[0-9a-fA-F]+)-(?P<end>[0-9a-fA-F]+)\s+:\s+"
        r"(?P<name>mpam_msc\.\S+)\s+MPAM:MSC\s*$"
    )
    regions = []
    with open(PROC_IOMEM, "r", encoding="utf-8") as iomem:
        for line in iomem:
            match = msc_iomem_re.match(line)
            if match is None:
                continue
            start = int(match.group("start"), 16)
            end = int(match.group("end"), 16)
            if start and end >= start:
                regions.append(
                    (match.group("name"), start, end - start + 1))
    return regions


def _mpam_read32(registers, offset):
    """Read a little-endian 32-bit value from an MSC register view."""
    return struct.unpack_from("<I", registers, offset)[0]


def _mpam_read_idr(registers):
    """Read the complete MPAM feature identification register."""
    mpamf_idr = 0x0000
    mpamf_idr_hi = 0x0004
    idr_ext = 1 << 28

    low = _mpam_read32(registers, mpamf_idr)
    high = _mpam_read32(registers, mpamf_idr_hi) if low & idr_ext else 0
    return low | (high << 32)


def _mpam_write32(registers, offset, value):
    """Write a little-endian 32-bit value to an MSC register view."""
    struct.pack_into("<I", registers, offset, value & 0xFFFFFFFF)


def _validate_mpam_init_registers(  # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
        registers, label):
    """Validate init-programmed allocation registers for each RIS/PARTID."""
    mpamf_iidr = 0x0018
    mpamf_cpor_idr = 0x0030
    mpamf_ccap_idr = 0x0038
    mpamf_mbw_idr = 0x0040
    mpamf_partid_nrw_idr = 0x0050
    mpamcfg_part_sel = 0x0100
    mpamcfg_cmax = 0x0108
    mpamcfg_cmin = 0x0110
    mpamcfg_mbw_min = 0x0200
    mpamcfg_mbw_max = 0x0208
    mpamcfg_cpbm = 0x1000

    idr_has_ccap_part = 1 << 24
    idr_has_cpor_part = 1 << 25
    idr_has_mbw_part = 1 << 26
    idr_has_partid_nrw = 1 << 31
    idr_has_ris = 1 << 32
    ccap_idr_has_cmin = 1 << 29
    ccap_idr_no_cmax = 1 << 30
    mbw_idr_has_min = 1 << 10
    mbw_idr_has_max = 1 << 11
    mbw_max_hardlim = 1 << 31
    part_sel_internal = 1 << 16
    t241_msc_iidr = 0x2410036B

    failures = []
    checks = 0

    original_selector = _mpam_read32(registers, mpamcfg_part_sel)
    ris_count = 0
    try:
        _mpam_write32(registers, mpamcfg_part_sel, 0)
        initial_idr = _mpam_read_idr(registers)
        iidr = _mpam_read32(registers, mpamf_iidr)
        ris_count = (
            ((initial_idr >> 56) & 0xF) + 1
            if initial_idr & idr_has_ris else 1
        )
        ris_features = []

        for ris in range(ris_count):
            _mpam_write32(registers, mpamcfg_part_sel, ris << 24)
            idr = _mpam_read_idr(registers)
            partid_max = idr & 0xFFFF
            if idr & idr_has_partid_nrw:
                partid_max = min(
                    partid_max,
                    _mpam_read32(
                        registers, mpamf_partid_nrw_idr) & 0xFFFF
                )
            ccap_idr = (
                _mpam_read32(registers, mpamf_ccap_idr)
                if idr & idr_has_ccap_part else 0
            )
            cpor_idr = (
                _mpam_read32(registers, mpamf_cpor_idr)
                if idr & idr_has_cpor_part else 0
            )
            mbw_idr = (
                _mpam_read32(registers, mpamf_mbw_idr)
                if idr & idr_has_mbw_part else 0
            )
            ris_features.append(
                (idr, ccap_idr, cpor_idr, mbw_idr, partid_max))

        for ris, features in enumerate(ris_features):
            idr, ccap_idr, cpor_idr, mbw_idr, partid_max = features
            for partid in range(min(2, partid_max + 1)):
                selector = (ris << 24) | partid
                if idr & idr_has_partid_nrw:
                    selector |= part_sel_internal
                _mpam_write32(registers, mpamcfg_part_sel, selector)

                if idr & idr_has_ccap_part:
                    cmax_width = ccap_idr & 0x3F
                    if cmax_width and not (
                            ccap_idr & ccap_idr_no_cmax):
                        actual = (
                            _mpam_read32(registers, mpamcfg_cmax) & 0xFFFF)
                        expected = (
                            (1 << cmax_width) - 1) << (16 - cmax_width)
                        checks += 1
                        if (actual & expected) != expected:
                            failures.append(
                                f"{label} RIS {ris} PARTID {partid} CMAX: "
                                f"expected allowed bits 0x{expected:x}, read "
                                f"0x{actual:04x}"
                            )

                    if ccap_idr & ccap_idr_has_cmin:
                        actual = (
                            _mpam_read32(registers, mpamcfg_cmin) & 0xFFFF)
                        checks += 1
                        if actual != 0:
                            failures.append(
                                f"{label} RIS {ris} PARTID {partid} CMIN: "
                                f"expected 0x0000, read 0x{actual:04x}"
                            )

                cpbm_width = cpor_idr & 0xFFFF
                for word in range((cpbm_width + 31) // 32):
                    implemented = min(32, cpbm_width - word * 32)
                    expected = (
                        0xFFFFFFFF
                        if implemented == 32 else (1 << implemented) - 1
                    )
                    actual = _mpam_read32(registers, mpamcfg_cpbm + word * 4)
                    checks += 1
                    if actual != expected:
                        failures.append(
                            f"{label} RIS {ris} PARTID {partid} CPBM word "
                            f"{word}: expected "
                            f"0x{expected:08x}, read 0x{actual:08x}"
                        )

                bwa_width = min(mbw_idr & 0x3F, 16)
                significant = ((1 << bwa_width) - 1) << (16 - bwa_width)
                if bwa_width and mbw_idr & mbw_idr_has_min:
                    actual = _mpam_read32(registers, mpamcfg_mbw_min) & 0xFFFF
                    expected = 0
                    if iidr == t241_msc_iidr:
                        granule = 1 << (16 - bwa_width)
                        delta = (5 * 0xFFFF) // 100 - 1
                        reduced = (
                            significant - delta
                            if significant > delta else 0
                        )
                        expected = max(reduced, granule)
                    checks += 1
                    if (actual & significant) != (expected & significant):
                        failures.append(
                            f"{label} RIS {ris} PARTID {partid} MBW_MIN: "
                            f"expected implemented "
                            f"bits 0x{expected & significant:04x}, read "
                            f"0x{actual:04x} (mask 0x{significant:04x})"
                        )

                if bwa_width and mbw_idr & mbw_idr_has_max:
                    actual = _mpam_read32(registers, mpamcfg_mbw_max)
                    checks += 1
                    if (actual & significant) != significant:
                        failures.append(
                            f"{label} RIS {ris} PARTID {partid} MBW_MAX: "
                            f"expected implemented "
                            f"bits 0x{significant:04x}, read "
                            f"0x{actual & 0xFFFF:04x}"
                        )

                    max_limit = (mbw_idr >> 8) & 0x3
                    if max_limit != 3:
                        expected_hardlim = int(max_limit == 2)
                        actual_hardlim = int(bool(actual & mbw_max_hardlim))
                        if actual_hardlim != expected_hardlim:
                            failures.append(
                                f"{label} RIS {ris} PARTID {partid} MBW_MAX "
                                f"HARDLIM: expected "
                                f"{expected_hardlim} for MAX_LIM={max_limit}, "
                                f"read {actual_hardlim}"
                            )
    finally:
        _mpam_write32(registers, mpamcfg_part_sel, original_selector)

    return checks, ris_count, failures


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_driver_init_registers(  # pylint: disable=too-many-branches,too-many-locals
        _, __):
    """Validate allocation defaults programmed during MPAM driver init.

    Skips when CONFIG_ARM64_MPAM is not enabled or none of the target
    allocation registers is supported. Fails if resctrl is already
    mounted so driver-init defaults can be observed without changing a
    caller-owned mount. Checks capability-gated CMAX, CMIN, CPBM, MBW_MIN,
    and MBW_MAX values for up to the first two supported PARTIDs on every
    RIS.

    Raw /dev/mem access cannot synchronize PART_SEL with the bound driver or
    enforce the MSC CPU-accessibility mask. Replace it with a kernel-mediated
    snapshot interface when one becomes available.
    """
    dev_mem = "/dev/mem"
    platform_device_path = "/sys/bus/platform/devices"

    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not an NVIDIA Vera CPU; test is Vera-specific.")

    mpam_config = "CONFIG_ARM64_MPAM"
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"Kernel config not found at {kconfig_path}; "
            f"cannot verify {mpam_config}."
        )
    mpam = kconfig.get(mpam_config)
    if mpam != "y":
        GLTCore.GLTTestResult.skip(
            f"{mpam_config} is not enabled (found {mpam!r}); MPAM unavailable."
        )

    try:
        resctrl_mounts = _resctrl_mount_points()
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot determine resctrl mount state from {PROC_MOUNTS}: "
            f"{error}"
        )
    if resctrl_mounts:
        GLTCore.GLTTestResult.fail(
            "resctrl is mounted at "
            + ", ".join(resctrl_mounts)
            + "; unmount it before running this test so MPAM driver-init "
            "defaults can be observed."
        )

    regions = []
    try:
        regions = _mpam_msc_regions()
        if not regions:
            GLTCore.GLTTestResult.fail(
                "No device-specific MPAM MSC MMIO regions were found.")
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot read MPAM MSC resources from {PROC_IOMEM}: {error}")

    unbound = [
        name for name, _, _ in regions
        if not os.path.islink(
            os.path.join(platform_device_path, name, "driver"))
    ]
    if unbound:
        GLTCore.GLTTestResult.fail(
            "MPAM MSC driver initialization is not complete for: "
            + ", ".join(unbound)
        )

    checked_registers = 0
    failures = []
    access_errors = []
    for name, base, size in regions:
        page_size = mmap.PAGESIZE
        map_base = base & ~(page_size - 1)
        register_base = base - map_base
        map_length = (
            register_base + size + page_size - 1) & ~(page_size - 1)
        descriptor = None
        try:
            descriptor = os.open(dev_mem, os.O_RDWR | os.O_SYNC)
            with mmap.mmap(
                    descriptor, map_length, flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=map_base) as mapping:
                with memoryview(mapping)[
                        register_base:register_base + size] as registers:
                    checks, _ris_count, register_failures = (
                        _validate_mpam_init_registers(
                            registers, f"{name} @ 0x{base:x}")
                    )
        except (OSError, ValueError, struct.error) as error:
            access_errors.append(
                f"{name} @ 0x{base:x} through {dev_mem}: {error}")
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)
        checked_registers += checks
        failures.extend(register_failures)

    if failures:
        GLTCore.GLTTestResult.fail(
            "MPAM MSC driver-init register validation failed:\n"
            + "\n".join(f"- {failure}" for failure in failures)
            + "\nKernel commit 09b89d2 should be backported."
        )
    if access_errors:
        GLTCore.GLTTestResult.fail(
            "Raw access was unavailable for one or more MPAM MSCs:\n"
            + "\n".join(f"- {error}" for error in access_errors)
        )
    if checked_registers == 0:
        GLTCore.GLTTestResult.skip(
            "No discovered MPAM MSC advertises CMAX, CMIN, CPBM, "
            "MBW_MIN, or MBW_MAX. driver-init register validation is not "
            "applicable."
        )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_driver_init_registers")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_mount(_, logger):
    """
    Verify the ARM64 MPAM resctrl filesystem can be mounted.

    Skips when MPAM/resctrl support is not enabled in the kernel; fails when
    the feature is enabled but the resctrl filesystem cannot be mounted.

    JIRA: DGX-16759
    """
    mpam_config = "CONFIG_ARM64_MPAM"
    resctrl_fs_config = "CONFIG_RESCTRL_FS"

    # Skip gate: kernel configuration must enable MPAM and the resctrl fs.
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"Kernel config not found at {kconfig_path}; "
            f"cannot verify {mpam_config} and {resctrl_fs_config}."
        )

    mpam = kconfig.get(mpam_config)
    if mpam != "y":
        GLTCore.GLTTestResult.skip(
            f"{mpam_config} is not enabled (found {mpam!r}); MPAM unavailable."
        )

    resctrl_fs = kconfig.get(resctrl_fs_config)
    if resctrl_fs != "y":
        GLTCore.GLTTestResult.skip(
            f"{resctrl_fs_config} is not enabled (found {resctrl_fs!r}); "
            "resctrl filesystem unavailable."
        )

    if not _resctrl_fs_supported():
        GLTCore.GLTTestResult.fail(
            f"'{RESCTRL_FS_TYPE}' filesystem not registered in {PROC_FILESYSTEMS}."
        )

    with _managed_resctrl_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_info(_, logger):  # pylint: disable=too-many-branches
    """
    Verify the ARM64 MPAM resctrl info directory is present and accessible.

    JIRA: DGX-16760
    """
    info_dir = os.path.join(RESCTRL_MOUNT_POINT, "info")

    with _managed_resctrl_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        if not os.path.isdir(info_dir):
            GLTCore.GLTTestResult.fail(
                f"resctrl info directory not found at {info_dir}; "
                "MPAM resctrl info interface unavailable."
            )

        platform_profile = ""
        nvdev = NvDev()
        for profile in MPAM_PLATFORM_CAPABILITIES:
            if nvdev.is_cpu_type(profile):
                platform_profile = profile
                break
        if not platform_profile:
            GLTCore.GLTTestResult.skip(
                "This is not an NVIDIA Grace or Vera CPU."
            )
        platform_capabilities = MPAM_PLATFORM_CAPABILITIES[platform_profile]

        errors = []
        for name in platform_capabilities["expected_dirs"]:
            path = os.path.join(info_dir, name)
            if not os.path.exists(path):
                errors.append(f"{name} (expected directory) is missing")
            elif not os.path.isdir(path):
                errors.append(f"{name} exists but is not a directory")

        for name in platform_capabilities["expected_files"]:
            path = os.path.join(info_dir, name)
            if not os.path.exists(path):
                errors.append(f"{name} (expected file) is missing")
            elif not os.path.isfile(path):
                errors.append(f"{name} exists but is not a file")

        if errors:
            GLTCore.GLTTestResult.fail(
                f"resctrl info directory {info_dir} is missing expected "
                f"entries: {'; '.join(errors)}."
            )

        expected_mb_mon_features = platform_capabilities["expected_mb_mon_features"]
        if expected_mb_mon_features:
            mon_features = os.path.join(info_dir, "MB_MON", MPAM_MON_FEATURES)
            try:
                with open(mon_features, "r", encoding="utf-8") as f:
                    features = f.read().split()
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Cannot read MB_MON {MPAM_MON_FEATURES} at {mon_features}: {e}."
                )
            missing_features = tuple(
                feature for feature in expected_mb_mon_features
                if feature not in features
            )
            if missing_features:
                GLTCore.GLTTestResult.fail(
                    f"MB_MON {MPAM_MON_FEATURES} is missing expected "
                    f"features: {', '.join(missing_features)}."
                )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_schemata(_, logger):
    """
    Verify the ARM64 MPAM resctrl schemata interface is readable.

    JIRA: DGX-16761
    """
    schemata_path = os.path.join(RESCTRL_MOUNT_POINT, "schemata")
    schemata_line_re = re.compile(r"^\s*\S+:\d+=\S+")

    with _managed_resctrl_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        if not os.path.isfile(schemata_path):
            GLTCore.GLTTestResult.fail(
                f"resctrl schemata {schemata_path} does not exist or is not "
                "a file after mount."
            )

        try:
            with open(schemata_path, "r", encoding="utf-8") as f:
                content = f.read()

                if not content.strip():
                    GLTCore.GLTTestResult.fail(
                        f"resctrl schemata {schemata_path} is empty."
                    )

                if not any(schemata_line_re.match(line) for line in content.splitlines()):
                    GLTCore.GLTTestResult.fail(
                        f"resctrl schemata {schemata_path} contains no "
                        "well-formed resource control line."
                    )

        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to read resctrl schemata {schemata_path}: {e}"
            )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_schemata")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_l3_occupancy(_, logger):  # pylint: disable=too-many-branches
    """
    Verify the ARM64 MPAM L3 cache occupancy monitor is exposed and readable.

    JIRA: DGX-16762
    """
    schemata_path = RESCTRL_MOUNT_POINT + "/schemata"
    l3_line_re = re.compile(r"^\s*L3:")
    l3_mon_features = RESCTRL_MOUNT_POINT + "/info/L3_MON/mon_features"
    l3_monitor_glob = "mon_data/mon_L3_*"

    with _managed_resctrl_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        # the L3 resource must be present in the schemata.
        try:
            with open(schemata_path, "r", encoding="utf-8") as f:
                l3_present = any(l3_line_re.match(line) for line in f)
        except OSError:
            l3_present = False
        if not l3_present:
            GLTCore.GLTTestResult.fail(
                "L3 resource not present in resctrl schemata.")

        # L3_MON must advertise the llc_occupancy feature.
        try:
            with open(l3_mon_features, "r", encoding="utf-8") as f:
                occupancy_advertised = any(
                    field == "llc_occupancy"
                    for line in f for field in line.split())
        except OSError:
            occupancy_advertised = False
        if not occupancy_advertised:
            GLTCore.GLTTestResult.fail(
                "L3 occupancy monitoring not advertised.")

        # Every L3 monitoring domain must expose a readable occupancy counter.
        monitor_dirs = [
            path for path in glob.glob(
                os.path.join(RESCTRL_MOUNT_POINT, l3_monitor_glob))
            if os.path.isdir(path)
        ]
        if not monitor_dirs:
            GLTCore.GLTTestResult.fail(
                "L3 occupancy advertised but no mon_L3 monitor directory found."
            )

        for monitor_dir in monitor_dirs:
            occupancy_file = os.path.join(monitor_dir, "llc_occupancy")
            if not os.path.isfile(occupancy_file):
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy file {occupancy_file} is missing."
                )

            try:
                with open(occupancy_file, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to read L3 occupancy file {occupancy_file}: {e}"
                )

            if not raw.isdigit():
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy file {occupancy_file} did not contain a "
                    f"non-negative integer (read {raw!r})."
                )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_l3_occupancy")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_l3_occupancy_workload(_, logger):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """
    Verify the ARM64 MPAM L3 cache allocation bounds occupancy under load.

    Restricts a resctrl control group to a partial L3 allocation (CBM) and
    drives a cache-filling stress-ng workload, verifying that the group's
    peak llc_occupancy stays bounded by the allocated capacity and does not
    leak past the CBM.

    JIRA: DGX-16956
    """
    workload_duration_sec = 10
    sample_interval_sec = 1
    tolerance = 0.25
    alloc_fraction = 4
    stress_workers = 4
    stress_vm_bytes = "256M"

    with _managed_resctrl_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        for cli in ("stress-ng", "numactl"):
            if not glt_utils.is_cli_available(cli):
                GLTCore.GLTTestResult.skip(
                    f"'{cli}' CLI not available, cannot drive L3 occupancy "
                    "workload.")

        # Pick the first NUMA node that actually owns CPUs (default node 0).
        node = 0
        for cpulist in sorted(glob.glob(
                "/sys/devices/system/node/node*/cpulist")):
            try:
                with open(cpulist, "r", encoding="utf-8") as f:
                    if f.read().strip():
                        match = re.search(r"node(\d+)", cpulist)
                        if match is None:
                            continue
                        node = int(match.group(1))
                        break
            except OSError:
                continue

        group = RESCTRL_MOUNT_POINT + "/glt_l3_occ_" + str(os.getpid())
        try:
            os.mkdir(group)
        except FileExistsError:
            # Stale own-named group from a prior killed run / PID reuse.
            # Removing it auto-reparents any stray tasks; retry once.
            try:
                os.rmdir(group)
                os.mkdir(group)
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not create a clean resctrl group {group}: {e}"
                )
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Could not create resctrl group {group}: {e}"
            )

        time.sleep(1)
        proc = None
        pgid = None
        try:
            node_l3_ids = set()
            node_l3_size = None
            for cpu in _cpus_of_node(node):
                for idx in glob.glob(
                        f"/sys/devices/system/cpu/cpu{cpu}/cache/index*"):
                    try:
                        with open(idx + "/level", encoding="utf-8") as f:
                            if f.read().strip() != "3":
                                continue
                        with open(idx + "/id", encoding="utf-8") as f:
                            node_l3_ids.add(int(f.read().strip()))
                        if node_l3_size is None:
                            with open(idx + "/size", encoding="utf-8") as f:
                                node_l3_size = f.read().strip()
                    except (OSError, ValueError):
                        continue

            occupancy_files = []
            for path in sorted(glob.glob(
                    os.path.join(group, "mon_data/mon_L3_*/llc_occupancy"))):
                m = re.search(r"mon_L3_(\d+)", path)
                if m and (not node_l3_ids or int(m.group(1)) in node_l3_ids):
                    occupancy_files.append(path)
            if not occupancy_files:
                GLTCore.GLTTestResult.fail(
                    f"No L3 monitoring domain matched node {node} "
                    f"(L3 cache id(s) {sorted(node_l3_ids)}) under "
                    f"{group}/mon_data; cannot correlate occupancy with the "
                    "bound workload."
                )

            try:
                baseline = _sum_l3_occupancy(occupancy_files)
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read baseline L3 occupancy from "
                    f"{occupancy_files}: {e}"
                )

            # Derive L3 geometry so the allocation cap can be expressed in
            # bytes.
            try:
                with open(RESCTRL_MOUNT_POINT + "/info/L3/cbm_mask",
                          "r", encoding="utf-8") as f:
                    total_ways = bin(int(f.read().strip(), 16)).count("1")
                with open(RESCTRL_MOUNT_POINT + "/info/L3/min_cbm_bits",
                          "r", encoding="utf-8") as f:
                    min_cbm_bits = int(f.read().strip())
                if node_l3_size is None:
                    raise ValueError(
                        f"no level-3 cache found for CPUs of node {node}")
                size_str = node_l3_size
                unit = size_str[-1]
                multiplier = {"K": 1024, "M": 1024 ** 2,
                              "G": 1024 ** 3}.get(unit.upper())
                if multiplier is not None:
                    l3_bytes = int(size_str[:-1]) * multiplier
                else:
                    l3_bytes = int(size_str)
                if total_ways <= 0:
                    raise ValueError("non-positive way count")
                bytes_per_way = l3_bytes // total_ways
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not determine L3 geometry for allocation test: {e}"
                )


            allocated_ways = max(min_cbm_bits, total_ways // alloc_fraction)
            partial_mask_val = (1 << allocated_ways) - 1
            partial_mask = format(partial_mask_val, "x")
            with open(group + "/schemata", "r", encoding="utf-8") as f:
                l3_line = next(
                    (ln for ln in f.read().splitlines()
                     if ln.strip().startswith("L3:")), None)
            domains = [entry.split("=", 1)[0].strip()
                       for entry in l3_line.strip()[len("L3:"):].split(";")
                       if "=" in entry]
            write_line = "L3:" + ";".join(
                f"{dom}={partial_mask}" for dom in domains)
            try:
                with open(group + "/schemata", "w", encoding="utf-8") as f:
                    f.write(write_line + "\n")
                with open(group + "/schemata", "r", encoding="utf-8") as f:
                    readback = next(
                        (ln for ln in f.read().splitlines()
                         if ln.strip().startswith("L3:")), "")
                applied = [entry.split("=", 1)[1].strip()
                           for entry in readback.strip()[len("L3:"):].split(";")
                           if "=" in entry]
                # Compare numeric masks: kernels may zero-pad hex on readback
                # (e.g. write "ff", read "00ff") without changing the value.
                if (not applied
                        or any(int(m, 16) != partial_mask_val for m in applied)):
                    raise OSError(f"readback {readback.strip()!r}")
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not apply partial L3 allocation {partial_mask}: {e}"
                )

            capacity_bound = int(
                allocated_ways * bytes_per_way * (1 + tolerance))


            try:
                with open(group + "/tasks", "w", encoding="utf-8") as f:
                    f.write(f"{threading.get_native_id()}\n")
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not move the test task into {group}: {e}"
                )
            try:
                proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    [
                        "numactl", f"--cpunodebind={node}",
                        f"--membind={node}",
                        "stress-ng", "--vm", str(stress_workers),
                        "--vm-bytes", stress_vm_bytes,
                        "--vm-keep", "--timeout", f"{workload_duration_sec}s",
                    ],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                pgid = proc.pid
            finally:
                with open(RESCTRL_MOUNT_POINT + "/tasks", "w",
                          encoding="utf-8") as f:
                    f.write(f"{threading.get_native_id()}\n")

            # Sample peak occupancy live, it decays slowly after the workload
            # exits, so the meaningful value is captured during the run.
            peak = baseline
            for _i in range(workload_duration_sec + 1):
                exited = proc.poll() is not None
                try:
                    peak = max(peak, _sum_l3_occupancy(occupancy_files))
                except (OSError, ValueError):
                    pass
                if exited:
                    break
                time.sleep(sample_interval_sec)

            logger.info(
                "L3 occupancy under workload: baseline=%.1f MiB, "
                "peak=%.1f MiB, cap=%.1f MiB (%d way(s) allocated).",
                baseline / (1024 * 1024), peak / (1024 * 1024),
                capacity_bound / (1024 * 1024), allocated_ways)

            try:
                returncode = proc.wait(timeout=workload_duration_sec + 5)
            except subprocess.TimeoutExpired:
                GLTCore.GLTTestResult.fail(
                    "stress-ng workload did not exit within the expected "
                    "duration, cannot verify L3 occupancy under load."
                )
            if returncode != 0:
                GLTCore.GLTTestResult.fail(
                    f"stress-ng workload exited with status {returncode}. "
                    "Cannot verify L3 occupancy under load."
                )

            if peak <= baseline:
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy did not rise above the "
                    f"{baseline / (1024 * 1024):.1f} MiB baseline. The "
                    "workload may not have been bound to the resctrl group."
                )

            if peak > capacity_bound:
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy {peak / (1024 * 1024):.1f} MiB exceeded the "
                    f"{capacity_bound / (1024 * 1024):.1f} MiB cap for a "
                    f"{allocated_ways}-way allocation (leaked past CBM)."
                )
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            try:
                os.rmdir(group)
            except OSError as e:
                logger.warning(
                    "Failed to remove resctrl group %s during cleanup: %s",
                    group, e)


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mba_schemata(_, logger):  # pylint: disable=too-many-branches,too-many-locals
    """
    Verify the ARM64 MPAM MBA percentage control path writes and reads back.

    JIRA: DGX-16763
    """
    group = None

    with _managed_resctrl_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        try:
            info_mb = RESCTRL_MOUNT_POINT + "/info/MB"
            try:
                with open(info_mb + "/schema_format", "r", encoding="utf-8") as f:
                    schema_format = f.read().strip()
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read MBA schema_format under {info_mb}: {e}")
            if schema_format != "percentage":
                GLTCore.GLTTestResult.skip(
                    f"MBA not in percentage mode (schema_format={schema_format}).")

            try:
                with open(info_mb + "/min_bandwidth", "r", encoding="utf-8") as f:
                    min_bw = int(f.read().strip())
                with open(info_mb + "/bandwidth_gran", "r", encoding="utf-8") as f:
                    gran = int(f.read().strip())
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read MBA percentage info files under {info_mb}: {e}")
            if gran <= 0:
                GLTCore.GLTTestResult.fail(
                    f"Invalid MBA bandwidth_gran {gran} under {info_mb}; "
                    "granularity must be positive.")
            if not 0 <= min_bw <= 100:
                GLTCore.GLTTestResult.fail(
                    f"Invalid MBA min_bandwidth {min_bw} under {info_mb}. "
                    "Expected a percentage between 0 and 100.")

            # Create a private control group
            group = RESCTRL_MOUNT_POINT + "/glt_mba_" + str(os.getpid())
            try:
                os.mkdir(group)
            except FileExistsError:
                try:
                    os.rmdir(group)
                    os.mkdir(group)
                except OSError as e:
                    GLTCore.GLTTestResult.fail(
                        f"Could not create a clean resctrl group {group}: {e}"
                    )
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not create resctrl group {group}: {e}"
                )

            # Parse the MB domains from the new group's schemata.
            try:
                with open(group + "/schemata", "r", encoding="utf-8") as f:
                    mb_line = next(
                        (ln for ln in f.read().splitlines()
                         if ln.strip().startswith("MB:")), "")
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read MB schemata from {group}: {e}"
                )
            domains = [entry.split("=", 1)[0].strip()
                       for entry in mb_line.strip()[len("MB:"):].split(";")
                       if "=" in entry]
            if not domains:
                GLTCore.GLTTestResult.fail(
                    "no MB domains parsed from group schemata.")

            # Valid steps are min_bw + N*gran; pick first, middle, and last so
            # coarse granularity still exercises distinct levels (e.g. 0/50/100).
            n_steps = (100 - min_bw) // gran
            valid = [min_bw + gran * i for i in range(n_steps + 1)]
            pick = {0, len(valid) // 2, len(valid) - 1}
            levels = sorted(valid[i] for i in pick)

            for level in levels:
                write_line = "MB:" + ";".join(
                    f"{dom}={level}" for dom in domains)
                try:
                    with open(group + "/schemata", "w", encoding="utf-8") as f:
                        f.write(write_line + "\n")
                    with open(group + "/schemata", "r", encoding="utf-8") as f:
                        readback = next(
                            (ln for ln in f.read().splitlines()
                             if ln.strip().startswith("MB:")), "")
                    applied = {entry.split("=", 1)[0].strip():
                               entry.split("=", 1)[1].strip()
                               for entry in readback.strip()[len("MB:"):].split(";")
                               if "=" in entry}
                    for dom in domains:
                        if int(applied.get(dom, "-1")) != level:
                            GLTCore.GLTTestResult.fail(
                                f"MBA domain {dom}: wrote {level}, read back "
                                f"{applied.get(dom)!r} (line {readback.strip()!r})"
                            )
                except (OSError, ValueError) as e:
                    GLTCore.GLTTestResult.fail(
                        f"Could not apply MBA level {level} to {group}: {e}"
                    )

        finally:
            if group is not None:
                try:
                    os.rmdir(group)
                except OSError as e:
                    logger.warning(
                        "Failed to remove resctrl group %s during cleanup: %s",
                        group, e)
