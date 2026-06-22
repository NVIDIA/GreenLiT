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
Python script to verify the ARM64 MPAM resctrl filesystem can be mounted.
"""

import glob
import os
import platform
import re
import subprocess

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


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


def _resctrl_is_mounted() -> bool:
    """Return True if a resctrl filesystem is mounted at RESCTRL_MOUNT_POINT."""
    try:
        with open(PROC_MOUNTS, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.split()
                if (len(fields) >= 3
                        and fields[1] == RESCTRL_MOUNT_POINT
                        and fields[2] == RESCTRL_FS_TYPE):
                    return True
    except OSError:
        return False
    return False


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
    mounted_by_test = False

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

    # Already mounted: mount ability is already demonstrated, leave it as-is.
    if _resctrl_is_mounted():
        logger.info("resctrl already mounted at %s; mount ability confirmed.",
                    RESCTRL_MOUNT_POINT)
        return

    for cli in ("mount", "umount"):
        if not glt_utils.is_cli_available(cli):
            GLTCore.GLTTestResult.skip(
                f"'{cli}' CLI not available; cannot test resctrl mount.")

    try:
        subprocess.run(
            ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE, RESCTRL_MOUNT_POINT],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=False,
        )
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(
            f"Failed to mount resctrl at {RESCTRL_MOUNT_POINT}: {e}"
        )
    mounted_by_test = True

    try:
        if not _resctrl_is_mounted():
            GLTCore.GLTTestResult.fail(
                f"mount reported success but no resctrl entry for "
                f"{RESCTRL_MOUNT_POINT} found in {PROC_MOUNTS}."
            )
    finally:
        if mounted_by_test:
            try:
                subprocess.run(
                    ["umount", RESCTRL_MOUNT_POINT],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, shell=False,
                )
            except subprocess.CalledProcessError as e:
                detail = e.stderr.strip() or e.stdout.strip() or "umount failed"
                logger.warning("Failed to unmount resctrl at %s during cleanup: %s",
                               RESCTRL_MOUNT_POINT, detail)


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
    mounted_by_test = False

    if not _resctrl_is_mounted():
        for cli in ("mount", "umount"):
            if not glt_utils.is_cli_available(cli):
                GLTCore.GLTTestResult.skip(
                    f"'{cli}' CLI not available; cannot validate resctrl info."
                )
        try:
            subprocess.run(
                ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE,
                 RESCTRL_MOUNT_POINT],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, shell=False,
            )
        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to mount resctrl at {RESCTRL_MOUNT_POINT}: {e}"
            )
        mounted_by_test = True

    try:
        if not _resctrl_is_mounted():
            GLTCore.GLTTestResult.fail(
                f"mount reported success but no resctrl entry for "
                f"{RESCTRL_MOUNT_POINT} found in {PROC_MOUNTS}; "
                "cannot validate resctrl info."
            )

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

    finally:
        if mounted_by_test:
            try:
                subprocess.run(
                    ["umount", RESCTRL_MOUNT_POINT],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, shell=False,
                )
            except subprocess.CalledProcessError as e:
                detail = e.stderr.strip() or e.stdout.strip() or "umount failed"
                logger.warning("Failed to unmount resctrl at %s during cleanup: %s",
                               RESCTRL_MOUNT_POINT, detail)


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
    mounted_by_test = False

    if not _resctrl_is_mounted():
        for cli in ("mount", "umount"):
            if not glt_utils.is_cli_available(cli):
                GLTCore.GLTTestResult.skip(
                    f"'{cli}' CLI not available, cannot test resctrl schemata.")

        try:
            subprocess.run(
                ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE,
                 RESCTRL_MOUNT_POINT],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, shell=False,
            )
        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to mount resctrl at {RESCTRL_MOUNT_POINT}: {e}"
            )
        mounted_by_test = True

        if not _resctrl_is_mounted():
            GLTCore.GLTTestResult.fail(
                f"mount reported success but no resctrl entry for "
                f"{RESCTRL_MOUNT_POINT} found in {PROC_MOUNTS}; "
                "cannot test resctrl schemata."
            )

    try:
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
    finally:
        if mounted_by_test:
            try:
                subprocess.run(
                    ["umount", RESCTRL_MOUNT_POINT],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, shell=False,
                )
            except subprocess.CalledProcessError as e:
                detail = e.stderr.strip() or e.stdout.strip() or "umount failed"
                logger.warning("Failed to unmount resctrl at %s during cleanup: %s",
                               RESCTRL_MOUNT_POINT, detail)


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
    l3_occupancy_glob = "mon_data/mon_L3_*/llc_occupancy"
    fs_mounted = False

    if not _resctrl_is_mounted():
        for cli in ("mount", "umount"):
            if not glt_utils.is_cli_available(cli):
                GLTCore.GLTTestResult.skip(
                    f"'{cli}' CLI not available, cannot test L3 occupancy.")

        try:
            subprocess.run(
                ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE,
                 RESCTRL_MOUNT_POINT],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, shell=False,
            )
        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to mount resctrl at {RESCTRL_MOUNT_POINT}: {e}"
            )
        fs_mounted = True

        if not _resctrl_is_mounted():
            GLTCore.GLTTestResult.fail(
                f"mount reported success but no resctrl entry for "
                f"{RESCTRL_MOUNT_POINT} found in {PROC_MOUNTS}; "
                "cannot test L3 occupancy."
            )

    try:
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

        # a concrete L3 occupancy counter file must exist.
        matches = sorted(glob.glob(
            os.path.join(RESCTRL_MOUNT_POINT, l3_occupancy_glob)))
        occupancy_file = matches[0] if matches else None
        if occupancy_file is None:
            GLTCore.GLTTestResult.fail(
                "L3 occupancy advertised but no mon_L3 occupancy file found."
            )

        # the counter must read back as a non-negative integer.
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
    finally:
        if fs_mounted:
            try:
                subprocess.run(
                    ["umount", RESCTRL_MOUNT_POINT],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, shell=False,
                )
            except subprocess.CalledProcessError as e:
                detail = e.stderr.strip() or e.stdout.strip() or "umount failed"
                logger.warning("Failed to unmount resctrl at %s during cleanup: %s",
                               RESCTRL_MOUNT_POINT, detail)
