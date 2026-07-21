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

"""Verify that Vera exposes the complete LS64 userspace reporting ABI."""

import ctypes
import os

from common import glt_core as GLTCore
from common.nv_device import NvDev


AT_HWCAP3 = 29
HWCAP3_LS64 = 1 << 3

CPUINFO_PATH = "/proc/cpuinfo"
LS64_CPUINFO_FEATURE = "ls64"
LS64_USERSPACE_PATCH = "58ce78667a64"
LS64_PATCH_SERIES = (
    "https://lore.kernel.org/all/"
    "20260119022928.149358-1-wangzhou1@hisilicon.com/"
)


def _read_cpu_features(cpuinfo_path=CPUINFO_PATH):
    """Return the union of feature flags exposed by /proc/cpuinfo."""
    features = set()

    with open(cpuinfo_path, "r", encoding="ascii") as cpuinfo:
        for line in cpuinfo:
            key, _, value = line.partition(":")
            if key.strip().lower() == "features":
                features.update(value.strip().lower().split())

    return features


def _getauxval(aux_type):
    """Return getauxval(aux_type), preserving errno for unsupported aux keys."""
    libc = ctypes.CDLL(None, use_errno=True)
    getauxval = libc.getauxval
    getauxval.argtypes = [ctypes.c_ulong]
    getauxval.restype = ctypes.c_ulong

    ctypes.set_errno(0)
    value = getauxval(aux_type)
    err = ctypes.get_errno()
    if value == 0 and err:
        raise OSError(err, os.strerror(err))

    return value


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_ls64_userspace_abi(_, logger):
    """
    Verify Vera exposes both FEAT_LS64 userspace reporting interfaces.

    JIRA: DGX-16333
    PATCH SERIES:
        https://lore.kernel.org/all/20260119022928.149358-1-wangzhou1@hisilicon.com/
    NATIVE ABI PATCH:
        https://git.kernel.org/torvalds/c/58ce78667a64
    EXTERNAL KVM PATCHES:
        https://git.kernel.org/torvalds/c/f174a9ffcd48
        https://git.kernel.org/torvalds/c/151b92c92a45
    REFERENCE TEST:
        https://git.kernel.org/torvalds/c/57a96356bb69

    This native test checks both userspace interfaces added by 58ce78667a64.
    The companion external LS64 executable opens /dev/kvm, creates an EL1 guest,
    and covers the outside-memslot KVM_EXIT_ARM_LDST64B behavior. It does not
    cover the valid-memslot exception injection added by 2937aeec9dc5.
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    try:
        cpu_features = _read_cpu_features()
    except OSError as error:
        GLTCore.GLTTestResult.fail(f"Failed to read {CPUINFO_PATH}: {error}.")
    else:
        if LS64_CPUINFO_FEATURE not in cpu_features:
            GLTCore.GLTTestResult.fail(
                f"{CPUINFO_PATH} does not expose the "
                f"'{LS64_CPUINFO_FEATURE}' feature; consider checking whether "
                f"the kernel is missing {LS64_USERSPACE_PATCH} from "
                f"{LS64_PATCH_SERIES}."
            )

    try:
        hwcap3 = _getauxval(AT_HWCAP3)
    except OSError as error:
        GLTCore.GLTTestResult.fail(f"Failed to read AT_HWCAP3: {error}.")
    else:
        if not hwcap3 & HWCAP3_LS64:
            GLTCore.GLTTestResult.fail(
                "AT_HWCAP3 does not expose HWCAP3_LS64; consider checking "
                f"whether the kernel is missing {LS64_USERSPACE_PATCH} from "
                f"{LS64_PATCH_SERIES}."
            )

    logger.info("Vera LS64 userspace ABI checks passed.")
