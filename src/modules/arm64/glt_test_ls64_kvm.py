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
import subprocess
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common import virt as glt_virt
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
LS64_PROBE_BUILD_TIMEOUT_SECONDS = 60
LS64_PROBE_RUN_TIMEOUT_SECONDS = 10
LS64_PROBE_SIGBUS = 2
LS64_PROBE_SIGILL = 3
LS64_PROBE_SOURCE_DIR = Path(__file__).resolve().parent / "ls64"
LS64_PROBE_MAKEFILE = LS64_PROBE_SOURCE_DIR / "Makefile"
KVM_MICROVM_SOURCE_DIR = (
    Path(glt_virt.__file__).resolve().parent / "kvm_microvm"
)
LS64_PROBE_REQUIRED_FILES = (
    LS64_PROBE_MAKEFILE,
    LS64_PROBE_SOURCE_DIR / "ls64_probe.c",
    KVM_MICROVM_SOURCE_DIR / "glt_arm64_kvm.c",
    KVM_MICROVM_SOURCE_DIR / "glt_arm64_kvm.h",
)


def _execute_ls64_probe(sd, test_name, mode, logger):
    """Build, run, and remove one LS64 probe executable."""
    probe_path = Path(sd.log_dir) / f"{test_name}.probe"
    result = None
    failure_reason = ""

    if not glt_utils.is_cli_available("make"):
        return None, "Please install 'make' to build the LS64 probe"
    missing_files = [
        str(path) for path in LS64_PROBE_REQUIRED_FILES
        if not path.is_file()
    ]
    if missing_files:
        return None, "LS64 probe source is missing: " + ", ".join(missing_files)

    try:
        probe_path.unlink(missing_ok=True)
        build_command = [
            "make", "-C", str(LS64_PROBE_SOURCE_DIR),
            f"OUTPUT={probe_path}",
            f"KVM_MICROVM_DIR={KVM_MICROVM_SOURCE_DIR}",
        ]
        try:
            build_result = glt_utils.run_command(
                build_command,
                "LS64 probe build failed",
                timeout=LS64_PROBE_BUILD_TIMEOUT_SECONDS,
            )
        except RuntimeError as error:
            failure_reason = str(error)
        else:
            if build_result.stdout:
                logger.debug(
                    "LS64 probe build output:\n%s",
                    build_result.stdout.rstrip(),
                )
        if not failure_reason and (
            not probe_path.is_file() or not os.access(probe_path, os.X_OK)
        ):
            failure_reason = f"LS64 probe build did not create {probe_path}"
        if not failure_reason:
            result = subprocess.run(
                [str(probe_path), mode],
                check=False,
                capture_output=True,
                text=True,
                timeout=LS64_PROBE_RUN_TIMEOUT_SECONDS,
            )
            if result.stdout:
                logger.debug("LS64 %s stdout:\n%s", mode, result.stdout.rstrip())
            if result.stderr:
                logger.debug("LS64 %s stderr:\n%s", mode, result.stderr.rstrip())
    except subprocess.TimeoutExpired as error:
        failure_reason = f"LS64 {mode} operation timed out after {error.timeout} seconds"
    except OSError as error:
        failure_reason = f"Unable to execute LS64 {mode} operation: {error}"

    try:
        probe_path.unlink(missing_ok=True)
    except OSError as error:
        cleanup_reason = f"Unable to remove LS64 probe {probe_path}: {error}"
        failure_reason = (
            f"{failure_reason}; additionally, {cleanup_reason}"
            if failure_reason else cleanup_reason
        )
    return result, failure_reason


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

    This test checks the two userspace reporting interfaces added by
    58ce78667a64. Separate native cases exercise instructions and the
    outside-memslot KVM_EXIT_ARM_LDST64B behavior. They do not cover the
    valid-memslot exception injection added by 2937aeec9dc5.
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


@GLTCore.GLTRequires.must_pass("arm64", "test_vera_ls64_userspace_abi")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_ls64_normal_memory(sd, logger):
    """Verify LS64 instructions produce a valid normal-memory result.

    PATCH: https://git.kernel.org/torvalds/c/58ce78667a64
    """
    result, reason = _execute_ls64_probe(
        sd, test_vera_ls64_normal_memory.__name__, "normal-memory", logger
    )
    if reason:
        GLTCore.GLTTestResult.fail(reason)
    if result.returncode == LS64_PROBE_SIGILL:
        GLTCore.GLTTestResult.fail(
            "HWCAP3_LS64 is set but LS64 instructions raised SIGILL"
        )
    if result.returncode not in (0, LS64_PROBE_SIGBUS):
        GLTCore.GLTTestResult.fail(
            "LS64 normal-memory probe returned unexpected status "
            f"{result.returncode}: {(result.stderr or '').strip()[-1000:]}"
        )
    if result.returncode == LS64_PROBE_SIGBUS:
        logger.info(
            "LS64 instructions raised SIGBUS on normal memory; "
            "FEAT_LS64_WB is not exposed"
        )
    else:
        logger.info(
            "LS64 instructions completed on normal memory; "
            "FEAT_LS64_WB may be available"
        )


@GLTCore.GLTRequires.must_pass("arm64", "test_vera_ls64_userspace_abi")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_ls64_alignment(sd, logger):
    """Require a misaligned ST64B to raise SIGBUS when normal memory works.

    This is architectural coverage, not a discriminator for the userspace
    reporting ABI patch. That ABI is checked by test_vera_ls64_userspace_abi.
    """
    normal_result, reason = _execute_ls64_probe(
        sd, test_vera_ls64_alignment.__name__, "normal-memory", logger
    )
    if reason:
        GLTCore.GLTTestResult.fail(reason)
    if normal_result.returncode == LS64_PROBE_SIGBUS:
        GLTCore.GLTTestResult.skip(
            "FEAT_LS64_WB is unavailable, so an aligned normal-memory fault "
            "prevents isolating the misalignment behavior"
        )
    if normal_result.returncode != 0:
        GLTCore.GLTTestResult.fail(
            "LS64 alignment prerequisite returned unexpected status "
            f"{normal_result.returncode}: "
            f"{(normal_result.stderr or '').strip()[-1000:]}"
        )

    result, reason = _execute_ls64_probe(
        sd, test_vera_ls64_alignment.__name__, "misaligned", logger
    )
    if reason:
        GLTCore.GLTTestResult.fail(reason)
    if result.returncode != LS64_PROBE_SIGBUS:
        GLTCore.GLTTestResult.fail(
            "Misaligned ST64B did not raise SIGBUS: "
            f"returncode={result.returncode}, "
            f"stderr={(result.stderr or '').strip()[-1000:]}"
        )
    logger.info("Misaligned ST64B raised SIGBUS as required")


@GLTCore.GLTRequires.must_pass("arm64", "test_vera_ls64_userspace_abi")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_ls64_kvm_exit(sd, logger):
    """Verify an outside-memslot LD64B exits to userspace through KVM.

    PATCH: https://git.kernel.org/torvalds/c/f174a9ffcd48
    PATCH: https://git.kernel.org/torvalds/c/151b92c92a45
    """
    if not os.access("/dev/kvm", os.R_OK | os.W_OK):
        GLTCore.GLTTestResult.fail("Read/write access to /dev/kvm is required")

    result, reason = _execute_ls64_probe(
        sd, test_vera_ls64_kvm_exit.__name__, "kvm-exit", logger
    )
    if reason:
        GLTCore.GLTTestResult.fail(reason)
    if result.returncode != 0:
        GLTCore.GLTTestResult.fail(
            f"LS64 KVM probe failed: returncode={result.returncode}, "
            f"stderr={(result.stderr or '').strip()}"
        )
    logger.info("LS64 guest produced the expected KVM_EXIT_ARM_LDST64B")
