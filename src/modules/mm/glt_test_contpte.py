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

"""Discriminate the arm64 contpte set_access_flags() false no-op."""
import ctypes
import fcntl
import os
import platform
import zlib

from common import glt_core as GLTCore
from common import glt_utils

CONTPTE_PROBE_MODULE = "contpte_access_flags_probe"
CONTPTE_PROBE_DEV = "/dev/glt_contpte_access_flags"
CONTPTE_FIX_SHA = "97c5550b763171dbef61e6239cab372b9f9cd4a2"
# _IOR('G', 0x51, struct glt_contpte_probe_result) with 28-byte struct
GLT_CONTPTE_IOCTL_PROBE = 0x801C4751


class ContpteProbeResult(ctypes.Structure):
    """Kernel probe output for contpte_ptep_set_access_flags discrimination."""

    _fields_ = [
        ("set_access_ret", ctypes.c_int32),
        ("target_rdonly_before", ctypes.c_uint32),
        ("slots_rdonly_after", ctypes.c_uint32),
        ("false_noop", ctypes.c_uint32),
        ("cont_ptes", ctypes.c_uint32),
        ("page_size", ctypes.c_uint32),
        ("gather_matches_entry", ctypes.c_uint32),
    ]


def _read_kernel_config_text():
    try:
        with open("/proc/config.gz", "rb") as cfg:
            return zlib.decompress(cfg.read(), 16 + zlib.MAX_WBITS).decode()
    except OSError:
        path = f"/boot/config-{platform.release()}"
        with open(path, "r", encoding="utf-8") as cfg:
            return cfg.read()


def _evaluate_probe_result(probe):
    """Return (None, None) for PASS, or ('skip'|'fail', reason)."""
    if not probe.target_rdonly_before:
        # Crafting failed to establish the writable-clean target; inconclusive.
        return (
            "skip",
            "Probe did not establish target PTE_RDONLY before "
            "set_access_flags; cannot discriminate causally.",
        )

    if not probe.gather_matches_entry:
        # Without this, an unfixed kernel can take the update path and PASS.
        return (
            "skip",
            "Gathered ptep_get() view did not match the dirtying entry "
            "before set_access_flags; buggy no-op predicate was not "
            "reachable (inconclusive, avoids false PASS).",
        )

    if probe.false_noop:
        return (
            "fail",
            "contpte_ptep_set_access_flags() falsely no-op'd while a CONT "
            "sub-PTE still had PTE_RDONLY set "
            f"(set_access_ret={probe.set_access_ret}, "
            f"slots_rdonly_after={probe.slots_rdonly_after}). "
            f"Include commit {CONTPTE_FIX_SHA} to fix it.",
        )

    if probe.slots_rdonly_after:
        return (
            "fail",
            "CONT sub-PTE(s) still have PTE_RDONLY after "
            f"set_access_flags (ret={probe.set_access_ret}, "
            f"slots_rdonly_after={probe.slots_rdonly_after}); "
            "expected the fixed kernel to clear it on every slot.",
        )

    if probe.set_access_ret != 1:
        return (
            "fail",
            "Expected set_access_flags to report a change "
            f"(ret=1), got {probe.set_access_ret}.",
        )

    return (None, None)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_contpte_set_access_flags_no_op(_, logger):  # pylint: disable=too-many-branches
    """Detect contpte set_access_flags() false no-op.

    Direct call of contpte_ptep_set_access_flags() on a crafted CONT block
    (dirty sibling, target still PTE_RDONLY). Unfixed kernels return 0 and
    leave PTE_RDONLY; fixed kernels update every slot and clear it.

    JIRA: DGX-15884
    PATCH: https://git.kernel.org/torvalds/c/97c5550b7631
    """
    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("Not ARM64; contpte path is arm64-specific.")

    try:
        kconfig_text = _read_kernel_config_text()
    except OSError as exc:
        GLTCore.GLTTestResult.skip(f"Cannot read running kernel config: {exc}")
    missing = [
        symbol
        for symbol in ("CONFIG_ARM64_CONTPTE=y", "CONFIG_ARM64=y")
        if symbol not in kconfig_text
    ]
    if missing:
        GLTCore.GLTTestResult.skip(
            "Required kernel config not enabled: " + ", ".join(missing)
        )

    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            GLTCore.GLTTestResult.skip(
                f"{command} is unavailable; cannot build or load the contpte probe."
            )

    if not os.path.isdir(f"/lib/modules/{platform.release()}/build"):
        GLTCore.GLTTestResult.skip(
            f"Kernel headers/build tree missing for {platform.release()}."
        )

    module_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "contpte_access_flags",
    )
    if not os.path.isdir(module_dir):
        GLTCore.GLTTestResult.skip(
            f"contpte probe sources missing at {module_dir}; "
            "package data may omit src/modules/mm/contpte_access_flags/*."
        )

    module_loaded = False

    with glt_utils.chdir_managed(module_dir):
        try:
            try:
                glt_utils.run_command(
                    ["make"],
                    "Cannot build contpte access-flags probe module",
                    timeout=120,
                )
            except RuntimeError as error:
                GLTCore.GLTTestResult.skip(str(error))

            if not glt_utils.sign_kernel_module(f"{CONTPTE_PROBE_MODULE}.ko", logger):
                GLTCore.GLTTestResult.fail(
                    f"Cannot sign {CONTPTE_PROBE_MODULE}.ko for Secure Boot. "
                    "See docs/user-guides/secure-boot.md for setup."
                )

            try:
                glt_utils.run_command(
                    ["insmod", f"{CONTPTE_PROBE_MODULE}.ko"],
                    "Cannot load contpte access-flags probe module",
                )
            except RuntimeError as error:
                GLTCore.GLTTestResult.skip(str(error))
            module_loaded = True

            if not os.path.exists(CONTPTE_PROBE_DEV):
                GLTCore.GLTTestResult.fail(
                    f"{CONTPTE_PROBE_DEV} was not created after insmod."
                )

            with open(CONTPTE_PROBE_DEV, "r+b", buffering=0) as dev:
                probe = ContpteProbeResult()
                try:
                    fcntl.ioctl(dev.fileno(), GLT_CONTPTE_IOCTL_PROBE, probe)
                except OSError as exc:
                    # Ioctl failures are harness/setup problems, not the
                    # regression under test (see cross-review DISP-003).
                    GLTCore.GLTTestResult.skip(
                        "contpte probe ioctl failed (inconclusive): "
                        f"{exc}."
                    )

                logger.info(
                    "contpte probe: set_access_ret=%s rdonly_before=%s "
                    "slots_rdonly_after=%s false_noop=%s cont_ptes=%s "
                    "page_size=%s gather_matches_entry=%s",
                    probe.set_access_ret,
                    probe.target_rdonly_before,
                    probe.slots_rdonly_after,
                    probe.false_noop,
                    probe.cont_ptes,
                    probe.page_size,
                    probe.gather_matches_entry,
                )
                outcome, reason = _evaluate_probe_result(probe)
                if outcome == "skip":
                    GLTCore.GLTTestResult.skip(reason)
                if outcome == "fail":
                    GLTCore.GLTTestResult.fail(reason)
        finally:
            if module_loaded:
                try:
                    glt_utils.run_command(
                        ["rmmod", CONTPTE_PROBE_MODULE],
                        f"Failed to unload {CONTPTE_PROBE_MODULE}",
                    )
                except RuntimeError as error:
                    logger.warning("%s", error)
            try:
                glt_utils.run_command(
                    ["make", "clean"],
                    "Failed to clean contpte access-flags probe build",
                )
            except RuntimeError as error:
                logger.warning("%s", error)
