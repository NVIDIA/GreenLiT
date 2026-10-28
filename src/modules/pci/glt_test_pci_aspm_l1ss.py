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

"""Validate paired PCIe L1SS saved-state capture using software sentinels."""

import errno
import platform
import subprocess
import sys
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils, pci_utils
from scripts import check_kernel_config


L1SS_PROBE_MODULE = "aspm_l1ss_saved_state_probe"
L1SS_PROBE_PROC = Path("/proc/greenlit_pci_aspm_l1ss_saved_state")
L1SS_PROBE_SYSFS = Path("/sys/module") / L1SS_PROBE_MODULE
L1SS_PROBE_DIR = Path(__file__).resolve().parent / L1SS_PROBE_MODULE
L1SS_MODES = {"parent_unsaved": "1db806ec06b7", "parent_saved_first": "7507eb3e7bfa"}
L1SS_REQUIRED_CONFIGS = ("CONFIG_PCIEASPM", "CONFIG_MODULES", "CONFIG_MODULE_UNLOAD")
L1SS_REGISTER_FIELDS = tuple(
    f"{device}_ctl{control}_{phase}"
    for device in ("child", "parent")
    for control in (1, 2)
    for phase in ("before", "saved", "after")
)
L1SS_FIELDS = {"mode", "child", "parent", "capture_error", "restore_error", *L1SS_REGISTER_FIELDS}


def _parse_l1ss_result(content):
    """Read the helper's raw observations, rejecting malformed or incomplete output."""
    result = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in L1SS_FIELDS or key in result:
            raise ValueError("Malformed, unknown or duplicate PCI L1SS probe field.")
        if key not in ("mode", "child", "parent"):
            value = int(value, 0)
            if key in L1SS_REGISTER_FIELDS and not 0 <= value <= 0xffffffff:
                raise ValueError(f"PCI L1SS register {key} is outside its valid range.")
            if key in ("capture_error", "restore_error") and not -(2**31) <= value < 2**31:
                raise ValueError(f"PCI L1SS error {key} is outside its valid range.")
        result[key] = value
    if set(result) != L1SS_FIELDS:
        raise ValueError("PCI L1SS probe result is missing required raw observations.")
    return result


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_pci_aspm_l1ss_save_parent(_, logger):  # pylint: disable=too-many-branches
    """Refresh both L1SS buffers with the parent initially unsaved and already saved.

    JIRA: DGX-13486
    PATCH: https://git.kernel.org/torvalds/c/1db806ec06b7 (upstream)
    PATCH: https://git.kernel.org/torvalds/c/7507eb3e7bfa (upstream)
    """
    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            GLTCore.GLTTestResult.skip(f"{command} is unavailable for PCI L1SS validation.")
    try:
        config = check_kernel_config.parse_kconfig(f"/boot/config-{platform.release()}")
    except OSError as error:
        GLTCore.GLTTestResult.skip(f"Cannot read running kernel configuration: {error}")
    missing = [name for name in L1SS_REQUIRED_CONFIGS if config.get(name) not in ("y", "m")]
    if missing:
        GLTCore.GLTTestResult.skip("Required kernel configuration not enabled: " + ", ".join(missing))
    if not Path(f"/lib/modules/{platform.release()}/build").is_dir():
        GLTCore.GLTTestResult.skip(
            f"Matching kernel headers/build tree is unavailable for {platform.release()}.")
    for source in ("Makefile", f"{L1SS_PROBE_MODULE}.c"):
        if not (L1SS_PROBE_DIR / source).is_file():
            GLTCore.GLTTestResult.fail(f"Packaged PCI L1SS source is missing: {source}.")
    if L1SS_PROBE_SYSFS.exists() or L1SS_PROBE_PROC.exists():
        GLTCore.GLTTestResult.fail("PCI L1SS helper is already present; refusing to reuse or remove it.")

    responsible = False
    with glt_utils.chdir_managed(L1SS_PROBE_DIR):
        try:
            try:
                glt_utils.run_command(["make"], "Failed to build PCI L1SS helper", timeout=120)
                if not glt_utils.sign_kernel_module(f"{L1SS_PROBE_MODULE}.ko", logger):
                    raise RuntimeError("Cannot sign PCI L1SS helper; see docs/user-guides/secure-boot.md.")
            except (OSError, RuntimeError, UnicodeError) as error:
                GLTCore.GLTTestResult.warn(f"PCI L1SS helper setup incomplete: {error}")

            responsible = True
            try:
                subprocess.run(["insmod", f"{L1SS_PROBE_MODULE}.ko"],
                               check=True, capture_output=True, text=True, timeout=30)
            except (subprocess.CalledProcessError, OSError) as error:
                # A signaled loader may already have inserted the module.
                responsible = isinstance(error, subprocess.CalledProcessError) and error.returncode < 0
                GLTCore.GLTTestResult.warn(f"PCI L1SS helper load failed: {error}")
            except (subprocess.TimeoutExpired, UnicodeError) as error:
                GLTCore.GLTTestResult.warn(f"PCI L1SS helper load is uncertain: {error}")

            failures = []
            for mode, commit in L1SS_MODES.items():
                logger.info("Checking PCI L1SS saved state: %s", mode)
                run_error = None
                try:
                    glt_utils.write_sysfs_value(L1SS_PROBE_PROC, mode)
                except OSError as error:
                    run_error = error
                try:
                    content = glt_utils.read_sysfs_value(L1SS_PROBE_PROC)
                    logger.info("PCI L1SS probe (%s):\n%s", mode, content.rstrip())
                    result = _parse_l1ss_result(content)
                except (OSError, UnicodeError, ValueError) as error:
                    GLTCore.GLTTestResult.fail(f"Invalid PCI L1SS probe report ({mode}): {error}")
                if result["mode"] != mode:
                    GLTCore.GLTTestResult.fail(f"Unexpected PCI L1SS probe mode ({mode}).")
                if result["restore_error"]:
                    GLTCore.GLTTestResult.fail(
                        f"PCI L1SS helper failed to restore original software caches ({mode}): "
                        f"{result['restore_error']}.")
                if run_error:
                    if failures:
                        GLTCore.GLTTestResult.fail(
                            "; ".join(failures) + f"; PCI L1SS capture failed ({mode}): {run_error}")
                    if run_error.errno in (errno.ENODEV, errno.EOPNOTSUPP, errno.EBUSY):
                        GLTCore.GLTTestResult.skip(f"No eligible PCI L1SS pair ({mode}): {run_error}")
                    if run_error.errno == errno.ENOMEM:
                        GLTCore.GLTTestResult.warn(f"PCI L1SS snapshot allocation failed ({mode}): {run_error}")
                    GLTCore.GLTTestResult.fail(f"PCI L1SS capture failed ({mode}): {run_error}")
                if result["capture_error"]:
                    GLTCore.GLTTestResult.fail(
                        f"PCI L1SS capture failed ({mode}): {result['capture_error']}.")
                if (not pci_utils.BDF_STRICT_RE.fullmatch(result["child"])
                        or not pci_utils.BDF_STRICT_RE.fullmatch(result["parent"])
                        or result["child"] == result["parent"]):
                    GLTCore.GLTTestResult.fail(f"Invalid PCI L1SS child/parent pair ({mode}).")
                mismatches = []
                for device in ("child", "parent"):
                    for control in (1, 2):
                        prefix = f"{device}_ctl{control}"
                        before = result[f"{prefix}_before"]
                        if before == 0xffffffff or result[f"{prefix}_after"] != before:
                            GLTCore.GLTTestResult.fail(
                                f"PCI L1SS {prefix} is unreadable or changed during capture ({mode}).")
                        saved = result[f"{prefix}_saved"]
                        if saved != before:
                            mismatches.append(
                                f"{mode}: PCI L1SS {prefix} saved {saved:#010x} instead of {before:#010x}; "
                                f"check for the behavior fixed by upstream commit {commit} "
                                "or an equivalent backport.")
                if mismatches:
                    failures.extend(mismatches)
                    logger.error("%s", "; ".join(mismatches))
                else:
                    logger.info("Saving %s refreshed both L1SS buffers (%s).", result["child"], mode)
            if failures:
                GLTCore.GLTTestResult.fail("; ".join(failures))
        finally:
            primary = sys.exc_info()[1]
            cleanup_errors = []
            if responsible and (L1SS_PROBE_SYSFS.exists() or L1SS_PROBE_PROC.exists()):
                try:
                    glt_utils.run_command(["rmmod", L1SS_PROBE_MODULE], "Failed to unload PCI L1SS helper")
                    if L1SS_PROBE_SYSFS.exists() or L1SS_PROBE_PROC.exists():
                        raise RuntimeError("PCI L1SS helper paths remain after removal.")
                except (OSError, RuntimeError) as error:
                    cleanup_errors.append(str(error))
            try:
                glt_utils.run_command(["make", "clean"], "Failed to clean PCI L1SS helper", timeout=120)
            except (OSError, RuntimeError) as error:
                cleanup_errors.append(str(error))
            if cleanup_errors:
                prefix = f"{primary}; " if primary else ""
                GLTCore.GLTTestResult.fail(prefix + "Cleanup failed: " + "; ".join(cleanup_errors))
