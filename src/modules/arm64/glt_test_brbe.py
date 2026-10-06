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

"""Tests for the Arm Branch Record Buffer Extension (BRBE)."""

import os
import platform
import re
import subprocess

from common import glt_core as GLTCore
from common import perf_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


BRBE_RECORD_TIMEOUT_SECONDS = 30
BRBE_DECODE_TIMEOUT_SECONDS = 30
ARM64_KERNEL_ADDRESS_MIN = 1 << 63
BRBE_STACK_ENTRY_PATTERN = re.compile(
    r"\b(?P<source>0x[0-9a-f]+)/(?P<target>0x[0-9a-f]+)/"
    r"(?:[^/\s]+/){4}(?P<branch_type>[a-z][a-z0-9_]*)/[^/\s]+"
    r"(?=\s|$)",
    re.IGNORECASE,
)


def _parse_brbe_stack_entries(
    perf_script_output: str,
) -> list[tuple[int, int, str]]:
    """Return typed user-space BRBE source/target entries."""
    entries = []
    for match in BRBE_STACK_ENTRY_PATTERN.finditer(perf_script_output):
        source_address = int(match.group("source"), 16)
        target_address = int(match.group("target"), 16)
        branch_type = match.group("branch_type").upper()
        if source_address >= ARM64_KERNEL_ADDRESS_MIN or \
                target_address >= ARM64_KERNEL_ADDRESS_MIN:
            raise ValueError(
                "Decoded BRBE stack contains a kernel-space address despite "
                "user-only branch filtering: "
                f"source={source_address:#x}, target={target_address:#x}, "
                f"type={branch_type}"
            )
        entries.append((source_address, target_address, branch_type))
    return entries


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_brbe_branch_stack(sd, logger):
    """Verify Vera host FEAT_BRBE support and perf branch-stack sampling.

    This is functional host-side coverage, not proof of exact Git commit
    provenance, and it does not exercise nVHE guest isolation.

    PATCH: https://git.kernel.org/torvalds/c/52e4a56ab8b8
    PATCH: https://git.kernel.org/torvalds/c/ae344bcb0d49
    PATCH: https://git.kernel.org/torvalds/c/58074a0fce66
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not an NVIDIA Vera CPU; BRBE validation is Vera-specific.")

    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except (OSError, ValueError) as err:
        GLTCore.GLTTestResult.fail(f"Unable to inspect kernel config {kconfig_path}: {err}")
    if kconfig.get("CONFIG_ARM64_BRBE") != "y":
        GLTCore.GLTTestResult.fail(f"CONFIG_ARM64_BRBE is not enabled in {kconfig_path}")
    try:
        capabilities = perf_utils.read_brbe_capabilities()
    except ValueError as err:
        GLTCore.GLTTestResult.fail(str(err))
    logger.info("Detected BRBE branch-record capabilities: %s", capabilities)

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)
    logger.info(
        "Using perf command for kernel %s: %s", platform.uname().release, perf_cmd
    )

    perf_data_path = os.path.join(sd.log_dir, "test_brbe_branch_stack.perf.data")
    record_command = [
        perf_cmd, "record", "-o", perf_data_path, "--no-buildid",
        "-e", "cycles:u", "-c", "10000",
        "--branch-filter", "any,save_type,u",
        "--", "dd", "if=/dev/zero", "of=/dev/null", "bs=64M", "count=8",
    ]
    perf_commands = (
        ("record", record_command, BRBE_RECORD_TIMEOUT_SECONDS),
        (
            "decode", [perf_cmd, "script", "-i", perf_data_path, "--fields", "brstack"],
            BRBE_DECODE_TIMEOUT_SECONDS,
        ),
    )

    results = {}
    for phase, command, timeout in perf_commands:
        try:
            result = subprocess.run(
                command, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            GLTCore.GLTTestResult.fail(f"perf BRBE {phase} timed out after {timeout} seconds")
        except OSError as err:
            GLTCore.GLTTestResult.fail(f"perf BRBE {phase} failed to start: {err}")
        if result.returncode != 0:
            GLTCore.GLTTestResult.fail(f"perf BRBE {phase} failed: returncode={result.returncode} "
                f"stderr={(result.stderr or '').strip()}")
        results[phase] = result

    try:
        branch_entries = _parse_brbe_stack_entries(results["decode"].stdout or "")
    except ValueError as err:
        GLTCore.GLTTestResult.fail(str(err))
    if not branch_entries:
        GLTCore.GLTTestResult.fail("perf BRBE decode produced no typed user-space branch-stack "
            "source/target entries")
    logger.info(
        "Decoded %d BRBE branch-stack entries from %s",
        len(branch_entries), perf_data_path,
    )
