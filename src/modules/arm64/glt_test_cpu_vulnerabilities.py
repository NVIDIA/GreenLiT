# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test CPU vulnerabilities.
"""
import os
from common import glt_core as GLTCore
from common.nv_device import NvDev

VULN_SYSFS_DIR = "/sys/devices/system/cpu/vulnerabilities"

CPU_VULNERABILITY_EXPECTED_STATES = {
    "vera": {
        "spec_rstack_overflow": "Not affected",
        "spec_store_bypass": "Mitigation: Speculative Store Bypass disabled via prctl",
        "spectre_v1": "Mitigation: __user pointer sanitization",
        "spectre_v2": "Mitigation: CSV2, BHB",
    },
}


def _read_vuln_file(name: str) -> tuple[str, str]:
    """
    Read /sys/devices/system/cpu/vulnerabilities/<name>.

    Returns:
        (content, error) tuple. On success, error is empty. On failure,
        content is empty and error describes the problem.
    """
    path = os.path.join(VULN_SYSFS_DIR, name)

    if not os.path.exists(path):
        return ("", f"The vulnerabilities file {name} is not found.")

    try:
        with open(path, "r", encoding="ascii") as f:
            return (f.read().strip(), "")
    except (IOError, OSError) as e:
        return ("", f"Failed to read {path}: {e}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_spectre_mitigation(_, logger):
    """Test if the Vera CPU's Spectre vulnerability state."""
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The cpu is not Vera.")

    if not os.path.isdir(VULN_SYSFS_DIR):
        logger.error("Spectre mitigation test: vulnerabilities sysfs directory not found.")
        GLTCore.GLTTestResult.fail("The vulnerabilities sysfs directory is not found.")

    expected_state = CPU_VULNERABILITY_EXPECTED_STATES["vera"]
    failures = []

    for filename, expected_value in expected_state.items():
        path = os.path.join(VULN_SYSFS_DIR, filename)

        if not os.path.exists(path):
            logger.error("Spectre mitigation test: vulnerabilities file %s not found.", filename)
            failures.append(("missing_file", filename, None, None))
            continue

        value, error_msg = _read_vuln_file(filename)
        if error_msg:
            logger.error("Spectre mitigation test: %s", error_msg)
            failures.append(("read_error", filename, None, error_msg))
            continue

        if value != expected_value:
            logger.error(
                "Spectre mitigation test: %s expected '%s', got '%s'.",
                filename,
                expected_value,
                value,
            )
            failures.append(("mismatch", filename, expected_value, value))

    if failures:
        # Build a single-space-separated message; no newlines.
        fail_msg = ["Spectre mitigation test failed."]
        for kind, fname, expected, detail in failures:
            if kind == "missing_file":
                fail_msg.append(f"The vulnerabilities file {fname} is not found.")
            elif kind == "read_error":
                fail_msg.append(detail)
            elif kind == "mismatch":
                fail_msg.append(
                    f"The vulnerability {fname} is not in the expected state."
                    f" Expected '{expected}', got '{detail}'."
                )

        GLTCore.GLTTestResult.fail(" ".join(fail_msg))
