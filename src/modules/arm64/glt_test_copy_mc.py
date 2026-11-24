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

"""Verify Vera copy_mc and ordinary copy behavior through the memcpy KUnit suite."""

import platform
import re
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config

KUNIT_RESULTS_PATH = Path("/sys/kernel/debug/kunit/memcpy/results")
KUNIT_MODULE_STATE_PATH = Path("/sys/module/memcpy_kunit")
KUNIT_MODULE_NAME = "memcpy_kunit"
KUNIT_LOAD_TIMEOUT_SECONDS = 120
KUNIT_UNLOAD_TIMEOUT_SECONDS = 30
REQUIRED_KUNIT_CASES = (
    "memset_test",
    "memcpy_test",
    "memcpy_large_test",
    "memmove_test",
    "memmove_large_test",
    "memmove_overlap_test",
    "copy_page_test",
    "copy_mc_page_test",
    "memcpy_mc_test",
    "memcpy_mc_large_test",
)
KUNIT_CASE_RE = re.compile(
    r"^ {4}(?P<status>ok|not ok)[ \t]+(?P<number>[0-9]+)[ \t]+"
    r"(?P<name>[A-Za-z0-9_.-]+)(?P<skip>[ \t]+#[ \t]*SKIP\b.*)?[ \t]*$"
)
KUNIT_PLAN_RE = re.compile(r"^ {4}1\.\.(?P<count>[0-9]+)[ \t]*$")
KUNIT_SUITE_RE = re.compile(
    r"^(?P<status>ok|not ok)[ \t]+1[ \t]+memcpy(?:[ \t]+#.*)?[ \t]*$"
)


def _config_skip_reason(kconfig_path):
    try:
        kconfig = check_kernel_config.parse_kconfig(str(kconfig_path))
    except (OSError, ValueError) as err:
        return f"Cannot inspect the running kernel configuration: {err}"

    missing = []
    if kconfig.get("CONFIG_ARCH_HAS_COPY_MC") != "y":
        missing.append("CONFIG_ARCH_HAS_COPY_MC=y")
    if kconfig.get("CONFIG_KUNIT") not in ("y", "m"):
        missing.append("CONFIG_KUNIT=y or m")
    if kconfig.get("CONFIG_KUNIT_DEBUGFS") != "y":
        missing.append("CONFIG_KUNIT_DEBUGFS=y")
    if kconfig.get("CONFIG_MEMCPY_KUNIT_TEST") not in ("y", "m"):
        missing.append("CONFIG_MEMCPY_KUNIT_TEST=y or m")
    if missing:
        return "copy_mc KUnit requires " + ", ".join(missing)
    return ""


def _parse_memcpy_results(results_text):
    lines = results_text.splitlines()
    if len(lines) < 5:
        raise ValueError("memcpy KUnit results are truncated")
    if lines[0] != "KTAP version 1" or lines[1] != "1..1":
        raise ValueError("memcpy KUnit results have no valid outer KTAP header")
    if lines[2] != "    KTAP version 1":
        raise ValueError("memcpy KUnit suite has no nested KTAP header")

    marker_indexes = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "# Subtest: memcpy"
    ]
    if (
        len(marker_indexes) != 1
        or marker_indexes[0] != 3
        or lines[3] != "    # Subtest: memcpy"
    ):
        raise ValueError("Expected exactly one '# Subtest: memcpy' marker")
    marker_index = marker_indexes[0]

    plan_count = None
    cases = []
    for line in lines[marker_index + 1:]:
        if line.strip() == "" or line.lstrip().startswith("#"):
            continue
        if KUNIT_SUITE_RE.fullmatch(line):
            continue
        plan_match = KUNIT_PLAN_RE.fullmatch(line)
        if plan_match:
            if plan_count is not None or cases:
                raise ValueError("memcpy KUnit suite has an unexpected extra plan")
            plan_count = int(plan_match.group("count"))
            continue
        case_match = KUNIT_CASE_RE.fullmatch(line)
        if case_match:
            if plan_count is None:
                raise ValueError("memcpy KUnit case appears before the test plan")
            cases.append({
                "number": int(case_match.group("number")),
                "name": case_match.group("name"),
                "failed": case_match.group("status") == "not ok",
                "skipped": case_match.group("skip") is not None,
            })
            continue
        raise ValueError(f"Unexpected memcpy KUnit record: {line!r}")

    if not plan_count:
        raise ValueError("memcpy KUnit suite has no valid test plan")

    if len(cases) != plan_count:
        raise ValueError(
            f"memcpy KUnit plan has {plan_count} cases but {len(cases)} results"
        )
    for expected_number, case in enumerate(cases, start=1):
        if case["number"] != expected_number:
            raise ValueError("memcpy KUnit results are not numbered consecutively")
    case_names = [case["name"] for case in cases]
    if len(set(case_names)) != len(case_names):
        raise ValueError("memcpy KUnit results contain duplicate case names")

    suite_matches = [KUNIT_SUITE_RE.fullmatch(line) for line in lines]
    suite_matches = [match for match in suite_matches if match]
    if len(suite_matches) != 1:
        raise ValueError("memcpy KUnit suite has no unique final result")
    nonempty_lines = [line for line in lines if line.strip()]
    if not nonempty_lines or not KUNIT_SUITE_RE.fullmatch(nonempty_lines[-1]):
        raise ValueError("memcpy KUnit suite result is not the final record")

    failed = [case["name"] for case in cases if case["failed"]]
    skipped = [case["name"] for case in cases if case["skipped"]]
    missing = [name for name in REQUIRED_KUNIT_CASES if name not in case_names]
    if suite_matches[0].group("status") == "not ok":
        failed.append("memcpy (suite)")
    return failed, skipped, missing


def _run_memcpy_kunit(kconfig_path=None, results_path=KUNIT_RESULTS_PATH,
                      module_state_path=KUNIT_MODULE_STATE_PATH):
    if kconfig_path is None:
        kconfig_path = Path("/boot") / f"config-{platform.release()}"
    results_path = Path(results_path)
    module_state_path = Path(module_state_path)

    skip_reason = _config_skip_reason(kconfig_path)
    if skip_reason:
        return "skip", skip_reason

    kconfig = check_kernel_config.parse_kconfig(str(kconfig_path))
    loaded_by_test = False
    outcome = "pass"
    reason = f"All {len(REQUIRED_KUNIT_CASES)} memcpy/copy_mc KUnit cases passed"

    try:
        if not results_path.is_file():
            if kconfig["CONFIG_MEMCPY_KUNIT_TEST"] != "m":
                outcome = "skip"
                reason = (
                    f"KUnit results are unavailable at {results_path}; "
                    "the built-in suite did not produce a readable result"
                )
            elif not glt_utils.is_cli_available("modprobe"):
                outcome = "skip"
                reason = "modprobe is required to load memcpy_kunit"
            elif not glt_utils.is_cli_available("rmmod"):
                outcome = "skip"
                reason = "rmmod is required to clean up memcpy_kunit"
            else:
                was_loaded = module_state_path.exists()
                try:
                    glt_utils.run_command(
                        ["modprobe", KUNIT_MODULE_NAME],
                        "Failed to load memcpy_kunit",
                        timeout=KUNIT_LOAD_TIMEOUT_SECONDS,
                    )
                    loaded_by_test = not was_loaded
                except RuntimeError as err:
                    loaded_by_test = not was_loaded and module_state_path.exists()
                    outcome = "skip"
                    reason = str(err)

        if outcome == "pass" or results_path.is_file():
            try:
                results_text = results_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as err:
                outcome = "fail"
                reason = f"Invalid memcpy KUnit results: not valid UTF-8: {err}"
            except OSError as err:
                outcome = "skip"
                reason = f"Cannot read KUnit results from {results_path}: {err}"
            else:
                try:
                    failed, skipped, missing = _parse_memcpy_results(results_text)
                except ValueError as err:
                    outcome = "fail"
                    reason = f"Invalid memcpy KUnit results: {err}"
                else:
                    if failed:
                        outcome = "fail"
                        reason = "memcpy KUnit failures: " + ", ".join(failed)
                    elif skipped or missing:
                        outcome = "skip"
                        reason_parts = []
                        if skipped:
                            reason_parts.append("skipped: " + ", ".join(skipped))
                        if missing:
                            reason_parts.append("missing: " + ", ".join(missing))
                        reason = "Incomplete copy_mc KUnit coverage; " + "; ".join(reason_parts)
    finally:
        if loaded_by_test:
            try:
                glt_utils.run_command(
                    ["rmmod", KUNIT_MODULE_NAME],
                    "Failed to unload memcpy_kunit",
                    timeout=KUNIT_UNLOAD_TIMEOUT_SECONDS,
                )
            except RuntimeError as err:
                if outcome == "fail":
                    reason = f"{reason}; cleanup also failed: {err}"
                else:
                    outcome = "fail"
                    reason = f"memcpy_kunit cleanup failed: {err}"

    return outcome, reason


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_copy_mc_memcpy_kunit(_, logger):
    """Validate copy_mc via v14 memcpy KUnit (https://lkml.iu.edu/2605.2/02391.html)."""
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("copy_mc KUnit validation is Vera-specific")

    outcome, reason = _run_memcpy_kunit()
    if outcome == "fail":
        GLTCore.GLTTestResult.fail(reason)
    if outcome == "skip":
        GLTCore.GLTTestResult.skip(reason)
    logger.info("%s", reason)
