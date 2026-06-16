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

"""Helpers for running perf selftests across perf versions."""

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional


class PerfTestStatus(Enum):
    """Normalized result for perf selftest execution."""

    FAIL = "fail"
    NO_MATCH = "no_match"
    PASS = "pass"


@dataclass(frozen=True)
class PerfTestResult:
    """Normalized perf selftest result."""

    status: PerfTestStatus
    reason: str
    test_name: Optional[str] = None


def _extract_perf_test_names(perf_test_list_output: str) -> set[str]:
    """Return perf selftest names parsed from `perf test list` output."""
    names = set()
    for line in perf_test_list_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if ":" in stripped:
            names.add(stripped.split(":", 1)[1].strip())
        else:
            names.add(stripped)

    return names


def _candidate_matches(candidate: str, available_tests: set[str]) -> bool:
    """Return True when candidate matches perf's substring selection behavior."""
    normalized_candidate = candidate.casefold()
    return any(normalized_candidate in test_name.casefold()
               for test_name in available_tests)


def _has_ok_line(output: str) -> bool:
    """Return True when perf output contains a passing selftest verdict."""
    return any(line.rstrip().endswith(": Ok") for line in output.splitlines())


def _summarize_output(output: str) -> str:
    """Return a compact failure summary from verbose perf output."""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return "\n".join(lines[-10:])


def _run_perf_test(test_name: str) -> PerfTestResult:
    """Run one perf selftest by name and normalize the result."""
    result = subprocess.run(
        ["perf", "test", "-vv", test_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")

    if result.returncode != 0:
        reason = _summarize_output(output) or f"perf test returned {result.returncode}"
        return PerfTestResult(
            PerfTestStatus.FAIL,
            reason,
            test_name=test_name,
        )

    if not output.strip():
        return PerfTestResult(
            PerfTestStatus.NO_MATCH,
            f"perf test '{test_name}' returned success with empty output",
            test_name=test_name,
        )

    if _has_ok_line(output):
        return PerfTestResult(
            PerfTestStatus.PASS,
            "",
            test_name=test_name,
        )

    return PerfTestResult(
        PerfTestStatus.FAIL,
        f"perf test '{test_name}' did not report ': Ok'",
        test_name=test_name,
    )


def has_coresight_fallback_evidence(
    event_sources_dir: Path = Path("/sys/bus/event_source/devices"),
    coresight_devices_dir: Path = Path("/sys/bus/coresight/devices"),
) -> bool:
    """Return True when perf exposes cs_etm and Coresight devices exist."""
    if not (event_sources_dir / "cs_etm").exists():
        return False
    if not coresight_devices_dir.is_dir():
        return False
    return any(coresight_devices_dir.iterdir())


def run_perf_test_candidates(candidates: Iterable[str]) -> PerfTestResult:
    """Run the first listed perf selftest candidate that exists."""
    candidate_names = tuple(candidates)
    list_result = subprocess.run(
        ["perf", "test", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if list_result.returncode != 0:
        reason = (list_result.stderr or list_result.stdout or "").strip()
        return PerfTestResult(
            PerfTestStatus.FAIL,
            reason or f"perf test list returned {list_result.returncode}",
        )

    available_tests = _extract_perf_test_names(list_result.stdout or "")
    for candidate in candidate_names:
        if _candidate_matches(candidate, available_tests):
            return _run_perf_test(candidate)

    return PerfTestResult(
        PerfTestStatus.NO_MATCH,
        "No matching perf test found for candidates: " + ", ".join(candidate_names),
    )
