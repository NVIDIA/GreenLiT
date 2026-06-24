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

import platform
import re
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


def get_perf_version(logger=None) -> str:
    """Return the perf userspace version string when available."""
    try:
        result = subprocess.run(
            ["perf", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as err:
        if logger:
            logger.debug("Unable to query perf version: %s", err)
        return ""

    if result.returncode != 0:
        if logger:
            reason = (result.stderr or result.stdout or "").strip()
            logger.debug("Unable to query perf version: %s",
                         reason or f"return code {result.returncode}")
        return ""

    return (result.stdout or result.stderr).strip()


def _parse_major_minor(version_text: str) -> Optional[tuple[int, int]]:
    """Return the first major.minor pair from a version-like string."""
    match = re.search(r"(\d+)\.(\d+)", version_text or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def get_perf_kernel_version_warning(
    perf_version: str,
    kernel_release: str,
    linux_tools_dir: Path = Path("/usr/lib/linux-tools"),
) -> str:
    """Return a warning when perf userspace may not match the running kernel."""
    warnings = []
    perf_major_minor = _parse_major_minor(perf_version)
    kernel_major_minor = _parse_major_minor(kernel_release)
    if perf_major_minor and kernel_major_minor and \
            perf_major_minor != kernel_major_minor:
        warnings.append(
            f"perf userspace ({perf_version}) does not match running kernel "
            f"major/minor ({kernel_release}); install the matching "
            "perf/linux-tools package for the running kernel."
        )

    if linux_tools_dir.is_dir():
        expected_perf = linux_tools_dir / kernel_release / "perf"
        if not expected_perf.exists():
            warnings.append(
                f"Matching kernel perf binary is missing: {expected_perf}; "
                "installed perf may not include events for the running kernel."
            )

    return " ".join(warnings)


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


def _get_perf_candidates():
    """Return perf command candidates without requiring them to exist."""
    candidates = ["perf"]
    candidates.extend(
        str(path) for path in sorted(Path("/usr/lib").glob("linux-tools*/perf"))
    )
    candidates.extend(
        str(path) for path in sorted(Path("/usr/lib/linux-tools").glob("*/perf"))
    )
    return list(dict.fromkeys(candidates))


def _candidate_matches_kernel(
    candidate: str,
    kernel_release: str,
    kernel_version: str,
    perf_version: str,
) -> bool:
    """Return True when a perf candidate matches the running kernel."""
    if perf_version == kernel_version:
        return True

    tool_dir = Path(candidate).parent.name
    if tool_dir in (kernel_release, kernel_version):
        return True
    return tool_dir.startswith(f"{kernel_version}-")


def _run_perf_test(perf_cmd: str, test_name: str) -> PerfTestResult:
    """Run one perf selftest by name and normalize the result."""
    result = subprocess.run(
        [perf_cmd, "test", "-vv", test_name],
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


def resolve_perf_command(logger=None) -> Optional[str]:
    """Resolve a perf command that works for the running kernel."""
    kernel_release = platform.uname().release
    kernel_version = kernel_release.split("-", 1)[0]

    for candidate in _get_perf_candidates():
        try:
            result = subprocess.run(
                [candidate, "--version"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as err:
            if logger:
                logger.debug("perf resolver skipped candidate %s: %s", candidate, err)
            continue

        output = result.stdout or ""
        if result.returncode != 0 or "perf not found for kernel" in output:
            if logger:
                logger.debug(
                    "perf resolver rejected candidate: %s --version returned %r",
                    candidate,
                    output.strip(),
                )
            continue

        version_match = re.search(r"perf version\s+([^\s]+)", output)
        perf_version = version_match.group(1) if version_match else ""
        if _candidate_matches_kernel(
                candidate, kernel_release, kernel_version, perf_version):
            return candidate

        if logger:
            logger.debug(
                "perf resolver skipped non-matching candidate: %s reports %r",
                candidate,
                perf_version,
            )

    return None


def run_perf_test_candidates(
    candidates: Iterable[str],
    perf_cmd: str = "perf",
) -> PerfTestResult:
    """Run the first listed perf selftest candidate that exists."""
    candidate_names = tuple(candidates)
    list_result = subprocess.run(
        [perf_cmd, "test", "list"],
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
            return _run_perf_test(perf_cmd, candidate)

    return PerfTestResult(
        PerfTestStatus.NO_MATCH,
        "No matching perf test found for candidates: " + ", ".join(candidate_names),
    )
