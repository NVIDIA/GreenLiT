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

"""Helpers for running perf commands and selftests across perf versions."""

import json
import platform
import re
import subprocess
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Iterable, Optional

PERF_VERSION_TIMEOUT_SECONDS = 5
PERF_TEST_TIMEOUT_SECONDS = 120
PERF_LIST_TIMEOUT_SECONDS = 30


class _PerfRejectRank(IntEnum):
    """How far a rejected perf candidate got before it was refused."""

    NOT_INSTALLED = 0
    PROBE_TIMEOUT = 1
    PROBE_FAILED = 2
    VERSION_MISMATCH = 3
    NO_KERNEL_BINARY = 4


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


def get_perf_list_json(perf_cmd: str, logger) -> list:
    """Return parsed JSON entries from ``perf list -j``."""
    try:
        cmd_output = subprocess.run(
            [perf_cmd, "list", "-j"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PERF_LIST_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        stderr = " ".join((exc.stderr or "").splitlines())
        logger.error("perf list failed: %s", stderr or exc)
        raise ValueError(
            f"perf list failed with return code {exc.returncode}: "
            f"{stderr or 'no stderr output'}"
        ) from exc
    except OSError as exc:
        logger.error("Failed to execute perf list: %s", exc)
        raise ValueError(f"Failed to execute perf list: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        logger.error("perf list timed out after %s seconds", exc.timeout)
        raise ValueError(
            f"perf list timed out after {exc.timeout} seconds"
        ) from exc

    try:
        perf_list_json = json.loads(cmd_output.stdout)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse perf list JSON output: %s", exc)
        raise ValueError(
            f"Failed to parse perf list JSON output: {exc}"
        ) from exc

    if not isinstance(perf_list_json, list):
        raise ValueError("perf list JSON output is not an array")
    return perf_list_json


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


def natural_sort_key(path: Path):
    """Return a key that sorts embedded numeric components numerically."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    )


def _get_perf_candidates():
    """Return perf command candidates without requiring them to exist."""
    candidates = ["perf"]
    candidates.extend(
        str(path)
        for path in sorted(
            Path("/usr/lib").glob("linux-tools*/perf"),
            key=natural_sort_key,
        )
    )
    candidates.extend(
        str(path)
        for path in sorted(
            Path("/usr/lib/linux-tools").glob("*/perf"),
            key=natural_sort_key,
        )
    )
    return list(dict.fromkeys(candidates))


def _perf_version_matches_kernel(perf_version: str, kernel_version: str) -> bool:
    """Return True when perf and kernel versions are compatible."""
    if not perf_version or not kernel_version:
        return False
    if perf_version == kernel_version:
        return True

    shorter = perf_version
    longer = kernel_version
    if len(shorter) > len(longer):
        shorter, longer = longer, shorter
    return longer.startswith((f"{shorter}.", f"{shorter}-", f"{shorter}+"))


def _normalized_tool_dir(candidate: str) -> str:
    """Return the linux-tools directory version token for a candidate path."""
    tool_dir = Path(candidate).parent.name
    if tool_dir.startswith("linux-tools-"):
        return tool_dir[len("linux-tools-"):]
    return tool_dir


def _candidate_is_kernel_compatible(
    candidate: str,
    kernel_release: str,
    kernel_version: str,
    perf_version: str,
) -> bool:
    """Return True for exact or abbreviated matches to the running kernel."""
    perf_base_version = perf_version.split("-", 1)[0]
    if _perf_version_matches_kernel(perf_base_version, kernel_version):
        return True

    normalized_tool_dir = _normalized_tool_dir(candidate)
    if normalized_tool_dir in (kernel_release, kernel_version):
        return True
    return normalized_tool_dir.startswith(f"{kernel_version}-")


def _candidate_is_exact_kernel_match(
    candidate: str,
    kernel_release: str,
    kernel_version: str,
    perf_version: str,
) -> bool:
    """Return True when a candidate exactly matches kernel base or release."""
    perf_base_version = perf_version.split("-", 1)[0]
    if perf_base_version == kernel_version:
        return True
    return _normalized_tool_dir(candidate) in (kernel_release, kernel_version)


def _run_perf_test(perf_cmd: str, test_name: str) -> PerfTestResult:
    """Run one perf selftest by name and normalize the result."""
    try:
        result = subprocess.run(
            [perf_cmd, "test", "-vv", test_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=PERF_TEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return PerfTestResult(
            PerfTestStatus.FAIL,
            f"perf test '{test_name}' timed out after "
            f"{PERF_TEST_TIMEOUT_SECONDS} seconds",
            test_name=test_name,
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


def _best_rejection_reason(rejections) -> str:
    """Return the reason from the candidate that got furthest before refusal."""
    best_rank = -1
    best_reason = ""
    for rank, reason in rejections:
        if rank > best_rank:
            best_rank = rank
            best_reason = reason
    return best_reason


def resolve_perf_command() -> tuple[Optional[str], str]:
    """Resolve a perf command that works for the running kernel.

    Returns:
        (command, ""):  a perf command usable for the running kernel.
        (None, reason): no usable perf was found. reason names the candidate
                        that got furthest, why it was refused, and the remedy,
                        so callers can report it instead of guessing.
    """
    kernel_release = platform.uname().release
    kernel_version = re.split(r"[-+]", kernel_release, 1)[0]
    compatible_candidate = None
    rejections: list[tuple[_PerfRejectRank, str]] = []

    for candidate in _get_perf_candidates():
        try:
            result = subprocess.run(
                [candidate, "--version"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=PERF_VERSION_TIMEOUT_SECONDS,
            )
        except OSError as err:
            rejections.append((
                _PerfRejectRank.NOT_INSTALLED,
                f"'{candidate}' could not be executed ({err}). Install the "
                f"linux-tools package matching kernel {kernel_release}, or "
                "build perf from the matching kernel tree.",
            ))
            continue
        except subprocess.TimeoutExpired:
            rejections.append((
                _PerfRejectRank.PROBE_TIMEOUT,
                f"'{candidate} --version' timed out after "
                f"{PERF_VERSION_TIMEOUT_SECONDS} seconds.",
            ))
            continue

        output = result.stdout or ""
        if "perf not found for kernel" in output:
            rejections.append((
                _PerfRejectRank.NO_KERNEL_BINARY,
                f"'{candidate}' has no perf binary for kernel "
                f"{kernel_release} ({output.strip()}). Install the matching "
                "linux-tools package, or build perf from the matching "
                "kernel tree.",
            ))
            continue
        if result.returncode != 0:
            rejections.append((
                _PerfRejectRank.PROBE_FAILED,
                f"'{candidate} --version' failed with exit status "
                f"{result.returncode} ({output.strip()}).",
            ))
            continue

        version_match = re.search(r"perf version\s+([^\s]+)", output)
        perf_version = version_match.group(1) if version_match else ""
        if not _candidate_is_kernel_compatible(
                candidate, kernel_release, kernel_version, perf_version):
            if perf_version:
                rejections.append((
                    _PerfRejectRank.VERSION_MISMATCH,
                    f"'{candidate}' is perf {perf_version}, which does not "
                    f"match kernel {kernel_release}. GreenLiT requires a "
                    "matching perf so tests are not run against a stale "
                    "perf ABI. Install the matching linux-tools package, or "
                    "build perf from the matching kernel tree.",
                ))
            else:
                rejections.append((
                    _PerfRejectRank.PROBE_FAILED,
                    "could not parse a perf version from "
                    f"'{candidate} --version' output ({output.strip()!r}).",
                ))
            continue

        if _candidate_is_exact_kernel_match(
                candidate, kernel_release, kernel_version, perf_version):
            return (candidate, "")

        if compatible_candidate is None:
            compatible_candidate = candidate

    if compatible_candidate is not None:
        return (compatible_candidate, "")

    best = _best_rejection_reason(rejections)
    reason = f"No usable perf found: {best}" if best else "No usable perf found."
    return (None, reason)


def run_perf_test_candidates(
    candidates: Iterable[str],
    perf_cmd: str = "perf",
) -> PerfTestResult:
    """Run the first listed perf selftest candidate using perf_cmd."""
    candidate_names = tuple(candidates)
    try:
        list_result = subprocess.run(
            [perf_cmd, "test", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=PERF_TEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return PerfTestResult(
            PerfTestStatus.FAIL,
            f"perf test list timed out after {PERF_TEST_TIMEOUT_SECONDS} seconds",
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
