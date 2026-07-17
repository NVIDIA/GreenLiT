# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers for reading GreenLiT cached session progress."""

from dataclasses import dataclass
import json
import os
from typing import Any


SESSION_MODULE_NAME = "glt-test-suite"
SESSION_PROGRESS_FILE = "test_progress.json"


@dataclass(frozen=True)
class GLTSessionSummary:
    """Stable metadata saved in a cached session Summary entry."""

    test_tags: set[str]
    categories: list[str]
    test_cases: list[str]
    verbose: bool = False


@dataclass(frozen=True)
class GLTSessionProgress:
    """Cached session progress plus parsed stable metadata."""

    progress_path: str
    progress: Any
    summary: GLTSessionSummary | None

    def summary_error(self):
        """Return why the cached session summary cannot be used."""
        if self.summary is not None:
            return ""
        return get_session_summary_error(self.progress, self.progress_path)


def get_session_progress_path(session_dir):
    """Return the progress JSON path for a cached GreenLiT session dir."""
    return os.path.join(session_dir, SESSION_MODULE_NAME, SESSION_PROGRESS_FILE)


def parse_session_summary(progress):
    """Return saved summary metadata from a loaded progress payload."""
    if not isinstance(progress, list):
        return None

    test_tags = None
    categories = None
    test_cases = []
    verbose = False

    for item in progress:
        if not isinstance(item, dict):
            continue
        summary = item.get("Summary")
        if isinstance(summary, dict):
            test_tags = summary.get("TestTags")
            categories = summary.get("Categories")
            verbose = bool(summary.get("Verbose", False))
            continue

        for test_name, test_info in item.items():
            if isinstance(test_info, dict) and test_info.get("Category"):
                test_cases.append(f"{test_info['Category']}:{test_name}")
                break

    if not test_tags or not categories:
        return None
    if not test_cases:
        return None

    return GLTSessionSummary(
        test_tags=set(test_tags),
        categories=list(categories),
        test_cases=test_cases,
        verbose=verbose)


def get_session_summary_error(progress, progress_path):
    """Return why a loaded progress payload has no usable summary."""
    if not isinstance(progress, list):
        return (
            f"Resume progress in {progress_path} has invalid format "
            "(expected a JSON array)")

    has_summary_fields = False
    has_test_cases = False

    for item in progress:
        if not isinstance(item, dict):
            continue
        summary = item.get("Summary")
        if isinstance(summary, dict):
            has_summary_fields = bool(
                summary.get("TestTags") and summary.get("Categories"))
            continue

        has_test_cases = has_test_cases or any(
            isinstance(test_info, dict) and test_info.get("Category")
            for test_info in item.values())

    if not has_summary_fields:
        return (
            f"Resume summary in {progress_path} is missing "
            "TestTags or Categories")
    if not has_test_cases:
        return (
            f"Resume progress in {progress_path} has no registered "
            "test cases")
    return ""


def read_progress_file(progress_path):
    """Read a progress JSON file and parse its summary."""
    with open(progress_path, "r", encoding="UTF-8") as progress_file:
        progress = json.load(progress_file)

    summary = parse_session_summary(progress)

    return GLTSessionProgress(
        progress_path=progress_path,
        progress=progress,
        summary=summary)
