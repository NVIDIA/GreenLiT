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

"""Validate Tegra410 PMU vendor event and metric metadata."""

import re

from common import glt_core as GLTCore
from common import perf_utils
from common.nv_device import NvDev

TEGRA410_PMU_EVENTS_COMMIT = (
    "86ff690f45cc perf vendor events arm64: Add Tegra410 Olympus PMU events"
)
TEGRA410_L1_PREFETCHER_METRIC_GROUP_COMMIT = (
    "2d85e13c5526 perf vendor events arm64: fix swapped MetricGroup for "
    "Tegra410 L1 prefetcher metrics"
)
TEGRA410_L1_PREFETCHER_METRIC_GROUP_LKML = (
    "https://lore.kernel.org/all/"
    "20260813170155.1156593-2-bwicaksono@nvidia.com/"
)
TEGRA410_L1_PREFETCHER_METRIC_GROUPS = {
    "l1d_cache_rw_miss_ratio": "L1D_Prefetcher_Effectiveness",
    "l1d_demand_accesses": "L1D_Prefetcher_Effectiveness",
    "l1d_demand_misses": "L1D_Prefetcher_Effectiveness",
    "l1d_prf_accuracy": "L1D_Prefetcher_Effectiveness",
    "l1d_prf_coverage": "L1D_Prefetcher_Effectiveness",
    "l1d_refilled_prf": "L1D_Prefetcher_Effectiveness",
    "l1d_useful_prf": "L1D_Prefetcher_Effectiveness",
    "l1i_cache_rd_miss_ratio": "L1I_Prefetcher_Effectiveness",
    "l1i_demand_accesses": "L1I_Prefetcher_Effectiveness",
    "l1i_demand_misses": "L1I_Prefetcher_Effectiveness",
    "l1i_prf_accuracy": "L1I_Prefetcher_Effectiveness",
    "l1i_prf_coverage": "L1I_Prefetcher_Effectiveness",
    "l1i_refilled_prf": "L1I_Prefetcher_Effectiveness",
    "l1i_useful_prf": "L1I_Prefetcher_Effectiveness",
}
TEGRA410_TXREQ_EVENT_COMMIT = (
    "9453bc6a69ef perf vendor events arm64: Fix Tegra410 Olympus event 0x0197"
)
TEGRA410_TXREQ_EVENT_LKML = (
    "https://lore.kernel.org/all/"
    "20260813170155.1156593-3-bwicaksono@nvidia.com/"
)


def _event_description(entry):
    """Return the normalized public description for a perf event entry."""
    return str(entry.get("PublicDescription", "")).lower()


def _event_entries_by_name(perf_list_json):
    """Index perf event entries by their case-insensitive event name."""
    event_entries = {}
    for item in perf_list_json:
        if not isinstance(item, dict):
            continue
        event_name = item.get("EventName")
        if isinstance(event_name, str):
            event_entries.setdefault(event_name.lower(), []).append(item)
    return event_entries


def _event_has_correct_txreq_metadata(entry):
    """Return whether one entry describes corrected Tegra410 event 0x0197."""
    encoding = str(entry.get("Encoding", "")).lower()
    description = _event_description(entry)
    return bool(
        re.search(r"event=0x0*197(?=$|[^0-9a-f])", encoding)
        and "between 0 and 1/2 of the l2_tq_size" in description
    )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra410_l1_prefetcher_metric_groups(_, logger):
    """
    Verify Tegra410 L1D and L1I prefetcher metrics use the correct groups.

    COMMIT:
    2d85e13c5526 perf vendor events arm64: fix swapped MetricGroup for Tegra410
    L1 prefetcher metrics
    LKML POSTING: https://lore.kernel.org/all/20260813170155.1156593-2-bwicaksono@nvidia.com/
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not NVIDIA Vera CPU; test is Tegra410-specific."
        )

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    try:
        perf_list_json = perf_utils.get_perf_list_json(perf_cmd, logger)
    except ValueError as exc:
        GLTCore.GLTTestResult.fail(str(exc))

    observed_groups = {}
    for item in perf_list_json:
        if not isinstance(item, dict):
            continue
        metric_name = item.get("MetricName")
        if metric_name not in TEGRA410_L1_PREFETCHER_METRIC_GROUPS:
            continue
        metric_group = item.get("MetricGroup", "")
        if not isinstance(metric_group, str):
            metric_group = ""
        observed_groups[metric_name] = metric_group

    incorrect_metrics = []
    for metric_name, expected_group in (
            TEGRA410_L1_PREFETCHER_METRIC_GROUPS.items()):
        observed_group = observed_groups.get(metric_name)
        if metric_name in observed_groups and observed_group != expected_group:
            incorrect_metrics.append(
                f"{metric_name}: expected {expected_group}, "
                f"observed {observed_group!r}"
            )
    if incorrect_metrics:
        GLTCore.GLTTestResult.fail(
            "Tegra410 L1 prefetcher metric groups are incorrect "
            f"(incorrect groups: {incorrect_metrics}). Consider checking "
            "whether the installed perf data is missing commit "
            f"{TEGRA410_L1_PREFETCHER_METRIC_GROUP_COMMIT}. "
            f"LKML posting: {TEGRA410_L1_PREFETCHER_METRIC_GROUP_LKML}."
        )

    missing_metrics = sorted(
        set(TEGRA410_L1_PREFETCHER_METRIC_GROUPS) - set(observed_groups)
    )
    if missing_metrics:
        GLTCore.GLTTestResult.fail(
            "Tegra410 L1 prefetcher metric metadata is unavailable or "
            f"incomplete (missing metrics: {missing_metrics}). Consider "
            f"checking for commit {TEGRA410_PMU_EVENTS_COMMIT}. The installed "
            "perf may have been built without vendor event support or may "
            "not have selected the Tegra410 CPU map."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra410_txreq_limit_event(_, logger):
    """
    Verify Tegra410 event 0x0197 has the below-half TXREQ metadata.

    COMMIT:
    9453bc6a69ef perf vendor events arm64: Fix Tegra410 Olympus event 0x0197
    LKML POSTING: https://lore.kernel.org/all/20260813170155.1156593-3-bwicaksono@nvidia.com/
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not NVIDIA Vera CPU; test is Tegra410-specific."
        )

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    try:
        perf_list_json = perf_utils.get_perf_list_json(perf_cmd, logger)
    except ValueError as exc:
        GLTCore.GLTTestResult.fail(str(exc))

    expected_name = "txreq_limit_below_half_cycles"
    legacy_name = "txreq_limit_1quarter_cycles"
    event_entries = _event_entries_by_name(perf_list_json)
    issues = []

    if legacy_name in event_entries:
        issues.append(f"legacy alias {legacy_name} is still present")

    expected_entries = event_entries.get(expected_name, [])
    if expected_entries and not any(
            map(_event_has_correct_txreq_metadata, expected_entries)):
        issues.append(
            f"{expected_name} does not describe event 0x0197 below one half"
        )

    companion_entries = event_entries.get("txreq_limit_count_cycles", [])
    if companion_entries:
        companion_description = " ".join(
            _event_description(entry) for entry in companion_entries
        )
        if legacy_name in companion_description:
            issues.append(
                "txreq_limit_count_cycles references the legacy alias"
            )
        elif expected_entries and expected_name not in companion_description:
            issues.append(
                "txreq_limit_count_cycles does not reference the corrected alias"
            )

    if issues:
        GLTCore.GLTTestResult.fail(
            "Tegra410 TXREQ limit event metadata is incorrect "
            f"({'; '.join(issues)}). Consider checking whether the installed "
            f"perf data is missing commit {TEGRA410_TXREQ_EVENT_COMMIT}. "
            f"LKML posting: {TEGRA410_TXREQ_EVENT_LKML}."
        )

    unavailable = []
    if not expected_entries:
        unavailable.append(
            "neither the legacy nor corrected event 0x0197 alias is exposed"
        )
    if not companion_entries:
        unavailable.append("companion event txreq_limit_count_cycles is missing")
    if unavailable:
        GLTCore.GLTTestResult.fail(
            "Tegra410 TXREQ limit event metadata is unavailable or incomplete "
            f"({'; '.join(unavailable)}). Consider checking whether the "
            f"installed perf includes commit {TEGRA410_PMU_EVENTS_COMMIT}. "
            "It may have been built without vendor event support or may not "
            "have selected the Tegra410 CPU map."
        )
