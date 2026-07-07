# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test memory zone properties.
"""
import re
from common import glt_core as GLTCore


def _parse_zoneinfo(logger):
    """Parse /proc/zoneinfo and return a list of zone dicts.

    Each dict has keys: node (int), zone (str), min (int|None), managed (int|None).
    """
    zones = []
    current = {}

    try:
        with open("/proc/zoneinfo", "r", encoding="utf-8") as f:
            for line in f:
                # Zone header line: "Node 0, zone      DMA"
                # Starts a new zone block; save the previous one and skip
                # to the next line (header itself has no watermark data).
                m = re.match(r"Node\s+(\d+),\s+zone\s+(\S+)", line)
                if m:
                    if current:
                        zones.append(current)
                    current = {
                        "node": int(m.group(1)),
                        "zone": m.group(2),
                        "min": None,
                        "managed": None,
                    }
                    continue

                # Non-header lines: extract watermark fields from the
                # indented key-value pairs inside the current zone block.
                stripped = line.strip()
                parts = stripped.split()
                if len(parts) == 2:
                    if parts[0] in ("min", "managed"):
                        try:
                            current[parts[0]] = int(parts[1])
                        except ValueError:
                            logger.error(
                                "Non-integer '%s' value in /proc/zoneinfo: %r",
                                parts[0], parts[1],
                            )
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"Failed to parse /proc/zoneinfo: {e}")
        return []

    # the last zone in the file.
    if current:
        zones.append(current)

    return zones


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_zone_movable_min_watermark(_, logger):
    """
    Test if ZONE_MOVABLE min watermark is properly capped.

    With the patch, ZONE_MOVABLE is treated like ZONE_HIGHMEM in
    __setup_per_zone_wmarks(): its min watermark is capped via
    clamp(managed/1024, 32, 128) instead of receiving a proportional
    share of min_free_kbytes.

    JIRA: DGX-9464
    PATCH: https://lore.kernel.org/all/20230625031656.23941-1-liuq131@chinatelecom.cn/T/#u
    """
    # min_pages = clamp(zone_managed_pages(zone) / 1024, SWAP_CLUSTER_MAX, 128)
    swap_cluster_max = 32
    wmark_min_cap = 128
    managed_threshold = wmark_min_cap * 1024

    zones = _parse_zoneinfo(logger)

    movable = [z for z in zones if z["zone"] == "Movable"]
    if not movable:
        GLTCore.GLTTestResult.skip("No ZONE_MOVABLE zones present.")

    qualifying = [z for z in movable
                  if z["min"] is not None
                  and z["managed"] is not None
                  and z["managed"] > managed_threshold]
    if not qualifying:
        GLTCore.GLTTestResult.skip(
            f"No ZONE_MOVABLE zones with managed > {managed_threshold} pages.")

    failures = []
    for z in qualifying:
        node = z["node"]
        wmark_min = z["min"]

        if wmark_min < swap_cluster_max or wmark_min > wmark_min_cap:
            failures.append(
                f"NUMA node {node}: min={wmark_min} "
                f"(expected [{swap_cluster_max}..{wmark_min_cap}])")

    if failures:
        GLTCore.GLTTestResult.fail(
            "ZONE_MOVABLE min watermark not capped. "
            "Consider backporting commit 416ef04 to fix this issue. "
            "If min_free_kbytes was manually adjusted, "
            "this failure can be ignored. "
            "Details: " + "; ".join(failures))
