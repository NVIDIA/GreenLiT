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

"""Helpers for Linux CPU topology sysfs values."""


def parse_cpu_mask(mask_text):
    """Parse a Linux CPU mask such as '0-3,8' into a set of CPU numbers."""
    cpus = set()
    mask_text = mask_text.strip()
    if not mask_text:
        return cpus

    for part in mask_text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid CPU mask with empty segment: {mask_text!r}")
        if "-" not in part:
            cpus.add(int(part))
            continue

        start_text, end_text = part.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid CPU range: {part}")
        cpus.update(range(start, end + 1))
    return cpus
