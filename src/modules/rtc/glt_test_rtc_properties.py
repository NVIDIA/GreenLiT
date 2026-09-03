# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test RTC alarm properties.
"""

import os
import glob
from common import glt_core as GLTCore

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_rtc_alarm_property(_, __):
    """
    Perform the RTC alarm property test.

    Checks if the RTC alarm content in /proc/driver/rtc is consistent
    with the wakealarm support status.

    JIRA: DGX-8201
    PATCH: https://git.kernel.org/torvalds/linux/c/101ca8d05913
    """
    proc_rtc_path = "/proc/driver/rtc"
    if not os.path.exists(proc_rtc_path):
        GLTCore.GLTTestResult.skip("SKIP: the /proc/driver/rtc does not exist.")

    # Check if wakealarm is supported by finding 'wakealarm' files in RTC devices
    has_wakealarm_support = False
    for rtc_path in glob.glob("/sys/class/rtc/rtc*/wakealarm"):
        if os.path.exists(rtc_path):
            has_wakealarm_support = True
            break

    # Check for alarm content in /proc/driver/rtc
    try:
        with open(proc_rtc_path, 'r', encoding='utf-8') as f:
            rtc_content = f.read()
    except (IOError, OSError) as e:
        GLTCore.GLTTestResult.fail(f"ERROR: Failed to read {proc_rtc_path}: {e}")
    has_alarm_content = "alarm" in rtc_content.lower()

    # Validate that alarm content matches wakealarm support status
    patch_url = "https://git.kernel.org/torvalds/linux/c/101ca8d05913"
    if has_wakealarm_support:
        if not has_alarm_content:
            error_msg = (
                f"ERROR: The alarm content is not found in {proc_rtc_path} but wakealarm is supported. "
                f"Please apply kernel patch: {patch_url}"
            )
            GLTCore.GLTTestResult.fail(error_msg)
    else:
        if has_alarm_content:
            error_msg = (
                f"ERROR: The alarm content is found in {proc_rtc_path} but wakealarm is not supported. "
                f"Please apply kernel patch: {patch_url}"
            )
            GLTCore.GLTTestResult.fail(error_msg)
