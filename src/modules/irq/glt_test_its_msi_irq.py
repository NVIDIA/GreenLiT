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

"""
Python script to test ITS-MSI IRQ.
"""

import os
import glob
from common import glt_core as GLTCore

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_its_msi_irq(_, __):
    """
    Perform the ITS-MSI IRQ test.

    JIRA: DGX-8209
    PATCH: https://git.kernel.org/torvalds/linux/c/db744ddd59be
    """

    # Check for ITS-MSI in /proc/interrupts
    try:
        with open('/proc/interrupts', 'r', encoding='utf-8') as f:
            interrupts_content = f.read()
            if "ITS-MSI" not in interrupts_content:
                GLTCore.GLTTestResult.skip("SKIP: No ITS-MSI irq.")
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip("SKIP: Could not access /proc/interrupts")

    # Get ITS-MSI hwirq list from /sys/kernel/irq/*/hwirq
    its_msi_irq_list = []
    for chip_name_path in glob.glob('/sys/kernel/irq/*/chip_name'):
        with open(chip_name_path, 'r', encoding='utf-8') as f:
            if "ITS-MSI" in f.read():
                hwirq_path = os.path.join(os.path.dirname(chip_name_path), 'hwirq')
                try:
                    with open(hwirq_path, 'r', encoding='utf-8') as hwirq_file:
                        hwirq = hwirq_file.read().strip()
                        its_msi_irq_list.append(hwirq)
                except FileNotFoundError:
                    continue

    # Check for duplicate ITS-MSI hwirqs
    if len(its_msi_irq_list) != len(set(its_msi_irq_list)):
        patch_url = "https://git.kernel.org/torvalds/linux/c/db744ddd59be"
        error_msg = f"ERROR: Found duplicated its-msi hwirq. Please apply kernel patch: {patch_url}"
        GLTCore.GLTTestResult.fail(error_msg)
