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
Python script to test power cycle.
"""
from common import glt_core as GLTCore
from common import glt_utils


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_reboot(sd, logger):
    """Reboot system to see if it's bootable"""
    for tests in sd.progress:
        if test_reboot.__name__ not in tests:
            continue
        if "TriggerReboot" not in tests[test_reboot.__name__]:
            logger.debug("Rebooting...")
            ret = glt_utils.reboot(test_reboot.__name__, sd)
            if ret is False:
                GLTCore.GLTTestResult.fail("Failed to trigger GLT reboot")
        else:
            logger.debug("Rebooted and resume successfully.")
            glt_utils.systemd_enable(enable=False)
