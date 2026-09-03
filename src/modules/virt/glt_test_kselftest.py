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
Python script to test virt kselftest.
"""
import platform
from packaging import version
from common import glt_core as GLTCore
from common import kselftest
from common.nv_device import NvDev

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_virt_kselftest(_, __):
    """Perform the kselftest for virt."""
    # Since nvidia kernel 6.5 is EOL, we need to drop 6.5 testing.
    # We drop the 6.5 kernel provided by Canonical for BaseOS, but
    # not all 6.5 kernel with nvidia string.
    # (TODO:DGX-9583) Will change to more flexible solution to
    # disable specific kernel kselftest case testing.
    uname = platform.release().split('.')

    if NvDev.get_release_info().get("NAME") == "Ubuntu" and \
        version.parse(f"{uname[0]}.{uname[1]}") == version.parse("6.5"):
        GLTCore.GLTTestResult.skip("Skip test on nvidia kernel 6.5.")

    # Get Kselftest object
    log_file_name = test_virt_kselftest.__name__ + ".kslog"
    try:
        kselftest_obj = kselftest.Kselftest(log_file_name)
    except ValueError as error:
        GLTCore.GLTTestResult.fail(f"Allocate Kselftest object error: {error}")

    # Define kselftest target test cases to run
    kselftest_test_targets = "TARGETS='kvm'"

    # Build kselftest kvm test case
    # (TODO): may use a python argument to skip build step in the future
    if not kselftest_obj.build_targets_test(kselftest_test_targets):
        GLTCore.GLTTestResult.fail("Encounter error when build targets.")

    # Run kselftest particular test case kvm in linux root folder
    if not kselftest_obj.run_targets_test(kselftest_test_targets, True):
        GLTCore.GLTTestResult.fail("Encounter error when run targets.")

    # Check kselftest test result and return 254 if detect failure
    if not kselftest_obj.check_log(log_file_name):
        GLTCore.GLTTestResult.fail("Kselftest for virt failed.")
