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
Python script to test KASLR.
"""
import platform
import subprocess
from common import glt_core as GLTCore
from common import glt_utils


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_kaslr_offset(_, __):
    """
    Test if kernel image address is really randomized by checking
    KASLR offset value.

    DGX-8289
    PATCH: https://git.kernel.org/torvalds/c/010338d729c1
    """
    # AMD64 has different memory layout, skip the test for now
    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("Not ARM64, skip the test")

    if not glt_utils.check_kernel_log("KASLR enabled"):
        GLTCore.GLTTestResult.skip("KASLR is not enabled, skip the test")

    # On ARM64 the kernel code, (i.e. _text) starts from 0xffff800080000000 [0]
    # when there's no KASLR, ARM64 also requires the kernel to be on 2M aligned
    # [1] address. If the _text resides below 0xffff800080000000 + 2M, that
    # means address randomization isn't in effect.
    #
    # [0] https://www.kernel.org/doc/html/latest/arch/arm64/memory.html
    # [1] arch/arm64/include/asm/boot.h
    size_2m = 0x00200000
    base_addr = 0xffff800080000000
    # Use shell=False to prevent command injection (CWE-78)
    result = subprocess.run(["grep", " _text$", "/proc/kallsyms"],
                            capture_output=True, text=True, shell=False, check=True)
    kernel_addr = int(result.stdout.split()[0], 16)  # Extract first field (address)

    if kernel_addr - base_addr < size_2m:
        GLTCore.GLTTestResult.fail("KASLR claims to be enabled, but it's not in effect. " +
                                   "Fix EFI RNG protocol to eliminate the issue. " +
                                   "Include kernel commit 010338d729c1 to avoid the confusion.")
