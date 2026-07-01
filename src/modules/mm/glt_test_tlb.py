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

"""
Python script to detect the arm64 memory hot-remove per-page TLB-flush storm.
"""
import ctypes.util
import os
import platform
import re
import subprocess
import time
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("stress")
def test_hotplug_tlb_flush_storm(_, __):  # pylint: disable=too-many-branches
    """Detect the arm64 memory hot-remove per-page TLB-flush regression.

    Fills GPU HBM as UVM coherent devmem, then times `rmmod nvidia_uvm` and
    scans the kernel log. A pre-fix arm64 kernel flushes the kernel TLB one
    page at a time in unmap_hotplug_range during arch_remove_memory, storming
    for minutes and soft-locking up; the fixed kernel batches the flush and
    tears down in seconds.

    JIRA: DGX-17016
    PATCH: https://github.com/torvalds/linux/commit/48478b9f7913
    """
    # Fixed batched-flush teardown is < 5 s. The buggy 4 KB per-page storm is
    # ~144 s. 30 s cleanly separates the two.
    threshold_s = 30
    # Minimum GPU memory the workload must make device-resident for the teardown to
    # be meaningfully exercised. This is a coarse floor to turn a grossly-failed or
    # near-empty fill into a SKIP (not fine calibration): a properly filled GB200
    # makes hundreds of GB resident, so ~64 GB is comfortably below a real run yet
    # far above a broken one. Without this, an under-filled workload on a buggy
    # kernel could tear down fast and be misread as a PASS.
    min_resident_mb = 65536

    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("Not ARM64; hot-remove TLB flush path is arm64-specific.")

    if not NvDev().get_nvidia_3d_bdf().strip():
        GLTCore.GLTTestResult.skip("No NVIDIA 3D-class GPU found.")

    if os.sysconf("SC_PAGE_SIZE") != 4096:
        GLTCore.GLTTestResult.skip(
            "Page size is not 4KB. The fix helps 64KB too, but this timing "
            "check is only reliable for the denser 4KB per-page flush storm.")

    if glt_utils.read_sysfs_value("/proc/sys/kernel/softlockup_panic") == "1":
        GLTCore.GLTTestResult.skip(
            "softlockup_panic=1: unsafe (would panic on the expected soft-lockup).")

    if subprocess.run(["modinfo", "nvidia_uvm"], capture_output=True,
                      check=False).returncode != 0:
        GLTCore.GLTTestResult.skip("nvidia_uvm module not available.")

    if ctypes.util.find_library("cuda") is None and \
            not os.path.exists("/usr/lib/aarch64-linux-gnu/libcuda.so.1"):
        GLTCore.GLTTestResult.fail(
            "NVIDIA GPU present but libcuda.so.1 is missing. "
            "Install the NVIDIA GPU driver to run this test.")

    elapsed = 0.0
    lockup = False
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.join(module_dir, "hotplug_tlb")
    with glt_utils.chdir_managed(module_dir):
        try:
            # Use shell=False to prevent command injection (CWE-78)
            subprocess.run(["make"], check=True, shell=False)

            uvm_was_loaded = os.path.exists("/sys/module/nvidia_uvm")
            try:
                subprocess.run(["modprobe", "nvidia_uvm"],
                               check=True, timeout=threshold_s)
            except subprocess.TimeoutExpired:
                GLTCore.GLTTestResult.fail(
                    f"modprobe nvidia_uvm timed out after {threshold_s}s")

            try:
                # Register all GPUs and fill ~90% of each GPU's HBM as UVM
                # coherent devmem so the hot-added region to tear down is large.
                try:
                    fill = subprocess.run(["./uvm_register"], capture_output=True,
                                          text=True, timeout=300, check=False)
                except subprocess.TimeoutExpired:
                    GLTCore.GLTTestResult.fail(
                        "UVM workload timed out after 300s while filling GPU memory.")

                # The teardown is only a valid test if the workload actually made a
                # large amount of GPU memory device-resident.
                match = re.search(r"TOTAL_RESIDENT_MB=(\d+)", fill.stdout)
                resident_mb = int(match.group(1)) if match else 0
                if fill.returncode != 0 or resident_mb < min_resident_mb:
                    GLTCore.GLTTestResult.skip(
                        f"UVM workload made only {resident_mb} MB resident "
                        f"({min_resident_mb} required, rc={fill.returncode}); "
                        "cannot exercise the teardown.")

                t0 = time.monotonic()
                try:
                    subprocess.run(["rmmod", "nvidia_uvm"], check=True,
                                   timeout=threshold_s)
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - t0
                    GLTCore.GLTTestResult.fail(
                        f"nvidia_uvm coherent-devmem teardown timed out after "
                        f"{elapsed:.1f}s (threshold {threshold_s}s). "
                        "The arm64 hot-remove per-page TLB flush is not batched. "
                        "Consider to backport commit 48478b9")
                except subprocess.CalledProcessError as e:
                    elapsed = time.monotonic() - t0
                    GLTCore.GLTTestResult.fail(
                        f"nvidia_uvm rmmod failed with rc={e.returncode} "
                        f"after {elapsed:.1f}s")
                else:
                    elapsed = time.monotonic() - t0

                # A pre-fix kernel soft-locks up in unmap_hotplug_range during the
                # teardown; the fixed kernel does not. Timing is the primary
                # signal. This corroborates it against the kernel log.
                lockup = (
                    (glt_utils.check_kernel_log("soft lockup")
                     or glt_utils.check_kernel_log("detected stall"))
                    and glt_utils.check_kernel_log("unmap_hotplug_range"))
            finally:
                if uvm_was_loaded:
                    try:
                        subprocess.run(["modprobe", "nvidia_uvm"],
                                       check=True, timeout=threshold_s)
                    except (subprocess.TimeoutExpired,
                            subprocess.CalledProcessError) as e:
                        GLTCore.GLTTestResult.fail(
                            f"modprobe nvidia_uvm restore failed: {e}")

            if lockup or elapsed > threshold_s:
                GLTCore.GLTTestResult.fail(
                    f"nvidia_uvm coherent-devmem teardown took {elapsed:.1f}s "
                    f"(threshold {threshold_s}s)"
                    + ". The arm64 hot-remove per-page TLB flush is not batched. "
                    "Consider to backport commit 48478b9")

        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(f"Error executing command: {e}")
        finally:
            # Best-effort: skip/fail raise and would otherwise leave build
            # artifacts in the installed module tree.
            subprocess.run(["make", "clean"], check=False, shell=False)
