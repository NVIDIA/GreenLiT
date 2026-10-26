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
Python script to test IOMMU.
"""
import ctypes.util
import glob
import os
import platform
import re
import subprocess
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from modules.iommu import cmdqv_hyp_own_runner as runner


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_iommu_id_mapping(_, __):
    """Test if could detect iommu id mapping logs in dmesg."""

    # This test is specific to ARM64 SMMU
    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("Not ARM64, skip the test")

    id_mapping_error_msg = "ID mapping index overflows valid mappings"

    try:
        id_mapping_error_found = glt_utils.check_kernel_log(id_mapping_error_msg)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve IOMMU ID mapping kernel log: {error}"
        )

    if id_mapping_error_found:
        GLTCore.GLTTestResult.fail(f'Find iommu error message {id_mapping_error_msg}. ' +
                                   'Include kernel commit 05da178ce0aa to fix it.')


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cmdqv_alignment(_, __):
    """Test Tegra241 CMDQV queue alignment fix.

    JIRA: DGX-10648
    Patch: https://lore.kernel.org/all/20241111030226.1940737-1-nicolinc@nvidia.com/
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("Not a Grace CPU, CMDQV is Grace-specific")

    page_size = os.sysconf("SC_PAGE_SIZE")
    if page_size != 4096:
        GLTCore.GLTTestResult.skip(
            f"Page size is {page_size}, not 4KB."
            " The alignment bug only triggered on 4KB page kernels.")

    if not glob.glob("/sys/devices/platform/NVDA200C:*"):
        GLTCore.GLTTestResult.skip("No NVDA200C CMDQV devices found in sysfs")

    try:
        cmdqv_warning_found = glt_utils.check_kernel_log("arm_smmu_init_one_queue")
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve CMDQV kernel log: {error}"
        )

    if cmdqv_warning_found:
        GLTCore.GLTTestResult.fail(
            "WARNING at arm_smmu_init_one_queue detected in kernel log."
            " The Tegra241 CMDQV max_n_shift alignment fix is missing."
            " Consider backporting commit a379971 to fix it.")


def _kprobe_events_write(kprobe_events, cmd):
    """Write a single kprobe_events command at offset 0"""
    fd = os.open(kprobe_events, os.O_WRONLY)
    try:
        os.write(fd, (cmd + "\n").encode())
    finally:
        os.close(fd)


def _sizes_for_pid(trace, pid):
    """Return gltsva size args attributed to `pid` only.

    The kprobe is process-global; without PID filtering, concurrent callers of
    arm_smmu_tlb_inv_range_asid can inject page-aligned hits and false-PASS.
    """
    token = f"-{pid} "
    sizes = []
    for line in trace.splitlines():
        if "gltsva:" not in line or token not in line:
            continue
        sizes.extend(
            int(m, 16) for m in re.findall(r"size=0x([0-9a-fA-F]+)", line))
    return sizes


def _unaligned_sva_sizes(sizes, page):
    """Return members of `sizes` that are not multiples of `page`.

    Raises:
        ValueError: if page <= 0 (sysconf failure must not modulo as -1).
    """
    if page <= 0:
        raise ValueError(f"invalid page size: {page}")
    return [sz for sz in sizes if sz % page != 0]


def _run_sva_workload(cmd, event_filter):
    """Run the sva_trigger argv `cmd` under a common_pid kprobe filter; return pid.

    Raises:
        subprocess.TimeoutExpired: soft-lock / hang during munmap invalidate.
        subprocess.CalledProcessError: non-zero workload exit.
    """
    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as proc:
        # Prefer an in-kernel common_pid filter so foreign hits never land in
        # the buffer; userspace PID matching below remains as a backstop.
        try:
            glt_utils.write_sysfs_value(
                event_filter, f"common_pid == {proc.pid}")
        except OSError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        finally:
            # Kill before Popen.__exit__ wait() so a soft-locked munmap
            # cannot hang harness cleanup.
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=stdout, stderr=stderr)
        return proc.pid


def _run_with_kprobe(tracefs, cmd, event="gltsva",
                     probe="arm_smmu_tlb_inv_range_asid size=$arg2"):
    """Set the `event` kprobe on `probe`, run argv `cmd`, return (trace, pid)."""
    kprobe_events = os.path.join(tracefs, "kprobe_events")
    enable = os.path.join(tracefs, f"events/kprobes/{event}/enable")
    event_filter = os.path.join(tracefs, f"events/kprobes/{event}/filter")
    tracing_on = os.path.join(tracefs, "tracing_on")
    trace = os.path.join(tracefs, "trace")

    # Pre-clean: drop any stale `event` left by a killed prior run, without
    # touching other probes. A stale *enabled* probe must be disabled first or
    # the delete is rejected. Both steps are best-effort.
    try:
        if os.path.exists(enable):
            glt_utils.write_sysfs_value(enable, 0)
    except OSError:
        pass
    try:
        _kprobe_events_write(kprobe_events, f"-:{event}")
    except OSError:
        pass

    try:
        # Install the probe with a single O_WRONLY write (never append/O_TRUNC).
        # A kernel that rejects the symbol or the $arg syntax raises OSError; let
        # it propagate so the decorated caller can SKIP (skip()/fail() must come
        # from a @GLTTestAttr function for the result to be recorded correctly).
        _kprobe_events_write(kprobe_events, f"p:{event} {probe}")
        glt_utils.write_sysfs_value(enable, 1)
        glt_utils.write_sysfs_value(trace, "")
        glt_utils.write_sysfs_value(tracing_on, 1)

        # Do not swallow TimeoutExpired: on an unfixed kernel with
        # ARM_SMMU_FEAT_RANGE_INV, munmap can soft-lock in invalidate.
        # Let the caller FAIL rather than parse a partial/empty trace as SKIP.
        # Non-zero exits (CUDA setup rc 2-4, SVA bind stimulus rc 5) raise
        # CalledProcessError instead of becoming an empty-trace SKIP.
        pid = _run_sva_workload(cmd, event_filter)
        return glt_utils.read_sysfs_value(trace), pid
    finally:
        try:
            glt_utils.write_sysfs_value(event_filter, "0")
        except OSError:
            pass
        try:
            glt_utils.write_sysfs_value(tracing_on, 0)
        except OSError:
            pass
        try:
            glt_utils.write_sysfs_value(enable, 0)
        except OSError:
            pass
        try:
            # Remove only our probe with a '-:<event>' delete directive.
            _kprobe_events_write(kprobe_events, f"-:{event}")
        except OSError:
            pass


def _resolve_tracefs():
    """Return a usable tracefs mount with kprobe_events, or None."""
    for path in ("/sys/kernel/tracing", "/sys/kernel/debug/tracing"):
        if os.path.exists(os.path.join(path, "kprobe_events")):
            return path
    return None


def _sva_workload_skip_reason(tracefs):
    """Return why the GPU SVA kprobe workload cannot run here, or None."""
    kprobe_events = os.path.join(tracefs or "", "kprobe_events")
    # Evaluated in order; later checks may rely on earlier ones passing.
    checks = (
        (lambda: platform.machine() != "aarch64", "Not ARM64, skip the test"),
        (lambda: not NvDev().get_nvidia_3d_bdf().strip(),
         "No NVIDIA 3D-class GPU found."),
        (lambda: not glob.glob("/sys/class/iommu/smmu3.*"),
         "No ARM SMMU-v3 found in /sys/class/iommu."),
        (lambda: not tracefs,
         "tracefs not available (/sys/kernel/tracing or "
         "/sys/kernel/debug/tracing with kprobe_events)"),
        (lambda: not os.access(kprobe_events, os.W_OK),
         f"{kprobe_events} not writable; kprobe privilege required"),
        (lambda: ctypes.util.find_library("cuda") is None and
         not os.path.exists("/usr/lib/aarch64-linux-gnu/libcuda.so.1"),
         "libcuda.so.1 not present; SVA workload needs the GPU driver."),
        (lambda: not glt_utils.is_cli_available("make")
         or not glt_utils.is_cli_available("gcc"),
         "make/gcc not available; cannot build SVA workload"),
    )
    for unavailable, reason in checks:
        if unavailable():
            return reason
    return None


# arm-smmu-v3 feature flags, as printed in "(features 0x...)" at SMMU probe.
SMMU_FEAT_HA = 1 << 21
SMMU_FEAT_HD = 1 << 22
# Context descriptor word 0 fields (Arm SMMUv3 CD layout).
CD0_TCR_EPD0 = 1 << 14
CD0_TCR_EPD1 = 1 << 30
CD0_V = 1 << 31
CD0_AA64 = 1 << 41
CD0_TCR_HD = 1 << 42
CD0_TCR_HA = 1 << 43
SVA_HTTU_COMMIT = ('74fa4c177ad0 ("iommu/arm-smmu-v3-sva: Enable Hardware '
                   'Access and Hardware Dirty bits", v7.2) or, on 6.18.y, '
                   '05dc22d5c4fa ("iommu/arm-smmu-v3: Add HAFT support for '
                   'SVA", v6.18.51, stable backport of f99d3b4bb095)')


def _gpu_addressing_modes():
    """Return the set of nvidia-smi GPU addressing modes, empty if unknown.

    Values are "ATS", "HMM" or "None". Older drivers without the
    addressing_mode query field, or a failing nvidia-smi, yield an empty set.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=addressing_mode",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _smmu_features(kernel_log):
    """Return the feature mask of every arm-smmu-v3 instance in the log."""
    return [int(m, 16)
            for m in re.findall(r"\(features 0x([0-9a-fA-F]+)\)", kernel_log)]


def _required_sva_httu_bits(features):
    """Return {name: CD bit} that every SMMU's SVA CDs must carry.

    Only flags every SMMU reports are required: the kprobe cannot tell which
    SMMU a CD belongs to, and SVA CDs set HA/HD per SMMU capability.
    """
    common = SMMU_FEAT_HA | SMMU_FEAT_HD
    for feat in features:
        common &= feat
    required = {}
    if common & SMMU_FEAT_HA:
        required["HA"] = CD0_TCR_HA
    if common & SMMU_FEAT_HD:
        required["HD"] = CD0_TCR_HD
    return required


def _sva_cds_for_pid(trace, pid):
    """Return (pasid, cd0) for live SVA CDs installed by `pid`.

    Live means valid (V=1) with TTB0 walks enabled (EPD0=0), which excludes
    the fault-everything CD written when an SVA mm is released. EPD1 and
    AA64, which arm_smmu_make_sva_cd() always sets, must also be set, so a
    probe that reads the wrong argument yields no CD rather than a bogus one.
    """
    token = f"-{pid} "
    cds = []
    for line in trace.splitlines():
        if "gltsvacd:" not in line or token not in line:
            continue
        match = re.search(r"pasid=(\d+) cd0=0x([0-9a-fA-F]+)", line)
        if not match:
            continue
        cd0 = int(match.group(2), 16)
        shape = CD0_V | CD0_TCR_EPD1 | CD0_AA64
        if cd0 & shape == shape and not cd0 & CD0_TCR_EPD0:
            cds.append((int(match.group(1)), cd0))
    return cds


def _cds_missing_bits(cds, required):
    """Return (pasid, cd0, [missing names]) for CDs lacking required bits."""
    bad = []
    for pasid, cd0 in cds:
        missing = [name for name, bit in required.items() if not cd0 & bit]
        if missing:
            bad.append((pasid, cd0, missing))
    return bad


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_sva_invalidate_range_size(_, __):  # pylint: disable=too-many-branches
    """Detect reintroduction of the ARM SMMU-v3 SVA invalidate-range off-by-one.

    Drives a GPU SVA workload while a kprobe captures the 'size' passed to
    arm_smmu_tlb_inv_range_asid, and FAILs if any captured size is not
    page-aligned.

    JIRA: DGX-8241
    Patch: https://lore.kernel.org/all/20220419210158.21320-1-nicolinc@nvidia.com/
    """
    tracefs = _resolve_tracefs()
    reason = _sva_workload_skip_reason(tracefs)
    if reason:
        GLTCore.GLTTestResult.skip(reason)

    sizes = []
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.join(module_dir, "sva_invalidate")
    with glt_utils.chdir_managed(module_dir):
        try:
            # Use shell=False to prevent command injection (CWE-78)
            subprocess.run(["make"], check=True, shell=False)
            try:
                trace, workload_pid = _run_with_kprobe(tracefs, ["./sva_trigger"])
            except OSError as e:
                GLTCore.GLTTestResult.skip(
                    f"could not install kprobe on arm_smmu_tlb_inv_range_asid: {e}")
            except subprocess.TimeoutExpired:
                GLTCore.GLTTestResult.fail(
                    "SVA workload timed out after 120s; possible soft-lock in "
                    "arm_smmu invalidate on an unfixed kernel. "
                    "Consider backporting commit 95a5885")
            sizes = _sizes_for_pid(trace, workload_pid)
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or e.stdout or "").strip()
            if e.returncode == 5:
                GLTCore.GLTTestResult.fail(
                    "SVA not bound: managed-alloc bind stimulus failed"
                    + (f": {detail}" if detail else ""))
            GLTCore.GLTTestResult.fail(
                f"Error executing command: {e}"
                + (f": {detail}" if detail else ""))
        except OSError as e:
            GLTCore.GLTTestResult.fail(f"Error executing command: {e}")
        finally:
            subprocess.run(["make", "clean"], check=False, shell=False)

    if not sizes:
        # Workload exit 0 means CUDA context + managed-alloc bind stimulus
        # succeeded. No PID-filtered TLBI hits => SVA invalidate path was not
        # actually active; fail closed with an explicit unbound reason.
        GLTCore.GLTTestResult.fail(
            "SVA not bound or invalidate path not exercised: workload "
            "completed but no sva_trigger-attributed "
            "arm_smmu_tlb_inv_range_asid hits were captured.")

    page = os.sysconf("SC_PAGE_SIZE")
    try:
        bad = _unaligned_sva_sizes(sizes, page)
    except ValueError:
        GLTCore.GLTTestResult.skip(
            f"Invalid page size from sysconf: {page}")
    for sz in bad:
        GLTCore.GLTTestResult.fail(
            f"arm_smmu SVA invalidation size {hex(sz)} is not page-aligned. "
            "Consider backporting commit 95a5885")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_sva_cd_hw_access_dirty(_, logger):  # pylint: disable=too-many-branches
    """Check that SVA context descriptors enable SMMU hardware Access/Dirty.

    When every SMMU reports HTTU (ARM_SMMU_FEAT_HA/HD), the CD that
    arm-smmu-v3 builds for an SVA bind must set TCR.HA/HD so the SMMU
    updates the Access flag and Dirty state of the shared CPU page table
    itself. Without them, every device write to a clean (DBM, read-only)
    PTE, for example in memory freed with MADV_FREE, faults and has to be
    fixed up in software one page at a time.

    Drives a GPU SVA bind while a kprobe on arm_smmu_set_pasid() records the
    CD word 0 installed for the workload, and FAILs if a required bit is
    clear. SKIPs when the SMMUs do not support HTTU or nvidia-smi reports no
    GPU using ATS addressing, since only ATS binds the mm via SVA. WARNs that
    the check is inconclusive if the workload succeeds but no CD is captured.

    JIRA: DGX-18407
    Patch: https://git.kernel.org/torvalds/c/74fa4c177ad0
    Patch (6.18.y): https://git.kernel.org/stable/c/05dc22d5c4fa
    """
    tracefs = _resolve_tracefs()
    reason = _sva_workload_skip_reason(tracefs)
    if reason:
        GLTCore.GLTTestResult.skip(reason)

    try:
        features = _smmu_features(glt_utils.get_kernel_log())
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve the kernel log for SMMU features: {error}")
    if not features:
        GLTCore.GLTTestResult.skip(
            "No arm-smmu-v3 '(features 0x...)' line in the kernel log")
    feature_list = (", ".join(f"{feat:#x}" for feat in sorted(set(features)))
                    + f" ({len(features)} SMMUs)")
    required = _required_sva_httu_bits(features)
    if not required:
        GLTCore.GLTTestResult.skip(
            "Not every SMMU supports hardware Access/Dirty updates (HTTU); "
            f"SMMU features: {feature_list}")
    logger.info(f"SMMU features: {feature_list}; SVA CDs must set "
                f"{'/'.join(required)}")

    modes = _gpu_addressing_modes()
    if modes and modes <= {"HMM", "None"}:
        GLTCore.GLTTestResult.skip(
            "No GPU uses ATS addressing (nvidia-smi addressing_mode: "
            f"{', '.join(sorted(modes))}), so the driver does not bind "
            "process address spaces via SVA.")

    cds = []
    module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "sva_invalidate")
    with glt_utils.chdir_managed(module_dir):
        try:
            subprocess.run(["make"], check=True, shell=False)
            try:
                # 0 iterations: bind and exit, skipping the munmap
                # invalidation loop that test_sva_invalidate_range_size uses.
                trace, workload_pid = _run_with_kprobe(
                    tracefs, ["./sva_trigger", "0"], event="gltsvacd",
                    probe="arm_smmu_set_pasid pasid=$arg3:u32 "
                          "cd0=+0($arg4):x64")
            except OSError as e:
                GLTCore.GLTTestResult.skip(
                    f"could not install kprobe on arm_smmu_set_pasid: {e}")
            except subprocess.TimeoutExpired:
                GLTCore.GLTTestResult.fail(
                    "SVA workload timed out after 120s")
            cds = _sva_cds_for_pid(trace, workload_pid)
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or e.stdout or "").strip()
            GLTCore.GLTTestResult.fail(
                f"Error executing command: {e}"
                + (f": {detail}" if detail else ""))
        except OSError as e:
            GLTCore.GLTTestResult.fail(f"Error executing command: {e}")
        finally:
            subprocess.run(["make", "clean"], check=False, shell=False)

    if not cds:
        cause = ("the probe may not match this kernel's arm_smmu_set_pasid() "
                 "arguments")
        if modes != {"ATS"}:
            cause += ", or the workload's GPU may not use ATS addressing"
        GLTCore.GLTTestResult.warn(
            "The SVA bind workload succeeded, but the arm_smmu_set_pasid() "
            "kprobe captured no live SVA CD for it (nvidia-smi "
            f"addressing_mode: {', '.join(sorted(modes)) or 'unknown'}). "
            f"The check is inconclusive: {cause}.")

    bad = _cds_missing_bits(cds, required)
    if bad:
        details = "; ".join(f"PASID {pasid} CD0 {cd0:#x} lacks {'/'.join(missing)}"
                            for pasid, cd0, missing in bad)
        GLTCore.GLTTestResult.fail(
            f"SVA CD does not enable hardware Access/Dirty updates although "
            f"every SMMU supports them ({details}). Consider checking whether "
            f"the kernel is missing commit {SVA_HTTU_COMMIT}.")
    logger.info(f"{len(cds)} SVA CD(s) set {'/'.join(required)}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cmdqv_hyp_own_supports_cmd(_, logger):
    """Test Tegra241 CMDQV host VINTF0 supports_cmd fix.

    JIRA: DGX-17036
    Patch: https://git.kernel.org/torvalds/c/803e41f36d22
    Discriminator: dual-function-probe COUNT over a host-activity window
        (no struct field reads beyond the SMMU-scoping arm_smmu_device->
        dev->kobj.name) -- counts tegra241_cmdqv_get_cmdq() returning
        non-NULL vs. tegra241_guest_vcmdq_supports_cmd() called while that
        scoped get_cmdq is in flight. On this bare-metal-only test, any
        guest_supports_cmd > 0 means an unpatched host, regardless of
        get_cmdq_nonnull; get_cmdq_nonnull > 0 with guest_supports_cmd == 0
        means a patched host. The test SKIPs before observation when this
        kernel is itself a guest via running_as_guest(). See
        src/modules/iommu/cmdqv_hyp_own/observe.bt.in and
        cmdqv_hyp_own_runner.evaluate_result() for the full rationale and
        the live-hardware measurement backing this design.
    """
    # The runner raises the shared GLTTestSkip/GLTTestFail, but only
    # GLTTestResult.skip()/fail() -- which must be called from an
    # @GLTTestAttr-decorated function -- record the reason on this test's attr
    # for the session report. Hence the re-raise through them here.
    try:
        runner.run_cmdqv_hyp_own_test(logger)
    except GLTCore.GLTTestSkip as exc:
        GLTCore.GLTTestResult.skip(str(exc))
    except GLTCore.GLTTestFail as exc:
        GLTCore.GLTTestResult.fail(str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        GLTCore.GLTTestResult.fail(
            f"unexpected infrastructure error ({type(exc).__name__}): {exc}"
        )
