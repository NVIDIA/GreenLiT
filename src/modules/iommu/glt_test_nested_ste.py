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
Regression test for arm-smmu-v3 nested STE S1DSS-bypass update sequencing.

Exercises both halves of the nested STE update-safety fix by toggling STE
CFG (S1DSS-bypass <-> S2-bypass) and STE.EATS simultaneously, so a kernel
missing either half of the series still fails:

Patch series (all Jason Gunthorpe, arm_smmu_get_ste_update_safe()):
  1. https://git.kernel.org/torvalds/c/2781f2a930ab
     "iommu/arm-smmu-v3: Add update_safe bits to fix STE update sequence"
     (introduces arm_smmu_get_ste_update_safe(); no relaxation yet)
  2. https://git.kernel.org/torvalds/c/f3c1d372dbb8
     "iommu/arm-smmu-v3: Mark STE MEV safe when computing the update sequence"
     (MEV half -- exercised by the CFG toggle alone)
  3. https://git.kernel.org/torvalds/c/7cad80048595
     "iommu/arm-smmu-v3: Mark EATS_TRANS safe when computing the update sequence"
     (EATS half -- exercised by the EATS_TRANS toggle riding on the same
     CFG change, mirroring that commit's "VM toggles EATS_TRANS off at the
     same time as changing the CFG" scenario)

Which half a given run actually exercises depends on the endpoint: the EATS
half is only reachable when pciutils detects ATS capability, because otherwise
the host may force the physical STE.EATS to 0 regardless of the guest STE
request (arm_smmu_ats_supported()). When ATS is not detected the test still
exercises the MEV / S1DSS-bypass CFG (Config 6<->7) transition, which is the
half commit f3c1d372dbb8 fixes. Discovery therefore prefers endpoints where
pciutils detects ATS but falls back to the remaining endpoints so the CFG half
still runs.

Plan: disruptive and stress (binds vfio-pci and runs bpftrace; both decorators
below register it on those two plans).

The companion bpftrace observer ``nested_ste/observe_ste.bt`` is licensed
GPL-2.0 (bpftrace scripts link the GPL-only kernel BPF ABI); it ships beside
this Apache-2.0 module as a separate, independently licensed artifact.
"""

import glob
import gzip
import os
import platform
import re
import signal
import subprocess
import sys
import time

from common import glt_core as GLTCore
from common import glt_utils
from common import pci_utils
from modules.iommu.nested_ste import trigger
from scripts import check_kernel_config

NESTED_STE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nested_ste")
BPFTRACE_SCRIPT = "observe_ste.bt"
ITERATIONS = 32
TRIGGER_TIMEOUT_SEC = 120

# PCI class prefix for display controllers (GPUs), as read from sysfs
# `class` (e.g. "0x030000" for a VGA controller).
_PCI_CLASS_DISPLAY_PREFIX = "0x03"

# PCI class prefix for bridge devices (e.g. "0x060400" for a PCI-to-PCI
# bridge / root port), as read from sysfs `class`.
_PCI_CLASS_BRIDGE_PREFIX = "0x06"

_BDF_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")

def _parse_running_kconfig():
    """Return the running kernel's config as a {CONFIG_x: value} dict.

    Tries the on-disk /boot/config-<release> first, then falls back to the
    kernel's own compressed /proc/config.gz (CONFIG_IKCONFIG_PROC) when the
    /boot copy is absent or unreadable -- container/appliance images often
    ship only the latter. Returns {} if neither source can be read."""
    try:
        return check_kernel_config.parse_kconfig(
            "/boot/config-" + platform.uname().release)
    except OSError:
        pass
    try:
        with gzip.open("/proc/config.gz", "rt", encoding="latin-1") as fh:
            content = fh.read()
    except OSError:
        return {}
    config = {}
    for line in content.splitlines():
        match = re.match(r"(CONFIG_\w+)=(.*)", line.strip())
        if match:
            config[match.group(1)] = match.group(2).strip()
    return config


def _skip_reason_kprobe_targets(symbols=None):
    # Missing kprobe target symbols mean the observer cannot attach at all --
    # that is "can't observe" (an environment/observability gap, like a
    # missing bpftrace), not a kernel defect, so it must SKIP rather than FAIL.
    if symbols is None:
        with open("/proc/kallsyms", "r", encoding="utf-8") as fh:
            symbols = fh.read()
    for sym in ("arm_smmu_write_entry", "arm_smmu_ste_writer_sync_entry"):
        if sym not in symbols:
            return (f"kprobe target {sym} missing from /proc/kallsyms "
                    "(cannot observe STE writes; environment gap, not a kernel defect)")
    return None


def _skip_reason_endianness():
    if sys.byteorder != "little":
        return ("nested STE trigger requires a little-endian host for STE "
                "qword layout")
    return None


def _is_arm_smmu_v3_backed(bdf):
    link = f"/sys/bus/pci/devices/{bdf}/iommu"
    if not os.path.islink(link):
        return False
    target = os.path.realpath(link)
    for smmu_node in glob.glob("/sys/devices/platform/arm-smmu-v3.*"):
        if target.startswith(os.path.realpath(smmu_node)):
            return True
    return False


def _current_driver(bdf):
    link = f"/sys/bus/pci/devices/{bdf}/driver"
    if not os.path.islink(link):
        return None
    return os.path.basename(os.path.realpath(link))


def _display_device_is_safe(bdf):
    """Return whether an idle display device is safe to bind to vfio-pci."""
    path = f"/sys/bus/pci/devices/{bdf}"
    if _current_driver(bdf) is not None or not os.path.isfile(f"{path}/reset"):
        return False
    try:
        boot_vga = f"{path}/boot_vga"
        if os.path.isfile(boot_vga) and glt_utils.read_sysfs_value(boot_vga) != "0":
            return False
        group_bdfs = {
            os.path.basename(device)
            for device in glob.glob(f"{path}/iommu_group/devices/*")
        }
        if group_bdfs != {bdf}:
            return False
        device_path = os.path.realpath(path)
        for framebuffer in glob.glob("/sys/class/graphics/fb*/device"):
            if os.path.commonpath(
                    (device_path, os.path.realpath(framebuffer))) == device_path:
                return False
    except (OSError, ValueError):
        return False
    return True


def _discover_candidate_bdfs(ats_bdfs, logger=None):
    """Safe vfio-pci endpoints; pciutils-detected ATS first, then by BDF.

    Returns:
        list: Candidate BDFs, or ``None`` when the unsafe-BDF safety probe
        itself failed (fail-closed). An empty list means the probe succeeded
        but no eligible endpoint was found.
    """
    try:
        unsafe = pci_utils.unsafe_pci_bdfs()
    except OSError as exc:
        if logger is not None:
            logger.warning(
                "fail-closed: cannot determine unsafe PCI BDFs (%s); "
                "refusing all candidates",
                exc,
            )
        return None
    candidates = []
    for path in glob.glob("/sys/bus/pci/devices/*"):
        bdf = os.path.basename(path)
        if not _BDF_RE.match(bdf) or bdf.lower() in unsafe:
            continue
        if not os.path.islink(f"{path}/iommu_group"):
            continue
        if not _is_arm_smmu_v3_backed(bdf):
            continue
        pci_class = pci_utils.get_pci_class(bdf)
        if not pci_class:
            continue
        if (pci_class.lower().startswith(_PCI_CLASS_DISPLAY_PREFIX)
                and not _display_device_is_safe(bdf)):
            continue
        if pci_class.lower().startswith(_PCI_CLASS_BRIDGE_PREFIX):
            continue
        if _current_driver(bdf) == "vfio-pci":
            continue
        candidates.append(bdf)
    candidates.sort(key=lambda b: (0 if b.lower() in ats_bdfs else 1, b))
    if logger is not None:
        logger.info("nested_ste candidates (ATS first): %s", candidates)
    return candidates


def _save_driver_state(store, bdf):
    orig_override = ""
    override_path = f"/sys/bus/pci/devices/{bdf}/driver_override"
    if os.path.isfile(override_path):
        with open(override_path, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
        orig_override = "" if raw == "(null)" else raw
    store[bdf] = {
        "orig_driver": _current_driver(bdf) or "",
        "orig_override": orig_override,
    }


def _restore_driver_state(store, bdf, logger):  # pylint: disable=too-many-return-statements
    """Unbind vfio-pci, restore override, drivers_probe once. Returns True on success."""
    state = store.pop(bdf, None)
    if not state or not os.path.exists(f"/sys/bus/pci/devices/{bdf}"):
        return True
    if _current_driver(bdf) == "vfio-pci":
        try:
            with open("/sys/bus/pci/drivers/vfio-pci/unbind", "w", encoding="utf-8") as fh:
                fh.write(bdf)
        except OSError as exc:
            logger.error("failed to unbind vfio-pci from %s: %s", bdf, exc)
            return False
    override_value = state["orig_override"]
    if override_value in ("", "(null)"):
        override_value = "\n"
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/driver_override", "w",
                  encoding="utf-8") as fh:
            fh.write(override_value)
    except OSError as exc:
        logger.error("failed to restore driver_override for %s: %s", bdf, exc)
        return False
    orig_driver = state["orig_driver"]
    if not orig_driver or orig_driver == "vfio-pci":
        return True
    try:
        with open("/sys/bus/pci/drivers_probe", "w", encoding="utf-8") as fh:
            fh.write(bdf)
    except OSError as exc:
        logger.error("drivers_probe failed for %s: %s", bdf, exc)
        return False
    if _current_driver(bdf) != orig_driver:
        logger.error(
            "failed to restore %s to %s (now %s)", bdf, orig_driver,
            _current_driver(bdf) or "none")
        return False
    return True


def _bind_vfio_pci(bdf, logger):  # pylint: disable=too-many-return-statements
    pci_class = pci_utils.get_pci_class(bdf)
    if not pci_class:
        return f"cannot determine PCI class for {bdf}"
    is_display = pci_class.lower().startswith(_PCI_CLASS_DISPLAY_PREFIX)
    if is_display and not _display_device_is_safe(bdf):
        return f"display device {bdf} is not safe to bind"
    subprocess.run(["modprobe", "vfio-pci"], check=False, shell=False)
    cur = _current_driver(bdf)
    if is_display and cur:
        return f"display device {bdf} became bound to {cur}"
    if cur and cur != "vfio-pci":
        try:
            with open(f"/sys/bus/pci/drivers/{cur}/unbind", "w", encoding="utf-8") as fh:
                fh.write(bdf)
        except OSError as exc:
            return f"failed to unbind {cur} from {bdf}: {exc}"
    override_path = f"/sys/bus/pci/devices/{bdf}/driver_override"
    try:
        with open(override_path, "w", encoding="utf-8") as fh:
            fh.write("vfio-pci")
        with open(override_path, "r", encoding="utf-8") as fh:
            readback = fh.read().strip()
    except OSError as exc:
        return f"failed to set/read driver_override for {bdf}: {exc}"
    if readback != "vfio-pci":
        return (f"driver_override readback mismatch for {bdf}: wrote 'vfio-pci', "
                f"read back {readback!r}")
    try:
        with open("/sys/bus/pci/drivers/vfio-pci/bind", "w", encoding="utf-8") as fh:
            fh.write(bdf)
    except OSError as exc:
        return f"failed to bind vfio-pci to {bdf}: {exc}"
    if not os.path.isdir(f"/sys/bus/pci/devices/{bdf}/vfio-dev"):
        return f"vfio-pci bind failed for {bdf}"
    logger.info("bound %s to vfio-pci", bdf)
    return None


def _bind_usable_candidate(candidates, store, logger):
    """Bind the first usable candidate; restore failures before trying the next."""
    errors = []
    for cand in candidates:
        _save_driver_state(store, cand)
        err = _bind_vfio_pci(cand, logger)
        if err is None:
            return cand, errors
        errors.append(f"{cand}: {err}")
        _restore_driver_state(store, cand, logger)
    return None, errors


def _terminate_proc(proc, timeout=10):
    # Prefer a graceful SIGINT so bpftrace runs its END block (printing the
    # final SUMMARY / STE_WRITE lines) and flushes before exiting; only fall
    # back to SIGKILL if it will not exit, since a kill can truncate the last
    # buffered lines into a false "no STE_WRITE" verdict.
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _await_trace_flush(trace_log, attempts=25, interval=0.1):
    """Wait (briefly) for bpftrace's END block to write the SUMMARY marker to
    trace_log, so the verdict never parses a half-flushed log. The process is
    already signaled to exit by the time this runs; this only covers the tail
    where the final lines may still be in flight."""
    for _ in range(attempts):
        try:
            with open(trace_log, "r", encoding="utf-8") as fh:
                if "SUMMARY" in fh.read():
                    return
        except OSError:
            pass
        time.sleep(interval)


def _start_tracing(trace_log, logger):
    with open(trace_log, "w", encoding="utf-8"):
        pass
    bpftrace = os.path.join(NESTED_STE_DIR, BPFTRACE_SCRIPT)
    bpftrace_cmd = ["bpftrace", bpftrace]
    if glt_utils.is_cli_available("stdbuf"):
        bpftrace_cmd = ["stdbuf", "-oL"] + bpftrace_cmd
    with open(trace_log, "a", encoding="utf-8") as logfh:
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            bpftrace_cmd,
            stdout=logfh,
            stderr=subprocess.STDOUT,
            shell=False,
        )
    for _ in range(50):
        if os.path.isfile(trace_log):
            with open(trace_log, "r", encoding="utf-8") as fh:
                content = fh.read()
            # PROBE_LIVE only, not the "Attaching N probes..." banner: that
            # banner prints when bpftrace ANNOUNCES it is about to attach,
            # not after attachment completes -- confirmed live on
            # vr-p5035-3690 as a false readiness signal in MR !326's
            # observer (same bpftrace startup sequence), which this test
            # shares. PROBE_LIVE is this script's own BEGIN action and only
            # prints once bpftrace's startup phase reaches it.
            if "PROBE_LIVE" in content:
                time.sleep(2)
                logger.info("bpftrace live (log: %s)", trace_log)
                return proc, None
        time.sleep(0.2)
    _terminate_proc(proc, timeout=5)
    with open(trace_log, "r", encoding="utf-8") as fh:
        return None, f"bpftrace failed to start (see {trace_log}): {fh.read()}"


def _stop_tracing(proc, trace_log=None):
    if proc is None:
        return
    _terminate_proc(proc, timeout=10)
    if trace_log is not None:
        _await_trace_flush(trace_log)


def _run_trigger(bdf, iters, logger):
    """Run the ioctl trigger. Returns (fail_reason, worker_stuck)."""
    try:
        status = trigger.run_trigger(bdf, iters, timeout=TRIGGER_TIMEOUT_SEC)
    except trigger.TriggerTimeout as exc:
        return str(exc), True
    except trigger.TriggerError as exc:
        return str(exc), False
    logger.info(status)
    return None, False


def _parse_trace(trace_log):
    """Parse the observer log into (disruptive, ste_writes, target_writes).

    - disruptive: transient V=0 rewrites on the transition under test.
    - ste_writes: every STE_WRITE observed (drives the liveness check).
    - target_writes: STE_WRITE events tagged is_target=1, i.e. the physical
      STE Config 6<->7 transition the test actually means to exercise --
      used to prove the transition happened before certifying it hitless."""
    disruptive = 0
    ste_writes = 0
    target_writes = 0
    with open(trace_log, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("STE_WRITE "):
                ste_writes += 1
                is_target = "is_target=1" in line
                if is_target:
                    target_writes += 1
                # Count a transient V=0 as disruptive only on the nested
                # update-safety transition under test (physical STE Config
                # 6<->7). Benign install/teardown writes (is_target=0) pass
                # through V=0 legitimately even on a patched kernel, so
                # gating on is_target avoids a false FAIL. ste_writes still
                # tallies every write, so the liveness check is unaffected.
                if "v_zero=1" in line and is_target:
                    disruptive += 1
    return disruptive, ste_writes, target_writes


def _nested_ste_cleanup(_sd, attr, logger):
    # Backstop cleanup hook. Always stop the (separate) bpftrace process,
    # which is safe regardless of trigger state.
    proc = getattr(attr, "nested_ste_trace_proc", None)
    if proc is not None:
        _stop_tracing(proc)
    # If the trigger worker is stuck in a kernel ioctl it still holds
    # /dev/iommu + vfio fds; touching the device (unbind/rebind) would race
    # that in-flight ioctl, so leave it alone (a reboot is recommended).
    if getattr(attr, "nested_ste_worker_stuck", False):
        return
    bdf = getattr(attr, "nested_ste_bdf", None)
    store = getattr(attr, "driver_state", None)
    if bdf and store is not None:
        _restore_driver_state(store, bdf, logger)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_nested_ste_hitless_update(sd, logger):
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    """Nested arm-smmu-v3 STE updates must stay hitless across S1DSS-bypass toggles.

    Exercises iommufd nested HWPT attach/replace (cfg=S1_TRANS+S1DSS_BYPASS
    versus cfg=BYPASS), with an EATS_TRANS toggle riding on the same CFG
    change, and uses bpftrace to detect disruptive STE.V=0 rewrites.
    Unpatched kernels take the disruptive path (missing either the MEV or
    the EATS safe-bit relaxation); fully patched kernels remain hitless.

    Registered on the disruptive and stress plans: requires vfio-pci bind and
    bpftrace, and is intended for IOMMU regression validation.

    Patch series: https://git.kernel.org/torvalds/c/2781f2a930ab
    (arm_smmu_get_ste_update_safe) -> https://git.kernel.org/torvalds/c/f3c1d372dbb8
    (MEV) -> https://git.kernel.org/torvalds/c/7cad80048595 (EATS)
    """
    test_func = test_nested_ste_hitless_update

    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("Not ARM64, skip the test", func=test_func)
    if not os.path.exists("/dev/iommu"):
        GLTCore.GLTTestResult.skip(
            "no /dev/iommu (nested HWPT / iommufd not available)", func=test_func)

    if not os.path.isdir("/sys/bus/platform/drivers/arm-smmu-v3"):
        GLTCore.GLTTestResult.skip(
            "no arm-smmu-v3 platform driver (not in affected family)", func=test_func)
    if not glob.glob("/sys/devices/platform/arm-smmu-v3.*"):
        GLTCore.GLTTestResult.skip(
            "arm-smmu-v3 driver present but no SMMU instances probed", func=test_func)

    # Accept both "=y" (built-in) and "=m" (module): a module build still
    # provides CONFIG_ARM_SMMU_V3_IOMMUFD once loaded, so it must not SKIP.
    kconfig = _parse_running_kconfig()
    if kconfig.get("CONFIG_ARM_SMMU_V3_IOMMUFD") not in ("y", "m"):
        GLTCore.GLTTestResult.skip(
            "kernel lacks CONFIG_ARM_SMMU_V3_IOMMUFD", func=test_func)

    if not glt_utils.is_cli_available("bpftrace"):
        GLTCore.GLTTestResult.skip(
            "required tool bpftrace missing (environment gap, not a kernel defect)",
            func=test_func)
    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.skip(
            "required tool lspci missing; install pciutils "
            "(environment gap, not a kernel defect)",
            func=test_func)
    # Missing kprobe targets mean "cannot observe", not a kernel defect -> SKIP.
    skip_reason = _skip_reason_kprobe_targets()
    if skip_reason:
        GLTCore.GLTTestResult.skip(skip_reason, func=test_func)
    skip_reason = _skip_reason_endianness()
    if skip_reason:
        GLTCore.GLTTestResult.skip(skip_reason, func=test_func)

    ats_bdfs = {
        bdf.lower()
        for bdf in pci_utils.get_bdfs_with_ext_cap(pci_utils.EXT_CAP_ATS)
    }
    trace_log = os.path.join(sd.log_dir, "nested_ste_observe.log")
    test_func.attr.nested_ste_bdf = None
    test_func.attr.nested_ste_trace_proc = None
    test_func.attr.nested_ste_worker_stuck = False
    # Driver-restore backup lives on attr (like CXL cleanup_state / I2C
    # rebind_devices) so it is shared with the post_hooks and never leaks
    # across tests in a long-lived GLT process.
    test_func.attr.driver_state = {}

    bdf = None
    trace_proc = None
    worker_stuck = False
    try:
        candidates = _discover_candidate_bdfs(ats_bdfs, logger=logger)
        if candidates is None:
            GLTCore.GLTTestResult.skip(
                "cannot determine unsafe PCI BDFs (fail-closed; unable to prove "
                "isolation from management NIC / mounted storage)",
                func=test_func)
        if not candidates:
            GLTCore.GLTTestResult.skip(
                "no suitable PCI BDF found (dynamic arm-smmu-v3 discovery found no "
                "safe-to-unbind endpoint)", func=test_func)

        # Iterate candidates (ATS-capable first): a bind/vfio-setup failure on
        # one is an environment condition, not a kernel defect, so restore it
        # and try the next; only if NONE bind do we SKIP.
        bdf, bind_errors = _bind_usable_candidate(
            candidates, test_func.attr.driver_state, logger)
        if bdf is None:
            GLTCore.GLTTestResult.skip(
                "no usable vfio-pci endpoint (environment gap, not a kernel "
                f"defect); tried {len(candidates)} candidate(s): "
                + "; ".join(bind_errors),
                func=test_func)
        test_func.attr.nested_ste_bdf = bdf
        ats_capable = bdf.lower() in ats_bdfs
        logger.info("using PCI device %s (ATS-detected=%s)", bdf, ats_capable)

        trace_proc, fail_reason = _start_tracing(trace_log, logger)
        test_func.attr.nested_ste_trace_proc = trace_proc
        if fail_reason:
            GLTCore.GLTTestResult.fail(fail_reason, func=test_func)

        fail_reason, worker_stuck = _run_trigger(bdf, ITERATIONS, logger)
        if fail_reason:
            if worker_stuck:
                test_func.attr.nested_ste_worker_stuck = True
                GLTCore.GLTTestResult.fail(
                    fail_reason + " -- the trigger worker is stuck holding "
                    "/dev/iommu and vfio fds; leaving the device bound to "
                    "vfio-pci (unbinding it now would race the in-flight "
                    "ioctl). Reboot recommended to recover the device.",
                    func=test_func)
            GLTCore.GLTTestResult.fail(fail_reason, func=test_func)

        _stop_tracing(trace_proc, trace_log)
        test_func.attr.nested_ste_trace_proc = None
        trace_proc = None

        disruptive, ste_writes, target_writes = _parse_trace(trace_log)
        logger.info(
            "TRACE_DISRUPTIVE=%d TRACE_STE_WRITES=%d TRACE_TARGET_WRITES=%d",
            disruptive, ste_writes, target_writes)
        if ste_writes < 1:
            GLTCore.GLTTestResult.fail(
                f"trace not live during trigger (no STE_WRITE events; see {trace_log})",
                func=test_func)
        if disruptive > 0:
            GLTCore.GLTTestResult.fail(
                "disruptive nested STE rewrite detected (transient V=0 during "
                "write_entry) — unpatched behavior; missing MEV and/or EATS "
                "arm_smmu_get_ste_update_safe() relaxation "
                "(https://git.kernel.org/torvalds/c/2781f2a930ab -> "
                "https://git.kernel.org/torvalds/c/f3c1d372dbb8 -> "
                "https://git.kernel.org/torvalds/c/7cad80048595)",
                func=test_func)
        # Liveness of the transition ITSELF: a run that never drove the
        # physical STE Config 6<->7 (target) transition observed 0 disruptive
        # writes vacuously and must NOT PASS. Each iteration drives the target
        # transition twice, so require at least ITERATIONS target writes.
        if target_writes < ITERATIONS:
            GLTCore.GLTTestResult.fail(
                f"transition under test not observed (target Config 6<->7 writes="
                f"{target_writes}, expected >= {ITERATIONS}); the attach/replace "
                f"sequence did not exercise the S1DSS<->S2bypass path, so hitless "
                f"behavior cannot be certified (see {trace_log})",
                func=test_func)

        if ats_capable:
            logger.info(
                "nested S1DSS-bypass/EATS_TRANS STE transitions are hitless across "
                "%d alternations (ATS-capable endpoint; MEV and EATS halves)",
                ITERATIONS)
        else:
            logger.info(
                "nested S1DSS-bypass STE transitions are hitless across "
                "%d alternations (ATS not detected; MEV/CFG half only)",
                ITERATIONS)
    finally:
        # Always terminate the (separate) bpftrace process. Do NOT null
        # nested_ste_bdf here: the post-hooks and this restore share the same
        # backup, mirroring the i2c tests.
        if trace_proc is not None:
            _stop_tracing(trace_proc, trace_log)
            test_func.attr.nested_ste_trace_proc = None
        # Restore the device unless the worker is stuck in a kernel ioctl
        # still holding its fds (touching it would race the in-flight ioctl).
        if bdf is not None and not worker_stuck:
            restore_ok = _restore_driver_state(
                test_func.attr.driver_state, bdf, logger)
            # Surface a restore failure so a botched cleanup is not swallowed
            # into a clean PASS -- but only when the body itself did not
            # already fail/skip (sys.exc_info() empty), so a genuine
            # disruptive/verdict FAIL keeps its own more actionable reason.
            if not restore_ok and sys.exc_info()[0] is None:
                GLTCore.GLTTestResult.fail(
                    f"failed to restore {bdf} to its original driver after an "
                    "otherwise-clean run (left bound to vfio-pci) — reboot "
                    "recommended", func=test_func)


test_nested_ste_hitless_update.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_nested_ste_cleanup,
    postskip=_nested_ste_cleanup,
    postfail=_nested_ste_cleanup,
    postafail=_nested_ste_cleanup,
)
