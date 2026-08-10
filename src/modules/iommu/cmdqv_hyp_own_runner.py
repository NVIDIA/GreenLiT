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

"""Orchestrator for tegra241 CMDQV host VINTF0 supports_cmd regression (803e41f36d22).

Dual-function-probe COUNT discriminator: over a bounded host-activity
window, count calls into two kernel functions (get_cmdq_nonnull:
tegra241_cmdqv_get_cmdq() scoped to the trigger SMMU returning non-NULL;
guest_supports_cmd: scoped calls into tegra241_guest_vcmdq_supports_cmd()
correlated with the trigger SMMU's get_cmdq path)
rather than reading a struct field off a one-shot kretprobe return.
guest_supports_cmd > 0 is the reliable unpatched signal (live hardware
measurement showed get_cmdq_nonnull stuck at 0 under host DMA I/O on an
unfixed kernel). See observe.bt.in for the full evidence/kernel-source
citations and evaluate_result() for the verdict this drives.
"""

# pylint: disable=consider-using-with

from __future__ import annotations

import ctypes
import glob
import logging
import mmap
import os
import platform
import re
import select
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from common import glt_core as GLTCore
from common import glt_utils, pci_utils

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmdqv_hyp_own")
OBSERVE_TEMPLATE = os.path.join(ARTIFACT_DIR, "observe.bt.in")
SYMBOL_LIVENESS = "tegra241_cmdqv_get_cmdq"
# Second probe of the discriminator: called for host commands only on an
# UNFIXED kernel (see module docstring/observe.bt.in). The real observer
# launch validates both symbols; no separate dry-run attach is needed.
SYMBOL_GUEST_SUPPORTS_CMD = "tegra241_guest_vcmdq_supports_cmd"

TRIGGER_NUM_READS = 32768
TRIGGER_MIN_READS = 512
TRIGGER_MAX_SEC = 5
PROBE_READY_TIMEOUT = 30
# Grace after the DMA trigger before SIGINT'ing bpftrace. Counts accumulate
# during the trigger itself; this window only covers DMA-FQ deferred flush
# (IOVA_FQ_TIMEOUT is ~10ms). Elapsing is the expected completion path: the
# runner sends SIGINT, bpftrace's END{} prints CMDQV_COUNTS, and the process
# exits. No CMDQV_COUNTS line is treated as liveness not met.
RESULT_TIMEOUT_SEC = 2
# Bound leftover bpftrace teardown after kill so a hung observer cannot
# stall the suite indefinitely.
BPFTRACE_TEARDOWN_TIMEOUT_SEC = 5

# get_cmdq_calls is diagnostic-only (total scoped calls, printed so the
# both-zero liveness case is debuggable) -- it plays no part in
# evaluate_result()'s verdict. Any scoped guest_supports_cmd call is the
# regression signal; get_cmdq_nonnull establishes the fixed-kernel PASS path.
COUNTS_RE = re.compile(
    r"^CMDQV_COUNTS get_cmdq_calls=(?P<get_cmdq_calls>\d+) "
    r"get_cmdq_nonnull=(?P<get_cmdq_nonnull>\d+) "
    r"guest_supports_cmd=(?P<guest_supports_cmd>\d+)$"
)


class _CandidateNotLive(Exception):
    """Internal control flow: this NVMe candidate did not meet the liveness gate.

    Not a test result. It only tells ``run_cmdqv_hyp_own_test()`` to move on to
    the next candidate; the loop decides the verdict once every candidate has
    been tried. Skip and fail are ``GLTCore.GLTTestSkip`` / ``GLTTestFail``,
    raised here and converted to a result at the ``@GLTTestAttr``-decorated
    entry point in glt_test_iommu.py.
    """


@dataclass
class CmdqvCounts:
    """Anti-correlated call counts from the dual-function-probe observer.

    ``get_cmdq_nonnull`` counts scoped tegra241_cmdqv_get_cmdq() returns
    that were non-NULL; ``guest_supports_cmd`` counts calls into
    tegra241_guest_vcmdq_supports_cmd() that occur while the scoped
    get_cmdq probe holds ``@in_scope[tid]``. No struct fields are read to
    produce either count -- see observe.bt.in for the full discriminator
    rationale and the live-hardware measurement backing it.

    ``get_cmdq_calls`` is the total scoped tegra241_cmdqv_get_cmdq() call
    count (diagnostic only -- helps distinguish "the trigger never reached
    the scoped SMMU at all" from "it reached it but retval was always
    NULL" in the both-zero liveness case). It plays no part in
    evaluate_result()'s verdict.
    """

    get_cmdq_nonnull: int
    guest_supports_cmd: int
    get_cmdq_calls: int = 0


def kallsyms_has(symbol: str) -> bool:
    """Return whether ``symbol`` is present in the running kernel's kallsyms."""
    try:
        with open("/proc/kallsyms", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3 and parts[2] == symbol:
                    return True
    except OSError:
        return False
    return False


def acpi_cmdqv_present() -> bool:
    """Return whether ACPI NVDA200C CMDQV hardware is present."""
    if glob.glob("/sys/bus/acpi/devices/NVDA200C:*"):
        return True
    if glob.glob("/sys/devices/platform/NVDA200C:*"):
        return True
    try:
        with open("/proc/iomem", encoding="utf-8", errors="ignore") as handle:
            if "NVDA200C" in handle.read():
                return True
    except OSError:
        pass
    return False


def cmdqv_admin_disabled_reason() -> str:
    """Return a semicolon-separated reason when CMDQV is administratively disabled."""
    reasons = []
    disable_path = "/sys/module/arm_smmu_v3/parameters/disable_cmdqv"
    bypass_path = "/sys/module/arm_smmu_v3/parameters/bypass_vcmdq"
    try:
        if os.path.isfile(disable_path):
            with open(disable_path, encoding="utf-8") as handle:
                if handle.read().strip() != "N":
                    reasons.append("disable_cmdqv=Y")
    except OSError:
        pass
    try:
        if os.path.isfile(bypass_path):
            with open(bypass_path, encoding="utf-8") as handle:
                if handle.read().strip() != "N":
                    reasons.append("bypass_vcmdq=Y")
    except OSError:
        pass
    return "; ".join(reasons)


def running_as_guest() -> bool:
    """Return whether this kernel appears to be running under a hypervisor.

    Fixed kernels inside a guest correctly install tegra241_guest_vcmdq_supports_cmd
    (HYP_OWN reads back 0). That produces the same guest_supports_cmd>0 signal as
    an unpatched host, so the discriminator is only valid on bare metal.
    """
    if glt_utils.is_cli_available("systemd-detect-virt"):
        try:
            result = subprocess.run(
                ["systemd-detect-virt", "--quiet", "--vm"],
                capture_output=True,
                check=False,
            )
        except OSError:
            result = None
        if result is not None:
            if result.returncode == 0:
                return True
            if result.returncode == 1:
                return False
    try:
        with open("/sys/hypervisor/type", encoding="utf-8", errors="ignore") as handle:
            hyp = handle.read().strip().lower()
        # Non-empty type other than "none" indicates a hypervisor presence.
        # Prefer systemd-detect-virt above when available (handles Dom0/etc.).
        if hyp and hyp != "none":
            return True
    except OSError:
        pass
    try:
        with open("/sys/class/dmi/id/sys_vendor", encoding="utf-8", errors="ignore") as handle:
            vendor = handle.read().strip().lower()
        if vendor in ("qemu", "kvm", "vmware, inc.", "innotek gmbh", "xen"):
            return True
    except OSError:
        pass
    return False


def check_btf_available() -> None:
    """Skip when bpftrace cannot resolve driver structs against kernel BTF."""
    if not os.path.isfile("/sys/kernel/btf/vmlinux"):
        raise GLTCore.GLTTestSkip(
            "kernel BTF unavailable (/sys/kernel/btf/vmlinux missing) — "
            "observer needs struct arm_smmu_device for SMMU scoping"
        )


def check_global_iommu_translated() -> None:
    """Skip when the kernel cmdline disables translating IOMMU DMA."""
    try:
        with open("/proc/cmdline", encoding="utf-8", errors="ignore") as handle:
            cmdline = handle.read()
    except OSError as exc:
        raise GLTCore.GLTTestFail(f"cannot read /proc/cmdline: {exc}") from exc
    if "iommu.passthrough=1" in cmdline:
        raise GLTCore.GLTTestSkip(
            "global iommu.passthrough=1 — translating DMA path not exercisable"
        )
    if re.search(r"\biommu=pt\b", cmdline) or re.search(r"\biommu=off\b", cmdline):
        raise GLTCore.GLTTestSkip(
            "global iommu=pt/off — translating DMA path not exercisable"
        )


def check_preconditions(logger: logging.Logger) -> None:
    """Validate platform applicability, tooling, and observer prerequisites."""
    if platform.machine() != "aarch64":
        raise GLTCore.GLTTestSkip("not aarch64 (tegra241 CMDQV is aarch64-only)")

    if not acpi_cmdqv_present():
        raise GLTCore.GLTTestSkip(
            "affected CMDQV family absent (no NVDA200C ACPI CMDQV)"
        )

    # Not root / bpftrace missing are environment-unsuitable ("cannot
    # evaluate") conditions on an otherwise-affected platform, not evidence
    # of a kernel regression -- SKIP rather than FAIL (DISP-001).
    if os.geteuid() != 0:
        raise GLTCore.GLTTestSkip("must run as root")
    if not shutil.which("bpftrace"):
        raise GLTCore.GLTTestSkip("bpftrace not installed")
    if not os.path.isfile("/proc/kallsyms"):
        raise GLTCore.GLTTestFail("cannot read /proc/kallsyms")

    disabled = cmdqv_admin_disabled_reason()
    if disabled:
        raise GLTCore.GLTTestSkip(
            "affected CMDQV hardware present but path disabled "
            f"({disabled}); cannot evaluate discriminator"
        )

    if running_as_guest():
        raise GLTCore.GLTTestSkip(
            "running as a guest — fixed kernels correctly use "
            "tegra241_guest_vcmdq_supports_cmd (HYP_OWN=0); host-only discriminator"
        )

    check_btf_available()
    check_global_iommu_translated()

    # Both probes must resolve in kallsyms. A missing symbol on a confirmed
    # NVDA200C platform is a real FAIL: the discriminator cannot fire without it.
    for symbol in (SYMBOL_LIVENESS, SYMBOL_GUEST_SUPPORTS_CMD):
        if not kallsyms_has(symbol):
            raise GLTCore.GLTTestFail(f"{symbol} absent from kallsyms on CMDQV platform")
    if not os.path.isfile(OBSERVE_TEMPLATE):
        raise GLTCore.GLTTestFail("observe.bt.in missing")

    logger.info("[cmdqv-hyp-own] platform: ACPI NVDA200C")


# Generic PCI/IOMMU sysfs primitives (pci_bdfs_backing_mount,
# sysfs_path_to_pci_bdf, bdf_in_translated_domain) live in common.pci_utils
# and are used directly — they carry no cmdqv-specific logic.


def _smmu_platform_name_from_iommu_path(iommu_path: str) -> Optional[str]:
    """Return the platform sysfs name that matches ``dev->kobj.name`` for bpftrace.

    Walk upward from the resolved ``iommu`` MMIO node and return the first
    ancestor whose ``driver`` symlink basename is ``arm-smmu-v3``. That
    directory name is what ``arm_smmu_device->dev->kobj.name`` exposes, so it
    stays correct for ACPI (``arm-smmu-v3.N[.auto]``), DT
    (``<reg>.arm-smmu-v3`` / ``<reg>.iommu``), and other naming schemes.
    Fall back to a name heuristic when ``driver`` links are absent (unit
    fixtures / incomplete sysfs).
    """
    path = iommu_path
    while path and path != "/":
        driver_link = os.path.join(path, "driver")
        try:
            if os.path.islink(driver_link) or os.path.exists(driver_link):
                driver = os.path.basename(os.path.realpath(driver_link))
                if driver == "arm-smmu-v3":
                    return os.path.basename(path)
        except OSError:
            pass
        path = os.path.dirname(path)

    path = iommu_path
    while path and path != "/":
        base = os.path.basename(path)
        if (
            base.startswith("arm-smmu-v3.")
            or base.endswith(".arm-smmu-v3")
            or ".arm-smmu-v3." in base
        ):
            return base
        path = os.path.dirname(path)
    return None


def resolve_smmu_identity_for_bdf(bdf: str) -> Tuple[str, str]:
    """Resolve the arm-smmu-v3 platform name and MMIO node for ``bdf``."""
    iommu_link = f"/sys/bus/pci/devices/{bdf}/iommu"
    if not os.path.exists(iommu_link):
        raise GLTCore.GLTTestFail(f"PCI_BDF={bdf} has no iommu symlink")
    iommu_path = os.path.realpath(iommu_link)
    platform_name = _smmu_platform_name_from_iommu_path(iommu_path)
    mmio = os.path.basename(iommu_path)
    if platform_name is None:
        raise GLTCore.GLTTestFail(f"PCI_BDF={bdf} SMMU is not arm-smmu-v3 ({iommu_path})")
    if not mmio.startswith("smmu3.0x"):
        raise GLTCore.GLTTestFail(f"PCI_BDF={bdf} unexpected SMMU MMIO node ({mmio})")
    return platform_name, mmio


def resolve_nvme_block_from_bdf(bdf: str) -> str:
    """Return the first resolvable NVMe namespace block device for ``bdf``."""
    for nvme_ctrl in sorted(glob.glob(f"/sys/bus/pci/devices/{bdf}/nvme/nvme*")):
        for ns in sorted(glob.glob(os.path.join(nvme_ctrl, "nvme*n1"))):
            dev = f"/dev/{os.path.basename(ns)}"
            if os.path.exists(dev):
                return dev
    raise GLTCore.GLTTestFail(f"PCI_BDF={bdf} has no resolvable block namespace")


def enumerate_nvme_candidates() -> List[str]:
    """Return BDF-ordered NVMe controllers behind CMDQV SMMUs in DMA domains."""
    candidates = []
    for bdf_path in sorted(glob.glob("/sys/bus/pci/devices/*")):
        bdf = os.path.basename(bdf_path)
        full_class = pci_utils.get_pci_class(bdf)
        if not full_class or full_class[2:6] != "0108":
            continue
        if not pci_utils.bdf_in_translated_domain(bdf):
            continue
        try:
            resolve_nvme_block_from_bdf(bdf)
            resolve_smmu_identity_for_bdf(bdf)
        except GLTCore.GLTTestFail:
            continue
        candidates.append(bdf)
    return candidates


def _trigger_dma_alarm_handler(_signum, _frame):
    """SIGALRM handler for the trigger's ITIMER_REAL cutoff.

    Raising (rather than returning) is what makes a blocking ``readv`` propagate
    the interruption under PEP 475 instead of being restarted. The caller treats
    the TimeoutError as a soft cutoff and discards it, so no context about which
    device or bound expired needs to travel with it.
    """
    raise TimeoutError("trigger_dma: wall-clock cutoff reached")


def run_trigger_dma(
    block_dev: str,
    num_reads: int,
    min_reads: int,
    max_sec: int,
) -> None:
    """Issue deterministic O_DIRECT reads on ``block_dev`` to trigger DMA.

    Pure-Python port of the former ``trigger_dma.c`` helper: opens the
    target block device O_DIRECT and issues up to ``num_reads`` page-aligned
    reads, bounded by ``max_sec`` wall-clock seconds, and fails when fewer
    than ``min_reads`` reads complete successfully.

    Runs as a direct in-process call rather than a subprocess with
    file-based IPC. No thread is needed for this to overlap with bpftrace's
    observation window: bpftrace already runs concurrently as its own OS
    process (started via ``subprocess.Popen`` in ``run_observed_trigger``),
    so a synchronous call here still executes concurrently with bpftrace at
    the OS level.
    """
    try:
        # os.O_DIRECT is Linux-only and this function only ever runs on the
        # aarch64 DUT (see check_preconditions). getattr() with a fallback
        # keeps this call-safe (and unit-testable) on non-Linux dev/CI hosts
        # that lack the attribute entirely, rather than AttributeError'ing.
        o_direct = getattr(os, "O_DIRECT", 0)
        fd = os.open(block_dev, os.O_RDONLY | o_direct)
    except OSError as exc:
        raise GLTCore.GLTTestFail(f"trigger_dma: open {block_dev} failed: {exc}") from exc

    ok_reads = 0
    # Interrupt blocking readv via ITIMER_REAL so a wedged device cannot hang
    # the suite (parity with the former C helper's SIGALRM cutoff). A raising
    # handler makes the interrupted syscall propagate under PEP 475.
    deadline = time.monotonic() + max_sec if max_sec > 0 else None
    previous_handler = None
    timer_armed = False

    try:
        if max_sec > 0:
            previous_handler = signal.signal(
                signal.SIGALRM, _trigger_dma_alarm_handler
            )
            signal.setitimer(signal.ITIMER_REAL, max_sec)
            timer_armed = True
        try:
            with mmap.mmap(-1, mmap.PAGESIZE) as buf:
                for _ in range(num_reads):
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    try:
                        n = os.readv(fd, [buf])
                    except OSError as exc:
                        raise GLTCore.GLTTestFail(
                            f"trigger_dma: read on {block_dev} failed: {exc}"
                        ) from exc
                    if n == 0:
                        break
                    ok_reads += 1
        except TimeoutError:
            # ITIMER_REAL can fire outside readv; treat as soft cutoff.
            pass
    finally:
        if timer_armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
        os.close(fd)

    if ok_reads < min_reads:
        raise GLTCore.GLTTestFail(
            f"trigger_dma: only {ok_reads} successful reads on {block_dev} "
            f"(need {min_reads})"
        )


def write_observe_script(
    target_smmu: str,
    out_path: str,
) -> None:
    """Materialize the scoped bpftrace observer script for one SMMU target.

    Only ``@@TARGET_SMMU@@`` is templated: it is the one token observe.bt.in
    actually uses in probe-scoping logic. ``@@TARGET_SMMU_MMIO@@`` was a
    dead, informational-only placeholder (never referenced by any probe
    predicate) and has been removed from the template.
    """
    with open(OBSERVE_TEMPLATE, encoding="utf-8") as template:
        content = template.read()
    content = content.replace("@@TARGET_SMMU@@", target_smmu)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(out_path, 0o755)


def parse_cmdqv_counts(output: str) -> Optional[CmdqvCounts]:
    """Parse the CMDQV_COUNTS line emitted by the dual-function-probe observer.

    The observer prints this line exactly once, from its END{} block, when
    the runner SIGINTs it at the end of the observation window (see
    RESULT_TIMEOUT_SEC and observe.bt.in). The last matching line is used
    in case the observer output ever contains more than one -- this stays
    correct in that case without the runner needing to change.
    """
    result = None
    for line in output.splitlines():
        match = COUNTS_RE.match(line.strip())
        if match:
            result = CmdqvCounts(
                get_cmdq_nonnull=int(match.group("get_cmdq_nonnull")),
                guest_supports_cmd=int(match.group("guest_supports_cmd")),
                get_cmdq_calls=int(match.group("get_cmdq_calls")),
            )
    return result


def wait_for_probe_ready(
    proc: "subprocess.Popen[bytes]",
    timeout_sec: int,
) -> str:
    """Block until bpftrace emits PROBE_READY or fail definitively.

    PROBE_READY is emitted only after every probe is attached. A plain
    process timeout cannot establish that ordering and could let the DMA
    trigger race bpftrace startup. Read the combined stdout/stderr pipe via
    ``os.read()`` so buffered text I/O cannot hide readiness from
    ``select()``.
    """
    output_buf = b""
    output_fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            remaining_out, _ = proc.communicate()
            output_buf += remaining_out or b""
            output_text = output_buf.decode("utf-8", errors="replace").strip()
            lowered = output_text.lower()
            if any(
                marker in lowered
                for marker in (
                    "unknown struct/union",
                    "unknown type",
                    "no btf found",
                    "failed to resolve",
                )
            ):
                raise GLTCore.GLTTestSkip(
                    "bpftrace could not resolve types against kernel BTF:\n"
                    f"{output_text}"
                )
            raise GLTCore.GLTTestFail(
                "bpftrace observer exited before PROBE_READY "
                f"(rc={proc.returncode}):\n{output_text or '(no output)'}"
            )

        readable, _, _ = select.select([output_fd], [], [], 0.5)
        if readable:
            try:
                output_buf += os.read(output_fd, 65536)
            except OSError:
                pass

        output_text = output_buf.decode("utf-8", errors="replace")
        if "PROBE_READY" in output_text:
            return output_text

    output_text = output_buf.decode("utf-8", errors="replace").strip()
    detail = f":\n{output_text}" if output_text else ""
    raise GLTCore.GLTTestFail(
        f"observer did not attach probes within {timeout_sec}s{detail}"
    )


def _set_parent_death_signal() -> None:
    """Best-effort: ask the kernel to SIGKILL this process if its parent dies.

    Runs in the forked child, before exec (see ``subprocess.Popen``'s
    ``preexec_fn``). Linux-only (``PR_SET_PDEATHSIG`` via ``prctl(2)``); a
    silent no-op anywhere the ``prctl`` symbol or a matching libc is
    unavailable (e.g. local dev/CI hosts). This is what makes the bpftrace
    observer self-terminate if the GreenLiT test process itself is killed
    out from under it (e.g. a harness SIGKILL or timeout) instead of being
    orphaned with kprobes still attached -- GreenLiT's own SIGINT/SIGTERM
    handler calls ``os._exit(0)`` immediately, without running any
    Python-level cleanup.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL)
    except OSError:
        pass


def _kill_bpftrace_proc(proc: "subprocess.Popen[bytes]") -> None:
    """SIGTERM then SIGKILL the observer process group; best-effort."""
    teardown = BPFTRACE_TEARDOWN_TIMEOUT_SEC
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.communicate(timeout=teardown)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.communicate(timeout=teardown)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
        proc.communicate(timeout=teardown)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_observed_trigger(  # pylint: disable=too-many-branches
    bdf: str,
    logger: logging.Logger,
) -> Optional[CmdqvCounts]:
    """Run the scoped dual-function-probe observer and DMA trigger for one NVMe PCI BDF."""
    block = resolve_nvme_block_from_bdf(bdf)
    target_smmu, target_mmio = resolve_smmu_identity_for_bdf(bdf)

    with tempfile.TemporaryDirectory(prefix="glt-cmdqv-") as workdir:
        probe_script = os.path.join(workdir, "observe.bt")

        write_observe_script(target_smmu, probe_script)

        logger.info(
            "[cmdqv-hyp-own] observe PCI_BDF=%s block=%s smmu=%s mmio=%s",
            bdf,
            block,
            target_smmu,
            target_mmio,
        )

        proc = subprocess.Popen(  # pylint: disable=subprocess-popen-preexec-fn
            ["bpftrace", probe_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_set_parent_death_signal,
        )
        try:
            output_prefix = wait_for_probe_ready(
                proc, PROBE_READY_TIMEOUT
            )

            # Probes are attached; drive the scoped liveness trigger so
            # the dual-function-probe counters have host DMA activity to
            # observe. A candidate-local trigger I/O failure here (bad
            # open/read on this one BDF's block device) does not mean
            # the discriminator itself is broken -- map it to the same
            # retryable liveness path as "no CMDQV_COUNTS observed" so a
            # single flaky NVMe candidate cannot abort the whole test
            # when a later candidate could still drive CMDQV (DISP-007).
            try:
                run_trigger_dma(
                    block,
                    TRIGGER_NUM_READS,
                    TRIGGER_MIN_READS,
                    TRIGGER_MAX_SEC,
                )
            except (GLTCore.GLTTestFail, TimeoutError) as exc:
                raise _CandidateNotLive(
                    f"trigger I/O failed on PCI_BDF={bdf} candidate: {exc}"
                ) from exc

            try:
                output_suffix, _ = proc.communicate(
                    timeout=RESULT_TIMEOUT_SEC
                )
                observer_exited_early = True
            except subprocess.TimeoutExpired:
                observer_exited_early = False
                # Expected: the observer counts for the whole window and
                # does not self-exit(), so it is still running when this
                # timeout elapses. Interrupt cleanly so its END{} block
                # prints the single CMDQV_COUNTS line before it exits.
                proc.send_signal(signal.SIGINT)
                try:
                    # Independent of attach/trigger wall time so a slow
                    # but allowed attach cannot starve END{} drain into a
                    # hard FAIL.
                    output_suffix, _ = proc.communicate(
                        timeout=BPFTRACE_TEARDOWN_TIMEOUT_SEC
                    )
                except subprocess.TimeoutExpired as exc:
                    _kill_bpftrace_proc(proc)
                    raise GLTCore.GLTTestFail(
                        f"bpftrace observer hung after trigger: {exc}"
                    ) from exc
        except Exception:
            if proc.poll() is None:
                _kill_bpftrace_proc(proc)
            raise

        output_suffix_text = (output_suffix or b"").decode("utf-8", errors="replace")
        observer_output = output_prefix + output_suffix_text

        for line in observer_output.splitlines():
            if line.startswith(("Attaching", "PROBE_", "CMDQV_COUNTS", "ERROR", "error:")):
                logger.info("[cmdqv-hyp-own] bpftrace: %s", line)

        if observer_exited_early:
            raise GLTCore.GLTTestFail(
                "bpftrace observer exited before the observation window ended "
                f"(rc={proc.returncode}):\n{observer_output.strip()}"
            )
        if any(
            line.strip().startswith(("ERROR", "error:"))
            or " ERROR:" in line
            for line in observer_output.splitlines()
        ):
            raise GLTCore.GLTTestFail(
                f"bpftrace observer reported an error:\n{observer_output.strip()}"
            )

        result = parse_cmdqv_counts(observer_output)
        if result is not None:
            logger.info(
                "[cmdqv-hyp-own] CMDQV_COUNTS PCI_BDF=%s get_cmdq_calls=%d "
                "get_cmdq_nonnull=%d guest_supports_cmd=%d",
                bdf,
                result.get_cmdq_calls,
                result.get_cmdq_nonnull,
                result.guest_supports_cmd,
            )
            return result

        # Honor a completed counts line above even if bpftrace's exit status
        # is version-fragile after SIGINT. Only gate on rc when no counts.
        if proc.returncode not in (0, -signal.SIGINT, 128 + signal.SIGINT):
            raise GLTCore.GLTTestFail(
                "bpftrace observer failed without CMDQV_COUNTS "
                f"(rc={proc.returncode}):\n{observer_output.strip()}"
            )
        logger.info(
            "[cmdqv-hyp-own] no CMDQV_COUNTS observed for PCI_BDF=%s", bdf
        )
        return None


def evaluate_result(result: Optional[CmdqvCounts]) -> None:
    """Apply the dual-function-probe count verdict.

    guest_supports_cmd > 0                           -> FAIL (unpatched host)
    guest_supports_cmd == 0 and get_cmdq_nonnull > 0 -> PASS (fixed host)
    both counts 0, or no CMDQV_COUNTS observed       -> liveness not met (retryable)

    ``check_preconditions()`` excludes guests before the observer starts, so
    every scoped guest callback call is a regression signal on the bare-metal
    host regardless of whether an invalidation command also returns a non-NULL
    cmdq during the observation window.
    """
    if result is None:
        raise _CandidateNotLive(
            "no CMDQV_COUNTS observed — trigger did not exercise the scoped "
            f"{SYMBOL_LIVENESS} / {SYMBOL_GUEST_SUPPORTS_CMD} probes"
        )
    if result.guest_supports_cmd > 0:
        raise GLTCore.GLTTestFail(
            f"tegra241_guest_vcmdq_supports_cmd() was called "
            f"{result.guest_supports_cmd} time(s) for host commands "
            "(unpatched kernel; see commit 803e41f36d22)"
        )
    if result.get_cmdq_nonnull > 0:
        return
    raise _CandidateNotLive(
        "both get_cmdq_nonnull and guest_supports_cmd are 0 — trigger did "
        f"not exercise the scoped {SYMBOL_LIVENESS} / "
        f"{SYMBOL_GUEST_SUPPORTS_CMD} probes"
    )


def run_cmdqv_hyp_own_test(logger: logging.Logger) -> None:
    """Run the validated CMDQV supports_cmd regression and return on pass."""
    check_preconditions(logger)
    candidates = enumerate_nvme_candidates()
    if not candidates:
        raise GLTCore.GLTTestSkip(
            "CMDQV platform present but no DMA/DMA-FQ NVMe trigger candidate found"
        )

    # Prefer non-root storage, but keep root-backed NVMe as a safe fallback:
    # this trigger only issues read-only O_DIRECT reads. Discovery happens once,
    # and the BDF tie-breaker keeps both partitions stable across runs.
    try:
        root_bdfs = pci_utils.pci_bdfs_backing_mount("/")
    except OSError:
        root_bdfs = set()
    candidates = sorted(candidates, key=lambda bdf: (bdf in root_bdfs, bdf))
    if root_bdfs:
        logger.info(
            "[cmdqv-hyp-own] root-backed NVMe candidates ordered last: %s",
            " ".join(sorted(root_bdfs & set(candidates))),
        )
    logger.info(
        "[cmdqv-hyp-own] NVMe candidates (non-root first, BDF-ordered): %s",
        " ".join(candidates),
    )

    last_liveness_error = ""
    for bdf in candidates:
        logger.info("[cmdqv-hyp-own] trying PCI_BDF=%s", bdf)
        try:
            result = run_observed_trigger(bdf, logger)
            evaluate_result(result)
            return
        except _CandidateNotLive as exc:
            last_liveness_error = str(exc)
            logger.info(
                "[cmdqv-hyp-own] causal liveness not met on %s: %s", bdf, exc
            )
            continue

    if last_liveness_error:
        # Environment-unsuitable/"cannot evaluate" (DISP-001), not a kernel
        # regression: this is ambiguous by construction, it can mean the
        # O_DIRECT trigger never routed any command through the scoped
        # VCMDQ path on any candidate (e.g. VINTF0/VCMDQ not enabled),
        # which produces the same zero counts on a fixed or an unpatched
        # kernel alike -- this failure alone does not distinguish them.
        raise GLTCore.GLTTestSkip(
            "no NVMe could drive a CMDQV SMMU (both get_cmdq_nonnull and "
            "guest_supports_cmd stayed 0 on every candidate)"
        )
    raise GLTCore.GLTTestFail("no NVMe candidates to exercise")
