# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test ARM coresight.
"""
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from packaging import version
from common import cpu_topology
from common import glt_core as GLTCore
from common import perf_utils
from common.nv_device import NvDev

# DGX-10928/DGX-10931: usable dynamic IDs are 1..0x6F (111). Perf opens one
# cs_etm event per CPU, so a global map fails mmap ENOMEM once CPU count
# exceeds that (need >= 112). Measured on a 352-CPU MGX Vera CX against the
# unfixed kernel 78bf04ba9c94: -C 0-110 (111 CPUs) records cleanly, -C 0-111
# (112 CPUs) fails mmap ENOMEM -- the boundary is exactly this threshold.
_MIN_CPUS_FOR_TRACE_ID_PRESSURE = 112
# Safe-side control: the same command over exactly one fewer CPU. 111 is the
# last count a global ID map can still serve, so this must succeed on fixed and
# unfixed kernels alike before an ENOMEM at 112 may be blamed on the missing
# per-sink backport. Control and trigger differ by one CPU at that
# 111-usable-ID boundary.
_CONTROL_CPU_COUNT = _MIN_CPUS_FOR_TRACE_ID_PRESSURE - 1
_CORESIGHT_DEVICES_DIR = Path("/sys/bus/coresight/devices")
_CS_ETM_EVENT_DIR = Path("/sys/bus/event_source/devices/cs_etm")
_ETM4X_MODULE = "coresight_etm4x"
_MODPROBE_TIMEOUT_SECONDS = 30
_RECORD_TIMEOUT_SECONDS = 300
# Per-CPU ETE/ETM trace sources and TRBE sinks encode their CPU ID in the
# device name (ete0, etm12, trbe12, ...). Match the whole name so unrelated
# CoreSight nodes cannot satisfy the selected-CPU coverage gate.
_ETE_SOURCE_RE = re.compile(r"^(?:ete|etm)(\d+)$")
_TRBE_SINK_RE = re.compile(r"^trbe(\d+)$")
# Both records name their CPUs explicitly with -C, and perf cannot open an
# event on an offline CPU, so the selectable set is the *online* mask.
_CPU_ONLINE_PATH = Path("/sys/devices/system/cpu/online")
# Match only mmap ENOMEM. Bare "Cannot allocate memory" would mis-blame the
# per-sink ID backport for unrelated aux/ring-buffer or host OOM failures.
_MMAP_ENOMEM_RE = re.compile(
    r"failed to mmap.*Cannot allocate memory",
    re.IGNORECASE,
)


def _coresight_cpu_ids(pattern) -> set:
    """Return CPU IDs encoded by matching CoreSight device names."""
    if not _CORESIGHT_DEVICES_DIR.is_dir():
        return set()

    cpu_ids = set()
    for path in _CORESIGHT_DEVICES_DIR.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            cpu_ids.add(int(match.group(1)))
    return cpu_ids


def _ensure_etm4x_loaded(logger) -> set:
    """Load coresight_etm4x when sources are absent; return source CPU IDs."""
    source_cpu_ids = _coresight_cpu_ids(_ETE_SOURCE_RE)
    if source_cpu_ids:
        return source_cpu_ids

    logger.info("No ETE/ETM devices visible; loading %s", _ETM4X_MODULE)
    # check=False on purpose: platforms without CoreSight should reach the
    # cs_etm unsupported skip below rather than turn into a red test. A hung
    # module load is bounded by timeout so it cannot become an unhandled
    # framework error either.
    try:
        result = subprocess.run(
            ["modprobe", _ETM4X_MODULE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_MODPROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Loading %s timed out after %ds; treating as unavailable",
            _ETM4X_MODULE, _MODPROBE_TIMEOUT_SECONDS,
        )
        return set()
    except OSError as err:
        # modprobe missing / not executable: still route to the cs_etm
        # unsupported skip rather than an unhandled framework error.
        logger.warning("Could not launch modprobe for %s: %s", _ETM4X_MODULE, err)
        return set()
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        logger.warning(
            "Failed to load %s: %s", _ETM4X_MODULE, detail or result.returncode
        )
        return set()

    source_cpu_ids = _coresight_cpu_ids(_ETE_SOURCE_RE)
    logger.info("ETE/ETM devices after modprobe: %d", len(source_cpu_ids))
    return source_cpu_ids


@dataclass(frozen=True)
class _RecordOutcome:
    """Result of one cs_etm ``perf record`` invocation."""

    returncode: int
    combined: str
    stderr: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Return True when the record completed successfully."""
        return self.error is None and self.returncode == 0

    @property
    def mmap_enomem(self) -> bool:
        """Return True when the record died on an AUX mmap ENOMEM."""
        return self.error is None and bool(_MMAP_ENOMEM_RE.search(self.combined))


def _run_cs_etm_record(perf_cmd, output_path, logger, cpu_range):
    """Run one ``perf record -e cs_etm//u`` and normalize the outcome.

    ``cpu_range`` is the explicit ``-C`` CPU list. Both the control and the
    trigger pass one, so both run in perf's per-CPU mode and the only thing
    that differs between them is the number of CPUs named.
    """
    cmd = [
        perf_cmd, "record", "-e", "cs_etm//u", "-C", cpu_range,
        "-o", output_path, "--", "true",
    ]

    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_RECORD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as err:
        return _RecordOutcome(
            returncode=-1, combined="", stderr="",
            error=f"timed out after {err.timeout} seconds",
        )
    except OSError as err:
        return _RecordOutcome(
            returncode=-1, combined="", stderr="",
            error=f"failed to launch: {err}",
        )

    combined = f"{result.stdout or ''}\n{result.stderr or ''}"
    logger.debug(
        "perf cs_etm record (cpu_range=%s) rc=%d stdout=%r stderr=%r",
        cpu_range or "all", result.returncode, result.stdout, result.stderr,
    )
    return _RecordOutcome(
        returncode=result.returncode,
        combined=combined,
        stderr=(result.stderr or "").strip(),
    )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_perf_coresight(_, logger):
    """Check ARM coresight tracing are available"""
    # FIXME: DGX-10447 perf seems not enable coresight on ubuntu.
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    # Coresight is targeted to RHEL95 for now. DGX-7793 and DGX-7787
    if "Red Hat" in NvDev.get_release_info().get("NAME") and \
            version.parse(NvDev.get_release_info().get("VERSION_ID")) < \
            version.parse("9.5"):
        GLTCore.GLTTestResult.skip("RHEL supports coresight with " +
                                   "version >= 9.5.")

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    result = perf_utils.run_perf_test_candidates(
        ["arm coresight"], perf_cmd=perf_cmd)
    logger.debug(
        "perf Coresight test status=%s test_name=%s reason=%s",
        result.status.value, result.test_name, result.reason)
    if result.status == perf_utils.PerfTestStatus.PASS:
        return

    # Some perf builds expose CoreSight through cs_etm/sysfs but do not
    # install or list the CoreSight shell selftest. In that case, avoid a
    # false hardware failure and accept this lightweight availability check.
    if result.status == perf_utils.PerfTestStatus.NO_MATCH and \
            perf_utils.has_coresight_fallback_evidence():
        logger.info("No perf Coresight selftest found; using cs_etm/sysfs fallback.")
        return

    GLTCore.GLTTestResult.fail(f"perf Coresight test failed: {result.reason}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_perf_cs_etm_multisocket_trace_id_map(sd, logger):
    """
    JIRA: DGX-10931

    Discriminate per-sink CoreSight trace ID allocation on multi-socket systems.

    Two ``perf record -e cs_etm//u`` runs, both with an explicit ``-C`` list of
    real online CPU IDs, differing by exactly one CPU:

    * control -- the first 111 online CPUs. 111 is the last count a *global*
      trace ID map can serve (usable dynamic IDs are 1..0x6F), so this must
      succeed on fixed and unfixed kernels alike.
    * trigger -- the same 111 CPUs plus one more, 112 total.

    Unfixed expectation: the trigger fails mmap with ENOMEM at 112 while the
    control succeeded at 111.

    Fixed expectation: both records complete, because per-sink ID maps remove
    the global ceiling.

    ENOMEM is a generic errno. Causation rests on the two runs differing by one
    CPU at the 111-usable-ID boundary (same perf command, same per-CPU mode,
    same host). That is consistent with a global trace ID map; it does not
    independently rule out AUX mmap failure. If the 111-CPU control itself
    fails after prerequisites are established, the test fails without citing
    the backport.

    CPU IDs come from ``/sys/devices/system/cpu/online``, not ``0..N-1``:
    ``-C`` runs in perf's per-CPU mode and ``perf_event_open()`` on an offline
    CPU fails, so sparse or partially-offlined numbering must be handled
    explicitly.

    Measured on a 352-CPU MGX Vera CX (present == online == 0-351, 352 ETE,
    352 TRBE), perf from linux-tools 6.17.0-2004::

        record                    unfixed 78bf04ba9c94   fixed 2a95eeed8432
        -C 0-110  (111 CPUs)      rc=0                   rc=0
        -C 0-111  (112 CPUs)      mmap ENOMEM            rc=0

    The unfixed boundary falls exactly at _MIN_CPUS_FOR_TRACE_ID_PRESSURE.

    NOTE: this duplicates no packaged coverage. Upstream's
    ``tools/perf/tests/shell/coresight/test_arm_coresight.sh`` does exercise
    this path, but it is not shipped by ``linux-tools`` on the target platform:
    on both proof kernels ``perf test list`` contains no CoreSight entry and
    ``perf test -vv "arm coresight"`` prints nothing and exits 0. GreenLiT's
    sibling ``test_perf_coresight`` therefore falls back to sysfs-availability
    evidence and performs no runtime discrimination at all, which is why this
    package-independent oracle exists.

    PATCH: https://git.kernel.org/torvalds/c/5ad628a76176
    coresight: Use per-sink trace ID maps for Perf sessions
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    try:
        online_cpu_ids = sorted(cpu_topology.parse_cpu_mask(
            _CPU_ONLINE_PATH.read_text(encoding="utf-8")
        ))
    except (OSError, ValueError) as err:
        GLTCore.GLTTestResult.skip(
            f"Cannot read a valid online CPU mask from {_CPU_ONLINE_PATH}: {err}"
        )
    if not online_cpu_ids:
        GLTCore.GLTTestResult.skip(
            f"Online CPU mask from {_CPU_ONLINE_PATH} is empty."
        )

    online_cpus = len(online_cpu_ids)
    if online_cpus < _MIN_CPUS_FOR_TRACE_ID_PRESSURE:
        GLTCore.GLTTestResult.skip(
            "System does not have enough online CPUs to exhaust the global "
            f"CoreSight trace ID map (online_cpus={online_cpus}, need >= "
            f"{_MIN_CPUS_FOR_TRACE_ID_PRESSURE})."
        )

    source_cpu_ids = _ensure_etm4x_loaded(logger)
    if not _CS_ETM_EVENT_DIR.exists():
        GLTCore.GLTTestResult.skip("cs_etm perf event source is not available.")

    sink_cpu_ids = _coresight_cpu_ids(_TRBE_SINK_RE)
    logger.info(
        "CoreSight ETE pressure: online_cpus=%d ete_devices=%d trbe_sinks=%d "
        "threshold=%d",
        online_cpus, len(source_cpu_ids), len(sink_cpu_ids),
        _MIN_CPUS_FOR_TRACE_ID_PRESSURE,
    )
    trigger_cpus = online_cpu_ids[:_MIN_CPUS_FOR_TRACE_ID_PRESSURE]
    selected_cpu_ids = set(trigger_cpus)

    # Counts cannot prove the selected CPUs have the per-CPU source and sink
    # topology needed by this oracle. Match the CPU IDs encoded by sysfs names
    # and require all 112 CPUs used by the trigger in both sets.
    missing_source_ids = sorted(selected_cpu_ids - source_cpu_ids)
    if missing_source_ids:
        GLTCore.GLTTestResult.skip(
            "Selected CPUs do not all have ETE/ETM sources "
            f"(missing_cpu_ids={missing_source_ids})."
        )

    missing_sink_ids = sorted(selected_cpu_ids - sink_cpu_ids)
    if missing_sink_ids:
        GLTCore.GLTTestResult.skip(
            "Selected CPUs do not all have TRBE sinks to exercise per-sink "
            f"trace ID maps (missing_cpu_ids={missing_sink_ids})."
        )

    # Prefer a kernel-matched perf; custom proof kernels rarely ship a matching
    # linux-tools package, so accept an unmatched perf too. The mmap ENOMEM
    # oracle is kernel-side, so userspace major/minor mismatch is acceptable.
    perf_cmd, perf_reason = perf_utils.resolve_perf_command(allow_unmatched=True)
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)
    logger.info("Using perf command: %s", perf_cmd)

    control_path = os.path.join(
        sd.log_dir, "test_perf_cs_etm_multisocket_trace_id_map.control.perf.data"
    )
    perf_data_path = os.path.join(
        sd.log_dir, "test_perf_cs_etm_multisocket_trace_id_map.perf.data"
    )

    # The trigger's CPU list is the control's plus exactly one more CPU.
    control_cpus = online_cpu_ids[:_CONTROL_CPU_COUNT]
    control_range = ",".join(str(cpu) for cpu in control_cpus)
    trigger_range = ",".join(str(cpu) for cpu in trigger_cpus)
    extra_cpu = trigger_cpus[-1]

    # Safe-side control first. 111 CPUs is the largest count a global ID map can
    # still serve, so this must succeed on fixed and unfixed kernels alike.
    # Prerequisites (platform, cs_etm, selected ETE/TRBE) are already
    # established, so a control miss is a recording failure, not a skip.
    control = _run_cs_etm_record(
        perf_cmd, control_path, logger, cpu_range=control_range
    )
    if not control.ok:
        detail = control.error or (
            f"returncode={control.returncode} stderr={control.stderr}"
        )
        GLTCore.GLTTestResult.fail(
            f"perf cs_etm record over {_CONTROL_CPU_COUNT} online CPUs did "
            "not succeed after platform, cs_etm, and selected ETE/TRBE CPUs "
            f"were established: {detail}"
        )
    logger.info(
        "Safe-side control (%d CPUs) recorded cleanly; cs_etm is usable at the "
        "last count a global trace ID map can serve",
        _CONTROL_CPU_COUNT,
    )

    # Trigger: same perf command, same per-CPU mode, same host, same moment,
    # same CPU IDs -- plus exactly one more CPU.
    trigger = _run_cs_etm_record(
        perf_cmd, perf_data_path, logger, cpu_range=trigger_range
    )
    if trigger.error is not None:
        GLTCore.GLTTestResult.fail(f"perf cs_etm record {trigger.error}")

    if trigger.ok:
        logger.info(
            "perf cs_etm record over %d CPUs succeeded (per-sink trace ID maps "
            "present)", _MIN_CPUS_FOR_TRACE_ID_PRESSURE,
        )
        return

    if trigger.mmap_enomem:
        GLTCore.GLTTestResult.fail(
            "perf cs_etm record failed to mmap with ENOMEM over "
            f"{_MIN_CPUS_FOR_TRACE_ID_PRESSURE} online CPUs, yet the identical "
            f"command over {_CONTROL_CPU_COUNT} of the same CPUs succeeded on "
            f"this host moments earlier. The two runs differ only by CPU "
            f"{extra_cpu} (same perf binary, same per-CPU mode). That is the "
            "111-usable-ID boundary (dynamic trace IDs are 1..0x6F), which is "
            "consistent with a global CoreSight trace ID map. Perf also emits "
            "this mmap ENOMEM message for AUX buffer allocation, so the extra "
            "buffer is not independently ruled out. Backport 5ad628a76176 "
            "(\"coresight: Use per-sink trace ID maps for Perf sessions\") "
            "and the remainder of the CoreSight per-sink ID series."
        )

    GLTCore.GLTTestResult.fail(
        "perf cs_etm record failed unexpectedly: "
        f"returncode={trigger.returncode} stderr={trigger.stderr}"
    )
