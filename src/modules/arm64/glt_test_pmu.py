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
Python script to test PMU.
"""
import os
import platform
import re
import signal
import subprocess
import sys
from pathlib import Path

from packaging import version

from common import glt_core as GLTCore
from common import glt_utils
from common import perf_utils
from common.nv_device import NvDev
from scripts import check_kernel_config

NVIDIA_PMU_SYSFS_PATH = "/sys/bus/event_source/devices"

KNOWN_PATCH_SERIES = {
    "t410_prep": "T410 preparatory arm_cspmu patches "
    "(reset_ev_filter, PMPIDR, revision-id matching, PMEVFILTR2)",
    "t410_pmu": "Tegra410 PMU support patches "
    "(UCF, PCIE, PCIE-TGT, and perf vendor events)",
}

VERA_REQUIRED_CS_PMU_FAMILIES = (
    "nvidia_ucf_pmu",
    "nvidia_pcie_pmu",
    "nvidia_pcie_tgt_pmu",
)

GRACE_PMU_FAMILIES = (
    "nvidia_cnvlink_pmu",
    "nvidia_nvlink_c2c0_pmu",
    "nvidia_nvlink_c2c1_pmu",
    "nvidia_pcie_pmu",
    "nvidia_scf_pmu",
)

GENERIC_FALLBACK_PMU_FAMILIES = (
    "nvidia_uncore_pmu",
)

def _normalize_nvidia_pmu_name(pmu_name):
    """Return the PMU family name without instance-specific suffixes."""
    normalized_name = pmu_name.lower()
    if not normalized_name.startswith("nvidia_"):
        return pmu_name
    match = re.match(r"(nvidia_.*?_pmu)(?:_.*)?$", normalized_name)
    if match:
        return match.group(1)
    return normalized_name


def _get_pmu_families(pmu_names):
    """Return normalized NVIDIA PMU families from device or perf unit names."""
    return {
        _normalize_nvidia_pmu_name(pmu_name)
        for pmu_name in pmu_names
        if isinstance(pmu_name, str)
        and pmu_name.lower().startswith("nvidia_")
        and re.search(r"_pmu(?:_|$)", pmu_name.lower())
    }


def _format_patch_guidance():
    """Format concise guidance for missing T410 PMU runtime support."""
    patch_lines = [
        f"{name}: {description}"
        for name, description in KNOWN_PATCH_SERIES.items()
    ]
    return (
        "Verify that the kernel includes the required T410 PMU patch groups: "
        + "; ".join(patch_lines)
    )


def _parse_nvidia_perf_units(perf_list_json):
    """Extract normalized NVIDIA PMU families from perf list JSON entries."""
    return _get_pmu_families(
        item.get("Unit", "")
        for item in perf_list_json
        if isinstance(item, dict)
    )

ARM_SPE_EVENT_SOURCE_PATH = "/sys/bus/event_source/devices"
ARM_SPE_PMUS = ("arm_spe_0", "arm_spe_1", "arm_spe_2", "arm_spe_3")
ARM_SPE_RECORD_TIMEOUT_SECONDS = 180
ARM_SPE_DECODE_TIMEOUT_SECONDS = 120
CSPMU_RECORD_TIMEOUT_SECONDS = 30
ARM_SPE_LLC_RECORD_TIMEOUT_SECONDS = 300
ARM_SPE_MEMFD_PROBE_TIMEOUT_SECONDS = 10
COMMON_DATA_SOURCE_LEVELS = ("LVL L2", "LVL Local RAM", "LVL Remote RAM")
DATA_SOURCE_OPS = ("OP LOAD", "OP STORE")
REQUIRED_SPE_REPORT_STRINGS = (
    "Samples:",
    "l1d-miss",
    "l1d-access",
    "llc-miss",
    "llc-access",
    "tlb-access",
    "memory",
    "branch",
)
PERF_REPORT_SAMPLE_COUNT_PATTERN = re.compile(
    r"^\s*#\s*Samples:\s*"
    r"(?P<count>[0-9][0-9,]*(?:\.[0-9]+)?)\s*[KMG]?"
    r"\s+of event\s+'(?P<event>[^']+)'",
    re.IGNORECASE,
)
ARM_SPE_LLC_THRASH_SIZE_BYTES = 512 * 1024 * 1024
ARM_SPE_LLC_THRASH_STRIDE_BYTES = 64
ARM_SPE_LLC_THRASH_PASSES = 2
# Anonymous mmap thrash sized above Grace LLC to produce llc-miss samples.
# Invoked via a script file under sd.log_dir (not -c) for shell safety.
ARM_SPE_LLC_THRASH_SOURCE = f"""\
import os, mmap
size = {ARM_SPE_LLC_THRASH_SIZE_BYTES}
stride = {ARM_SPE_LLC_THRASH_STRIDE_BYTES}
passes = {ARM_SPE_LLC_THRASH_PASSES}
fd = os.memfd_create('thrash', 0)
os.ftruncate(fd, size)
m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
for _ in range(passes):
    for i in range(0, size, stride):
        m[i] = (m[i] + 1) & 0xff
"""
ARM_SPE_MEMFD_PROBE_SOURCE = (
    "import os; fd = os.memfd_create('greenlit-spe-probe', 0); os.close(fd)"
)


def _arm_spe_memfd_probe_failure():
    """Return a failure reason if this interpreter cannot create an anonymous memfd."""
    try:
        memfd_probe = subprocess.run(
            [sys.executable, "-c", ARM_SPE_MEMFD_PROBE_SOURCE],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=ARM_SPE_MEMFD_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as err:
        return f"Python memfd_create probe timed out after {err.timeout} seconds"

    if memfd_probe.returncode != 0:
        return (
            "Python os.memfd_create support is required for ARM SPE LLC thrash "
            f"workload: stderr={memfd_probe.stderr}"
        )
    return None


def _log_arm_spe_aux_collision_warnings(stderr, logger):
    """Log perf AUX collision warnings at debug level only."""
    for line in (stderr or "").splitlines():
        lowered = line.lower()
        if "aux" in lowered and "collision" in lowered:
            logger.debug("Ignoring perf AUX collision warning: %s", line)


def _has_positive_llc_miss_samples(perf_report_output):
    """Return True when perf report shows positive samples for llc-miss."""
    for line in perf_report_output.splitlines():
        match = PERF_REPORT_SAMPLE_COUNT_PATTERN.match(line)
        if not match or match.group("event").strip() != "llc-miss":
            continue
        sample_count = match.group("count").replace(",", "")
        try:
            if float(sample_count) > 0:
                return True
        except ValueError:
            continue
    return False


def _terminate_arm_spe_record_process(process, pgid):
    """Terminate a perf record process group and its descendants."""
    try:
        os.killpg(pgid, signal.SIGTERM)
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        try:
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _record_arm_spe_llc_workload(perf_cmd, perf_data_path, spe_event,
                                 thrash_script_path, cpu):
    """Record ARM SPE LLC-thrash workload with process-group timeout cleanup."""
    record_cmd = [
        perf_cmd, "record", "-e", spe_event, "-o", perf_data_path, "--",
        "taskset", "--cpu-list", str(cpu), sys.executable, thrash_script_path,
    ]
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        record_cmd,
        start_new_session=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pgid = process.pid

    try:
        stdout, stderr = process.communicate(
            timeout=ARM_SPE_LLC_RECORD_TIMEOUT_SECONDS)
    except BaseException:
        _terminate_arm_spe_record_process(process, pgid)
        raise

    return subprocess.CompletedProcess(
        record_cmd, process.returncode, stdout, stderr)


def _count_decoded_data_source(output):
    """Count lines that contain both operation and memory-level decode fields."""
    return sum(
        1 for line in output.splitlines()
        if any(op in line for op in DATA_SOURCE_OPS)
        and any(level in line for level in COMMON_DATA_SOURCE_LEVELS)
    )


def _decode_arm_spe(perf_cmd, perf_data_path, logger):
    """Decode an ARM SPE perf.data file with perf script data-source fields."""
    logger.info("Decoding ARM SPE data-source fields with perf script")
    return subprocess.run(
        [perf_cmd, "script", "-i", perf_data_path, "-F", "period,data_src,weight,ip,sym"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=ARM_SPE_DECODE_TIMEOUT_SECONDS,
    )


def _arm_spe_pmu_candidates(base_path=ARM_SPE_EVENT_SOURCE_PATH, logger=None):
    """Return discovered ARM SPE PMUs in the existing deterministic order."""
    pmus = [pmu for pmu in ARM_SPE_PMUS if os.path.isdir(os.path.join(base_path, pmu))]
    try:
        pmus.extend(sorted([
            name for name in os.listdir(base_path)
            if name not in pmus and name.startswith("arm_spe")
            and os.path.isdir(os.path.join(base_path, name))
        ], key=lambda name: perf_utils.natural_sort_key(Path(name))))
    except OSError as err:
        if logger:
            logger.debug("Unable to list ARM SPE PMUs under %s: %s", base_path, err)
    return pmus


def _find_arm_spe_pmu(base_path=ARM_SPE_EVENT_SOURCE_PATH, logger=None):
    """Find the first available ARM SPE PMU."""
    pmus = _arm_spe_pmu_candidates(base_path=base_path, logger=logger)
    return pmus[0] if pmus else None


_PMCCNTR_SMT_BUSY_FLOOR = 100_000_000
_PMCCNTR_SMT_IDLE_RATIO_MAX = 0.10
_PMCCNTR_SMT_PROBE_TIMEOUT_SECONDS = 10


def _parse_cpu_list(s):
    """Parse a Linux CPU list string into a sorted list of integer CPU IDs.

    Handles comma-separated values, ranges, and mixed syntax (e.g. '0,2-3').
    """
    cpus = []
    for part in s.strip().split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            cpus.append(int(part))
    return sorted(set(cpus))


def _select_arm_spe_cpu(
    pmu, base_path=ARM_SPE_EVENT_SOURCE_PATH, allowed_cpus=None
):
    """Return the first CPU supported by the selected ARM SPE PMU."""
    cpumask_path = Path(base_path) / pmu / "cpumask"
    try:
        cpumask = cpumask_path.read_text(encoding="utf-8").strip()
    except OSError as err:
        return None, (
            f"Unable to read ARM SPE PMU cpumask at {cpumask_path}: {err}"
        )

    if not cpumask:
        return None, f"ARM SPE PMU cpumask at {cpumask_path} is empty"

    try:
        cpus = _parse_cpu_list(cpumask)
    except (OverflowError, ValueError) as err:
        return None, (
            f"Unable to parse ARM SPE PMU cpumask at {cpumask_path}: "
            f"{cpumask!r}: {err}"
        )

    unusable_reason = ""
    usable_cpus = []
    if not cpus:
        unusable_reason = (
            f"ARM SPE PMU cpumask at {cpumask_path} did not select any CPU"
        )
    else:
        try:
            available_cpus = (
                set(os.sched_getaffinity(0))
                if allowed_cpus is None else set(allowed_cpus)
            )
        except OSError as err:
            unusable_reason = (
                f"Unable to read current CPU affinity for ARM SPE PMU {pmu}: "
                f"{err}"
            )
        else:
            usable_cpus = sorted(set(cpus) & available_cpus)
            if not usable_cpus:
                unusable_reason = (
                    f"ARM SPE PMU cpumask at {cpumask_path} has no CPUs in "
                    "the current process CPU affinity"
                )

    if unusable_reason:
        return None, unusable_reason
    return usable_cpus[0], ""


def _select_arm_spe_pmu_cpu(
    base_path=ARM_SPE_EVENT_SOURCE_PATH, allowed_cpus=None, logger=None
):
    """Return the first discovered ARM SPE PMU with an affinity-usable CPU."""
    unusable_reasons = []

    for pmu in _arm_spe_pmu_candidates(base_path=base_path, logger=logger):
        cpu, reason = _select_arm_spe_cpu(pmu, base_path=base_path,
                                          allowed_cpus=allowed_cpus)
        if cpu is not None:
            return pmu, cpu, ""
        unusable_reasons.append(f"{pmu}: {reason}")
        if logger:
            logger.debug("Skipping ARM SPE PMU %s: %s", pmu, reason)

    if unusable_reasons:
        return None, None, (
            "No discovered ARM SPE PMU has a CPU usable by the current process "
            f"CPU affinity: {'; '.join(unusable_reasons)}")
    return None, None, (
        f"ARM SPE PMU is not available under {base_path}. "
        "SPE requires OS and firmware support, please check both.")


def _find_online_smt_pair():
    """Return (busy_cpu, idle_cpu) from the first fully-online SMT sibling pair.

    Returns None when no such pair exists or sysfs is unavailable.
    """
    try:
        online_cpus = set(_parse_cpu_list(
            Path("/sys/devices/system/cpu/online").read_text(encoding="utf-8")
        ))
    except (OSError, ValueError):
        return None

    for cpu in sorted(online_cpus):
        siblings_path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        try:
            siblings = _parse_cpu_list(siblings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        online_siblings = [c for c in siblings if c in online_cpus]
        if len(online_siblings) >= 2:
            return online_siblings[0], online_siblings[1]
    return None


def _parse_perf_stat_csv(stderr, busy_cpu, idle_cpu):
    """Parse per-CPU cycle counts from perf stat -x, -A stderr output.

    Returns (busy_cycles, idle_cycles) or raises ValueError.
    """
    target = {busy_cpu, idle_cpu}
    counts = {}

    for line in stderr.splitlines():
        line = line.strip()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or parts[0] not in {
                str(busy_cpu), f"CPU{busy_cpu}", str(idle_cpu), f"CPU{idle_cpu}"}:
            continue
        if parts[3].split(":", 1)[0] != "cycles":
            continue
        cpu_label = parts[0]
        count_field = parts[1]

        for cpu_id in target:
            if cpu_label in (str(cpu_id), f"CPU{cpu_id}"):
                if count_field in ("<not counted>", "<not supported>", ""):
                    raise ValueError(
                        f"CPU {cpu_id} cycles counter unavailable: {count_field!r}"
                    )
                try:
                    counts[cpu_id] = int(count_field)
                except ValueError as exc:
                    raise ValueError(
                        f"CPU {cpu_id} cycle count not parseable: {count_field!r}"
                    ) from exc

    for cpu_id in target:
        if cpu_id not in counts:
            raise ValueError(
                f"No cycle count found for CPU {cpu_id} in perf stat output; "
                f"stderr tail: {stderr.strip()[-300:]!r}"
            )

    return counts[busy_cpu], counts[idle_cpu]


def _probe_pmccntr_smt(busy_cpu, idle_cpu, logger, perf_cmd):
    """Run a single perf stat SMT probe and return (busy_cycles, idle_cycles).

    Uses one subprocess so perf stat owns the workload lifetime exactly.
    Raises ValueError on parse failure or missing per-CPU counts.
    """
    try:
        result = subprocess.run(
            [
                perf_cmd, "stat", "-x,", "-e", "cycles", "-A",
                "-C", f"{busy_cpu},{idle_cpu}",
                "--", "taskset", "-c", str(busy_cpu), "timeout", "2s", "yes",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=_PMCCNTR_SMT_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            "perf stat PMCCNTR SMT probe timed out after "
            f"{_PMCCNTR_SMT_PROBE_TIMEOUT_SECONDS} seconds"
        ) from exc
    except OSError as exc:
        raise ValueError(f"perf stat PMCCNTR SMT probe failed to start: {exc}") from exc
    logger.debug(
        "perf stat rc=%d stderr=%r",
        result.returncode, (result.stderr or "")[:500],
    )
    return _parse_perf_stat_csv(result.stderr or "", busy_cpu, idle_cpu)


def _get_nvidia_perf_list(perf_cmd, logger):
    """Get NVIDIA-specific perf event unit families."""
    return _parse_nvidia_perf_units(
        perf_utils.get_perf_list_json(perf_cmd, logger)
    )


def _get_nvidia_pmu_devices(pmu_path=NVIDIA_PMU_SYSFS_PATH):
    """Get sorted NVIDIA PMU device names from sysfs."""
    path = Path(pmu_path)
    if not path.is_dir():
        return []

    try:
        return sorted(
            device.name
            for device in path.iterdir()
            if device.name.lower().startswith("nvidia_")
            and re.search(r"_pmu(?:_|$)", device.name.lower())
            and device.is_dir()
        )
    except OSError:
        return []


def _is_fallback_pmu(pmu_name):
    """Return True for generic NVIDIA CoreSight PMU fallback devices."""
    return _normalize_nvidia_pmu_name(pmu_name) in GENERIC_FALLBACK_PMU_FAMILIES


def _classify_pmu_devices(pmu_devices):
    """Classify PMU devices into named and generic fallback devices."""
    named = []
    fallback = []
    for pmu_device in pmu_devices:
        if _is_fallback_pmu(pmu_device):
            fallback.append(pmu_device)
        else:
            named.append(pmu_device)
    return named, fallback


def _missing_pmu_families(observed_families, required_families):
    """Return required PMU families absent from observed families."""
    return sorted(set(required_families) - set(observed_families))


def _validate_common_nvidia_pmus(logger, nvevent_from_dut, pmu_devices,
                                 guidance=""):
    """Validate NVIDIA PMU visibility that applies to all NVIDIA CPUs."""
    logger.debug("Supported NVIDIA PMU families from perf: %s",
                 sorted(nvevent_from_dut))
    if not nvevent_from_dut:
        return f"No NVIDIA perf events found. {guidance}".strip()

    logger.debug("Registered NVIDIA PMU devices: %s", pmu_devices)
    if not pmu_devices:
        return f"No NVIDIA PMU devices found in sysfs. {guidance}".strip()

    named_pmus, fallback_pmus = _classify_pmu_devices(pmu_devices)
    logger.debug("Named NVIDIA PMU devices: %s", named_pmus)
    logger.debug("Fallback NVIDIA PMU devices: %s", fallback_pmus)
    if fallback_pmus and not named_pmus:
        return (
            f"Only generic fallback NVIDIA PMU devices found: {fallback_pmus}. "
            f"{guidance}"
        ).strip()

    logger.debug("Observed NVIDIA PMU families from sysfs: %s",
                 sorted(_get_pmu_families(pmu_devices)))
    return None


def validate_grace_pmus(logger, nvevent_from_dut, pmu_devices):
    """Validate Grace-specific NVIDIA PMU expectations."""
    failure_reason = _validate_common_nvidia_pmus(
        logger, nvevent_from_dut, pmu_devices)
    if failure_reason:
        return failure_reason

    pmu_families = _get_pmu_families(pmu_devices)
    expected_families = sorted(GRACE_PMU_FAMILIES)
    detected_perf_families = sorted(nvevent_from_dut)
    detected_sysfs_families = sorted(pmu_families)
    missing_perf_families = _missing_pmu_families(
        nvevent_from_dut, GRACE_PMU_FAMILIES)
    missing_sysfs_families = _missing_pmu_families(
        pmu_families, GRACE_PMU_FAMILIES)
    if missing_perf_families or missing_sysfs_families:
        return (
            "Missing expected Grace NVIDIA PMU families. "
            f"expected: {expected_families}; "
            f"detected perf: {detected_perf_families}; "
            f"detected sysfs: {detected_sysfs_families}; "
            f"missing perf: {missing_perf_families}; "
            f"missing sysfs: {missing_sysfs_families}"
        )

    allowed_families = set(GRACE_PMU_FAMILIES) | set(
        GENERIC_FALLBACK_PMU_FAMILIES)
    extra_perf_families = sorted(set(nvevent_from_dut) - allowed_families)
    extra_sysfs_families = sorted(pmu_families - allowed_families)
    if extra_perf_families or extra_sysfs_families:
        return (
            "Found unexpected Grace NVIDIA PMU families. "
            f"expected: {expected_families}; "
            f"detected perf: {detected_perf_families}; "
            f"detected sysfs: {detected_sysfs_families}; "
            f"extra perf: {extra_perf_families}; "
            f"extra sysfs: {extra_sysfs_families}"
        )
    return None


def validate_vera_pmus(logger, nvevent_from_dut, pmu_devices):
    """Validate Vera-specific NVIDIA PMU expectations."""
    patch_guidance = _format_patch_guidance()
    failure_reason = _validate_common_nvidia_pmus(
        logger, nvevent_from_dut, pmu_devices, patch_guidance)
    if failure_reason:
        return failure_reason

    missing_sysfs_families = _missing_pmu_families(
        _get_pmu_families(pmu_devices), VERA_REQUIRED_CS_PMU_FAMILIES)
    if missing_sysfs_families:
        return (
            "Missing required Vera T410 CoreSight PMU families from sysfs: "
            f"{', '.join(missing_sysfs_families)}. {patch_guidance}"
        )

    missing_perf_families = _missing_pmu_families(
        nvevent_from_dut, VERA_REQUIRED_CS_PMU_FAMILIES)
    if missing_perf_families:
        return (
            "Missing required Vera T410 CoreSight PMU families from perf: "
            f"{', '.join(missing_perf_families)}. {patch_guidance}"
        )
    return None


def _has_decoded_data_source(output):
    """Return True if perf script output includes decoded operation and level fields."""
    return _count_decoded_data_source(output) > 0


def _record_arm_spe(perf_cmd, perf_data_path, pmu, logger):
    """Record a small workload with the requested ARM SPE PMU."""
    event = f"{pmu}/ts_enable=1,load_filter=1,store_filter=1,min_latency=1/"
    logger.info("Recording ARM SPE data to %s", perf_data_path)
    return subprocess.run(
        [
            perf_cmd, "record", "-o", perf_data_path, "-e", event, "--",
            "dd", "if=/dev/zero", "of=/dev/null", "bs=64M", "count=8",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=ARM_SPE_RECORD_TIMEOUT_SECONDS,
    )


@GLTCore.GLTRequires.must_pass("hwtracing", "test_perf_coresight")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_perf_events(_, logger):
    """Check whether NVIDIA perf PMU events match platform expectations."""
    # FIXME: DGX-10447 perf seems not enable coresight on ubuntu.
    nvdev = NvDev()
    is_grace = nvdev.is_cpu_type("grace")
    is_vera = nvdev.is_cpu_type("vera")
    if not is_grace and not is_vera:
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    # Coresight is targeted to RHEL95 for now. DGX-7793 and DGX-7787
    if "Red Hat" in NvDev.get_release_info().get("NAME") and \
            version.parse(NvDev.get_release_info().get("VERSION_ID")) < \
            version.parse("9.5"):
        GLTCore.GLTTestResult.skip("RHEL supports coresight if version >= 9.5.")

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    try:
        nvevent_from_dut = _get_nvidia_perf_list(perf_cmd, logger)
    except ValueError as exc:
        GLTCore.GLTTestResult.fail(str(exc))
    pmu_devices = _get_nvidia_pmu_devices()
    failure_reason = None
    if is_grace:
        failure_reason = validate_grace_pmus(
            logger, nvevent_from_dut, pmu_devices)
    elif is_vera:
        failure_reason = validate_vera_pmus(
            logger, nvevent_from_dut, pmu_devices)

    if failure_reason:
        GLTCore.GLTTestResult.fail(failure_reason)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_all_pmu_events(_, logger):
    """Perform perf all pmu events check"""
    # FIXME: DGX-10165 pylint: disable=fixme
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    result = perf_utils.run_perf_test_candidates(
        ["perf all PMU test", "Parsing of all PMU events from sysfs"],
        perf_cmd=perf_cmd,
    )
    logger.debug(
        "perf PMU test status=%s test_name=%s reason=%s",
        result.status.value, result.test_name, result.reason)
    if result.status != perf_utils.PerfTestStatus.PASS:
        GLTCore.GLTTestResult.fail(f"perf PMU test failed: {result.reason}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
def test_perf_cspmu_rejects_foreign_events(_, logger):
    """Verify CSPMU rejects events meant for other PMUs (commit 15c7ef73).

    Uses `perf stat -e r2` so Core PMU can run the raw event when CSPMU
    correctly returns ENOENT for foreign events.
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    # Commit 15c7ef73 only applies when nvidia_*_pmu devices are registered.
    pmu_devices = _get_nvidia_pmu_devices()
    if not pmu_devices:
        GLTCore.GLTTestResult.skip(
            "No CoreSight nvidia_*_pmu devices registered; "
            "cannot verify commit 15c7ef73."
        )

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    try:
        # Repro from commit 15c7ef73: Core PMU should handle raw r2 after CSPMU
        # rejects the foreign event with ENOENT.
        cmd_output = subprocess.run(
            [perf_cmd, "stat", "-e", "r2", "--", "sleep", "1"],
            check=True,
            text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=CSPMU_RECORD_TIMEOUT_SECONDS)
        logger.debug("perf stat completed: %s", cmd_output)
    except subprocess.TimeoutExpired:
        GLTCore.GLTTestResult.fail(
            f"perf command timed out after {CSPMU_RECORD_TIMEOUT_SECONDS} seconds. "
            "This may indicate that perf or the workload stalled."
        )
    except subprocess.CalledProcessError as e:
        stderr_line = " ".join((e.stderr or "").splitlines())
        GLTCore.GLTTestResult.fail(
            f"perf command failed with return code {e.returncode}: {stderr_line}. "
            "This may indicate that commit 15c7ef73 is missing; consider backporting it."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_spe_pmu(sd, logger):
    """Verify ARM SPE PMU support with a deterministic LLC-thrash workload."""
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
        if kconfig.get("CONFIG_ARM_SPE_PMU") not in ("y", "m"):
            GLTCore.GLTTestResult.fail("CONFIG_ARM_SPE_PMU is not enabled in kernel config")
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {kconfig_path}, cannot verify CONFIG_ARM_SPE_PMU"
        )

    pmu, spe_cpu, cpu_reason = _select_arm_spe_pmu_cpu(logger=logger)
    if cpu_reason:
        GLTCore.GLTTestResult.fail(cpu_reason)

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    if not glt_utils.is_cli_available("taskset"):
        GLTCore.GLTTestResult.fail(
            "Required tool 'taskset' not found; install it to run this test.")

    memfd_failure = _arm_spe_memfd_probe_failure()
    if memfd_failure:
        GLTCore.GLTTestResult.fail(memfd_failure)

    perf_data_path = os.path.join(sd.log_dir, "test_arm_spe_pmu.perf.data")
    thrash_script_path = os.path.join(sd.log_dir, "arm_spe_llc_thrash.py")
    with open(thrash_script_path, "w", encoding="utf-8") as thrash_file:
        thrash_file.write(ARM_SPE_LLC_THRASH_SOURCE)

    # dd if=/dev/zero rarely produces llc-miss on Grace; thrash anonymous memory
    # above LLC size for deterministic llc-miss decode coverage. Deliver via a
    # script file (not -c) so remote shell quoting cannot mangle it.
    spe_event = f"{pmu}//"
    try:
        record_result = _record_arm_spe_llc_workload(
            perf_cmd,
            perf_data_path,
            spe_event,
            thrash_script_path,
            spe_cpu,
        )
    except subprocess.TimeoutExpired as err:
        GLTCore.GLTTestResult.fail(
            f"perf record with arm_spe timed out after {err.timeout} seconds"
        )

    _log_arm_spe_aux_collision_warnings(record_result.stderr, logger)

    if record_result.returncode != 0:
        GLTCore.GLTTestResult.fail(
            f"perf record failed with arm_spe event: returncode={record_result.returncode} "
            f"stderr={record_result.stderr}"
        )

    try:
        cmd_output = subprocess.run(
            [perf_cmd, "report", "-i", perf_data_path, "--stdio"],
            check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=ARM_SPE_DECODE_TIMEOUT_SECONDS)
        logger.debug(cmd_output.stdout)
    except subprocess.TimeoutExpired as err:
        GLTCore.GLTTestResult.fail(
            f"perf report timed out after {err.timeout} seconds for {perf_data_path}"
        )
    except subprocess.CalledProcessError as err:
        GLTCore.GLTTestResult.fail(
            f"perf report failed with {perf_data_path}: {err} - {err.stdout}"
        )

    # Since we are using `perf record -e <arm_spe_*>//` instead of `perf mem`,
    # the output format changes. We verify that the SPE event was actually sampled.
    report_stdout = cmd_output.stdout or ""
    missing = [
        required_string
        for required_string in REQUIRED_SPE_REPORT_STRINGS
        if required_string not in report_stdout
    ]
    if missing:
        GLTCore.GLTTestResult.fail(
            f"Missing expected strings {missing} in perf report with {perf_data_path}"
        )
    if not _has_positive_llc_miss_samples(report_stdout):
        GLTCore.GLTTestResult.fail(
            f"Missing positive llc-miss samples in perf report with {perf_data_path}"
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_pmccntr_smt_handling(_, logger):
    """Verify PMCCNTR_EL0 is not used for CPU_CYCLES on SMT cores.

    Tests the fix from commits c3d78c34ad00 (arm_pmuv3: Don't use PMCCNTR_EL0
    on SMT cores), f8f89e8cf3d6, and 7ab06ea41af5.
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not a Vera CPU; test is Vera-specific.")

    pair = _find_online_smt_pair()
    if pair is None:
        GLTCore.GLTTestResult.skip(
            "No online SMT sibling pair found; cannot probe PMCCNTR_EL0 SMT behavior."
        )
    busy_cpu, idle_cpu = pair

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)

    for tool in ("taskset", "timeout", "yes"):
        if not glt_utils.is_cli_available(tool):
            GLTCore.GLTTestResult.fail(
                f"Required tool '{tool}' not found; install it to run this test."
            )

    try:
        busy_cycles, idle_cycles = _probe_pmccntr_smt(
            busy_cpu, idle_cpu, logger, perf_cmd)
    except ValueError as exc:
        GLTCore.GLTTestResult.fail(f"PMCCNTR_EL0 SMT probe failed: {exc}")

    ratio = idle_cycles / busy_cycles if busy_cycles else float("inf")
    logger.debug(
        "PMCCNTR_EL0 SMT probe: busy_cpu=%d busy=%d idle_cpu=%d idle=%d ratio=%.4f",
        busy_cpu, busy_cycles, idle_cpu, idle_cycles, ratio,
    )

    if busy_cycles < _PMCCNTR_SMT_BUSY_FLOOR:
        GLTCore.GLTTestResult.fail(
            f"Busy CPU {busy_cpu} only counted {busy_cycles} cycles "
            f"(floor {_PMCCNTR_SMT_BUSY_FLOOR}); workload may not have run."
        )

    if ratio > _PMCCNTR_SMT_IDLE_RATIO_MAX:
        GLTCore.GLTTestResult.fail(
            f"Idle CPU {idle_cpu} cycle ratio {ratio:.4f} exceeds threshold "
            f"{_PMCCNTR_SMT_IDLE_RATIO_MAX} (idle={idle_cycles}, busy={busy_cycles}). "
            "PMCCNTR_EL0 appears to count across SMT siblings. "
            "Backport commits c3d78c34ad00, f8f89e8cf3d6, 7ab06ea41af5."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_perf_arm_spe_olympus_data_source(sd, logger):
    """
    Verify perf decodes Olympus ARM SPE data-source fields.

    Vera-only: this test does not use GLTRequires.must_pass("hwtracing",
    "test_perf_coresight") because it validates ARM SPE data_src decode via
    perf script, not CoreSight ETM selftest availability. CoreSight selftests
    can be absent while SPE PMU decode still works on Vera.

    PATCH: d5e4c710b54a tools headers arm64: Add NVIDIA Olympus part
    PATCH: d852b838eb05 perf arm-spe: Add NVIDIA Olympus to neoverse list
    """
    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Vera CPU.")

    logger.info("Detected platform: vera=True; CPU core expected: Olympus")

    perf_cmd, perf_reason = perf_utils.resolve_perf_command()
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail(perf_reason)
    logger.info("Using perf command: %s", perf_cmd)

    pmu = _find_arm_spe_pmu(logger=logger)
    if pmu is None:
        GLTCore.GLTTestResult.fail("ARM SPE PMU is not available under /sys/bus/event_source/devices.")
    logger.info("Using ARM SPE PMU: %s", pmu)

    os.makedirs(sd.log_dir, exist_ok=True)
    perf_data_path = os.path.join(
        sd.log_dir, "test_perf_arm_spe_olympus_data_source.perf.data"
    )

    try:
        result = _record_arm_spe(perf_cmd, perf_data_path, pmu, logger)
    except subprocess.TimeoutExpired as err:
        GLTCore.GLTTestResult.fail(
            f"perf ARM SPE record timed out after {err.timeout} seconds"
        )

    if result.returncode != 0:
        GLTCore.GLTTestResult.fail(
            f"perf ARM SPE record failed: stdout={result.stdout}, stderr={result.stderr}"
        )

    try:
        result = _decode_arm_spe(perf_cmd, perf_data_path, logger)
    except subprocess.TimeoutExpired as err:
        GLTCore.GLTTestResult.fail(
            f"perf ARM SPE decode timed out after {err.timeout} seconds"
        )
    if result.returncode != 0:
        GLTCore.GLTTestResult.fail(
            f"perf ARM SPE decode failed: stdout={result.stdout}, stderr={result.stderr}"
        )

    decoded_lines = _count_decoded_data_source(result.stdout)
    logger.info("Decoded ARM SPE data-source summary: decoded_lines=%d",
                decoded_lines)

    if not _has_decoded_data_source(result.stdout):
        GLTCore.GLTTestResult.fail(
            "perf ARM SPE output did not include decoded Olympus data-source fields."
        )

    logger.info("perf ARM SPE Olympus data-source decode is present")
