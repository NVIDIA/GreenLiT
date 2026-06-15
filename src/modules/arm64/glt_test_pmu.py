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
import json
import os
import platform
import re
import subprocess
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
        "Consider checking whether the kernel is missing the T410 patch "
        "groups from data/kernel-patches/vera.txt with check-kernel-patches: "
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
COMMON_DATA_SOURCE_LEVELS = ("LVL L2", "LVL Local RAM", "LVL Remote RAM")
DATA_SOURCE_OPS = ("OP LOAD", "OP STORE")


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
    )


def _find_arm_spe_pmu(base_path=ARM_SPE_EVENT_SOURCE_PATH):
    """Find the first available ARM SPE PMU."""
    for pmu in ARM_SPE_PMUS:
        if os.path.exists(os.path.join(base_path, pmu)):
            return pmu

    try:
        pmus = sorted(
            name for name in os.listdir(base_path)
            if name.startswith("arm_spe")
        )
    except OSError:
        return None

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


def _probe_pmccntr_smt(busy_cpu, idle_cpu, logger):
    """Run a single perf stat SMT probe and return (busy_cycles, idle_cycles).

    Uses one subprocess so perf stat owns the workload lifetime exactly.
    Raises ValueError on parse failure or missing per-CPU counts.
    """
    try:
        result = subprocess.run(
            [
                "perf", "stat", "-x,", "-e", "cycles", "-A",
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


def _get_nvidia_perf_list(logger):
    """Get NVIDIA-specific perf event unit families."""
    try:
        cmd_output = subprocess.run(
            ["perf", "list", "-j"],
            check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, OSError) as e:
        logger.error("Error executing command: %s", e)
        return set()

    try:
        perf_list_json = json.loads(cmd_output.stdout)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse perf list JSON output: %s", e)
        return set()

    return _parse_nvidia_perf_units(perf_list_json)


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


def _get_perf_candidates():
    """Return perf command candidates without requiring them to exist."""
    candidates = ["perf"]
    candidates.extend(str(path) for path in sorted(Path("/usr/lib").glob("linux-tools*/perf")))
    candidates.extend(str(path) for path in sorted(Path("/usr/lib/linux-tools").glob("*/perf")))
    return list(dict.fromkeys(candidates))


def _get_perf_command(logger=None):
    """Resolve a perf command that works for the running kernel."""
    kernel_release = platform.uname().release
    kernel_version = kernel_release.split("-", 1)[0]

    for candidate in _get_perf_candidates():
        try:
            result = subprocess.run(
                [candidate, "--version"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as err:
            if logger:
                logger.debug("perf resolver skipped candidate %s: %s", candidate, err)
            continue

        output = result.stdout or ""
        if result.returncode != 0 or "perf not found for kernel" in output:
            if logger:
                logger.debug("perf resolver rejected candidate: %s --version returned %r",
                             candidate, output.strip())
            continue

        version_match = re.search(r"perf version\s+([^\s]+)", output)
        perf_version = version_match.group(1) if version_match else ""
        if kernel_release in candidate or kernel_version in candidate or perf_version == kernel_version:
            return candidate

        if logger:
            logger.debug("perf resolver skipped non-matching candidate: %s reports %r",
                         candidate, perf_version)

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
            "dd", "if=/dev/zero", "of=/dev/null", "bs=1M", "count=512", "status=none",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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

    nvevent_from_dut = _get_nvidia_perf_list(logger)
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

    if not glt_utils.is_cli_available("perf"):
        GLTCore.GLTTestResult.fail("Please install 'perf' to peform the test.")

    result = perf_utils.run_perf_test_candidates(
        ["perf all PMU test", "Parsing of all PMU events from sysfs"]
    )
    logger.debug(
        "perf PMU test status=%s test_name=%s reason=%s",
        result.status.value, result.test_name, result.reason)
    if result.status != perf_utils.PerfTestStatus.PASS:
        GLTCore.GLTTestResult.fail(f"perf PMU test failed: {result.reason}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
def test_perf_cspmu_rejects_foreign_events(_, logger):
    """Verify perf rejects CSPMU events from other PMUs (related to commit 15c7ef73)."""
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("This is not NVIDIA Grace or Vera CPU.")

    if not glt_utils.is_cli_available("perf"):
        GLTCore.GLTTestResult.fail("Please install 'perf' to perform the test.")

    try:
        # Test if the kernel properly rejects PMU events meant for other PMUs (commit 15c7ef73).
        cmd_output = subprocess.run(
            "perf record -q -o /dev/null -e cycles:Pu -a -- sleep 0.1",
            shell=True, executable="/bin/bash", check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug("perf record completed: %s", cmd_output)
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
    """Verify ARM SPE PMU support."""
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
        if kconfig.get("CONFIG_ARM_SPE_PMU") not in ("y", "m"):
            GLTCore.GLTTestResult.fail("CONFIG_ARM_SPE_PMU is not enabled in kernel config")
    except FileNotFoundError:
        GLTCore.GLTTestResult.fail(
            f"Kernel config not found at {kconfig_path}, cannot verify CONFIG_ARM_SPE_PMU"
        )

    if not os.path.exists("/sys/bus/event_source/devices/arm_spe_0/"):
        GLTCore.GLTTestResult.fail(
            "/sys/bus/event_source/devices/arm_spe_0/ does not exist. "
            "SPE requires OS and firmware support, please check both."
        )

    if not glt_utils.is_cli_available("perf"):
        GLTCore.GLTTestResult.fail("Please install 'perf' to perform the test.")

    perf_data_path = os.path.join(sd.log_dir, "test_arm_spe_pmu.perf.data")

    try:
        subprocess.run(
            ["perf", "record", "-e", "arm_spe//", "-o", perf_data_path, "--",
             "dd", "if=/dev/zero", "of=/dev/null", "bs=1M", "count=500"],
            check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr_out = e.stderr.decode(errors="replace") if e.stderr else ""
        GLTCore.GLTTestResult.fail(f"perf record failed with arm_spe event: {e} - {stderr_out}")

    try:
        cmd_output = subprocess.run(
            ["perf", "report", "-i", perf_data_path, "--stdio"],
            check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logger.debug(cmd_output.stdout)
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"perf report failed with {perf_data_path}: {e} - {e.stdout}")

    # Since we are using `perf record -e arm_spe//` instead of `perf mem`,
    # the output format changes. We verify that the SPE event was actually sampled.
    expected_strings = [
        "Samples:", "l1d-miss", "l1d-access", "llc-miss", "llc-access",
        "tlb-access", "memory", "branch"
    ]
    missing = [s for s in expected_strings if s not in cmd_output.stdout]
    if missing:
        GLTCore.GLTTestResult.fail(
            f"Missing expected strings {missing} in perf report with {perf_data_path}"
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

    for tool in ("perf", "taskset", "timeout", "yes"):
        if not glt_utils.is_cli_available(tool):
            GLTCore.GLTTestResult.fail(
                f"Required tool '{tool}' not found; install it to run this test."
            )

    try:
        busy_cycles, idle_cycles = _probe_pmccntr_smt(busy_cpu, idle_cpu, logger)
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

    PATCH: d5e4c710b54a tools headers arm64: Add NVIDIA Olympus part
    PATCH: d852b838eb05 perf arm-spe: Add NVIDIA Olympus to neoverse list
    """
    nvdev = NvDev()
    is_grace = nvdev.is_cpu_type("grace")
    is_vera = nvdev.is_cpu_type("vera")
    logger.info("Detected platform: grace=%s, vera=%s; Vera CPU core expected: Olympus",
                is_grace, is_vera)

    if not is_vera:
        GLTCore.GLTTestResult.skip("This is not NVIDIA Vera CPU.")

    perf_cmd = _get_perf_command(logger)
    if perf_cmd is None:
        GLTCore.GLTTestResult.fail("Please install 'perf' to perform the test.")
    logger.info("Using perf command: %s", perf_cmd)

    pmu = _find_arm_spe_pmu()
    if pmu is None:
        GLTCore.GLTTestResult.fail("ARM SPE PMU is not available under /sys/bus/event_source/devices.")
    logger.info("Using ARM SPE PMU: %s", pmu)

    perf_data_path = os.path.join(sd.log_dir, "test_perf_arm_spe_olympus_data_source.perf.data")

    result = _record_arm_spe(perf_cmd, perf_data_path, pmu, logger)
    if result.returncode != 0:
        GLTCore.GLTTestResult.fail(
            f"perf ARM SPE record failed: stdout={result.stdout}, stderr={result.stderr}"
        )

    result = _decode_arm_spe(perf_cmd, perf_data_path, logger)
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
