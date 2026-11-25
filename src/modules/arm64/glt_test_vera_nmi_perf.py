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

"""Validate Vera pseudo-NMI PMU overflow delivery in an IRQ-off probe."""

import ctypes
import errno
import fcntl
import json
import os
import platform
import zlib
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.cpu_topology import parse_cpu_mask
from common.nv_device import NvDev

NMI_PERF_MODULE = "vera_nmi_perf_probe"
NMI_PERF_MODULE_PATH = Path(__file__).resolve().parent / "vera_nmi_perf_probe"
NMI_PERF_MODULE_STATE_PATH = Path("/sys/module") / NMI_PERF_MODULE
NMI_PERF_DEVICE = "/dev/glt_vera_nmi_perf"
NMI_PERF_MANIFEST_PATH = Path("/etc/greenlit/vera-nmi-perf.json")
DMI_SERIAL_PATH = Path("/sys/class/dmi/id/product_serial")
CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
NMI_PERF_MAX_IRQOFF_MS = 1000
NMI_PERF_MIN_IRQOFF_MS = 100


class NmiPerfTrigger(ctypes.Structure):
    _fields_ = [
        ("cpu", ctypes.c_uint32),
        ("duration_ms", ctypes.c_uint32),
        ("armed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("overflows", ctypes.c_uint32),
        ("nmi_overflows", ctypes.c_uint32),
        ("irqoff_overflows", ctypes.c_uint32),
        ("irqoff_nmi_overflows", ctypes.c_uint32),
        ("module_irqoff_nmi_overflows", ctypes.c_uint32),
    ]


NMI_PERF_IOCTL_TRIGGER = (
    (3 << 30) | (ctypes.sizeof(NmiPerfTrigger) << 16) | (ord("G") << 8) | 0x71
)


def _read_kernel_config_text():
    try:
        with open("/proc/config.gz", "rb") as config_file:
            return zlib.decompress(config_file.read(), 16 + zlib.MAX_WBITS).decode()
    except OSError:
        with open(f"/boot/config-{platform.release()}", encoding="utf-8") as config_file:
            return config_file.read()


def _read_cmdline():
    return Path("/proc/cmdline").read_text(encoding="utf-8")


def _load_manifest(manifest_path, serial_path):
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Cannot load NMI perf DUT manifest: {err}") from err
    if not isinstance(manifest, dict):
        raise ValueError("NMI perf DUT manifest must be an object")
    dut_id = manifest.get("dut_id")
    if not isinstance(dut_id, str) or not dut_id.strip():
        raise ValueError("NMI perf DUT manifest requires a non-empty dut_id")
    serial = Path(serial_path).read_text(encoding="utf-8").strip()
    if not serial or serial != dut_id.strip():
        raise ValueError("NMI perf DUT manifest does not match the running DUT identity")
    if manifest.get("recovery") != "ready":
        raise ValueError("NMI perf DUT manifest requires recovery=ready")
    if manifest.get("backend") != "pmu_pseudo_nmi":
        raise ValueError("NMI perf sampling requires backend=pmu_pseudo_nmi")
    cpu = manifest.get("cpu")
    if not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0:
        raise ValueError("NMI perf manifest cpu must be a non-negative integer")
    if "max_irqoff_ms" not in manifest:
        raise ValueError("NMI perf DUT manifest requires explicit max_irqoff_ms")
    max_irqoff = manifest["max_irqoff_ms"]
    if not isinstance(max_irqoff, int) or isinstance(max_irqoff, bool):
        raise ValueError("max_irqoff_ms must be an integer")
    if max_irqoff <= 0 or max_irqoff > NMI_PERF_MAX_IRQOFF_MS:
        raise ValueError(f"max_irqoff_ms must be in 1..{NMI_PERF_MAX_IRQOFF_MS}")
    return cpu, max_irqoff


def _kernel_bool_param(cmdline, expected_name):
    expected_name = expected_name.replace("-", "_")
    enabled = False
    for token in cmdline.split():
        if token == "--":
            break
        name, separator, value = token.partition("=")
        if name.replace("-", "_") != expected_name or not separator or not value:
            continue
        first = value[0].lower()
        if first in ("y", "t", "1"):
            enabled = True
        elif first in ("n", "f", "0"):
            enabled = False
        elif first == "o" and len(value) > 1:
            if value[1].lower() == "n":
                enabled = True
            elif value[1].lower() == "f":
                enabled = False
    return enabled


def _pseudo_nmi_reason(config_text, cmdline, boot_log):
    if "CONFIG_ARM64_PSEUDO_NMI=y" not in config_text:
        return "CONFIG_ARM64_PSEUDO_NMI=y is required"
    if not _kernel_bool_param(cmdline, "irqchip.gicv3_pseudo_nmi"):
        return "irqchip.gicv3_pseudo_nmi is not enabled"
    if "Pseudo-NMIs enabled using " not in boot_log:
        return "GIC pseudo-NMI enablement was not observed in the current boot log"
    return ""


def _trigger_duration_ms(max_irqoff_ms):
    duration = min(max_irqoff_ms, NMI_PERF_MAX_IRQOFF_MS)
    if duration < NMI_PERF_MIN_IRQOFF_MS:
        raise ValueError(
            f"NMI perf sampling requires at least {NMI_PERF_MIN_IRQOFF_MS}ms "
            "of authorized IRQ-off time"
        )
    return duration


def _evaluate_trigger(trigger, cpu_id, duration_ms):
    if trigger.cpu != cpu_id or trigger.duration_ms != duration_ms:
        return "fail", "The kernel changed the requested CPU or duration"
    total = trigger.overflows
    nmi = trigger.nmi_overflows
    irqoff = trigger.irqoff_overflows
    irqoff_nmi = trigger.irqoff_nmi_overflows
    module_irqoff_nmi = trigger.module_irqoff_nmi_overflows
    counts = (
        f"total={total} nmi={nmi} irqoff={irqoff} "
        f"irqoff_nmi={irqoff_nmi} module_irqoff_nmi={module_irqoff_nmi}"
    )
    if not total:
        return "fail", f"No PMU overflow was observed: {counts}"
    if nmi != total:
        return "fail", f"Not every PMU overflow was in NMI context: {counts}"
    if irqoff > total or irqoff_nmi != irqoff:
        return "fail", f"Inconsistent IRQ-off/NMI overflow accounting: {counts}"
    if not irqoff_nmi:
        return "fail", f"No PMU overflow interrupted the IRQ-off probe: {counts}"
    if module_irqoff_nmi > irqoff_nmi:
        return "fail", f"Invalid probe-module overflow accounting: {counts}"
    if not module_irqoff_nmi:
        return "fail", f"No IRQ-off NMI overflow PC was inside the probe: {counts}"
    return "pass", f"Pseudo-NMI PMU overflow sampling on CPU {cpu_id}: {counts}"


def _run_nmi_perf_probe(logger, manifest_path=NMI_PERF_MANIFEST_PATH,
                        serial_path=DMI_SERIAL_PATH, cpu_root=CPU_SYSFS_ROOT,
                        module_dir=NMI_PERF_MODULE_PATH,
                        module_state_path=NMI_PERF_MODULE_STATE_PATH,
                        headers_root=Path("/lib/modules"), kernel_release=None,
                        device_path=NMI_PERF_DEVICE, proc_root=Path("/proc")):
    if kernel_release is None:
        kernel_release = platform.release()
    module_dir = Path(module_dir)
    module_state_path = Path(module_state_path)

    try:
        cpu_id, max_irqoff = _load_manifest(manifest_path, serial_path)
        config_text = _read_kernel_config_text()
        cmdline = _read_cmdline()
        backend_reason = _pseudo_nmi_reason(
            config_text, cmdline, glt_utils.get_kernel_log()
        )
        if backend_reason:
            return "skip", backend_reason

        required_configs = [
            "CONFIG_MODULES=y",
            "CONFIG_MODULE_UNLOAD=y",
            "CONFIG_PERF_EVENTS=y",
            "CONFIG_HW_PERF_EVENTS=y",
            "CONFIG_ARM_PMU=y",
            "CONFIG_ARM_PMUV3=y",
        ]
        missing = [name for name in required_configs if name not in config_text]
        if missing:
            return "skip", "NMI perf sampling requires " + ", ".join(missing)

        nmi_watchdog = (proc_root / "sys/kernel/nmi_watchdog").read_text(
            encoding="utf-8"
        ).strip()
        if nmi_watchdog != "0":
            return "skip", "nmi_watchdog=0 is required to isolate the perf sampling event"
        hardlockup_panic = (proc_root / "sys/kernel/hardlockup_panic").read_text(
            encoding="utf-8"
        ).strip()
        if hardlockup_panic != "0":
            return "skip", "hardlockup_panic=0 is required for the IRQ-off probe"
        duration_ms = _trigger_duration_ms(max_irqoff)
        online_cpus = parse_cpu_mask(Path(cpu_root, "online").read_text(encoding="utf-8"))
        allowed_cpus = set(os.sched_getaffinity(0))
        if cpu_id not in online_cpus or cpu_id not in allowed_cpus:
            return "skip", f"CPU {cpu_id} is not online and allowed by process affinity"
    except (OSError, ValueError, RuntimeError) as err:
        return "skip", f"Cannot validate NMI perf runtime preconditions: {err}"

    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return "skip", f"{command} is required for NMI perf validation"
    headers = Path(headers_root) / kernel_release / "build"
    if not headers.is_dir():
        return "skip", f"Matching kernel headers are unavailable: {headers}"
    if not module_dir.is_dir():
        return "skip", f"NMI perf probe sources are missing: {module_dir}"
    if module_state_path.exists():
        return "skip", f"{NMI_PERF_MODULE} is already loaded; refusing an unknown instance"

    loaded_by_test = False
    outcome = "pass"
    reason = ""
    cleanup_errors = []
    device_fd = None

    with glt_utils.chdir_managed(module_dir):
        try:
            try:
                glt_utils.run_command(["make"], "Cannot build NMI perf probe", timeout=120)
            except RuntimeError as err:
                outcome = "skip"
                reason = str(err)
            if outcome == "pass" and not glt_utils.sign_kernel_module(
                f"{NMI_PERF_MODULE}.ko", logger
            ):
                outcome = "fail"
                reason = f"Cannot sign {NMI_PERF_MODULE}.ko for Secure Boot"
            if outcome == "pass":
                try:
                    glt_utils.run_command(
                        ["insmod", f"{NMI_PERF_MODULE}.ko"],
                        "Cannot load NMI perf probe",
                        timeout=30,
                    )
                    loaded_by_test = True
                except RuntimeError as err:
                    if module_state_path.exists():
                        outcome = "fail"
                        reason = f"{err}; module ownership is unknown; manual cleanup required"
                    else:
                        outcome = "skip"
                        reason = str(err)
            if outcome == "pass":
                try:
                    device_fd = os.open(device_path, os.O_RDWR)
                except OSError as err:
                    outcome = "skip"
                    reason = f"Cannot open {device_path}: {err}"
            if outcome == "pass":
                try:
                    checkpoint = glt_utils.capture_kernel_log_checkpoint()
                    invalid_triggers = (
                        ("Unarmed", NmiPerfTrigger(
                            cpu=cpu_id, duration_ms=duration_ms, armed=0
                        )),
                        ("Zero-duration", NmiPerfTrigger(cpu=cpu_id, armed=1)),
                        ("Invalid-CPU", NmiPerfTrigger(
                            cpu=0xFFFFFFFF, duration_ms=duration_ms, armed=1
                        )),
                    )
                    for name, invalid_trigger in invalid_triggers:
                        rejected = False
                        try:
                            fcntl.ioctl(
                                device_fd, NMI_PERF_IOCTL_TRIGGER,
                                invalid_trigger, True,
                            )
                        except OSError as err:
                            if err.errno == errno.EINVAL:
                                rejected = True
                            else:
                                raise
                        if not rejected:
                            raise ValueError(f"{name} trigger was accepted")
                    kernel_log = glt_utils.get_kernel_log(checkpoint)
                    if "Watchdog detected hard LOCKUP" in kernel_log:
                        outcome = "fail"
                        reason = "Unexpected hardlockup before the active sampling trigger"
                    else:
                        checkpoint = glt_utils.capture_kernel_log_checkpoint()
                        trigger = NmiPerfTrigger(
                            cpu=cpu_id, duration_ms=duration_ms, armed=1
                        )
                        try:
                            fcntl.ioctl(
                                device_fd, NMI_PERF_IOCTL_TRIGGER, trigger, True
                            )
                        except OSError as err:
                            if err.errno in (errno.ENOSPC, errno.EBUSY):
                                outcome = "skip"
                                reason = f"No PMU counter was available for the probe: {err}"
                            else:
                                raise
                        else:
                            kernel_log = glt_utils.get_kernel_log(checkpoint)
                            if "Watchdog detected hard LOCKUP" in kernel_log:
                                outcome = "fail"
                                reason = "The bounded sampling probe caused a hardlockup"
                            else:
                                outcome, reason = _evaluate_trigger(
                                    trigger, cpu_id, duration_ms
                                )
                except (OSError, RuntimeError, ValueError) as err:
                    outcome = "fail"
                    reason = f"NMI perf trigger failed: {err}"
        finally:
            if device_fd is not None:
                try:
                    os.close(device_fd)
                except OSError as err:
                    cleanup_errors.append(str(err))
            if loaded_by_test:
                try:
                    glt_utils.run_command(
                        ["rmmod", NMI_PERF_MODULE],
                        "Failed to unload NMI perf probe",
                        timeout=30,
                    )
                except RuntimeError as err:
                    cleanup_errors.append(str(err))
            try:
                glt_utils.run_command(
                    ["make", "clean"], "Failed to clean NMI perf probe", timeout=120
                )
            except RuntimeError as err:
                cleanup_errors.append(str(err))

    if cleanup_errors:
        cleanup_reason = "; ".join(cleanup_errors)
        if outcome == "fail":
            reason = f"{reason}; cleanup also failed: {cleanup_reason}"
        else:
            outcome = "fail"
            reason = f"NMI perf probe cleanup failed: {cleanup_reason}"
    return outcome, reason


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_vera_nmi_perf_sampling(_, logger):
    """Verify pseudo-NMI PMU overflows interrupt a bounded IRQ-off probe."""
    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("NMI perf sampling validation is arm64-specific")
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("NMI perf sampling validation is Vera-specific")
    outcome, reason = _run_nmi_perf_probe(logger)
    if outcome == "fail":
        GLTCore.GLTTestResult.fail(reason)
    if outcome == "skip":
        GLTCore.GLTTestResult.skip(reason)
    logger.info("%s", reason)
