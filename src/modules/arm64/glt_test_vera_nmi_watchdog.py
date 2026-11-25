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

"""Exercise bounded IRQ-off hardlockup detection on an authorized Vera DUT."""

import ctypes
import errno
import fcntl
import json
import os
import platform
import re
import time
import zlib
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.cpu_topology import parse_cpu_mask
from common.nv_device import NvDev

NMI_MODULE = "vera_nmi_watchdog"
NMI_MODULE_PATH = Path(__file__).resolve().parent / "vera_nmi_watchdog"
NMI_MODULE_STATE_PATH = Path("/sys/module") / NMI_MODULE
NMI_DEVICE = "/dev/glt_vera_nmi_watchdog"
NMI_MANIFEST_PATH = Path("/etc/greenlit/vera-nmi-watchdog.json")
DMI_SERIAL_PATH = Path("/sys/class/dmi/id/product_serial")
CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
NMI_MAX_IRQOFF_MS = 25000
NMI_RESULT_TIMEOUT_SECONDS = 15
NMI_RESULT_POLL_SECONDS = 0.2
NMI_RESULT_QUIET_SECONDS = 0.2


class NmiTrigger(ctypes.Structure):
    _fields_ = [
        ("cpu", ctypes.c_uint32),
        ("duration_ms", ctypes.c_uint32),
        ("armed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


NMI_IOCTL_TRIGGER = (
    (1 << 30) | (ctypes.sizeof(NmiTrigger) << 16) | (ord("G") << 8) | 0x70
)
NMI_BLOCK_RE = re.compile(r"Watchdog detected hard LOCKUP on cpu (?P<cpu>[0-9]+)")


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
        raise ValueError(f"Cannot load NMI DUT manifest: {err}") from err
    if not isinstance(manifest, dict):
        raise ValueError("NMI DUT manifest must be an object")
    dut_id = manifest.get("dut_id")
    if not isinstance(dut_id, str) or not dut_id.strip():
        raise ValueError("NMI DUT manifest requires a non-empty dut_id")
    serial = Path(serial_path).read_text(encoding="utf-8").strip()
    if not serial or serial != dut_id.strip():
        raise ValueError("NMI DUT manifest does not match the running DUT identity")
    if manifest.get("recovery") != "ready":
        raise ValueError("NMI DUT manifest requires recovery=ready")
    backend = manifest.get("backend")
    if backend not in ("sdei", "pmu_pseudo_nmi"):
        raise ValueError("NMI backend must be sdei or pmu_pseudo_nmi")
    cpu = manifest.get("cpu")
    if not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0:
        raise ValueError("NMI manifest cpu must be a non-negative integer")
    max_irqoff = manifest.get("max_irqoff_ms", 15000)
    if not isinstance(max_irqoff, int) or isinstance(max_irqoff, bool):
        raise ValueError("max_irqoff_ms must be an integer")
    if max_irqoff <= 0 or max_irqoff > NMI_MAX_IRQOFF_MS:
        raise ValueError(f"max_irqoff_ms must be in 1..{NMI_MAX_IRQOFF_MS}")
    return manifest, backend, cpu, max_irqoff


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


def _backend_reason(backend, config_text, cmdline, boot_log):
    if backend == "sdei" and (
        "disable_sdei_nmi_watchdog" in cmdline or "disable-sdei-nmi-watchdog" in cmdline
    ):
        return "SDEI watchdog is disabled by kernel cmdline"
    sdei_registered = "SDEI Watchdog registered successfully" in boot_log
    if backend == "sdei":
        if not sdei_registered:
            return "SDEI backend was not registered in the current boot log"
        return ""
    if sdei_registered:
        return "SDEI backend is active; refusing to treat the PMU watchdog as the selected backend"
    if "CONFIG_ARM64_PSEUDO_NMI=y" not in config_text:
        return "CONFIG_ARM64_PSEUDO_NMI=y is required"
    if not _kernel_bool_param(cmdline, "irqchip.gicv3_pseudo_nmi"):
        return "irqchip.gicv3_pseudo_nmi is not enabled"
    if "Pseudo-NMIs enabled using " not in boot_log:
        return "GIC pseudo-NMI enablement was not observed in the current boot log"
    enabled_pos = boot_log.rfind("NMI watchdog: Enabled. Permanently consumes one hw-PMU counter.")
    disabled_pos = boot_log.rfind("Perf NMI watchdog permanently disabled")
    if disabled_pos > enabled_pos:
        return "Perf NMI watchdog is disabled"
    if enabled_pos < 0:
        return "PMU NMI watchdog enablement was not observed in the current boot log"
    return ""


def _hardlockup_blocks(kernel_log):
    blocks = []
    current = None
    for line in kernel_log.splitlines():
        match = NMI_BLOCK_RE.search(line)
        if match:
            if current is not None:
                blocks.append(current)
            current = {"cpu": int(match.group("cpu")), "text": line + "\n"}
        elif current is not None:
            current["text"] += line + "\n"
    if current is not None:
        blocks.append(current)
    return blocks


def _has_helper_callframe(block):
    call_trace = block["text"].find("Call trace:")
    return call_trace >= 0 and "glt_nmi_spin_thread" in block["text"][call_trace:]


def _await_hardlockup(checkpoint, cpu_id):
    deadline = time.monotonic() + NMI_RESULT_TIMEOUT_SECONDS
    candidate = None
    stable_since = None
    while True:
        blocks = _hardlockup_blocks(glt_utils.get_kernel_log(checkpoint))
        if len(blocks) > 1:
            raise ValueError(f"Unexpected additional hardlockup blocks: {len(blocks)}")
        if blocks and blocks[0]["cpu"] != cpu_id:
            raise ValueError(
                f"Unexpected hardlockup on CPU {blocks[0]['cpu']}, expected CPU {cpu_id}"
            )
        if blocks and _has_helper_callframe(blocks[0]):
            text = blocks[0]["text"]
            if candidate is None or candidate["text"] != text:
                candidate = blocks[0]
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= NMI_RESULT_QUIET_SECONDS:
                if len(blocks) == 1 and _has_helper_callframe(candidate):
                    return candidate
        else:
            candidate = None
            stable_since = None
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No stable complete hardlockup block for CPU {cpu_id} and "
                "glt_nmi_spin_thread arrived before timeout"
            )
        time.sleep(NMI_RESULT_POLL_SECONDS)


def _trigger_duration_ms(watchdog_thresh, max_irqoff_ms):
    duration = (2 * watchdog_thresh + 2) * 1000
    if duration > max_irqoff_ms:
        raise ValueError(
            f"Required IRQ-off duration {duration} exceeds manifest limit {max_irqoff_ms}"
        )
    return duration


def _run_nmi_probe(logger, manifest_path=NMI_MANIFEST_PATH, serial_path=DMI_SERIAL_PATH,
                   cpu_root=CPU_SYSFS_ROOT, module_dir=NMI_MODULE_PATH,
                   module_state_path=NMI_MODULE_STATE_PATH,
                   headers_root=Path("/lib/modules"), kernel_release=None,
                   device_path=NMI_DEVICE, proc_root=Path("/proc")):
    if kernel_release is None:
        kernel_release = platform.release()
    module_dir = Path(module_dir)
    module_state_path = Path(module_state_path)

    try:
        manifest, backend, cpu_id, max_irqoff = _load_manifest(manifest_path, serial_path)
    except (OSError, ValueError) as err:
        return "skip", str(err)
    try:
        config_text = _read_kernel_config_text()
        cmdline = _read_cmdline()
        backend_reason = _backend_reason(backend, config_text, cmdline, glt_utils.get_kernel_log())
        if backend_reason:
            return "skip", backend_reason

        required_configs = ["CONFIG_MODULES=y", "CONFIG_MODULE_UNLOAD=y", "CONFIG_HARDLOCKUP_DETECTOR=y"]
        if backend == "sdei":
            required_configs.append("CONFIG_SDEI_WATCHDOG=y")
        else:
            required_configs.append("CONFIG_HARDLOCKUP_DETECTOR_PERF=y")
        missing = [name for name in required_configs if name not in config_text]
        if missing:
            return "skip", "NMI watchdog requires " + ", ".join(missing)

        nmi_watchdog = (proc_root / "sys/kernel/nmi_watchdog").read_text(encoding="utf-8").strip()
        if nmi_watchdog != "1":
            return "skip", "nmi_watchdog must be enabled for this test"
        hardlockup_panic = (proc_root / "sys/kernel/hardlockup_panic").read_text(encoding="utf-8").strip()
        if hardlockup_panic != "0":
            return "skip", "hardlockup_panic=0 is required for the non-panic scenario"
        watchdog_thresh = int((proc_root / "sys/kernel/watchdog_thresh").read_text(encoding="utf-8").strip())
        if watchdog_thresh <= 0:
            return "skip", f"Invalid watchdog_thresh {watchdog_thresh}"
        try:
            duration_ms = _trigger_duration_ms(watchdog_thresh, max_irqoff)
        except ValueError as err:
            return "skip", str(err)

        online_cpus = parse_cpu_mask(Path(cpu_root, "online").read_text(encoding="utf-8"))
        allowed_cpus = set(os.sched_getaffinity(0))
        if cpu_id not in online_cpus or cpu_id not in allowed_cpus:
            return "skip", f"CPU {cpu_id} is not online and allowed by process affinity"
    except (OSError, ValueError, RuntimeError) as err:
        return "skip", f"Cannot validate NMI runtime preconditions: {err}"

    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return "skip", f"{command} is required for NMI hardlockup validation"
    headers = Path(headers_root) / kernel_release / "build"
    if not headers.is_dir():
        return "skip", f"Matching kernel headers are unavailable: {headers}"
    if not module_dir.is_dir():
        return "skip", f"NMI probe sources are missing: {module_dir}"
    if module_state_path.exists():
        return "skip", f"{NMI_MODULE} is already loaded; refusing an unknown instance"

    loaded_by_test = False
    outcome = "pass"
    reason = ""
    cleanup_errors = []
    device_fd = None
    original_affinity = allowed_cpus

    with glt_utils.chdir_managed(module_dir):
        try:
            try:
                glt_utils.run_command(["make"], "Cannot build NMI hardlockup probe", timeout=120)
            except RuntimeError as err:
                outcome = "skip"
                reason = str(err)
            if outcome == "pass" and not glt_utils.sign_kernel_module(f"{NMI_MODULE}.ko", logger):
                outcome = "fail"
                reason = f"Cannot sign {NMI_MODULE}.ko for Secure Boot"
            if outcome == "pass":
                try:
                    glt_utils.run_command(
                        ["insmod", f"{NMI_MODULE}.ko"], "Cannot load NMI hardlockup probe", timeout=30,
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
                os.sched_setaffinity(0, {cpu_id})
                try:
                    checkpoint = glt_utils.capture_kernel_log_checkpoint()
                    unarmed_rejected = False
                    try:
                        fcntl.ioctl(device_fd, NMI_IOCTL_TRIGGER, NmiTrigger(cpu_id, 0, 0, 0))
                    except OSError as err:
                        if err.errno == errno.EINVAL:
                            unarmed_rejected = True
                        else:
                            raise
                    if not unarmed_rejected:
                        raise ValueError("Unarmed zero-duration trigger was accepted")
                    blocks = _hardlockup_blocks(glt_utils.get_kernel_log(checkpoint))
                    if blocks:
                        outcome = "fail"
                        reason = "Unexpected hardlockup before the active trigger"
                    else:
                        checkpoint = glt_utils.capture_kernel_log_checkpoint()
                        fcntl.ioctl(
                            device_fd, NMI_IOCTL_TRIGGER,
                            NmiTrigger(cpu_id, duration_ms, 1, 0),
                        )
                        _await_hardlockup(checkpoint, cpu_id)
                        outcome = "pass"
                        reason = f"Hardlockup on CPU {cpu_id} was detected with the helper stack"
                except (OSError, RuntimeError, ValueError) as err:
                    outcome = "fail"
                    reason = f"NMI hardlockup trigger failed: {err}"
        finally:
            try:
                os.sched_setaffinity(0, original_affinity)
            except OSError as err:
                cleanup_errors.append(str(err))
            if device_fd is not None:
                try:
                    os.close(device_fd)
                except OSError as err:
                    cleanup_errors.append(str(err))
            if loaded_by_test:
                try:
                    glt_utils.run_command(
                        ["rmmod", NMI_MODULE], "Failed to unload NMI hardlockup probe", timeout=30,
                    )
                except RuntimeError as err:
                    cleanup_errors.append(str(err))
            try:
                glt_utils.run_command(
                    ["make", "clean"], "Failed to clean NMI hardlockup probe", timeout=120,
                )
            except RuntimeError as err:
                cleanup_errors.append(str(err))

    if cleanup_errors:
        cleanup_reason = "; ".join(cleanup_errors)
        if outcome == "fail":
            reason = f"{reason}; cleanup also failed: {cleanup_reason}"
        else:
            outcome = "fail"
            reason = f"NMI probe cleanup failed: {cleanup_reason}"
    return outcome, reason


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_vera_nmi_hardlockup(_, logger):
    """Verify bounded IRQ-off hardlockup detection on an authorized Vera DUT."""
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("NMI hardlockup validation is Vera-specific")
    outcome, reason = _run_nmi_probe(logger)
    if outcome == "fail":
        GLTCore.GLTTestResult.fail(reason)
    if outcome == "skip":
        GLTCore.GLTTestResult.skip(reason)
    logger.info("%s", reason)
