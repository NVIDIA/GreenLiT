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

"""GIC and irqchip validation tests."""

import glob
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from common.virt.glt_libvirt import GLTLibvirt


# GICv4 IRQ scaling test configuration.
#
# DGX-8229 and DGX-8257 both require GICv4 to be active, so this test enables
# kvm-arm.vgic_v4_enable=1 once and validates both signals in the same boot:
# - DGX-8229: repeated high-fanout VM create/teardown to catch the GICv4 VPE
#   activation/deactivation doorbell race.
# - DGX-8257: large interrupt descriptor pressure from many high-vCPU VMs.
#
# Progress marker: this test appended the GICv4 boot parameter and owns cleanup.
GICV4_IRQ_BOOT_PARAM_MARKER = "GICv4BootParamAdded"
# Progress marker: persistent boot config was removed and a restore reboot began.
GICV4_IRQ_RESTORE_REBOOT_MARKER = "GICv4RestoreReboot"
# Progress value: original pass/fail reason to preserve across the restore reboot.
GICV4_IRQ_RESTORE_RESULT_REASON = "GICv4RestoreResultReason"
GICV4_IRQ_CMDLINE_ARG = "kvm-arm.vgic_v4_enable=1"
GLT_REBOOT_MARKER = "TriggerReboot"
GICV4_IRQ_TEST_PARAM_KEYS = {
    "duration_seconds": "GICv4DurationSeconds",
    "memory_mb": "GICv4MemoryMb",
    "num_vcpus": "GICv4NumVcpus",
    "num_vms": "GICv4NumVms",
    "settle_seconds": "GICv4SettleSeconds",
}
# libvirt's QEMU confinement profiles normally allow direct-boot artifacts here.
GICV4_IRQ_LIBVIRT_STAGING_DIR = "/var/lib/libvirt/boot"
GICV4_IRQ_LOG_MESSAGES = (
    "GICv4 support enabled",
    "GICv4.1 support enabled",
)
GICV4_EXTRACT_VMLINUX_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class GICv4TestParams:
    """Runtime parameters for the GICv4 stress test."""

    duration_seconds: int
    num_vms: int
    num_vcpus: int
    memory_mb: int
    settle_seconds: int


@dataclass(frozen=True)
class GICv4StressContext:
    """Runtime context for one GICv4 IRQ stress run."""

    initrd_path: str
    kernel_path: str
    libvirt_conn: GLTLibvirt
    logger: object
    params: GICv4TestParams


def _get_progress_value(sd, test_name: str, key: str):
    """Return a saved progress value for a test case."""
    for entry in getattr(sd, "progress", []):
        if test_name in entry:
            return entry[test_name].get(key)
    return None


def _disable_gicv4_resume_service_if_needed(test_name: str, sd) -> None:
    """Disable GreenLiT auto-resume after this test consumes a reboot marker."""
    if sd.progress_check(test_name, GLT_REBOOT_MARKER):
        glt_utils.systemd_enable(enable=False)


def _get_saved_gicv4_irq_test_parameters(test_name: str, sd) -> Optional[GICv4TestParams]:
    """Return previously selected runtime parameters after a reboot resume."""
    values = {}
    for field_name, progress_key in GICV4_IRQ_TEST_PARAM_KEYS.items():
        value = _get_progress_value(sd, test_name, progress_key)
        if value is None:
            return None
        values[field_name] = int(value)
    return GICv4TestParams(**values)


def _save_gicv4_irq_test_parameters(test_name: str, sd, params: GICv4TestParams) -> None:
    """Persist runtime parameters so reboot resume uses the same test shape."""
    values = {
        "duration_seconds": params.duration_seconds,
        "memory_mb": params.memory_mb,
        "num_vcpus": params.num_vcpus,
        "num_vms": params.num_vms,
        "settle_seconds": params.settle_seconds,
    }
    for field_name, progress_key in GICV4_IRQ_TEST_PARAM_KEYS.items():
        sd.progress_update(test_name, progress_key, values[field_name])


def _get_env_int(name: str, default: int) -> int:
    """Read a positive integer environment override."""
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer") from e

    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _get_irq_count() -> int:
    """Return the number of allocated kernel IRQ descriptors."""
    return len(os.listdir("/sys/kernel/irq"))


def _find_first_existing(patterns: list[str], name_tokens: tuple[str, ...]) -> str:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        for match in matches:
            basename = os.path.basename(match)
            if os.path.isfile(match) and any(token in basename for token in name_tokens):
                return match
    return ""


def _get_kernel_initrd_path() -> tuple[str, str]:
    """Find boot artifacts for direct kernel boot domains."""
    release = platform.release()
    kernel_path = _find_first_existing([
        f"/boot/Image-{release}",
        "/boot/Image",
        f"/boot/vmlinuz-{release}",
        "/boot/vmlinuz",
        f"/boot/*{release}*",
    ], ("vmlinuz", "Image"))
    initrd_path = _find_first_existing([
        f"/boot/initrd.img-{release}",
        f"/boot/initramfs-{release}.img",
        f"/boot/initrd-{release}",
        "/boot/initrd.img",
        "/boot/initrd",
        f"/boot/*{release}*",
    ], ("initrd", "initramfs"))

    return kernel_path, initrd_path


def _check_gicv4_memory_capacity(num_vms: int, memory_mb: int) -> tuple[bool, str]:
    """Make sure the GICv4 VM fanout does not knowingly overcommit memory."""
    try:
        meminfo = _read_file("/proc/meminfo")
    except OSError as e:
        return False, f"Unable to read /proc/meminfo: {e}"

    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", meminfo, re.MULTILINE)
    if not match:
        return False, "Unable to determine available memory"

    available_mb = int(match.group(1)) // 1024
    required_mb = num_vms * memory_mb
    if available_mb < required_mb:
        return False, (
            f"Need at least {required_mb} MiB available for {num_vms} VMs, "
            f"but only {available_mb} MiB is available"
        )

    return True, ""


def _get_gicv4_platform_skip_reason() -> str:
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        return "This is not an NVIDIA Grace CPU."

    socket_count = NvDev.get_cpu_socket_count()
    if socket_count == -1:
        return "Unable to determine CPU socket count."
    if socket_count > 2:
        return (
            f"GICv4 is disabled on {socket_count}-socket Grace systems "
            "affected by T241-FABRIC-4."
        )

    return ""


def _get_qemu_emulator_path() -> str:
    """Find a local QEMU emulator usable by libvirt on aarch64 hosts."""
    for candidate in (
        shutil.which("qemu-system-aarch64"),
        shutil.which("qemu-kvm"),
        "/usr/libexec/qemu-kvm",
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def _get_gicv4_log_failure_reason() -> str:
    if not any(glt_utils.check_kernel_log(message) for message in GICV4_IRQ_LOG_MESSAGES):
        return "GICv4 support is not enabled in current boot logs."

    return ""


def _check_gicv4_restore_reboot_complete(test_name: str, sd) -> tuple[bool, str]:
    """Complete the test after rebooting out of a GLT-added GICv4 boot parameter."""

    # This marker means the test session rebooted after removing the GICv4
    # boot parameter.
    if not sd.progress_check(test_name, GICV4_IRQ_RESTORE_REBOOT_MARKER):
        return False, ""

    status = glt_utils.check_kernel_parameter(
        GICV4_IRQ_CMDLINE_ARG,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )

    if status.active:
        return True, (
            f"System rebooted after removing {GICV4_IRQ_CMDLINE_ARG}, "
            "but the parameter is still active."
        )
    if status.configured or status.added_by_glt:
        return True, (
            f"System rebooted after removing {GICV4_IRQ_CMDLINE_ARG}, "
            "but the parameter is still configured."
        )

    # Return the original pass/fail reason to preserve across the restore reboot.
    return True, _get_progress_value(sd, test_name, GICV4_IRQ_RESTORE_RESULT_REASON) or ""


# pylint: disable=too-many-return-statements
def _prepare_gicv4_boot_parameter(test_name, sd, logger) -> tuple[bool, str]:
    """Append the GICv4 kernel parameter and reboot when current boot lacks it."""
    status = glt_utils.check_kernel_parameter(
        GICV4_IRQ_CMDLINE_ARG,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )

    # If parameter is active, it means the system is ready for performing the stress.
    if status.active:
        if sd.progress_check(test_name, GICV4_IRQ_BOOT_PARAM_MARKER):
            # The setup reboot succeeded. Drop persistent config before stress
            # so an unexpected later reboot naturally returns to the original boot.
            ok, reason = _remove_gicv4_boot_parameter(test_name, sd)
            if not ok:
                return False, reason
        return True, ""

    if sd.progress_check(test_name, GLT_REBOOT_MARKER):
        reason = (
            f"System rebooted but {GICV4_IRQ_CMDLINE_ARG} is still missing from /proc/cmdline."
        )
        if sd.progress_check(test_name, GICV4_IRQ_BOOT_PARAM_MARKER):
            ok, restore_reason = _remove_gicv4_boot_parameter(test_name, sd)
            if not ok:
                reason = (
                    f"{reason}; additionally, {restore_reason}"
                )
        return False, reason

    logger.info(
        "Appending %s for GICv4 validation.",
        GICV4_IRQ_CMDLINE_ARG,
    )
    ok, reason = glt_utils.append_kernel_parameter(
        GICV4_IRQ_CMDLINE_ARG,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if not ok:
        cleanup_ok, cleanup_reason = _remove_gicv4_boot_parameter(test_name, sd)
        if cleanup_ok:
            return False, f"Failed to append kernel parameter: {reason}"
        return False, (
            f"Failed to append kernel parameter: {reason}; additionally, {cleanup_reason}"
        )

    status = glt_utils.check_kernel_parameter(
        GICV4_IRQ_CMDLINE_ARG,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if status.added_by_glt:
        sd.progress_record_time(test_name, GICV4_IRQ_BOOT_PARAM_MARKER)

    # Current persistent bootloader backends normally need a reboot after a
    # missing parameter is configured. Keep honoring the API for future backends
    # that can make the parameter effective without rebooting.
    if not status.reboot_required:
        return True, ""

    glt_utils.reboot(test_name, sd)

    # glt_utils.reboot() should not return after a real reboot request. If it
    # does return, roll back any GLT-owned persistent config before failing.
    # remove_kernel_parameter() is a safe no-op when this test does not own it.
    ok, reason = _remove_gicv4_boot_parameter(test_name, sd)
    if not ok:
        return False, f"Reboot did not exit, and {reason}"
    return False, "Reboot was triggered but the process did not exit."


def _remove_gicv4_boot_parameter(test_name: str, sd) -> tuple[bool, str]:
    """Remove the GICv4 parameter only if it was appended by GreenLiT."""
    ok, reason = glt_utils.remove_kernel_parameter(
        GICV4_IRQ_CMDLINE_ARG,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if not ok:
        return False, f"Failed to remove kernel command-line configuration: {reason}"
    return True, ""


def _restore_gicv4_boot_parameter(
    test_name: str,
    sd,
    logger,
    result_reason: str = "",
) -> tuple[bool, str]:
    """Remove GLT-owned config and reboot if needed to restore active kernel state."""
    if result_reason:
        sd.progress_update(test_name, GICV4_IRQ_RESTORE_RESULT_REASON, result_reason)

    # This means GreenLiT doesn't own the GICv4 boot parameter.
    if not sd.progress_check(test_name, GICV4_IRQ_BOOT_PARAM_MARKER):
        return True, ""

    ok, reason = _remove_gicv4_boot_parameter(test_name, sd)
    if not ok:
        return False, reason

    status = glt_utils.check_kernel_parameter(
        GICV4_IRQ_CMDLINE_ARG,
        test_case_name=test_name,
        session_dir=sd.log_dir,
    )
    if not status.active:
        # The active boot already matches the original state, so no restore
        # reboot is needed.
        return True, ""

    logger.info("Rebooting to restore kernel command-line state after GICv4 validation.")
    sd.progress_record_time(test_name, GICV4_IRQ_RESTORE_REBOOT_MARKER)
    glt_utils.reboot(test_name, sd)
    return False, "Restore reboot was triggered but the process did not exit."


def _get_gicv4_tool_failure_reason() -> str:
    if not glt_utils.is_cli_available("virsh"):
        return "Please install 'virsh' to perform the test."
    if not _get_qemu_emulator_path():
        return "Please install 'qemu-system-aarch64' or 'qemu-kvm' to perform the test."
    kernel_path, _ = _get_kernel_initrd_path()
    if (
        kernel_path and
        os.path.basename(kernel_path).startswith("vmlinuz") and
        not _get_extract_vmlinux_path()
    ):
        return f"Please install 'extract-vmlinux' to use compressed kernel {kernel_path}."

    return ""


def _get_gicv4_irq_test_parameters(test_name: str, sd) -> GICv4TestParams:
    """Return GICv4 IRQ stress settings, allowing per-run environment overrides."""
    saved_params = _get_saved_gicv4_irq_test_parameters(test_name, sd)
    if saved_params is not None:
        return saved_params

    params = GICv4TestParams(
        duration_seconds=_get_env_int("GLT_GICV4_DURATION_SECONDS", 1800),
        num_vms=_get_env_int("GLT_GICV4_NUM_VMS", 100),
        num_vcpus=_get_env_int("GLT_GICV4_VCPUS", 128),
        memory_mb=_get_env_int("GLT_GICV4_MEMORY_MB", 512),
        settle_seconds=_get_env_int("GLT_GICV4_SETTLE_SECONDS", 5),
    )
    _save_gicv4_irq_test_parameters(test_name, sd, params)
    return params


def _get_gicv4_boot_artifacts() -> tuple[str, str, str]:
    kernel_path, initrd_path = _get_kernel_initrd_path()
    if not kernel_path:
        return "", "", "Unable to find a bootable host kernel under /boot."
    if not initrd_path:
        return "", "", "Unable to find a bootable host initrd under /boot."

    return kernel_path, initrd_path, ""


def _stage_boot_artifact(path: str) -> tuple[str, str]:
    destination = os.path.join(
        GICV4_IRQ_LIBVIRT_STAGING_DIR,
        f"glt-gicv4-{os.getpid()}-{os.path.basename(path)}",
    )
    try:
        os.makedirs(GICV4_IRQ_LIBVIRT_STAGING_DIR, exist_ok=True)
        shutil.copy2(path, destination)
        os.chmod(destination, 0o644)
    except OSError as e:
        try:
            os.remove(destination)
        except FileNotFoundError:
            pass
        return "", f"Failed to stage {path} for libvirt: {e}"
    return destination, ""


def _get_extract_vmlinux_path() -> str:
    """Find the kernel helper that extracts raw Images from compressed vmlinuz files."""
    release = platform.release()
    candidates = [
        shutil.which("extract-vmlinux"),
        f"/usr/src/linux-headers-{release}/scripts/extract-vmlinux",
    ]
    candidates.extend(sorted(glob.glob("/usr/src/linux-*/scripts/extract-vmlinux"), reverse=True))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def _stage_gicv4_kernel_artifact(path: str) -> tuple[str, str]:
    """Stage a QEMU-loadable kernel image, extracting compressed vmlinuz if needed."""
    if not os.path.basename(path).startswith("vmlinuz"):
        return _stage_boot_artifact(path)

    # Ubuntu arm64 vmlinuz files can be EFI zboot images compressed with zstd.
    # QEMU/libvirt direct boot expects a raw Image, so extract before staging.
    # SLES/RHEL often expose a directly bootable Image and skip this path.
    extractor = _get_extract_vmlinux_path()
    if not extractor:
        return "", f"Unable to find extract-vmlinux for compressed kernel {path}"

    destination = os.path.join(
        GICV4_IRQ_LIBVIRT_STAGING_DIR,
        f"glt-gicv4-{os.getpid()}-Image-{platform.release()}",
    )
    try:
        os.makedirs(GICV4_IRQ_LIBVIRT_STAGING_DIR, exist_ok=True)
        with open(destination, "wb") as output:
            # extract-vmlinux writes the raw kernel Image to stdout.
            result = subprocess.run(
                [extractor, path],
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=GICV4_EXTRACT_VMLINUX_TIMEOUT_SECONDS,
            )
        os.chmod(destination, 0o644)
    except subprocess.TimeoutExpired as e:
        try:
            os.remove(destination)
        except FileNotFoundError:
            pass
        return "", f"Timed out extracting kernel image from {path}: {e}"
    except OSError as e:
        try:
            os.remove(destination)
        except FileNotFoundError:
            pass
        return "", f"Failed to extract {path} for libvirt: {e}"
    if result.returncode != 0:
        try:
            os.remove(destination)
        except FileNotFoundError:
            pass
        reason = result.stderr.decode("utf-8", errors="replace").strip()
        return "", reason or f"Failed to extract kernel image from {path}"
    return destination, ""


def _stage_boot_artifacts(kernel_path: str, initrd_path: str) -> tuple[str, str, list[str], str]:
    staged_paths = []
    staged_kernel, reason = _stage_gicv4_kernel_artifact(kernel_path)
    if reason:
        return "", "", staged_paths, reason
    staged_paths.append(staged_kernel)

    staged_initrd = ""
    if initrd_path:
        staged_initrd, reason = _stage_boot_artifact(initrd_path)
        if reason:
            return "", "", staged_paths, reason
        staged_paths.append(staged_initrd)

    return staged_kernel, staged_initrd, staged_paths, ""


def _cleanup_staged_boot_artifacts(staged_paths: list[str]) -> None:
    for path in staged_paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _create_gicv4_domains(
    libvirt_conn: GLTLibvirt,
    iteration: int,
    params: GICv4TestParams,
    kernel_path: str,
    initrd_path: str,
) -> tuple[list[str], str]:
    domain_names = []
    qemu_emulator = _get_qemu_emulator_path()
    kernel_args = "console=ttyAMA0 panic=0"
    if initrd_path:
        kernel_args += " rdinit=/bin/sh"
    for idx in range(1, params.num_vms + 1):
        domain_name = f"glt-gicv4-{os.getpid()}-{iteration}-{idx}"
        xml_config = libvirt_conn.generate_domain_xml({
            "domain_name": domain_name,
            "memory_mb": params.memory_mb,
            "vcpus": params.num_vcpus,
            "arch": "aarch64",
            "emulator": qemu_emulator,
            "kernel_path": kernel_path,
            "initrd_path": initrd_path,
            "kernel_args": kernel_args,
        })

        if not libvirt_conn.define_and_start(xml_config):
            return domain_names, f"Failed to create VM {domain_name}"
        domain_names.append(domain_name)

    return domain_names, ""


def _get_domain_state_failure(libvirt_conn: GLTLibvirt, domain_names: list[str]) -> str:
    for domain_name in domain_names:
        success, output, _ = libvirt_conn.run_virsh_command(["domstate", domain_name])
        if not success or "running" not in output.lower():
            return f"VM {domain_name} is not running after creation: {output}"

    return ""


def _cleanup_gicv4_domains(libvirt_conn: GLTLibvirt, domain_names: list[str]) -> None:
    """Destroy all GICv4 domains first, then undefine them."""
    for domain_name in reversed(domain_names):
        libvirt_conn.destroy(domain_name)
    for domain_name in reversed(domain_names):
        libvirt_conn.undefine(domain_name)


def _run_gicv4_irq_iteration(context: GICv4StressContext, iteration: int) -> str:
    domain_names = []
    original_irqs = _get_irq_count()

    try:
        domain_names, failure_reason = _create_gicv4_domains(
            context.libvirt_conn,
            iteration,
            context.params,
            context.kernel_path,
            context.initrd_path,
        )
        if failure_reason:
            return failure_reason

        time.sleep(context.params.settle_seconds)

        failure_reason = _get_domain_state_failure(context.libvirt_conn, domain_names)
        if failure_reason:
            return failure_reason

        final_irqs = _get_irq_count()
        irq_delta = final_irqs - original_irqs
        context.logger.info(
            "IRQ descriptors changed from %d to %d, delta %d",
            original_irqs, final_irqs, irq_delta,
        )
        # DGX-8257 is about IRQ descriptor capacity. GICv4 allocates per-vCPU
        # VPE/doorbell resources, so the VM fanout should scale with both VM
        # count and vCPU count.
        min_expected_irqs = context.params.num_vms * context.params.num_vcpus
        if irq_delta < min_expected_irqs:
            return (
                f"Expected at least {min_expected_irqs} new IRQ descriptors "
                f"({context.params.num_vms} VMs x {context.params.num_vcpus} vCPUs), got {irq_delta}"
            )
        return ""
    finally:
        _cleanup_gicv4_domains(context.libvirt_conn, domain_names)


def _run_gicv4_irq_stress(context: GICv4StressContext) -> str:
    """Run repeated GICv4 VM create/teardown cycles for the configured duration."""
    deadline = time.monotonic() + context.params.duration_seconds
    iteration = 0

    while iteration == 0 or time.monotonic() < deadline:
        iteration += 1
        context.logger.info("Starting GICv4 stress iteration %d", iteration)
        failure_reason = _run_gicv4_irq_iteration(context, iteration)
        if failure_reason:
            return failure_reason

    context.logger.info("Completed %d GICv4 stress iterations", iteration)
    return ""


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches,broad-exception-caught
def test_gicv4_irq_scaling(sd, logger):
    """
    Validate Grace GICv4 VM fanout and IRQ scaling in one GICv4-enabled boot.

    DGX-8229: Stress repeated create/teardown of about 100 high-vCPU VMs to
        verify the GICv4.1 VPE activation/deactivation doorbell race fix.
    DGX-8257: Reuse the same VM fanout to exercise interrupt descriptor
        pressure from many vCPUs without a second kernel-parameter reboot.

    PATCHES:
        DGX-8229: https://git.kernel.org/torvalds/c/ef3691683d7bfd0a2acf48812e4ffe894f10bfa8
        DGX-8257: https://git.kernel.org/torvalds/c/721255b9826bd11c7a38b585905fc2dd0fb94e52
    """
    test_name = test_gicv4_irq_scaling.__name__

    # If this invocation is a systemd resume after a GreenLiT reboot, consume
    # the resume service immediately. Any later planned reboot will re-enable
    # it through glt_utils.reboot().
    _disable_gicv4_resume_service_if_needed(test_name, sd)

    restore_complete, restore_reason = _check_gicv4_restore_reboot_complete(test_name, sd)
    if restore_complete:
        # If the restore_reason is given, it means the restore happened because a failure occurred during the test.
        if restore_reason:
            GLTCore.GLTTestResult.fail(restore_reason)
        return

    skip_reason = _get_gicv4_platform_skip_reason()
    if skip_reason:
        GLTCore.GLTTestResult.skip(skip_reason)

    failure_reason = _get_gicv4_tool_failure_reason()
    if failure_reason:
        GLTCore.GLTTestResult.fail(failure_reason)

    kernel_path, initrd_path, failure_reason = _get_gicv4_boot_artifacts()
    if failure_reason:
        GLTCore.GLTTestResult.fail(failure_reason)

    try:
        params = _get_gicv4_irq_test_parameters(test_name, sd)
    except ValueError as e:
        GLTCore.GLTTestResult.fail(str(e))

    enough_memory, reason = _check_gicv4_memory_capacity(params.num_vms, params.memory_mb)
    if not enough_memory:
        GLTCore.GLTTestResult.fail(reason)

    try:
        libvirt_conn = GLTLibvirt(logger=logger)
    except ConnectionError as e:
        GLTCore.GLTTestResult.fail(str(e))

    # This may reboot and resume. When it returns successfully, current boot is
    # ready for GICv4 validation.
    ok, reason = _prepare_gicv4_boot_parameter(test_name, sd, logger)
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    failure_reason = _get_gicv4_log_failure_reason()
    if failure_reason:
        ok, reason = _restore_gicv4_boot_parameter(test_name, sd, logger, failure_reason)
        if not ok:
            GLTCore.GLTTestResult.fail(f"{failure_reason}; additionally, {reason}")
        GLTCore.GLTTestResult.fail(failure_reason)

    logger.info(
        "Starting GICv4 stress for %d seconds with %d VMs, %d vCPUs per VM, %d MiB per VM",
        params.duration_seconds,
        params.num_vms, params.num_vcpus, params.memory_mb,
    )

    kernel_path, initrd_path, staged_paths, failure_reason = _stage_boot_artifacts(
        kernel_path, initrd_path
    )
    if failure_reason:
        _cleanup_staged_boot_artifacts(staged_paths)
        ok, reason = _restore_gicv4_boot_parameter(test_name, sd, logger, failure_reason)
        if not ok:
            GLTCore.GLTTestResult.fail(f"{failure_reason}; additionally, {reason}")
        GLTCore.GLTTestResult.fail(failure_reason)

    stress_failure_reason = ""
    try:
        context = GICv4StressContext(
            initrd_path=initrd_path,
            kernel_path=kernel_path,
            libvirt_conn=libvirt_conn,
            logger=logger,
            params=params,
        )
        stress_failure_reason = _run_gicv4_irq_stress(context)
    except Exception as e:
        stress_failure_reason = f"GICv4 stress raised unexpected exception: {e!r}"
    finally:
        _cleanup_staged_boot_artifacts(staged_paths)

    if stress_failure_reason:
        ok, reason = _restore_gicv4_boot_parameter(test_name, sd, logger, stress_failure_reason)
        if not ok:
            GLTCore.GLTTestResult.fail(f"{stress_failure_reason}; additionally, {reason}")
        GLTCore.GLTTestResult.fail(stress_failure_reason)

    ok, reason = _restore_gicv4_boot_parameter(test_name, sd, logger)
    if not ok:
        GLTCore.GLTTestResult.fail(reason)
