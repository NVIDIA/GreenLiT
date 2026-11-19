# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Arm64 KVM instruction fetches from cacheable device PFNMAP memory."""

import os
import platform
import subprocess
import tempfile
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common import pci_utils
from common import virt as glt_virt


PROBE_DIR = Path(__file__).resolve().parent / "kvm_cacheable_pfnmap"
KVM_DIR = Path(glt_virt.__file__).resolve().parent / "kvm_microvm"
# Optional full-domain PCI address; see the testcase docstring for selection policy.
BDF_ENV = "GLT_KVM_CACHEABLE_PFNMAP_BDF"
GPU_DRIVER = "nvgrace_gpu_vfio_pci"
# Override-only IDs from drivers/vfio/pci/nvgrace-gpu/main.c.
GPU_DEVICE_IDS = {"0x2342", "0x2345", "0x2348", "0x2941", "0x31c2"}


def _unused_gpu_reason(bdf):  # pylint: disable=too-many-return-statements
    """Apply the existing GPU tests' unbound, reset and display ownership gates."""
    driver = pci_utils.get_pci_driver(bdf)
    if driver is not None:
        return f"GPU {bdf} is owned by {driver}"
    gpu = Path("/sys/bus/pci/devices") / bdf
    try:
        gpu.stat()
    except FileNotFoundError:
        return f"GPU {bdf} is absent"
    if (glt_utils.read_sysfs_value(gpu / "vendor") != "0x10de"
            or not glt_utils.read_sysfs_value(gpu / "class").startswith("0x03")):
        return f"Device {bdf} is not an NVIDIA display device"
    if glt_utils.read_sysfs_value(gpu / "device") not in GPU_DEVICE_IDS:
        return f"GPU {bdf} is not a supported Grace GPU"
    try:
        boot_vga = glt_utils.read_sysfs_value(gpu / "boot_vga")
    except FileNotFoundError:
        boot_vga = "0"
    if boot_vga != "0":
        return f"GPU {bdf} is the boot display"
    try:
        frames = list(Path("/sys/class/graphics").iterdir())
    except FileNotFoundError:
        frames = []
    for frame in frames:
        if not frame.name.startswith("fb") or not frame.name[2:].isdecimal():
            continue
        try:
            target = (frame / "device").resolve(strict=True)
        except FileNotFoundError:
            continue
        if target.is_relative_to(gpu.resolve(strict=True)):
            return f"GPU {bdf} owns a framebuffer"
    try:
        (gpu / "reset").stat()
    except FileNotFoundError:
        return f"GPU {bdf} has no supported reset"
    return None


def _isolated_gpu_group(bdf, *, allow_absent=False):
    """Reject shared groups; some kernels create the group only during binding."""
    try:
        members = {path.name for path in (
            Path('/sys/bus/pci/devices') / bdf / 'iommu_group/devices').iterdir()}
    except FileNotFoundError:
        return allow_absent
    return members == {bdf}


def _discover_unused_gpu(logger, excluded=()):
    """Choose an isolated unused Grace GPU without stealing a staged override."""
    try:
        devices = sorted(Path("/sys/bus/pci/devices").iterdir())
    except FileNotFoundError:
        return None
    unsafe = pci_utils.unsafe_pci_bdfs()
    for gpu in devices:
        bdf = gpu.name
        if not pci_utils.BDF_STRICT_RE.fullmatch(bdf) or bdf in excluded:
            continue
        if (glt_utils.read_sysfs_value(gpu / "vendor") != "0x10de"
                or not glt_utils.read_sysfs_value(gpu / "class").startswith("0x03")):
            continue
        if bdf in unsafe:
            logger.info("Not selecting GPU %s: used by management or mounted storage", bdf)
            continue
        reason = _unused_gpu_reason(bdf)
        if reason:
            logger.info("Not selecting GPU %s: %s", bdf, reason)
            continue
        if glt_utils.read_sysfs_value(gpu / "driver_override") not in ("", "(null)"):
            logger.info("Not selecting GPU %s: a driver override is already staged", bdf)
            continue
        if _isolated_gpu_group(bdf, allow_absent=True):
            return bdf
        logger.info("Not selecting GPU %s: its IOMMU group is shared", bdf)
    return None


def _module_load_reason(owned_bdf):
    """Reject module loading that could bind a PCI assignment we do not own."""
    for device in Path('/sys/bus/pci/devices').iterdir():
        try:
            override = glt_utils.read_sysfs_value(device / 'driver_override')
        except FileNotFoundError:
            continue
        if override == GPU_DRIVER and device.name != owned_bdf:
            return f"Driver loading could bind an unowned PCI assignment for {device.name}"
    return None


def _prepare_device(  # pylint: disable=too-many-branches,too-many-return-statements
        bdf, state, logger, automatic=False):
    """Return (unavailable reason, device-specific) or None after owned setup."""
    if pci_utils.get_pci_driver(bdf) == GPU_DRIVER:
        return (f"GPU {bdf} was acquired during discovery", True) if automatic else None
    reason = _unused_gpu_reason(bdf)
    if reason:
        return reason, True
    gpu = Path("/sys/bus/pci/devices") / bdf
    override = glt_utils.read_sysfs_value(gpu / "driver_override")
    power = glt_utils.read_sysfs_value(gpu / "power/control")
    if automatic and (override not in ('', '(null)')
                      or not _isolated_gpu_group(bdf, allow_absent=True)):
        return f"GPU {bdf} is no longer available for automatic selection", True
    if not (Path("/sys/bus/pci/drivers") / GPU_DRIVER).exists():
        reason = _module_load_reason(bdf if not automatic and override == GPU_DRIVER else None)
        if reason:
            return reason, False
        if not automatic:
            state.update(bdf=bdf, override=override, power=power)
        try:
            glt_utils.run_command(["modprobe", GPU_DRIVER],
                                  "Grace GPU VFIO driver unavailable", timeout=30)
        except RuntimeError:
            return "Grace GPU VFIO driver unavailable", False
    # Recheck after module loading, which may have bound a matching override.
    driver = pci_utils.get_pci_driver(bdf)
    if driver is not None and override != GPU_DRIVER:
        state.clear()
        return f"GPU {bdf} acquired an owner during setup", True
    if automatic and glt_utils.read_sysfs_value(gpu / "driver_override") != override:
        return f"GPU {bdf} acquired a driver override during setup", True
    state.update(bdf=bdf, override=override, power=power)
    if driver is None:
        glt_utils.write_sysfs_value(gpu / "driver_override", GPU_DRIVER)
        if glt_utils.read_sysfs_value(gpu / "driver_override") != GPU_DRIVER:
            raise OSError("GPU driver override did not take effect")
        glt_utils.write_sysfs_value(Path("/sys/bus/pci/drivers") / GPU_DRIVER / "bind", bdf)
    if pci_utils.get_pci_driver(bdf) != GPU_DRIVER:
        raise OSError(f"Selected GPU {bdf} did not bind to {GPU_DRIVER}")
    if automatic and not _isolated_gpu_group(bdf):
        return f"GPU {bdf} does not have an isolated VFIO group after binding", True
    logger.info("Bound unused GPU %s to %s", bdf, GPU_DRIVER)
    return None


def _restore_device(state, logger):
    """Restore owned binding/override/power state; retain failed work for post hooks."""
    if not state:
        return None
    bdf = state["bdf"]
    gpu = Path("/sys/bus/pci/devices") / bdf
    try:
        driver = pci_utils.get_pci_driver(bdf)
        if driver == GPU_DRIVER:
            glt_utils.write_sysfs_value(Path("/sys/bus/pci/drivers") / GPU_DRIVER / "unbind", bdf)
        elif driver is not None:
            raise OSError(f"GPU ownership changed to {driver}")
        if pci_utils.get_pci_driver(bdf) is not None:
            raise OSError("GPU remains bound after cleanup")
    except OSError as error:
        return f"Cannot restore selected GPU {bdf}: {error}"
    override = "" if state["override"] == "(null)" else state["override"]
    errors = []
    for name, value in (("driver_override", override), ("power/control", state["power"])):
        try:
            actual = glt_utils.read_sysfs_value(gpu / name)
            if name == "driver_override" and actual == "(null)":
                actual = ""
            if actual == value:
                continue
            glt_utils.write_sysfs_value(gpu / name, value)
            actual = glt_utils.read_sysfs_value(gpu / name)
            if name == "driver_override" and actual == "(null)":
                actual = ""
            if actual != value:
                raise OSError("readback differs from the original value")
        except OSError as error:
            errors.append(f"{name}: {error}")
    if errors:
        return f"Cannot restore selected GPU {bdf}: {'; '.join(errors)}"
    state.clear()
    logger.info("Restored GPU %s to its unbound state", bdf)
    return None


def _cleanup_device(_, attr, logger):
    """Retry the same owned cleanup through the framework's outcome hooks."""
    error = _restore_device(getattr(attr, "pfnmap_gpu_state", None), logger)
    if error:
        logger.error("%s; keep the GPU reserved for recovery", error)


def _find_device(bdf):
    """Return the selected bound device's group node, or None if absent.

    Shared driver-inspection errors and non-absence group/node errors propagate.
    """
    if pci_utils.get_pci_driver(bdf) != GPU_DRIVER:
        return None
    group_link = Path("/sys/bus/pci/devices") / bdf / "iommu_group"
    try:
        group = group_link.resolve(strict=True).name
    except FileNotFoundError:
        return None
    if not group.isdecimal():
        return None
    for node in (Path("/dev/vfio") / group,
                 Path("/dev/vfio") / f"noiommu-{group}"):
        try:
            node.stat()
        except FileNotFoundError:
            continue
        return node
    return None


def _log_probe_output(logger, stdout, stderr):
    """Retain readable output from completed probes and interrupted runs."""
    for stream, output in (("stdout", stdout), ("stderr", stderr)):
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        logger.info("PFNMAP probe %s:\n%s", stream, (output or "").rstrip())


def _run_probe(sd, device, node, logger):
    """Build into a session-local directory and execute the bounded probe."""
    with tempfile.TemporaryDirectory(prefix="pfnmap-", dir=sd.log_dir) as temp:
        executable = Path(temp) / "cacheable_pfnmap_probe"
        build = glt_utils.run_command(
            ["make", "-C", str(PROBE_DIR), f"OUTPUT={executable}",
             f"KVM_MICROVM_DIR={KVM_DIR}"],
            "Cacheable PFNMAP probe build failed", timeout=60,
        )
        logger.debug("Probe build: %s", build.stdout)
        return subprocess.run(
            [str(executable), device, str(node)], check=False,
            capture_output=True, text=True, timeout=90,
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_kvm_cacheable_pfnmap(sd, logger):  # pylint: disable=too-many-branches
    """Execute a tiny guest from cacheable VFIO device memory.

    PATCH: https://git.kernel.org/torvalds/c/8cc9dc1ae4fb
    PATCH: https://git.kernel.org/torvalds/c/216887f79d98
    PATCH: https://git.kernel.org/torvalds/c/2a8dfab26677
    PATCH: https://git.kernel.org/torvalds/c/0c67288e0c8b
    PATCH: https://git.kernel.org/torvalds/c/f55ce5a6cd33

    An unset or empty GLT_KVM_CACHEABLE_PFNMAP_BDF selects an unused,
    isolated Grace GPU automatically. To pin an exclusively reserved GPU,
    set it to a full PCI address (domain:bus:device.function), for example
    0008:01:00.0. An explicit choice never falls back to another GPU.

    Requires CPU-accessible USEMEM (BAR4), FEAT_S2FWB and CACHE_DIC. An unused GPU
    is temporarily bound to nvgrace_gpu_vfio_pci; an already bound GPU is left
    as configured. GPUs owned by another driver or used as a display skip.
    VFIO can reset the GPU on open/close. The GPU VFIO module is loaded if
    needed and left loaded; reset methods, module parameters and boot
    configuration are not changed.
    The probe acquires its VFIO group, saves/restores eight scratch bytes, and requires
    guest execution to produce the expected MMIO store. It also checks the
    VM-scoped capability ABI. It restores its own binding, override and power
    policy changes and never enables no-IOMMU.
    An outer probe timeout leaves BAR4 scratch restoration unverified; keep
    that GPU reserved for recovery even if its PCI state is restored.

    The first two PATCH links are preparatory series context. This covers
    the positive cacheable-PFNMAP path and ABI, rather than every
    partial-backport permutation or the rejection path on unsupported CPUs.
    """
    if platform.machine().lower() not in ("aarch64", "arm64"):
        GLTCore.GLTTestResult.skip("Requires Arm64 KVM")
    requested = os.environ.get(BDF_ENV, "").strip().lower()
    if requested and not pci_utils.BDF_STRICT_RE.fullmatch(requested):
        GLTCore.GLTTestResult.skip(
            f"Invalid {BDF_ENV}: use a full PCI address"
        )
    state = {}
    setattr(GLTCore.get_test_attr(test_kvm_cacheable_pfnmap), "pfnmap_gpu_state", state)
    attempted = set()
    result = None
    unavailable = None
    while True:
        try:
            try:
                device = requested or _discover_unused_gpu(logger, attempted)
                if device is None:
                    break
                if not Path("/dev/kvm").exists():
                    GLTCore.GLTTestResult.skip("KVM is unavailable")
                if not os.access("/dev/kvm", os.R_OK | os.W_OK):
                    GLTCore.GLTTestResult.fail("Read/write access to /dev/kvm is required")
                result = None
                attempted.add(device)
                preparation = _prepare_device(device, state, logger, automatic=not requested)
                if preparation is not None:
                    unavailable, device_specific = preparation
                    if requested or not device_specific:
                        GLTCore.GLTTestResult.skip(unavailable)
                    logger.info("GPU %s unavailable during setup: %s", device, unavailable)
                    continue
                node = _find_device(device)
                if node is None:
                    unavailable = f"Selected GPU {device} has no available VFIO group node"
                    if requested:
                        GLTCore.GLTTestResult.skip(unavailable)
                    logger.info("%s", unavailable)
                    continue
                selection = "explicitly nominated" if requested else "automatically selected"
                logger.info("Using %s VFIO GPU %s (%s)", selection, device, node)
                result = _run_probe(sd, device, node, logger)
                _log_probe_output(logger, result.stdout, result.stderr)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                diagnostic = str(error)
                if isinstance(error, subprocess.TimeoutExpired):
                    _log_probe_output(logger, error.stdout, error.stderr)
                    diagnostic = (
                        f"GPU {device}: outer probe timed out after {error.timeout} seconds; "
                        "BAR4 scratch restoration is unverified; "
                        "keep the GPU reserved for recovery even if PCI cleanup succeeds"
                    )
                logger.error("Cacheable PFNMAP setup or probe failed: %s", diagnostic)
                GLTCore.GLTTestResult.fail(
                    f"Cacheable PFNMAP setup or probe error (non-conclusive): {diagnostic}"
                )
        finally:
            cleanup_error = _restore_device(state, logger)
            if cleanup_error:
                GLTCore.GLTTestResult.fail(f"GPU cleanup failed (non-conclusive): {cleanup_error}")
        if requested or result.returncode != 3:
            break
        logger.info("GPU %s lacks probe prerequisites; trying another unused GPU", device)
    if result is None:
        GLTCore.GLTTestResult.skip(unavailable or "No unused, isolated Grace GPU is available")
    if result.returncode < 0:
        GLTCore.GLTTestResult.fail(
            f"Cacheable PFNMAP probe error: terminated by signal {-result.returncode}; "
            "inspect device readiness and kernel fault logs before retrying"
        )
    if result.returncode in (3, 4):
        GLTCore.GLTTestResult.skip(f"PFNMAP prerequisite unavailable: {result.stderr.strip()}")
    if result.returncode == 1:
        GLTCore.GLTTestResult.fail(
            f"Cacheable PFNMAP probe returned {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    if result.returncode != 0:
        GLTCore.GLTTestResult.fail(
            f"Cacheable PFNMAP probe error (exit {result.returncode}, "
            f"non-conclusive): {result.stderr.strip()}"
        )
    logger.info("Guest executed from cacheable PFNMAP memory; capability ABI passed")


GLTCore.get_test_attr(test_kvm_cacheable_pfnmap).post_hooks = GLTCore.PostHooks.create(
    postsuccess=_cleanup_device, postskip=_cleanup_device, postfail=_cleanup_device,
    postafail=_cleanup_device, postwarn=_cleanup_device,
)
