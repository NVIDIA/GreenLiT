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
Python script to test I2C.
"""
# The checkpoint RuntimeError-to-GLT-failure path intentionally mirrors the
# CXL cleanup-safe path so each decorated test can run its registered hooks.
# pylint: disable=duplicate-code
import os
import platform
import subprocess
import stat
import time
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


_I2C_ACPI_RUNTIME_PM_BUG_HEADER = "BUG: sleeping function called from invalid context"
_I2C_ACPI_RUNTIME_PM_COMMAND_TIMEOUT = 10
_I2C_ACPI_RUNTIME_PM_IPMI_TRIGGER_ATTEMPTS = 5
_I2C_ACPI_RUNTIME_PM_QUIET_INTERVAL_SECONDS = 2
_IPMI_TEGRA_TOPOLOGY_ERROR = "error"
_IPMI_TEGRA_TOPOLOGY_MATCH = "match"
_IPMI_TEGRA_TOPOLOGY_NO_MATCH = "no_match"


def _tegra_i2c_devices(prefix: str = "NVDA2017:",
                       drv: Path = Path("/sys/bus/platform/drivers/tegra-i2c"),
                       strict_errors: bool = False) -> list[str]:
    """
    Return sorted tegra-i2c platform devices whose names start with prefix.
    """
    try:
        entries = list(drv.iterdir())
    except FileNotFoundError:
        return []
    except OSError:
        if strict_errors:
            raise
        return []

    devices = []
    for entry in entries:
        if not entry.name.startswith(prefix):
            continue
        try:
            if not stat.S_ISLNK(entry.lstat().st_mode):
                continue
            entry.stat()
        except FileNotFoundError:
            continue
        except OSError:
            if strict_errors:
                raise
            continue
        devices.append(entry.name)
    return sorted(devices)


def _tegra_i2c_kernel_error_messages(dev_name: str) -> list[str]:
    """Return kernel log fragments that indicate Tegra I2C probe/transfer failure."""
    return [
        f"tegra-i2c {dev_name}: probe with driver tegra-i2c failed",
        f"tegra-i2c {dev_name}: config load timed out",
        f"tegra-i2c {dev_name}: i2c transfer timed out",
        f"tegra-i2c {dev_name}: rx fifo flush timed out",
        f"tegra-i2c {dev_name}: tx fifo flush timed out",
    ]


def _dev_unbind_bind(dev_name: str) -> tuple[bool, str]:
    """Unbind then rebind dev_name on the tegra-i2c platform driver.
    Returns (True, "") on success, or (False, reason) on any OSError.
    """
    drv = Path("/sys/bus/platform/drivers/tegra-i2c")
    unbind = drv / "unbind"
    bind = drv / "bind"
    try:
        with open(unbind, "w", encoding="ascii") as fh:
            fh.write(dev_name)
    except OSError as exc:
        return (False, f"write {unbind}: {exc}")
    try:
        with open(bind, "w", encoding="ascii") as fh:
            fh.write(dev_name)
    except OSError as exc:
        return (False, f"write {bind}: {exc}")

    time.sleep(0.5)
    return (True, "")


def _has_tegra_i2c_acpi_irq_safe_bug(log: str) -> bool:
    """Return true when log shows Tegra I2C sleeping in ACPI runtime resume."""
    required_markers = (
        "acpi_subsys_runtime_resume",
        "rpm_resume",
        "tegra_i2c_xfer",
    )
    for block in log.split(_I2C_ACPI_RUNTIME_PM_BUG_HEADER)[1:]:
        has_mutex_frame = "mutex_lock" in block
        if has_mutex_frame and all(marker in block for marker in required_markers):
            return True

    return False


def _tegra_i2c_parent_from_ipmi_devpath(
    udev_output: str, tegra_i2c_devices: set[str]
) -> str | None:
    """Return the Tegra I2C parent of /dev/ipmi0 when the topology matches."""
    devpaths = []
    for line in udev_output.splitlines():
        if line.startswith("P: "):
            devpaths.append(line.removeprefix("P: ").strip())
        elif line.startswith("E: DEVPATH="):
            devpaths.append(line.removeprefix("E: DEVPATH=").strip())

    for devpath in devpaths:
        if "IPI0001" not in devpath:
            continue
        path_parts = set(Path(devpath).parts)
        for dev_name in tegra_i2c_devices:
            if dev_name in path_parts:
                return dev_name

    return None


def _ipmi_over_tegra_i2c_acpi_topology(logger) -> tuple[str, str, Path | None]:
    """Inspect whether /dev/ipmi0 is connected through ACPI Tegra I2C."""
    if not Path("/dev/ipmi0").exists():
        return (_IPMI_TEGRA_TOPOLOGY_NO_MATCH, "/dev/ipmi0 is not present.", None)

    try:
        # Use shell=False to prevent command injection (CWE-78)
        result = subprocess.run(
            ["udevadm", "info", "-n", "/dev/ipmi0"],
            capture_output=True, shell=False, text=True, check=False,
            timeout=_I2C_ACPI_RUNTIME_PM_COMMAND_TIMEOUT
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired) as exc:
        logger.error(f"udevadm failed for /dev/ipmi0: {exc}")
        return (_IPMI_TEGRA_TOPOLOGY_ERROR, f"udevadm failed for /dev/ipmi0: {exc}", None)

    if result.returncode != 0:
        logger.error(result.stdout)
        logger.error(result.stderr)
        return (
            _IPMI_TEGRA_TOPOLOGY_ERROR,
            "udevadm failed while inspecting /dev/ipmi0.",
            None,
        )

    try:
        tegra_i2c_devices = set(_tegra_i2c_devices(prefix="NVDA", strict_errors=True))
    except OSError as exc:
        reason = f"Cannot inspect tegra-i2c devices while validating /dev/ipmi0 topology: {exc}"
        logger.error(reason)
        return (_IPMI_TEGRA_TOPOLOGY_ERROR, reason, None)

    parent = _tegra_i2c_parent_from_ipmi_devpath(result.stdout, tegra_i2c_devices)
    if parent is None:
        return (
            _IPMI_TEGRA_TOPOLOGY_NO_MATCH,
            "/dev/ipmi0 is not under IPI0001 on a bound ACPI Tegra I2C device.",
            None,
        )

    return (_IPMI_TEGRA_TOPOLOGY_MATCH, parent, Path("/sys/devices/platform") / parent)


def _tegra_i2c_runtime_status(tegra_i2c_device_path: Path) -> str | None:
    """Return the runtime PM status for a Tegra I2C platform device."""
    try:
        return (tegra_i2c_device_path / "power" / "runtime_status").read_text(
            encoding="utf-8"
        ).strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _run_ipmitool_mc_info(logger) -> tuple[bool, str]:
    """Run the IPMI command that exercises the SSIF I2C path."""
    try:
        # Use shell=False to prevent command injection (CWE-78)
        result = subprocess.run(
            ["ipmitool", "mc", "info"], capture_output=True,
            shell=False, text=True, check=False,
            timeout=_I2C_ACPI_RUNTIME_PM_COMMAND_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.error(f"ipmitool mc info failed: {exc}")
        return (False, f"ipmitool mc info failed; cannot exercise Tegra I2C: {exc}")
    if result.returncode != 0:
        logger.error(result.stdout)
        logger.error(result.stderr)
        return (False, "ipmitool mc info failed; cannot exercise Tegra I2C.")
    if "Device Available          : yes" not in result.stdout:
        logger.error(result.stdout)
        return (False, "Invalid IPMI device state detected by ipmitool.")

    return (True, "")


def is_NVDA0301_present(logger): # pylint: disable=invalid-name
    """
    Check if could find i2c device NVDA0301 from stdout of
    'udevadm info -n /dev/ipmi0'
    """
    # Use shell=False to prevent command injection (CWE-78)
    result = subprocess.run(["udevadm", "info", "-n", "/dev/ipmi0"],
                            capture_output=True, shell=False, text=True, check=False)
    if result.returncode != 0:
        logger.error(result.stdout)
        return False

    return "NVDA0301" in result.stdout and "IPI0001" in result.stdout


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_i2c_and_ipmi_device(_, logger):
    """Test if could detect i2c and ipmi device.

    kernel patch:
    https://git.kernel.org/torvalds/c/5bf71889ad9a4d39b7665c105a005c5a33d730ba
    """

    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("System is not using a Grace CPU.")

    if not is_NVDA0301_present(logger):
        GLTCore.GLTTestResult.fail("NVDA0301 or IPI0001 device not found in /dev/ipmi0 device node. " +
                                   "Include kernel commit 5bf71889ad9 to fix it.")

    # Check if could find correct ipmi device state by ipmitool
    if glt_utils.is_cli_available("ipmitool"):
        # Use shell=False to prevent command injection (CWE-78)
        result = subprocess.run(["ipmitool", "mc", "info"],
                                capture_output=True, shell=False, text=True, check=False)
        if "Device Available          : yes" not in result.stdout:
            logger.error(result.stdout)
            GLTCore.GLTTestResult.fail("Invalid IPMI device state detected by ipmitool. "
                                       "Include kernel commit 5bf71889ad9 to fix it")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_ipmi_i2c_pec(_, logger):
    """
    Test if I2C PEC is working.

    JIRA: DGX-8249
    PATCH: https://git.kernel.org/torvalds/c/9f855779a387 v6.4+
    """

    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("System is not using a Grace CPU.")

    if not is_NVDA0301_present(logger):
        GLTCore.GLTTestResult.fail("NVDA0301 or IPI0001 device not found in /dev/ipmi0 device node. "
                                   "Include kernel commit 5bf71889ad9 to fix it.")

    if not glt_utils.is_cli_available("ipmitool"):
        GLTCore.GLTTestResult.fail("Please install 'ipmitool' to perform the test.")

    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.join(module_dir, "kretprobes")
    with glt_utils.chdir_managed(module_dir):
        try:
            # Use shell=False to prevent command injection (CWE-78)
            subprocess.run(["make"], check=True, shell=False)
            if not glt_utils.sign_kernel_module("kretprobe.ko", logger):
                GLTCore.GLTTestResult.fail(
                    "Cannot sign kretprobe.ko for Secure Boot. "
                    "See docs/user-guides/secure-boot.md for setup.")
            subprocess.run(["insmod", "kretprobe.ko"], check=True, shell=False)
            subprocess.run(["ipmitool", "mc", "info"],
                           capture_output=True, shell=False, text=True, check=False)
            result = subprocess.run(["cat", "/proc/greenlit"],
                                    capture_output=True, shell=False, text=True, check=False)
            subprocess.run(["rmmod", "kretprobe"], check=True, shell=False)
            subprocess.run(["make", "clean"], check=True, shell=False)
            if result.returncode != 0 or "0" in result.stdout:
                GLTCore.GLTTestResult.fail("IPMI PEC validation failed. "
                                           "Include kernel commit 9f855779a387 to fix it.")

        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(f"Error executing command: {e}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra_i2c_acpi_runtime_pm_irq_safe(_, logger):  # pylint: disable=too-many-branches
    """
    Verify ACPI-backed Tegra I2C runtime PM does not sleep while atomic
    during IPMI-triggered runtime resume.

    PATCH: https://git.kernel.org/torvalds/c/14d069d92951 v6.11+
    """

    if not Path("/sys/firmware/acpi").is_dir():
        GLTCore.GLTTestResult.skip(
            "System is not ACPI-booted; this test only runs on ACPI systems.")

    topology_status, topology_reason, tegra_i2c_path = _ipmi_over_tegra_i2c_acpi_topology(
        logger
    )
    if topology_status == _IPMI_TEGRA_TOPOLOGY_ERROR:
        GLTCore.GLTTestResult.fail(topology_reason)
    if topology_status == _IPMI_TEGRA_TOPOLOGY_NO_MATCH:
        GLTCore.GLTTestResult.skip(topology_reason)

    if not glt_utils.is_cli_available("ipmitool"):
        GLTCore.GLTTestResult.fail("Please install 'ipmitool' to perform the test.")

    config_path = Path(f"/boot/config-{platform.uname().release}")
    try:
        kernel_config = check_kernel_config.parse_kconfig(str(config_path))
    except (FileNotFoundError, PermissionError, OSError):
        kernel_config = None

    try:
        checkpoint = glt_utils.capture_kernel_log_checkpoint()
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            "Failed to capture kernel log checkpoint before IPMI trigger: "
            f"{error}"
        )

    try:
        baseline_log = glt_utils.get_kernel_log()
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            "Cannot read kernel log; cannot check for sleeping-while-atomic "
            f"during Tegra I2C ACPI runtime resume: {error}"
        )
    if not baseline_log:
        GLTCore.GLTTestResult.fail(
            "Cannot read kernel log; cannot check for sleeping-while-atomic "
            "during Tegra I2C ACPI runtime resume.")
    if _has_tegra_i2c_acpi_irq_safe_bug(baseline_log):
        GLTCore.GLTTestResult.fail(
            "Kernel log already contains sleeping-while-atomic BUG with "
            "mutex_lock during acpi_subsys_runtime_resume/tegra_i2c_xfer.")

    saw_suspended_at_trigger = False
    runtime_status_at_trigger = None
    for _ in range(_I2C_ACPI_RUNTIME_PM_IPMI_TRIGGER_ATTEMPTS):
        time.sleep(_I2C_ACPI_RUNTIME_PM_QUIET_INTERVAL_SECONDS)
        runtime_status_at_trigger = _tegra_i2c_runtime_status(tegra_i2c_path)
        if runtime_status_at_trigger == "suspended":
            saw_suspended_at_trigger = True
        ok, reason = _run_ipmitool_mc_info(logger)
        if not ok:
            GLTCore.GLTTestResult.fail(reason)

    bug_reason = (
        "Found sleeping-while-atomic BUG in kernel log during Tegra I2C "
        "ACPI runtime resume (mutex_lock from tegra_i2c_xfer via "
        "acpi_subsys_runtime_resume). "
        "Backport kernel commit 14d069d92951 to fix it.")

    try:
        new_log = glt_utils.get_kernel_log(checkpoint)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            "Cannot read post-trigger kernel log; cannot check for "
            f"sleeping-while-atomic during Tegra I2C ACPI runtime resume: {error}"
        )

    if _has_tegra_i2c_acpi_irq_safe_bug(new_log):
        GLTCore.GLTTestResult.fail(bug_reason)

    if not saw_suspended_at_trigger:
        GLTCore.GLTTestResult.fail(
            f"Tegra I2C runtime_status at trigger was {runtime_status_at_trigger}; "
            "runtime resume was not confirmed, so absence of "
            "sleeping-while-atomic BUG is non-conclusive.")

    if kernel_config is None:
        GLTCore.GLTTestResult.skip(
            "Cannot read kernel config; ipmitool exercised IPMI-over-Tegra-I2C, "
            "but absence of sleeping-while-atomic BUG is non-conclusive.")
    else:
        missing_configs = [
            config
            for config in ("CONFIG_DEBUG_ATOMIC_SLEEP", "CONFIG_KALLSYMS")
            if kernel_config.get(config) != "y"
        ]
        if missing_configs:
            GLTCore.GLTTestResult.skip(
                f"{', '.join(missing_configs)} is not enabled; ipmitool exercised "
                "IPMI-over-Tegra-I2C, but absence of sleeping-while-atomic BUG "
                "is non-conclusive.")


def _restore_tegra_i2c_binding(_scheduler, attr, logger):
    dev_names = getattr(attr, "rebind_devices", None)
    if dev_names is None:
        dev_name = getattr(attr, "picked_device", None)
        dev_names = () if dev_name is None else (dev_name,)
    elif isinstance(dev_names, str):
        dev_names = (dev_names,)

    for dev_name in dev_names:
        try:
            with open("/sys/bus/platform/drivers/tegra-i2c/bind", "w", encoding="ascii") as fh:
                fh.write(dev_name)
        except OSError as e:
            logger.debug(f"post-hook: bind {dev_name}: {e}")

    if getattr(attr, "dyndbg_enabled", False):
        try:
            with open("/sys/kernel/debug/dynamic_debug/control",
                      "w", encoding="ascii") as fh:
                fh.write("module i2c_tegra -p")
        except OSError as e:
            logger.debug(f"post-hook: dyndbg restore: {e}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra_i2c_dma_pio_fallback(_, __):
    """
    Verify kernel commit b53232fd220a takes effect on tegra-i2c device.

    JIRA: DGX-16326
    PATCH: https://lore.kernel.org/linux-tegra/20251118140620.549-2-akhilrajeev@nvidia.com/
    """

    test_tegra_i2c_dma_pio_fallback.attr.picked_device = None
    test_tegra_i2c_dma_pio_fallback.attr.dyndbg_enabled = False

    if not Path("/sys/firmware/acpi").is_dir():
        GLTCore.GLTTestResult.skip(
            "System is not ACPI-booted; this test only runs on ACPI-booted systems.")

    try:
        devices = _tegra_i2c_devices(strict_errors=True)
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to inspect tegra-i2c devices before rebind: {error}"
        )
    if not devices:
        GLTCore.GLTTestResult.skip("No tegra-i2c device bound to the platform driver.")
    dev_name = devices[0]

    test_tegra_i2c_dma_pio_fallback.attr.picked_device = dev_name

    # Enable dev_dbg() output for the i2c_tegra module so the fix-specific
    # "DMA not available, falling back to PIO" line reaches the kernel log.
    try:
        with open("/sys/kernel/debug/dynamic_debug/control", "w", encoding="ascii") as fh:
            fh.write("module i2c_tegra +p")
    except OSError:
        GLTCore.GLTTestResult.skip(
            "Could not enable dynamic debug for module i2c_tegra")

    test_tegra_i2c_dma_pio_fallback.attr.dyndbg_enabled = True

    try:
        checkpoint = glt_utils.capture_kernel_log_checkpoint()
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to capture kernel log checkpoint before I2C rebind: {error}"
        )

    ok, reason = _dev_unbind_bind(dev_name)
    if not ok:
        GLTCore.GLTTestResult.skip(f"unbind/bind cycle on {dev_name} failed: {reason}")

    message = "DMA not available, falling back to PIO"
    try:
        found = glt_utils.check_kernel_log(message, checkpoint=checkpoint)
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve kernel log after rebind: {error}"
        )
    if not found:
        GLTCore.GLTTestResult.fail(
            f"missing '{message}' in kernel log for {dev_name} after rebind, "
            "consider backporting kernel commit b53232fd220a"
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra410_i2c_device(_, logger):
    """
    Verify Tegra410 I2C controllers bind and survive driver rebind on Vera.

    JIRA: DGX-16337
    PATCH: https://git.kernel.org/torvalds/c/4eeb19aaff55
    PATCH: https://git.kernel.org/torvalds/c/0c0e440b0c93
    PATCH: https://git.kernel.org/torvalds/c/59717f260183
    """

    test_tegra410_i2c_device.attr.rebind_devices = None

    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    if not Path("/sys/firmware/acpi").is_dir():
        GLTCore.GLTTestResult.skip(
            "System is not ACPI-booted; Tegra410 I2C ACPI coverage is not applicable.")

    try:
        devices = _tegra_i2c_devices(strict_errors=True)
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to inspect Tegra410 I2C devices before rebind: {error}"
        )
    if not devices:
        GLTCore.GLTTestResult.fail(
            "No NVDA2017 Tegra410 I2C device is bound to tegra-i2c. "
            "Include kernel commit 59717f260183 to add Tegra410 support.")
    test_tegra410_i2c_device.attr.rebind_devices = tuple(devices)

    logger.info("Detected Tegra410 I2C platform devices: %s", ", ".join(devices))

    kernel_errors = []

    for dev_name in devices:
        try:
            checkpoint = glt_utils.capture_kernel_log_checkpoint()
        except RuntimeError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to capture kernel log checkpoint before rebinding "
                f"{dev_name}: {error}"
            )

        ok, reason = _dev_unbind_bind(dev_name)
        if not ok:
            GLTCore.GLTTestResult.fail(
                f"unbind/bind cycle on {dev_name} failed: {reason}. "
                "Include kernel commits 0c0e440b0c93 and 59717f260183 to fix Tegra410 "
                "I2C support.")

        try:
            rebound_devices = _tegra_i2c_devices(strict_errors=True)
        except OSError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to inspect tegra-i2c devices after rebinding "
                f"{dev_name}: {error}"
            )

        if dev_name not in rebound_devices:
            GLTCore.GLTTestResult.fail(
                f"{dev_name} is not bound to tegra-i2c after rebind. "
                "Include kernel commit 59717f260183 to add Tegra410 support.")

        try:
            kernel_log = glt_utils.get_kernel_log(checkpoint)
        except RuntimeError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to retrieve kernel log after rebinding {dev_name}: {error}"
            )

        kernel_errors.extend(
            message
            for message in _tegra_i2c_kernel_error_messages(dev_name)
            if message in kernel_log
        )

    if kernel_errors:
        GLTCore.GLTTestResult.fail(
            "Found Tegra410 I2C kernel errors after probing devices: "
            f"{'; '.join(kernel_errors)}. Include kernel commits 0c0e440b0c93 "
            "and 59717f260183 to fix Tegra410 register-offset support.")


test_tegra410_i2c_device.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_restore_tegra_i2c_binding,
    postskip=_restore_tegra_i2c_binding,
    postfail=_restore_tegra_i2c_binding,
    postafail=_restore_tegra_i2c_binding,
)


test_tegra_i2c_dma_pio_fallback.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_restore_tegra_i2c_binding,
    postskip=_restore_tegra_i2c_binding,
    postfail=_restore_tegra_i2c_binding,
    postafail=_restore_tegra_i2c_binding,
)
