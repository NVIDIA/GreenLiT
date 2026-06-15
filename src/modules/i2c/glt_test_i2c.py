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
import os
import subprocess
import time
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev


def _first_eligible_tegra_i2c_device() -> str | None:
    """
    Return the basename of the first NVDA2017:* device bound to the
    tegra-i2c platform driver, or None if none is bound.
    """
    drv = Path("/sys/bus/platform/drivers/tegra-i2c")
    if not drv.is_dir():
        return None

    candidates = [
        entry.name
        for entry in drv.iterdir()
        if entry.name.startswith("NVDA2017:") and entry.is_symlink()
    ]
    if not candidates:
        return None
    return sorted(candidates)[0]


def _tegra_i2c_devices(prefix: str = "NVDA2017:",
                       drv: Path = Path("/sys/bus/platform/drivers/tegra-i2c")) -> list[str]:
    """
    Return sorted tegra-i2c platform devices whose names start with prefix.
    """
    if not drv.is_dir():
        return []

    try:
        entries = list(drv.iterdir())
    except OSError:
        return []

    return sorted(
        entry.name
        for entry in entries
        if entry.name.startswith(prefix) and entry.is_symlink() and entry.exists()
    )


def _tegra_i2c_kernel_error_messages(dev_name: str) -> list[str]:
    """Return kernel log fragments that indicate Tegra I2C probe/transfer failure."""
    return [
        f"tegra-i2c {dev_name}: probe with driver tegra-i2c failed",
        f"tegra-i2c {dev_name}: config load timed out",
        f"tegra-i2c {dev_name}: i2c transfer timed out",
        f"tegra-i2c {dev_name}: rx fifo flush timed out",
        f"tegra-i2c {dev_name}: tx fifo flush timed out",
    ]


def _kernel_log_text() -> str:
    """Return the current boot kernel log text, or an empty string if unavailable."""
    if glt_utils.is_cli_available("journalctl"):
        try:
            result = subprocess.run(["journalctl", "--no-pager", "-b0", "-k"],
                                    capture_output=True, text=True, check=False, timeout=10)
        except subprocess.TimeoutExpired:
            pass
        else:
            if result.returncode == 0:
                return result.stdout

    if glt_utils.is_cli_available("dmesg"):
        try:
            result = subprocess.run(["dmesg"], capture_output=True, text=True, check=False, timeout=10)
        except subprocess.TimeoutExpired:
            return ""
        if result.returncode == 0:
            return result.stdout

    return ""


def _new_tegra_i2c_kernel_errors(dev_name: str, previous_log: str, current_log: str) -> list[str]:
    """Return Tegra I2C errors added after previous_log was captured."""
    if not previous_log:
        current_log = ""
    elif current_log.startswith(previous_log):
        current_log = current_log[len(previous_log):]
    else:
        current_log = ""

    return [
        message
        for message in _tegra_i2c_kernel_error_messages(dev_name)
        if message in current_log
    ]


def _dev_unbind_bind(dev_name: str) -> tuple[bool, str]:
    """Unbind then rebind dev_name on the tegra-i2c platform driver.
    Returns (True, "") on success, or (False, reason) on any OSError so
    the caller can map it to a SKIP verdict.
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
@GLTCore.GLTPlan("sanity")
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
@GLTCore.GLTPlan("sanity")
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
@GLTCore.GLTPlan("sanity")
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

    dev_name = _first_eligible_tegra_i2c_device()
    if dev_name is None:
        GLTCore.GLTTestResult.skip("No tegra-i2c device bound to the platform driver.")

    test_tegra_i2c_dma_pio_fallback.attr.picked_device = dev_name

    # Enable dev_dbg() output for the i2c_tegra module so the new
    # "DMA not available, falling back to PIO" line reaches the kernel log.
    try:
        with open("/sys/kernel/debug/dynamic_debug/control", "w", encoding="ascii") as fh:
            fh.write("module i2c_tegra +p")
    except OSError:
        GLTCore.GLTTestResult.skip(
            "Could not enable dynamic debug for module i2c_tegra")

    test_tegra_i2c_dma_pio_fallback.attr.dyndbg_enabled = True

    ok, reason = _dev_unbind_bind(dev_name)
    if not ok:
        GLTCore.GLTTestResult.skip(f"unbind/bind cycle on {dev_name} failed: {reason}")

    message = "DMA not available, falling back to PIO"
    if not glt_utils.check_kernel_log(message):
        GLTCore.GLTTestResult.fail(
            f"missing '{message}' in kernel log for {dev_name} after rebind, "
            "consider backporting kernel commit b53232fd220a")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
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

    devices = _tegra_i2c_devices()
    if not devices:
        GLTCore.GLTTestResult.fail(
            "No NVDA2017 Tegra410 I2C device is bound to tegra-i2c. "
            "Include kernel commit 59717f260183 to add Tegra410 support.")
    test_tegra410_i2c_device.attr.rebind_devices = tuple(devices)

    logger.info("Detected Tegra410 I2C platform devices: %s", ", ".join(devices))

    kernel_log = _kernel_log_text()
    if not kernel_log:
        GLTCore.GLTTestResult.skip(
            "Kernel log is unavailable; cannot validate Tegra410 I2C probe/transfer errors.")
    kernel_errors = []

    for dev_name in devices:
        ok, reason = _dev_unbind_bind(dev_name)
        if not ok:
            GLTCore.GLTTestResult.fail(
                f"unbind/bind cycle on {dev_name} failed: {reason}. "
                "Include kernel commits 0c0e440b0c93 and 59717f260183 to fix Tegra410 "
                "I2C support.")

        if dev_name not in _tegra_i2c_devices():
            GLTCore.GLTTestResult.fail(
                f"{dev_name} is not bound to tegra-i2c after rebind. "
                "Include kernel commit 59717f260183 to add Tegra410 support.")

        current_log = _kernel_log_text()
        if not current_log:
            GLTCore.GLTTestResult.skip(
                f"Kernel log unavailable after rebinding {dev_name}; cannot validate new errors.")
        if not current_log.startswith(kernel_log):
            GLTCore.GLTTestResult.skip(
                "Kernel log cursor is non-contiguous; cannot reliably diff new Tegra410 I2C errors.")
        kernel_errors.extend(_new_tegra_i2c_kernel_errors(dev_name, kernel_log, current_log))
        kernel_log = current_log

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
