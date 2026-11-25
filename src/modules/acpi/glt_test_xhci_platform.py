# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify ACPI matching for platform xHCI controllers."""

from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils

PLATFORM_BUS = Path("/sys/bus/platform")
XHCI_DRIVER = "xhci-hcd"


def _uses_pnp0d15(device: Path) -> bool:
    """Identify controllers that exercise the PNP0D15 match after boot."""
    # The platform modalias applies the same ACPI companion selection as driver matching.
    try:
        aliases = glt_utils.read_sysfs_value(device / "modalias").splitlines()
    except FileNotFoundError:
        return False

    acpi_ids = next((alias.split(":")[1:] for alias in aliases if alias.startswith("acpi:")), [])
    if "PNP0D15" not in acpi_ids or "PNP0D10" in acpi_ids:
        return False

    # An OF or platform-name match can bind without the PNP0D15 entry.
    if any(alias.startswith("of:") for alias in aliases) or device.name.partition(".")[0] == XHCI_DRIVER:
        return False

    # A functional but absent ACPI node can expose a modalias without matching a driver.
    try:
        status = glt_utils.read_sysfs_value(device / "firmware_node" / "status")
    except FileNotFoundError:  # Without _STA, ACPI treats the device as present.
        status = "1"
    if int(status) & 1 == 0:
        return False

    return glt_utils.read_sysfs_value(device / "driver_override") == "(null)"


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
def test_xhci_pnp0d15_binding(_, logger):
    """Check that ACPI PNP0D15 platform controllers bind to xhci-hcd at boot.

    PATCH: https://github.com/torvalds/linux/commit/eb124822186be8ea433f568e92cc3bfbf6117a30
    """
    try:
        controllers = [device for device in sorted((PLATFORM_BUS / "devices").iterdir()) if _uses_pnp0d15(device)]
    except (OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(f"Cannot inspect ACPI platform devices: {error}")

    if not controllers:
        GLTCore.GLTTestResult.skip("No eligible ACPI PNP0D15 platform controller found")

    expected_driver = PLATFORM_BUS / "drivers" / XHCI_DRIVER
    for device in controllers:
        try:
            bound_driver = (device / "driver").resolve(strict=True)
        except FileNotFoundError:
            bound_driver = None
        except OSError as error:
            GLTCore.GLTTestResult.fail(f"Cannot inspect driver binding for {device.name}: {error}")

        if bound_driver != expected_driver:
            binding = f"bound to {bound_driver}" if bound_driver else "unbound"
            GLTCore.GLTTestResult.fail(
                f"ACPI PNP0D15 platform controller {device.name} is {binding}, expected {XHCI_DRIVER}. "
                f"Check whether {XHCI_DRIVER} is available, whether its probe succeeded, and whether "
                "the running kernel includes the PNP0D15 match added by eb124822186b."
            )
        logger.info("ACPI PNP0D15 platform controller %s binds to %s", device.name, XHCI_DRIVER)
