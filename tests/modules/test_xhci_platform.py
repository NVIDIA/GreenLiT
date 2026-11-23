# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test ACPI xHCI platform binding against synthetic sysfs devices."""

import logging

import pytest

from common import glt_core as GLTCore
from src.modules.acpi import glt_test_xhci_platform as xhci


LOGGER = logging.getLogger(__name__)
PNP0D15_MODALIAS = "acpi:TEST0001:PNP0D15:\n"


@pytest.fixture(name="add_controller")
def controller_factory(tmp_path, monkeypatch):
    """Create platform devices, ACPI companions, and driver links on request."""
    platform_bus = tmp_path / "bus" / "platform"
    bus_devices = platform_bus / "devices"
    bus_devices.mkdir(parents=True)
    drivers = platform_bus / "drivers"
    drivers.mkdir()
    monkeypatch.setattr(xhci, "PLATFORM_BUS", platform_bus)

    def add(name, modalias=PNP0D15_MODALIAS, driver=None, override="(null)"):
        device = tmp_path / "devices" / "platform" / name
        device.mkdir(parents=True)
        (bus_devices / name).symlink_to(device, target_is_directory=True)
        if modalias is not None:
            (device / "modalias").write_text(modalias, encoding="ascii")
        if modalias and modalias.startswith("acpi:"):
            firmware_node = tmp_path / "devices" / "ACPI" / name
            firmware_node.mkdir(parents=True)
            (firmware_node / "modalias").write_text(modalias, encoding="ascii")
            (device / "firmware_node").symlink_to(firmware_node, target_is_directory=True)
        if driver is not None:
            driver_path = drivers / driver
            driver_path.mkdir(exist_ok=True)
            (device / "driver").symlink_to(driver_path, target_is_directory=True)
        if override is not None:
            (device / "driver_override").write_text(f"{override}\n", encoding="ascii")
        return device

    return add


def test_matching_controller_bound_to_xhci_passes(add_controller):
    """Recognize PNP0D15 as a compatible ID of a vendor-named controller."""
    add_controller("TEST0001:00", driver="xhci-hcd")

    xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_missing_driver_override_fails_inspection(add_controller):
    """Require an explicit unset driver override before accepting a binding."""
    add_controller("TEST0001:00", driver="xhci-hcd", override=None)

    with pytest.raises(GLTCore.GLTTestFail, match=r"Cannot inspect ACPI platform devices: .*driver_override"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_matching_controller_unbound_fails_with_patch_hint(add_controller):
    """A driver registered for another controller cannot satisfy this one."""
    add_controller("PNP0D10:00", modalias="acpi:PNP0D10:\n", driver="xhci-hcd")
    add_controller("TEST0001:00")

    with pytest.raises(GLTCore.GLTTestFail, match="eb124822186b"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_matching_controller_bound_to_a_different_driver_fails(add_controller):
    """A binding to another driver does not exercise xHCI platform matching."""
    add_controller("TEST0001:00", driver="test-other-driver")

    with pytest.raises(GLTCore.GLTTestFail, match="test-other-driver"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)


@pytest.mark.parametrize("modalias", [
    None,
    "of:NusbT(null)CPNP0D15\n",
    "acpi:PNP0D10:\n",
    "acpi:TEST0001:PNP0D15:PNP0D10:\n",
    "acpi:TEST0001:PNP0D150:\n",
    "acpi:TEST0001:PNP0D15:\nof:NusbT(null)Cgeneric-xhci\n",
])
def test_no_eligible_controller_skips(add_controller, modalias):
    """Require an exact ACPI ID without another firmware match."""
    add_controller("TEST0001:00", modalias=modalias, driver="xhci-hcd")

    with pytest.raises(GLTCore.GLTTestSkip, match="PNP0D15"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_secondary_device_sharing_acpi_companion_is_ignored(add_controller):
    """Only the primary device is matched using the shared companion's ACPI IDs."""
    primary = add_controller("TEST0001:00", driver="xhci-hcd")
    secondary = add_controller("test-secondary.0", modalias="platform:test-secondary\n")
    (secondary / "firmware_node").symlink_to(
        (primary / "firmware_node").resolve(), target_is_directory=True
    )

    xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_functional_but_absent_controller_skips(add_controller):
    """An ACPI node can expose a modalias with just the functional status bit set."""
    controller = add_controller("TEST0001:00")
    (controller / "firmware_node" / "status").write_text("8\n", encoding="ascii")

    with pytest.raises(GLTCore.GLTTestSkip, match="PNP0D15"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_platform_name_match_skips(add_controller):
    """The platform device name can match the driver independently of ACPI."""
    add_controller("xhci-hcd.0", driver="xhci-hcd")

    with pytest.raises(GLTCore.GLTTestSkip, match="PNP0D15"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_forced_binding_skips(add_controller):
    """A driver override can bypass the ACPI match table."""
    add_controller("TEST0001:00", driver="xhci-hcd", override="xhci-hcd")

    with pytest.raises(GLTCore.GLTTestSkip, match="PNP0D15"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)


def test_later_matching_controller_must_also_bind(add_controller):
    """An earlier bound controller cannot hide another controller's failure."""
    add_controller("TEST0001:00", driver="xhci-hcd")
    add_controller("TEST0001:01")

    with pytest.raises(GLTCore.GLTTestFail, match="TEST0001:01"):
        xhci.test_xhci_pnp0d15_binding(None, LOGGER)
