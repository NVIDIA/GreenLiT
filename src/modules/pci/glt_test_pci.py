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
Python script to test PCI.
"""

import platform
import re
import subprocess
from common import glt_core as GLTCore
from common import glt_utils
from common import pci_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_pci_device_reset_method(sd, logger):
    """
    Test primary reset method of Hopper/NVIDIA GFX device.

    nvbug: 4193665
    DGX-8277
    PATCH: https://git.kernel.org/torvalds/c/6937b7dd4349 v5.15+
    """

    for tests in sd.progress:
        if test_nvidia_pci_device_reset_method.__name__ not in tests:
            continue
        if "MightReset" in tests[test_nvidia_pci_device_reset_method.__name__]:
            sd.progress_delete_time(test_nvidia_pci_device_reset_method.__name__, "MightReset")

            # If timer was activated and GLT gets resumed here, it means
            # the system didn't reset, therefore the test was passed.
            p = subprocess.run("""sudo systemctl show -p LastTriggerUSecMonotonic greenlit.timer | cut -d'=' -f2-""",
                               capture_output=True, shell=True, text=True, check=False)

            if p.returncode == 0 and p.stdout.strip() != "0":
                logger.info("System reset didn't happen")
                glt_utils.systemd_enable(enable=False)
                glt_utils.start_systemd_timer(start=False)
                return

            GLTCore.GLTTestResult.fail("System was reset. " +
                                       "Include kernel commit 05da178ce0aa to fix it.")

    lspci_output = NvDev.get_nvidia_3d_bdf()
    pci_dev_addrs = list(lspci_output.splitlines())

    if not pci_dev_addrs:
        GLTCore.GLTTestResult.skip("SKIP: NVIDIA hopper pcie device ID not found.")

    # Check reset_method for each pci device
    for pci_dev_addr in pci_dev_addrs:
        reset_method_file = f"/sys/bus/pci/devices/{pci_dev_addr}/reset_method"
        with open(reset_method_file, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if not content.startswith("acpi"):
                GLTCore.GLTTestResult.fail(
                    f"ERROR: reset_method '{content}' is unexpected. " +
                    "Expected primary reset method 'acpi'. " +
                    "Include kernel commit 05da178ce0aa to fix it."
                )

    for pci_dev_addr in pci_dev_addrs:
        # pylint: disable=consider-using-with
        subprocess.Popen(f"sudo sh -c 'sleep 5; echo 1 > /sys/bus/pci/devices/{pci_dev_addr}/reset'",
                         start_new_session=True, restore_signals=False, shell=True)

    glt_utils.arm_for_reset(test_nvidia_pci_device_reset_method.__name__, sd)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_pci_secondary_mabort(_, logger):
    """
    Verifies that the PCI secondary status register is clean after PCI re-enumeration.

    DGX-9488
    PATCH: https://git.kernel.org/torvalds/c/7bf9d2af
    """

    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.fail("Please install 'pciutils' to perform the test.")

    # Step 1: Check if PCI/PCIe devices exist on the machine
    device_blocks = []
    devices = []
    empty_bridges = []
    try:
        cmd_output = subprocess.run(
            ["lspci", "-Dv"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if not cmd_output.stdout.strip():
            GLTCore.GLTTestResult.skip("No PCI/PCIe devices found on this machine. PCI enumeration is not valid.")
        device_blocks = cmd_output.stdout.strip().split("\n\n")
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"Failed to enumerate PCI devices with 'lspci': {e}")

    for block in device_blocks:
        lines = block.splitlines()
        if not lines:
            continue

        slot = lines[0].split()[0]
        domain_str, bus_str, _ = slot.split(":")
        domain = int(domain_str, 16)
        bus = int(bus_str, 16)

        is_pci2pci_bridge = any(
            "PCI bridge" in line or "PCI Express Root Port" in line
            for line in lines
        )

        sec_bus = sub_bus = None
        for line in lines:
            m = re.search(
                r"Bus:\s+primary=\w+,\s+secondary=(\w+),\s+subordinate=(\w+)",
                line,
            )
            if m:
                sec_bus = int(m.group(1), 16)
                sub_bus = int(m.group(2), 16)
                break

        devices.append(
            {
                "slot": slot,
                "domain": domain,
                "bus": bus,
                "is_pci2pci_bridge": is_pci2pci_bridge,
                "sec_bus": sec_bus,
                "sub_bus": sub_bus,
            }
        )


    occupied_buses = set()
    for d in devices:
        occupied_buses.add((d["domain"], d["bus"]))

    for d in devices:
        if not d["is_pci2pci_bridge"]:
            continue
        if d["sec_bus"] is None or d["sub_bus"] is None: # Non bus range info, can't reason about children
            continue

        dom = d["domain"]
        primary_bus = d["bus"]
        sec = d["sec_bus"]
        sub = d["sub_bus"]

        has_child = False
        for (odomain, obus) in occupied_buses:
            if odomain != dom:
                continue
            if sec <= obus <= sub and obus != primary_bus:
                has_child = True
                break

        if not has_child:
            empty_bridges.append(d["slot"])

    if len(empty_bridges) != 0:
        logger.warning("Empty bridges found: %s", ", ".join(empty_bridges))

    # Step 2: Re-scan/re-enumerate all PCI/PCIe devices
    try:
        _ = subprocess.run(
            ["sudo", "sh", "-c", "echo 1 > /sys/bus/pci/rescan"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"Failed to re-scan PCI bus: {e}")

    # Step 3: Check for MAbort+ flags in secondary status
    try:
        cmd_output = subprocess.run(
            ["sudo", "lspci", "-vv"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.debug("lspci completed: rc=%s", cmd_output.returncode)

        offending_lines = [
            line for line in cmd_output.stdout.splitlines()
            if "Secondary" in line and "MAbort+" in line
        ]
        if offending_lines:
            fail_msg = (
                "Secondary status register is not clean:"
                + " ".join(offending_lines)
                + " This may indicate that commit 7bf9d2af is missing; consider backporting it."
            )
            if empty_bridges:
                fail_msg += (
                    " Empty bridges "
                    + ", ".join(empty_bridges)
                    + " may have caused the Master Abort."
                )
            logger.error(fail_msg)
            GLTCore.GLTTestResult.fail(fail_msg)
    except subprocess.CalledProcessError as e:
        GLTCore.GLTTestResult.fail(f"Failed to run 'sudo lspci -vv': {e}")


_acs_kexec_conditional_reboot_hook = GLTCore.PostHooks.for_conditional_reboot(
    "TriggerKexec",
    glt_utils.is_kexec_boot,
)


# pylint: disable=too-many-branches
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_pci_acs_configurability(sd, logger):
    """
    Verifies that the ACS configurability is workable and
    Test that the kernel has the patch PCI/ACS config_acs= fix (patch 9cf8a95).

    Finds ACS-capable PCIe devices via sysfs; reboots via kexec with
    pci=config_acs= for up to 3 devices. After reboot: verifies config_acs=
    in cmdline; fail if kernel log contains "PCI: Unknown option `config_acs`"
    or "PCI: Can't parse ACS command line parameter" (suggest backport).
    When CONFIG_DYNAMIC_DEBUG is set, requires "ACS control" and "ACS fw_ctrl"
    messages in kernel log.

    DGX-9408, DGX-12197
    PATCH: https://git.kernel.org/torvalds/c/47c8846, https://git.kernel.org/torvalds/c/9cf8a95
    """
    kconfig_path = "/boot/config-" + platform.release()
    parse_err = "PCI: Can't parse ACS command line parameter"
    acs_control_msg = "ACS control"
    acs_fw_ctrl_msg = "ACS fw_ctrl"

    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.skip("lspci not available; Please install 'pciutils' to perform the test.")

    acs_bdfs = pci_utils.get_bdfs_with_ext_cap(pci_utils.EXT_CAP_ACS)
    if not acs_bdfs:
        GLTCore.GLTTestResult.skip(
            "No PCIe devices with ACS capability found; extended config"
            " unavailable or no ACS-capable devices."
        )

    extra_cmdline = (
        "pci=config_acs=" + ";".join(f"101xxxx@{bdf}" for bdf in acs_bdfs[:3])
        + ' "dyndbg=file drivers/pci/pci.c +p"'
    )

    ok, reason = glt_utils.handle_kexec_flow(
        test_pci_acs_configurability.__name__, sd, logger,
        custom_cmdline=extra_cmdline,
    )
    if not ok:
        GLTCore.GLTTestResult.fail(reason)


    try:
        with open("/proc/cmdline", "r", encoding="utf-8") as f:
            cmdline = f.read()
            if "config_acs=" not in cmdline:
                GLTCore.GLTTestResult.fail("config_acs= not found in kernel cmdline after kexec.")
    except (FileNotFoundError, PermissionError) as e:
        GLTCore.GLTTestResult.fail(f"Cannot read kernel cmdline: {e}")

    # Check if the kernel supports ACS configurability
    if glt_utils.check_kernel_log("PCI: Unknown option `config_acs"):
        GLTCore.GLTTestResult.fail(
            "PCI ACS configurability is not supported in the kernel." +
            " Consider backporting commit 47c8846 to fix it."
        )

    # Check if the kernel has config_acs fixed
    if glt_utils.check_kernel_log(parse_err):
        GLTCore.GLTTestResult.fail(
            f"Kernel log contains '{parse_err}'. "
            "Backport patch 9cf8a95 (PCI/ACS: Fix 'pci=config_acs=' parameter)."
        )

    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
        if kconfig.get("CONFIG_DYNAMIC_DEBUG") not in ("y", "m"):
            logger.warning(
                "CONFIG_DYNAMIC_DEBUG is not enabled in kernel config." +
                " Consider enabling dynamic debug to get the full testing coverage.")
            return
    except OSError:
        logger.error(f"Cannot read kernel config: {kconfig_path}")
        return

    has_acs_control = glt_utils.check_kernel_log(acs_control_msg)
    has_acs_fw_ctrl = glt_utils.check_kernel_log(acs_fw_ctrl_msg)

    missing = []
    if not has_acs_control:
        missing.append(acs_control_msg)
    if not has_acs_fw_ctrl:
        missing.append(acs_fw_ctrl_msg)

    if missing:
        GLTCore.GLTTestResult.fail(
            f"CONFIG_DYNAMIC_DEBUG is set but '{', '.join(missing)}' "
            "missing from kernel log; kernel may lack patch 9cf8a95 or dyndbg "
            "did not enable drivers/pci/pci.c."
        )


test_pci_acs_configurability.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_acs_kexec_conditional_reboot_hook,
    postfail=_acs_kexec_conditional_reboot_hook,
    postafail=_acs_kexec_conditional_reboot_hook,
)
