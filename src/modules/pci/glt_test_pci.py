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

import os
import platform
import re
import subprocess
from common import glt_core as GLTCore
from common import glt_utils
from common import pci_utils
from common.nv_device import NvDev
from scripts import check_kernel_config

ASPM_POLICY_PATH = "/sys/module/pcie_aspm/parameters/policy"
ASPM_PROBE_PROC_PATH = "/proc/greenlit_pci_aspm_saved_state"
ASPM_SAVED_STATE_MODULE = "aspm_saved_state_probe"
ASPM_REQUIRED_CONFIGS = ("CONFIG_PCIEASPM", "CONFIG_KPROBES", "CONFIG_KRETPROBES")


def _parse_aspm_policy_content(content):
    """Return (available policies, current policy) from the ASPM policy file."""
    policies = []
    current_policy = None

    for token in content.split():
        is_current = token.startswith("[") and token.endswith("]")
        policy = token.strip("[]")
        policies.append(policy)
        if is_current:
            current_policy = policy

    return policies, current_policy


def _candidate_aspm_policies(policies, current_policy):
    """Return deterministic ASPM policy candidates that differ from current."""
    candidates = []
    for policy in ("performance", "powersave", "powersupersave", "default"):
        if policy in policies and policy != current_policy and policy not in candidates:
            candidates.append(policy)
    for policy in policies:
        if policy != current_policy and policy not in candidates:
            candidates.append(policy)
    return candidates


def _check_aspm_saved_state_kernel_config():
    """Return missing kernel configs required by the ASPM saved-state test."""
    kconfig_path = "/boot/config-" + platform.uname().release
    kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    return [
        config for config in ASPM_REQUIRED_CONFIGS
        if kconfig.get(config) not in ("y", "m")
    ]


def _cleanup_aspm_saved_state_probe(module_loaded, policy_changed, original_policy, logger):
    """Restore ASPM policy and unload the saved-state probe module."""
    if policy_changed:
        try:
            _write_aspm_policy(original_policy)
        except OSError as e:
            logger.error(
                "Failed to restore PCIe ASPM policy %s: %s",
                original_policy,
                e,
            )
    if module_loaded:
        result = subprocess.run(
            ["rmmod", ASPM_SAVED_STATE_MODULE],
            capture_output=True,
            check=False,
            shell=False,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            logger.error("Failed to unload %s: %s", ASPM_SAVED_STATE_MODULE, detail)
    if glt_utils.is_cli_available("make"):
        subprocess.run(["make", "clean"], check=False, shell=False)


def _read_aspm_probe_hits(proc_path=ASPM_PROBE_PROC_PATH):
    """Read ASPM config and saved-state probe counters."""
    values = {}
    with open(proc_path, "r", encoding="utf-8") as fh:
        for line in fh:
            key, value = line.strip().split("=", maxsplit=1)
            values[key] = int(value)
    return values


def _read_aspm_policy(policy_path=ASPM_POLICY_PATH):
    """Read the ASPM policy sysfs file."""
    with open(policy_path, "r", encoding="utf-8") as fh:
        return _parse_aspm_policy_content(fh.read())


def _write_aspm_policy(policy, policy_path=ASPM_POLICY_PATH):
    """Write the ASPM policy sysfs file."""
    with open(policy_path, "w", encoding="utf-8") as fh:
        fh.write(policy)
    _, current_policy = _read_aspm_policy(policy_path)
    if current_policy != policy:
        raise OSError(
            f"PCIe ASPM policy write to {policy} did not take effect; "
            f"current policy is {current_policy or 'unknown'}."
        )


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
            p = subprocess.run("""systemctl show -p LastTriggerUSecMonotonic greenlit.timer | cut -d'=' -f2-""",
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
        subprocess.Popen(f"sh -c 'sleep 5; echo 1 > /sys/bus/pci/devices/{pci_dev_addr}/reset'",
                         start_new_session=True, restore_signals=False, shell=True)

    glt_utils.arm_for_reset(test_nvidia_pci_device_reset_method.__name__, sd)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_renesas_xhci_pci_config(_, __):
    """
    Verify the Renesas xHCI PCI host controller driver is enabled.

    JIRA: DGX-11145
    PATCH: https://git.kernel.org/torvalds/c/25f51b76f90f
    """
    config_name = "CONFIG_USB_XHCI_PCI_RENESAS"
    kconfig_path = "/boot/config-" + platform.release()

    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace"):
        GLTCore.GLTTestResult.skip("This is not an NVIDIA Grace CPU.")

    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except OSError as e:
        GLTCore.GLTTestResult.fail(
            f"Cannot read kernel config at {kconfig_path}; "
            f"cannot verify {config_name}: {e}"
        )
    except ValueError as e:
        GLTCore.GLTTestResult.fail(
            f"Cannot parse kernel config at {kconfig_path}; "
            f"cannot verify {config_name}: {e}"
        )

    value = kconfig.get(config_name)
    if value not in ("y", "m"):
        GLTCore.GLTTestResult.fail(
            f"{config_name} must be enabled as built-in or module; "
            f"found {value!r}. Apply kernel config support for "
            "https://git.kernel.org/torvalds/c/25f51b76f90f."
        )


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
            ["sh", "-c", "echo 1 > /sys/bus/pci/rescan"],
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
            ["lspci", "-vv"],
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
        GLTCore.GLTTestResult.fail(f"Failed to run 'lspci -vv': {e}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_pci_aspm_saved_state_update(_, logger):
    """
    Verify ASPM policy changes update PCIe saved Link Control state.

    The test changes the system-wide PCIe ASPM policy and restores it in cleanup.
    SIGKILL or host failure before cleanup can leave the policy changed.

    JIRA: DGX-8962
    PATCH: https://git.kernel.org/torvalds/c/6d4266675279
    """
    if not os.path.exists(ASPM_POLICY_PATH):
        GLTCore.GLTTestResult.skip(
            "PCIe ASPM policy control is not available; CONFIG_PCIEASPM may be disabled."
        )

    try:
        missing_configs = _check_aspm_saved_state_kernel_config()
    except OSError as e:
        GLTCore.GLTTestResult.skip(f"Cannot read running kernel config: {e}")
    if missing_configs:
        GLTCore.GLTTestResult.skip(
            "Required kernel config not enabled: " + ", ".join(missing_configs)
        )

    try:
        policies, original_policy = _read_aspm_policy()
    except OSError as e:
        GLTCore.GLTTestResult.fail(f"Cannot read PCIe ASPM policy: {e}")

    if not original_policy:
        GLTCore.GLTTestResult.fail("Cannot determine current PCIe ASPM policy.")

    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            GLTCore.GLTTestResult.skip(
                f"{command} is unavailable; cannot build or load the ASPM saved-state probe."
            )

    candidate_policies = _candidate_aspm_policies(policies, original_policy)
    if not candidate_policies:
        GLTCore.GLTTestResult.skip(
            "Only one PCIe ASPM policy is available; policy-change path is not observable."
        )

    module_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ASPM_SAVED_STATE_MODULE,
    )
    module_loaded = False
    policy_changed = False

    with glt_utils.chdir_managed(module_dir):
        try:
            subprocess.run(["make"], check=True, shell=False)
            if not glt_utils.sign_kernel_module(f"{ASPM_SAVED_STATE_MODULE}.ko", logger):
                GLTCore.GLTTestResult.fail(
                    f"Cannot sign {ASPM_SAVED_STATE_MODULE}.ko for Secure Boot. "
                    "See docs/user-guides/secure-boot.md for setup."
                )

            result = subprocess.run(
                ["insmod", f"{ASPM_SAVED_STATE_MODULE}.ko"],
                capture_output=True,
                shell=False,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "insmod failed"
                GLTCore.GLTTestResult.skip(
                    "Cannot load ASPM saved-state probe module; kprobes or "
                    f"pcie_config_aspm_link() are unavailable. Detail: {detail}"
                )
            module_loaded = True

            before_counts = _read_aspm_probe_hits()
            after_counts = before_counts
            observed_policy = None

            for alternate_policy in candidate_policies:
                logger.info(
                    "Switching PCIe ASPM policy from %s to %s.",
                    original_policy,
                    alternate_policy,
                )
                _write_aspm_policy(alternate_policy)
                policy_changed = True
                _write_aspm_policy(original_policy)
                policy_changed = False
                after_counts = _read_aspm_probe_hits()

                transition_delta = (
                    after_counts.get("transition_hits", 0) -
                    before_counts.get("transition_hits", 0)
                )
                if transition_delta > 0:
                    observed_policy = alternate_policy
                    break

            config_delta = (
                after_counts.get("config_hits", 0) -
                before_counts.get("config_hits", 0)
            )
            saved_state_delta = (
                after_counts.get("saved_state_hits", 0) -
                before_counts.get("saved_state_hits", 0)
            )

            if config_delta <= 0:
                GLTCore.GLTTestResult.skip(
                    "PCIe ASPM policy changed but did not reprogram any link; "
                    "saved-state update path is not observable on this system."
                )

            if transition_delta <= 0:
                GLTCore.GLTTestResult.skip(
                    "PCIe ASPM policy changed but did not change any inspected "
                    "Link Control ASPM/CLKREQ bits; saved-state update path is "
                    "not observable on this system."
                )

            inspection_delta = (
                after_counts.get("saved_state_inspections", 0) -
                before_counts.get("saved_state_inspections", 0)
            )
            mismatch_delta = (
                after_counts.get("saved_state_mismatches", 0) -
                before_counts.get("saved_state_mismatches", 0)
            )
            missed_delta = (
                after_counts.get("probe_missed_instances", 0) -
                before_counts.get("probe_missed_instances", 0)
            )

            if inspection_delta <= 0:
                GLTCore.GLTTestResult.skip(
                    "PCIe ASPM policy changed, but no saved PCIe Link Control "
                    "state was available to inspect on the reprogrammed links."
                )

            if missed_delta > 0:
                GLTCore.GLTTestResult.skip(
                    "ASPM saved-state probe missed kretprobe instances; "
                    "saved-state update path is not fully observable on this system."
                )

            if mismatch_delta > 0:
                GLTCore.GLTTestResult.fail(
                    "PCIe ASPM link configuration changed, but "
                    "saved PCIe Link Control state did not match live state; "
                    "ASPM state may remain stale across PCIe error recovery. Apply "
                    "commit 6d4266675279."
                )

            if saved_state_delta <= 0:
                GLTCore.GLTTestResult.fail(
                    "PCIe ASPM link configuration changed, but no matching "
                    "saved PCIe Link Control state was observed; apply commit "
                    "6d4266675279."
                )

            logger.info(
                "ASPM policy %s produced %d config writes, %d Link Control "
                "transitions, and %d saved-state matches.",
                observed_policy,
                config_delta,
                transition_delta,
                saved_state_delta,
            )
        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(f"Error executing command: {e}")
        except (OSError, ValueError) as e:
            GLTCore.GLTTestResult.fail(f"PCIe ASPM saved-state validation failed: {e}")
        finally:
            _cleanup_aspm_saved_state_probe(
                module_loaded,
                policy_changed,
                original_policy,
                logger,
            )


def _has_usb_devices_behind_hub(controller_bdf: str) -> tuple[bool, str]:
    """Return (found, reason) where found is True if a USB device is enumerated behind a hub.

    """
    ctrl_path = f"/sys/bus/pci/devices/{controller_bdf}"
    bus_numbers = []
    try:
        for entry in os.listdir(ctrl_path):
            if entry.startswith("usb") and entry[3:].isdigit():
                bus_numbers.append(entry[3:])
    except OSError as e:
        return False, f"unable to inspect {ctrl_path}: {e}"
    if not bus_numbers:
        return False, f"no USB bus found under {ctrl_path}"

    bus_alt = "|".join(re.escape(b) for b in bus_numbers)
    behind_hub_re = re.compile(r"^(?:" + bus_alt + r")-\d+(?:\.\d+)+$")
    try:
        found = any(behind_hub_re.match(name) for name in os.listdir("/sys/bus/usb/devices"))
    except OSError as e:
        return False, f"unable to inspect /sys/bus/usb/devices: {e}"
    if not found:
        return False, "no USB devices enumerated behind hub"
    return True, ""


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_aspeed_ast1150_bridge_alias_quirk(_, logger):
    """
    Verify devices behind an ASPEED AST1150 bridge are not affected by missing
    PCI_BRIDGE_NO_ALIAS quirk.
    If USB host controllers are present the test additionally checks that USB
    devices are reachable behind the ASPEED hub.

    JIRA: DGX-16330
    PATCH: https://git.kernel.org/torvalds/c/eeb95c07d5fc
    PATCH: https://git.kernel.org/torvalds/c/550a190494a0
    PATCH: https://github.com/NVIDIA/NV-Kernels/commit/738fff0e2060
    """
    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.skip("lspci not available; install 'pciutils' to perform the test.")

    ast1150_bdfs = pci_utils.get_bdfs_by_pci_id(
        pci_utils.ASPEED_VENDOR_ID,
        pci_utils.ASPEED_AST1150_DEVICE_ID,
    )
    if not ast1150_bdfs:
        GLTCore.GLTTestResult.skip(
            "ASPEED AST1150 PCI-to-PCI bridge not found; quirk is not observable."
        )

    logger.info("Found ASPEED AST1150 bridge at: %s", ", ".join(ast1150_bdfs))

    all_children: list[tuple[str, str]] = []  # (bridge_bdf, child_bdf)
    for bridge_bdf in ast1150_bdfs:
        for child_bdf in pci_utils.get_pci_children(bridge_bdf):
            all_children.append((bridge_bdf, child_bdf))
            logger.info("Found device %s behind AST1150 %s.", child_bdf, bridge_bdf)

    if not all_children:
        GLTCore.GLTTestResult.fail(
            "No devices found behind AST1150; expected child devices on NVIDIA platforms."
        )

    aliasing_bdfs = []
    usb_controllers: list[tuple[str, str]] = []  # (bridge_bdf, usb_bdf)
    for bridge_bdf, child_bdf in all_children:
        if glt_utils.check_kernel_log(f"pci {child_bdf}: Aliasing StreamID"):
            aliasing_bdfs.append(child_bdf)
            logger.error(
                "Device %s behind AST1150 %s: kernel log shows 'Aliasing StreamID' — "
                "PCI_BRIDGE_NO_ALIAS quirk is missing, DMA is broken.",
                child_bdf, bridge_bdf,
            )
        dev_class = pci_utils.get_pci_class(child_bdf)
        if dev_class and dev_class.startswith(pci_utils.PCI_CLASS_USB_HOST):
            usb_controllers.append((bridge_bdf, child_bdf))
            logger.info("USB controller %s found behind AST1150 %s.", child_bdf, bridge_bdf)

    if aliasing_bdfs:
        GLTCore.GLTTestResult.fail(
            f"Aliasing StreamID errors in kernel log for devices behind AST1150: "
            f"{', '.join(aliasing_bdfs)}. "
            "The AST1150 PCI_BRIDGE_NO_ALIAS quirk is missing; DMA will be broken. "
            "Apply https://git.kernel.org/torvalds/c/eeb95c07d5fc or equivalent."
        )

    if not usb_controllers:
        logger.info("No USB controllers behind AST1150; hub enumeration check not applicable.")
        return

    no_devices_behind_hub = []  # list of (usb_bdf, reason)
    for bridge_bdf, usb_bdf in usb_controllers:
        ok, reason = _has_usb_devices_behind_hub(usb_bdf)
        if not ok:
            no_devices_behind_hub.append((usb_bdf, reason))
            logger.warning(
                "USB controller %s behind AST1150 %s: %s",
                usb_bdf, bridge_bdf, reason,
            )
        else:
            logger.info(
                "USB controller %s behind AST1150 %s: USB devices behind hub OK.",
                usb_bdf, bridge_bdf,
            )

    if no_devices_behind_hub:
        details = "; ".join(f"{bdf}: {reason}" for bdf, reason in no_devices_behind_hub)
        GLTCore.GLTTestResult.warn(
            f"USB host controllers behind AST1150 with issues: {details}. "
            "If no USB devices behind hub, apply "
            "https://github.com/NVIDIA/NV-Kernels/commit/738fff0e2060"
        )


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
