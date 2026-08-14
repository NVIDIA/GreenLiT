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

# pylint: disable=too-many-lines

import os
import platform
import re
import subprocess
import time
import uuid
from common import cxl_utils
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


def _get_cxl_cache_bdfs(bdfs, logger):
    """Return BDFs that advertise a readable CXL.cache capability."""
    cxl_cache_bdfs = []
    for bdf in bdfs:
        dvsec_off = cxl_utils.find_cxl_device_dvsec_base(bdf)
        cap_bits = (
            cxl_utils.read_cxl_device_dvsec_cap_bits(bdf, dvsec_off)
            if dvsec_off is not None else None
        )
        if cap_bits and cap_bits["cache_capable"]:
            cxl_cache_bdfs.append(bdf)
            continue
        reason = (
            "CXL Device DVSEC unavailable or unreadable"
            if cap_bits is None else "CXL.cache capability is clear"
        )
        logger.debug("PCI device %s is not an applicable CXL.cache device: %s.",
                     bdf, reason)
    return cxl_cache_bdfs


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
def test_cxl_cache_ats_always_on(_, logger):
    """
    Verifies ATS remains enabled for CXL.cache devices in identity domains.

    PATCH: PCI: Add pci_ats_required() for CXL.cache capable devices
           (91561e1dc94b)
    PATCH: iommu/arm-smmu-v3: Allow ATS to be always on (5ff98a0da6b4)
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("Test only applies to Vera-based platforms.")

    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.skip("lspci not available; install 'pciutils' to perform the test.")

    try:
        target_bdfs = sorted(os.listdir("/sys/bus/pci/devices"))
    except OSError as error:
        GLTCore.GLTTestResult.skip(f"Unable to enumerate PCI devices: {error}.")

    cxl_cache_bdfs = _get_cxl_cache_bdfs(
        target_bdfs, logger
    )

    if not cxl_cache_bdfs:
        GLTCore.GLTTestResult.skip(
            "No PCI device with readable CXL.cache capability found; "
            f"ATS always-on path is not observable after checking "
            f"{len(target_bdfs)} PCI device(s)."
        )

    disabled_bdfs = []
    unvalidated_bdfs = []
    for bdf in cxl_cache_bdfs:
        group_type = pci_utils.get_iommu_group_type(bdf)
        ats_enabled = pci_utils.is_ats_enabled(bdf)
        if group_type is None:
            unvalidated_bdfs.append(bdf)
            logger.warning(
                "CXL.cache device %s: IOMMU group type is unreadable; cannot "
                "validate the always-on bypass path.",
                bdf,
            )
            continue
        if group_type != "identity":
            unvalidated_bdfs.append(bdf)
            logger.warning(
                "CXL.cache device %s is not in an identity IOMMU domain "
                "(type=%s); its ATS state does not validate the always-on bypass path.",
                bdf, group_type,
            )
            continue

        if ats_enabled is None:
            unvalidated_bdfs.append(bdf)
            logger.warning(
                "Unable to determine ATS state for CXL.cache device %s; lspci "
                "failed or did not expose a conclusive ATSCtl value.",
                bdf,
            )
            continue
        if ats_enabled:
            logger.info("ATS is enabled on CXL.cache device %s.", bdf)
        else:
            disabled_bdfs.append(bdf)

    if disabled_bdfs:
        GLTCore.GLTTestResult.fail(
            "CXL.cache device(s) expected to use ATS always-on do not show "
            f"'ATSCtl: Enable+': {', '.join(disabled_bdfs)}. "
            "Apply the CXL.cache PCI ATS requirement (91561e1dc94b) and "
            "ARM SMMUv3 ATS always-on support (5ff98a0da6b4)."
        )

    if unvalidated_bdfs:
        GLTCore.GLTTestResult.warn(
            "CXL.cache device(s) did not conclusively exercise the ATS always-on "
            f"bypass path: {', '.join(unvalidated_bdfs)}. Devices must use "
            "an identity IOMMU domain and show 'ATSCtl: Enable+'."
        )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
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
        cmd_output = glt_utils.run_command(
            ["lspci", "-Dv"],
            "Failed to enumerate PCI devices with 'lspci'",
        )

        if not cmd_output.stdout.strip():
            GLTCore.GLTTestResult.skip("No PCI/PCIe devices found on this machine. PCI enumeration is not valid.")
        device_blocks = cmd_output.stdout.strip().split("\n\n")
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(str(error))

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
        cmd_output = glt_utils.run_command(
            ["lspci", "-vv"],
            "Failed to run 'lspci -vv'",
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
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(str(error))


@GLTCore.GLTTestAttr()
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

    try:
        kernel_log = glt_utils.get_kernel_log()
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to retrieve kernel log for ASPEED alias check: {error}"
        )

    aliasing_bdfs = []
    usb_controllers: list[tuple[str, str]] = []  # (bridge_bdf, usb_bdf)
    for bridge_bdf, child_bdf in all_children:
        if f"pci {child_bdf}: Aliasing StreamID" in kernel_log:
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
@GLTCore.GLTPlan("disruptive")
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

    try:
        kernel_log = glt_utils.get_kernel_log()
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to collect kernel logs: {error}"
        )

    # Check if the kernel supports ACS configurability
    if "PCI: Unknown option `config_acs" in kernel_log:
        GLTCore.GLTTestResult.fail(
            "PCI ACS configurability is not supported in the kernel." +
            " Consider backporting commit 47c8846 to fix it."
        )

    # Check if the kernel has config_acs fixed
    if parse_err in kernel_log:
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

    has_acs_control = acs_control_msg in kernel_log
    has_acs_fw_ctrl = acs_fw_ctrl_msg in kernel_log

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


def _setpci_read_word(bdf: str, offset: str):
    """Read a 16-bit word from PCI config space at `offset` on `bdf` via setpci."""
    try:
        result = subprocess.run(
            ["setpci", "-s", bdf, f"{offset}.w"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return int(result.stdout.strip(), 16)
    except ValueError:
        return None


def _setpci_write_word(bdf: str, offset: str, value: int) -> bool:
    """Write a 16-bit `value` to `offset` on `bdf` via setpci."""
    try:
        subprocess.run(
            ["setpci", "-s", bdf, f"{offset}.w={value:#x}"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _pci_bdfs_on_path(sysfs_path: str) -> list[str]:
    """Return the PCI BDF components on the resolved sysfs path, root-first.

    For a PCI endpoint this yields [root_port, bridge..., endpoint]; for a
    non-PCI device (or a missing path) it yields an empty list.
    """
    bdf_component_re = re.compile(
        r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
    real = os.path.realpath(sysfs_path)
    return [comp for comp in real.split(os.sep) if bdf_component_re.match(comp)]


# pylint: disable=too-many-return-statements,too-many-branches
def _resolve_and_validate_target(bdf: str):
    """Validate that `bdf` is a safe, expendable ConnectX target."""
    mlx_vendor_id = "15b3"
    # Block devices that are RAM-backed or file-backed (a loop device's backing
    # file lives on a real filesystem that is itself in /proc/mounts and thus
    # already checked). These are outside the DPC blast radius by construction,
    # so the storage-domain gate ignores them instead of failing closed.
    virtual_block_dev_re = re.compile(r"^(loop|zram|ram)\d+(p\d+)?$")

    connectx_bdfs = [b.lower() for b in pci_utils.get_bdfs_by_pci_id(mlx_vendor_id)]
    if bdf not in connectx_bdfs:
        return False, f"configured device {bdf} is not a ConnectX/mlx5 device"

    target_chain = _pci_bdfs_on_path(f"/sys/bus/pci/devices/{bdf}")
    target_root = target_chain[0] if target_chain else None
    if not target_root:
        return False, (
            f"refusing: cannot resolve the PCIe root port of {bdf}; unable "
            "to prove domain isolation from management and storage devices"
        )

    # Refuse any function of the device that carries the default route
    # (the management network typically rides the other port/function), and
    # any management NIC that shares the target's root-port (DPC
    # containment) domain. Unreadable route tables fail closed: an empty
    # iface set must not skip this gate.
    target_slot = bdf.rsplit(".", 1)[0]
    try:
        mgmt_ifaces = pci_utils.default_route_ifaces()
    except OSError:
        return False, (
            "refusing: cannot read default-route tables. Unable to prove "
            "isolation from the management network"
        )
    for iface in mgmt_ifaces:
        iface_bdfs = pci_utils.iface_pci_bdfs(iface)
        if not iface_bdfs:
            return False, (
                f"refusing: cannot resolve PCI backing device for "
                f"default-route interface {iface}. Unable to prove isolation"
            )
        for iface_bdf in iface_bdfs:
            if iface_bdf.rsplit(".", 1)[0] == target_slot:
                return False, f"refusing: {bdf} carries the management network route"
            mgmt_chain = _pci_bdfs_on_path(f"/sys/bus/pci/devices/{iface_bdf}")
            if not mgmt_chain:
                return False, (
                    f"refusing: cannot resolve PCI chain for management device "
                    f"{iface_bdf}. Unable to prove isolation from {bdf}"
                )
            if mgmt_chain[0] == target_root:
                return False, (
                    f"refusing: {bdf} shares PCIe domain {target_root} with "
                    f"the management network device ({iface_bdf})"
                )

    # Refuse if the target shares a PCIe root-port (DPC containment) domain
    # with the root disk or any mounted filesystem. Mounted devices are
    # resolved through dm/md/crypt stacks to their physical disks; anything
    # that cannot be traced to a PCI device fails closed (refused), except
    # loop/zram/ram devices, which are RAM- or file-backed and therefore
    # outside the DPC blast radius. Unreadable /proc/mounts fails closed.
    try:
        mounted_devs = pci_utils.mounted_block_devices()
    except OSError:
        return False, (
            "refusing: cannot read /proc/mounts; unable to prove isolation "
            "from mounted storage"
        )
    for dev in mounted_devs:
        if virtual_block_dev_re.match(dev):
            continue
        leaves = pci_utils.block_device_physical_leaves(dev)
        if not leaves:
            return False, (
                f"refusing: mounted block device {dev} resolves to no "
                "backing devices; cannot prove it is outside the DPC "
                "blast radius"
            )
        for leaf in leaves:
            if virtual_block_dev_re.match(leaf):
                continue
            dev_chain = _pci_bdfs_on_path(f"/sys/class/block/{leaf}")
            if not dev_chain:
                return False, (
                    f"refusing: mounted block device {dev} (backing device "
                    f"{leaf}) does not resolve to a PCI device; cannot prove "
                    "it is outside the DPC blast radius"
                )
            if dev_chain[0] == target_root:
                return False, (
                    f"refusing: {bdf} shares PCIe domain {target_root} "
                    f"with mounted storage ({dev})"
                )

    return True, ""
# pylint: enable=too-many-return-statements,too-many-branches


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_msix_mem_restore_no_ur(sd, logger):
    """
    Verify the kernel restores MSI-X messages with memory decoding enabled:
    FLR a vfio-bound ConnectX PF whose saved config has COMMAND.Memory
    cleared and check that the MSI-X-table restore does not fault as an
    Unsupported Request / RP PIO error / DPC containment.

    The fix is NOT yet upstream: a FAIL is remediated by backporting
    "PCI/MSI: Enable memory decoding before restoring MSI-X messages"
    (patch 4/4 of the v20 "Error recovery for vfio-pci devices on s390x"
    series).

    DESTRUCTIVE: on an unpatched kernel this wedges the target NIC and a BMC
    power cycle may be required to recover it.

    JIRA: DGX-17078
    PATCH: https://lore.kernel.org/all/20260622171840.1618-5-alifm@linux.ibm.com/
    """
    target_bdf_env = "GLT_MSIX_MEM_RESTORE_BDF"
    pci_command_offset = "0x04"
    pci_command_memory = 0x2
    # Kernel-log fragments that identify the UR / RP PIO / DPC / AER signature
    # of the unpatched MSI-X restore path. Each is checked for as a substring
    # of the post-FLR kernel log window via glt_utils.get_kernel_log().
    fault_signatures = (
        "DPC: containment event",
        "RP PIO",
        "Memory Request received UR Completion",
        "AER: Uncorrected",
        "severity=Uncorrected",
    )
    patch_fail_msg = (
        "MSI-X restore raised an Unsupported Request / DPC containment — kernel "
        "lacks the memory-decoding fix. Consider backporting "
        "https://lore.kernel.org/all/20260622171840.1618-5-alifm@linux.ibm.com/ "
        "(PCI/MSI: Enable memory decoding before restoring MSI-X messages) to "
        "fix it."
    )
    msix_bdf_re = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")

    if (not glt_utils.is_cli_available("lspci")
            or not glt_utils.is_cli_available("setpci")):
        GLTCore.GLTTestResult.skip("lspci/setpci not available; install pciutils")

    bdf = os.environ.get(target_bdf_env, "").strip().lower()
    if not bdf:
        GLTCore.GLTTestResult.skip(
            f"set {target_bdf_env} to an expendable ConnectX BDF; this test "
            "is destructive and may require a BMC power cycle to recover")
    dev_path = f"/sys/bus/pci/devices/{bdf}"
    if not msix_bdf_re.match(bdf) or not os.path.exists(dev_path):
        GLTCore.GLTTestResult.skip(f"configured device {bdf} not present")

    safe, reason = _resolve_and_validate_target(bdf)
    if not safe:
        GLTCore.GLTTestResult.skip(reason)

    try:
        iommu_groups = os.listdir("/sys/kernel/iommu_groups")
    except OSError:
        iommu_groups = []
    if not iommu_groups:
        GLTCore.GLTTestResult.skip("IOMMU not enabled; vfio requires IOMMU")

    if not glt_utils.is_cli_available("vfio_mlx5"):
        GLTCore.GLTTestResult.skip("vfio_mlx5 not found; install DOCA libvfio-mlx5")

    try:
        subprocess.run(["modprobe", "vfio-pci"],
                       capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        GLTCore.GLTTestResult.skip("vfio-pci module unavailable")

    drv_link = f"{dev_path}/driver"

    sd.progress_record_time(test_msix_mem_restore_no_ur.__name__,
                            "MsixMemRestoreMutated")

    if os.path.islink(drv_link):
        orig_driver = os.path.basename(os.path.realpath(drv_link))
        try:
            glt_utils.write_sysfs_value(
                f"/sys/bus/pci/devices/{bdf}/driver/unbind", bdf)
        except OSError as e:
            GLTCore.GLTTestResult.skip(
                f"could not unbind {orig_driver} from {bdf}: {e}")
    try:
        glt_utils.write_sysfs_value(
            f"/sys/bus/pci/devices/{bdf}/driver_override", "vfio-pci")
        glt_utils.write_sysfs_value("/sys/bus/pci/drivers/vfio-pci/bind", bdf)
    except OSError as e:
        GLTCore.GLTTestResult.skip(f"could not bind {bdf} to vfio-pci: {e}")
    if (not os.path.islink(drv_link)
            or os.path.basename(os.path.realpath(drv_link)) != "vfio-pci"):
        GLTCore.GLTTestResult.skip(f"{bdf} did not bind to vfio-pci")
    logger.info("%s bound to vfio-pci", bdf)

    vf_token = str(uuid.uuid4())
    try:
        # pylint: disable-next=consider-using-with
        proc = subprocess.Popen(
            ["vfio_mlx5", f"--device={bdf},vf_token={vf_token}",
             "--memsize=1G"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        GLTCore.GLTTestResult.skip(f"could not launch vfio_mlx5: {e}")

    try:
        lspci_out = ""
        msix_enabled = False
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                GLTCore.GLTTestResult.skip(
                    f"vfio_mlx5 exited early (rc={proc.returncode}) before "
                    f"enabling MSI-X on {bdf}")
            try:
                result = subprocess.run(["lspci", "-vv", "-s", bdf],
                                        capture_output=True, text=True,
                                        check=False)
                lspci_out = result.stdout if result.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                lspci_out = ""
            if "MSI-X: Enable+" in lspci_out:
                msix_enabled = True
                break
            time.sleep(1)
        if not msix_enabled:
            GLTCore.GLTTestResult.skip(
                f"vfio_mlx5 did not enable MSI-X on {bdf} within 60s")

        # BAR0 base + MSI-X vector-table offset, for log attribution only.
        bar0 = re.search(r"Region 0: Memory at ([0-9a-f]+)", lspci_out)
        table = re.search(r"Vector table: BAR=(\d+) offset=([0-9a-f]+)",
                          lspci_out)
        logger.info(
            "MSI-X enabled on %s (BAR0=%s, vector table BAR=%s offset=0x%s)",
            bdf,
            bar0.group(1) if bar0 else "?",
            table.group(1) if table else "?",
            table.group(2) if table else "?")

        try:
            checkpoint = glt_utils.capture_kernel_log_checkpoint()
        except RuntimeError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to capture kernel log checkpoint before FLR on "
                f"{bdf}: {error}"
            )

        command = _setpci_read_word(bdf, pci_command_offset)
        if command is None:
            GLTCore.GLTTestResult.skip(
                f"could not read COMMAND on {bdf} via setpci")

        if not _setpci_write_word(bdf, pci_command_offset,
                                  command & ~pci_command_memory):
            GLTCore.GLTTestResult.skip(f"failed to clear COMMAND.Memory on {bdf}")
        readback = _setpci_read_word(bdf, pci_command_offset)
        if readback is None or readback & pci_command_memory:
            GLTCore.GLTTestResult.skip(
                f"COMMAND.Memory did not clear on {bdf} "
                f"(read back {'unreadable' if readback is None else hex(readback)})")

        # FLR is the chosen reliable trigger for this regression. There might be
        # other reset paths that could hit the MSI-X path under test.
        logger.info("Triggering FLR on %s", bdf)
        try:
            subprocess.run(
                ["sh", "-c", f"echo 1 > {dev_path}/reset"],
                capture_output=True, text=True, check=True, timeout=120)
        except subprocess.TimeoutExpired:
            GLTCore.GLTTestResult.fail(
                f"FLR write on {bdf} did not complete within 120s")
        except subprocess.CalledProcessError as e:
            GLTCore.GLTTestResult.fail(
                f"FLR write on {bdf} failed (rc={e.returncode}): "
                f"{(e.stderr or '').strip()}")
        except (OSError, subprocess.SubprocessError) as e:
            GLTCore.GLTTestResult.fail(f"FLR could not be issued on {bdf} ({e})")

        try:
            with open(f"/sys/bus/pci/devices/{bdf}/config", "rb") as cfg:
                vendor = cfg.read(2)
            speed = glt_utils.read_sysfs_value(
                f"/sys/bus/pci/devices/{bdf}/current_link_speed")
            healthy = (len(vendor) == 2 and vendor != b"\xff\xff"
                       and not speed.startswith("Unknown"))
        except OSError:
            healthy = False

        try:
            kernel_log = glt_utils.get_kernel_log(checkpoint)
        except RuntimeError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to retrieve kernel log after FLR on {bdf}: {error}"
            )

        for signature in fault_signatures:
            if signature in kernel_log:
                logger.error("fault signature in kernel log: %s", signature)
                GLTCore.GLTTestResult.fail(patch_fail_msg)

        if not healthy:
            GLTCore.GLTTestResult.fail(
                "device did not recover after FLR "
                "(config inaccessible / link down)")

        logger.info("FLR completed cleanly on %s; no UR/DPC/AER signature", bdf)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
# pylint: enable=too-many-branches


_msix_mem_restore_reboot_hook = GLTCore.PostHooks.for_conditional_reboot(
    "MsixMemRestoreMutated", lambda: True)
test_msix_mem_restore_no_ur.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_msix_mem_restore_reboot_hook,
    postskip=_msix_mem_restore_reboot_hook,
    postfail=_msix_mem_restore_reboot_hook,
    postafail=_msix_mem_restore_reboot_hook,
)
