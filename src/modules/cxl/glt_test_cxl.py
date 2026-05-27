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

"""
CXL bus tests.
"""

import glob
import mmap
import os
import platform
import re
import struct
import subprocess
import time

from common import cxl_utils
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


def _find_root_port_bdf(dev_bdf: str, sysfs_root: str = "/") -> str | None:
    """Walk up sysfs from `dev_bdf` and return the topmost PCI BDF on the path.

    Returns None if the device is missing or has no PCI parent (i.e. the chain
    contains only the device itself).
    """
    bdf_pattern = r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$"
    pci_devices_dir = os.path.join(sysfs_root, "sys", "bus", "pci", "devices")
    start = os.path.join(pci_devices_dir, dev_bdf)
    if not os.path.exists(start):
        return None

    ancestors = []
    cur = os.path.realpath(start)
    while True:
        base = os.path.basename(cur)
        if not base:
            break
        if re.match(bdf_pattern, base):
            ancestors.append(base)
        else:
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    if len(ancestors) < 2:
        return None
    return ancestors[-1]


def _setpci_read_word(bdf: str, offset: str) -> int | None:
    """Read a 16-bit word from PCI config space at `offset` on `bdf` via setpci.

    Returns None if setpci is missing, exits non-zero, or output is unparseable.
    """
    try:
        result = subprocess.run(
            ["setpci", "-s", bdf, f"{offset}.w"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip(), 16)
    except ValueError:
        return None


def _setpci_write_word(bdf: str, offset: str, value: int) -> bool:
    """Write a 16-bit `value` (unmasked) to `offset` on `bdf` via setpci.

    Returns True on success, False if setpci is missing or exits non-zero.
    """
    try:
        result = subprocess.run(
            ["setpci", "-s", bdf, f"{offset}.w={value:#x}"],
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _setpci_write_byte(bdf: str, offset: str, value: int, mask: int) -> bool:
    """Masked 8-bit write of `value` (under `mask`) to `offset` on `bdf`.

    Returns True on success, False if setpci is missing or exits non-zero.
    """
    try:
        result = subprocess.run(
            [
                "setpci", "-s", bdf,
                f"{offset}.b={value:#x}:{mask:#x}",
            ],
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return result.returncode == 0


def _pci_rescan() -> bool:
    """Trigger a full PCI bus rescan. Returns True on success."""
    try:
        result = subprocess.run(
            ["sh", "-c", "echo 1 > /sys/bus/pci/rescan"],
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return result.returncode == 0


def _postsuccess_restore_link(_scheduler, attr, logger):
    """Restore the root port to its pre-test state.    """
    state = getattr(attr, "cleanup_state", None)
    if state is None:
        return
    rp_bdf = state["rp_bdf"]
    orig_pc = state["orig_pc"]
    pco_offset = state["pco_offset"]
    if not _setpci_write_byte(rp_bdf, "CAP_EXP+0x10", 0x00, mask=0x10):
        logger.error(
            "CXL cleanup: failed to clear Link Disable on %s", rp_bdf,
        )
    if not _setpci_write_word(rp_bdf, f"{pco_offset:#x}", orig_pc):
        logger.error(
            "CXL cleanup: failed to restore Port Control Override "
            "at %#x on %s", pco_offset, rp_bdf,
        )
    if not _pci_rescan():
        logger.error("CXL cleanup: PCI rescan failed")
        return
    # Rescan returns immediately, but the kernel link-training, AER recovery
    # and device probe sequence takes several seconds. Poll for the CXL
    # device to reappear so subsequent tests do not see a transient missing-
    # device state.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if cxl_utils.find_first_cxl_bdf() is not None:
            return
        time.sleep(0.5)
    logger.warning(
        "CXL cleanup: device did not re-enumerate within 15s on %s",
        rp_bdf,
    )


# Post-hook that reboots only if the test got far enough to bring the link
# down (the "LinkDisabled" progress key is set in the test body just before
# the Link Disable setpci write). Failures earlier than that leave the system
# in a clean state and do not trigger a reboot.
_link_down_reboot_hook = GLTCore.PostHooks.for_conditional_reboot(
    "LinkDisabled",
    lambda: True,
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches
def test_cxl_link_disable_led_error(sd, logger):
    """
    Verifies that a CXL root port can issue Link Disable without the kernel
    emitting "Setting an LED's brightness failed (-19)" during the link
    teardown path.

    JIRA: DGX-16338
    PATCH: https://github.com/torvalds/linux/commit/16d021c878dca22532c984668c9e8cf4722d6a49
    """

    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("Test only applies to Vera-based platforms")

    kconfig_path = "/boot/config-" + platform.release()
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"Kernel config not found at {kconfig_path}"
        )
    if kconfig.get("CONFIG_PCI_NPEM") != "y":
        GLTCore.GLTTestResult.skip(
            "CONFIG_PCI_NPEM is not enabled in the running kernel; "
            "the LED-brightness code path under test is not present"
        )

    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.skip("lspci not available; install pciutils")
    if not glt_utils.is_cli_available("setpci"):
        GLTCore.GLTTestResult.skip("setpci not available; install pciutils")

    cxl_bdf = cxl_utils.find_first_cxl_bdf()
    if cxl_bdf is None:
        GLTCore.GLTTestResult.skip("No CXL device found")
    logger.info("CXL device BDF: %s", cxl_bdf)

    if cxl_utils.any_cxl_region_committed():
        GLTCore.GLTTestResult.skip(
            "CXL regions are committed/online, "
            "refusing to bring the link down"
        )

    rp_bdf = _find_root_port_bdf(cxl_bdf)
    if rp_bdf is None:
        GLTCore.GLTTestResult.skip(
            f"Could not locate root port for CXL device {cxl_bdf}"
        )
    logger.info("CXL root port BDF: %s", rp_bdf)

    dvsec_base = cxl_utils.find_cxl_port_dvsec_base(rp_bdf)
    if dvsec_base is None:
        GLTCore.GLTTestResult.fail(
            f"CXL Port Extensions DVSEC not found on root port {rp_bdf}"
        )
    # Port Control Override sits at offset 0x0C within the CXL Port Extensions DVSEC.
    pco_offset = dvsec_base + 0x0C
    pco_offset_str = f"{pco_offset:#x}"
    logger.info(
        "CXL Port DVSEC base on %s: %#x; Port Control Override at %s",
        rp_bdf, dvsec_base, pco_offset_str,
    )

    orig_pc = _setpci_read_word(rp_bdf, pco_offset_str)
    if orig_pc is None:
        GLTCore.GLTTestResult.fail(
            f"Could not read PCI config at {pco_offset_str} on {rp_bdf}"
        )
    logger.info("Original %s.w on %s: %#06x", pco_offset_str, rp_bdf, orig_pc)

    # Stash cleanup state BEFORE the first PCO write so any subsequent
    # skip path (postskip hook) restores the register.
    test_cxl_link_disable_led_error.attr.cleanup_state = {
        "rp_bdf": rp_bdf,
        "orig_pc": orig_pc,
        "pco_offset": pco_offset,
    }

    if not _setpci_write_word(rp_bdf, pco_offset_str, orig_pc | 0x0002):
        GLTCore.GLTTestResult.skip(
            f"setpci write to {pco_offset_str} on {rp_bdf} failed"
        )
    readback = _setpci_read_word(rp_bdf, pco_offset_str)
    if readback is None:
        GLTCore.GLTTestResult.fail(
            f"Could not read back PCI config at {pco_offset_str} on {rp_bdf}"
        )
    if (readback & 0x0002) == 0:
        GLTCore.GLTTestResult.skip(
            "UnmaskLinkDisable did not stick - firmware/HW restricts it"
        )

    logger.info("Issuing Link Disable on %s", rp_bdf)
    sd.progress_record_time(test_cxl_link_disable_led_error.__name__, "LinkDisabled")
    if not _setpci_write_byte(rp_bdf, "CAP_EXP+0x10", 0x10, mask=0x10):
        GLTCore.GLTTestResult.skip(
            f"setpci write of Link Disable on {rp_bdf} failed"
        )

    time.sleep(5)

    if glt_utils.check_kernel_log("Setting an LED's brightness failed (-19)"):
        GLTCore.GLTTestResult.fail(
            "Kernel log contains LED brightness failure after CXL link disable."
            " Consider backporting commit 16d021c8"
        )


test_cxl_link_disable_led_error.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_postsuccess_restore_link,
    postskip=_postsuccess_restore_link,
    postfail=_link_down_reboot_hook,
    postafail=_link_down_reboot_hook,
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_cxl_switch_decoder_target_list_complete(_, logger):
    """
    Verify programmed interleaved CXL switch decoders have a fully populated
    target_list. A switch decoder with interleave_ways > 1 whose target_list
    is empty or shorter than interleave_ways is the failure of the bug.

    JIRA: DGX-16570
    PATCH: https://git.kernel.org/torvalds/c/3e8aaacdad4f
    """
    qualifying_found = False

    for decoder_dir in sorted(glob.glob("/sys/bus/cxl/devices/decoder*")):
        devtype = ""
        interleave_ways = 0
        region = ""
        decoder_size = 0
        target_list_raw = ""
        target_list_entries = []
        try:
            with open(os.path.join(decoder_dir, "devtype"),
                      "r", encoding="utf-8") as f:
                devtype = f.read().strip()
        except OSError:
            continue
        if devtype != "cxl_decoder_switch":
            continue

        decoder_name = os.path.basename(decoder_dir)
        try:
            with open(os.path.join(decoder_dir, "interleave_ways"),
                      "r", encoding="utf-8") as f:
                interleave_ways = int(f.read().strip())
        except (OSError, ValueError) as e:
            GLTCore.GLTTestResult.fail(
                f"Unable to read CXL switch decoder interleave_ways: "
                f"{decoder_name}: {e}"
            )
        if interleave_ways <= 1:
            continue

        try:
            with open(os.path.join(decoder_dir, "region"),
                      "r", encoding="utf-8") as f:
                region = f.read().strip()
        except OSError:
            region = ""
        try:
            with open(os.path.join(decoder_dir, "size"),
                      "r", encoding="utf-8") as f:
                decoder_size = int(f.read().strip(), 0)
        except (OSError, ValueError) as e:
            GLTCore.GLTTestResult.fail(
                f"Unable to read CXL switch decoder size: "
                f"{decoder_name}: {e}"
            )
        # Zero-size decoders are idle slots, not programmed decode windows.
        if decoder_size == 0:
            continue

        qualifying_found = True
        try:
            with open(os.path.join(decoder_dir, "target_list"),
                      "r", encoding="utf-8") as f:
                target_list_raw = f.read().strip()
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Unable to read CXL switch decoder target_list: "
                f"{decoder_name}: {e}"
            )
        target_list_entries = [
            entry.strip()
            for entry in target_list_raw.split(",")
            if entry.strip()
        ]
        logger.info(
            "CXL switch decoder %s: interleave_ways=%d size=%#x "
            "region=%s target_list=%s",
            decoder_name, interleave_ways, decoder_size, region,
            target_list_entries,
        )
        if len(target_list_entries) < interleave_ways:
            GLTCore.GLTTestResult.fail(
                f"Incomplete CXL switch decoder target list: "
                f"{decoder_name} interleave_ways={interleave_ways} "
                f"size={decoder_size:#x} region={region!r} "
                f"target_list={target_list_entries}. "
                f"Consider backporting commit 3e8aaacdad4f"
            )

    if not qualifying_found:
        GLTCore.GLTTestResult.skip(
            "No programmed interleaved CXL switch decoder found; "
            "patch path not exercised"
        )


def _snapshot_dvsec(bdf: str, dvsec_off: int) -> dict[str, int]:
    """
    Read the Device DVSEC fields. Read via
    /sys/bus/pci/devices/<bdf>/config and return them as {name: value}.
    """
    # (name, offset within Device DVSEC, access width: 'w'=u16, 'l'=u32)
    saved_fields: list[tuple[str, int, str]] = [
        ("CTRL",     0x0C, "w"),
        ("CTRL2",    0x10, "w"),
        ("LOCK",     0x14, "w"),
        ("R0_SZ_HI", 0x18, "l"),
        ("R0_SZ_LO", 0x1C, "l"),
        ("R0_BS_HI", 0x20, "l"),
        ("R0_BS_LO", 0x24, "l"),
        ("R1_SZ_HI", 0x28, "l"),
        ("R1_SZ_LO", 0x2C, "l"),
        ("R1_BS_HI", 0x30, "l"),
        ("R1_BS_LO", 0x34, "l"),
    ]
    cfg_path = f"/sys/bus/pci/devices/{bdf}/config"
    with open(cfg_path, "rb") as f:
        cfg = f.read()
    out: dict[str, int] = {}
    for name, offset, width in saved_fields:
        fmt = "<H" if width == "w" else "<I"
        (val,) = struct.unpack_from(fmt, cfg, dvsec_off + offset)
        out[name] = val
    return out


# pylint: disable=too-many-locals
def _snapshot_hdm(bdf: str, comp_bar: int, comp_off: int) -> dict[str, int]:
    """
    mmap resource<comp_bar>, walk the CXL Component Register CM cap array
    at comp_off + CXL_CM_OFFSET to find the HDM Decoder cap (ID=0x5), then
    read GLOBAL_CTRL and, for each decoder, BASE_LO/HI, SIZE_LO/HI, CTRL,
    TL_LO/HI. Return a dict {name: int}.
    """
    # Offset of the CXL Component Register CM cap array within the component
    # register block (CXL_CM_OFFSET in the kernel).
    cxl_cm_offset = 0x1000
    # CM Capability ID for HDM Decoder Capability (CXL_CM_CAP_CAP_ID_HDM).
    cxl_cm_cap_id_hdm = 0x5
    # 64 KiB is enough to cover the CM cap array plus all per-cap blocks
    # (including HDM) on currently shipping CXL endpoints. Round up to the
    # runtime page size so this works on both 4K (x86) and 64K (Grace/Vera
    # ARM64) kernels.
    page_size = mmap.PAGESIZE
    read_span = 0x10000
    map_base = comp_off & ~(page_size - 1)
    map_span = (
        (comp_off - map_base) + read_span + page_size - 1
    ) & ~(page_size - 1)

    resource_path = f"/sys/bus/pci/devices/{bdf}/resource{comp_bar}"
    with open(resource_path, "rb") as f:
        mm = mmap.mmap(
            f.fileno(), map_span,
            mmap.MAP_SHARED, mmap.PROT_READ,
            offset=map_base,
        )
    try:
        # All reads below are u32 little-endian at an absolute BAR offset
        # `off`; the mmap starts at `map_base`, so the in-buffer position is
        # `off - map_base`.
        cm_base = comp_off + cxl_cm_offset
        cap_hdr = struct.unpack_from("<I", mm, cm_base - map_base)[0]
        cap_count = (cap_hdr >> 24) & 0xFF
        hdm_cap_off: int | None = None
        for i in range(1, cap_count + 1):
            hdr = struct.unpack_from(
                "<I", mm, (cm_base + 4 * i) - map_base,
            )[0]
            cap_id = hdr & 0xFFFF
            cap_offset = (hdr >> 20) & 0xFFF
            if cap_id == cxl_cm_cap_id_hdm:
                hdm_cap_off = cap_offset
                break
        if hdm_cap_off is None:
            return {}

        hdm_base = cm_base + hdm_cap_off
        # CXL r3.1 8.2.4.20.1 HDM Decoder Capability:
        #   Decoder Count [3:0] encodes 1, 2, 4, 6, 8, ... decoders.
        dec_count_field = struct.unpack_from(
            "<I", mm, (hdm_base + 0x0) - map_base,
        )[0] & 0xF
        decoders = dec_count_field * 2 if dec_count_field else 1

        out: dict[str, int] = {
            "GLOBAL_CTRL": struct.unpack_from(
                "<I", mm, (hdm_base + 0x4) - map_base,
            )[0],
        }
        for d in range(decoders):
            db = hdm_base + 0x10 + 0x20 * d
            out[f"D{d}_BASE_LO"] = struct.unpack_from(
                "<I", mm, (db + 0x00) - map_base,
            )[0]
            out[f"D{d}_BASE_HI"] = struct.unpack_from(
                "<I", mm, (db + 0x04) - map_base,
            )[0]
            out[f"D{d}_SIZE_LO"] = struct.unpack_from(
                "<I", mm, (db + 0x08) - map_base,
            )[0]
            out[f"D{d}_SIZE_HI"] = struct.unpack_from(
                "<I", mm, (db + 0x0C) - map_base,
            )[0]
            out[f"D{d}_CTRL"] = struct.unpack_from(
                "<I", mm, (db + 0x10) - map_base,
            )[0]
            out[f"D{d}_TL_LO"] = struct.unpack_from(
                "<I", mm, (db + 0x14) - map_base,
            )[0]
            out[f"D{d}_TL_HI"] = struct.unpack_from(
                "<I", mm, (db + 0x18) - map_base,
            )[0]
        return out
    finally:
        mm.close()
# pylint: enable=too-many-locals


def _restore_reset_method(_scheduler, attr, logger):
    """
    Restore /sys/bus/pci/devices/<bdf>/reset_method to its pre-test value.
    """
    state = getattr(attr, "cleanup_state", None)
    if state is None:
        return
    bdf = state["bdf"]
    orig = state["orig_reset_method"]
    try:
        with open(
            f"/sys/bus/pci/devices/{bdf}/reset_method", "w", encoding="utf-8",
        ) as f:
            f.write(orig)
    except OSError as e:
        logger.warning(
            "CXL cleanup: could not restore reset_method on %s: %s", bdf, e,
        )


# A failed cxl_bus reset can leave the CXL link in an indeterminate state;
# a reboot is the only reliable recovery. The for_conditional_reboot pattern
# with a `lambda: True` condition is effectively unconditional once the
# gating progress key is recorded (just before the reset trigger).
_cxl_bus_reset_reboot_hook = GLTCore.PostHooks.for_conditional_reboot(
    "CxlBusResetTriggered",
    lambda: True,
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches,too-many-statements,too-many-locals
def test_cxl_save_restore_across_cxl_bus_reset(sd, logger):
    """
    Verify that CXL Device DVSEC and HDM decoder state are preserved across
    a `cxl_bus` PCI reset on a Type-2 CXL endpoint.

    JIRA: DGX-16568
    PATCH: 
    https://lore.kernel.org/linux-cxl/SA3PR12MB9131BF1477CA8B87E8FD70EBBB40A@SA3PR12MB9131.namprd12.prod.outlook.com/T/#u
    """

    kconfig_path = "/boot/config-" + platform.release()
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"Kernel config not found at {kconfig_path}"
        )
    if kconfig.get("CONFIG_PCI_CXL") != "y":
        GLTCore.GLTTestResult.skip(
            "CONFIG_PCI_CXL is not enabled in the running kernel; "
            "the CXL DVSEC/HDM save/restore code path under test does not exist"
        )

    if (not glt_utils.is_cli_available("lspci")
            or not glt_utils.is_cli_available("setpci")):
        GLTCore.GLTTestResult.skip(
            "lspci/setpci not available; install pciutils"
        )

    bdf = cxl_utils.find_first_cxl_device_dvsec_bdf()
    if bdf is None:
        GLTCore.GLTTestResult.skip(
            "No PCI device exposes a CXL Device DVSEC (Vendor=0x1E98, ID=0x0000)"
        )
    logger.info("Candidate CXL endpoint: %s", bdf)

    if cxl_utils.any_cxl_region_committed():
        GLTCore.GLTTestResult.skip(
            "CXL regions are committed/online, "
            "refusing to reset a device with active mappings"
        )

    # Stash cleanup state BEFORE the first reset_method write so any
    # subsequent skip/fail path's post-hook can restore the original method.
    reset_method_path = f"/sys/bus/pci/devices/{bdf}/reset_method"
    try:
        with open(reset_method_path, "r", encoding="utf-8") as f:
            orig_reset_method = f.read().strip()
    except OSError as e:
        GLTCore.GLTTestResult.skip(
            f"Could not read reset_method on {bdf}: {e}"
        )
    logger.info("Original reset_method on %s: %r", bdf, orig_reset_method)

    test_cxl_save_restore_across_cxl_bus_reset.attr.cleanup_state = {
        "bdf": bdf,
        "orig_reset_method": orig_reset_method,
    }

    try:
        with open(reset_method_path, "w", encoding="utf-8") as f:
            f.write("cxl_bus")
    except OSError as e:
        GLTCore.GLTTestResult.skip(
            f"cxl_bus not selectable as reset_method on {bdf}: {e}"
        )
    try:
        with open(reset_method_path, "r", encoding="utf-8") as f:
            selected = f.read().strip()
    except OSError as e:
        GLTCore.GLTTestResult.skip(
            f"Could not read back reset_method on {bdf}: {e}"
        )
    if selected != "cxl_bus":
        GLTCore.GLTTestResult.skip(
            f"Could not select cxl_bus reset method on {bdf} "
            f"(got {selected!r}); kernel does not support the reset path"
        )

    dvsec_off = cxl_utils.find_cxl_device_dvsec_base(bdf)
    if dvsec_off is None:
        # The candidate was discovered via this exact DVSEC; a None here
        # means the device disappeared between discovery and now.
        GLTCore.GLTTestResult.fail(
            f"CXL Device DVSEC not found on {bdf} (re-discovery failed)"
        )
    logger.info("CXL Device DVSEC base on %s: %#x", bdf, dvsec_off)

    comp_regs = cxl_utils.find_cxl_component_regs(bdf)
    if comp_regs is None:
        GLTCore.GLTTestResult.skip(
            f"Register Locator DVSEC not found / unparseable on {bdf}; "
            "cannot locate component registers to snapshot HDM state"
        )
    comp_bar, comp_off = comp_regs

    try:
        pre_dvsec = _snapshot_dvsec(bdf, dvsec_off)
    except OSError as e:
        # /sys/.../config unreadable: device disappeared between discovery
        # and now, permission denied, etc. Skip rather than crash.
        GLTCore.GLTTestResult.skip(
            f"Could not read PCI config on {bdf} for PRE DVSEC snapshot: {e}"
        )
    try:
        pre_hdm = _snapshot_hdm(bdf, comp_bar, comp_off)
    except OSError as e:
        # BAR not assigned, permissions, etc. Treat as skip per spec.
        GLTCore.GLTTestResult.skip(
            f"Could not mmap BAR{comp_bar} on {bdf} for PRE HDM snapshot: {e}"
        )
    if not pre_hdm:
        GLTCore.GLTTestResult.skip(
            f"HDM Decoder cap absent in CM cap array on {bdf}; "
            "nothing for the save/restore patches to preserve"
        )
    logger.info("PRE DVSEC snapshot on %s: %s", bdf, pre_dvsec)
    logger.info("PRE HDM snapshot on %s: %s", bdf, pre_hdm)

    # Past this point a failure leaves the CXL link in an indeterminate
    # state; record the progress key so postfail/postafail triggers a reboot.
    sd.progress_record_time(
        test_cxl_save_restore_across_cxl_bus_reset.__name__,
        "CxlBusResetTriggered",
    )
    logger.info("Triggering cxl_bus reset on %s", bdf)
    try:
        with open(
            f"/sys/bus/pci/devices/{bdf}/reset", "w", encoding="utf-8",
        ) as f:
            f.write("1")
    except OSError as e:
        GLTCore.GLTTestResult.fail(f"cxl_bus reset on {bdf} failed: {e}")

    sysfs_path = f"/sys/bus/pci/devices/{bdf}"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if os.path.exists(sysfs_path):
            break
        time.sleep(0.1)
    if not os.path.exists(sysfs_path):
        GLTCore.GLTTestResult.fail(
            f"Device {bdf} did not reappear under sysfs within 5s "
            "after cxl_bus reset"
        )

    try:
        post_dvsec = _snapshot_dvsec(bdf, dvsec_off)
    except OSError as e:
        GLTCore.GLTTestResult.fail(
            f"Could not read PCI config on {bdf} for POST DVSEC snapshot: {e}"
        )
    try:
        post_hdm = _snapshot_hdm(bdf, comp_bar, comp_off)
    except OSError as e:
        GLTCore.GLTTestResult.fail(
            f"Could not mmap BAR{comp_bar} on {bdf} for POST HDM snapshot: {e}"
        )
    logger.info("POST DVSEC snapshot on %s: %s", bdf, post_dvsec)
    logger.info("POST HDM snapshot on %s: %s", bdf, post_hdm)

    if pre_dvsec != post_dvsec:
        GLTCore.GLTTestResult.fail(
            f"DVSEC drift across cxl_bus reset on {bdf}: "
            f"pre={pre_dvsec} post={post_dvsec}. "
            "Consider backporting the CXL DVSEC/HDM save/restore patch series "
            "(https://lore.kernel.org/linux-cxl/"
            "20260306080014.116789-1-smadhavan@nvidia.com/)"
        )
    if pre_hdm != post_hdm:
        GLTCore.GLTTestResult.fail(
            f"HDM drift across cxl_bus reset on {bdf}: "
            f"pre={pre_hdm} post={post_hdm}. "
            "Consider backporting the CXL DVSEC/HDM save/restore patch series "
            "(https://lore.kernel.org/linux-cxl/"
            "20260306080014.116789-1-smadhavan@nvidia.com/)"
        )
# pylint: enable=too-many-branches,too-many-statements,too-many-locals


test_cxl_save_restore_across_cxl_bus_reset.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_restore_reset_method,
    postskip=_restore_reset_method,
    postfail=_cxl_bus_reset_reboot_hook,
    postafail=_cxl_bus_reset_reboot_hook,
)


# A failed cxl_reset can leave the device's CXL state machine in an
# indeterminate place; a reboot is the only reliable recovery. The
# for_conditional_reboot pattern with a `lambda: True` condition is
# effectively unconditional once the gating progress key is recorded
# (just before the write to the cxl_reset sysfs attribute). Failures
# before that point leave the system in a clean state and do not trigger
# a reboot.
_cxl_reset_reboot_hook = GLTCore.PostHooks.for_conditional_reboot(
    "CxlResetTriggered",
    lambda: True,
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
# pylint: disable=too-many-branches,too-many-statements
def test_cxl_reset_dvsec_sysfs(sd, logger):
    """
    Exercise the cxl_reset sysfs attribute (in-band CXL DVSEC reset).

    Verifies kernel gating (file presence matches RST_CAPABLE) and that
    writing the file triggers the DVSEC reset sequence - not SBR - and
    completes with the expected STATUS2 bits.

    JIRA: DGX-16578
    PATCH:
    https://lore.kernel.org/linux-cxl/20260306092322.148765-1-smadhavan@nvidia.com/
    """

    if not glt_utils.is_cli_available("lspci"):
        GLTCore.GLTTestResult.skip("lspci not available; install pciutils")

    bdf = cxl_utils.find_first_cxl_device_dvsec_bdf()
    if bdf is None:
        GLTCore.GLTTestResult.skip(
            "No PCI device exposes a CXL Device DVSEC "
            "(Vendor=0x1E98, ID=0x0000)"
        )
    logger.info("Candidate CXL endpoint: %s", bdf)

    if cxl_utils.any_cxl_region_committed():
        GLTCore.GLTTestResult.skip(
            "CXL regions are committed/online, "
            "refusing to reset a device with active mappings"
        )

    dvsec_off = cxl_utils.find_cxl_device_dvsec_base(bdf)

    cap = cxl_utils.read_cxl_device_dvsec_cap_bits(bdf, dvsec_off)
    if cap is None:
        GLTCore.GLTTestResult.fail(
            f"Could not read CXL Device DVSEC CAP on {bdf}"
        )
    logger.info(
        "CXL Device DVSEC base on %s: %#x; CAP bits: %s",
        bdf, dvsec_off, cap,
    )

    if not cap["rst_capable"]:
        GLTCore.GLTTestResult.skip(
            f"{bdf} does not advertise RST_CAPABLE; nothing to exercise"
        )

    # Kernel gating: with RST_CAPABLE=1, the kernel must have exposed the
    # cxl_reset sysfs attribute.
    sysfs_path = f"/sys/bus/pci/devices/{bdf}/cxl_reset"
    if not os.path.exists(sysfs_path):
        GLTCore.GLTTestResult.fail(
            f"RST_CAPABLE set on {bdf} but cxl_reset attribute is missing. "
            "The most likely cause is that the kernel under test lacks the "
            "cxl_reset patch series; consider backporting it "
            "(https://lore.kernel.org/linux-cxl/"
            "20260306092322.148765-1-smadhavan@nvidia.com/)"
        )

    # Past this point a failure can leave the CXL state machine in an
    # indeterminate state.
    sd.progress_record_time(
        test_cxl_reset_dvsec_sysfs.__name__,
        "CxlResetTriggered",
    )
    logger.info("Triggering cxl_reset on %s", bdf)
    try:
        with open(sysfs_path, "w", encoding="utf-8") as f:
            f.write("1")
    except OSError as e:
        GLTCore.GLTTestResult.fail(
            f"write to {sysfs_path} failed: {e}"
        )

    time.sleep(0.1)

    s2 = cxl_utils.read_cxl_device_dvsec_status2_bits(bdf, dvsec_off)
    if s2 is None:
        GLTCore.GLTTestResult.fail(
            f"Could not read CXL Device DVSEC STATUS2 on {bdf} after write"
        )

    if not s2["rst_done"]:
        GLTCore.GLTTestResult.fail(
            f"RST_DONE not set in STATUS2 on {bdf} after successful "
            "cxl_reset write"
        )
    if s2["rst_err"]:
        GLTCore.GLTTestResult.fail(
            f"RST_ERR set in STATUS2 on {bdf} after cxl_reset"
        )
    if cap["cache_wbi_capable"] and not s2["cache_inv"]:
        GLTCore.GLTTestResult.fail(
            f"{bdf} advertised CACHE_WBI_CAPABLE but CACHE_INV was not "
            "set in STATUS2 after reset - WB+I phase did not run"
        )

    if not os.path.exists(f"/sys/bus/pci/devices/{bdf}"):
        GLTCore.GLTTestResult.fail(
            f"device {bdf} disappeared from sysfs after cxl_reset"
        )
# pylint: enable=too-many-branches,too-many-statements


test_cxl_reset_dvsec_sysfs.attr.post_hooks = GLTCore.PostHooks.create(
    postfail=_cxl_reset_reboot_hook,
    postafail=_cxl_reset_reboot_hook,
)
