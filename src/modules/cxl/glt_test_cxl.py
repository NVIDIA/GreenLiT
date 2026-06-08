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
import os
import platform
import re
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
