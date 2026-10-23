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

"""tegra_get_chip_id() calls ARM SMCCC for the running task."""

import os
import platform
import sys
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config

# drivers/soc/tegra/fuse/tegra-apbmisc.c
SERIES_SHA = "8b8ee2e56f951ccf41d98eebe73195cea487cd48"
PROBE_MODULE = "tegra_smccc_chipid_probe"
PROBE_PARAMETERS = f"/sys/module/{PROBE_MODULE}/parameters"
FUSE_ACPI_MODALIAS = "acpi:NVDA200F:"
COMMAND_TIMEOUT_SECS = 120
_REQUIRED_CONFIGS = (
    "CONFIG_KPROBES",
    "CONFIG_HAVE_ARM_SMCCC_DISCOVERY",
)


def _chip_id_from_smccc_soc_id(soc_id):
    """Return ``(implementation >> 4) & 0xff`` (``TEGRA_SMCCC_CHIP_ID``)."""
    parts = soc_id.strip().split(":")
    if len(parts) != 3 or parts[0] != "jep106" or len(parts[2]) != 4:
        return None
    try:
        implementation = int(parts[2], 16)
    except ValueError:
        return None
    return (implementation >> 4) & 0xFF


def _smccc_chip_id():
    chips = []
    root = Path("/sys/devices")
    if not root.is_dir():
        return None, "sysfs devices directory is absent"
    for path in sorted(root.glob("soc*/soc_id")):
        try:
            text = glt_utils.read_sysfs_value(str(path))
        except OSError as error:
            return None, f"cannot read {path}: {error}"
        chip = _chip_id_from_smccc_soc_id(text)
        if chip is not None:
            chips.append(chip)
    if not chips:
        return None, "SMCCC SoC id is not available"
    if len(set(chips)) != 1:
        return None, f"ambiguous SMCCC chip ids: {sorted(set(chips))}"
    return chips[0], ""


def _fuse_device_present():
    root = Path("/sys/bus/platform/devices")
    if not root.is_dir():
        return False, "platform device sysfs is absent"
    try:
        devices = sorted(root.iterdir())
    except OSError as error:
        return False, f"cannot list {root}: {error}"
    for device in devices:
        modalias = device / "modalias"
        if not modalias.is_file():
            continue
        try:
            text = glt_utils.read_sysfs_value(str(modalias))
        except OSError as error:
            return False, f"cannot read {modalias}: {error}"
        if text == FUSE_ACPI_MODALIAS:
            return True, ""
    return False, "ACPI NVDA200F fuse device is absent"


def _precheck(release):  # pylint: disable=too-many-return-statements
    """Return why this kernel cannot run the probe, or an empty string."""
    if platform.machine() not in {"aarch64", "arm64"}:
        return "SMCCC chip-id probe is aarch64-only"
    nvdev = NvDev()
    if not nvdev.is_cpu_type("grace") and not nvdev.is_cpu_type("vera"):
        return "SMCCC chip-id probe is only supported on Grace and Vera"

    config_path = Path("/boot") / f"config-{release}"
    if not config_path.is_file():
        return f"{config_path} is absent"
    try:
        kconfig = check_kernel_config.parse_kconfig(str(config_path))
    except OSError as error:
        return f"cannot read {config_path}: {error}"
    missing = [
        name for name in _REQUIRED_CONFIGS if kconfig.get(name) not in ("y", "m")
    ]
    if missing:
        return "SMCCC chip-id probe requires " + ", ".join(missing)

    present, reason = _fuse_device_present()
    if reason and not present:
        return reason

    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return f"The '{command}' command is required to load the SMCCC probe."
    headers = Path("/lib/modules") / release / "build"
    if not headers.is_dir():
        return f"Matching running-kernel headers are unavailable: {headers}"
    if Path(f"/sys/module/{PROBE_MODULE}").exists():
        return f"{PROBE_MODULE} is already loaded"
    return ""


def _load_probe(logger):
    try:
        glt_utils.run_command(
            ["make"],
            "Failed to build the SMCCC chip-id probe",
            timeout=COMMAND_TIMEOUT_SECS,
        )
    except RuntimeError as error:
        return False, f"Cannot build the SMCCC chip-id probe: {error}"
    if not glt_utils.sign_kernel_module(f"{PROBE_MODULE}.ko", logger):
        return False, (
            f"Cannot sign {PROBE_MODULE}.ko. "
            "See docs/user-guides/secure-boot.md for setup."
        )
    try:
        glt_utils.run_command(
            ["insmod", f"{PROBE_MODULE}.ko"],
            "Cannot load the SMCCC chip-id probe",
            timeout=COMMAND_TIMEOUT_SECS,
        )
    except RuntimeError as error:
        loaded = Path(f"/sys/module/{PROBE_MODULE}").exists()
        return loaded, str(error)
    return True, ""


def _cleanup_probe(loaded, logger):
    errors = []
    if loaded and Path(f"/sys/module/{PROBE_MODULE}").exists():
        try:
            glt_utils.run_command(
                ["rmmod", PROBE_MODULE],
                f"Cannot remove {PROBE_MODULE}",
                timeout=COMMAND_TIMEOUT_SECS,
            )
        except RuntimeError as error:
            errors.append(str(error))
            logger.error(errors[-1])
    try:
        glt_utils.run_command(
            ["make", "clean"],
            "Failed to clean the SMCCC chip-id probe",
            timeout=COMMAND_TIMEOUT_SECS,
        )
    except RuntimeError as error:
        errors.append(f"Cannot clean the SMCCC chip-id probe: {error}")
        logger.error(errors[-1])
    return errors


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_tegra_smccc_chip_id(_, logger):
    """Require one tegra_get_chip_id() call to hit ARM SMCCC.

    The unfixed helper returns the HIDREV value cached at boot and does
    not call arm_smccc_get_soc_id_version(). A kprobe on that getter
    counts hits from the module's own task and exposes smccc_hits and
    chip_id as read-only module parameters.

    PATCH: 8b8ee2e56f951ccf41d98eebe73195cea487cd48
    """
    release = platform.uname().release
    reason = _precheck(release)
    if reason:
        GLTCore.GLTTestResult.skip(reason)
    expected, reason = _smccc_chip_id()
    if reason:
        GLTCore.GLTTestResult.skip(reason)

    module_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        PROBE_MODULE,
    )
    loaded = False
    try:
        with glt_utils.chdir_managed(module_dir):
            loaded, detail = _load_probe(logger)
            if detail:
                GLTCore.GLTTestResult.skip(detail)
            try:
                hits = int(
                    glt_utils.read_sysfs_value(f"{PROBE_PARAMETERS}/smccc_hits")
                )
                chip_id = int(
                    glt_utils.read_sysfs_value(f"{PROBE_PARAMETERS}/chip_id")
                )
                if hits < 0 or chip_id < 0:
                    raise ValueError("probe counters must be non-negative")
            except (OSError, ValueError) as error:
                GLTCore.GLTTestResult.warn(
                    f"Cannot read the SMCCC chip-id probe: {error}"
                )
            if hits == 0:
                GLTCore.GLTTestResult.fail(
                    "tegra_get_chip_id() did not call ARM SMCCC; the chip id "
                    "still comes from the HIDREV value cached at boot. "
                    f"Include {SERIES_SHA}."
                )
            if chip_id != expected:
                GLTCore.GLTTestResult.fail(
                    f"tegra_get_chip_id() returned {chip_id}, expected "
                    f"{expected} from the SMCCC SoC id. Include {SERIES_SHA}."
                )
    except OSError as error:
        GLTCore.GLTTestResult.skip(
            f"Cannot enter the SMCCC chip-id probe directory: {error}"
        )
    finally:
        cleanup_errors = []
        try:
            with glt_utils.chdir_managed(module_dir):
                cleanup_errors = _cleanup_probe(loaded, logger)
        except OSError as error:
            cleanup_errors.append(
                f"Cannot clean the SMCCC chip-id probe: {error}"
            )
        if cleanup_errors:
            pending = sys.exc_info()[1]
            prior = str(pending).strip() if pending else ""
            cleanup = "; ".join(cleanup_errors)
            if prior:
                cleanup = f"{prior}; additionally, {cleanup}"
            GLTCore.GLTTestResult.fail(cleanup)
