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

"""NVGrace GPU VFIO driver checks."""

from pathlib import Path
import platform
import re

from common import cxl_utils
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


def _find_nvgrace_override():
    """Return (matching device or None, error message or None)."""
    try:
        for device in Path("/sys/bus/pci/devices").iterdir():
            override = (device / "driver_override").read_text(encoding="utf-8").strip()
            if override == "nvgrace_gpu_vfio_pci":
                return device.name, None
    except (OSError, UnicodeError) as error:
        return None, f"Cannot inspect PCI driver_override before module load: {error}"
    return None, None


def _module_is_loaded(module_name):
    """Check an exact lsmod name; return (loaded or None, error or None)."""
    try:
        result = glt_utils.run_command(["lsmod"], "Cannot inspect loaded modules")
    except RuntimeError as error:
        return None, str(error)
    loaded = any(
        fields and fields[0] == module_name
        for fields in (line.split() for line in result.stdout.splitlines())
    )
    return loaded, None


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvgrace_gpu_cxl_readiness(_, logger):
    """Check that the NVGrace VFIO driver contains CXL readiness support.

    Reference: https://lore.kernel.org/r/20260602063015.3915-1-ankita@nvidia.com
    Check symbol presence only when an NVIDIA GPU exposes a CXL Device DVSEC;
    non-CXL GPUs use the legacy readiness path and are outside this check.
    Load a modular driver only after checking PCI driver overrides. This
    preflight assumes no concurrent changes to PCI devices or overrides.
    """
    config_name = "CONFIG_NVGRACE_GPU_VFIO_PCI"
    module_field = "[nvgrace_gpu_vfio_pci]"
    patch_commit = "682ecb14e83840e87ea36c6d7c16c5111ce18784"
    symbol = "nvgrace_gpu_wait_device_ready_cxl"
    symbol_pattern = re.compile(rf"{re.escape(symbol)}(?:\.isra\.\d+)?")
    # 1. Establish whether the driver is modular or built in.
    config_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(config_path)
    except OSError as error:
        GLTCore.GLTTestResult.skip(
            f"Cannot inspect kernel configuration {config_path}: {error}"
        )

    config = kconfig.get(config_name)
    if config not in ("m", "y"):
        GLTCore.GLTTestResult.skip(
            f"{config_name} is not enabled"
        )

    gpu_bdfs = NvDev.get_nvidia_3d_bdf().split()
    if not gpu_bdfs:
        GLTCore.GLTTestResult.skip("No NVIDIA 3D GPU is present")
    if not any(
        cxl_utils.find_cxl_device_dvsec_base(bdf) is not None
        for bdf in gpu_bdfs
    ):
        GLTCore.GLTTestResult.skip(
            "No NVIDIA 3D GPU exposes a CXL Device DVSEC"
        )

    module_fields = [module_field] if config == "m" else []
    # 2. Built-in drivers do not appear in lsmod.
    module_loaded, error = (
        (True, None) if config == "y" else _module_is_loaded("nvgrace_gpu_vfio_pci")
    )
    if error:
        GLTCore.GLTTestResult.skip(error)

    # 3. Check all PCI overrides before registering an unloaded driver.
    if not module_loaded:
        # Overrides can match devices outside the driver's static ID table.
        override_device, error = _find_nvgrace_override()
        if error:
            GLTCore.GLTTestResult.skip(error)

        if override_device is not None:
            GLTCore.GLTTestResult.skip(
                f"{override_device} driver_override selects NVGrace VFIO; "
                "loading the module could bind a device"
            )

        try:
            glt_utils.run_command(
                ["modprobe", "nvgrace-gpu-vfio-pci"],
                "Cannot load NVGrace VFIO driver",
            )
        except RuntimeError as error:
            GLTCore.GLTTestResult.skip(str(error))

    try:
        with open(
            "/proc/kallsyms", encoding="utf-8", errors="ignore"
        ) as kallsyms:
            found = any(
                len(fields) >= 3
                and symbol_pattern.fullmatch(fields[2])
                and fields[3:] == module_fields
                for fields in (line.split() for line in kallsyms)
            )
    except OSError as error:
        GLTCore.GLTTestResult.skip(f"Cannot inspect /proc/kallsyms: {error}")

    if not found:
        GLTCore.GLTTestResult.fail(
            f"{symbol} is absent from the configured NVGrace VFIO driver; "
            f"consider checking for upstream commit {patch_commit} or an "
            "equivalent backport."
        )

    logger.info("Found %s in the configured NVGrace VFIO driver", symbol)
