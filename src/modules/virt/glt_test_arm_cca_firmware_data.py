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

"""Verify running-kernel fingerprints for Arm CCA firmware data on Vera."""

import platform
from pathlib import Path

from common import glt_core as GLTCore
from common.nv_device import NvDev
from scripts import check_kernel_config


KALLSYMS_PATH = Path("/proc/kallsyms")
PATCHED_RSI_SYMBOL = "arm64_rsi_is_protected"
PREPATCH_RSI_SYMBOL = "__arm64_is_protected_mmio"

EFI_SECRET_CONFIG = "CONFIG_EFI_SECRET"
EFI_COCO_SECRET_CONFIG = "CONFIG_EFI_COCO_SECRET"
SECURITYFS_CONFIG = "CONFIG_SECURITYFS"

REALM_ENCRYPTED_MAPPING_COMMIT = "fa84e534c3ec2904d8718a83180294f7b5afecc7"
ARM64_EFI_SECRET_COMMIT = "9e8a3df3e7f762966762a6fbf3282b9da2074127"

# d02c2e45b1e7767b177f6854026e4ad0d70b4a4d added an arm64 read-only
# EFI_ACPI_MEMORY_NVS mapping for CCEL, but it introduced no stable procfs,
# sysfs, symbol, or Kconfig discriminator. It was subsequently reverted by
# 9238fa35a3c04835a77988a14e1397763cf0eabe because other NVS users, including
# GHES error acknowledgement, need writable mappings. Safely distinguishing
# those states requires a controlled CCEL/GHES fixture or build provenance, so
# these running-kernel tests intentionally do not validate that change.


def _read_kernel_text_symbols(path=KALLSYMS_PATH):
    """Return exact text-symbol names from a readable kallsyms file.

    Raises:
        OSError: if the kallsyms file cannot be opened or read.
        UnicodeError: if the kallsyms file is not valid UTF-8 text.
        ValueError: if the file contains no recognizable symbol records.
    """
    symbols = set()
    valid_records = 0

    with path.open(encoding="utf-8") as kallsyms:
        for line in kallsyms:
            fields = line.split()
            if len(fields) < 3 or len(fields[1]) != 1:
                continue
            try:
                int(fields[0], 16)
            except ValueError:
                continue
            valid_records += 1
            if fields[1] in ("T", "t"):
                symbols.add(fields[2])

    if valid_records == 0:
        raise ValueError(f"{path} contains no recognizable symbol records")
    return symbols


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_cca_efi_secret_kernel_config(_, logger):
    """
    Verify that the Vera kernel config enables ARM64 EFI secret support.

    PATCH: https://git.kernel.org/torvalds/c/9e8a3df3e7f7

    This checks the exact-release config artifact, not Git ancestry, module
    installation/loading, firmware data, or CCEL/NVS mapping behavior.
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    config_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(config_path)
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot read exact-release kernel config {config_path}: {error}; "
            f"check for {ARM64_EFI_SECRET_COMMIT} or an equivalent downstream "
            "backport."
        )

    required_values = {
        EFI_SECRET_CONFIG: ("y", "m"),
        EFI_COCO_SECRET_CONFIG: ("y",),
        SECURITYFS_CONFIG: ("y",),
    }
    invalid_configs = [
        f"{name}={kconfig.get(name)!r} (expected {' or '.join(allowed)})"
        for name, allowed in required_values.items()
        if kconfig.get(name) not in allowed
    ]
    if invalid_configs:
        GLTCore.GLTTestResult.fail(
            "Arm CCA EFI secret-area support is not configured: "
            + ", ".join(invalid_configs)
            + f"; check for {ARM64_EFI_SECRET_COMMIT} or an equivalent downstream "
            "backport."
        )

    logger.info(
        "Exact-release kernel config enables EFI secret support: "
        "%s=%s, %s=y, %s=y",
        EFI_SECRET_CONFIG,
        kconfig[EFI_SECRET_CONFIG],
        EFI_COCO_SECRET_CONFIG,
        SECURITYFS_CONFIG,
    )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_arm_cca_realm_encrypted_mapping_kernel_fingerprint(_, logger):
    """
    Check the running Vera kernel for the protected Realm mapping fingerprint.

    PATCH: https://git.kernel.org/torvalds/c/fa84e534c3ec

    The exported RSI symbol is a strong running-image fingerprint, but symbol
    inspection does not prove Git ancestry or the compiled function body.
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("The CPU is not Vera.")

    try:
        kernel_symbols = _read_kernel_text_symbols()
    except (OSError, UnicodeError, ValueError) as error:
        GLTCore.GLTTestResult.skip(
            f"Cannot inspect running-kernel symbols in {KALLSYMS_PATH}: {error}."
        )

    if PATCHED_RSI_SYMBOL not in kernel_symbols:
        if PREPATCH_RSI_SYMBOL in kernel_symbols:
            detail = f"only the prepatch symbol {PREPATCH_RSI_SYMBOL!r} is present"
        else:
            detail = f"required symbol {PATCHED_RSI_SYMBOL!r} is absent"
        GLTCore.GLTTestResult.fail(
            f"Arm CCA protected Realm mapping was not confirmed: {detail}; "
            f"check for {REALM_ENCRYPTED_MAPPING_COMMIT} or an equivalent "
            "downstream backport."
        )

    logger.info(
        "Running kernel exposes the Arm CCA RSI symbol: %s",
        PATCHED_RSI_SYMBOL,
    )
    if PREPATCH_RSI_SYMBOL in kernel_symbols:
        logger.warning(
            "Running kernel also exposes legacy RSI symbol %s. The required "
            "fingerprint is present, but symbol inspection cannot determine "
            "whether the legacy export is only a compatibility alias.",
            PREPATCH_RSI_SYMBOL,
        )
