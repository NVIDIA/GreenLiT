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

"""Read the firmware-advertised TPM2 FF-A identity independently of Linux discovery."""

import struct
from pathlib import Path


def read_tpm2_ffa_partition(table_path):
    """Validate an ACPI TPM2 table and return its FF-A partition ID, or None.

    ACPI fields are little-endian regardless of the running kernel's endian.
    Return None for valid tables using other start methods. File access errors
    propagate; malformed advertised tables raise ValueError.
    """
    table = Path(table_path).read_bytes()
    if len(table) < 52 or table[:4] != b"TPM2":
        raise ValueError("TPM2 ACPI table has an invalid signature or truncated fixed header")
    length = struct.unpack_from("<I", table, 4)[0]
    if length != len(table):
        raise ValueError(f"TPM2 ACPI table length {length} does not match {len(table)} bytes read")
    if sum(table) % 256:
        raise ValueError("TPM2 ACPI table checksum is invalid")
    start_method = struct.unpack_from("<I", table, 48)[0]
    if start_method != 15:
        return None
    if length < 64:
        raise ValueError("TPM2 ACPI FF-A start method parameters are truncated")
    # struct tpm2_crb_ffa starts at offset 52: flags, attributes, then u16 partition ID.
    return struct.unpack_from("<H", table, 54)[0]
