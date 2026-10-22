# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared TPM helpers."""

import struct


def make_tpm2_table(partition_id=0x8001, start_method=15, length=64):
    """Build a checksum-correct synthetic ACPI TPM2 table."""
    table = bytearray(length)
    table[:4] = b"TPM2"
    struct.pack_into("<I", table, 4, length)
    struct.pack_into("<I", table, 48, start_method)
    if length >= 56:
        struct.pack_into("<H", table, 54, partition_id)
    table[9] = -sum(table) % 256
    return bytes(table)
