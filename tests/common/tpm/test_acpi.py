# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable ACPI TPM2 identity parsing, independent of Linux FF-A discovery."""

import struct
from unittest import mock

import pytest

from common.tpm.acpi import read_tpm2_ffa_partition
from tests.common.tpm import make_tpm2_table


def test_decodes_firmware_id_in_acpi_byte_order():
    """The wire-format ID is little-endian even on a big-endian consumer."""
    table = make_tpm2_table(partition_id=0x8123)
    assert table[54:56] == b"\x23\x81"
    with mock.patch("common.tpm.acpi.Path.read_bytes", return_value=table):
        assert read_tpm2_ffa_partition("/unit/TPM2") == 0x8123


def test_other_start_method_has_no_ffa_identity():
    """A valid non-FF-A TPM interface supplies no expected FF-A partition."""
    with mock.patch("common.tpm.acpi.Path.read_bytes", return_value=make_tpm2_table(start_method=7)):
        assert read_tpm2_ffa_partition("/unit/TPM2") is None


@pytest.mark.parametrize("damage,expected", [
    ("signature", "invalid signature"),
    ("length", "does not match"),
    ("checksum", "checksum is invalid"),
    ("parameters", "parameters are truncated"),
])
def test_rejects_corrupt_advertised_identity(damage, expected):
    """Damaged firmware data cannot become an independently trusted identity."""
    table = bytearray(make_tpm2_table(length=52 if damage == "parameters" else 64))
    if damage == "signature":
        table[:4] = b"NOPE"
    elif damage == "length":
        struct.pack_into("<I", table, 4, len(table) + 1)
    elif damage == "checksum":
        table[10] ^= 1
    with mock.patch("common.tpm.acpi.Path.read_bytes", return_value=bytes(table)), \
            pytest.raises(ValueError, match=expected):
        read_tpm2_ffa_partition("/unit/TPM2")


def test_preserves_absent_firmware_table_error():
    """Callers can distinguish absent optional ACPI from malformed advertised data."""
    with mock.patch("common.tpm.acpi.Path.read_bytes", side_effect=FileNotFoundError), \
            pytest.raises(FileNotFoundError):
        read_tpm2_ffa_partition("/unit/TPM2")
