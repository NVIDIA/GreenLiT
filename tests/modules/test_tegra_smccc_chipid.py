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

"""Unit tests for the tegra_get_chip_id SMCCC probe verdict."""

import contextlib
from unittest import mock

from common.glt_core import GLTTestFail, GLTTestWarn
from tests.test_base import GLTTestBase
from src.modules.acpi import glt_test_tegra_smccc_chipid as smccc


class TestTegraSmcccChipId(GLTTestBase):
    """Parsing and verdict coverage for the SMCCC chip-id probe."""

    def test_grace_soc_id_is_tegra241(self):
        """jep106:036b:0241 encodes chip id 0x24."""
        self.assertEqual(
            smccc._chip_id_from_smccc_soc_id("jep106:036b:0241\n"),  # pylint: disable=protected-access
            0x24,
        )

    def test_rejects_non_smccc_soc_ids(self):
        """Only the four-digit jep106 form is the SMCCC observation."""
        self.assertIsNone(
            smccc._chip_id_from_smccc_soc_id("Tegra")  # pylint: disable=protected-access
        )
        self.assertIsNone(
            smccc._chip_id_from_smccc_soc_id("jep106:036b:241")  # pylint: disable=protected-access
        )

    def _run_with_parameters(self, values):
        logger = mock.MagicMock()
        with mock.patch.object(
            smccc, "_precheck", return_value=""
        ), mock.patch.object(
            smccc, "_smccc_chip_id", return_value=(0x24, "")
        ), mock.patch.object(
            smccc.glt_utils, "chdir_managed", return_value=contextlib.nullcontext()
        ), mock.patch.object(
            smccc, "_load_probe", return_value=(True, "")
        ), mock.patch.object(
            smccc, "_cleanup_probe", return_value=[]
        ), mock.patch.object(
            smccc.glt_utils, "read_sysfs_value", side_effect=values
        ):
            smccc.test_tegra_smccc_chip_id(None, logger)

    def test_passes_when_probe_records_hit_and_matching_chip_id(self):
        """Non-zero hits and matching chip_id pass."""
        self._run_with_parameters(["1", "36"])

    def test_fails_when_probe_records_zero_hits(self):
        """Zero hits mean tegra_get_chip_id() did not call ARM SMCCC."""
        with self.assertRaisesRegex(GLTTestFail, "did not call ARM SMCCC"):
            self._run_with_parameters(["0", "36"])

    def test_fails_when_chip_id_disagrees_with_smccc(self):
        """Mismatched chip_id fails even with non-zero hits."""
        with self.assertRaisesRegex(GLTTestFail, "returned 35, expected 36"):
            self._run_with_parameters(["1", "35"])

    def test_warns_when_module_parameter_cannot_be_read(self):
        """Unreadable module parameters after insmod surface as a warning."""
        with self.assertRaisesRegex(GLTTestWarn, "Cannot read the SMCCC chip-id probe"):
            self._run_with_parameters(OSError("missing parameter"))
