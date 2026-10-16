# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked timeout contracts for callers that opt into bounded PCI discovery."""

import subprocess
from unittest.mock import MagicMock

from common.pci_utils import EXT_CAP_ACS, get_bdfs_by_pci_id, get_bdfs_with_ext_cap
from tests.common.test_pci_utils import LSPCI_OUTPUT_TWO_DEVICES
from tests.test_base import GLTTestBase


class TestPciDiscoveryTimeout(GLTTestBase):
    """Keep timeout propagation separate from default PCI discovery behavior."""

    def _mock_lspci(self, stdout="", returncode=0):
        """Create a mock subprocess.run result for lspci."""
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.returncode = returncode
        return mock_result

    def test_discovery_timeout_is_optional_and_forwarded(self):
        """Bounded callers opt in without changing discovery results/default callers."""
        for helper, args, output, expected in (
                (get_bdfs_with_ext_cap, (EXT_CAP_ACS,), LSPCI_OUTPUT_TWO_DEVICES, ['0000:00:01.0']),
                (get_bdfs_by_pci_id, ('10de',), '0000:02:00.0 NVIDIA\n', ['0000:02:00.0'])):
            for timeout in (None, 10):
                with self.subTest(helper=helper.__name__, timeout=timeout), self.mock.patch(
                        'common.pci_utils.subprocess.run',
                        return_value=self._mock_lspci(stdout=output)) as command:
                    result = helper(*args) if timeout is None else helper(*args, timeout=timeout)
                    self.assertEqual(result, expected)
                    self.assertEqual(command.call_args.kwargs.get('timeout'), timeout)

    def test_discovery_timeout_propagates_without_retry(self):
        """The worker needs actual timeout exceptions to classify pre-ACK refusal."""
        for helper, args in ((get_bdfs_with_ext_cap, (EXT_CAP_ACS,)),
                             (get_bdfs_by_pci_id, ('10de',))):
            with self.subTest(helper=helper.__name__), self.mock.patch(
                    'common.pci_utils.subprocess.run',
                    side_effect=subprocess.TimeoutExpired(['lspci'], 10)) as command:
                with self.assertRaises(subprocess.TimeoutExpired):
                    helper(*args, timeout=10)
                command.assert_called_once()
