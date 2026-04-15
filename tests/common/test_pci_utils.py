"""Unit test script for common/pci_utils.py"""

from unittest.mock import MagicMock

from tests.test_base import GLTTestBase
from src.common.pci_utils import (
    EXT_CAP_ACS,
    get_bdfs_with_ext_cap,
)

LSPCI_OUTPUT_TWO_DEVICES = """\
0000:00:01.0 PCI bridge: NVIDIA Corporation Device 0001 (rev a1)
\tControl: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
\tCapabilities: [40] Express Root Port (Slot+), MSI 00
\tCapabilities: [2b0 v1] Access Control Services
\t\tACSCap:\tSrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd- EgressCtrl- DirectTrans-

0000:00:02.0 PCI bridge: NVIDIA Corporation Device 0002 (rev a1)
\tControl: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
\tCapabilities: [40] Express Root Port (Slot+), MSI 00
"""

LSPCI_OUTPUT_NO_ACS = """\
0000:00:01.0 PCI bridge: NVIDIA Corporation Device 0001 (rev a1)
\tCapabilities: [40] Express Root Port (Slot+), MSI 00

0000:00:02.0 PCI bridge: NVIDIA Corporation Device 0002 (rev a1)
\tCapabilities: [40] Express Root Port (Slot+), MSI 00
"""

LSPCI_OUTPUT_MULTIPLE_ACS = """\
0000:00:01.0 PCI bridge: NVIDIA Corporation Device 0001 (rev a1)
\tCapabilities: [2b0 v1] Access Control Services

0000:00:02.0 PCI bridge: NVIDIA Corporation Device 0002 (rev a1)
\tCapabilities: [40] Express Root Port (Slot+), MSI 00

0000:00:03.0 PCI bridge: NVIDIA Corporation Device 0003 (rev a1)
\tCapabilities: [2b0 v1] Access Control Services
"""


class TestPciUtils(GLTTestBase):
    """Test Class for PCI utilities"""

    def _mock_lspci(self, stdout="", returncode=0):
        """Create a mock subprocess.run result for lspci."""
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.returncode = returncode
        return mock_result

    def test_get_bdfs_with_ext_cap_found(self):
        """Returns BDFs that have the requested capability."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_TWO_DEVICES),
        ):
            result = get_bdfs_with_ext_cap(EXT_CAP_ACS)
        self.assertEqual(result, ["0000:00:01.0"])

    def test_get_bdfs_with_ext_cap_not_found(self):
        """Returns empty list when no device has the capability."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_NO_ACS),
        ):
            result = get_bdfs_with_ext_cap(EXT_CAP_ACS)
        self.assertEqual(result, [])

    def test_get_bdfs_with_ext_cap_multiple_devices(self):
        """Returns multiple BDFs when several devices have the capability."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_MULTIPLE_ACS),
        ):
            result = get_bdfs_with_ext_cap(EXT_CAP_ACS)
        self.assertEqual(result, ["0000:00:01.0", "0000:00:03.0"])

    def test_get_bdfs_with_ext_cap_lspci_failure(self):
        """Returns empty list when lspci exits with non-zero."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(returncode=1),
        ):
            result = get_bdfs_with_ext_cap(EXT_CAP_ACS)
        self.assertEqual(result, [])

    def test_get_bdfs_with_ext_cap_lspci_missing(self):
        """Returns empty list when lspci is not installed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = get_bdfs_with_ext_cap(EXT_CAP_ACS)
        self.assertEqual(result, [])


if __name__ == "__main__":
    GLTTestBase.unittest.main()
