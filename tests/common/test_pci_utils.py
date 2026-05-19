"""Unit test script for common/pci_utils.py"""

from unittest.mock import MagicMock

from tests.test_base import GLTTestBase
from src.common.pci_utils import (
    ASPEED_AST1150_DEVICE_ID,
    ASPEED_VENDOR_ID,
    EXT_CAP_ACS,
    PCI_CLASS_USB_HOST,
    get_bdfs_by_pci_id,
    get_bdfs_with_ext_cap,
    get_pci_children,
    get_pci_class,
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

LSPCI_OUTPUT_AST1150 = """\
0000:e0:00.0 PCI bridge [0604]: ASPEED Technology, Inc. AST1150 PCI-to-PCI Bridge [1a03:1150] (rev 04)
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

    def test_get_bdfs_by_pci_id_found(self):
        """Returns BDFs for devices matching vendor and device IDs."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_AST1150),
        ) as run_mock:
            result = get_bdfs_by_pci_id(ASPEED_VENDOR_ID, ASPEED_AST1150_DEVICE_ID)
        self.assertEqual(result, ["0000:e0:00.0"])
        run_mock.assert_called_once_with(
            ["lspci", "-Dnn", "-d", "1a03:1150"],
            capture_output=True, text=True, check=False,
        )

    def test_get_bdfs_by_pci_id_vendor_only(self):
        """Returns BDFs for devices matching a vendor ID without device filter."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_AST1150),
        ) as run_mock:
            result = get_bdfs_by_pci_id(ASPEED_VENDOR_ID)
        self.assertEqual(result, ["0000:e0:00.0"])
        run_mock.assert_called_once_with(
            ["lspci", "-Dnn", "-d", "1a03:"],
            capture_output=True, text=True, check=False,
        )

    def test_get_bdfs_by_pci_id_not_found(self):
        """Returns empty list when no device matches the PCI ID."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=""),
        ):
            result = get_bdfs_by_pci_id(ASPEED_VENDOR_ID, ASPEED_AST1150_DEVICE_ID)
        self.assertEqual(result, [])

    def test_get_bdfs_by_pci_id_lspci_failure(self):
        """Returns empty list when lspci exits with non-zero."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(returncode=1),
        ):
            result = get_bdfs_by_pci_id(ASPEED_VENDOR_ID, ASPEED_AST1150_DEVICE_ID)
        self.assertEqual(result, [])

    def test_get_bdfs_by_pci_id_lspci_missing(self):
        """Returns empty list when lspci is missing."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = get_bdfs_by_pci_id(ASPEED_VENDOR_ID, ASPEED_AST1150_DEVICE_ID)
        self.assertEqual(result, [])

    def test_get_pci_children_returns_bdf_entries(self):
        """Returns only entries that look like BDFs, ignoring sysfs files."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            return_value=[
                "0000:e1:00.0",
                "0000:e1:00.1",
                "class",
                "vendor",
                "subsystem_device",
            ],
        ):
            result = get_pci_children("0000:e0:00.0")
        self.assertEqual(result, ["0000:e1:00.0", "0000:e1:00.1"])

    def test_get_pci_children_missing_bridge(self):
        """Returns empty list when the bridge sysfs node is absent."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            side_effect=FileNotFoundError,
        ):
            result = get_pci_children("0000:e0:00.0")
        self.assertEqual(result, [])

    def test_get_pci_class_returns_hex_string(self):
        """Returns the trimmed contents of the sysfs class file."""
        mock_file = GLTTestBase.unittest.mock.mock_open(read_data="0x0c0330\n")
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open", mock_file, create=True,
        ):
            result = get_pci_class("0000:e1:00.0")
        self.assertEqual(result, "0x0c0330")

    def test_get_pci_class_missing_file(self):
        """Returns None when the class file cannot be read."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=FileNotFoundError,
            create=True,
        ):
            result = get_pci_class("0000:e1:00.0")
        self.assertIsNone(result)

    def test_pci_class_usb_host_constant(self):
        """USB host controller class prefix matches the PCI spec."""
        self.assertEqual(PCI_CLASS_USB_HOST, "0x0c03")


if __name__ == "__main__":
    GLTTestBase.unittest.main()
