"""Unit test script for common/pci_utils.py"""

import os
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock

from tests.test_base import GLTTestBase
from src.common.pci_utils import (
    ASPEED_AST1150_DEVICE_ID,
    ASPEED_VENDOR_ID,
    BDF_STRICT_RE,
    EXT_CAP_ACS,
    EXT_CAP_ATS,
    PCI_CLASS_USB_HOST,
    bdf_in_translated_domain,
    block_device_physical_leaves,
    default_route_ifaces,
    get_bdfs_by_pci_id,
    get_bdfs_with_ext_cap,
    get_iommu_group_type,
    get_pci_children,
    get_pci_class,
    get_pci_parent,
    get_reset_methods,
    iface_pci_bdfs,
    is_ats_enabled,
    mounted_block_devices,
    pci_bdfs_backing_mount,
    read_config_word,
    unsafe_pci_bdfs,
    sysfs_path_to_pci_bdf,
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

LSPCI_OUTPUT_ATS = """\
0000:00:01.0 PCI bridge: NVIDIA Corporation Device 0001 (rev a1)
\tCapabilities: [40] Express Root Port (Slot+), MSI 00
\tCapabilities: [2b0 v1] Access Control Services

0000:01:00.0 3D controller: NVIDIA Corporation Device 0002 (rev a1)
\tCapabilities: [78] Express Endpoint, MSI 00
\tCapabilities: [148 v1] Address Translation Service (ATS)
\t\tATSCap: Invalidate Queue Depth: 00
\t\tATSCtl: Enable+, Smallest Translation Unit: 00
"""

LSPCI_OUTPUT_AST1150 = """\
0000:e0:00.0 PCI bridge [0604]: ASPEED Technology, Inc. AST1150 PCI-to-PCI Bridge [1a03:1150] (rev 04)
"""

PROC_NET_ROUTE_DEFAULT = """\
Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT
eth0\t00000000\t0102A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0
eth0\t0002A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\t0\t0\t0
"""

PROC_NET_ROUTE_NO_DEFAULT = """\
Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT
eth0\t0002A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\t0\t0\t0
"""

PROC_NET_IPV6_ROUTE_DEFAULT = (
    "00000000000000000000000000000000 00 "
    "00000000000000000000000000000000 00 "
    "fe800000000000000000000000000001 "
    "00000400 00000001 00000000 00000003 bond0\n"
    "fe800000000000000000000000000000 40 "
    "00000000000000000000000000000000 00 "
    "00000000000000000000000000000000 "
    "00000100 00000001 00000000 00000001 eth1\n"
)

PROC_MOUNTS = """\
proc /proc proc rw,nosuid 0 0
/dev/mapper/vg-root / ext4 rw,relatime 0 0
/dev/nvme0n1p1 /boot vfat rw,relatime 0 0
tmpfs /run tmpfs rw,nosuid 0 0
"""

LSPCI_OUTPUT_ATS_ENABLED = """\
0000:02:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:0001]
\tCapabilities: [3a8 v1] Address Translation Service (ATS)
\t\tATSCap: Invalidate Queue Depth: 00
\t\tATSCtl: Enable+, Smallest Translation Unit: 00
"""

LSPCI_OUTPUT_ATS_DISABLED = """\
0000:02:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:0001]
\tCapabilities: [3a8 v1] Address Translation Service (ATS)
\t\tATSCap: Invalidate Queue Depth: 00
\t\tATSCtl: Enable-, Smallest Translation Unit: 00
"""


class TestPciIommuGroupType(GLTTestBase):
    """Test the per-device IOMMU group type helper."""

    def test_get_iommu_group_type_returns_runtime_domain(self):
        """Returns the active per-device IOMMU group domain type."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            return_value="identity",
        ) as read_mock:
            result = get_iommu_group_type("0000:02:00.0")
        self.assertEqual(result, "identity")
        read_mock.assert_called_once_with(
            "/sys/bus/pci/devices/0000:02:00.0/iommu_group/type"
        )

    def test_get_iommu_group_type_unavailable(self):
        """Returns None when the device IOMMU group type is unavailable."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            side_effect=FileNotFoundError,
        ):
            result = get_iommu_group_type("0000:02:00.0")
        self.assertIsNone(result)

    def test_get_iommu_group_type_unreadable(self):
        """Returns None when permissions prevent validating the group type."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            side_effect=PermissionError,
        ):
            result = get_iommu_group_type("0000:02:00.0")
        self.assertIsNone(result)


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

    def test_get_bdfs_with_ext_cap_ats(self):
        """Returns only endpoints advertising Address Translation Service."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_ATS),
        ):
            result = get_bdfs_with_ext_cap(EXT_CAP_ATS)
        self.assertEqual(result, ["0000:01:00.0"])

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


class TestGetPciParent(GLTTestBase):
    """Tests for get_pci_parent() against a fake sysfs tree."""

    PARENT = "0000:00:01.0"
    CHILD = "0000:01:00.0"

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _link_pci_device(self, real_dir, bdf):
        bus_dir = os.path.join(self.tmpdir, "sys", "bus", "pci", "devices")
        os.makedirs(bus_dir, exist_ok=True)
        os.symlink(
            os.path.relpath(real_dir, bus_dir),
            os.path.join(bus_dir, bdf),
        )

    def test_returns_immediate_upstream_bridge(self):
        """A downstream device's sysfs parent BDF is the upstream bridge."""
        parent_dir = os.path.join(
            self.tmpdir, "sys", "devices", "pci0000:00", self.PARENT,
        )
        child_dir = os.path.join(parent_dir, self.CHILD)
        os.makedirs(child_dir)
        self._link_pci_device(parent_dir, self.PARENT)
        self._link_pci_device(child_dir, self.CHILD)

        parent, error = get_pci_parent(self.CHILD, sysfs_root=self.tmpdir)
        self.assertEqual(parent, self.PARENT)
        self.assertIsNone(error)

    def test_returns_none_for_root_bus_device(self):
        """A device whose sysfs parent is not a BDF has no upstream bridge."""
        parent_dir = os.path.join(
            self.tmpdir, "sys", "devices", "pci0000:00", self.PARENT,
        )
        os.makedirs(parent_dir)
        self._link_pci_device(parent_dir, self.PARENT)

        parent, error = get_pci_parent(self.PARENT, sysfs_root=self.tmpdir)
        self.assertIsNone(parent)
        self.assertIsNone(error)

    def test_reports_error_when_device_missing(self):
        """Missing sysfs entries are lookup failures, not root-bus devices."""
        parent, error = get_pci_parent(self.CHILD, sysfs_root=self.tmpdir)
        self.assertIsNone(parent)
        self.assertIsNotNone(error)
        self.assertIn("cannot resolve", error)
        self.assertIn(self.CHILD, error)

    def test_reports_error_for_malformed_bdf(self):
        """Rejects values that are not domain-prefixed BDFs."""
        parent, error = get_pci_parent("../etc/passwd", sysfs_root=self.tmpdir)
        self.assertIsNone(parent)
        self.assertIsNotNone(error)
        self.assertIn("malformed BDF", error)

    def test_reports_error_when_parent_missing_from_bus(self):
        """A BDF-shaped parent without a bus/pci/devices entry is unresolved."""
        parent_dir = os.path.join(
            self.tmpdir, "sys", "devices", "pci0000:00", self.PARENT,
        )
        child_dir = os.path.join(parent_dir, self.CHILD)
        os.makedirs(child_dir)
        self._link_pci_device(child_dir, self.CHILD)

        parent, error = get_pci_parent(self.CHILD, sysfs_root=self.tmpdir)
        self.assertIsNone(parent)
        self.assertIsNotNone(error)
        self.assertIn(self.PARENT, error)
        self.assertIn("missing from sysfs", error)


class TestGetResetMethods(GLTTestBase):
    """Tests for get_reset_methods()."""

    BDF = "0000:01:00.0"

    def test_splits_space_separated_tokens(self):
        """Returns each advertised reset method as its own token."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            return_value="flr bus cxl_bus",
        ) as read_mock:
            result = get_reset_methods(self.BDF)
        self.assertEqual(result, ["flr", "bus", "cxl_bus"])
        read_mock.assert_called_once_with(
            f"/sys/bus/pci/devices/{self.BDF}/reset_method"
        )

    def test_does_not_treat_cxl_bus_as_bus(self):
        """The cxl_bus method is distinct from the bus method."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            return_value="flr cxl_bus",
        ):
            result = get_reset_methods(self.BDF)
        self.assertEqual(result, ["flr", "cxl_bus"])
        self.assertNotIn("bus", result)

    def test_missing_file_returns_empty_list(self):
        """An absent reset_method file means no methods are advertised."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            side_effect=FileNotFoundError,
        ):
            self.assertEqual(get_reset_methods(self.BDF), [])

    def test_unreadable_file_returns_none(self):
        """Permission errors are unvalidated rather than empty."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            side_effect=PermissionError,
        ):
            self.assertIsNone(get_reset_methods(self.BDF))

    def test_empty_file_returns_empty_list(self):
        """A present but empty reset_method file advertises nothing."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            return_value="",
        ):
            self.assertEqual(get_reset_methods(self.BDF), [])


class TestReadConfigWord(GLTTestBase):
    """Tests for read_config_word() against a fake sysfs config file."""

    BDF = "0000:00:01.0"
    OFFSET = 0x0C
    VALUE = 0xABCD

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _write_config(self, size, word_off=None, word=None):
        device_dir = os.path.join(
            self.tmpdir, "sys", "bus", "pci", "devices", self.BDF,
        )
        os.makedirs(device_dir, exist_ok=True)
        buf = bytearray(size)
        if word_off is not None:
            buf[word_off] = word & 0xFF
            buf[word_off + 1] = (word >> 8) & 0xFF
        with open(os.path.join(device_dir, "config"), "wb") as cfg:
            cfg.write(buf)

    def test_reads_little_endian_word(self):
        """Returns the 16-bit little-endian value at the requested offset."""
        self._write_config(self.OFFSET + 2, self.OFFSET, self.VALUE)
        self.assertEqual(
            read_config_word(self.BDF, self.OFFSET, sysfs_root=self.tmpdir),
            self.VALUE,
        )

    def test_returns_none_when_config_missing(self):
        """Missing config file is unreadable rather than zero."""
        self.assertIsNone(
            read_config_word(self.BDF, self.OFFSET, sysfs_root=self.tmpdir)
        )

    def test_returns_none_on_short_read(self):
        """A file too short to cover the offset is unreadable."""
        self._write_config(self.OFFSET + 1)
        self.assertIsNone(
            read_config_word(self.BDF, self.OFFSET, sysfs_root=self.tmpdir)
        )

    def test_returns_none_for_malformed_bdf(self):
        """Rejects values that are not domain-prefixed BDFs."""
        self.assertIsNone(
            read_config_word("../etc/passwd", 0, sysfs_root=self.tmpdir)
        )

    def test_returns_none_for_negative_offset(self):
        """Negative config offsets are rejected before seeking."""
        self._write_config(2, 0, self.VALUE)
        self.assertIsNone(
            read_config_word(self.BDF, -1, sysfs_root=self.tmpdir)
        )

    def test_returns_none_on_permission_error(self):
        """OSError on open is unreadable rather than a raised exception."""
        self._write_config(self.OFFSET + 2, self.OFFSET, self.VALUE)
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=PermissionError("no perm"),
            create=True,
        ):
            self.assertIsNone(
                read_config_word(self.BDF, self.OFFSET, sysfs_root=self.tmpdir)
            )


class TestPciUtilsAts(GLTTestBase):
    """Test Class for the PCI ATS-state helper."""

    @staticmethod
    def _mock_lspci(stdout="", returncode=0):
        """Create a mock subprocess.run result for lspci."""
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.returncode = returncode
        return mock_result

    def test_is_ats_enabled_disabled(self):
        """Returns False when ATSCtl reports ATS disabled."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_ATS_DISABLED),
        ):
            result = is_ats_enabled("0000:02:00.0")
        self.assertFalse(result)

    def test_is_ats_enabled_lspci_failure(self):
        """Returns None when lspci exits with non-zero."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["lspci"]),
        ):
            result = is_ats_enabled("0000:02:00.0")
        self.assertIsNone(result)

    def test_is_ats_enabled_lspci_missing(self):
        """Returns None when lspci is missing."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = is_ats_enabled("0000:02:00.0")
        self.assertIsNone(result)

    def test_is_ats_enabled_missing_capability(self):
        """Returns None when the device lacks ATS capability output."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_NO_ACS),
        ):
            result = is_ats_enabled("0000:02:00.0")
        self.assertIsNone(result)

    def test_is_ats_enabled_true(self):
        """Returns True when ATSCtl reports ATS enabled."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.subprocess.run",
            return_value=self._mock_lspci(stdout=LSPCI_OUTPUT_ATS_ENABLED),
        ) as run_mock:
            result = is_ats_enabled("0000:02:00.0")
        self.assertTrue(result)
        run_mock.assert_called_once_with(
            ["lspci", "-vvv", "-s", "0000:02:00.0"],
            capture_output=True, text=True, check=True,
        )


class TestPciUtilsNetAndBlock(GLTTestBase):
    """Test Class for the network and block-device PCI helpers"""

    @staticmethod
    def _open_from_files(files):
        """Return an open() side effect serving read_data keyed by path."""
        def _opener(path, *args, **kwargs):  # pylint: disable=unused-argument
            if path not in files:
                raise FileNotFoundError(path)
            mock_file = GLTTestBase.unittest.mock.mock_open(
                read_data=files[path])
            return mock_file.return_value
        return _opener

    def test_default_route_ifaces_ipv4(self):
        """IPv4 default route yields its interface; other routes ignored."""
        files = {"/proc/net/route": PROC_NET_ROUTE_DEFAULT}
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=self._open_from_files(files),
            create=True,
        ):
            result = default_route_ifaces()
        self.assertEqual(result, {"eth0"})

    def test_default_route_ifaces_ipv6(self):
        """IPv6 all-zero destination with prefix 00 yields the last field."""
        files = {"/proc/net/ipv6_route": PROC_NET_IPV6_ROUTE_DEFAULT}
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=self._open_from_files(files),
            create=True,
        ):
            result = default_route_ifaces()
        self.assertEqual(result, {"bond0"})

    def test_default_route_ifaces_no_default(self):
        """Non-default routes do not contribute any interface."""
        files = {"/proc/net/route": PROC_NET_ROUTE_NO_DEFAULT}
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=self._open_from_files(files),
            create=True,
        ):
            result = default_route_ifaces()
        self.assertEqual(result, set())

    def test_default_route_ifaces_unreadable(self):
        """Unreadable route tables raise so callers can fail closed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=OSError,
            create=True,
        ):
            with self.assertRaises(OSError):
                default_route_ifaces()

    def test_iface_pci_bdfs_direct(self):
        """A directly PCI-backed interface maps to its lowercased BDF."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.realpath",
            return_value="/sys/devices/pci0000:00/0000:0A:00.0",
        ):
            result = iface_pci_bdfs("eth0")
        self.assertEqual(result, {"0000:0a:00.0"})

    def test_iface_pci_bdfs_non_pci(self):
        """A non-PCI interface with no lowers yields an empty set."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.realpath",
            return_value="/sys/class/net/lo/device",
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            return_value=["mtu", "operstate"],
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            side_effect=OSError,
        ):
            result = iface_pci_bdfs("lo")
        self.assertEqual(result, set())

    def test_iface_pci_bdfs_bond(self):
        """A bond descends lower_* and bonding/slaves to physical BDFs."""
        realpaths = {
            "/sys/class/net/bond0/device": "/sys/class/net/bond0/device",
            "/sys/class/net/eth0/device":
                "/sys/devices/pci0000:00/0000:01:00.0",
            "/sys/class/net/eth1/device":
                "/sys/devices/pci0000:00/0000:02:00.0",
        }

        def _listdir(path):
            if path == "/sys/class/net/bond0":
                return ["lower_eth0", "bonding", "mtu"]
            return []

        def _read_sysfs(path):
            if path == "/sys/class/net/bond0/bonding/slaves":
                return "eth0 eth1"
            raise OSError(path)

        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.realpath",
            side_effect=lambda path: realpaths.get(path, path),
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            side_effect=_listdir,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            side_effect=_read_sysfs,
        ):
            result = iface_pci_bdfs("bond0")
        self.assertEqual(result, {"0000:01:00.0", "0000:02:00.0"})

    def test_iface_pci_bdfs_recursion_guard(self):
        """Mutually-lower interfaces terminate and yield an empty set."""
        def _listdir(path):
            if path == "/sys/class/net/vetha":
                return ["lower_vethb"]
            if path == "/sys/class/net/vethb":
                return ["lower_vetha"]
            return []

        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.realpath",
            side_effect=lambda path: path,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            side_effect=_listdir,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.glt_utils.read_sysfs_value",
            side_effect=OSError,
        ):
            result = iface_pci_bdfs("vetha")
        self.assertEqual(result, set())

    def test_mounted_block_devices(self):
        """/dev/* mount sources resolve to kernel device base names."""
        realpaths = {
            "/dev/mapper/vg-root": "/dev/dm-0",
            "/dev/nvme0n1p1": "/dev/nvme0n1p1",
        }
        files = {"/proc/mounts": PROC_MOUNTS}
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=self._open_from_files(files),
            create=True,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.realpath",
            side_effect=lambda path: realpaths.get(path, path),
        ):
            result = mounted_block_devices()
        self.assertEqual(result, {"dm-0", "nvme0n1p1"})

    def test_mounted_block_devices_unreadable(self):
        """Unreadable /proc/mounts raises so callers can fail closed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.open",
            side_effect=OSError,
            create=True,
        ):
            with self.assertRaises(OSError):
                mounted_block_devices()

    def test_block_device_physical_leaves_no_slaves(self):
        """A device with no slaves directory is its own leaf."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            side_effect=FileNotFoundError,
        ):
            result = block_device_physical_leaves("nvme0n1p2")
        self.assertEqual(result, {"nvme0n1p2"})

    def test_block_device_physical_leaves_stacked(self):
        """A stacked device resolves to its physical backing disks."""
        def _listdir(path):
            if path == "/sys/class/block/dm-0/slaves":
                return ["sda", "sdb"]
            raise FileNotFoundError(path)

        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            side_effect=_listdir,
        ):
            result = block_device_physical_leaves("dm-0")
        self.assertEqual(result, {"sda", "sdb"})

    def test_block_device_physical_leaves_cycle_guard(self):
        """Mutually-slaved devices terminate without infinite recursion."""
        def _listdir(path):
            if path == "/sys/class/block/dm-0/slaves":
                return ["dm-1"]
            if path == "/sys/class/block/dm-1/slaves":
                return ["dm-0"]
            raise FileNotFoundError(path)

        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            side_effect=_listdir,
        ):
            result = block_device_physical_leaves("dm-0")
        self.assertEqual(result, set())

    def test_unsafe_pci_bdfs_collects_nic_ancestors_and_storage(self):
        """Management NIC path BDFs and mounted-storage leaf BDF are excluded."""
        realpaths = {
            "/sys/class/net/eth0/device":
                "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0",
            "/sys/class/block/nvme0n1":
                "/sys/devices/pci0000:00/0000:00:02.0/0000:02:00.0/nvme/nvme0/nvme0n1",
        }

        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.default_route_ifaces",
            return_value={"eth0"},
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.iface_pci_bdfs",
            return_value={"0000:01:00.0"},
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.mounted_block_devices",
            return_value={"nvme0n1p1"},
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.block_device_physical_leaves",
            return_value={"nvme0n1"},
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.exists",
            return_value=True,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.realpath",
            side_effect=lambda path: realpaths.get(path, path),
        ):
            result = unsafe_pci_bdfs()

        self.assertEqual(
            result,
            {"0000:00:01.0", "0000:01:00.0", "0000:02:00.0"},
        )

    def test_unsafe_pci_bdfs_propagates_oserror(self):
        """Unreadable route tables raise so callers can fail closed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.default_route_ifaces",
            side_effect=OSError("no route tables"),
        ):
            with self.assertRaises(OSError):
                unsafe_pci_bdfs()

    def test_unsafe_pci_bdfs_propagates_mounted_block_oserror(self):
        """Unreadable /proc/mounts discards any partial NIC unsafe set."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.default_route_ifaces",
            return_value={"eth0"},
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.iface_pci_bdfs",
            return_value={"0000:01:00.0"},
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.exists",
            return_value=False,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.mounted_block_devices",
            side_effect=OSError("no /proc/mounts"),
        ):
            with self.assertRaises(OSError):
                unsafe_pci_bdfs()

    def test_iface_pci_bdfs_propagates_permission_error(self):
        """Permission errors resolving net sysfs must not look like 'no NIC'."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.path.realpath",
            return_value="/sys/class/net/eth0/device",
        ), GLTTestBase.unittest.mock.patch(
            "src.common.pci_utils.os.listdir",
            side_effect=PermissionError("denied"),
        ):
            with self.assertRaises(PermissionError):
                iface_pci_bdfs("eth0")


class TestBdfStrictRe(GLTTestBase):
    """Strict BDF form used by callers that splice input into sysfs paths."""

    def test_accepts_canonical_bdf(self):
        """A domain-qualified lowercase BDF is accepted."""
        self.assertIsNotNone(BDF_STRICT_RE.fullmatch("0000:01:00.0"))

    def test_rejects_path_traversal(self):
        """Path-like input cannot escape the PCI sysfs directory."""
        self.assertIsNone(BDF_STRICT_RE.fullmatch("../../etc/passwd"))

    def test_rejects_invalid_pci_function(self):
        """PCI function numbers are 0-7; .8/.9 must not match."""
        self.assertIsNone(BDF_STRICT_RE.fullmatch("0000:01:00.8"))
        self.assertIsNone(BDF_STRICT_RE.fullmatch("0000:01:00.9"))


class TestSysfsPathToPciBdf(GLTTestBase):
    """Resolve PCI BDF from sysfs device paths."""

    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.os.path.exists")
    def test_finds_bdf_at_varying_depths(self, mock_exists):
        """Valid BDF segments are found at shallow and deep sysfs paths."""
        mock_exists.return_value = True
        cases = [
            ("/sys/bus/pci/devices/0000:01:00.0", "0000:01:00.0"),
            (
                "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/nvme/nvme0/nvme0n1",
                "0000:01:00.0",
            ),
        ]
        for path, expected in cases:
            with self.subTest(path=path):
                self.assertEqual(sysfs_path_to_pci_bdf(path), expected)

    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.os.path.exists")
    def test_returns_none_when_no_match(self, mock_exists):
        """Paths without a present PCI device return None."""
        mock_exists.return_value = False
        self.assertIsNone(sysfs_path_to_pci_bdf("/sys/class/block/nvme0n1/device"))

    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.os.path.exists")
    def test_rejects_malformed_bdf(self, mock_exists):
        """Malformed BDF basename segments are rejected."""
        mock_exists.return_value = True
        self.assertIsNone(
            sysfs_path_to_pci_bdf("/sys/bus/pci/devices/not-a-bdf")
        )


class TestBdfInTranslatedDomain(GLTTestBase):
    """Detect DMA/DMA-FQ IOMMU group membership for a PCI BDF."""

    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.os.path.exists")
    def test_returns_false_when_no_iommu_group(self, mock_exists):
        """Devices without an iommu_group symlink are not translated."""
        mock_exists.return_value = False
        self.assertFalse(bdf_in_translated_domain("0000:01:00.0"))

    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.os.path.exists")
    def test_rejects_malformed_bdf_before_touching_sysfs(self, mock_exists):
        """A malformed BDF returns False without building a sysfs path."""
        self.assertFalse(bdf_in_translated_domain("../../etc/passwd"))
        mock_exists.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.open")
    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.os.path.realpath")
    @GLTTestBase.unittest.mock.patch("src.common.pci_utils.os.path.exists")
    def test_returns_true_for_dma_group_type(
        self, mock_exists, mock_realpath, mock_open
    ):
        """A DMA-typed iommu_group is reported as translated."""
        mock_exists.return_value = True
        mock_realpath.return_value = "/sys/kernel/iommu_groups/5"
        mock_open.return_value.__enter__.return_value.read.return_value = "DMA\n"
        self.assertTrue(bdf_in_translated_domain("0000:01:00.0"))


class TestPciBdfsBackingMount(GLTTestBase):
    """Mount-point resolution follows stacked storage to physical PCI leaves."""

    @GLTTestBase.unittest.mock.patch(
        "src.common.pci_utils.sysfs_path_to_pci_bdf",
        return_value="0000:01:00.0",
    )
    @GLTTestBase.unittest.mock.patch(
        "src.common.pci_utils.block_device_physical_leaves",
        return_value=["nvme0n1"],
    )
    @GLTTestBase.unittest.mock.patch(
        "src.common.pci_utils.mounted_block_devices",
        return_value={"dm-0"},
    )
    @GLTTestBase.unittest.mock.patch(
        "src.common.pci_utils.os.path.realpath",
        side_effect=["/dev/dm-0", "/sys/devices/pci0000:00/0000:01:00.0/nvme/nvme0"],
    )
    @GLTTestBase.unittest.mock.patch(
        "src.common.pci_utils.open",
        new_callable=GLTTestBase.unittest.mock.mock_open,
        read_data="/dev/mapper/vg-root / ext4 rw 0 0\n",
    )
    def test_resolves_mapper_root_to_physical_pci_device(
        self, _open, _realpath, _mounted, mock_leaves, mock_bdf
    ):
        """A dm/LVM root resolves to the physical disk's PCI device."""
        self.assertEqual(pci_bdfs_backing_mount("/"), {"0000:01:00.0"})
        mock_leaves.assert_called_once_with("dm-0")
        mock_bdf.assert_called_once_with(
            "/sys/devices/pci0000:00/0000:01:00.0/nvme/nvme0"
        )

    @GLTTestBase.unittest.mock.patch(
        "src.common.pci_utils.mounted_block_devices",
        return_value={"nvme0n1p2"},
    )
    @GLTTestBase.unittest.mock.patch(
        "src.common.pci_utils.open",
        new_callable=GLTTestBase.unittest.mock.mock_open,
        read_data="tmpfs /run tmpfs rw 0 0\n",
    )
    def test_returns_empty_when_mount_has_no_dev_source(self, _open, _mounted):
        """A mount point with no /dev/* source yields no BDFs."""
        self.assertEqual(pci_bdfs_backing_mount("/run"), set())


if __name__ == "__main__":
    GLTTestBase.unittest.main()
