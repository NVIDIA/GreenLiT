"""Unit test script for common/pci_utils.py"""

from unittest.mock import MagicMock

from tests.test_base import GLTTestBase
from src.common.pci_utils import (
    ASPEED_AST1150_DEVICE_ID,
    ASPEED_VENDOR_ID,
    EXT_CAP_ACS,
    EXT_CAP_ATS,
    PCI_CLASS_USB_HOST,
    block_device_physical_leaves,
    default_route_ifaces,
    get_bdfs_by_pci_id,
    get_bdfs_with_ext_cap,
    get_pci_children,
    get_pci_class,
    iface_pci_bdfs,
    mounted_block_devices,
    unsafe_pci_bdfs,
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


if __name__ == "__main__":
    GLTTestBase.unittest.main()
