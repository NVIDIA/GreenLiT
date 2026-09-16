"""Unit test script for common/cxl_utils.py"""
# pylint: disable=too-many-lines

import os
import shutil
import struct
import subprocess
import tempfile
from unittest.mock import MagicMock

from tests.test_base import GLTTestBase
from src.common import cxl_utils


def _mock_run(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Build a MagicMock that imitates subprocess.run's CompletedProcess."""
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


# ---------------------------------------------------------------------------
# find_first_cxl_bdf
# ---------------------------------------------------------------------------

LSPCI_ONE_CXL_DEVICE = (
    "0000:e1:00.0 CXL: Vendor 8086 Device 0d93\n"
)

LSPCI_MULTIPLE_CXL_DEVICES = (
    "0000:e1:00.0 CXL: Vendor 8086 Device 0d93\n"
    "\n"
    "0000:e2:00.0 CXL: Vendor 1b96 Device 0010\n"
)


class TestFindFirstCxlBdf(GLTTestBase):
    """Tests for find_first_cxl_bdf()."""

    def test_returns_first_bdf(self):
        """Picks the BDF from the first non-empty lspci output line."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_ONE_CXL_DEVICE),
        ) as run_mock:
            self.assertEqual(cxl_utils.find_first_cxl_bdf(), "0000:e1:00.0")
        run_mock.assert_called_once_with(
            ["lspci", "-d", "::0502", "-D"],
            capture_output=True, text=True, check=False,
        )

    def test_returns_first_when_multiple_devices(self):
        """Returns only the first device when several are listed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_MULTIPLE_CXL_DEVICES),
        ):
            self.assertEqual(cxl_utils.find_first_cxl_bdf(), "0000:e1:00.0")

    def test_returns_none_when_no_devices(self):
        """Empty lspci output yields None."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=""),
        ):
            self.assertIsNone(cxl_utils.find_first_cxl_bdf())

    def test_returns_none_on_nonzero_exit(self):
        """A non-zero lspci exit yields None even if stdout is populated."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_ONE_CXL_DEVICE, returncode=1),
        ):
            self.assertIsNone(cxl_utils.find_first_cxl_bdf())

    def test_returns_none_when_lspci_missing(self):
        """A FileNotFoundError raising lspci yields None."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(cxl_utils.find_first_cxl_bdf())


# ---------------------------------------------------------------------------
# find_cxl_port_dvsec_base
# ---------------------------------------------------------------------------

LSPCI_VV_MATCHING_DVSEC = (
    "0000:00:01.0 PCI bridge: ...\n"
    "\tCapabilities: [468 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0003 Rev=2 Len=14: CXL\n"
)

LSPCI_VV_WRONG_VENDOR = (
    "0000:00:01.0 PCI bridge: ...\n"
    "\tCapabilities: [468 v1] Designated Vendor-Specific: "
    "Vendor=1234 ID=0003 Rev=2 Len=14: Other\n"
)

LSPCI_VV_WRONG_DVSEC_ID = (
    "0000:00:01.0 PCI bridge: ...\n"
    "\tCapabilities: [468 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0002 Rev=2 Len=14: CXL\n"
)

LSPCI_VV_NO_DVSEC = (
    "0000:00:01.0 PCI bridge: ...\n"
    "\tCapabilities: [40] Express Root Port (Slot+), MSI 00\n"
)

LSPCI_VV_MULTIPLE_DVSEC = (
    "0000:00:01.0 PCI bridge: ...\n"
    "\tCapabilities: [100 v1] Designated Vendor-Specific: "
    "Vendor=1234 ID=0001 Rev=1 Len=14: Other\n"
    "\tCapabilities: [468 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0003 Rev=2 Len=14: CXL\n"
    "\tCapabilities: [500 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0003 Rev=2 Len=14: CXL\n"
)


class TestFindCxlPortDvsecBase(GLTTestBase):
    """Tests for find_cxl_port_dvsec_base()."""

    def test_returns_offset_when_dvsec_matches(self):
        """Returns the bracketed offset from the matching DVSEC capability line."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_MATCHING_DVSEC),
        ) as run_mock:
            self.assertEqual(
                cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"), 0x468,
            )
        run_mock.assert_called_once_with(
            ["lspci", "-vv", "-s", "0000:00:01.0"],
            capture_output=True, text=True, check=False,
        )

    def test_returns_first_matching_when_multiple(self):
        """When multiple matching DVSECs exist, returns the first one seen."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_MULTIPLE_DVSEC),
        ):
            self.assertEqual(
                cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"), 0x468,
            )

    def test_returns_none_when_no_dvsec(self):
        """Returns None when no Designated Vendor-Specific cap is present."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_NO_DVSEC),
        ):
            self.assertIsNone(cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"))

    def test_returns_none_on_wrong_vendor(self):
        """Returns None when the DVSEC vendor is not the CXL Consortium."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_WRONG_VENDOR),
        ):
            self.assertIsNone(cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"))

    def test_returns_none_on_wrong_dvsec_id(self):
        """Returns None when the DVSEC ID is not the CXL Port Extensions ID."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_WRONG_DVSEC_ID),
        ):
            self.assertIsNone(cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"))

    def test_returns_none_on_nonzero_exit(self):
        """Returns None when lspci exits non-zero."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_MATCHING_DVSEC, returncode=1),
        ):
            self.assertIsNone(cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"))

    def test_returns_none_when_lspci_missing(self):
        """Returns None when lspci binary is not installed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"))

    def test_returns_none_on_oserror(self):
        """Returns None when lspci raises an OSError other than not-found."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=OSError,
        ):
            self.assertIsNone(cxl_utils.find_cxl_port_dvsec_base("0000:00:01.0"))


# ---------------------------------------------------------------------------
# any_cxl_region_committed
# ---------------------------------------------------------------------------


class TestAnyCxlRegionCommitted(GLTTestBase):
    """Tests for any_cxl_region_committed()."""

    def setUp(self):
        """Build a fake sysfs tree under a tempdir."""
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.regions_dir = os.path.join(
            self.tmpdir, "sys", "bus", "cxl", "devices",
        )
        os.makedirs(self.regions_dir)

    def tearDown(self):
        """Tear down the fake sysfs tree."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _make_region(self, name: str, commit_value: str | None) -> str:
        """Create a region dir; if commit_value is None, omit the commit file."""
        region_dir = os.path.join(self.regions_dir, name)
        os.makedirs(region_dir)
        if commit_value is not None:
            with open(
                os.path.join(region_dir, "commit"), "w", encoding="utf-8",
            ) as fh:
                fh.write(commit_value)
        return region_dir

    def test_false_when_no_regions(self):
        """No region directories at all yields False."""
        self.assertFalse(
            cxl_utils.any_cxl_region_committed(sysfs_root=self.tmpdir),
        )

    def test_true_when_region_commit_is_one(self):
        """A region whose commit file contains '1' is committed."""
        self._make_region("region0", "1\n")
        self.assertTrue(
            cxl_utils.any_cxl_region_committed(sysfs_root=self.tmpdir),
        )

    def test_false_when_region_commit_is_zero(self):
        """A region whose commit file contains '0' is not committed."""
        self._make_region("region0", "0\n")
        self.assertFalse(
            cxl_utils.any_cxl_region_committed(sysfs_root=self.tmpdir),
        )

    def test_true_when_commit_file_missing(self):
        """An unreadable commit file is treated conservatively as committed."""
        self._make_region("region0", commit_value=None)
        self.assertTrue(
            cxl_utils.any_cxl_region_committed(sysfs_root=self.tmpdir),
        )

    def test_true_when_any_one_of_many_is_committed(self):
        """If any region in the set is committed, returns True."""
        self._make_region("region0", "0\n")
        self._make_region("region1", "1\n")
        self.assertTrue(
            cxl_utils.any_cxl_region_committed(sysfs_root=self.tmpdir),
        )

    def test_false_when_all_regions_uncommitted(self):
        """All regions reporting '0' yields False."""
        self._make_region("region0", "0\n")
        self._make_region("region1", "0\n")
        self.assertFalse(
            cxl_utils.any_cxl_region_committed(sysfs_root=self.tmpdir),
        )


# ---------------------------------------------------------------------------
# find_cxl_device_dvsec_base
# ---------------------------------------------------------------------------

LSPCI_VV_DEVICE_DVSEC_MATCH = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\tCapabilities: [600 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0000 Rev=2 Len=3c: CXL\n"
)

LSPCI_VV_DEVICE_DVSEC_WRONG_ID = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\tCapabilities: [600 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0003 Rev=2 Len=14: CXL\n"
)

LSPCI_VV_DEVICE_DVSEC_WRONG_VENDOR = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\tCapabilities: [600 v1] Designated Vendor-Specific: "
    "Vendor=1234 ID=0000 Rev=2 Len=3c: Other\n"
)

LSPCI_VV_DEVICE_DVSEC_MULTI = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\tCapabilities: [400 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0003 Rev=2 Len=14: CXL\n"
    "\tCapabilities: [600 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0000 Rev=2 Len=3c: CXL\n"
    "\tCapabilities: [800 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0000 Rev=2 Len=3c: CXL\n"
)


class TestFindCxlDeviceDvsecBase(GLTTestBase):
    """Tests for find_cxl_device_dvsec_base()."""

    def test_returns_offset_when_device_dvsec_matches(self):
        """Returns the bracketed offset from the matching Device DVSEC line."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_DEVICE_DVSEC_MATCH),
        ) as run_mock:
            self.assertEqual(
                cxl_utils.find_cxl_device_dvsec_base("0000:e1:00.0"), 0x600,
            )
        run_mock.assert_called_once_with(
            ["lspci", "-vv", "-s", "0000:e1:00.0"],
            capture_output=True, text=True, check=True,
        )

    def test_returns_first_matching_when_multiple(self):
        """When multiple Device DVSECs exist, returns the first one seen."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_DEVICE_DVSEC_MULTI),
        ):
            self.assertEqual(
                cxl_utils.find_cxl_device_dvsec_base("0000:e1:00.0"), 0x600,
            )

    def test_returns_none_when_only_port_dvsec_present(self):
        """A device with only the Port Extensions DVSEC (ID=0x0003) returns None."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_DEVICE_DVSEC_WRONG_ID),
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_device_dvsec_base("0000:e1:00.0"),
            )

    def test_returns_none_on_wrong_vendor(self):
        """Vendor != CXL Consortium yields None."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_DEVICE_DVSEC_WRONG_VENDOR),
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_device_dvsec_base("0000:e1:00.0"),
            )

    def test_returns_none_when_lspci_missing(self):
        """Returns None when lspci binary is not installed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_device_dvsec_base("0000:e1:00.0"),
            )

    def test_returns_none_on_calledprocesserror(self):
        """Returns None when lspci exits non-zero (check=True raises)."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["lspci"]),
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_device_dvsec_base("0000:e1:00.0"),
            )

    def test_returns_none_on_oserror(self):
        """Returns None when lspci raises a generic OSError."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=OSError,
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_device_dvsec_base("0000:e1:00.0"),
            )


# ---------------------------------------------------------------------------
# find_first_cxl_device_dvsec_bdf
# ---------------------------------------------------------------------------

LSPCI_D_TWO_DEVICES = (
    "0000:00:01.0 PCI bridge: Some bridge\n"
    "0000:e1:00.0 Coprocessor: Some accel\n"
)


def _make_lspci_dispatcher(d_stdout: str, per_bdf_stdout: dict[str, str]):
    """Build a side_effect callable that responds to both lspci -D and
    lspci -vv -s <bdf>. Unknown commands raise AssertionError so tests fail
    loudly on unexpected calls.
    """
    def _side_effect(cmd, **_kwargs):
        if cmd == ["lspci", "-D"]:
            return _mock_run(stdout=d_stdout)
        if len(cmd) == 4 and cmd[:3] == ["lspci", "-vv", "-s"]:
            return _mock_run(stdout=per_bdf_stdout.get(cmd[3], ""))
        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")
    return _side_effect


class TestFindFirstCxlDeviceDvsecBdf(GLTTestBase):
    """Tests for find_first_cxl_device_dvsec_bdf()."""

    def test_returns_first_bdf_with_device_dvsec(self):
        """Skips devices without a Device DVSEC and returns the first that has one."""
        side_effect = _make_lspci_dispatcher(
            d_stdout=LSPCI_D_TWO_DEVICES,
            per_bdf_stdout={
                "0000:00:01.0": LSPCI_VV_DEVICE_DVSEC_WRONG_ID,
                "0000:e1:00.0": LSPCI_VV_DEVICE_DVSEC_MATCH,
            },
        )
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run", side_effect=side_effect,
        ):
            self.assertEqual(
                cxl_utils.find_first_cxl_device_dvsec_bdf(), "0000:e1:00.0",
            )

    def test_returns_none_when_no_device_dvsec_anywhere(self):
        """Returns None when no device on the bus exposes a Device DVSEC."""
        side_effect = _make_lspci_dispatcher(
            d_stdout=LSPCI_D_TWO_DEVICES,
            per_bdf_stdout={
                "0000:00:01.0": LSPCI_VV_DEVICE_DVSEC_WRONG_ID,
                "0000:e1:00.0": LSPCI_VV_DEVICE_DVSEC_WRONG_VENDOR,
            },
        )
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run", side_effect=side_effect,
        ):
            self.assertIsNone(cxl_utils.find_first_cxl_device_dvsec_bdf())

    def test_returns_none_when_lspci_missing(self):
        """Returns None when lspci binary is not installed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(cxl_utils.find_first_cxl_device_dvsec_bdf())

    def test_returns_none_on_calledprocesserror(self):
        """Returns None when lspci -D itself exits non-zero."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["lspci", "-D"]),
        ):
            self.assertIsNone(cxl_utils.find_first_cxl_device_dvsec_bdf())

    def test_returns_none_when_no_devices(self):
        """Empty lspci -D output yields None."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=""),
        ):
            self.assertIsNone(cxl_utils.find_first_cxl_device_dvsec_bdf())


# ---------------------------------------------------------------------------
# find_cxl_component_regs
# ---------------------------------------------------------------------------

LSPCI_VV_REGLOC_COMPONENT = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\tCapabilities: [800 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0008 Rev=0 Len=20: CXL\n"
    "\t\tCXL RegisterLocator\n"
    "\t\t\tBlock1: BIR: bar0, ID: component registers, offset: "
    "0000000000ec0000\n"
    "\t\t\tBlock2: BIR: bar0, ID: device registers, offset: "
    "0000000000800000\n"
)

LSPCI_VV_REGLOC_NO_COMPONENT = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\tCapabilities: [800 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0008 Rev=0 Len=20: CXL\n"
    "\t\tCXL RegisterLocator\n"
    "\t\t\tBlock1: BIR: bar0, ID: device registers, offset: "
    "0000000000800000\n"
)

LSPCI_VV_REGLOC_COMPONENT_BAR2 = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\t\t\tBlock3: BIR: bar2, ID: component registers, offset: "
    "0000000000001000\n"
)

LSPCI_VV_REGLOC_MULTIPLE_COMPONENT = (
    "0000:e1:00.0 Coprocessor: ...\n"
    "\t\t\tBlock1: BIR: bar0, ID: component registers, offset: "
    "0000000000ec0000\n"
    "\t\t\tBlock4: BIR: bar4, ID: component registers, offset: "
    "0000000000000000\n"
)


class TestFindCxlComponentRegs(GLTTestBase):
    """Tests for find_cxl_component_regs()."""

    def test_returns_bar_and_offset_for_component_block(self):
        """Returns (bar, offset) from the first 'component registers' Block line."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_REGLOC_COMPONENT),
        ) as run_mock:
            self.assertEqual(
                cxl_utils.find_cxl_component_regs("0000:e1:00.0"),
                (0, 0xEC0000),
            )
        run_mock.assert_called_once_with(
            ["lspci", "-vv", "-s", "0000:e1:00.0"],
            capture_output=True, text=True, check=True,
        )

    def test_returns_none_when_no_component_block(self):
        """Returns None when the Register Locator has only non-component blocks."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_REGLOC_NO_COMPONENT),
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_component_regs("0000:e1:00.0"),
            )

    def test_parses_non_bar0_correctly(self):
        """Correctly parses Block lines with a BIR other than bar0."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_REGLOC_COMPONENT_BAR2),
        ):
            self.assertEqual(
                cxl_utils.find_cxl_component_regs("0000:e1:00.0"),
                (2, 0x1000),
            )

    def test_returns_first_when_multiple_component_blocks(self):
        """When multiple 'component registers' blocks exist, returns the first."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            return_value=_mock_run(stdout=LSPCI_VV_REGLOC_MULTIPLE_COMPONENT),
        ):
            self.assertEqual(
                cxl_utils.find_cxl_component_regs("0000:e1:00.0"),
                (0, 0xEC0000),
            )

    def test_returns_none_when_lspci_missing(self):
        """Returns None when lspci binary is not installed."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_component_regs("0000:e1:00.0"),
            )

    def test_returns_none_on_calledprocesserror(self):
        """Returns None when lspci exits non-zero."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["lspci"]),
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_component_regs("0000:e1:00.0"),
            )

    def test_returns_none_on_oserror(self):
        """Returns None on a generic OSError raised by lspci."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.subprocess.run",
            side_effect=OSError,
        ):
            self.assertIsNone(
                cxl_utils.find_cxl_component_regs("0000:e1:00.0"),
            )


# ---------------------------------------------------------------------------
# read_cxl_device_dvsec_cap_bits / read_cxl_device_dvsec_status2_bits
# ---------------------------------------------------------------------------


def _write_sparse_config(path: str, words: dict[int, int]) -> None:
    """Write a fake config-space file at `path`.

    `words` maps absolute byte offsets to 16-bit little-endian values.
    Bytes outside the listed words are zero-filled. The file is sized just
    large enough to hold the highest word.
    """
    size = (max(words.keys()) + 2) if words else 0
    buf = bytearray(size)
    for off, val in words.items():
        struct.pack_into("<H", buf, off, val)
    with open(path, "wb") as fh:
        fh.write(bytes(buf))


def _make_device_config(
    sysfs_root: str, bdf: str, words: dict[int, int],
) -> str:
    """Create <sysfs_root>/sys/bus/pci/devices/<bdf>/config containing `words`.

    Returns the absolute path to the device directory.
    """
    device_dir = os.path.join(
        sysfs_root, "sys", "bus", "pci", "devices", bdf,
    )
    os.makedirs(device_dir, exist_ok=True)
    _write_sparse_config(os.path.join(device_dir, "config"), words)
    return device_dir


class TestReadCxlDeviceDvsecCapBits(GLTTestBase):
    """Tests for read_cxl_device_dvsec_cap_bits()."""

    # The CAP register sits 0xA bytes into the CXL Device DVSEC.
    _CAP_REG_OFF = 0xA
    _DVSEC_OFF = 0x10C
    _CAP_ABS_OFF = _DVSEC_OFF + _CAP_REG_OFF  # 0x116
    _BDF = "0002:81:00.0"

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _run(self, cap_word: int) -> dict[str, bool] | None:
        """Build fake sysfs with CAP=cap_word at the test's DVSEC and read."""
        _make_device_config(
            self.tmpdir, self._BDF, {self._CAP_ABS_OFF: cap_word},
        )
        return cxl_utils.read_cxl_device_dvsec_cap_bits(
            self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
        )

    def test_decodes_observed_hardware_value(self):
        """Hardware-observed value 0x8bd7 → all four decoded bits True.

        Baseline from VR-NVL72 box, GPU 0002:81:00.0: CAP=0x8bd7 advertises
        RST_CAPABLE=1, CACHE_WBI_CAPABLE=1, CACHE_CAPABLE=1, MEM_CAPABLE=1.
        """
        self.assertEqual(
            self._run(0x8BD7),
            {
                "cache_capable":     True,
                "mem_capable":       True,
                "cache_wbi_capable": True,
                "rst_capable":       True,
            },
        )

    def test_all_bits_clear_when_raw_zero(self):
        """Raw word 0x0000 → every decoded flag is False."""
        self.assertEqual(
            self._run(0x0000),
            {
                "cache_capable":     False,
                "mem_capable":       False,
                "cache_wbi_capable": False,
                "rst_capable":       False,
            },
        )

    def test_only_rst_capable_when_only_bit7_set(self):
        """Raw word 0x0080 (bit 7) → only rst_capable is True."""
        self.assertEqual(
            self._run(0x0080),
            {
                "cache_capable":     False,
                "mem_capable":       False,
                "cache_wbi_capable": False,
                "rst_capable":       True,
            },
        )

    def test_ignores_undecoded_bits(self):
        """Bits the helper doesn't decode (e.g. bit 1, bit 3, bit 4) don't
        leak into the returned dict."""
        # 0x003A = bits 1, 3, 4, 5; none of which are decoded.
        self.assertEqual(
            self._run(0x003A),
            {
                "cache_capable":     False,
                "mem_capable":       False,
                "cache_wbi_capable": False,
                "rst_capable":       False,
            },
        )

    def test_returns_none_when_config_missing(self):
        """Device directory absent → None."""
        # Don't materialize any device under self.tmpdir.
        self.assertIsNone(
            cxl_utils.read_cxl_device_dvsec_cap_bits(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_returns_none_on_short_read(self):
        """Config file too small to cover the CAP offset → None."""
        device_dir = os.path.join(
            self.tmpdir, "sys", "bus", "pci", "devices", self._BDF,
        )
        os.makedirs(device_dir)
        # Only 0x10 bytes; CAP would be at 0x116 - way out of range.
        with open(os.path.join(device_dir, "config"), "wb") as fh:
            fh.write(b"\x00" * 0x10)
        self.assertIsNone(
            cxl_utils.read_cxl_device_dvsec_cap_bits(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_returns_none_on_oserror(self):
        """A generic OSError on open() → None.

        The fake config file is materialized first so that, without the
        patch, the helper would succeed; this guarantees the None result
        comes from the patched PermissionError path and not from a missing
        path (which would also raise OSError and produce a false pass).
        """
        _make_device_config(
            self.tmpdir, self._BDF, {self._CAP_ABS_OFF: 0x0000},
        )
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.pci_utils.open",
            side_effect=PermissionError("no perm"),
            create=True,
        ) as open_mock:
            self.assertIsNone(
                cxl_utils.read_cxl_device_dvsec_cap_bits(
                    self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
                )
            )
        open_mock.assert_called_once()


class TestReadCxlDeviceDvsecStatus2Bits(GLTTestBase):
    """Tests for read_cxl_device_dvsec_status2_bits()."""

    # The STATUS2 register sits 0x12 bytes into the CXL Device DVSEC.
    _STATUS2_REG_OFF = 0x12
    _DVSEC_OFF = 0x10C
    _STATUS2_ABS_OFF = _DVSEC_OFF + _STATUS2_REG_OFF  # 0x11E
    _BDF = "0002:81:00.0"

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _run(self, status2_word: int) -> dict[str, bool] | None:
        """Build fake sysfs with STATUS2=status2_word and read."""
        _make_device_config(
            self.tmpdir, self._BDF, {self._STATUS2_ABS_OFF: status2_word},
        )
        return cxl_utils.read_cxl_device_dvsec_status2_bits(
            self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
        )

    def test_all_three_bits_set(self):
        """Raw word 0x0007 → cache_inv, rst_done, rst_err all True."""
        self.assertEqual(
            self._run(0x0007),
            {"cache_inv": True, "rst_done": True, "rst_err": True},
        )

    def test_all_bits_clear_when_raw_zero(self):
        """Raw word 0x0000 → every decoded flag is False."""
        self.assertEqual(
            self._run(0x0000),
            {"cache_inv": False, "rst_done": False, "rst_err": False},
        )

    def test_only_cache_inv_when_only_bit0_set(self):
        """Raw word 0x0001 (bit 0) → only cache_inv is True."""
        self.assertEqual(
            self._run(0x0001),
            {"cache_inv": True, "rst_done": False, "rst_err": False},
        )

    def test_only_rst_done_when_only_bit1_set(self):
        """Raw word 0x0002 (bit 1) → only rst_done is True."""
        self.assertEqual(
            self._run(0x0002),
            {"cache_inv": False, "rst_done": True, "rst_err": False},
        )

    def test_only_rst_err_when_only_bit2_set(self):
        """Raw word 0x0004 (bit 2) → only rst_err is True."""
        self.assertEqual(
            self._run(0x0004),
            {"cache_inv": False, "rst_done": False, "rst_err": True},
        )

    def test_ignores_undecoded_bits(self):
        """Bits the helper doesn't decode (e.g. bit 3+) don't leak."""
        # 0xFFF8 = every bit set except bits 0/1/2.
        self.assertEqual(
            self._run(0xFFF8),
            {"cache_inv": False, "rst_done": False, "rst_err": False},
        )

    def test_returns_none_when_config_missing(self):
        """Device directory absent → None."""
        self.assertIsNone(
            cxl_utils.read_cxl_device_dvsec_status2_bits(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_returns_none_on_short_read(self):
        """Config file too small to cover the STATUS2 offset → None."""
        device_dir = os.path.join(
            self.tmpdir, "sys", "bus", "pci", "devices", self._BDF,
        )
        os.makedirs(device_dir)
        with open(os.path.join(device_dir, "config"), "wb") as fh:
            fh.write(b"\x00" * 0x10)
        self.assertIsNone(
            cxl_utils.read_cxl_device_dvsec_status2_bits(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_returns_none_on_oserror(self):
        """A generic OSError on open() → None.

        The fake config file is materialized first so that, without the
        patch, the helper would succeed; this guarantees the None result
        comes from the patched PermissionError path and not from a missing
        path (which would also raise OSError and produce a false pass).
        """
        _make_device_config(
            self.tmpdir, self._BDF, {self._STATUS2_ABS_OFF: 0x0000},
        )
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.pci_utils.open",
            side_effect=PermissionError("no perm"),
            create=True,
        ) as open_mock:
            self.assertIsNone(
                cxl_utils.read_cxl_device_dvsec_status2_bits(
                    self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
                )
            )
        open_mock.assert_called_once()


# ---------------------------------------------------------------------------
# read_cxl_port_unmask_sbr / read_cxl_flexbus_port_cxl_mode /
# is_cxl_sbr_masked_port
# ---------------------------------------------------------------------------

LSPCI_VV_FLEXBUS_DVSEC = (
    "0000:00:01.0 PCI bridge: ...\n"
    "\tCapabilities: [47c v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0007 Rev=1 Len=20: CXL\n"
)

LSPCI_VV_PORT_AND_FLEXBUS = (
    "0000:00:01.0 PCI bridge: ...\n"
    "\tCapabilities: [468 v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0003 Rev=2 Len=14: CXL\n"
    "\tCapabilities: [47c v1] Designated Vendor-Specific: "
    "Vendor=1e98 ID=0007 Rev=1 Len=20: CXL\n"
)


class TestReadCxlPortUnmaskSbr(GLTTestBase):
    """Tests for read_cxl_port_unmask_sbr()."""

    _DVSEC_OFF = 0x468
    _CTL_ABS_OFF = _DVSEC_OFF + 0x0C
    _BDF = "0000:00:01.0"

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_false_when_unmask_sbr_clear(self):
        """Unmask SBR bit 0 clear means SBR is masked."""
        _make_device_config(
            self.tmpdir, self._BDF, {self._CTL_ABS_OFF: 0x0000},
        )
        self.assertFalse(
            cxl_utils.read_cxl_port_unmask_sbr(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_true_when_unmask_sbr_set(self):
        """Unmask SBR bit 0 set means SBR can take effect."""
        _make_device_config(
            self.tmpdir, self._BDF, {self._CTL_ABS_OFF: 0x0001},
        )
        self.assertTrue(
            cxl_utils.read_cxl_port_unmask_sbr(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_returns_none_when_config_missing(self):
        """Device directory absent → None."""
        self.assertIsNone(
            cxl_utils.read_cxl_port_unmask_sbr(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )


class TestReadCxlFlexbusPortCxlMode(GLTTestBase):
    """Tests for read_cxl_flexbus_port_cxl_mode()."""

    _DVSEC_OFF = 0x47C
    _STATUS_ABS_OFF = _DVSEC_OFF + 0x0E
    _BDF = "0000:00:01.0"

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_true_when_cache_status_set(self):
        """Cache status bit matches pcie_is_cxl()."""
        _make_device_config(
            self.tmpdir, self._BDF, {self._STATUS_ABS_OFF: 0x0001},
        )
        self.assertTrue(
            cxl_utils.read_cxl_flexbus_port_cxl_mode(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_true_when_mem_status_set(self):
        """Mem status bit matches pcie_is_cxl()."""
        _make_device_config(
            self.tmpdir, self._BDF, {self._STATUS_ABS_OFF: 0x0004},
        )
        self.assertTrue(
            cxl_utils.read_cxl_flexbus_port_cxl_mode(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )

    def test_false_when_neither_cxl_status_bit_set(self):
        """PCIe-mode Flex Bus status is not pcie_is_cxl()."""
        _make_device_config(
            self.tmpdir, self._BDF, {self._STATUS_ABS_OFF: 0x0000},
        )
        self.assertFalse(
            cxl_utils.read_cxl_flexbus_port_cxl_mode(
                self._BDF, self._DVSEC_OFF, sysfs_root=self.tmpdir,
            )
        )


class TestIsCxlSbrMaskedPort(GLTTestBase):
    """Tests for is_cxl_sbr_masked_port()."""

    _BDF = "0000:00:01.0"
    _PORT_CTL_ABS = 0x468 + 0x0C
    _FLEX_STATUS_ABS = 0x47C + 0x0E

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _config(self, port_ctl, flex_status):
        _make_device_config(
            self.tmpdir, self._BDF,
            {self._PORT_CTL_ABS: port_ctl, self._FLEX_STATUS_ABS: flex_status},
        )

    def test_true_when_cxl_mode_and_sbr_masked(self):
        """CXL.cache status with Unmask SBR clear is the hide-bus case."""
        self._config(port_ctl=0x0000, flex_status=0x0001)
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.glt_utils.run_command",
            return_value=_mock_run(stdout=LSPCI_VV_PORT_AND_FLEXBUS),
        ):
            self.assertTrue(
                cxl_utils.is_cxl_sbr_masked_port(
                    self._BDF, sysfs_root=self.tmpdir,
                )
            )

    def test_false_when_unmask_sbr_set(self):
        """Unmask SBR set means bus reset is a real hardware method."""
        self._config(port_ctl=0x0001, flex_status=0x0001)
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.glt_utils.run_command",
            return_value=_mock_run(stdout=LSPCI_VV_PORT_AND_FLEXBUS),
        ):
            self.assertFalse(
                cxl_utils.is_cxl_sbr_masked_port(
                    self._BDF, sysfs_root=self.tmpdir,
                )
            )

    def test_false_when_link_in_pcie_mode(self):
        """Unmask SBR has no effect when the link is not in CXL mode."""
        self._config(port_ctl=0x0000, flex_status=0x0000)
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.glt_utils.run_command",
            return_value=_mock_run(stdout=LSPCI_VV_PORT_AND_FLEXBUS),
        ):
            self.assertFalse(
                cxl_utils.is_cxl_sbr_masked_port(
                    self._BDF, sysfs_root=self.tmpdir,
                )
            )

    def test_false_when_port_dvsec_missing(self):
        """No Port Extensions DVSEC means cxl_sbr_masked() is false."""
        self._config(port_ctl=0x0000, flex_status=0x0001)
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.glt_utils.run_command",
            return_value=_mock_run(stdout=LSPCI_VV_FLEXBUS_DVSEC),
        ):
            self.assertFalse(
                cxl_utils.is_cxl_sbr_masked_port(
                    self._BDF, sysfs_root=self.tmpdir,
                )
            )

    def test_none_when_config_unreadable(self):
        """DVSECs present but config missing cannot be classified."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.glt_utils.run_command",
            return_value=_mock_run(stdout=LSPCI_VV_PORT_AND_FLEXBUS),
        ):
            self.assertIsNone(
                cxl_utils.is_cxl_sbr_masked_port(
                    self._BDF, sysfs_root=self.tmpdir,
                )
            )

    def test_none_when_lspci_fails(self):
        """lspci timeout, missing binary, or non-zero exit cannot be classified."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.cxl_utils.glt_utils.run_command",
            side_effect=RuntimeError("Failed to run lspci"),
        ):
            self.assertIsNone(
                cxl_utils.is_cxl_sbr_masked_port(
                    self._BDF, sysfs_root=self.tmpdir,
                )
            )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
