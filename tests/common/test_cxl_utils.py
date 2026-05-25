"""Unit test script for common/cxl_utils.py"""

import os
import shutil
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


if __name__ == "__main__":
    GLTTestBase.unittest.main()
