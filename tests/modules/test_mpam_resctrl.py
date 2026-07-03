"""Unit tests for ARM64 MPAM resctrl schema policy."""
# pylint: disable=wrong-import-position

import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_mpam_resctrl as mpam_resctrl


class TestMpamResctrlInfo(GLTTestBase):
    """Validate Grace/Vera-specific MPAM resctrl info behavior."""

    def _patch_mounted_info(self, mb_mon_present):
        def exists(path):
            if path.endswith("MB_MON"):
                return mb_mon_present
            return True

        def isdir(path):
            if path.endswith("MB_MON"):
                return mb_mon_present
            return True

        return (
            self.mock.patch.object(mpam_resctrl, "_resctrl_is_mounted",
                                   return_value=True),
            self.mock.patch.object(mpam_resctrl.os.path, "isdir",
                                   side_effect=isdir),
            self.mock.patch.object(mpam_resctrl.os.path, "exists",
                                   side_effect=exists),
            self.mock.patch.object(mpam_resctrl.os.path, "isfile",
                                   return_value=True),
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_grace_does_not_require_mb_mon(self, mock_nvdev):
        """Grace expects the base schema but not the MB_MON directory."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        patches = self._patch_mounted_info(mb_mon_present=False)
        mock_open = self.mock.mock_open(read_data="mbm_total_bytes\n")

        with patches[0], patches[1], patches[2], patches[3], \
                self.mock.patch("builtins.open", mock_open):
            mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        mock_open.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_vera_missing_mb_mon_fails(self, mock_nvdev):
        """Vera is the full-schema control, so missing MB_MON is a regression."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = self._patch_mounted_info(mb_mon_present=False)

        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        self.assertIn("MB_MON (expected directory) is missing",
                      str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_unknown_cpu_type_skips(self, mock_nvdev):
        """The platform-specific schema is not applicable to other CPUs."""
        mock_nvdev.return_value.is_cpu_type.return_value = False
        patches = self._patch_mounted_info(mb_mon_present=True)

        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        self.assertIn("not an NVIDIA Grace or Vera CPU", str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_vera_mb_mon_requires_mbm_total_bytes(self, mock_nvdev):
        """Vera full-schema pass includes the MBM total-bytes event."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = self._patch_mounted_info(mb_mon_present=True)
        mock_open = self.mock.mock_open(read_data="mbm_total_bytes\n")

        with patches[0], patches[1], patches[2], patches[3], \
                self.mock.patch("builtins.open", mock_open):
            mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        mock_open.assert_called_once_with(
            f"{mpam_resctrl.RESCTRL_MOUNT_POINT}/info/MB_MON/mon_features",
            "r",
            encoding="utf-8",
        )

    def test_info_unmounts_when_post_mount_check_fails(self):
        """A test-owned mount is cleaned up even when validation fails early."""
        with self.mock.patch.object(mpam_resctrl.subprocess, "run") as mock_run, \
                self.mock.patch.object(mpam_resctrl.glt_utils, "is_cli_available",
                                       return_value=True), \
                self.mock.patch.object(mpam_resctrl, "NvDev"), \
                self.mock.patch.object(mpam_resctrl, "_resctrl_is_mounted",
                                       side_effect=(False, False)):
            with self.assertRaises(GLTCore.GLTTestFail):
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(
            mock_run.call_args_list[0].args[0],
            ["mount", "-t", "resctrl", "resctrl", "/sys/fs/resctrl"],
        )
        self.assertEqual(
            mock_run.call_args_list[1].args[0],
            ["umount", "/sys/fs/resctrl"],
        )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
