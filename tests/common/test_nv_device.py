"""Unit test script for common/nv_device.py"""

import platform
import subprocess
from unittest.mock import mock_open

from tests.test_base import GLTTestBase
from src.common.nv_device import NvDev


class TestNvDevice(GLTTestBase):
    """Test Class for NvDev"""

    def setUp(self):
        """Set up test environment"""
        super().setUp()
        self.sample_dmi_modalias = (
            "dmi:bvnAMI:bvrF7A:bd12/10/2022:svnNVIDIA:pnDGX-A100:pvr:"
            "rvnNVIDIA:rnDGX-A100:rvrV1.0:cvnNVIDIA:ct16:cvr:"
        )
        # Update the manifest to have a pattern that will actually match
        self.sample_hw_manifest = {"tpm": ["dmi:bvnAMI:bvrF7A:*ct16*"]}

    def test_init(self):
        """Test NvDev initialization"""
        # Test with NCA available
        with GLTTestBase.unittest.mock.patch(
            "os.path.exists", return_value=True
        ):
            nv_dev = NvDev()
            self.assertTrue(nv_dev.nca_available)

        # Test without NCA available
        with GLTTestBase.unittest.mock.patch(
            "os.path.exists", return_value=False
        ):
            nv_dev = NvDev()
            self.assertFalse(nv_dev.nca_available)

    def test_get_nvidia_3d_bdf(self):
        """Test get_nvidia_3d_bdf method"""
        # Test with NVIDIA 3D device present
        with GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "01:00.0\n"
            result = NvDev.get_nvidia_3d_bdf()
            self.assertEqual(result, "01:00.0\n")
            mock_run.assert_called_once()

        # Test without NVIDIA 3D device
        with GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            result = NvDev.get_nvidia_3d_bdf()
            self.assertEqual(result, "")
            mock_run.assert_called_once()

    def test_get_release_info(self):
        """Test get_release_info method"""
        # Test with platform.freedesktop_os_release available
        sample_os_release = {
            "NAME": "Ubuntu",
            "VERSION": "22.04",
            "ID": "ubuntu",
        }

        with GLTTestBase.unittest.mock.patch.object(
            platform, "freedesktop_os_release", return_value=sample_os_release
        ):
            result = NvDev.get_release_info()
            self.assertEqual(result, sample_os_release)

        # Test with /etc/os-release file
        sample_os_release_file = 'NAME="Ubuntu"\nVERSION="22.04"\nID=ubuntu\n'

        # Remove the attribute temporarily to test the fallback
        original_attr = getattr(platform, "freedesktop_os_release", None)
        if hasattr(platform, "freedesktop_os_release"):
            delattr(platform, "freedesktop_os_release")

        with GLTTestBase.unittest.mock.patch(
            "builtins.open", mock_open(read_data=sample_os_release_file)
        ):
            result = NvDev.get_release_info()
            self.assertEqual(result["NAME"], "Ubuntu")
            self.assertEqual(result["VERSION"], "22.04")
            self.assertEqual(result["ID"], "ubuntu")

        # Restore the attribute if it existed
        if original_attr is not None:
            setattr(platform, "freedesktop_os_release", original_attr)

    def test_has_hopper_gpu(self):
        """Test has_hopper_gpu method"""
        nv_dev = NvDev()

        # Test with NCA available and Hopper GPU present
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", True
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            self.assertTrue(nv_dev.has_hopper_gpu())
            mock_run.assert_called_once()

        # Test with NCA available and no Hopper GPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", True
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            self.assertFalse(nv_dev.has_hopper_gpu())
            mock_run.assert_called_once()

        # Test without NCA but with Hopper GPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", False
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.check_output"
        ) as mock_check_output:
            mock_check_output.return_value = (
                b"10de:2302 NVIDIA Corporation Device\n"
            )
            self.assertTrue(nv_dev.has_hopper_gpu())
            mock_check_output.assert_called_once()

        # Test without NCA and no Hopper GPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", False
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.check_output"
        ) as mock_check_output:
            mock_check_output.return_value = (
                b"10de:1234 NVIDIA Corporation Device\n"
            )
            self.assertFalse(nv_dev.has_hopper_gpu())
            mock_check_output.assert_called_once()

        # Test with subprocess error
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", False
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "cmd"),
        ):
            self.assertFalse(nv_dev.has_hopper_gpu())

    def test_is_cpu_type(self):
        """Test is_cpu_type method"""
        nv_dev = NvDev()

        # Test with NCA available and Grace CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", True
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            self.assertTrue(nv_dev.is_cpu_type("grace"))
            mock_run.assert_called_once()

        # Test with NCA available and no Grace CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", True
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            self.assertFalse(nv_dev.is_cpu_type("grace"))
            mock_run.assert_called_once()

        # Test without NCA but with Grace CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", False
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            self.assertTrue(nv_dev.is_cpu_type("grace"))
            mock_run.assert_called_once()

        # Test without NCA and no Grace CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", False
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            self.assertFalse(nv_dev.is_cpu_type("grace"))
            mock_run.assert_called_once()

        # Test with NCA available and Vera CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", True
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            self.assertTrue(nv_dev.is_cpu_type("vera"))
            mock_run.assert_called_once()

        # Test with NCA available and no Vera CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", True
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            self.assertFalse(nv_dev.is_cpu_type("vera"))
            mock_run.assert_called_once()

        # Test without NCA but with Vera CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", False
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            self.assertTrue(nv_dev.is_cpu_type("vera"))
            mock_run.assert_called_once()

        # Test without NCA and no Vera CPU
        with GLTTestBase.unittest.mock.patch.object(
            nv_dev, "nca_available", False
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            self.assertFalse(nv_dev.is_cpu_type("vera"))
            mock_run.assert_called_once()


    def test_manifest_skip_func(self):
        """Test manifest_skip_func method"""
        nv_dev = NvDev()

        # Create a mock for re.match that returns a match for our
        # sample modalias
        def mock_re_match(_, string):
            # For tests with our sample DMI modalias, return a match
            if string == self.sample_dmi_modalias:
                return GLTTestBase.unittest.mock.Mock()
            return None

        # Test with matching modalias and specified parser
        with GLTTestBase.unittest.mock.patch(
            "os.path.isfile", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "src.common.nv_device.NvDev._NvDev__dmi_modalias_parser",
            return_value=[self.sample_dmi_modalias],
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open", mock_open()
        ), GLTTestBase.unittest.mock.patch(
            "json.load", return_value=self.sample_hw_manifest
        ), GLTTestBase.unittest.mock.patch(
            "re.match", side_effect=mock_re_match
        ):
            self.assertTrue(nv_dev.manifest_skip_func("tpm", "dmi"))

        # Test with non-matching modalias
        different_modalias = (
            "dmi:bvnDell:bvrF7A:bd12/10/2022:svnDell:pnXPS:pvr:"
            "rvnDell:rnXPS:rvrV1.0:cvnDell:ct16:cvr:"
        )
        with GLTTestBase.unittest.mock.patch(
            "os.path.isfile", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "src.common.nv_device.NvDev._NvDev__dmi_modalias_parser",
            return_value=[different_modalias],
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open", mock_open()
        ), GLTTestBase.unittest.mock.patch(
            "json.load", return_value=self.sample_hw_manifest
        ), GLTTestBase.unittest.mock.patch(
            "re.match", side_effect=mock_re_match
        ):
            self.assertFalse(nv_dev.manifest_skip_func("tpm", "dmi"))

        # Test with missing hardware manifest
        with GLTTestBase.unittest.mock.patch(
            "os.path.isfile", return_value=False
        ):
            self.assertFalse(nv_dev.manifest_skip_func("tpm", "dmi"))

        # Test with invalid parser
        with GLTTestBase.unittest.mock.patch(
            "os.path.isfile", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open", mock_open()
        ), GLTTestBase.unittest.mock.patch(
            "json.load", return_value=self.sample_hw_manifest
        ):
            self.assertFalse(
                nv_dev.manifest_skip_func("tpm", "invalid_parser")
            )

        # Test with no specified parser but matching category
        with GLTTestBase.unittest.mock.patch(
            "os.path.isfile", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open", mock_open()
        ), GLTTestBase.unittest.mock.patch(
            "json.load", return_value=self.sample_hw_manifest
        ), GLTTestBase.unittest.mock.patch.object(
            nv_dev,
            "_NvDev__modalias_parser",
            return_value=[self.sample_dmi_modalias],
        ), GLTTestBase.unittest.mock.patch(
            "re.match", side_effect=mock_re_match
        ):
            self.assertTrue(nv_dev.manifest_skip_func("tpm"))


if __name__ == "__main__":
    GLTTestBase.unittest.main()
