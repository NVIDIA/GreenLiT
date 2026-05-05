"""Unit test script for common/glt_conf.py"""

import os

from tests.test_base import GLTTestBase
from src.common.glt_conf import cond_to_func, load_glt_yaml_files


class TestGLTConf(GLTTestBase):
    """Test Class for GLT Conf"""

    def setUp(self):
        super().setUp()
        self.test_yaml_dir = os.path.join(self.test_data_dir, "yaml")

    def test_cond_to_func_script(self):
        """Ensure the filter function is working by using bash script"""
        # Test successful script execution
        condition = {"script": "echo 'test'"}
        func = cond_to_func(self.mock_logger, condition)
        self.assertTrue(func())

        # Test failed script execution
        condition = {"script": "exit 1"}
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

        # Test non-existent command
        condition = {"script": "nonexistentcommand"}
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

    @GLTTestBase.unittest.mock.patch(
        "platform.freedesktop_os_release", create=True
    )
    @GLTTestBase.unittest.mock.patch("builtins.open")
    def test_cond_to_func_distro(self, mock_open, mock_freedesktop):
        """Ensure the filter function is working by using distro"""
        release_info = {
            "NAME": "Ubuntu",
            "VERSION_ID": "22.04",
        }

        # Setup mock for platform.freedesktop_os_release
        mock_freedesktop.return_value = release_info

        # In case freedesktop_os_release is not used, also mock file reading
        mock_file = GLTTestBase.unittest.mock.MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.readlines.return_value = [
            'NAME="Ubuntu"\n',
            'VERSION_ID="22.04"\n',
        ]
        mock_open.return_value = mock_file

        # Run the tests
        condition = {"distro": "Ubuntu"}
        func = cond_to_func(self.mock_logger, condition)
        self.assertTrue(func())

        condition = {"distro": "Ubuntu", "release": "22.04"}
        func = cond_to_func(self.mock_logger, condition)
        self.assertTrue(func())

        condition = {"distro": "Ubuntu", "release": "20.04"}
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

    @GLTTestBase.unittest.mock.patch("platform.release")
    @GLTTestBase.unittest.mock.patch(
        "platform.freedesktop_os_release", create=True
    )
    @GLTTestBase.unittest.mock.patch("builtins.open")
    def test_cond_to_func_kernel(
        self, mock_open, mock_freedesktop, mock_platform_release
    ):
        """Ensure the filter function is working by using kernel version"""
        # Setup mocks for OS release info
        release_info = {
            "NAME": "Ubuntu",
            "VERSION_ID": "22.04",
        }
        mock_freedesktop.return_value = release_info

        # In case freedesktop_os_release is not used, also mock file reading
        mock_file = GLTTestBase.unittest.mock.MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.readlines.return_value = [
            'NAME="Ubuntu"\n',
            'VERSION_ID="22.04"\n',
        ]
        mock_open.return_value = mock_file

        # Mock kernel version
        mock_platform_release.return_value = "5.15.0-56-generic"

        # Test kernel version comparison
        test_cases = [
            ({"distro": "Ubuntu", "kernel-uname": ">= 5.15.0"}, True),
            ({"distro": "Ubuntu", "kernel-uname": "> 5.16.0"}, False),
            ({"distro": "Ubuntu", "kernel-uname": "<= 5.15.0-57-generic"}, True),
            ({"distro": "Ubuntu", "kernel-uname": "< 5.14.0"}, False),
            ({"distro": "Ubuntu", "kernel-uname": "== 5.15.0-56-generic"}, True),
        ]

        for condition, expected in test_cases:
            func = cond_to_func(self.mock_logger, condition)
            self.assertEqual(
                func(), expected, f"Failed for condition: {condition}"
            )

    def test_cond_to_func_unsupported(self):
        """Unsupported conditions log a warning and return a False stub"""
        condition = {"foo": "bar"}
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())
        self.mock_logger.warning.assert_called_once()

    def test_load_glt_yaml_files(self):
        """Load the sample yaml file and check fields"""
        # Test successful yaml loading
        with self.mock.patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["00_test.yaml"]

            result = load_glt_yaml_files(self.mock_logger, self.test_yaml_dir)
            self.assertEqual(len(result), 1)
            for _, value in result[0].items():
                for key, test in value.items():
                    self.assertIn(
                        key,
                        ["allow-failure", "conditional-skip"],
                        f"Invalid key: {key}",
                    )
                    for cond in test:
                        for k, v in cond.items():
                            self.assertIn(
                                k, ["condition"], f"Invalid key: {k}"
                            )
                            for k2, _ in v.items():
                                self.assertIn(
                                    k2,
                                    [
                                        "name",
                                        "distro",
                                        "release",
                                        "kernel-uname",
                                        "script",
                                    ],
                                    f"Invalid key: {k2}",
                                )

        # Test file not found
        with self.mock.patch("os.listdir") as mock_listdir, self.mock.patch(
            "builtins.open", side_effect=FileNotFoundError
        ):
            mock_listdir.return_value = ["test.yaml"]

            result = load_glt_yaml_files(self.mock_logger, "/test/dir")
            self.assertEqual(len(result), 0)

        # Test permission error
        with self.mock.patch("os.listdir") as mock_listdir, self.mock.patch(
            "builtins.open", side_effect=PermissionError
        ):
            mock_listdir.return_value = ["test.yaml"]

            result = load_glt_yaml_files(self.mock_logger, "/test/dir")
            self.assertEqual(len(result), 0)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
