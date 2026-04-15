"""Unit tests for the kselftest module"""

import os
import tempfile
import subprocess
from tests.test_base import GLTTestBase

from src.common.kselftest import Kselftest, main


class TestKselftest(GLTTestBase):
    """Test cases for Kselftest class"""

    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        # Use with statement for creating temporary file
        with tempfile.NamedTemporaryFile(delete=False) as self.temp_log:
            self.temp_log_path = self.temp_log.name
        # File is automatically closed after the with block

        # Create a patcher for os.path.isdir to always return True
        self.isdir_patcher = self.mock.patch(
            "os.path.isdir", return_value=True
        )
        self.isdir_mock = self.isdir_patcher.start()

        # Create a patcher for subprocess.run
        self.subprocess_patcher = self.mock.patch("subprocess.run")
        self.subprocess_mock = self.subprocess_patcher.start()

        # Configure subprocess mock
        self.process_mock = self.mock.MagicMock()
        self.process_mock.stdout = b"Command output"
        self.process_mock.stderr = b""
        self.subprocess_mock.return_value = self.process_mock

    def tearDown(self):
        """Tear down test fixtures"""
        os.unlink(self.temp_log_path)
        self.isdir_patcher.stop()
        self.subprocess_patcher.stop()
        super().tearDown()

    def test_initialization(self):
        """Test Kselftest initialization"""
        kselftest = Kselftest(self.temp_log_path)
        self.assertEqual(
            kselftest.linux_root_directory, "/usr/local/share/greenlit/linux"
        )

        # Check if log file contains the exact reset message
        with open(self.temp_log_path, "r", encoding="latin-1") as log_file:
            log_content = log_file.read()
            self.assertIn("Log file reset at script start.", log_content)

    def test_execute_command_success(self):
        """Test successful command execution"""
        result = Kselftest.execute_command("echo test", "/tmp")
        self.assertTrue(result)
        self.subprocess_mock.assert_called_once()

    def test_execute_command_invalid_dir(self):
        """Test command execution with invalid directory"""
        # Set up isdir to return False for this test
        self.isdir_mock.return_value = False

        result = Kselftest.execute_command("echo test", "/invalid/dir")
        self.assertFalse(result)
        self.subprocess_mock.assert_not_called()

    def test_execute_command_failure(self):
        """Test command execution failure"""
        self.subprocess_mock.side_effect = subprocess.CalledProcessError(
            1, "cmd", output=b"", stderr=b"error"
        )

        result = Kselftest.execute_command("failing_command", "/tmp")
        self.assertFalse(result)

    def test_build_targets_test_success(self):
        """Test building targets successfully"""
        kselftest = Kselftest(self.temp_log_path)

        with self.mock.patch.object(
            kselftest, "execute_command", return_value=True
        ) as mock_exec:
            result = kselftest.build_targets_test("TARGETS='kvm'")
            self.assertTrue(result)
            self.assertEqual(mock_exec.call_count, 2)

    def test_build_targets_test_invalid_target(self):
        """Test building with invalid target"""
        kselftest = Kselftest(self.temp_log_path)
        result = kselftest.build_targets_test("kvm")
        self.assertFalse(result)

    def test_build_targets_test_all(self):
        """Test building all targets"""
        kselftest = Kselftest(self.temp_log_path)

        with self.mock.patch.object(
            kselftest, "execute_command", return_value=True
        ) as mock_exec:
            result = kselftest.build_targets_test("TARGETS=all")
            self.assertTrue(result)
            # Check we're using the constant ALL_KSELFTEST_TARGET_TESTS
            mock_exec.assert_any_call(
                "make -C tools/testing/selftests "
                "TARGETS='size rtc/property kvm'",
                kselftest.linux_root_directory,
            )

    def test_run_targets_test(self):
        """Test running targets"""
        kselftest = Kselftest(self.temp_log_path)

        with self.mock.patch.object(
            kselftest, "execute_command", return_value=True
        ) as mock_exec:
            # Test without root
            result = kselftest.run_targets_test("TARGETS='kvm'")
            self.assertTrue(result)
            mock_exec.assert_called_with(
                "make -C tools/testing/selftests run_tests TARGETS='kvm'",
                kselftest.linux_root_directory,
            )

            # Test with root
            result = kselftest.run_targets_test("TARGETS='kvm'", True)
            self.assertTrue(result)
            mock_exec.assert_called_with(
                "sudo make -C tools/testing/selftests run_tests TARGETS='kvm'",
                kselftest.linux_root_directory,
            )

    def test_check_log_pass(self):
        """Test log checking with passing tests"""
        # Create a mock log file with no failures
        with open(self.temp_log_path, "w", encoding="latin-1") as f:
            f.write("ok 1 - test1\nok 2 - test2\n")

        result = Kselftest.check_log(self.temp_log_path)
        self.assertTrue(result)

    def test_check_log_fail(self):
        """Test log checking with failing tests"""
        # Create a mock log file with failures
        with open(self.temp_log_path, "w", encoding="latin-1") as f:
            f.write("ok 1 - test1\nnot ok 2 - test2\n")

        result = Kselftest.check_log(self.temp_log_path)
        self.assertFalse(result)

    def test_check_log_missing(self):
        """Test log checking with missing log file"""
        result = Kselftest.check_log("/nonexistent/file.log")
        self.assertFalse(result)

    def test_main_success(self):
        """Test main function success path"""
        with self.mock.patch("src.common.kselftest.Kselftest") as mock_class:
            instance = mock_class.return_value
            instance.build_targets_test.return_value = True

            result = main()
            self.assertEqual(result, 0)
            instance.build_targets_test.assert_called_with("TARGETS=all")

    def test_main_failure(self):
        """Test main function failure path"""
        with self.mock.patch("src.common.kselftest.Kselftest") as mock_class:
            instance = mock_class.return_value
            instance.build_targets_test.return_value = False

            result = main()
            self.assertEqual(result, 255)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
