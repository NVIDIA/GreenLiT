"""
Test cases for external test launcher.
"""
import os
import tempfile
from unittest import mock
from tests.test_base import GLTTestBase
from src.common.glt_ext_launcher import discover, CATEGORY_NAME, GLTExtShellExitCode


class TestGLTExtLauncher(GLTTestBase):
    """Test cases for external test launcher."""

    def setUp(self):
        super().setUp()
        # Create a temp directory structure
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root_dir = self.temp_dir.name
        self.log_dir = os.path.join(self.root_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.mock_sd.log_dir = self.log_dir

    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()

    def _create_executable(self, rel_path):
        """Helper to create a fake executable file."""
        abs_path = os.path.join(self.root_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(abs_path, 0o755)
        return abs_path

    @mock.patch("subprocess.run")
    def test_discover_and_execution(self, mock_run):
        """Test discover and execution of external test cases."""
        # Create a structure: a/b.sh, a/c.sh, z.sh, y.sh, d/e/f.sh
        files = ["a/b.sh", "a/c.sh", "z.sh", "y.sh", "w.sh", "d/e/f.sh"]
        for f in files:
            self._create_executable(f)
        # Mock subprocess.run to simulate different return codes

        def fake_run(args, *_, **__):
            if args[0].endswith("b.sh"):
                return mock.Mock(returncode=GLTExtShellExitCode.PASS, stdout=b"pass stdout", stderr=b"pass")
            if args[0].endswith("z.sh"):
                return mock.Mock(returncode=GLTExtShellExitCode.XFAIL, stdout=b"xfail stdout", stderr=b"xfail")
            if args[0].endswith("y.sh"):
                return mock.Mock(returncode=GLTExtShellExitCode.XPASS, stdout=b"xpass stdout", stderr=b"xpass")
            if args[0].endswith("c.sh"):
                return mock.Mock(returncode=GLTExtShellExitCode.SKIP, stdout=b"skip stdout", stderr=b"skip")
            if args[0].endswith("w.sh"):
                return mock.Mock(returncode=127, stdout=b"unexpected stdout", stderr=b"unexpected fail")
            return mock.Mock(returncode=GLTExtShellExitCode.FAIL, stdout=b"fail stdout", stderr=b"fail")
        mock_run.side_effect = fake_run
        # Discover (DFS, a-z)
        tests = discover(self.root_dir, self.root_dir)
        # Should be 5 tests, DFS order: a/b.sh, a/c.sh, d/e/f.sh, y.sh, z.sh
        names = [t.attr.name for t in tests]
        self.assertEqual(names, ["a_b_sh", "a_c_sh", "d_e_f_sh", "w_sh", "y_sh", "z_sh"])
        # Check attributes
        for t in tests:
            self.assertEqual(t.attr.category, CATEGORY_NAME)
            self.assertEqual(t.attr.plans, ["ext_executable"])
            self.assertTrue(callable(t))
        # Simulate running each test
        for t in tests:
            try:
                t(self.mock_sd, self.mock_logger)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        # Check results
        results = [t.attr.result[0] for t in tests]
        expected = ["PASSED", "SKIPPED", "FAILED", "FAILED", "FAILED", "Allowed FAILURE"]
        actual = [r.value for r in results]
        self.assertEqual(actual, expected)
