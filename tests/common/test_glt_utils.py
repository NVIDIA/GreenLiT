"""Unit test script for common/glt_utils.py"""

import os
import subprocess
import signal

from tests.test_base import GLTTestBase
from src.common.glt_utils import (
    _load_kernel_for_kexec,  # pylint: disable=protected-access
    append_kernel_parameter,
    arm_for_reset,
    check_kernel_log,
    chdir_managed,
    check_kernel_parameter,
    collect_logs,
    compare_version,
    create_cache_session,
    get_cache_session_list,
    get_cpu_model_name,
    is_cli_available,
    parse_linux_version,
    reboot,
    reboot_with_kexec,
    reboot_with_reisub,
    remove_kernel_parameter,
    start_systemd_timer,
    systemd_enable,
)


class TestGLTUtils(GLTTestBase):  # pylint: disable=too-many-public-methods
    """Test Class for GLT Utils"""

    @GLTTestBase.unittest.mock.patch("subprocess.run")
    @GLTTestBase.unittest.mock.patch(
        "src.common.glt_utils.start_systemd_timer"
    )
    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.systemd_enable")
    def test_arm_for_reset(
        self, mock_systemd_enable, mock_start_timer, mock_run
    ):
        """Test arm_for_reset function"""
        mock_testcase = GLTTestBase.unittest.mock.MagicMock()

        result = arm_for_reset(mock_testcase, self.mock_sd)

        self.mock_sd.progress_record_time.assert_called_once_with(
            mock_testcase, "MightReset"
        )
        mock_start_timer.assert_called_once()
        mock_systemd_enable.assert_called_once()
        mock_run.assert_called_once_with(
            ["pkill", "-INT", "glt-test-suite"], shell=False, check=False
        )
        self.assertFalse(result)

    @GLTTestBase.unittest.mock.patch("subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.is_cli_available")
    def test_check_kernel_log(self, mock_is_cli, mock_run):
        """Test check_kernel_log function"""
        test_msg = "test message"

        # Test with journalctl available
        mock_is_cli.side_effect = [True, False]
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = f"some log with {test_msg} in it"
        self.assertTrue(check_kernel_log(test_msg))

        # Test with journalctl unavailable but dmesg available
        mock_is_cli.side_effect = [False, True]
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = f"dmesg output with {test_msg}"
        self.assertTrue(check_kernel_log(test_msg))

        # Test with both commands unavailable
        mock_is_cli.side_effect = [False, False]
        self.assertFalse(check_kernel_log(test_msg))

    def test_collect_logs(self):
        """Test collect_logs function"""
        # Test with None directory
        success, msg = collect_logs(None)
        self.assertFalse(success)
        self.assertEqual(msg, "directory is empty")

        # Test with nvidia-bug-report.sh unavailable
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=False
        ):
            success, msg = collect_logs("/tmp")
            self.assertFalse(success)
            self.assertEqual(msg, "nvidia-bug-report.sh is unavailable")

        # Test successful collection
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            success, msg = collect_logs("/tmp")
            self.assertTrue(success)
            self.assertEqual(msg, "")

    def test_compare_version(self):
        """Test compare_version function with basic comparison"""
        # Mock both dpkg and rpmdev-vercmp as unavailable to test
        # basic comparison
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=False
        ):
            test_cases = [
                ("5.15.0-91-generic", "5.15.0-92-generic", -1),
                ("6.1.0-18-amd64", "6.1.0-17-amd64", 1),
                ("3.10.0-1160.el7", "3.10.0-1160.el7", 0),
                ("5.4.0-100", "5.4.0-99", 1),
                ("4.18.0", "4.19.0", -1),
                # Test cases with different length
                ("5.15.0", "5.15.0-91-generic", -1),
                ("6.1.0-18", "6.1", 1),
                # Test with string vs numeric components
                ("5.4.0-generic", "5.4.0-100", 1),  # 'generic' > 100
                (
                    "5.4.0-generic",
                    "5.4.0-generid",
                    -1,
                ),  # 'generic' < 'generid'
                ("5.4.0-100-generic", "5.4.0-99-amd64", 1),  # 100 > 99
            ]

            for v1, v2, expected in test_cases:
                with self.subTest(method="basic", v1=v1, v2=v2):
                    result = compare_version(v1, v2)
                    self.assertEqual(result, expected)

    def test_compare_version_dpkg(self):
        """Test compare_version with dpkg"""
        # Mock dpkg as available, rpmdev-vercmp as unavailable
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available",
            side_effect=lambda x: x == "dpkg",
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            # Test case 1: v1 > v2
            test_cases = [
                # v1 > v2
                {
                    "versions": ("6.1.0-18-amd64", "6.1.0-17-amd64"),
                    "returncodes": {
                        "gt": 0,
                        "lt": 1,
                        "eq": 1,
                    },  # gt succeeds, others fail
                    "expected": 1,
                },
                # v1 < v2
                {
                    "versions": ("5.15.0-91-generic", "5.15.0-92-generic"),
                    "returncodes": {
                        "gt": 1,
                        "lt": 0,
                        "eq": 1,
                    },  # lt succeeds, others fail
                    "expected": -1,
                },
                # v1 == v2
                {
                    "versions": ("3.10.0-1160.el7", "3.10.0-1160.el7"),
                    "returncodes": {
                        "gt": 1,
                        "lt": 1,
                        "eq": 0,
                    },  # eq succeeds, others fail
                    "expected": 0,
                },
            ]

            for case in test_cases:
                v1, v2 = case["versions"]
                returncodes = case["returncodes"]
                expected = case["expected"]

                with self.subTest(method="dpkg", v1=v1, v2=v2):
                    # Configure mock to return appropriate return codes
                    mock_run.side_effect = [
                        GLTTestBase.unittest.mock.Mock(
                            returncode=returncodes["gt"]
                        ),
                        GLTTestBase.unittest.mock.Mock(
                            returncode=returncodes["lt"]
                        ),
                        GLTTestBase.unittest.mock.Mock(
                            returncode=returncodes["eq"]
                        ),
                    ]

                    result = compare_version(v1, v2)
                    self.assertEqual(result, expected)

                    # Verify dpkg was called with correct arguments
                    self.assertEqual(mock_run.call_count, 3)
                    mock_run.reset_mock()

    def test_compare_version_rpm(self):
        """Test compare_version with rpmdev-vercmp"""
        # Mock dpkg as unavailable, rpmdev-vercmp as available
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available",
            side_effect=lambda x: x
            == "rpmdev-vercmp",  # Only return True for rpmdev-vercmp
        ), GLTTestBase.unittest.mock.patch("subprocess.run") as mock_run:
            test_cases = [
                # v1 == v2
                {
                    "versions": ("3.10.0-1160.el7", "3.10.0-1160.el7"),
                    "returncode": 0,  # Equal
                    "expected": 0,
                },
                # v1 > v2
                {
                    "versions": ("6.1.0-18-amd64", "6.1.0-17-amd64"),
                    "returncode": 11,  # v1 > v2
                    "expected": 1,
                },
                # v1 < v2
                {
                    "versions": ("5.15.0-91-generic", "5.15.0-92-generic"),
                    "returncode": 12,  # v1 < v2
                    "expected": -1,
                },
            ]

            for case in test_cases:
                v1, v2 = case["versions"]
                returncode = case["returncode"]
                expected = case["expected"]

                with self.subTest(method="rpm", v1=v1, v2=v2):
                    mock_run.return_value = GLTTestBase.unittest.mock.Mock(
                        returncode=returncode
                    )

                    result = compare_version(v1, v2)
                    self.assertEqual(result, expected)

                    # Verify rpmdev-vercmp was called with correct arguments
                    mock_run.assert_called_with(
                        ["rpmdev-vercmp", v1, v2], shell=False, check=False
                    )
                    mock_run.reset_mock()

    def test_create_cache_session(self):
        """Test create_cache_session function"""
        fixed_datetime = "2024-01-01-12:00:00"

        # Create a real datetime object for more reliable mocking
        mock_dt = GLTTestBase.unittest.mock.Mock()
        mock_dt.strftime.return_value = fixed_datetime

        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.datetime"
        ) as mock_datetime, GLTTestBase.unittest.mock.patch(
            "os.makedirs"
        ) as mock_makedirs:

            # Ensure the mock datetime.now() returns our fixed mock
            mock_datetime.now.return_value = mock_dt

            cache_dir = create_cache_session()

            # Verify the mock was actually used
            mock_datetime.now.assert_called_once()
            mock_dt.strftime.assert_called_once_with("%Y-%m-%d-%H:%M:%S")

            self.assertEqual(
                cache_dir, f"/var/cache/greenlit/{fixed_datetime}"
            )
            mock_makedirs.assert_called_once_with(cache_dir, exist_ok=True)

    @GLTTestBase.unittest.mock.patch("subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.is_cli_available")
    def test_get_cpu_model_name(self, mock_is_cli, mock_run):
        """Test get_cpu_model_name function"""
        mock_logger = GLTTestBase.unittest.mock.MagicMock()

        # Test when lscpu is not available
        mock_is_cli.return_value = False
        result = get_cpu_model_name(mock_logger)
        self.assertIsNone(result)
        mock_logger.warning.assert_called_once_with(
            "lscpu not found. Install util-linux package."
        )

        # Reset mocks for next tests
        mock_is_cli.return_value = True
        mock_logger.reset_mock()

        # Test when lscpu command fails
        mock_run.return_value = GLTTestBase.unittest.mock.Mock(
            returncode=1, stdout="", stderr="error"
        )
        result = get_cpu_model_name(mock_logger)
        self.assertIsNone(result)

        # Test when lscpu succeeds with model name
        mock_run.return_value = GLTTestBase.unittest.mock.Mock(
            returncode=0,
            stdout="Architecture:          x86_64\n"
                   "Model name:            AMD EPYC Olympus\n"
                   "CPU(s):                64\n",
            stderr=""
        )
        result = get_cpu_model_name(mock_logger)
        self.assertEqual(result, "AMD EPYC Olympus")

        # Test when lscpu succeeds but model name not found
        mock_run.return_value = GLTTestBase.unittest.mock.Mock(
            returncode=0,
            stdout="Architecture:          x86_64\nCPU(s):                64\n",
            stderr=""
        )
        result = get_cpu_model_name(mock_logger)
        self.assertIsNone(result)

    @GLTTestBase.unittest.mock.patch("os.listdir")
    @GLTTestBase.unittest.mock.patch("os.path.isfile")
    @GLTTestBase.unittest.mock.patch("os.path.isdir")
    def test_get_cache_session_list(self, mock_isdir, mock_isfile, mock_listdir):
        """Test get_cache_session_list function"""
        # Test with unsorted input and a non-session directory.
        mock_isdir.return_value = True
        mock_listdir.return_value = ["session2", "not_a_dir", "boot-config", "session1"]
        valid_progress_paths = {
            "/var/cache/greenlit/session1/glt-test-suite/test_progress.json",
            "/var/cache/greenlit/session2/glt-test-suite/test_progress.json",
        }
        mock_isfile.side_effect = lambda x: x in valid_progress_paths

        result = get_cache_session_list()

        # Verify the length
        self.assertEqual(len(result), 2)

        # Verify all paths have correct prefix
        self.assertTrue(
            all(x.startswith("/var/cache/greenlit/") for x in result)
        )

        # Verify the order is sorted
        expected = [
            "/var/cache/greenlit/session1",
            "/var/cache/greenlit/session2",
        ]
        self.assertEqual(result, expected)

    @GLTTestBase.unittest.mock.patch("os.listdir")
    @GLTTestBase.unittest.mock.patch("os.path.isdir")
    def test_get_cache_session_list_missing_root(self, mock_isdir, mock_listdir):
        """Missing GreenLiT cache root means there are no resumable sessions."""
        mock_isdir.return_value = False

        result = get_cache_session_list()

        self.assertEqual(result, [])
        mock_listdir.assert_not_called()

    def test_parse_linux_version(self):
        """Test parse_linux_version function"""
        test_cases = [
            # Standard Ubuntu/Debian style versions
            ("5.15.0-91-generic", [5, 15, 0, 91, "generic"]),
            ("6.1.0-18-amd64", [6, 1, 0, 18, "amd64"]),
            # RHEL/CentOS style versions
            ("3.10.0-1160.108.1.el7", [3, 10, 0, 1160, 108, 1, "el7"]),
            # Simple versions
            ("4.18.0", [4, 18, 0]),
            # Complex versions
            ("5.4.0-100-generic-lpae", [5, 4, 0, 100, "generic", "lpae"]),
            ("4.19.0-23-cloud-amd64", [4, 19, 0, 23, "cloud", "amd64"]),
        ]

        for version_string, expected_parts in test_cases:
            with self.subTest(version=version_string):
                result = parse_linux_version(version_string)
                self.assertEqual(result, expected_parts)

    @GLTTestBase.unittest.mock.patch("shutil.which")
    def test_is_cli_available(self, mock_which):
        """Test is_cli_available function"""
        mock_which.side_effect = ["/usr/bin/test", None]

        self.assertTrue(is_cli_available("test"))
        self.assertFalse(is_cli_available("nonexistent"))

    def test_reboot_with_reisub(self):
        """Test reboot_with_reisub function with and without systemd"""
        mock_process = GLTTestBase.unittest.mock.MagicMock()

        # Test with systemd enabled
        with GLTTestBase.unittest.mock.patch(
            "subprocess.Popen", return_value=mock_process
        ) as mock_popen, GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=mock_process
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.systemd_enable", return_value=True
        ):
            mock_process.communicate.return_value = ("", "")
            mock_process.returncode = 0

            result = reboot_with_reisub(self.mock_logger, self.mock_sd)
            self.assertFalse(result)

            # Verify progress_record_time was called
            self.mock_sd.progress_record_time.assert_called_once_with(
                self.mock_logger, "REISUB"
            )

            expected_cmd = (
                "sh -c 'sleep 5; \\\n"
                "                         echo s > /proc/sysrq-trigger; \\\n"
                "                         echo u > /proc/sysrq-trigger; \\\n"
                "                         echo b > /proc/sysrq-trigger'"
            )

            mock_popen.assert_called_once()
            actual_cmd = mock_popen.call_args[0][0]
            self.assertEqual(actual_cmd, expected_cmd)

        # Test without systemd
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.systemd_enable", return_value=False
        ), GLTTestBase.unittest.mock.patch(
            "os.kill"
        ) as mock_kill, GLTTestBase.unittest.mock.patch(
            "os.getpid", return_value=12345
        ):
            # Reset mock to verify it's called again
            self.mock_sd.progress_record_time.reset_mock()

            result = reboot_with_reisub(self.mock_logger, self.mock_sd)
            self.assertFalse(result)  # Function should return False

            # Verify progress_record_time was called in this case too
            self.mock_sd.progress_record_time.assert_called_once_with(
                self.mock_logger, "REISUB"
            )
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    @GLTTestBase.unittest.mock.patch(
        "src.common.glt_utils._find_extract_vmlinux",
        return_value="/usr/bin/extract-vmlinux",
    )
    @GLTTestBase.unittest.mock.patch(
        "src.common.glt_utils.subprocess.run"
    )
    def test_load_kernel_for_kexec_extracts_unknown_format(
        self, mock_run, _mock_find
    ):
        """An unrecognized vmlinuz is extracted, retried, and removed."""
        extracted_paths = []

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "kexec" and cmd[2].startswith("/boot/"):
                raise subprocess.CalledProcessError(
                    1,
                    cmd,
                    stderr="Cannot determine the file type of /boot/vmlinuz-test",
                )
            if cmd[0] == "/usr/bin/extract-vmlinux":
                extracted_paths.append(kwargs["stdout"].name)
                return self.mock.Mock(returncode=0, stderr=b"")
            return self.mock.Mock(returncode=0)

        mock_run.side_effect = run_side_effect

        loaded, reason = _load_kernel_for_kexec(
            "/boot/vmlinuz-test",
            "/boot/initrd-test",
            "root=test",
        )

        self.assertTrue(loaded)
        self.assertEqual(reason, "")
        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual(len(extracted_paths), 1)
        self.assertEqual(
            mock_run.call_args_list[2].args[0][2],
            extracted_paths[0],
        )
        self.assertFalse(os.path.exists(extracted_paths[0]))

    def test_reboot_with_kexec_stops_when_kernel_load_fails(self):
        """A load failure is returned without triggering a kexec reboot."""
        files = [
            "/boot/initrd.img-test-kernel",
            "/boot/vmlinuz-test-kernel",
        ]
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.platform.release",
            return_value="test-kernel",
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.glob.glob", return_value=files
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open",
            GLTTestBase.unittest.mock.mock_open(
                read_data="BOOT_IMAGE=/boot/vmlinuz-test-kernel root=test"
            ),
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._load_kernel_for_kexec",
            return_value=(False, "extraction failed"),
        ) as mock_loader, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.systemd_enable"
        ) as mock_systemd, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.os.system"
        ) as mock_system:
            loaded, reason = reboot_with_kexec(
                "test_case", self.mock_sd, custom_cmdline="test.option=1"
            )

        self.assertFalse(loaded)
        self.assertEqual(reason, "extraction failed")
        mock_loader.assert_called_once_with(
            "/boot/vmlinuz-test-kernel",
            "/boot/initrd.img-test-kernel",
            (
                "BOOT_IMAGE=/boot/vmlinuz-test-kernel root=test "
                "test.option=1 glt_kexec_boot=1"
            ),
        )
        mock_systemd.assert_not_called()
        mock_system.assert_not_called()
        self.mock_sd.progress_record_time.assert_not_called()

    @GLTTestBase.unittest.mock.patch("os._exit")
    @GLTTestBase.unittest.mock.patch("os.system")
    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.systemd_enable")
    def test_reboot(self, mock_systemd_enable, mock_system, mock_exit):
        """Test reboot function"""
        mock_testcase = GLTTestBase.unittest.mock.MagicMock()

        # Test with systemd enabled
        mock_systemd_enable.return_value = True
        result = reboot(mock_testcase, self.mock_sd)
        self.mock_sd.progress_record_time.assert_called_once_with(
            mock_testcase, "TriggerReboot"
        )
        mock_system.assert_called_once_with("reboot")
        mock_exit.assert_called_once_with(0)
        self.assertFalse(result)

        # Test with systemd disabled
        mock_systemd_enable.return_value = False
        with GLTTestBase.unittest.mock.patch("os.kill") as mock_kill:
            result = reboot(mock_testcase, self.mock_sd)
            mock_kill.assert_called_once()
            self.assertFalse(result)

    @GLTTestBase.unittest.mock.patch("subprocess.run")
    def test_systemd_timer(self, mock_run):
        """Test systemd timer start/stop functions"""
        mock_run.return_value.returncode = 0

        # Test start timer
        self.assertTrue(start_systemd_timer())
        mock_run.assert_called_with(
            ["systemctl", "start", "greenlit.timer"],
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Test stop timer
        self.assertTrue(start_systemd_timer(start=False))
        mock_run.assert_called_with(
            ["systemctl", "stop", "greenlit.timer"],
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @GLTTestBase.unittest.mock.patch("subprocess.run")
    def test_systemd_enable(self, mock_run):
        """Test systemd service enable/disable functions"""
        mock_run.return_value.returncode = 0

        # Test enable service
        self.assertTrue(systemd_enable())
        mock_run.assert_called_with(
            ["systemctl", "enable", "greenlit"],
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Test disable service
        self.assertTrue(systemd_enable(enable=False))
        mock_run.assert_called_with(
            ["systemctl", "disable", "greenlit"],
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.KernelCmdlineManager")
    def test_append_kernel_parameter(self, mock_manager_cls):
        """Test appending kernel parameter through detected backend."""
        mock_manager = GLTTestBase.unittest.mock.MagicMock()
        mock_manager.append_parameter.return_value = GLTTestBase.unittest.mock.Mock(
            ok=True, reason=""
        )
        mock_manager_cls.detect.return_value = mock_manager

        ok, reason = append_kernel_parameter(
            "kvm-arm.vgic_v4_enable=1",
            test_case_name="test_gicv4_irq_scaling",
            session_dir="/session/glt-test-suite",
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        mock_manager_cls.detect.assert_called_once_with(
            transaction_root="/session/glt-test-suite/boot-config"
        )
        mock_manager.append_parameter.assert_called_once_with(
            "kvm-arm.vgic_v4_enable=1",
            transaction_id="test_gicv4_irq_scaling"
        )

    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.KernelCmdlineManager")
    def test_append_kernel_parameter_rejects_invalid_parameter(
        self, mock_manager_cls
    ):
        """Test invalid kernel parameter validation."""
        mock_manager = GLTTestBase.unittest.mock.MagicMock()
        mock_manager.append_parameter.return_value = GLTTestBase.unittest.mock.Mock(
            ok=False, reason="kernel parameter contains invalid characters"
        )
        mock_manager_cls.detect.return_value = mock_manager

        ok, reason = append_kernel_parameter("bad\nparam")

        self.assertFalse(ok)
        self.assertIn("invalid characters", reason)
        mock_manager.append_parameter.assert_called_once_with(
            "bad\nparam", transaction_id=None
        )

    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.KernelCmdlineManager")
    def test_check_kernel_parameter_returns_backend_result(self, mock_manager_cls):
        """Test result-preserving kernel parameter check helper."""
        mock_manager = GLTTestBase.unittest.mock.MagicMock()
        expected = GLTTestBase.unittest.mock.Mock(ok=True, active=True)
        mock_manager.check_parameter.return_value = expected
        mock_manager_cls.detect.return_value = mock_manager

        result = check_kernel_parameter("foo=bar", test_case_name="test")

        self.assertIs(result, expected)
        mock_manager.check_parameter.assert_called_once_with(
            "foo=bar", transaction_id="test"
        )

    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.KernelCmdlineManager")
    def test_check_kernel_parameter_uses_session_dir(self, mock_manager_cls):
        """Test kernel parameter helpers can use the active GreenLiT session."""
        mock_manager = GLTTestBase.unittest.mock.MagicMock()
        mock_manager.check_parameter.return_value = GLTTestBase.unittest.mock.Mock(
            ok=True, reason=""
        )
        mock_manager_cls.detect.return_value = mock_manager

        result = check_kernel_parameter(
            "foo=bar", test_case_name="test", session_dir="/session/glt-test-suite"
        )

        self.assertTrue(result.ok)
        mock_manager_cls.detect.assert_called_once_with(
            transaction_root="/session/glt-test-suite/boot-config"
        )
        mock_manager.check_parameter.assert_called_once_with(
            "foo=bar", transaction_id="test"
        )

    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.KernelCmdlineManager")
    def test_check_kernel_parameter_ignores_empty_session_dir(self, mock_manager_cls):
        """Empty session directories use the backend default transaction root."""
        mock_manager = GLTTestBase.unittest.mock.MagicMock()
        mock_manager.check_parameter.return_value = GLTTestBase.unittest.mock.Mock(
            ok=True, reason=""
        )
        mock_manager_cls.detect.return_value = mock_manager

        result = check_kernel_parameter("foo=bar", test_case_name="test", session_dir=" ")

        self.assertTrue(result.ok)
        mock_manager_cls.detect.assert_called_once_with(transaction_root=None)

    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.KernelCmdlineManager")
    def test_remove_kernel_parameter_from_tag(
        self, mock_manager_cls
    ):
        """Test removing kernel parameters through detected backend."""
        mock_manager = GLTTestBase.unittest.mock.MagicMock()
        mock_manager.remove_parameter.return_value = GLTTestBase.unittest.mock.Mock(
            ok=True, reason=""
        )
        mock_manager_cls.detect.return_value = mock_manager

        ok, reason = remove_kernel_parameter(
            "foo=bar", test_case_name="test", session_dir="/session/glt-test-suite"
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        mock_manager_cls.detect.assert_called_once_with(
            transaction_root="/session/glt-test-suite/boot-config"
        )
        mock_manager.remove_parameter.assert_called_once_with(
            "foo=bar", transaction_id="test"
        )

    def test_chdir_managed(self):
        """Test chdir_managed context manager function"""
        # Create mock directories to test with
        original_dir = "/original/path"
        target_dir = "/target/path"

        with GLTTestBase.unittest.mock.patch("os.getcwd", return_value=original_dir) as mock_getcwd, \
                GLTTestBase.unittest.mock.patch("os.chdir") as mock_chdir, \
                GLTTestBase.unittest.mock.patch.dict("os.environ", {"PWD": original_dir}):

            # Test the context manager
            with chdir_managed(target_dir):
                # Verify we changed to the target directory
                mock_chdir.assert_called_with(target_dir)
                # getcwd() is called twice: once to save old_cwd, once to set PWD
                self.assertEqual(mock_getcwd.call_count, 2)
                # Verify PWD was set
                self.assertEqual(os.environ['PWD'], original_dir)

                # Reset the mock to verify the exit behavior separately
                mock_chdir.reset_mock()

            # Verify we returned to the original directory after exiting the context
            mock_chdir.assert_called_with(original_dir)
            # Verify PWD was restored
            self.assertEqual(os.environ['PWD'], original_dir)

            # Test with exception inside the context
            mock_chdir.reset_mock()
            try:
                with chdir_managed(target_dir):
                    # Verify directory change occurred
                    mock_chdir.assert_called_with(target_dir)
                    mock_chdir.reset_mock()

                    # Raise an exception to ensure cleanup still happens
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Verify we still returned to the original directory despite the exception
            mock_chdir.assert_called_with(original_dir)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
