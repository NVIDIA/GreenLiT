"""Unit test script for common/glt_utils.py"""
# pylint: disable=duplicate-code,too-many-lines

import os
import subprocess
import signal

from tests.test_base import GLTTestBase
from src.common import glt_utils as glt_utils_module
from src.common.glt_utils import (
    _find_kexec_boot_files,  # pylint: disable=protected-access
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
    run_command,
    start_systemd_timer,
    systemd_enable,
)


def _command_failure(command, stderr):
    """Return the failure raised by subprocess.run(check=True)."""
    return subprocess.CalledProcessError(
        returncode=1,
        cmd=command,
        output="",
        stderr=stderr,
    )


JOURNAL_LIST_BOOTS_OUTPUT = """
IDX BOOT ID                          FIRST ENTRY                 LAST ENTRY
 -1 ABCDEF0123456789ABCDEF0123456789 Tue 2026-09-08 10:00:00 UTC Tue 2026-09-08 10:05:00 UTC
  0 01234567-89AB-CDEF-0123-456789ABCDEF Tue 2026-09-08 10:10:00 UTC Tue 2026-09-08 10:15:00 UTC
"""


class TestGLTUtils(GLTTestBase):  # pylint: disable=too-many-public-methods
    """Test Class for GLT Utils"""

    def test_kernel_log_checkpoint_has_public_type_alias(self):
        """Expose the opaque checkpoint union through a public type name."""
        self.assertIsNotNone(
            getattr(glt_utils_module, "KernelLogCheckpoint", None)
        )

    def test_module_unload_unavailable_reason_accepts_enabled_config(self):
        """Both module-loading and unloading booleans must be enabled."""
        with GLTTestBase.unittest.mock.patch.object(
            glt_utils_module.platform, "release", return_value="probe-kernel"
        ), GLTTestBase.unittest.mock.patch.object(
            glt_utils_module.check_kernel_config, "parse_kconfig",
            return_value={"CONFIG_MODULES": "y", "CONFIG_MODULE_UNLOAD": "y"},
        ) as config_reader:
            self.assertEqual(glt_utils_module.module_unload_unavailable_reason(), "")

        config_reader.assert_called_once_with("/boot/config-probe-kernel")

    def test_module_unload_unavailable_reason_rejects_disabled_or_invalid_config(self):
        """Missing, disabled and invalid booleans cannot authorize a temporary probe."""
        for option in ("CONFIG_MODULES", "CONFIG_MODULE_UNLOAD"):
            for value in ("n", "m", None, "invalid", True):
                config = {"CONFIG_MODULES": "y", "CONFIG_MODULE_UNLOAD": "y"}
                if value is None:
                    del config[option]
                else:
                    config[option] = value
                with self.subTest(option=option, value=value), GLTTestBase.unittest.mock.patch.object(
                    glt_utils_module.check_kernel_config, "parse_kconfig", return_value=config,
                ):
                    reason = glt_utils_module.module_unload_unavailable_reason()

                self.assertIn(option + "=y", reason)
                self.assertIn("use a kernel with module loading and unloading enabled", reason)

    def test_module_unload_unavailable_reason_reports_config_read_errors(self):
        """An unavailable or invalid config leaves unload support unverified."""
        errors = (
            FileNotFoundError("missing config"), PermissionError("unreadable config"),
            OSError("config read failed"), ValueError("invalid config"),
        )
        for error in errors:
            with self.subTest(error=error), GLTTestBase.unittest.mock.patch.object(
                glt_utils_module.platform, "release", return_value="probe-kernel"
            ), GLTTestBase.unittest.mock.patch.object(
                glt_utils_module.check_kernel_config, "parse_kconfig", side_effect=error,
            ):
                reason = glt_utils_module.module_unload_unavailable_reason()

            self.assertIn("/boot/config-probe-kernel", reason)
            self.assertIn(str(error), reason)
            self.assertIn("provide a readable running-kernel configuration", reason)

    def test_parse_journal_boots_normalizes_exact_boot_ids(self):
        """Parse retained indexes and normalize exact boot IDs."""
        boots = glt_utils_module._parse_journal_boots(  # pylint: disable=protected-access
            JOURNAL_LIST_BOOTS_OUTPUT
        )

        self.assertEqual(
            boots,
            [
                glt_utils_module.JournalBoot(
                    index=-1,
                    boot_id="abcdef0123456789abcdef0123456789",
                ),
                glt_utils_module.JournalBoot(
                    index=0,
                    boot_id="01234567-89ab-cdef-0123-456789abcdef",
                ),
            ],
        )

    def test_parse_journal_boots_rejects_malformed_lines(self):
        """Surface malformed journal indexes, IDs, and short records."""
        cases = (
            ("0\n", "output"),
            (
                "not-an-index 0123456789abcdef0123456789abcdef date\n",
                "index",
            ),
            ("0 not-a-boot-id date\n", "boot ID"),
        )

        for output, expected_message in cases:
            with self.subTest(output=output), self.assertRaises(
                RuntimeError
            ) as context:
                glt_utils_module._parse_journal_boots(  # pylint: disable=protected-access
                    output
                )

            self.assertIn(expected_message, str(context.exception))
            self.assertIn("line 1", str(context.exception))

    def test_list_journal_boots_uses_supported_arguments(self):
        """List retained boots without consulting current boot state."""
        result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout=JOURNAL_LIST_BOOTS_OUTPUT,
            stderr="",
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=result
        ) as mock_run, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._read_boot_id"
        ) as mock_read_boot_id:
            boots = glt_utils_module.list_journal_boots()

        self.assertEqual(len(boots), 2)
        mock_read_boot_id.assert_not_called()
        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "--list-boots"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_list_journal_boots_rejects_empty_history(self):
        """Do not turn an empty journal history into a clean observation."""
        result = subprocess.CompletedProcess(
            ["journalctl"], 0, stdout="", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=result
        ), self.assertRaisesRegex(RuntimeError, "no retained boot records"):
            glt_utils_module.list_journal_boots()

    def test_list_journal_boots_propagates_command_failure(self):
        """Report journal boot-enumeration failures."""
        journal_error = _command_failure(["journalctl"], "journal error")

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=journal_error
        ), self.assertRaisesRegex(
            RuntimeError,
            "Failed to enumerate retained journal boots with journalctl",
        ):
            glt_utils_module.list_journal_boots()

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

    def test_capture_kernel_log_checkpoint_rejects_bad_journal_record(self):
        """Reject a checkpoint when journalctl cannot provide a valid cursor."""
        bad_results = [
            _command_failure(["journalctl"], "journal error"),
            subprocess.CompletedProcess(
                ["journalctl"], 0, stdout="", stderr=""
            ),
            subprocess.CompletedProcess(
                ["journalctl"], 0, stdout="not json", stderr=""
            ),
            subprocess.CompletedProcess(
                ["journalctl"], 0, stdout="[]", stderr=""
            ),
            subprocess.CompletedProcess(
                ["journalctl"], 0, stdout='{"MESSAGE": "missing cursor"}', stderr=""
            ),
        ]

        for bad_result in bad_results:
            with self.subTest(result=bad_result), GLTTestBase.unittest.mock.patch(
                "builtins.open",
                GLTTestBase.unittest.mock.mock_open(read_data="boot-1\n"),
            ), GLTTestBase.unittest.mock.patch(
                "subprocess.run", side_effect=[bad_result]
            ):
                with self.assertRaises(RuntimeError):
                    glt_utils_module.capture_kernel_log_checkpoint()

    def test_capture_kernel_log_checkpoint_raises_when_journalctl_is_missing(self):
        """Reject a checkpoint when journalctl is not installed."""
        missing_error = FileNotFoundError(
            2, "No such file or directory", "journalctl"
        )

        with GLTTestBase.unittest.mock.patch(
            "builtins.open",
            GLTTestBase.unittest.mock.mock_open(read_data="boot-1\n"),
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=missing_error
        ) as mock_run:
            with self.assertRaises(RuntimeError) as context:
                glt_utils_module.capture_kernel_log_checkpoint()

        self.assertIn("Failed to run journalctl", str(context.exception))
        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "-b0", "-k", "-n1", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_capture_kernel_log_checkpoint_uses_last_journal_cursor(self):
        """Capture and reuse the exact last current-boot journal cursor."""
        capture_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-1", "MESSAGE": "old"}\n',
            stderr="",
        )
        validation_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-1", "MESSAGE": "old"}\n',
            stderr="",
        )
        after_result = subprocess.CompletedProcess(
            ["journalctl"], 0, stdout="new one\nnew two\n", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "builtins.open",
            GLTTestBase.unittest.mock.mock_open(read_data="boot-1\n"),
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run",
            side_effect=[
                capture_result,
                validation_result,
                after_result,
            ],
        ) as mock_run:
            checkpoint = glt_utils_module.capture_kernel_log_checkpoint()
            result = glt_utils_module.get_kernel_log(checkpoint)

        self.assertEqual(result, "new one\nnew two\n")
        self.assertEqual(
            mock_run.call_args_list,
            [
                GLTTestBase.unittest.mock.call(
                    [
                        "journalctl",
                        "--no-pager",
                        "-b0",
                        "-k",
                        "-n1",
                        "-o",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30.0,
                ),
                GLTTestBase.unittest.mock.call(
                    [
                        "journalctl",
                        "--no-pager",
                        "-b0",
                        "-k",
                        "--cursor=cursor-1",
                        "-n1",
                        "-o",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30.0,
                ),
                GLTTestBase.unittest.mock.call(
                    [
                        "journalctl",
                        "--no-pager",
                        "-b0",
                        "-k",
                        "--after-cursor=cursor-1",
                        "-o",
                        "cat",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30.0,
                ),
            ],
        )

    def test_check_kernel_log_does_not_interpolate_message(self):
        """Keep caller-controlled text out of kernel-log commands."""
        test_msg = '$(touch /tmp/not-created) "quoted"'
        journal_result = subprocess.CompletedProcess(
            ["journalctl"], 0, stdout=f"prefix {test_msg} suffix", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=journal_result
        ) as mock_run:
            self.assertTrue(check_kernel_log(test_msg))

        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "-b0", "-k", "-o", "cat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_check_kernel_log_forwards_checkpoint(self):
        """Forward an opaque checkpoint to the retrieval helper."""
        checkpoint = object()

        with GLTTestBase.unittest.mock.patch.object(
            glt_utils_module, "get_kernel_log", return_value="needle", create=True
        ) as mock_get_log, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=False
        ):
            self.assertTrue(check_kernel_log("needle", checkpoint=checkpoint))

        mock_get_log.assert_called_once_with(checkpoint)

    def test_check_kernel_log_propagates_retrieval_errors(self):
        """Do not turn retrieval failures into a false no-match result."""
        with GLTTestBase.unittest.mock.patch.object(
            glt_utils_module,
            "get_kernel_log",
            side_effect=RuntimeError("log unavailable"),
            create=True,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=False
        ):
            with self.assertRaises(RuntimeError):
                check_kernel_log("needle")

    def test_check_kernel_log_uses_case_sensitive_literal_membership(self):
        """Preserve case-sensitive literal substring matching."""
        with GLTTestBase.unittest.mock.patch.object(
            glt_utils_module,
            "get_kernel_log",
            return_value="Alpha [literal] entry",
            create=True,
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=False
        ):
            self.assertTrue(check_kernel_log("Alpha [literal]"))
            self.assertFalse(check_kernel_log("alpha [literal]"))

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

    def test_get_kernel_log_after_journal_checkpoint_rejects_boot_change(self):
        """Reject a journal checkpoint captured before the current boot."""
        capture_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-1", "MESSAGE": "old"}',
            stderr="",
        )
        boot_file = GLTTestBase.unittest.mock.mock_open()
        boot_file.return_value.read.side_effect = [
            "boot-1\n",
            "boot-2\n",
        ]

        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open", boot_file
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=capture_result
        ) as mock_run:
            checkpoint = glt_utils_module.capture_kernel_log_checkpoint()
            with self.assertRaises(RuntimeError):
                glt_utils_module.get_kernel_log(checkpoint)

        self.assertEqual(mock_run.call_count, 1)

    def test_get_kernel_log_after_journal_checkpoint_rejects_invalid_input(self):
        """Reject values that are not opaque kernel-log checkpoints."""
        with self.assertRaises(RuntimeError) as context:
            glt_utils_module.get_kernel_log(object())
        self.assertIn("unexpected type", str(context.exception))
        self.assertIn("object", str(context.exception))

    def test_get_kernel_log_after_journal_checkpoint_rejects_moved_cursor(self):
        """Reject a cursor seek that resolves to a different journal record."""
        capture_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-1", "MESSAGE": "old"}',
            stderr="",
        )
        moved_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-2", "MESSAGE": "nearby"}',
            stderr="",
        )

        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open",
            GLTTestBase.unittest.mock.mock_open(read_data="boot-1\n"),
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=[capture_result, moved_result]
        ) as mock_run:
            checkpoint = glt_utils_module.capture_kernel_log_checkpoint()
            with self.assertRaises(RuntimeError):
                glt_utils_module.get_kernel_log(checkpoint)

        self.assertEqual(mock_run.call_count, 2)

    def test_parse_journal_cursor_reports_distinct_failures(self):
        """Parser RuntimeErrors must distinguish JSON, type, and cursor problems."""
        parse = glt_utils_module._parse_journal_cursor  # pylint: disable=protected-access

        with self.assertRaises(RuntimeError) as invalid_json:
            parse("not json")
        self.assertIn("not valid JSON", str(invalid_json.exception))

        with self.assertRaises(RuntimeError) as not_object:
            parse("[]")
        self.assertIn("must be an object", str(not_object.exception))
        self.assertIn("list", str(not_object.exception))

        with self.assertRaises(RuntimeError) as missing_cursor:
            parse('{"MESSAGE": "no cursor"}')
        self.assertIn("missing a non-empty __CURSOR", str(missing_cursor.exception))

        with self.assertRaises(RuntimeError) as wrong_type:
            parse(None)  # type: ignore[arg-type]
        self.assertIn("not text that can be parsed as JSON", str(wrong_type.exception))

    def test_get_kernel_log_after_journal_checkpoint_rejects_validation_errors(self):
        """Reject failed or malformed exact-cursor validation."""
        capture_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-1", "MESSAGE": "old"}',
            stderr="",
        )
        cases = (
            (
                _command_failure(["journalctl"], "journal error"),
                "Failed to validate the journal checkpoint",
            ),
            (
                subprocess.CompletedProcess(
                    ["journalctl"], 0, stdout="not json", stderr=""
                ),
                "not valid JSON",
            ),
            (
                subprocess.CompletedProcess(
                    ["journalctl"], 0, stdout="[]", stderr=""
                ),
                "must be an object",
            ),
            (
                subprocess.CompletedProcess(
                    ["journalctl"], 0, stdout='{"MESSAGE": "no cursor"}', stderr=""
                ),
                "missing a non-empty __CURSOR",
            ),
        )

        for bad_result, expected_message in cases:
            with self.subTest(result=bad_result), GLTTestBase.unittest.mock.patch(
                "src.common.glt_utils.is_cli_available", return_value=True
            ), GLTTestBase.unittest.mock.patch(
                "builtins.open",
                GLTTestBase.unittest.mock.mock_open(read_data="boot-1\n"),
            ), GLTTestBase.unittest.mock.patch(
                "subprocess.run", side_effect=[capture_result, bad_result]
            ):
                checkpoint = glt_utils_module.capture_kernel_log_checkpoint()
                with self.assertRaises(RuntimeError) as context:
                    glt_utils_module.get_kernel_log(checkpoint)
            self.assertIn(expected_message, str(context.exception))

    def test_get_kernel_log_after_journal_checkpoint_returns_empty(self):
        """Return an empty string when no journal record follows the cursor."""
        cursor_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-1", "MESSAGE": "old"}',
            stderr="",
        )
        empty_result = subprocess.CompletedProcess(
            ["journalctl"], 0, stdout="", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "builtins.open",
            GLTTestBase.unittest.mock.mock_open(read_data="boot-1\n"),
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run",
            side_effect=[
                cursor_result,
                cursor_result,
                empty_result,
            ],
        ):
            checkpoint = glt_utils_module.capture_kernel_log_checkpoint()
            result = glt_utils_module.get_kernel_log(checkpoint)

        self.assertEqual(result, "")

    def test_get_kernel_log_after_journal_checkpoint_raises_on_query_failure(self):
        """Raise when querying records strictly after a valid cursor fails."""
        cursor_result = subprocess.CompletedProcess(
            ["journalctl"],
            0,
            stdout='{"__CURSOR": "cursor-1", "MESSAGE": "old"}',
            stderr="",
        )
        failed_error = _command_failure(["journalctl"], "journal error")

        with GLTTestBase.unittest.mock.patch(
            "builtins.open",
            GLTTestBase.unittest.mock.mock_open(read_data="boot-1\n"),
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run",
            side_effect=[cursor_result, cursor_result, failed_error],
        ):
            checkpoint = glt_utils_module.capture_kernel_log_checkpoint()
            with self.assertRaises(RuntimeError):
                glt_utils_module.get_kernel_log(checkpoint)

    def test_get_kernel_log_after_checkpoint_rejects_malformed_boot_id(self):
        """Reject malformed checkpoint boot IDs before reading the system boot."""
        checkpoint_type = getattr(
            glt_utils_module, "_JournalKernelLogCheckpoint"
        )
        cases = (
            ("", "boot_id is empty"),
            (1, "boot_id must be a string"),
        )

        for boot_id, expected_message in cases:
            checkpoint = checkpoint_type(
                boot_id=boot_id,
                cursor="cursor-1",
            )
            with self.subTest(boot_id=boot_id), GLTTestBase.unittest.mock.patch(
                "src.common.glt_utils._read_boot_id",
                return_value="boot-1",
            ) as mock_read_boot_id:
                with self.assertRaises(RuntimeError) as context:
                    glt_utils_module.get_kernel_log(checkpoint)
            self.assertIn(expected_message, str(context.exception))
            mock_read_boot_id.assert_not_called()

    def test_get_kernel_log_rejects_malformed_journal_checkpoint_data(self):
        """Reject empty or wrongly typed cursors before constructing a command."""
        checkpoint_type = getattr(
            glt_utils_module, "_JournalKernelLogCheckpoint"
        )
        cases = (
            ("", "cursor is empty"),
            (1, "cursor must be a string"),
        )

        for cursor, expected_message in cases:
            checkpoint = checkpoint_type(
                boot_id="boot-1",
                cursor=cursor,
            )
            with self.subTest(cursor=cursor), GLTTestBase.unittest.mock.patch(
                "src.common.glt_utils._read_boot_id",
                return_value="boot-1",
            ), GLTTestBase.unittest.mock.patch(
                "src.common.glt_utils.run_command",
            ) as mock_run:
                with self.assertRaises(RuntimeError) as context:
                    glt_utils_module.get_kernel_log(checkpoint)
            self.assertIn(expected_message, str(context.exception))
            mock_run.assert_not_called()

    def test_get_kernel_log_returns_selected_boot_journal_text(self):
        """Retrieve complete kernel logs by exact compact or UUID boot ID."""
        boot_ids = (
            "ABCDEF0123456789ABCDEF0123456789",
            "01234567-89AB-CDEF-0123-456789ABCDEF",
        )

        for boot_id in boot_ids:
            journal_result = subprocess.CompletedProcess(
                ["journalctl"],
                0,
                stdout="first message\nsecond message\n",
                stderr="",
            )
            with self.subTest(boot_id=boot_id), GLTTestBase.unittest.mock.patch(
                "subprocess.run", return_value=journal_result
            ) as mock_run:
                result = glt_utils_module.get_kernel_log(boot_id=boot_id)

            self.assertEqual(result, "first message\nsecond message\n")
            mock_run.assert_called_once_with(
                [
                    "journalctl",
                    "--no-pager",
                    f"--boot={boot_id}",
                    "-k",
                    "-o",
                    "cat",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30.0,
            )

    def test_get_kernel_log_rejects_invalid_selected_boot_id(self):
        """Reject ambiguous or malformed retained-boot selectors."""
        cases = (
            ("", "boot_id is empty"),
            (1, "boot_id must be a string"),
            (" ", "complete 128-bit boot ID"),
            ("0", "complete 128-bit boot ID"),
            ("-1", "complete 128-bit boot ID"),
            ("g" * 32, "complete 128-bit boot ID"),
            ("0" * 31, "complete 128-bit boot ID"),
        )

        for boot_id, expected_message in cases:
            with self.subTest(boot_id=boot_id), GLTTestBase.unittest.mock.patch(
                "src.common.glt_utils.run_command"
            ) as mock_run, GLTTestBase.unittest.mock.patch(
                "src.common.glt_utils._read_boot_id"
            ) as mock_read_boot_id:
                with self.assertRaises(RuntimeError) as context:
                    glt_utils_module.get_kernel_log(boot_id=boot_id)

            self.assertIn(expected_message, str(context.exception))
            mock_run.assert_not_called()
            mock_read_boot_id.assert_not_called()

    def test_get_kernel_log_rejects_checkpoint_with_selected_boot(self):
        """Reject conflicting log windows before reading any system state."""
        checkpoint_type = getattr(
            glt_utils_module, "_JournalKernelLogCheckpoint"
        )
        checkpoint = checkpoint_type(boot_id="boot-1", cursor="cursor-1")

        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.run_command"
        ) as mock_run, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._read_boot_id"
        ) as mock_read_boot_id:
            with self.assertRaises(RuntimeError) as context:
                glt_utils_module.get_kernel_log(
                    checkpoint,
                    boot_id="0123456789abcdef0123456789abcdef",
                )

        self.assertIn("mutually exclusive", str(context.exception))
        mock_run.assert_not_called()
        mock_read_boot_id.assert_not_called()

    def test_get_kernel_log_rejects_empty_selected_boot_journal(self):
        """Do not interpret an unreadable retained boot as signature-free."""
        boot_id = "0123456789abcdef0123456789abcdef"
        empty_journal = subprocess.CompletedProcess(
            ["journalctl"], 0, stdout="", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=empty_journal
        ):
            with self.assertRaises(RuntimeError) as context:
                glt_utils_module.get_kernel_log(boot_id=boot_id)

        self.assertIn(
            f"no kernel records for boot {boot_id}",
            str(context.exception),
        )

    def test_get_kernel_log_selected_boot_propagates_journal_failure(self):
        """Report retained-boot journal retrieval failures."""
        boot_id = "0123456789abcdef0123456789abcdef"
        journal_error = _command_failure(["journalctl"], "journal error")

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=journal_error
        ):
            with self.assertRaises(RuntimeError) as context:
                glt_utils_module.get_kernel_log(boot_id=boot_id)

        self.assertIn(f"boot {boot_id}", str(context.exception))
        self.assertIn("journalctl", str(context.exception))

    def test_get_kernel_log_raises_when_journal_command_fails(self):
        """Propagate journalctl failure without trying another backend."""
        journal_error = _command_failure(["journalctl"], "journal error")

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=journal_error
        ) as mock_run:
            with self.assertRaises(RuntimeError) as context:
                glt_utils_module.get_kernel_log()

        self.assertIn("journalctl", str(context.exception))
        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "-b0", "-k", "-o", "cat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_get_kernel_log_raises_when_journalctl_is_missing(self):
        """Report a missing journalctl instead of reading another backend."""
        missing_error = FileNotFoundError(
            2, "No such file or directory", "journalctl"
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=missing_error
        ) as mock_run:
            with self.assertRaises(RuntimeError) as context:
                glt_utils_module.get_kernel_log()

        self.assertIn("Failed to run journalctl", str(context.exception))
        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "-b0", "-k", "-o", "cat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_get_kernel_log_raises_on_empty_journal(self):
        """Reject an empty current-boot read instead of reporting no match."""
        empty_journal = subprocess.CompletedProcess(
            ["journalctl"], 0, stdout="", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=empty_journal
        ) as mock_run:
            with self.assertRaises(RuntimeError) as context:
                glt_utils_module.get_kernel_log()
            with self.assertRaises(RuntimeError):
                check_kernel_log("searched signature")

        self.assertIn("no current-boot kernel records", str(context.exception))
        self.assertEqual(mock_run.call_count, 2)

    def test_get_kernel_log_raises_on_command_timeout(self):
        """Map a stalled journalctl query to RuntimeError."""
        timeout_error = subprocess.TimeoutExpired(
            ["journalctl"], 30.0
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=timeout_error
        ) as mock_run:
            with self.assertRaises(RuntimeError) as ctx:
                glt_utils_module.get_kernel_log()

        self.assertIn("journalctl", str(ctx.exception))
        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "-b0", "-k", "-o", "cat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_get_kernel_log_raises_on_decode_failure(self):
        """Map undecodable journalctl output to RuntimeError."""
        decode_error = UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, "invalid start byte"
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", side_effect=decode_error
        ) as mock_run:
            with self.assertRaises(RuntimeError) as ctx:
                glt_utils_module.get_kernel_log()

        self.assertIn("journalctl", str(ctx.exception))
        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "-b0", "-k", "-o", "cat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_get_kernel_log_returns_journal_text_exactly(self):
        """Return message-only current-boot journal text unchanged."""
        journal_result = subprocess.CompletedProcess(
            ["journalctl"], 0, stdout="first message\nsecond message\n", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=journal_result
        ) as mock_run:
            result = glt_utils_module.get_kernel_log()

        self.assertEqual(result, "first message\nsecond message\n")
        mock_run.assert_called_once_with(
            ["journalctl", "--no-pager", "-b0", "-k", "-o", "cat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

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
            "src.common.glt_utils.is_systemd_running", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.systemd_enable", return_value=True
        ):
            mock_process.communicate.return_value = ("", "")
            mock_process.returncode = 0

            result = reboot_with_reisub(self.mock_logger, self.mock_sd)
            self.assertEqual(result, (True, None))

            # Verify progress_record_time was called
            self.mock_sd.progress_record_time.assert_called_once_with(
                self.mock_logger, "REISUB"
            )

            mock_popen.assert_called_once()
            actual_cmd = mock_popen.call_args[0][0]
            self.assertIn("sleep 5", actual_cmd)
            for action in ("s", "u", "b"):
                self.assertIn(
                    f"echo {action} > /proc/sysrq-trigger", actual_cmd
                )

        # Test without systemd
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_systemd_running", return_value=False
        ), GLTTestBase.unittest.mock.patch(
            "os.kill"
        ) as mock_kill, GLTTestBase.unittest.mock.patch(
            "os.getpid", return_value=12345
        ):
            # Reset mock to verify it's called again
            self.mock_sd.progress_record_time.reset_mock()

            result = reboot_with_reisub(self.mock_logger, self.mock_sd)
            self.assertEqual(
                result,
                (
                    False,
                    "SIGTERM did not terminate the process while "
                    "emulating shutdown",
                ),
            )

            self.mock_sd.progress_record_time.assert_not_called()
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)

        # systemd runs but the unit cannot be enabled: do not kill the suite
        self.mock_sd.progress_record_time.reset_mock()
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_systemd_running", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.systemd_enable", return_value=False
        ), GLTTestBase.unittest.mock.patch(
            "subprocess.Popen"
        ) as mock_popen, GLTTestBase.unittest.mock.patch("os.kill") as mock_kill:
            ok, reason = reboot_with_reisub(self.mock_logger, self.mock_sd)
            self.assertFalse(ok)
            self.assertIn("greenlit.service", reason)
            self.mock_sd.progress_record_time.assert_not_called()
            mock_popen.assert_not_called()
            mock_kill.assert_not_called()

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

    def _find_kexec_boot_files_in(self, files):
        """Run _find_kexec_boot_files() against a fake /boot listing."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.glob.glob", return_value=files
        ):
            return _find_kexec_boot_files("6.12.0-160100.1-default")

    def test_find_kexec_boot_files_accepts_image(self):
        """arm64 kernels can be installed as Image-<release>."""
        kernel, initrd = self._find_kexec_boot_files_in(
            [
                "/boot/config-6.12.0-160100.1-default",
                "/boot/Image-6.12.0-160100.1-default",
                "/boot/initrd-6.12.0-160100.1-default",
                "/boot/symvers-6.12.0-160100.1-default.gz",
                "/boot/sysctl.conf-6.12.0-160100.1-default",
                "/boot/System.map-6.12.0-160100.1-default",
                "/boot/vmlinux-6.12.0-160100.1-default.gz",
            ]
        )

        self.assertEqual(kernel, "/boot/Image-6.12.0-160100.1-default")
        self.assertEqual(initrd, "/boot/initrd-6.12.0-160100.1-default")

    def test_find_kexec_boot_files_prefers_vmlinuz_over_image(self):
        """vmlinuz keeps priority when both names are present."""
        kernel, _ = self._find_kexec_boot_files_in(
            [
                "/boot/Image-6.12.0-160100.1-default",
                "/boot/vmlinuz-6.12.0-160100.1-default",
                "/boot/initrd-6.12.0-160100.1-default",
            ]
        )

        self.assertEqual(kernel, "/boot/vmlinuz-6.12.0-160100.1-default")

    def test_reboot_with_kexec_loads_image(self):
        """An Image-<release> kernel reaches the kexec loader."""
        files = [
            "/boot/initrd-test-kernel",
            "/boot/Image-test-kernel",
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
                read_data="BOOT_IMAGE=/boot/Image-test-kernel root=test"
            ),
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._load_kernel_for_kexec",
            return_value=(False, "stop here"),
        ) as mock_loader:
            loaded, reason = reboot_with_kexec("test_case", self.mock_sd)

        self.assertFalse(loaded)
        self.assertEqual(reason, "stop here")
        mock_loader.assert_called_once()
        self.assertEqual(
            mock_loader.call_args.args[:2],
            ("/boot/Image-test-kernel", "/boot/initrd-test-kernel"),
        )

    def test_reboot_with_kexec_reports_missing_kernel_image(self):
        """The error names both accepted kernel image names."""
        with GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_cli_available", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.platform.release",
            return_value="test-kernel",
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.glob.glob",
            return_value=["/boot/initrd-test-kernel"],
        ):
            loaded, reason = reboot_with_kexec("test_case", self.mock_sd)

        self.assertFalse(loaded)
        self.assertEqual(
            reason, "No matching vmlinuz or Image in /boot for current kernel"
        )

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

    def test_reboot_with_kexec_reports_service_that_cannot_be_enabled(self):
        """A masked greenlit.service is reported, not silently fatal."""
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
            GLTTestBase.unittest.mock.mock_open(read_data="1"),
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._load_kernel_for_kexec",
            return_value=(True, ""),
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_systemd_running", return_value=True
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.systemd_enable", return_value=False
        ) as mock_systemd, GLTTestBase.unittest.mock.patch(
            "os._exit"
        ) as mock_exit, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._unload_kernel_for_kexec",
            return_value=(True, ""),
        ) as mock_unload, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.os.system"
        ) as mock_system, GLTTestBase.unittest.mock.patch(
            "os.kill"
        ) as mock_kill:
            ok, reason = reboot_with_kexec("test_case", self.mock_sd)

            self.assertFalse(ok)
            self.assertIn("greenlit.service", reason)
            self.mock_sd.progress_record_time.assert_not_called()
            mock_unload.assert_called_once_with()
            mock_system.assert_not_called()
            mock_kill.assert_not_called()

            # The service is enabled, but systemctl rejects the kexec request.
            self.mock_sd.progress_record_time.reset_mock()
            mock_systemd.return_value = True
            mock_system.return_value = 1
            mock_unload.reset_mock()
            ok, reason = reboot_with_kexec("test_case", self.mock_sd)

            self.assertFalse(ok)
            self.assertEqual(reason, "systemctl kexec failed with status 1")
            self.mock_sd.progress_record_time.assert_called_once_with(
                "test_case", "TriggerKexec"
            )
            self.mock_sd.progress_delete_time.assert_called_once_with(
                "test_case", "TriggerKexec"
            )
            mock_unload.assert_called_once_with()
            mock_exit.assert_not_called()

    def test_reboot_with_kexec_reports_when_sigterm_does_not_terminate(self):
        """Report failure if the development-mode SIGTERM returns."""
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
            GLTTestBase.unittest.mock.mock_open(read_data="1"),
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._load_kernel_for_kexec",
            return_value=(True, ""),
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.is_systemd_running", return_value=False
        ), GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils._unload_kernel_for_kexec",
            return_value=(True, ""),
        ) as mock_unload, GLTTestBase.unittest.mock.patch(
            "src.common.glt_utils.os.system"
        ) as mock_system, GLTTestBase.unittest.mock.patch(
            "os.kill"
        ) as mock_kill, GLTTestBase.unittest.mock.patch(
            "os.getpid", return_value=12345
        ):
            ok, reason = reboot_with_kexec("test_case", self.mock_sd)

        self.assertFalse(ok)
        self.assertEqual(
            reason,
            "SIGTERM did not terminate the process while emulating shutdown",
        )
        self.mock_sd.progress_record_time.assert_not_called()
        mock_unload.assert_called_once_with()
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        mock_system.assert_not_called()

    @GLTTestBase.unittest.mock.patch("os._exit")
    @GLTTestBase.unittest.mock.patch("os.system")
    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.is_systemd_running")
    @GLTTestBase.unittest.mock.patch("src.common.glt_utils.systemd_enable")
    def test_reboot(
        self, mock_systemd_enable, mock_is_systemd, mock_system, mock_exit
    ):
        """Test reboot function"""
        mock_testcase = GLTTestBase.unittest.mock.MagicMock()

        # systemd is running and the service can be enabled
        mock_is_systemd.return_value = True
        mock_systemd_enable.return_value = True
        mock_system.return_value = 0
        result = reboot(mock_testcase, self.mock_sd)
        self.mock_sd.progress_record_time.assert_called_once_with(
            mock_testcase, "TriggerReboot"
        )
        mock_system.assert_called_once_with("reboot")
        mock_exit.assert_called_once_with(0)
        self.assertFalse(result)

        # The reboot command itself fails: remove the saved reboot marker.
        self.mock_sd.progress_record_time.reset_mock()
        mock_system.reset_mock()
        mock_system.return_value = 1
        mock_exit.reset_mock()
        result = reboot(mock_testcase, self.mock_sd)
        self.mock_sd.progress_record_time.assert_called_once_with(
            mock_testcase, "TriggerReboot"
        )
        self.mock_sd.progress_delete_time.assert_called_once_with(
            mock_testcase, "TriggerReboot"
        )
        mock_exit.assert_not_called()
        self.assertFalse(result)

        # No systemd at all: emulate system off
        self.mock_sd.progress_record_time.reset_mock()
        mock_is_systemd.return_value = False
        with GLTTestBase.unittest.mock.patch("os.kill") as mock_kill:
            result = reboot(mock_testcase, self.mock_sd)
            self.mock_sd.progress_record_time.assert_not_called()
            mock_kill.assert_called_once()
            self.assertFalse(result)

        # systemd runs but the unit cannot be enabled (masked, missing, ...):
        # report failure instead of silently killing the suite.
        mock_is_systemd.return_value = True
        mock_systemd_enable.return_value = False
        mock_system.reset_mock()
        with GLTTestBase.unittest.mock.patch("os.kill") as mock_kill:
            result = reboot(mock_testcase, self.mock_sd)
            self.mock_sd.progress_record_time.assert_not_called()
            mock_kill.assert_not_called()
            mock_system.assert_not_called()
            self.assertFalse(result)

    def test_run_command_returns_output_with_checked_text_defaults(self):
        """Run an argument vector with capture, text, check, and a bounded default."""
        completed = subprocess.CompletedProcess(
            ["lspci", "-D"], 0, stdout="0000:00:00.0\n", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=completed
        ) as mock_run:
            result = run_command(["lspci", "-D"], "Failed to list PCI devices")

        self.assertIs(result, completed)
        mock_run.assert_called_once_with(
            ["lspci", "-D"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )

    def test_run_command_forwards_a_caller_timeout(self):
        """Let a caller bound a command that is expected to run longer."""
        completed = subprocess.CompletedProcess(
            ["grub2-mkconfig"], 0, stdout="", stderr=""
        )

        with GLTTestBase.unittest.mock.patch(
            "subprocess.run", return_value=completed
        ) as mock_run:
            run_command(
                ["grub2-mkconfig"],
                "Failed to refresh the boot configuration",
                timeout=120.0,
            )

        mock_run.assert_called_once_with(
            ["grub2-mkconfig"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120.0,
        )

    def test_run_command_maps_command_failures_to_the_caller_message(self):
        """Report one domain message for timeout, non-zero exit, and decode failures."""
        failures = (
            subprocess.TimeoutExpired(["dmesg"], 30.0),
            _command_failure(["dmesg"], "dmesg error"),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        )

        for failure in failures:
            with self.subTest(
                failure=type(failure).__name__
            ), GLTTestBase.unittest.mock.patch(
                "subprocess.run", side_effect=failure
            ):
                with self.assertRaises(RuntimeError) as context:
                    run_command(["dmesg"], "Failed to retrieve the kernel log")

            self.assertEqual(
                str(context.exception), "Failed to retrieve the kernel log"
            )
            # Callers surface these messages in test results, so the raw
            # subprocess failure must stay suppressed rather than chained.
            self.assertTrue(context.exception.__suppress_context__)

    def test_run_command_names_an_executable_that_cannot_run(self):
        """Identify the executable when the command never starts."""
        for failure in (FileNotFoundError(), PermissionError()):
            with self.subTest(
                failure=type(failure).__name__
            ), GLTTestBase.unittest.mock.patch(
                "subprocess.run", side_effect=failure
            ):
                with self.assertRaises(RuntimeError) as context:
                    run_command(["lspci", "-D"], "Failed to list PCI devices")

            self.assertEqual(str(context.exception), "Failed to run lspci")
            self.assertTrue(context.exception.__suppress_context__)

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
