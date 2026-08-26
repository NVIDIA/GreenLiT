"""Unit tests for the CPPC cpc_read sysfs oracle."""
# pylint: disable=protected-access

import contextlib
import errno
import subprocess
from unittest import mock

from common import glt_core as GLTCore
from src.modules.cpu import glt_test_cppc_cpc_read as cppc_cpc_read
from tests.test_base import GLTTestBase

ORACLE_PASS = GLTCore.GLTTestResult.PASS
ORACLE_FAIL = GLTCore.GLTTestResult.FAIL
ORACLE_WARN = GLTCore.GLTTestResult.WARN
ORACLE_SKIP = GLTCore.GLTTestResult.SKIP


def _runtime_isdir(*, missing_prefix=None, missing_suffix=None):
    """Provide safe directory overrides for runtime prerequisite tests."""
    real_isdir = cppc_cpc_read.os.path.isdir

    def directory_exists(path):
        try:
            path_text = cppc_cpc_read.os.fsdecode(path)
        except TypeError:
            return real_isdir(path)
        if missing_prefix and path_text.startswith(missing_prefix):
            return False
        if missing_suffix and path_text.endswith(missing_suffix):
            return False
        if path_text.startswith("/lib/modules/") or path_text.endswith(
                cppc_cpc_read.CPPC_CPC_READ_PROBE_MODULE):
            return True
        return real_isdir(path)

    return directory_exists


def _enter_runtime_prerequisites(stack, stub_probe_load=True):
    """Install successful preflight mocks for the runtime test."""
    nvdev = stack.enter_context(mock.patch.object(cppc_cpc_read, "NvDev"))
    nvdev.return_value.is_cpu_type.return_value = True
    stack.enter_context(mock.patch.object(
        cppc_cpc_read,
        "_find_cppc_cpc_read_sysfs_oracles",
        return_value=(
            "cpu0",
            {
                "highest_perf": "/sys/cpu0/acpi_cppc/highest_perf",
                "feedback_ctrs": "/sys/cpu0/acpi_cppc/feedback_ctrs",
            },
            {"highest_perf": "373", "feedback_ctrs": "ref:1 del:1"},
            None,
        ),
    ))
    stack.enter_context(mock.patch.object(
        cppc_cpc_read.glt_utils, "is_cli_available", return_value=True
    ))
    stack.enter_context(mock.patch.object(
        cppc_cpc_read.check_kernel_config,
        "parse_kconfig",
        return_value={
            config: "y"
            for config in cppc_cpc_read.CPPC_CPC_READ_REQUIRED_CONFIGS
        },
    ))
    stack.enter_context(mock.patch.object(
        cppc_cpc_read,
        "_is_kernel_symbol_available",
        return_value=True,
    ))
    stack.enter_context(mock.patch.object(
        cppc_cpc_read,
        "_is_cppc_cpc_read_probe_present",
        return_value=False,
    ))
    stack.enter_context(mock.patch.object(
        cppc_cpc_read.os.path,
        "isdir",
        side_effect=_runtime_isdir(),
    ))
    stack.enter_context(mock.patch.object(
        cppc_cpc_read.glt_utils,
        "chdir_managed",
        side_effect=lambda _path: contextlib.nullcontext(),
    ))
    run_command = stack.enter_context(mock.patch.object(
        cppc_cpc_read.glt_utils, "run_command"
    ))
    stack.enter_context(mock.patch.object(
        cppc_cpc_read.glt_utils,
        "sign_kernel_module",
        return_value=True,
    ))
    if stub_probe_load:
        stack.enter_context(mock.patch.object(
            cppc_cpc_read,
            "_load_cppc_cpc_read_probe",
            return_value=(True, None, True),
        ))
    return run_command


class TestCppcCpcReadRuntime(GLTTestBase):
    """Test CPPC cpc_read oracle execution and verdicts."""

    def test_sysfs_oracle_accepts_eio_with_one_injection(self):
        """An injected EIO reaching userspace identifies a patched getter."""
        sysfs_path = "/sys/cpu0/acpi_cppc/highest_perf"

        def read_value(path):
            if path == sysfs_path:
                raise OSError(errno.EIO, "input/output error")
            return "1 3"

        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=read_value,
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ) as write_value:
            result = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf", sysfs_path, self.mock_logger
            )

        self.assertEqual(result, (ORACLE_PASS, None))
        self.assertEqual(
            write_value.call_args_list,
            [
                mock.call(cppc_cpc_read.CPPC_CPC_READ_PROBE_PROC_PATH, "arm"),
                mock.call(cppc_cpc_read.CPPC_CPC_READ_PROBE_PROC_PATH, "disarm"),
            ],
        )

    def test_sysfs_oracle_is_inconclusive_when_arm_fails(self):
        """An observer arm error is not evidence about the target patch."""
        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "write_sysfs_value",
            side_effect=OSError(errno.EACCES, "denied"),
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "read_sysfs_value"
        ) as read_value:
            verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf",
                "/sys/cpu0/acpi_cppc/highest_perf",
                self.mock_logger,
            )

        self.assertIs(verdict, ORACLE_SKIP)
        self.assertIn("Cannot arm", reason)
        read_value.assert_not_called()

    def test_sysfs_oracle_is_inconclusive_without_error_number(self):
        """An errno-less read failure cannot identify kernel behavior."""
        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=[OSError("unknown read error"), "1 0"],
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf",
                "/sys/cpu0/acpi_cppc/highest_perf",
                self.mock_logger,
            )

        self.assertIs(verdict, ORACLE_WARN)
        self.assertIn("without errno", reason)

    def test_sysfs_oracle_rejects_success_after_injection(self):
        """A successful injected read identifies an unpatched getter."""
        sysfs_path = "/sys/cpu0/acpi_cppc/highest_perf"

        def read_value(path):
            if path == sysfs_path:
                return "373"
            return "1 3"

        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=read_value,
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf", sysfs_path, self.mock_logger
            )

        self.assertIs(verdict, ORACLE_FAIL)
        self.assertIn("still succeeded", reason)
        self.assertIn(cppc_cpc_read.CPPC_CPC_READ_PATCH, reason)

    def test_sysfs_oracle_is_inconclusive_without_injection(self):
        """Do not claim a kernel result when the cpc_read path was not hit."""
        for status, expected in (
            ("0 0", "did not complete"),
            ("0 3", "ran out of instances system-wide 3 times"),
        ):
            with self.subTest(status=status), mock.patch.object(
                cppc_cpc_read.glt_utils,
                "read_sysfs_value",
                side_effect=["ref:1 del:1", status],
            ), mock.patch.object(
                cppc_cpc_read.glt_utils, "write_sysfs_value"
            ):
                verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                    "feedback_ctrs",
                    "/sys/cpu0/acpi_cppc/feedback_ctrs",
                    self.mock_logger,
                )
            self.assertIs(verdict, ORACLE_WARN)
            self.assertIn(expected, reason)

    def test_sysfs_oracle_does_not_claim_genuine_eio_without_injection(self):
        """A real read failure with zero injections remains inconclusive."""
        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=[OSError(errno.EIO, "firmware failure"), "0 0"],
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf",
                "/sys/cpu0/acpi_cppc/highest_perf",
                self.mock_logger,
            )

        self.assertIs(verdict, ORACLE_WARN)
        self.assertIn("did not complete", reason)

    def test_sysfs_oracle_is_inconclusive_with_multiple_injections(self):
        """A violated one-shot contract is an observer malfunction."""
        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=["373", "2 0"],
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf",
                "/sys/cpu0/acpi_cppc/highest_perf",
                self.mock_logger,
            )

        self.assertIs(verdict, ORACLE_WARN)
        self.assertIn("reported 2 injections", reason)

    def test_sysfs_oracle_is_inconclusive_with_unexpected_errno(self):
        """A different userspace errno is outside the patch oracle model."""
        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=[OSError(errno.EACCES, "denied"), "1 0"],
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf",
                "/sys/cpu0/acpi_cppc/highest_perf",
                self.mock_logger,
            )

        self.assertIs(verdict, ORACLE_WARN)
        self.assertIn(f"expected errno {errno.EIO}", reason)
        self.assertIn("inconclusive", reason)

    def test_sysfs_oracle_disarms_when_status_is_invalid(self):
        """Unreadable or malformed injection counts remain inconclusive."""
        status_errors = (
            PermissionError(errno.EACCES, "denied"),
            "not-a-number",
            "1 -1",
            "1 0 extra",
        )
        for status_error in status_errors:
            with self.subTest(status=status_error), mock.patch.object(
                cppc_cpc_read.glt_utils,
                "read_sysfs_value",
                side_effect=["373", status_error],
            ), mock.patch.object(
                cppc_cpc_read.glt_utils, "write_sysfs_value"
            ) as write_value:
                verdict, reason = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                    "highest_perf",
                    "/sys/cpu0/acpi_cppc/highest_perf",
                    self.mock_logger,
                )

            self.assertIs(verdict, ORACLE_WARN)
            self.assertIn("Cannot read the cpc_read observer status", reason)
            self.assertEqual(
                write_value.call_args_list[-1],
                mock.call(
                    cppc_cpc_read.CPPC_CPC_READ_PROBE_PROC_PATH, "disarm"
                ),
            )

    def test_sysfs_oracle_preserves_result_when_disarm_fails(self):
        """A failed disarm must not replace the synchronous oracle result."""
        with mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=[OSError(errno.EIO, "input/output error"), "1 0"],
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "write_sysfs_value",
            side_effect=[None, OSError(errno.EIO, "disarm failed")],
        ):
            result = cppc_cpc_read._run_cppc_cpc_read_sysfs_oracle(
                "highest_perf",
                "/sys/cpu0/acpi_cppc/highest_perf",
                self.mock_logger,
            )

        self.assertEqual(result, (ORACLE_PASS, None))
        self.mock_logger.warning.assert_called_once()

    def test_runtime_passes_both_sysfs_oracles_and_cleans(self):
        """Run both agreed sysfs discriminators and clean the module."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            oracle = stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_run_cppc_cpc_read_sysfs_oracle",
                side_effect=[(ORACLE_PASS, None), (ORACLE_PASS, None)],
            ))
            cleanup = stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_cleanup_cppc_cpc_read_probe",
                return_value=[],
            ))

            cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                None, self.mock_logger
            )

        self.assertEqual(oracle.call_count, 2)
        cleanup.assert_called_once_with(True, self.mock_logger, True)
        run_command.assert_called_once_with(
            ["make", "clean"],
            "Failed to clean CPPC cpc_read probe build artifacts",
            timeout=cppc_cpc_read.CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
        )

    def test_runtime_aggregates_explicit_oracle_verdicts(self):
        """FAIL wins, then PASS, WARN, and all-SKIP in that order."""
        cases = (
            (
                "failure outranks skip",
                [
                    (ORACLE_SKIP, "highest_perf", "caps inconclusive"),
                    (ORACLE_FAIL, "feedback_ctrs", "counters prove absence"),
                ],
                GLTCore.GLTTestFail,
                "counters prove absence",
            ),
            (
                "failure outranks pass",
                [
                    (ORACLE_PASS, "highest_perf", None),
                    (ORACLE_FAIL, "feedback_ctrs", "counters prove absence"),
                ],
                GLTCore.GLTTestFail,
                "counters prove absence",
            ),
            (
                "pass outranks skip",
                [
                    (ORACLE_PASS, "highest_perf", None),
                    (ORACLE_SKIP, "feedback_ctrs", "counters inconclusive"),
                ],
                None,
                None,
            ),
            (
                "pass outranks warn",
                [
                    (ORACLE_PASS, "highest_perf", None),
                    (ORACLE_WARN, "feedback_ctrs", "counters inconclusive"),
                ],
                None,
                None,
            ),
            (
                "warn outranks skip",
                [
                    (ORACLE_WARN, "highest_perf", "caps inconclusive"),
                    (ORACLE_SKIP, "feedback_ctrs", "counters unavailable"),
                ],
                GLTCore.GLTTestWarn,
                "caps inconclusive",
            ),
            (
                "all skip",
                [
                    (ORACLE_SKIP, "highest_perf", "caps inconclusive"),
                    (ORACLE_SKIP, "feedback_ctrs", "counters inconclusive"),
                ],
                GLTCore.GLTTestSkip,
                "caps inconclusive",
            ),
        )
        for name, results, expected_exception, expected_text in cases:
            with self.subTest(case=name), contextlib.ExitStack() as stack:
                _enter_runtime_prerequisites(stack)
                stack.enter_context(mock.patch.object(
                    cppc_cpc_read,
                    "_run_cppc_cpc_read_probe",
                    return_value=(results, []),
                ))

                if expected_exception is None:
                    cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                        None, self.mock_logger
                    )
                else:
                    with self.assertRaises(expected_exception) as context:
                        cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                            None, self.mock_logger
                        )
                    self.assertIn(expected_text, str(context.exception))

    def test_runtime_logs_non_winning_oracle_before_failure(self):
        """Preserve an inconclusive getter diagnostic when another getter fails."""
        results = [
            (ORACLE_FAIL, "highest_perf", "caps prove absence"),
            (ORACLE_WARN, "feedback_ctrs", "counter observer degraded"),
        ]
        with contextlib.ExitStack() as stack:
            _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_run_cppc_cpc_read_probe",
                return_value=(results, []),
            ))

            with self.assertRaises(GLTCore.GLTTestFail):
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.mock_logger.warning.assert_any_call(
            "Inconclusive CPPC sysfs oracle: %s",
            "counter observer degraded",
        )

    def test_runtime_refuses_preloaded_probe(self):
        """Skip before build when an unknown probe proc entry exists."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_is_cppc_cpc_read_probe_present",
                return_value=True,
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("already loaded", str(context.exception))
        self.assertIn("stale or unknown module", str(context.exception))
        self.assertIn("rmmod cppc_cpc_read_probe", str(context.exception))
        run_command.assert_not_called()

    def test_runtime_make_clean_failures_fail_safely(self):
        """Clean partial builds and preserve the preceding oracle context."""
        cases = (
            ("pass", [(ORACLE_PASS, None), (ORACLE_PASS, None)], None, []),
            (
                "oracle failure",
                [(ORACLE_FAIL, "caps prove absence"), (ORACLE_PASS, None)],
                None,
                ["caps prove absence"],
            ),
            (
                "build failure",
                [],
                (False, "probe build failed", False),
                ["probe build failed"],
            ),
            (
                "warn and skip",
                [(ORACLE_WARN, "caps degraded"),
                 (ORACLE_SKIP, "counters unavailable")],
                None,
                ["caps degraded", "counters unavailable"],
            ),
        )
        for name, oracle_results, load_result, prior_reasons in cases:
            with self.subTest(case=name), contextlib.ExitStack() as stack:
                self.mock_logger.reset_mock()
                run_command = _enter_runtime_prerequisites(stack)
                stack.enter_context(mock.patch.object(
                    cppc_cpc_read,
                    "_run_cppc_cpc_read_sysfs_oracle",
                    side_effect=oracle_results,
                ))
                stack.enter_context(mock.patch.object(
                    cppc_cpc_read,
                    "_cleanup_cppc_cpc_read_probe",
                    return_value=[],
                ))
                if load_result is not None:
                    stack.enter_context(mock.patch.object(
                        cppc_cpc_read,
                        "_load_cppc_cpc_read_probe",
                        return_value=load_result,
                    ))
                run_command.side_effect = RuntimeError("clean failed")

                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                        None, self.mock_logger
                    )

                reason = str(context.exception)
                if name == "pass":
                    prior_reasons = ["highest_perf", "feedback_ctrs"]
                positions = [reason.index(item) for item in prior_reasons]
                positions.append(reason.index("Cannot clean"))
                self.assertEqual(positions, sorted(positions))
                self.mock_logger.error.assert_called_once_with(
                    "Cannot clean the cpc_read observer build artifacts: "
                    "clean failed"
                )

    def test_runtime_probe_cleanup_error_overrides_pass(self):
        """Never report PASS while the cpc_read probe may remain active."""
        results = [
            (ORACLE_PASS, "highest_perf", None),
            (ORACLE_PASS, "feedback_ctrs", None),
        ]
        with contextlib.ExitStack() as stack:
            _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_run_cppc_cpc_read_probe",
                return_value=(results, ["probe remains loaded"]),
            ))

            with self.assertRaises(GLTCore.GLTTestFail) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("propagated an injected", str(context.exception))
        self.assertIn("probe remains loaded", str(context.exception))

    def test_runtime_skips_missing_sysfs_oracles(self):
        """Missing public CPPC sysfs interfaces make the test inapplicable."""
        with contextlib.ExitStack() as stack:
            _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_find_cppc_cpc_read_sysfs_oracles",
                return_value=(None, None, None, "CPPC sysfs is unavailable"),
            ))
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("CPPC sysfs is unavailable", str(context.exception))


class TestCppcCpcReadPreflight(GLTTestBase):
    """Test environment checks for the CPPC cpc_read observer."""

    def test_runtime_skips_non_vera_before_discovery(self):
        """The Vera-only oracle exits before touching observer state."""
        with mock.patch.object(
            cppc_cpc_read, "NvDev"
        ) as nvdev, mock.patch.object(
            cppc_cpc_read, "_find_cppc_cpc_read_sysfs_oracles"
        ) as find_oracles, mock.patch.object(
            cppc_cpc_read.glt_utils, "run_command"
        ) as run_command:
            nvdev.return_value.is_cpu_type.return_value = False

            with self.assertRaises(GLTCore.GLTTestSkip):
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        find_oracles.assert_not_called()
        run_command.assert_not_called()

    def test_runtime_skips_missing_command_before_build(self):
        """Missing build/load tools are inconclusive and do not start a build."""
        for missing_command in ("make", "insmod", "rmmod"):
            with self.subTest(command=missing_command), contextlib.ExitStack() as stack:
                run_command = _enter_runtime_prerequisites(stack)
                stack.enter_context(mock.patch.object(
                    cppc_cpc_read.glt_utils,
                    "is_cli_available",
                    side_effect=lambda command, missing=missing_command: command != missing,
                ))

                with self.assertRaises(GLTCore.GLTTestSkip):
                    cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                        None, self.mock_logger
                    )

                run_command.assert_not_called()

    def test_runtime_skips_module_build_context_error_without_cleanup(self):
        """A pre-build chdir failure is an environment SKIP, not cleanup FAIL."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.glt_utils,
                "chdir_managed",
                side_effect=PermissionError(errno.EACCES, "denied"),
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("module build context failed", str(context.exception))
        self.assertIn("denied", str(context.exception))
        run_command.assert_not_called()

    def test_runtime_skips_missing_kernel_headers_before_build(self):
        """Missing running-kernel headers are inconclusive."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.os.path,
                "isdir",
                side_effect=_runtime_isdir(missing_prefix="/lib/modules/"),
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("kernel headers", str(context.exception))
        run_command.assert_not_called()

    def test_runtime_fails_when_packaged_probe_is_missing(self):
        """Missing packaged probe sources identify a broken installation."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.os.path,
                "isdir",
                side_effect=_runtime_isdir(
                    missing_suffix=cppc_cpc_read.CPPC_CPC_READ_PROBE_MODULE
                ),
            ))

            with self.assertRaises(GLTCore.GLTTestFail) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("installation is incomplete", str(context.exception))
        run_command.assert_not_called()

    def test_runtime_skips_missing_kernel_config_before_build(self):
        """Missing kprobe configuration is inconclusive."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            find_oracles = stack.enter_context(mock.patch.object(
                cppc_cpc_read, "_find_cppc_cpc_read_sysfs_oracles"
            ))
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.platform,
                "uname",
                return_value=mock.Mock(release="test-kernel"),
            ))
            parse_kconfig = stack.enter_context(mock.patch.object(
                cppc_cpc_read.check_kernel_config,
                "parse_kconfig",
                return_value={
                    "CONFIG_KPROBES": "y",
                    "CONFIG_MODULE_UNLOAD": "y",
                },
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

            parse_kconfig.assert_called_once_with("/boot/config-test-kernel")
            self.assertIn("CONFIG_KRETPROBES", str(context.exception))
            find_oracles.assert_not_called()
            run_command.assert_not_called()

    def test_runtime_skips_unreadable_kernel_config_before_build(self):
        """An unreadable running-kernel config is inconclusive."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            find_oracles = stack.enter_context(mock.patch.object(
                cppc_cpc_read, "_find_cppc_cpc_read_sysfs_oracles"
            ))
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.check_kernel_config,
                "parse_kconfig",
                side_effect=OSError(errno.EACCES, "denied"),
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

            self.assertIn("Cannot inspect the running kernel", str(context.exception))
            self.assertIn("denied", str(context.exception))
            find_oracles.assert_not_called()
            run_command.assert_not_called()

    def test_runtime_skips_missing_kernel_symbol_before_build(self):
        """An unavailable cpc_read symbol is inconclusive."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_is_kernel_symbol_available",
                return_value=False,
            ))

            with self.assertRaises(GLTCore.GLTTestSkip):
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

            run_command.assert_not_called()

    def test_runtime_skips_unreadable_kallsyms_before_build(self):
        """A kallsyms read error is inconclusive and starts no build."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_is_kernel_symbol_available",
                side_effect=PermissionError(errno.EACCES, "denied"),
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("Cannot inspect /proc/kallsyms", str(context.exception))
        run_command.assert_not_called()

    def test_runtime_skips_signing_failure_without_loading(self):
        """A module-signing limitation is not a patch verdict."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(
                stack, stub_probe_load=False
            )
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.glt_utils,
                "sign_kernel_module",
                return_value=False,
            ))
            probe_present = stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_is_cppc_cpc_read_probe_present",
                return_value=False,
            ))
            insmod = stack.enter_context(mock.patch.object(
                cppc_cpc_read.subprocess, "run"
            ))

            with self.assertRaises(GLTCore.GLTTestSkip):
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        insmod.assert_not_called()
        probe_present.assert_called_once_with()
        self.assertFalse(any(
            call.args and call.args[0][0] == "rmmod"
            for call in run_command.call_args_list
        ))

    def test_runtime_skips_probe_build_failure(self):
        """A broken observer build environment is inconclusive."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(
                stack, stub_probe_load=False
            )
            run_command.side_effect = [
                RuntimeError("compiler mismatch"),
                mock.Mock(),
            ]
            probe_present = stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_is_cppc_cpc_read_probe_present",
                return_value=False,
            ))
            insmod = stack.enter_context(mock.patch.object(
                cppc_cpc_read.subprocess, "run"
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("Cannot build the cpc_read observer", str(context.exception))
        insmod.assert_not_called()
        probe_present.assert_called_once_with()
        self.assertFalse(any(
            call.args and call.args[0][0] == "rmmod"
            for call in run_command.call_args_list
        ))

    def test_runtime_skips_probe_already_loaded_failure(self):
        """An EEXIST load result preserves the kernel diagnostic."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(
                stack, stub_probe_load=False
            )
            insmod = stack.enter_context(mock.patch.object(
                cppc_cpc_read.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(
                    1, ["insmod"], output="", stderr="File exists"
                ),
            ))
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("Cannot load the cpc_read observer", str(context.exception))
        self.assertNotIn("Another GreenLiT run may own", str(context.exception))
        self.assertTrue(insmod.call_args.kwargs["check"])
        self.assertFalse(any(
            call.args and call.args[0][0] == "rmmod"
            for call in run_command.call_args_list
        ))

    def test_runtime_skips_other_probe_load_failure_without_ownership_claim(self):
        """A non-EEXIST load failure preserves its actual diagnostic."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(
                stack, stub_probe_load=False
            )
            insmod = stack.enter_context(mock.patch.object(
                cppc_cpc_read.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(
                    1, ["insmod"], output="", stderr="Invalid module format"
                ),
            ))
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("Invalid module format", str(context.exception))
        self.assertNotIn("Another GreenLiT run may own", str(context.exception))
        self.assertTrue(insmod.call_args.kwargs["check"])
        self.assertFalse(any(
            call.args and call.args[0][0] == "rmmod"
            for call in run_command.call_args_list
        ))

    def test_runtime_skips_insmod_startup_error_without_unloading(self):
        """An insmod execution error remains inconclusive and unowned."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(
                stack, stub_probe_load=False
            )
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.subprocess,
                "run",
                side_effect=OSError(errno.ENOENT, "insmod unavailable"),
            ))
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("Cannot execute insmod", str(context.exception))
        self.assertFalse(any(
            call.args and call.args[0][0] == "rmmod"
            for call in run_command.call_args_list
        ))

    def test_runtime_recovers_probe_after_ambiguous_load_timeout(self):
        """A timed-out load permits recovery only when probe paths appear."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(
                stack, stub_probe_load=False
            )
            stack.enter_context(mock.patch.object(
                cppc_cpc_read.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(["insmod"], 120),
            ))
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_is_cppc_cpc_read_probe_present",
                side_effect=[False, True, False],
            ))

            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("load outcome is uncertain", str(context.exception))
        self.assertTrue(any(
            call.args and call.args[0][0] == "rmmod"
            for call in run_command.call_args_list
        ))

    def test_runtime_warns_on_unexpected_oracle_exception(self):
        """Post-load observer I/O errors produce an advisory result."""
        with contextlib.ExitStack() as stack:
            run_command = _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_run_cppc_cpc_read_sysfs_oracle",
                side_effect=OSError(errno.EIO, "observer read failed"),
            ))
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_cleanup_cppc_cpc_read_probe",
                return_value=[],
            ))

            with self.assertRaises(GLTCore.GLTTestWarn) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("observer read failed", str(context.exception))
        run_command.assert_called_once_with(
            ["make", "clean"],
            "Failed to clean CPPC cpc_read probe build artifacts",
            timeout=cppc_cpc_read.CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
        )

    def test_runtime_keeps_first_failure_when_second_oracle_raises(self):
        """A later observer exception cannot erase proof the patch is absent."""
        with contextlib.ExitStack() as stack:
            _enter_runtime_prerequisites(stack)
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_run_cppc_cpc_read_sysfs_oracle",
                side_effect=[
                    (ORACLE_FAIL, "highest_perf proves the patch is absent"),
                    OSError(errno.EIO, "feedback observer failed"),
                ],
            ))
            stack.enter_context(mock.patch.object(
                cppc_cpc_read,
                "_cleanup_cppc_cpc_read_probe",
                return_value=[],
            ))

            with self.assertRaises(GLTCore.GLTTestFail) as context:
                cppc_cpc_read.test_vera_cppc_cpc_read_error_propagation(
                    None, self.mock_logger
                )

        self.assertIn("highest_perf proves", str(context.exception))
