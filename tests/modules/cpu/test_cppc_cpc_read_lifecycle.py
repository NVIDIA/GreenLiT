"""Unit tests for the CPPC cpc_read observer lifecycle and contracts."""
# pylint: disable=protected-access

import errno
import os
from unittest import mock

from src.modules.cpu import glt_test_cppc_cpc_read as cppc_cpc_read
from tests.test_base import GLTTestBase


class TestCppcCpcReadLifecycle(GLTTestBase):
    """Test the lifecycle and contracts of the CPPC cpc_read observer."""

    def test_find_sysfs_oracles_reads_same_cpu(self):
        """Select one CPU with readable paths for both affected getters."""
        cppc_dir = "/sys/devices/system/cpu/cpu7/acpi_cppc"

        def read_value(path):
            if path.endswith("highest_perf"):
                return "373"
            return "ref:10 del:9"

        with mock.patch.object(
            cppc_cpc_read.glob, "glob", return_value=[cppc_dir]
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=read_value,
        ):
            cpu_name, paths, baselines, reason = (
                cppc_cpc_read._find_cppc_cpc_read_sysfs_oracles(
                    path_isfile=lambda _path: True
                )
            )

        self.assertEqual(cpu_name, "cpu7")
        self.assertEqual(set(paths), {"highest_perf", "feedback_ctrs"})
        self.assertEqual(baselines["highest_perf"], "373")
        self.assertIsNone(reason)

    def test_find_sysfs_oracles_reports_unreadable_candidates(self):
        """Return a useful reason when no candidate has a valid baseline."""
        cppc_dir = "/sys/devices/system/cpu/cpu0/acpi_cppc"
        with mock.patch.object(
            cppc_cpc_read.glob, "glob", return_value=[cppc_dir]
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=PermissionError(errno.EACCES, "denied"),
        ):
            cpu_name, paths, baselines, reason = (
                cppc_cpc_read._find_cppc_cpc_read_sysfs_oracles(
                    path_isfile=lambda _path: True
                )
            )

        self.assertIsNone(cpu_name)
        self.assertIsNone(paths)
        self.assertIsNone(baselines)
        self.assertIn("No CPU has either readable", reason)

    def test_find_sysfs_oracles_reports_empty_baseline(self):
        """Distinguish empty readable attributes from absent sysfs files."""
        cppc_dir = "/sys/devices/system/cpu/cpu0/acpi_cppc"
        with mock.patch.object(
            cppc_cpc_read.glob, "glob", return_value=[cppc_dir]
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=["", ""],
        ):
            cpu_name, paths, baselines, reason = (
                cppc_cpc_read._find_cppc_cpc_read_sysfs_oracles(
                    path_isfile=lambda _path: True
                )
            )

        self.assertIsNone(cpu_name)
        self.assertIsNone(paths)
        self.assertIsNone(baselines)
        self.assertIn("highest_perf: empty value", reason)

    def test_find_sysfs_oracles_accepts_one_readable_getter(self):
        """Return one getter while preserving the other read diagnostic."""
        cppc_dir = "/sys/devices/system/cpu/cpu0/acpi_cppc"
        with mock.patch.object(
            cppc_cpc_read.glob, "glob", return_value=[cppc_dir]
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=["373", PermissionError(errno.EACCES, "denied")],
        ):
            cpu_name, paths, baselines, reason = (
                cppc_cpc_read._find_cppc_cpc_read_sysfs_oracles(
                    path_isfile=lambda _path: True
                )
            )

        self.assertEqual(cpu_name, "cpu0")
        self.assertEqual(paths, {
            "highest_perf": f"{cppc_dir}/highest_perf",
        })
        self.assertEqual(baselines, {"highest_perf": "373"})
        self.assertIn("feedback_ctrs", reason)
        self.assertIn("denied", reason)

    def test_find_sysfs_oracles_prefers_complete_later_cpu(self):
        """Keep paths and baselines on the CPU with the most usable getters."""
        cpu0_dir = "/sys/devices/system/cpu/cpu0/acpi_cppc"
        cpu1_dir = "/sys/devices/system/cpu/cpu1/acpi_cppc"

        def isfile(path):
            return not path.endswith("cpu0/acpi_cppc/feedback_ctrs")

        def read_value(path):
            if path.endswith("feedback_ctrs"):
                return "ref:10 del:9"
            if path.startswith(cpu0_dir):
                return "100"
            return "373"

        with mock.patch.object(
            cppc_cpc_read.glob, "glob", return_value=[cpu0_dir, cpu1_dir]
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "read_sysfs_value",
            side_effect=read_value,
        ):
            cpu_name, paths, baselines, reason = (
                cppc_cpc_read._find_cppc_cpc_read_sysfs_oracles(
                    path_isfile=isfile
                )
            )

        self.assertEqual(cpu_name, "cpu1")
        self.assertTrue(all(path.startswith(cpu1_dir) for path in paths.values()))
        self.assertEqual(
            baselines,
            {"highest_perf": "373", "feedback_ctrs": "ref:10 del:9"},
        )
        self.assertIsNone(reason)

    def test_probe_presence_checks_both_userspace_paths(self):
        """Detect procfs-only, sysfs-only, and absent probe state."""
        cases = (
            ("absent", [False, False], False),
            ("procfs", [True], True),
            ("sysfs", [False, True], True),
        )
        for name, path_results, expected in cases:
            with self.subTest(case=name):
                path_exists = mock.Mock(side_effect=path_results)
                self.assertEqual(
                    cppc_cpc_read._is_cppc_cpc_read_probe_present(path_exists),
                    expected,
                )

    def test_cleanup_does_not_unload_unowned_probe(self):
        """A failed load must not remove a concurrent run's probe."""
        with mock.patch.object(
            cppc_cpc_read,
            "_is_cppc_cpc_read_probe_present",
            return_value=True,
        ) as probe_present, mock.patch.object(
            cppc_cpc_read.glt_utils, "run_command"
        ) as run_command:
            errors = cppc_cpc_read._cleanup_cppc_cpc_read_probe(
                False, self.mock_logger, False
            )

        self.assertEqual(errors, [])
        probe_present.assert_not_called()
        run_command.assert_not_called()

    def test_cleanup_unloads_probe(self):
        """Unload the probe and verify its userspace paths disappear."""
        with mock.patch.object(
            cppc_cpc_read,
            "_is_cppc_cpc_read_probe_present",
            return_value=False,
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "run_command"
        ) as run_command, mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ) as write_value:
            errors = cppc_cpc_read._cleanup_cppc_cpc_read_probe(
                True, self.mock_logger, True
            )

        self.assertEqual(errors, [])
        write_value.assert_called_once_with(
            cppc_cpc_read.CPPC_CPC_READ_PROBE_PROC_PATH,
            cppc_cpc_read.CPPC_CPC_READ_PROBE_DISARM_COMMAND,
        )
        run_command.assert_called_once_with(
            ["rmmod", cppc_cpc_read.CPPC_CPC_READ_PROBE_MODULE],
            "Failed to unload cppc_cpc_read_probe",
            timeout=cppc_cpc_read.CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
        )

    def test_cleanup_reports_rmmod_error(self):
        """Log a stale rmmod error when observer paths are already gone."""
        with mock.patch.object(
            cppc_cpc_read,
            "_is_cppc_cpc_read_probe_present",
            return_value=False,
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "run_command",
            side_effect=RuntimeError("rmmod failed"),
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            errors = cppc_cpc_read._cleanup_cppc_cpc_read_probe(
                True, self.mock_logger, True
            )

        self.assertEqual(errors, [])
        self.mock_logger.warning.assert_called_once()

    def test_cleanup_still_unloads_when_best_effort_disarm_fails(self):
        """A cleanup disarm error must not prevent an unload attempt."""
        with mock.patch.object(
            cppc_cpc_read,
            "_is_cppc_cpc_read_probe_present",
            return_value=False,
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "run_command"
        ) as run_command, mock.patch.object(
            cppc_cpc_read.glt_utils,
            "write_sysfs_value",
            side_effect=OSError("disarm failed"),
        ):
            errors = cppc_cpc_read._cleanup_cppc_cpc_read_probe(
                True, self.mock_logger, True
            )

        self.assertEqual(errors, [])
        self.mock_logger.warning.assert_called_once()
        run_command.assert_called_once()

    def test_cleanup_reports_rmmod_error_when_probe_remains(self):
        """Keep an unload error verdict-bearing while probe paths remain."""
        with mock.patch.object(
            cppc_cpc_read,
            "_is_cppc_cpc_read_probe_present",
            return_value=True,
        ), mock.patch.object(
            cppc_cpc_read.glt_utils,
            "run_command",
            side_effect=RuntimeError("rmmod failed"),
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            errors = cppc_cpc_read._cleanup_cppc_cpc_read_probe(
                True, self.mock_logger, True
            )

        self.assertEqual(errors[0], "rmmod failed")
        self.assertIn("remains loaded", errors[1])

    def test_cleanup_reports_probe_that_remains_loaded(self):
        """Fail cleanup when observer state remains after rmmod."""
        with mock.patch.object(
            cppc_cpc_read,
            "_is_cppc_cpc_read_probe_present",
            return_value=True,
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "run_command"
        ) as run_command, mock.patch.object(
            cppc_cpc_read.glt_utils, "write_sysfs_value"
        ):
            errors = cppc_cpc_read._cleanup_cppc_cpc_read_probe(
                True, self.mock_logger, True
            )

        run_command.assert_called_once_with(
            ["rmmod", cppc_cpc_read.CPPC_CPC_READ_PROBE_MODULE],
            "Failed to unload cppc_cpc_read_probe",
            timeout=cppc_cpc_read.CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
        )
        self.assertIn("remains loaded", errors[0])

    def test_cleanup_detects_probe_without_loaded_flag(self):
        """Unload observer state when an uncertain load exposed probe paths."""
        with mock.patch.object(
            cppc_cpc_read,
            "_is_cppc_cpc_read_probe_present",
            side_effect=[True, False],
        ), mock.patch.object(
            cppc_cpc_read.glt_utils, "run_command"
        ) as run_command:
            errors = cppc_cpc_read._cleanup_cppc_cpc_read_probe(
                False, self.mock_logger, True
            )

        self.assertEqual(errors, [])
        run_command.assert_called_once_with(
            ["rmmod", cppc_cpc_read.CPPC_CPC_READ_PROBE_MODULE],
            "Failed to unload cppc_cpc_read_probe",
            timeout=cppc_cpc_read.CPPC_CPC_READ_COMMAND_TIMEOUT_SECS,
        )

    def test_kernel_symbol_check_requires_exact_name(self):
        """Do not accept a similarly prefixed kallsyms entry."""
        kallsyms = "0 t cpc_read_clone\n1 t cpc_read\n"
        opener = mock.mock_open(read_data=kallsyms)
        self.assertTrue(cppc_cpc_read._is_kernel_symbol_available(
            "cpc_read", "/proc/test-kallsyms", opener
        ))
        self.assertFalse(cppc_cpc_read._is_kernel_symbol_available(
            "cpc", "/proc/test-kallsyms", opener
        ))

    def test_probe_python_and_c_abi_match(self):
        """Keep Python procfs commands aligned with the packaged C probe."""
        module_dir = os.path.join(
            os.path.dirname(os.path.abspath(cppc_cpc_read.__file__)),
            cppc_cpc_read.CPPC_CPC_READ_PROBE_MODULE,
        )
        source_path = os.path.join(
            module_dir, f"{cppc_cpc_read.CPPC_CPC_READ_PROBE_MODULE}.c"
        )
        with open(source_path, encoding="utf-8") as source_file:
            source = source_file.read()

        proc_name = cppc_cpc_read.CPPC_CPC_READ_PROBE_PROC_PATH.rsplit(
            "/", maxsplit=1
        )[1]
        self.assertIn(
            f'#define GREENLIT_PROC_NAME "{proc_name}"', source
        )
        arm_start = source.index('if (sysfs_streq(command, "arm"))')
        disarm_start = source.index(
            'if (sysfs_streq(command, "disarm"))', arm_start
        )
        arm_body = source[arm_start:disarm_start]
        self.assertIn("injections = 0;", arm_body)
        self.assertIn(
            "missed_at_arm = READ_ONCE(cpc_read_kretprobe.nmissed);",
            arm_body,
        )
        call_start = source.index('seq_printf(seq, "%u %u\\n"')
        call = source[call_start:source.index(");", call_start)]
        self.assertLess(
            call.index("observed_injections"),
            call.index("observed_misses"),
        )
        for command in (
            cppc_cpc_read.CPPC_CPC_READ_PROBE_ARM_COMMAND,
            cppc_cpc_read.CPPC_CPC_READ_PROBE_DISARM_COMMAND,
        ):
            self.assertIn(f'sysfs_streq(command, "{command}")', source)
