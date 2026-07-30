"""Unit tests for ARM SPE PMU llc-miss workload gating."""
# pylint: disable=protected-access,unused-argument

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from modules.arm64 import glt_test_pmu


FULL_SPE_REPORT = """\
# Samples: 12K of event 'l1d-miss'
# Samples: 20K of event 'l1d-access'
# Samples: 1K of event 'llc-miss'
# Samples: 4K of event 'llc-access'
# Samples: 3K of event 'tlb-access'
# Samples: 8K of event 'memory'
# Samples: 2K of event 'branch'
"""

MISSING_LLC_MISS_REPORT = """\
# Samples: 12K of event 'l1d-miss'
# Samples: 20K of event 'l1d-access'
# Samples: 4K of event 'llc-access'
# Samples: 3K of event 'tlb-access'
# Samples: 8K of event 'memory'
# Samples: 2K of event 'branch'
"""

ZERO_LLC_MISS_REPORT = """\
# Samples: 12K of event 'l1d-miss'
# Samples: 20K of event 'l1d-access'
# Samples: 0 of event 'llc-miss'
# Samples: 4K of event 'llc-access'
# Samples: 3K of event 'tlb-access'
# Samples: 8K of event 'memory'
# Samples: 2K of event 'branch'
"""


class TestSpeReportRequirements(GLTTestBase):
    """Contract coverage for SPE report string gating."""

    def test_required_spe_report_strings_include_llc_miss(self):
        """llc-miss remains a hard required report string."""
        self.assertIn("llc-miss", glt_test_pmu.REQUIRED_SPE_REPORT_STRINGS)

    def test_positive_llc_miss_samples_accepts_realistic_report(self):
        """A real perf report sample header proves llc-miss sample evidence."""
        self.assertTrue(
            glt_test_pmu._has_positive_llc_miss_samples(FULL_SPE_REPORT))

    def test_positive_llc_miss_samples_rejects_metadata_only_token(self):
        """The llc-miss token alone is metadata, not decoded sample evidence."""
        metadata_only_output = """\
# Samples: 12K of event 'l1d-miss'
# event metadata: llc-miss
llc-miss
"""
        self.assertFalse(
            glt_test_pmu._has_positive_llc_miss_samples(metadata_only_output))

    def test_select_arm_spe_cpu_uses_first_affinity_allowed_cpumask_cpu(self):
        """The selected ARM SPE PMU runs on a usable cpumask CPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pmu_path = Path(tmpdir) / "arm_spe_2"
            pmu_path.mkdir()
            (pmu_path / "cpumask").write_text("4-5,2\n", encoding="utf-8")

            cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                "arm_spe_2", base_path=tmpdir, allowed_cpus={2, 4})

        self.assertEqual(cpu, 2)
        self.assertEqual(reason, "")

    def test_select_arm_spe_cpu_intersects_cpumask_with_affinity(self):
        """The selected ARM SPE CPU must be usable by the current process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pmu_path = Path(tmpdir) / "arm_spe_1"
            pmu_path.mkdir()
            (pmu_path / "cpumask").write_text("1-4\n", encoding="utf-8")

            cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                "arm_spe_1", base_path=tmpdir, allowed_cpus={3, 6})

        self.assertEqual(cpu, 3)
        self.assertEqual(reason, "")

    def test_select_arm_spe_cpu_reports_empty_affinity_intersection(self):
        """A stale SPE cpumask fails when no CPU is usable by this process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pmu_path = Path(tmpdir) / "arm_spe_3"
            pmu_path.mkdir()
            (pmu_path / "cpumask").write_text("4-7\n", encoding="utf-8")

            cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                "arm_spe_3", base_path=tmpdir, allowed_cpus={0, 1})

        self.assertIsNone(cpu)
        self.assertIn("no CPUs in the current process CPU affinity", reason)

    def test_select_arm_spe_pmu_cpu_falls_back_to_later_usable_pmu(self):
        """The LLC workload uses a later PMU when the first PMU is affinity-blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            first_pmu_path = Path(tmpdir) / "arm_spe_0"
            second_pmu_path = Path(tmpdir) / "arm_spe_1"
            first_pmu_path.mkdir()
            second_pmu_path.mkdir()
            (first_pmu_path / "cpumask").write_text("0-1\n", encoding="utf-8")
            (second_pmu_path / "cpumask").write_text("4-5\n", encoding="utf-8")

            pmu, cpu, reason = glt_test_pmu._select_arm_spe_pmu_cpu(
                base_path=tmpdir,
                allowed_cpus={4},
                logger=self.mock_logger,
            )

        self.assertEqual(pmu, "arm_spe_1")
        self.assertEqual(cpu, 4)
        self.assertEqual(reason, "")
        self.mock_logger.debug.assert_any_call(
            "Skipping ARM SPE PMU %s: %s",
            "arm_spe_0",
            self.mock.ANY,
        )

    def test_select_arm_spe_cpu_reports_affinity_failure(self):
        """Affinity read failures surface before the decorated test records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pmu_path = Path(tmpdir) / "arm_spe_4"
            pmu_path.mkdir()
            (pmu_path / "cpumask").write_text("0-1\n", encoding="utf-8")

            with self.mock.patch.object(
                glt_test_pmu.os,
                "sched_getaffinity",
                side_effect=OSError("affinity unavailable"),
            ):
                cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                    "arm_spe_4", base_path=tmpdir)

        self.assertIsNone(cpu)
        self.assertIn("Unable to read current CPU affinity", reason)
        self.assertIn("affinity unavailable", reason)

    def test_select_arm_spe_cpu_reports_cpumask_failures(self):
        """Bad PMU cpumasks produce failure reasons for the decorated test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pmu_path = Path(tmpdir) / "arm_spe_0"
            pmu_path.mkdir()

            with self.subTest(case="missing"):
                cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                    "arm_spe_0", base_path=tmpdir)
                self.assertIsNone(cpu)
                self.assertIn("Unable to read ARM SPE PMU cpumask", reason)

            with self.subTest(case="unreadable"):
                with self.mock.patch.object(
                    glt_test_pmu.Path,
                    "read_text",
                    side_effect=PermissionError("denied"),
                ):
                    cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                        "arm_spe_0", base_path=tmpdir)
                self.assertIsNone(cpu)
                self.assertIn("Unable to read ARM SPE PMU cpumask", reason)

            cpumask_path = pmu_path / "cpumask"
            with self.subTest(case="empty"):
                cpumask_path.write_text("\n", encoding="utf-8")
                cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                    "arm_spe_0", base_path=tmpdir)
                self.assertIsNone(cpu)
                self.assertIn("is empty", reason)

            with self.subTest(case="malformed"):
                cpumask_path.write_text("0,bad\n", encoding="utf-8")
                cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                    "arm_spe_0", base_path=tmpdir, allowed_cpus={0})
                self.assertIsNone(cpu)
                self.assertIn("Unable to parse ARM SPE PMU cpumask", reason)

            with self.subTest(case="overflow"):
                cpumask_path.write_text(
                    "0-999999999999999999999999999999999\n",
                    encoding="utf-8",
                )
                with self.mock.patch.object(
                    glt_test_pmu,
                    "_parse_cpu_list",
                    side_effect=OverflowError("range too large"),
                ):
                    cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                        "arm_spe_0", base_path=tmpdir, allowed_cpus={0})
                self.assertIsNone(cpu)
                self.assertIn("Unable to parse ARM SPE PMU cpumask", reason)
                self.assertIn("range too large", reason)

            with self.subTest(case="empty-range"):
                cpumask_path.write_text("3-1\n", encoding="utf-8")
                cpu, reason = glt_test_pmu._select_arm_spe_cpu(
                    "arm_spe_0", base_path=tmpdir, allowed_cpus={0})
                self.assertIsNone(cpu)
                self.assertIn("did not select any CPU", reason)


class TestArmSpePmu(GLTTestBase):
    """Decorated test_arm_spe_pmu behavior without real SPE hardware."""

    def setUp(self):
        super().setUp()
        glt_test_pmu.test_arm_spe_pmu.attr.result = (
            GLTCore.GLTTestResult.NA, "")
        self.mock_sd.log_dir = tempfile.mkdtemp(prefix="arm_spe_pmu_")
        self.addCleanup(shutil.rmtree, self.mock_sd.log_dir, ignore_errors=True)

    def _completed(self, args=None, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(
            args or ["perf"],
            returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _record_process(self, returncode=0, stderr="",
                        communicate_side_effect=None, wait_side_effect=None):
        process = self.mock.MagicMock()
        process.pid = 1234
        process.returncode = returncode
        if communicate_side_effect is None:
            process.communicate.return_value = ("", stderr)
        else:
            process.communicate.side_effect = communicate_side_effect
        if wait_side_effect is None:
            process.wait.return_value = returncode
        else:
            process.wait.side_effect = wait_side_effect
        return process

    def _patch_prereqs(self):
        return self.mock.patch.multiple(
            glt_test_pmu,
            check_kernel_config=self.mock.DEFAULT,
            _select_arm_spe_pmu_cpu=self.mock.DEFAULT,
            glt_utils=self.mock.DEFAULT,
            perf_utils=self.mock.DEFAULT,
        )

    def _configure_prereqs(self, mocks, perf_cmd="/usr/bin/perf", perf_reason=""):
        mocks["check_kernel_config"].parse_kconfig.return_value = {
            "CONFIG_ARM_SPE_PMU": "y",
        }
        mocks["_select_arm_spe_pmu_cpu"].return_value = ("arm_spe_0", 8, "")
        mocks["glt_utils"].is_cli_available.return_value = True
        mocks["perf_utils"].resolve_perf_command.return_value = (
            perf_cmd, perf_reason)

    def _record_call(self, mock_popen):
        self.assertTrue(mock_popen.called, "perf record was not invoked")
        return mock_popen.call_args

    def _record_cmd_args(self, mock_popen):
        record_call = self._record_call(mock_popen)
        return record_call.args[0] if record_call.args else record_call[0][0]

    def test_arm_spe_pmu_pass_when_all_strings_present(self):
        """Full SPE report with thrash workload passes."""
        record_process = self._record_process(
            stderr="Warning: AUX data detected collision\n")
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen", return_value=record_process):
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
                self._completed(
                    args=["perf", "report"],
                    stdout=FULL_SPE_REPORT,
                ),
            ]

            glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

        self.mock_logger.debug.assert_any_call(
            "Ignoring perf AUX collision warning: %s",
            "Warning: AUX data detected collision",
        )
        self.assertEqual(
            glt_test_pmu.test_arm_spe_pmu.attr.result[0],
            GLTCore.GLTTestResult.NA,
        )

    def test_arm_spe_pmu_fails_when_llc_miss_missing(self):
        """Missing llc-miss alone fails with that string named."""
        record_process = self._record_process()
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen", return_value=record_process):
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
                self._completed(
                    args=["perf", "report"],
                    stdout=MISSING_LLC_MISS_REPORT,
                ),
            ]

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "llc-miss"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

        self.assertEqual(
            glt_test_pmu.test_arm_spe_pmu.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_arm_spe_pmu_fails_when_llc_miss_samples_are_zero(self):
        """An llc-miss report header must still carry positive samples."""
        record_process = self._record_process()
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen", return_value=record_process):
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
                self._completed(
                    args=["perf", "report"],
                    stdout=ZERO_LLC_MISS_REPORT,
                ),
            ]

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "positive llc-miss"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

        self.assertEqual(
            glt_test_pmu.test_arm_spe_pmu.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_arm_spe_pmu_fails_on_record_nonzero(self):
        """Non-zero perf record return code fails the test."""
        record_process = self._record_process(returncode=1, stderr="record failed")
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen", return_value=record_process):
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
            ]

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "perf record failed"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

    def test_arm_spe_pmu_fails_on_record_timeout(self):
        """Record timeout cleans up the captured process group."""
        record_process = self._record_process(
            communicate_side_effect=subprocess.TimeoutExpired(
                ["perf", "record"], 300),
            wait_side_effect=[
                subprocess.TimeoutExpired(["perf", "record"], 5),
                0,
            ],
        )
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen", return_value=record_process), \
                self.mock.patch.object(glt_test_pmu.os, "killpg") as mock_killpg:
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
            ]

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "timed out"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

        mock_killpg.assert_has_calls([
            self.mock.call(1234, glt_test_pmu.signal.SIGTERM),
            self.mock.call(1234, glt_test_pmu.signal.SIGKILL),
        ])
        record_process.wait.assert_has_calls([
            self.mock.call(timeout=5),
            self.mock.call(timeout=2),
        ])

    def test_record_arm_spe_unwinds_exceptions_with_cleanup(self):
        """KeyboardInterrupt/SystemExit still clean up the perf process group."""
        for exception in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(exception=type(exception).__name__):
                record_process = self._record_process(
                    communicate_side_effect=exception)
                with self.mock.patch.object(
                    glt_test_pmu.subprocess,
                    "Popen",
                    return_value=record_process,
                ), self.mock.patch.object(
                    glt_test_pmu.os, "killpg"
                ) as mock_killpg:
                    with self.assertRaises(type(exception)):
                        glt_test_pmu._record_arm_spe_llc_workload(
                            "/perf",
                            "/tmp/perf.data",
                            "arm_spe_0//",
                            "/tmp/thrash.py",
                            8,
                        )

                mock_killpg.assert_called_once_with(
                    1234, glt_test_pmu.signal.SIGTERM)
                record_process.wait.assert_called_once_with(timeout=5)

    def test_arm_spe_pmu_fails_when_pmu_is_missing(self):
        """Missing ARM SPE sysfs PMU fails before perf record."""
        with self._patch_prereqs() as mocks:
            self._configure_prereqs(mocks)
            mocks["_select_arm_spe_pmu_cpu"].return_value = (
                None,
                None,
                "ARM SPE PMU is not available",
            )

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "ARM SPE PMU"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

    def test_arm_spe_pmu_fails_when_cpumask_is_invalid(self):
        """Invalid PMU cpumask reasons surface from the decorated test."""
        with self._patch_prereqs() as mocks:
            self._configure_prereqs(mocks)
            mocks["_select_arm_spe_pmu_cpu"].return_value = (
                None, None, "Unable to parse ARM SPE PMU cpumask")

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "cpumask"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

    def test_arm_spe_pmu_fails_when_taskset_missing(self):
        """Missing taskset fails before recording the pinned workload."""
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen") as mock_popen:
            self._configure_prereqs(mocks)
            mocks["glt_utils"].is_cli_available.return_value = False

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "taskset"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

        mock_popen.assert_not_called()

    def test_arm_spe_pmu_fails_on_report_called_process_error(self):
        """CalledProcessError from perf report fails clearly."""
        record_process = self._record_process()
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen", return_value=record_process):
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
                subprocess.CalledProcessError(
                    1, ["perf", "report"], output="bad report"),
            ]

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "perf report failed"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

    def test_arm_spe_pmu_fails_on_report_timeout(self):
        """TimeoutExpired from perf report fails clearly."""
        record_process = self._record_process()
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen", return_value=record_process):
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
                subprocess.TimeoutExpired(["perf", "report"], 120),
            ]

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "perf report timed out"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

    def test_arm_spe_pmu_uses_thrash_script_file_not_dd_or_inline_c(self):
        """Workload is taskset-pinned interpreter plus <script-path>."""
        record_process = self._record_process()
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run, \
                self.mock.patch.object(
                    glt_test_pmu.subprocess, "Popen",
                    return_value=record_process) as mock_popen:
            self._configure_prereqs(mocks)
            mock_run.side_effect = [
                self._completed(args=[sys.executable, "-c"]),
                self._completed(
                    args=["perf", "report"],
                    stdout=FULL_SPE_REPORT,
                ),
            ]

            glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

            record_cmd = self._record_cmd_args(mock_popen)

        self.assertEqual(record_cmd[0], "/usr/bin/perf")
        self.assertIn("arm_spe_0//", record_cmd)
        self.assertNotIn("arm_spe//", record_cmd)
        workload_cmd = record_cmd[record_cmd.index("--") + 1:]
        self.assertEqual(
            workload_cmd[:4],
            ["taskset", "--cpu-list", "8", sys.executable],
        )
        self.assertEqual(record_cmd[-2], sys.executable)
        self.assertNotIn("python3", record_cmd)
        self.assertNotIn("dd", record_cmd)
        self.assertNotIn("-c", record_cmd)

        thrash_path = Path(self.mock_sd.log_dir) / "arm_spe_llc_thrash.py"
        self.assertIn(str(thrash_path), record_cmd)
        self.assertTrue(thrash_path.is_file())
        thrash_text = thrash_path.read_text(encoding="utf-8")
        self.assertIn("memfd_create", thrash_text)
        self.assertIn("size = 536870912", thrash_text)
        self.assertIn("stride = 64", thrash_text)
        self.assertIn("passes = 2", thrash_text)

        record_kwargs = self._record_call(mock_popen).kwargs
        self.assertTrue(record_kwargs.get("start_new_session"))
        self.assertTrue(record_kwargs.get("text"))
        self.assertEqual(record_kwargs.get("stdout"), subprocess.PIPE)
        self.assertEqual(record_kwargs.get("stderr"), subprocess.PIPE)
        self.assertEqual(
            record_process.communicate.call_args.kwargs.get("timeout"),
            glt_test_pmu.ARM_SPE_LLC_RECORD_TIMEOUT_SECONDS,
        )

    def test_arm_spe_pmu_fails_when_perf_missing(self):
        """Missing resolved perf binary reports the resolver reason."""
        with self._patch_prereqs() as mocks:
            self._configure_prereqs(
                mocks, perf_cmd=None, perf_reason="no usable perf")

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "no usable perf"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

    def test_arm_spe_pmu_fails_when_memfd_probe_fails(self):
        """Failed memfd_create probe fails before the workload is recorded."""
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run:
            self._configure_prereqs(mocks)
            mock_run.return_value = self._completed(
                args=[sys.executable, "-c"],
                returncode=1,
                stderr="AttributeError: os has no attribute memfd_create",
            )

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "memfd_create"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

        self.assertEqual(mock_run.call_count, 1)

    def test_arm_spe_pmu_fails_when_memfd_probe_times_out(self):
        """Timed out memfd_create probe fails before perf record."""
        with self._patch_prereqs() as mocks, \
                self.mock.patch.object(glt_test_pmu.subprocess, "run") as mock_run:
            self._configure_prereqs(mocks)
            mock_run.side_effect = subprocess.TimeoutExpired(
                [sys.executable, "-c"], 10)

            with self.assertRaisesRegex(GLTCore.GLTTestFail, "timed out"):
                glt_test_pmu.test_arm_spe_pmu(self.mock_sd, self.mock_logger)

        self.assertEqual(mock_run.call_count, 1)
