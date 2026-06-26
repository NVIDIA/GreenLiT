"""Unit tests for PMU perf compatibility."""
# pylint: disable=wrong-import-position,protected-access,unused-argument,too-many-public-methods

import subprocess
import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from common import perf_utils
from src.modules.arm64 import glt_test_pmu


class TestPerfPmuCompatibility(GLTTestBase):
    """Test PMU behavior around perf selftest compatibility."""

    def _perf_pass(self, test_name):
        return perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.PASS,
            "",
            test_name=test_name,
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_all_pmu_events_uses_old_and_617_candidates(
        self, mock_nvdev, mock_glt_utils, mock_perf_utils
    ):
        """PMU test tries the old perf name first, then the 6.17 name."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_glt_utils.is_cli_available.return_value = True
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = self._perf_pass(
            "Parsing of all PMU events from sysfs"
        )

        glt_test_pmu.test_all_pmu_events(None, self.mock_logger)

        mock_perf_utils.run_perf_test_candidates.assert_called_once_with(
            ["perf all PMU test", "Parsing of all PMU events from sysfs"]
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_all_pmu_events_fails_when_no_candidate_matches(
        self, mock_nvdev, mock_glt_utils, mock_perf_utils
    ):
        """PMU test fails clearly when neither old nor 6.17 names exist."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_glt_utils.is_cli_available.return_value = True
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.NO_MATCH,
            "No matching perf test found",
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_pmu.test_all_pmu_events(None, self.mock_logger)
        self.assertIn("No matching perf test found", str(context.exception))

    # --- _parse_cpu_list ---

    def test_parse_cpu_list_single(self):
        """Single CPU ID."""
        self.assertEqual(glt_test_pmu._parse_cpu_list("3"), [3])

    def test_parse_cpu_list_range(self):
        """Hyphen range expands correctly."""
        self.assertEqual(glt_test_pmu._parse_cpu_list("0-3"), [0, 1, 2, 3])

    def test_parse_cpu_list_mixed(self):
        """Comma list mixed with range."""
        self.assertEqual(glt_test_pmu._parse_cpu_list("0,2-3"), [0, 2, 3])

    # --- _parse_perf_stat_csv ---

    def test_parse_perf_stat_csv_valid_cpu_prefix(self):
        """Parses CPU<n> labels and returns correct busy/idle counts."""
        stderr = (
            "WARNING: some diagnostic line\n"
            "CPU2,2558580492,,cycles,1000004265,100.00,,\n"
            "CPU3,305749,,cycles,1000004265,100.00,,\n"
        )
        busy, idle = glt_test_pmu._parse_perf_stat_csv(stderr, 2, 3)
        self.assertEqual(busy, 2558580492)
        self.assertEqual(idle, 305749)

    def test_parse_perf_stat_csv_numeric_label(self):
        """Also accepts bare numeric CPU labels."""
        stderr = (
            "2,2558580492,,cycles,1000004265,100.00,,\n"
            "3,305749,,cycles,1000004265,100.00,,\n"
        )
        busy, idle = glt_test_pmu._parse_perf_stat_csv(stderr, 2, 3)
        self.assertEqual(busy, 2558580492)
        self.assertEqual(idle, 305749)

    def test_parse_perf_stat_csv_not_counted(self):
        """<not counted> raises ValueError with clear message."""
        stderr = "CPU0,<not counted>,,cycles,0,0.00,,\n"
        with self.assertRaises(ValueError) as ctx:
            glt_test_pmu._parse_perf_stat_csv(stderr, 0, 1)
        self.assertIn("unavailable", str(ctx.exception))

    def test_parse_perf_stat_csv_missing_cpu(self):
        """Missing CPU raises ValueError naming the absent CPU."""
        stderr = "CPU2,2558580492,,cycles,1000004265,100.00,,\n"
        with self.assertRaises(ValueError) as ctx:
            glt_test_pmu._parse_perf_stat_csv(stderr, 2, 3)
        self.assertIn("CPU 3", str(ctx.exception))

    def test_parse_perf_stat_csv_warning_lines_filtered(self):
        """WARNING: lines are ignored and do not corrupt the parse."""
        stderr = (
            "WARNING: A requested CPU is not supported by PMU 'cpu_atom'\n"
            "CPU0,500000000,,cycles,1000004265,100.00,,\n"
            "CPU1,40000,,cycles,1000004265,100.00,,\n"
        )
        busy, idle = glt_test_pmu._parse_perf_stat_csv(stderr, 0, 1)
        self.assertEqual(busy, 500000000)
        self.assertEqual(idle, 40000)

    def test_parse_perf_stat_csv_non_csv_diagnostic_filtered(self):
        """Non-CSV perf diagnostics are ignored and reported if counts are missing."""
        stderr = "Couldn't open the event\nCPU0,500000000,,cycles,1000004265,100.00,,\n"
        with self.assertRaises(ValueError) as ctx:
            glt_test_pmu._parse_perf_stat_csv(stderr, 0, 1)
        self.assertIn("Couldn't open the event", str(ctx.exception))

    def test_parse_perf_stat_csv_uses_count_field_not_unit_field(self):
        """The count field remains stable if perf ever emits a non-empty unit field."""
        stderr = (
            "CPU0,500000000,cycles-unit,cycles,1000004265,100.00,,\n"
            "CPU1,40000,cycles-unit,cycles,1000004265,100.00,,\n"
        )
        busy, idle = glt_test_pmu._parse_perf_stat_csv(stderr, 0, 1)
        self.assertEqual(busy, 500000000)
        self.assertEqual(idle, 40000)

    def test_parse_perf_stat_csv_requires_cycles_event_field(self):
        """Only the perf event field identifies the cycles rows."""
        stderr = (
            "CPU0,500000000,,instructions,cycles,100.00,,\n"
            "CPU1,40000,,cycles,1000004265,100.00,,\n"
        )
        with self.assertRaises(ValueError) as ctx:
            glt_test_pmu._parse_perf_stat_csv(stderr, 0, 1)
        self.assertIn("CPU 0", str(ctx.exception))

    def test_parse_perf_stat_csv_accepts_cycles_event_modifier(self):
        """Perf event modifiers keep cycles as the base event name."""
        stderr = (
            "CPU0,500000000,,cycles:u,1000004265,100.00,,\n"
            "CPU1,40000,,cycles:k,1000004265,100.00,,\n"
        )
        busy, idle = glt_test_pmu._parse_perf_stat_csv(stderr, 0, 1)
        self.assertEqual(busy, 500000000)
        self.assertEqual(idle, 40000)

    # --- _probe_pmccntr_smt ---

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.subprocess.run")
    def test_probe_pmccntr_smt_timeout_raises_value_error(self, mock_run):
        """Outer perf timeout is converted to a probe failure."""
        mock_run.side_effect = subprocess.TimeoutExpired(["perf"], 10)
        with self.assertRaises(ValueError) as ctx:
            glt_test_pmu._probe_pmccntr_smt(0, 1, self.mock_logger)
        self.assertIn("timed out", str(ctx.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.subprocess.run")
    def test_probe_pmccntr_smt_os_error_raises_value_error(self, mock_run):
        """Subprocess startup errors are converted to probe failures."""
        mock_run.side_effect = PermissionError("perf not executable")
        with self.assertRaises(ValueError) as ctx:
            glt_test_pmu._probe_pmccntr_smt(0, 1, self.mock_logger)
        self.assertIn("failed to start", str(ctx.exception))

    # --- test_pmccntr_smt_handling (decorated function) ---

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._probe_pmccntr_smt")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._find_online_smt_pair")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_pmccntr_smt_skip_non_vera(
        self, mock_nvdev, mock_glt_utils, mock_pair, mock_probe
    ):
        """Non-Vera CPU skips without touching the probe."""
        mock_nvdev.return_value.is_cpu_type.return_value = False
        with self.assertRaises(GLTCore.GLTTestSkip):
            glt_test_pmu.test_pmccntr_smt_handling(None, self.mock_logger)
        mock_pair.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._probe_pmccntr_smt")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._find_online_smt_pair")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_pmccntr_smt_skip_no_smt_pair(
        self, mock_nvdev, mock_glt_utils, mock_pair, mock_probe
    ):
        """Vera with no online SMT siblings skips."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_pair.return_value = None
        with self.assertRaises(GLTCore.GLTTestSkip):
            glt_test_pmu.test_pmccntr_smt_handling(None, self.mock_logger)

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._probe_pmccntr_smt")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._find_online_smt_pair")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_pmccntr_smt_fail_missing_tool(
        self, mock_nvdev, mock_glt_utils, mock_pair, mock_probe
    ):
        """Missing required tool causes GLTTestFail naming the tool."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_pair.return_value = (2, 3)
        mock_glt_utils.is_cli_available.side_effect = lambda t: t != "taskset"
        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            glt_test_pmu.test_pmccntr_smt_handling(None, self.mock_logger)
        self.assertIn("taskset", str(ctx.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._probe_pmccntr_smt")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._find_online_smt_pair")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_pmccntr_smt_pass_low_ratio(
        self, mock_nvdev, mock_glt_utils, mock_pair, mock_probe
    ):
        """Low idle/busy ratio passes (patch present, from commit example values)."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_pair.return_value = (2, 3)
        mock_glt_utils.is_cli_available.return_value = True
        mock_probe.return_value = (2_558_580_492, 305_749)
        glt_test_pmu.test_pmccntr_smt_handling(None, self.mock_logger)

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._probe_pmccntr_smt")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._find_online_smt_pair")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_pmccntr_smt_fail_high_ratio(
        self, mock_nvdev, mock_glt_utils, mock_pair, mock_probe
    ):
        """High idle/busy ratio fails and names the backport commits."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_pair.return_value = (2, 3)
        mock_glt_utils.is_cli_available.return_value = True
        # Unpatched: idle CPU reports cycles near the busy CPU
        mock_probe.return_value = (2_880_457_316, 2_880_459_810)
        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            glt_test_pmu.test_pmccntr_smt_handling(None, self.mock_logger)
        self.assertIn("c3d78c34ad00", str(ctx.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._probe_pmccntr_smt")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._find_online_smt_pair")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_pmccntr_smt_fail_low_busy_cycles(
        self, mock_nvdev, mock_glt_utils, mock_pair, mock_probe
    ):
        """Busy CPU below cycle floor fails with floor mentioned."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_pair.return_value = (2, 3)
        mock_glt_utils.is_cli_available.return_value = True
        mock_probe.return_value = (50_000, 1_000)
        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            glt_test_pmu.test_pmccntr_smt_handling(None, self.mock_logger)
        self.assertIn("floor", str(ctx.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._probe_pmccntr_smt")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu._find_online_smt_pair")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.glt_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_pmccntr_smt_fail_probe_parse_error(
        self, mock_nvdev, mock_glt_utils, mock_pair, mock_probe
    ):
        """ValueError from probe propagates as GLTTestFail."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_pair.return_value = (2, 3)
        mock_glt_utils.is_cli_available.return_value = True
        mock_probe.side_effect = ValueError("No cycle count found for CPU 3")
        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            glt_test_pmu.test_pmccntr_smt_handling(None, self.mock_logger)
        self.assertIn("CPU 3", str(ctx.exception))


if __name__ == "__main__":
    GLTTestBase.unittest.main()
