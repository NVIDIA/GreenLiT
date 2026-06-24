"""Unit tests for PMU perf compatibility."""
# pylint: disable=wrong-import-position,protected-access,unused-argument,too-many-public-methods

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from common import perf_utils
from src.modules.arm64 import glt_test_pmu


class TestPerfPmuCompatibility(GLTTestBase):
    """Test PMU behavior around perf selftest compatibility."""

    def _completed_process(self, args=None, returncode=0, stdout="", stderr=""):
        """Build a subprocess result for helper tests."""
        return subprocess.CompletedProcess(
            args or ["cmd"],
            returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _perf_pass(self, test_name):
        return perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.PASS,
            "",
            test_name=test_name,
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_all_pmu_events_uses_old_and_617_candidates(
        self, mock_nvdev, mock_perf_utils
    ):
        """PMU test uses resolved perf for old and 6.17 selftest names."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_perf_utils.resolve_perf_command.return_value = "/perf"
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = self._perf_pass(
            "Parsing of all PMU events from sysfs"
        )

        glt_test_pmu.test_all_pmu_events(None, self.mock_logger)

        mock_perf_utils.resolve_perf_command.assert_called_once_with(self.mock_logger)
        mock_perf_utils.run_perf_test_candidates.assert_called_once_with(
            ["perf all PMU test", "Parsing of all PMU events from sysfs"],
            perf_cmd="/perf",
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_all_pmu_events_fails_when_no_candidate_matches(
        self, mock_nvdev, mock_perf_utils
    ):
        """PMU test fails clearly when neither old nor 6.17 names exist."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_perf_utils.resolve_perf_command.return_value = "/perf"
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.NO_MATCH,
            "No matching perf test found",
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_pmu.test_all_pmu_events(None, self.mock_logger)
        self.assertIn("No matching perf test found", str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_pmu.NvDev")
    def test_all_pmu_events_fails_when_no_perf_command_exists(
        self, mock_nvdev, mock_perf_utils
    ):
        """PMU test fails before selftest discovery when no usable perf exists."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_perf_utils.resolve_perf_command.return_value = None

        with self.assertRaises(GLTCore.GLTTestFail):
            glt_test_pmu.test_all_pmu_events(None, self.mock_logger)

        mock_perf_utils.run_perf_test_candidates.assert_not_called()

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

    def test_decode_arm_spe_uses_data_source_fields(self):
        """The decode helper asks perf script for data_src output."""
        completed = self._completed_process(stdout="decoded")
        with self.mock.patch.object(
            glt_test_pmu.subprocess, "run", return_value=completed
        ) as mock_run:
            result = glt_test_pmu._decode_arm_spe(
                "/usr/lib/linux-tools/6.17/perf",
                "/tmp/perf.data",
                self.mock_logger,
            )

        self.assertIs(result, completed)
        mock_run.assert_called_once_with(
            [
                "/usr/lib/linux-tools/6.17/perf",
                "script",
                "-i",
                "/tmp/perf.data",
                "-F",
                "period,data_src,weight,ip,sym",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=glt_test_pmu.ARM_SPE_DECODE_TIMEOUT_SECONDS,
        )

    def test_find_arm_spe_pmu_returns_first_existing_pmu(self):
        """ARM SPE PMU discovery prefers the first configured PMU name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "arm_spe_1"))

            self.assertEqual(
                glt_test_pmu._find_arm_spe_pmu(base_path=tmpdir),
                "arm_spe_1",
            )

    def test_find_arm_spe_pmu_ignores_non_directory_entries(self):
        """ARM SPE PMU discovery ignores sysfs files that share the prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "arm_spe_note"), "w", encoding="utf-8") as handle:
                handle.write("not a directory\n")
            os.mkdir(os.path.join(tmpdir, "arm_spe_2"))

            self.assertEqual(
                glt_test_pmu._find_arm_spe_pmu(base_path=tmpdir),
                "arm_spe_2",
            )

    def test_has_decoded_data_source_accepts_load_with_l2_level(self):
        """Decoded Olympus data source accepts common data-source L2 evidence."""
        output = "100000 |OP LOAD|LVL L2 hit| 42 0xffff memcpy"

        self.assertTrue(glt_test_pmu._has_decoded_data_source(output))

    def test_has_decoded_data_source_accepts_store_with_local_ram_level(self):
        """Decoded Olympus data source accepts real perf Local RAM evidence."""
        output = "100000 |OP STORE|LVL Local RAM hit| 42 0xffff memcpy"

        self.assertTrue(glt_test_pmu._has_decoded_data_source(output))

    def test_has_decoded_data_source_accepts_store_with_remote_ram_level(self):
        """Decoded Olympus data source accepts real perf Remote RAM evidence."""
        output = "100000 |OP STORE|LVL Remote RAM (1 hop) hit| 42 0xffff memcpy"

        self.assertTrue(glt_test_pmu._has_decoded_data_source(output))

    def test_has_decoded_data_source_rejects_generic_spe_output(self):
        """Generic SPE report strings do not prove Olympus data-source decode."""
        output = "Samples: 10K of event 'l1d-miss', Event count: 42"

        self.assertFalse(glt_test_pmu._has_decoded_data_source(output))

    def test_has_decoded_data_source_rejects_l1_level(self):
        """Generic ARM SPE fallback can emit L1, so it is not Olympus evidence."""
        output = "100000 |OP LOAD|LVL L1 hit| 42 0xffff memcpy"

        self.assertFalse(glt_test_pmu._has_decoded_data_source(output))

    def test_has_decoded_data_source_rejects_l3_level(self):
        """Generic ARM SPE fallback can emit L3, so it is not Olympus evidence."""
        output = "100000 |OP LOAD|LVL L3 hit| 42 0xffff memcpy"

        self.assertFalse(glt_test_pmu._has_decoded_data_source(output))

    def test_has_decoded_data_source_rejects_split_op_and_level_lines(self):
        """Operation and memory level must appear in the same decoded sample."""
        output = "100000 |OP LOAD| memcpy\n100000 |LVL L2 hit| memcpy"

        self.assertFalse(glt_test_pmu._has_decoded_data_source(output))

    def test_record_arm_spe_uses_explicit_pmu_filters(self):
        """The record helper uses the verified ARM SPE event filter shape."""
        completed = self._completed_process()
        with self.mock.patch.object(
            glt_test_pmu.subprocess, "run", return_value=completed
        ) as mock_run:
            result = glt_test_pmu._record_arm_spe(
                "/usr/lib/linux-tools/6.17/perf",
                "/tmp/perf.data",
                "arm_spe_0",
                self.mock_logger,
            )

        self.assertIs(result, completed)
        mock_run.assert_called_once_with(
            [
                "/usr/lib/linux-tools/6.17/perf",
                "record",
                "-o",
                "/tmp/perf.data",
                "-e",
                "arm_spe_0/ts_enable=1,load_filter=1,store_filter=1,min_latency=1/",
                "--",
                "dd",
                "if=/dev/zero",
                "of=/dev/null",
                "bs=1M",
                "count=512",
                "status=none",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=glt_test_pmu.ARM_SPE_RECORD_TIMEOUT_SECONDS,
        )

    def test_olympus_data_source_test_skips_on_grace(self):
        """The Olympus data-source test is Vera-only but logs platform state."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.side_effect = lambda name: name == "grace"

        with self.mock.patch.object(
            glt_test_pmu, "NvDev", return_value=fake_nvdev
        ), self.assertRaises(GLTCore.GLTTestSkip):
            glt_test_pmu.test_perf_arm_spe_olympus_data_source(
                self.mock_sd,
                self.mock_logger,
            )

        self.mock_logger.info.assert_not_called()

    def test_olympus_data_source_test_passes_when_decode_has_data_source(self):
        """The decorated test passes when perf script shows op and level fields."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.side_effect = lambda name: name == "vera"
        self.mock_sd.log_dir = "/tmp/greenlit"

        with self.mock.patch.object(
            glt_test_pmu, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_pmu.perf_utils, "resolve_perf_command", return_value="/perf"
        ), self.mock.patch.object(
            glt_test_pmu, "_find_arm_spe_pmu", return_value="arm_spe_0"
        ), self.mock.patch.object(
            glt_test_pmu,
            "_record_arm_spe",
            return_value=self._completed_process(stdout="recorded"),
        ) as mock_record, self.mock.patch.object(
            glt_test_pmu,
            "_decode_arm_spe",
            return_value=self._completed_process(
                stdout="1 |OP LOAD|LVL L2 hit| memcpy"
            ),
        ):
            glt_test_pmu.test_perf_arm_spe_olympus_data_source(
                self.mock_sd,
                self.mock_logger,
            )

        mock_record.assert_called_once_with(
            "/perf",
            "/tmp/greenlit/test_perf_arm_spe_olympus_data_source.perf.data",
            "arm_spe_0",
            self.mock_logger,
        )
        self.mock_logger.info.assert_any_call(
            "Decoded ARM SPE data-source summary: decoded_lines=%d",
            1,
        )

    def test_olympus_data_source_test_fails_when_decode_lacks_data_source(self):
        """A successful perf script command still fails without decoded fields."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.side_effect = lambda name: name == "vera"
        self.mock_sd.log_dir = "/tmp/greenlit"

        with self.mock.patch.object(
            glt_test_pmu, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_pmu.perf_utils, "resolve_perf_command", return_value="/perf"
        ), self.mock.patch.object(
            glt_test_pmu, "_find_arm_spe_pmu", return_value="arm_spe_0"
        ), self.mock.patch.object(
            glt_test_pmu,
            "_record_arm_spe",
            return_value=self._completed_process(stdout="recorded"),
        ), self.mock.patch.object(
            glt_test_pmu,
            "_decode_arm_spe",
            return_value=self._completed_process(stdout="Samples: l1d-miss"),
        ), self.assertRaises(GLTCore.GLTTestFail):
            glt_test_pmu.test_perf_arm_spe_olympus_data_source(
                self.mock_sd,
                self.mock_logger,
            )

    def test_olympus_data_source_test_fails_when_decode_command_fails(self):
        """A non-zero perf script command fails the GreenLiT test."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.side_effect = lambda name: name == "vera"
        self.mock_sd.log_dir = "/tmp/greenlit"

        with self.mock.patch.object(
            glt_test_pmu, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_pmu.perf_utils, "resolve_perf_command", return_value="/perf"
        ), self.mock.patch.object(
            glt_test_pmu, "_find_arm_spe_pmu", return_value="arm_spe_0"
        ), self.mock.patch.object(
            glt_test_pmu,
            "_record_arm_spe",
            return_value=self._completed_process(stdout="recorded"),
        ), self.mock.patch.object(
            glt_test_pmu,
            "_decode_arm_spe",
            return_value=self._completed_process(returncode=1, stderr="decode failed"),
        ), self.assertRaises(GLTCore.GLTTestFail):
            glt_test_pmu.test_perf_arm_spe_olympus_data_source(
                self.mock_sd,
                self.mock_logger,
            )

    def test_olympus_data_source_test_fails_when_pmu_is_missing(self):
        """A Vera system without an ARM SPE PMU fails before perf record."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.side_effect = lambda name: name == "vera"

        with self.mock.patch.object(
            glt_test_pmu, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_pmu.perf_utils, "resolve_perf_command", return_value="/perf"
        ), self.mock.patch.object(
            glt_test_pmu, "_find_arm_spe_pmu", return_value=None
        ), self.mock.patch.object(
            glt_test_pmu, "_record_arm_spe"
        ) as mock_record, self.assertRaises(GLTCore.GLTTestFail):
            glt_test_pmu.test_perf_arm_spe_olympus_data_source(
                self.mock_sd,
                self.mock_logger,
            )

        mock_record.assert_not_called()

    def test_olympus_data_source_test_fails_when_record_command_fails(self):
        """A non-zero perf record command fails before decode."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.side_effect = lambda name: name == "vera"
        self.mock_sd.log_dir = "/tmp/greenlit"

        with self.mock.patch.object(
            glt_test_pmu, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_pmu.perf_utils, "resolve_perf_command", return_value="/perf"
        ), self.mock.patch.object(
            glt_test_pmu, "_find_arm_spe_pmu", return_value="arm_spe_0"
        ), self.mock.patch.object(
            glt_test_pmu,
            "_record_arm_spe",
            return_value=self._completed_process(returncode=1, stderr="record failed"),
        ), self.mock.patch.object(
            glt_test_pmu, "_decode_arm_spe"
        ) as mock_decode, self.assertRaises(GLTCore.GLTTestFail):
            glt_test_pmu.test_perf_arm_spe_olympus_data_source(
                self.mock_sd,
                self.mock_logger,
            )

        mock_decode.assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
