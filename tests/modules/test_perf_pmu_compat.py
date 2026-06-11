"""Unit tests for PMU perf compatibility."""
# pylint: disable=wrong-import-position

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


if __name__ == "__main__":
    GLTTestBase.unittest.main()
