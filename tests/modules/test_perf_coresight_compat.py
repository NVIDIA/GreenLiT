"""Unit tests for CoreSight perf compatibility."""
# pylint: disable=wrong-import-position,duplicate-code

import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from common import perf_utils
from src.modules.hwtracing import glt_test_coresight


class TestPerfCoresightCompatibility(GLTTestBase):
    """Test CoreSight behavior around perf selftest compatibility."""

    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_perf_coresight_uses_fallback_when_selftest_is_missing(
        self, mock_nvdev, mock_perf_utils
    ):
        """Coresight test accepts fallback evidence when no selftest is listed."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_nvdev.get_release_info.return_value = {
            "NAME": "Ubuntu",
            "VERSION_ID": "24.04",
        }
        mock_perf_utils.resolve_perf_command.return_value = ("/perf", "")
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.NO_MATCH,
            "No matching perf test found",
        )
        mock_perf_utils.has_coresight_fallback_evidence.return_value = True

        glt_test_coresight.test_perf_coresight(None, self.mock_logger)

        mock_perf_utils.run_perf_test_candidates.assert_called_once_with(
            ["arm coresight"], perf_cmd="/perf"
        )
        mock_perf_utils.has_coresight_fallback_evidence.assert_called_once_with()
        self.mock_logger.info.assert_called_once_with(
            "No perf Coresight selftest found; using cs_etm/sysfs fallback."
        )

    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_perf_coresight_fails_when_selftest_and_fallback_are_missing(
        self, mock_nvdev, mock_perf_utils
    ):
        """Coresight test fails when neither perf selftest nor fallback exists."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_nvdev.get_release_info.return_value = {
            "NAME": "Ubuntu",
            "VERSION_ID": "24.04",
        }
        mock_perf_utils.resolve_perf_command.return_value = ("/perf", "")
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.NO_MATCH,
            "No matching perf test found",
        )
        mock_perf_utils.has_coresight_fallback_evidence.return_value = False

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_coresight(None, self.mock_logger)
        self.assertIn("No matching perf test found", str(context.exception))


if __name__ == "__main__":
    GLTTestBase.unittest.main()
