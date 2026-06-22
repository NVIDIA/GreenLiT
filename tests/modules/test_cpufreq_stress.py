"""Unit tests for CPU frequency stress test helpers."""
# pylint: disable=protected-access

from unittest import mock

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.cpu import glt_test_cpufreq_stress as cpufreq_stress


class TestCpuFreqStress(GLTTestBase):
    """Test CPU frequency helper behavior."""

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.glt_utils.check_kernel_log")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._validate_cppc_prerequisites")
    def test_cppc_cpufreq_online_get_fails_on_kernel_log_match(
        self, mock_validate, mock_check_log
    ):
        """Fail when the current boot log contains the cpufreq online error."""
        mock_validate.return_value = (True, None)
        mock_check_log.return_value = True

        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            cpufreq_stress.test_cppc_cpufreq_online_get_did_not_fail(
                None, self.mock_logger
            )

        self.assertIn(cpufreq_stress.CPUFREQ_ONLINE_GET_FAILED_MSG, str(ctx.exception))

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._get_cpus_with_invalid_scaling_cur_freq")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.glt_utils.check_kernel_log")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._validate_cppc_prerequisites")
    def test_cppc_cpufreq_online_get_fails_on_invalid_scaling_cur_freq(
        self, mock_validate, mock_check_log, mock_invalid_cpus
    ):
        """Fail when a cpufreq CPU does not report positive scaling_cur_freq."""
        mock_validate.return_value = (True, None)
        mock_check_log.return_value = False
        mock_invalid_cpus.return_value = [1]

        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            cpufreq_stress.test_cppc_cpufreq_online_get_did_not_fail(
                None, self.mock_logger
            )

        self.assertIn("CPUs expose cppc_cpufreq", str(ctx.exception))

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.glt_utils.is_cli_available")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._get_cpus_with_invalid_scaling_cur_freq")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.glt_utils.check_kernel_log")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._validate_cppc_prerequisites")
    def test_cppc_cpufreq_online_get_does_not_require_stress_ng(
        self, mock_validate, mock_check_log, mock_invalid_cpus, mock_is_cli_available
    ):
        """Pass the non-stress check without consulting stress-ng availability."""
        mock_validate.return_value = (True, None)
        mock_check_log.return_value = False
        mock_invalid_cpus.return_value = []

        cpufreq_stress.test_cppc_cpufreq_online_get_did_not_fail(None, self.mock_logger)

        mock_validate.assert_called_once_with()
        mock_is_cli_available.assert_not_called()

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._get_cpu_scaling_cur_freqs")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._is_cpu_online")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._cpu_cpufreq_glob")
    def test_get_cpus_with_invalid_scaling_cur_freq_reports_missing_cpu(
        self, mock_cpufreq_glob, mock_is_cpu_online, mock_get_freqs
    ):
        """Report CPUs with cpufreq driver but no positive scaling_cur_freq."""
        mock_cpufreq_glob.return_value = [
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
            "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
        ]
        mock_is_cpu_online.return_value = True
        mock_get_freqs.return_value = {0: 2400000}

        result = cpufreq_stress._get_cpus_with_invalid_scaling_cur_freq(self.mock_logger)

        self.assertEqual(result, [1])

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._get_cpu_scaling_cur_freqs")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._is_cpu_online")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._cpu_cpufreq_glob")
    def test_get_cpus_with_invalid_scaling_cur_freq_ignores_offline_cpu(
        self, mock_cpufreq_glob, mock_is_cpu_online, mock_get_freqs
    ):
        """Do not report offline CPUs that lack scaling_cur_freq."""
        mock_cpufreq_glob.return_value = [
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
            "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
        ]
        mock_is_cpu_online.side_effect = lambda cpu, logger=None: cpu == 0
        mock_get_freqs.return_value = {0: 2400000}

        result = cpufreq_stress._get_cpus_with_invalid_scaling_cur_freq(self.mock_logger)

        self.assertEqual(result, [])

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._get_cpu_scaling_cur_freqs")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._is_cpu_online")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._cpu_cpufreq_glob")
    def test_get_cpus_with_invalid_scaling_cur_freq_accepts_all_positive_cpus(
        self, mock_cpufreq_glob, mock_is_cpu_online, mock_get_freqs
    ):
        """Do not report CPUs when every cpufreq CPU has positive scaling_cur_freq."""
        mock_cpufreq_glob.return_value = [
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
            "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
        ]
        mock_is_cpu_online.return_value = True
        mock_get_freqs.return_value = {0: 2400000, 1: 2400000}

        result = cpufreq_stress._get_cpus_with_invalid_scaling_cur_freq(self.mock_logger)

        self.assertEqual(result, [])

    @mock.patch("builtins.open")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._cpu_cpufreq_glob")
    def test_validate_cppc_prerequisites_rejects_mixed_cpufreq_drivers(
        self, mock_cpufreq_glob, mock_open
    ):
        """Fail when any cpufreq policy is not using cppc_cpufreq."""
        paths = [
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
            "/sys/devices/system/cpu/cpu1/cpufreq/scaling_driver",
        ]
        values = {
            paths[0]: "cppc_cpufreq\n",
            paths[1]: "other_driver\n",
        }
        mock_cpufreq_glob.return_value = paths

        def open_side_effect(path, *args, **kwargs):  # pylint: disable=unused-argument
            return mock.mock_open(read_data=values[path]).return_value

        mock_open.side_effect = open_side_effect

        ok, reason = cpufreq_stress._validate_cppc_prerequisites()

        self.assertFalse(ok)
        self.assertIn(paths[1], reason)
        self.assertIn("other_driver", reason)
