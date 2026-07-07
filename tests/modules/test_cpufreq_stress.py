"""Unit tests for CPU frequency stress test helpers."""
# pylint: disable=protected-access,too-many-arguments,too-many-public-methods

import contextlib
import types
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

    def test_parse_cpu_mask_handles_ranges_and_lists(self):
        """CPU masks are the data structure behind hotplug and policy checks."""
        self.assertEqual(
            cpufreq_stress._parse_cpu_mask("0-2,8,10-11"),
            {0, 1, 2, 8, 10, 11},
        )
        self.assertEqual(cpufreq_stress._parse_cpu_mask(""), set())

    def test_select_hotplug_cpu_skips_cpu0_and_requires_writable_online_file(self):
        """The hotplug probe must avoid CPU0 and choose a CPU it can restore."""
        writable = {
            "/sys/devices/system/cpu/cpu1/online": False,
            "/sys/devices/system/cpu/cpu2/online": True,
        }
        values = {
            "/sys/devices/system/cpu/cpufreq/policy2/affected_cpus": "2",
            "/sys/devices/system/cpu/cpufreq/policy2/related_cpus": "2",
        }

        cpu, reason = cpufreq_stress._select_hotplug_cpu(
            {0, 1, 2},
            is_writable=lambda path, _mode: writable.get(path, False),
            path_exists=lambda path: path == "/sys/devices/system/cpu/cpufreq/policy2",
            read_file=values.__getitem__,
        )

        self.assertEqual(cpu, 2)
        self.assertEqual(reason, "")

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.glt_utils.read_sysfs_value")
    def test_validate_cppc_policy_for_cpu_accepts_recreated_policy(self, mock_read):
        """A recreated policy must expose target CPU masks and positive frequencies."""
        values = {
            "/sys/devices/system/cpu/cpufreq/policy1/affected_cpus": "1",
            "/sys/devices/system/cpu/cpufreq/policy1/related_cpus": "1",
            "/sys/devices/system/cpu/cpufreq/policy1/scaling_cur_freq": "3240000",
            "/sys/devices/system/cpu/cpufreq/policy1/cpuinfo_max_freq": "3348000",
        }
        mock_read.side_effect = values.__getitem__

        ok, reason = cpufreq_stress._validate_cppc_policy_for_cpu(
            1,
            path_exists=lambda path: path == "/sys/devices/system/cpu/cpufreq/policy1",
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_verify_cppc_policy_absent_for_cpu_accepts_missing_policy(self):
        """Offline teardown is proven when the selected CPU policy directory is gone."""
        ok, reason = cpufreq_stress._verify_cppc_policy_absent_for_cpu(
            1,
            path_exists=lambda _path: False,
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_verify_cppc_policy_absent_for_cpu_accepts_unreadable_policy(self):
        """Offline teardown is also proven when the policy path exists but is unreadable."""
        def read_file(_path):
            raise OSError("permission denied")

        ok, reason = cpufreq_stress._verify_cppc_policy_absent_for_cpu(
            1,
            path_exists=lambda path: path == "/sys/devices/system/cpu/cpufreq/policy1",
            read_file=read_file,
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_verify_cppc_policy_absent_for_cpu_rejects_readable_policy(self):
        """A still-readable policy while offline means teardown was not observed."""
        values = {
            "/sys/devices/system/cpu/cpufreq/policy1/scaling_cur_freq": "3240000",
        }

        ok, reason = cpufreq_stress._verify_cppc_policy_absent_for_cpu(
            1,
            path_exists=lambda path: path == "/sys/devices/system/cpu/cpufreq/policy1",
            read_file=values.__getitem__,
        )

        self.assertFalse(ok)
        self.assertIn("still readable", reason)

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.time.sleep")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.time.monotonic")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._is_cpu_online")
    def test_wait_for_cpu_online_polls_until_online(
        self, mock_is_online, mock_monotonic, mock_sleep
    ):
        """Online confirmation must poll instead of relying on one fixed settle sleep."""
        mock_is_online.side_effect = [False, False, True]
        mock_monotonic.side_effect = [0.0, 0.0, 1.0, 2.0]

        ok, reason = cpufreq_stress._wait_for_cpu_online(
            1, timeout_secs=5, poll_secs=1, logger=self.mock_logger
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        self.assertEqual(mock_is_online.call_count, 3)
        mock_sleep.assert_has_calls([mock.call(1), mock.call(1)])

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.time.sleep")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.time.monotonic")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._is_cpu_online")
    def test_wait_for_cpu_online_times_out(
        self, mock_is_online, mock_monotonic, mock_sleep
    ):
        """Bounded polling must fail when the CPU never reports online."""
        mock_is_online.return_value = False
        mock_monotonic.side_effect = [0.0, 0.0, 1.0, 5.0]

        ok, reason = cpufreq_stress._wait_for_cpu_online(
            1, timeout_secs=5, poll_secs=1, logger=self.mock_logger
        )

        self.assertFalse(ok)
        self.assertIn("did not come back online", reason)
        mock_sleep.assert_called()

    def _patch_hotplug_teardown_and_online(self):
        """Patch offline policy teardown and online polling helpers for hotplug tests."""
        return (
            self.mock.patch.object(
                cpufreq_stress, "_verify_cppc_policy_absent_for_cpu",
                return_value=(True, "")),
            self.mock.patch.object(
                cpufreq_stress, "_wait_for_cpu_online",
                return_value=(True, "")),
        )

    @contextlib.contextmanager
    def _hotplug_policy_mocks(self, *, include_is_online=True):
        """Enter the common patches used by hotplug policy unit tests."""
        with contextlib.ExitStack() as stack:
            ns = types.SimpleNamespace()
            ns.sleep = stack.enter_context(
                mock.patch("src.modules.cpu.glt_test_cpufreq_stress.time.sleep"))
            ns.machine = stack.enter_context(
                mock.patch("src.modules.cpu.glt_test_cpufreq_stress.platform.machine"))
            ns.write = stack.enter_context(
                mock.patch(
                    "src.modules.cpu.glt_test_cpufreq_stress.glt_utils.write_sysfs_value"))
            ns.read = stack.enter_context(
                mock.patch(
                    "src.modules.cpu.glt_test_cpufreq_stress.glt_utils.read_sysfs_value"))
            ns.log_check = stack.enter_context(
                mock.patch(
                    "src.modules.cpu.glt_test_cpufreq_stress.glt_utils.check_kernel_log"))
            ns.checkpoint = stack.enter_context(
                mock.patch(
                    "src.modules.cpu.glt_test_cpufreq_stress.glt_utils."
                    "capture_kernel_log_checkpoint"))
            if include_is_online:
                ns.is_online = stack.enter_context(
                    mock.patch(
                        "src.modules.cpu.glt_test_cpufreq_stress._is_cpu_online"))
            ns.prereq = stack.enter_context(
                mock.patch(
                    "src.modules.cpu.glt_test_cpufreq_stress._validate_cppc_prerequisites"))
            ns.select_cpu = stack.enter_context(
                mock.patch(
                    "src.modules.cpu.glt_test_cpufreq_stress._select_hotplug_cpu"))
            ns.validate_policy = stack.enter_context(
                mock.patch(
                    "src.modules.cpu.glt_test_cpufreq_stress._validate_cppc_policy_for_cpu"))
            yield ns

    def _configure_default_hotplug_mocks(self, ns, *, is_online=False):
        """Apply aarch64 / prereq-ok / CPU1 defaults shared by hotplug policy tests."""
        ns.machine.return_value = "aarch64"
        ns.prereq.return_value = (True, None)
        ns.read.return_value = "0-2"
        ns.select_cpu.return_value = (1, "")
        ns.validate_policy.return_value = (True, "")
        ns.log_check.return_value = False
        ns.checkpoint.return_value = object()
        if hasattr(ns, "is_online"):
            ns.is_online.return_value = is_online

    def test_hotplug_policy_test_restores_cpu_after_validation_failure(self):
        """The runtime test must restore a CPU even when recreated policy validation fails."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m)
            teardown_patch, online_patch = self._patch_hotplug_teardown_and_online()

            with teardown_patch, online_patch, self.mock.patch.object(
                    cpufreq_stress, "_wait_for_cppc_policy_for_cpu",
                    return_value=(False, "policy1 did not recreate")):
                with self.assertRaises(GLTCore.GLTTestFail):
                    cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                        None, self.mock_logger
                    )

            online_path = "/sys/devices/system/cpu/cpu1/online"
            m.write.assert_has_calls([
                mock.call(online_path, "0"),
                mock.call(online_path, "1"),
                mock.call(online_path, "1"),
            ])

    def test_hotplug_policy_test_passes_and_restores_cpu(self):
        """The DGX-17150 path cycles a CPU and restores it after a clean validation."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m)
            teardown_patch, online_patch = self._patch_hotplug_teardown_and_online()

            with teardown_patch, online_patch, self.mock.patch.object(
                    cpufreq_stress, "_wait_for_cppc_policy_for_cpu",
                    return_value=(True, "")):
                cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                    None, self.mock_logger
                )

            online_path = "/sys/devices/system/cpu/cpu1/online"
            m.write.assert_has_calls([
                mock.call(online_path, "0"),
                mock.call(online_path, "1"),
                mock.call(online_path, "1"),
            ])
            m.checkpoint.assert_called_once_with()
            m.log_check.assert_called_once_with(
                cpufreq_stress.CPUFREQ_ONLINE_GET_FAILED_MSG,
                checkpoint=m.checkpoint.return_value,
            )
            self.mock_logger.info.assert_any_call(
                "PASS: CPU%d cppc_cpufreq policy recreated after hotplug", 1
            )

    def test_hotplug_policy_test_ignores_stale_kernel_log_signature(self):
        """A boot log signature that existed before the checkpoint must not fail."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m)
            m.log_check.return_value = False
            teardown_patch, online_patch = self._patch_hotplug_teardown_and_online()

            with teardown_patch, online_patch, self.mock.patch.object(
                    cpufreq_stress, "_wait_for_cppc_policy_for_cpu",
                    return_value=(True, "")):
                cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                    None, self.mock_logger
                )

            m.log_check.assert_called_once_with(
                cpufreq_stress.CPUFREQ_ONLINE_GET_FAILED_MSG,
                checkpoint=m.checkpoint.return_value,
            )

    def test_hotplug_policy_test_fails_on_new_kernel_log_signature(self):
        """A new online-get failure emitted by this hotplug cycle must fail."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m)
            m.log_check.return_value = True
            teardown_patch, online_patch = self._patch_hotplug_teardown_and_online()

            with teardown_patch, online_patch, self.mock.patch.object(
                    cpufreq_stress, "_wait_for_cppc_policy_for_cpu",
                    return_value=(True, "")):
                with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                    cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                        None, self.mock_logger
                    )

            self.assertIn(
                cpufreq_stress.CPUFREQ_ONLINE_GET_FAILED_MSG, str(ctx.exception)
            )

    def test_hotplug_policy_test_fails_when_cpu_never_goes_offline(self):
        """The test must prove the CPU actually left online state before validating policy."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m, is_online=True)

            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                    None, self.mock_logger
                )

            self.assertIn("remained online", str(ctx.exception))

    def test_hotplug_policy_test_fails_when_policy_still_readable_while_offline(self):
        """The test must prove policy teardown before bringing the CPU back online."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m)

            with self.mock.patch.object(
                    cpufreq_stress, "_verify_cppc_policy_absent_for_cpu",
                    return_value=(False, "cpufreq policy still readable for offline CPU1")):
                with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                    cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                        None, self.mock_logger
                    )

            self.assertIn("still readable", str(ctx.exception))

    def test_hotplug_policy_test_fails_when_cpu_does_not_come_back_online(self):
        """The test must prove the CPU returned online before checking policy state."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m)

            with self.mock.patch.object(
                    cpufreq_stress, "_verify_cppc_policy_absent_for_cpu",
                    return_value=(True, "")), self.mock.patch.object(
                    cpufreq_stress, "_wait_for_cpu_online",
                    return_value=(False, "CPU1 did not come back online after hotplug.")):
                with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                    cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                        None, self.mock_logger
                    )

            self.assertIn("did not come back online", str(ctx.exception))

    def test_validate_cppc_policy_for_cpu_rejects_shared_policy(self):
        """Post-hotplug validation must still prove a single-CPU policy."""
        values = {
            "/sys/devices/system/cpu/cpufreq/policy1/scaling_driver": "cppc_cpufreq",
            "/sys/devices/system/cpu/cpufreq/policy1/affected_cpus": "1-2",
            "/sys/devices/system/cpu/cpufreq/policy1/related_cpus": "1-2",
            "/sys/devices/system/cpu/cpufreq/policy1/scaling_cur_freq": "3240000",
            "/sys/devices/system/cpu/cpufreq/policy1/cpuinfo_max_freq": "3348000",
        }

        with self.mock.patch.object(
                cpufreq_stress.glt_utils, "read_sysfs_value", side_effect=values.__getitem__):
            ok, reason = cpufreq_stress._validate_cppc_policy_for_cpu(
                1,
                path_exists=lambda path: path == "/sys/devices/system/cpu/cpufreq/policy1",
            )

        self.assertFalse(ok)
        self.assertIn("affected_cpus is not CPU1", reason)

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.platform.machine")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._validate_cppc_prerequisites")
    def test_hotplug_policy_test_skips_when_cppc_is_not_active(
        self, mock_prereq, mock_machine
    ):
        """Generic ARM64 systems without cppc_cpufreq are not applicable."""
        mock_machine.return_value = "aarch64"
        mock_prereq.return_value = (
            False,
            "cppc_cpufreq not active for all cpufreq policies (policy0: acpi-cpufreq).",
        )

        with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
            cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                None, self.mock_logger
            )

        self.assertIn("cppc_cpufreq not active", str(ctx.exception))

    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.platform.machine")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress.glt_utils.read_sysfs_value")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._validate_cppc_prerequisites")
    @mock.patch("src.modules.cpu.glt_test_cpufreq_stress._select_hotplug_cpu")
    def test_hotplug_policy_test_skips_when_no_hotplug_cpu_is_available(
        self, mock_select_cpu, mock_prereq, mock_read, mock_machine
    ):
        """Shared-policy or non-writable systems should skip instead of pretending coverage."""
        mock_machine.return_value = "aarch64"
        mock_prereq.return_value = (True, None)
        mock_read.return_value = "0-2"
        mock_select_cpu.return_value = (
            None,
            "policy1 is shared; cannot prove policy recreation.",
        )

        with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
            cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                None, self.mock_logger
            )

        self.assertIn("cannot prove policy recreation", str(ctx.exception))

    def test_hotplug_policy_test_reports_restore_failure_with_original_failure(self):
        """A cleanup failure must be visible even when validation already failed."""
        with self._hotplug_policy_mocks() as m:
            self._configure_default_hotplug_mocks(m)
            teardown_patch, online_patch = self._patch_hotplug_teardown_and_online()

            def write_side_effect(path, value):
                if path == "/sys/devices/system/cpu/cpu1/online" and value == "1" and \
                        m.write.call_count >= 3:
                    raise OSError("restore failed")

            m.write.side_effect = write_side_effect

            with teardown_patch, online_patch, self.mock.patch.object(
                    cpufreq_stress, "_wait_for_cppc_policy_for_cpu",
                    return_value=(False, "policy1 did not recreate")):
                with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                    cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                        None, self.mock_logger
                    )

            self.assertIn("policy1 did not recreate", str(ctx.exception))
            self.assertIn("restore failed", str(ctx.exception))

    def test_hotplug_policy_test_reports_offline_write_failure(self):
        """A failure to offline the CPU must be reported as a hotplug failure."""
        with self._hotplug_policy_mocks(include_is_online=False) as m:
            self._configure_default_hotplug_mocks(m)

            def write_side_effect(path, value):
                if path == "/sys/devices/system/cpu/cpu1/online" and value == "0":
                    raise OSError("offline failed")

            m.write.side_effect = write_side_effect

            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                cpufreq_stress.test_cppc_cpufreq_policy_after_cpu_hotplug(
                    None, self.mock_logger
                )

            self.assertIn("Cannot hotplug CPU1", str(ctx.exception))
            self.assertIn("offline failed", str(ctx.exception))
