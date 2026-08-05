"""Unit tests for DGX-16331 CPPC online-CPU coverage."""
# pylint: disable=wrong-import-position,protected-access,too-many-public-methods

import sys
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.modules.cpu import glt_test_cppc_online_cpu as cppc_online


class TestCppcOnlineCpuHelpers(GLTTestBase):
    """Test deterministic helpers for CPPC online-CPU validation."""

    def test_parse_cpu_mask_handles_ranges_and_lists(self):
        """CPU topology masks must be parsed before comparing boot state."""
        self.assertEqual(cppc_online._parse_cpu_mask("0-3,8,10-11"), {0, 1, 2, 3, 8, 10, 11})
        self.assertEqual(cppc_online._parse_cpu_mask(""), set())

    def test_parse_cpu_mask_rejects_reversed_range(self):
        """Malformed masks should fail loudly instead of producing bad topology."""
        with self.assertRaises(ValueError):
            cppc_online._parse_cpu_mask("4-2")

    def test_topology_is_partial_when_present_exceeds_online(self):
        """The bug is exercisable only when boot topology has offline CPUs."""
        online = {0, 1, 2, 3}
        self.assertTrue(cppc_online._has_partial_online_topology({0, 1, 2, 3, 4}, online))
        self.assertFalse(cppc_online._has_partial_online_topology({0, 1, 2, 3}, online))

    def test_topology_is_not_partial_for_possible_only_cpus(self):
        """Possible-only CPUs are not enough evidence for never-onlined present CPUs."""
        online = {0, 1, 2, 3}
        present = {0, 1, 2, 3}
        possible = {0, 1, 2, 3, 4}

        self.assertTrue(possible - online)
        self.assertFalse(cppc_online._has_partial_online_topology(present, online))

    def test_recommended_maxcpus_uses_half_of_present_cpus(self):
        """The reboot setup should leave enough CPUs online while making the bug visible."""
        self.assertEqual(cppc_online._recommended_maxcpus(set(range(72))), 36)
        self.assertEqual(cppc_online._recommended_maxcpus({0, 1}), 1)

    def test_get_maxcpus_parameter_ignores_timestamp_marker(self):
        """The progress marker is a timestamp, not a recoverable maxcpus value."""
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_MAXCPUS_PARAM_MARKER: "2026-06-22 06:16:58",
            }
        }]

        self.assertEqual(cppc_online._get_maxcpus_parameter("test_case", self.mock_sd), "")

    def test_cmdline_evidence_detects_boot_time_cpu_limit(self):
        """Runtime masks need command-line evidence to be strong proof."""
        self.assertTrue(cppc_online._has_boot_time_cpu_limit("BOOT_IMAGE=/vmlinuz maxcpus=36"))
        self.assertTrue(cppc_online._has_boot_time_cpu_limit("BOOT_IMAGE=/vmlinuz nr_cpus=36"))
        self.assertFalse(cppc_online._has_boot_time_cpu_limit("BOOT_IMAGE=/vmlinuz quiet"))

    def test_prepare_maxcpus_boot_parameter_reboots_when_all_cpus_online(self):
        """All-online DUTs need GreenLiT-owned maxcpus setup before validation."""
        initial_status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        final_status = self.mock.Mock(
            active=False, configured=True, added_by_glt=True, reboot_required=True
        )
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    side_effect=[initial_status, final_status]), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(True, "")) as mock_append, \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(True, "")) as mock_remove, \
                self.mock.patch.object(cppc_online.glt_utils, "reboot",
                                       return_value=False) as mock_reboot:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("Reboot", reason)
        mock_append.assert_called_once_with(
            "maxcpus=4",
            test_case_name="test_case",
            session_dir=self.mock_sd.log_dir,
        )
        self.mock_sd.progress_record_time.assert_called_once_with(
            "test_case", cppc_online.CPPC_MAXCPUS_PARAM_MARKER
        )
        self.mock_sd.progress_update.assert_called_once_with(
            "test_case", cppc_online.CPPC_MAXCPUS_PARAM_VALUE, "maxcpus=4"
        )
        mock_reboot.assert_called_once_with("test_case", self.mock_sd)
        mock_remove.assert_called_once_with(
            "maxcpus=4",
            test_case_name="test_case",
            session_dir=self.mock_sd.log_dir,
        )

    def test_prepare_maxcpus_boot_parameter_uses_present_cpu_count(self):
        """maxcpus should be sized from present CPUs, not possible-only CPUs."""
        initial_status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        final_status = self.mock.Mock(
            active=False, configured=True, added_by_glt=True, reboot_required=True
        )
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    side_effect=[initial_status, final_status]), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(True, "")) as mock_append, \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot",
                                       return_value=False):
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(4)),
                online_cpus=set(range(4)),
            )

        self.assertFalse(ok)
        self.assertIn("Reboot", reason)
        mock_append.assert_called_once_with(
            "maxcpus=2",
            test_case_name="test_case",
            session_dir=self.mock_sd.log_dir,
        )

    def test_prepare_maxcpus_boot_parameter_does_not_reboot_partial_topology(self):
        """Existing partial topology should proceed without mutating boot config."""
        with self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter") as mock_append, \
                self.mock.patch.object(cppc_online.glt_utils, "reboot") as mock_reboot:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(4)),
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        mock_append.assert_not_called()
        mock_reboot.assert_not_called()

    def test_prepare_maxcpus_boot_parameter_cleans_up_after_append_failure(self):
        """Failed append attempts still need explicit cleanup of the chosen parameter."""
        status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(False, "append failed")), \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(True, "")) as mock_remove:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("append failed", reason)
        mock_remove.assert_called_once_with(
            "maxcpus=4",
            test_case_name="test_case",
            session_dir=self.mock_sd.log_dir,
        )

    def test_prepare_maxcpus_boot_parameter_fails_when_parameter_already_active(self):
        """An active maxcpus setting without partial topology cannot prove the bug state."""
        status = self.mock.Mock(active=True)
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter"
                                       ) as mock_append:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("maxcpus=4 is active", reason)
        mock_append.assert_not_called()

    def test_prepare_maxcpus_boot_parameter_fails_when_reboot_did_not_make_partial(self):
        """A completed setup reboot must actually produce partial CPU topology."""
        status = self.mock.Mock(active=False)
        self.mock_sd.progress_check.return_value = True

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter"
                                       ) as mock_append:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("rebooted after configuring", reason)
        mock_append.assert_not_called()

    def test_prepare_maxcpus_boot_parameter_restores_after_ineffective_setup_reboot(self):
        """Ineffective setup reboots must not leave GreenLiT-owned maxcpus active."""
        self.mock_sd.progress_check.return_value = True
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
            }
        }]
        status = self.mock.Mock(active=True)

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(True, "")) as mock_remove, \
                self.mock.patch.object(cppc_online.glt_utils, "reboot",
                                       return_value=False) as mock_reboot:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("rebooted after configuring", reason)
        self.assertIn("Restore reboot", reason)
        self.mock_sd.progress_update.assert_called_once_with(
            "test_case",
            cppc_online.CPPC_RESTORE_RESULT_REASON,
            "System rebooted after configuring maxcpus=4, but all present CPUs are still online.",
        )
        self.mock_sd.progress_record_time.assert_called_once_with(
            "test_case", cppc_online.CPPC_RESTORE_REBOOT_MARKER
        )
        mock_remove.assert_called_once_with(
            "maxcpus=4",
            test_case_name="test_case",
            session_dir=self.mock_sd.log_dir,
        )
        mock_reboot.assert_called_once_with("test_case", self.mock_sd)

    def test_prepare_maxcpus_boot_parameter_reports_cleanup_failure_after_append_failure(self):
        """Append failures should include cleanup failures so parameter leaks are visible."""
        status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(False, "append failed")), \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(False, "remove failed")):
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("append failed", reason)
        self.assertIn("remove failed", reason)

    def test_prepare_maxcpus_boot_parameter_records_configured_parameter(self):
        """Configured boot parameters must be tracked even when ownership metadata is absent."""
        initial_status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        final_status = self.mock.Mock(
            active=False, configured=True, added_by_glt=False, reboot_required=True
        )
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    side_effect=[initial_status, final_status]), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot", return_value=False):
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("Reboot", reason)
        self.mock_sd.progress_record_time.assert_called_once_with(
            "test_case", cppc_online.CPPC_MAXCPUS_PARAM_MARKER
        )
        self.mock_sd.progress_update.assert_called_once_with(
            "test_case", cppc_online.CPPC_MAXCPUS_PARAM_VALUE, "maxcpus=4"
        )

    def test_prepare_maxcpus_boot_parameter_records_parameter_on_inconsistent_status(self):
        """A successful append must leave enough state for later cleanup."""
        initial_status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        final_status = self.mock.Mock(
            active=False, configured=False, added_by_glt=False, reboot_required=True
        )
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    side_effect=[initial_status, final_status]), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot", return_value=False):
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("Reboot", reason)
        self.mock_sd.progress_record_time.assert_called_once_with(
            "test_case", cppc_online.CPPC_MAXCPUS_PARAM_MARKER
        )
        self.mock_sd.progress_update.assert_called_once_with(
            "test_case", cppc_online.CPPC_MAXCPUS_PARAM_VALUE, "maxcpus=4"
        )

    def test_prepare_maxcpus_boot_parameter_fails_when_reboot_is_not_required(self):
        """A boot-only maxcpus setup that needs no reboot cannot create the target topology."""
        initial_status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        final_status = self.mock.Mock(
            active=True, configured=True, added_by_glt=True, reboot_required=False
        )
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    side_effect=[initial_status, final_status]), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(True, "")) as mock_remove, \
                self.mock.patch.object(cppc_online.glt_utils, "reboot") as mock_reboot:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("no reboot was required", reason)
        mock_remove.assert_called_once_with(
            "maxcpus=4",
            test_case_name="test_case",
            session_dir=self.mock_sd.log_dir,
        )
        self.mock_sd.progress_delete_time.assert_has_calls([
            self.mock.call("test_case", cppc_online.CPPC_MAXCPUS_PARAM_MARKER),
            self.mock.call("test_case", cppc_online.CPPC_MAXCPUS_PARAM_VALUE),
        ])
        self.assertEqual(self.mock_sd.progress_delete_time.call_count, 2)
        mock_reboot.assert_not_called()

    def test_prepare_maxcpus_boot_parameter_preserves_markers_when_no_reboot_cleanup_fails(self):
        """Failed no-reboot cleanup should keep progress state for later recovery."""
        initial_status = self.mock.Mock(active=False, configured=False, added_by_glt=False)
        final_status = self.mock.Mock(
            active=True, configured=True, added_by_glt=True, reboot_required=False
        )
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    side_effect=[initial_status, final_status]), \
                self.mock.patch.object(cppc_online.glt_utils, "append_kernel_parameter",
                                       return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                       return_value=(False, "remove failed")), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot") as mock_reboot:
            ok, reason = cppc_online._prepare_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger,
                present_cpus=set(range(8)),
                online_cpus=set(range(8)),
            )

        self.assertFalse(ok)
        self.assertIn("no reboot was required", reason)
        self.assertIn("remove failed", reason)
        self.mock_sd.progress_delete_time.assert_not_called()
        mock_reboot.assert_not_called()

    def test_restore_maxcpus_boot_parameter_removes_and_reboots_when_active(self):
        """Restore cleanup must reboot when the old maxcpus setting remains active."""
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
            }
        }]
        status = self.mock.Mock(active=True)

        with self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                    return_value=(True, "")) as mock_remove, \
                self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                       return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot",
                                       return_value=False) as mock_reboot:
            ok, reason = cppc_online._restore_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger, "original failure"
            )

        self.assertFalse(ok)
        self.assertIn("Restore reboot", reason)
        self.mock_sd.progress_update.assert_called_once_with(
            "test_case",
            cppc_online.CPPC_RESTORE_RESULT_REASON,
            "original failure",
        )
        self.mock_sd.progress_record_time.assert_called_once_with(
            "test_case", cppc_online.CPPC_RESTORE_REBOOT_MARKER
        )
        mock_remove.assert_called_once_with(
            "maxcpus=4",
            test_case_name="test_case",
            session_dir=self.mock_sd.log_dir,
        )
        mock_reboot.assert_called_once_with("test_case", self.mock_sd)

    def test_restore_maxcpus_boot_parameter_reboots_without_result_reason(self):
        """Restore reboot should not persist an empty failure reason."""
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
            }
        }]
        status = self.mock.Mock(active=True)

        with self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                    return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                       return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot",
                                       return_value=False) as mock_reboot:
            ok, reason = cppc_online._restore_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger
            )

        self.assertFalse(ok)
        self.assertIn("Restore reboot", reason)
        self.mock_sd.progress_update.assert_not_called()
        self.mock_sd.progress_record_time.assert_called_once_with(
            "test_case", cppc_online.CPPC_RESTORE_REBOOT_MARKER
        )
        mock_reboot.assert_called_once_with("test_case", self.mock_sd)

    def test_restore_maxcpus_boot_parameter_ignores_reason_without_parameter(self):
        """Failure reasons should not be persisted when there is no GLT-owned maxcpus."""
        self.mock_sd.progress = [{"test_case": {}}]

        ok, reason = cppc_online._restore_maxcpus_boot_parameter(
            "test_case", self.mock_sd, self.mock_logger, "original failure"
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        self.mock_sd.progress_update.assert_not_called()
        self.mock_sd.progress_delete_time.assert_not_called()

    def test_restore_maxcpus_boot_parameter_reports_remove_failure(self):
        """Restore cleanup failures must be returned without rebooting."""
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
            }
        }]

        with self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                    return_value=(False, "remove failed")), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot") as mock_reboot:
            ok, reason = cppc_online._restore_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger, "original failure"
            )

        self.assertFalse(ok)
        self.assertIn("remove failed", reason)
        self.mock_sd.progress_update.assert_called_once_with(
            "test_case",
            cppc_online.CPPC_RESTORE_RESULT_REASON,
            "original failure",
        )
        mock_reboot.assert_not_called()

    def test_restore_maxcpus_boot_parameter_clears_markers_when_no_reboot_needed(self):
        """Restore cleanup should clear stale maxcpus markers when removal is immediately done."""
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_MAXCPUS_PARAM_MARKER: "timestamp",
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
                cppc_online.CPPC_RESTORE_RESULT_REASON: "original failure",
            }
        }]
        status = self.mock.Mock(active=False)

        with self.mock.patch.object(cppc_online.glt_utils, "remove_kernel_parameter",
                                    return_value=(True, "")), \
                self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                       return_value=status), \
                self.mock.patch.object(cppc_online.glt_utils, "reboot") as mock_reboot:
            ok, reason = cppc_online._restore_maxcpus_boot_parameter(
                "test_case", self.mock_sd, self.mock_logger
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        self.mock_sd.progress_delete_time.assert_has_calls([
            self.mock.call("test_case", cppc_online.CPPC_MAXCPUS_PARAM_MARKER),
            self.mock.call("test_case", cppc_online.CPPC_MAXCPUS_PARAM_VALUE),
            self.mock.call("test_case", cppc_online.CPPC_RESTORE_RESULT_REASON),
        ])
        self.assertEqual(self.mock_sd.progress_delete_time.call_count, 3)
        mock_reboot.assert_not_called()

    def test_finish_restore_reboot_fails_if_parameter_is_still_active(self):
        """A restore reboot is not complete while maxcpus remains active."""
        self.mock_sd.progress_check.return_value = True
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_RESTORE_REBOOT_MARKER: "timestamp",
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
            }
        }]
        status = self.mock.Mock(active=True, configured=False, added_by_glt=False)

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status):
            complete, reason = cppc_online._finish_restore_reboot_if_needed(
                "test_case", self.mock_sd
            )

        self.assertTrue(complete)
        self.assertIn("still active", reason)
        self.mock_sd.progress_delete_time.assert_not_called()

    def test_finish_restore_reboot_fails_if_parameter_is_still_configured(self):
        """Restore reboot is not complete while maxcpus remains configured."""
        self.mock_sd.progress_check.return_value = True
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_RESTORE_REBOOT_MARKER: "timestamp",
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
            }
        }]
        status = self.mock.Mock(active=False, configured=True, added_by_glt=False)

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status):
            complete, reason = cppc_online._finish_restore_reboot_if_needed(
                "test_case", self.mock_sd
            )

        self.assertTrue(complete)
        self.assertIn("still configured", reason)
        self.mock_sd.progress_delete_time.assert_not_called()

    def test_finish_restore_reboot_clears_marker_when_parameter_is_missing(self):
        """Defensive restore errors should not trap every later resumed run."""
        self.mock_sd.progress_check.return_value = True
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_RESTORE_REBOOT_MARKER: "timestamp",
            }
        }]

        complete, reason = cppc_online._finish_restore_reboot_if_needed(
            "test_case", self.mock_sd
        )

        self.assertTrue(complete)
        self.assertIn("no GreenLiT-owned", reason)
        self.mock_sd.progress_delete_time.assert_called_once_with(
            "test_case", cppc_online.CPPC_RESTORE_REBOOT_MARKER
        )

    def test_finish_restore_reboot_clears_progress_when_parameter_is_absent(self):
        """Once maxcpus is gone, restore markers should not poison the next run."""
        self.mock_sd.progress_check.return_value = True
        self.mock_sd.progress = [{
            "test_case": {
                cppc_online.CPPC_RESTORE_REBOOT_MARKER: "timestamp",
                cppc_online.CPPC_RESTORE_RESULT_REASON: "original failure",
                cppc_online.CPPC_MAXCPUS_PARAM_MARKER: "timestamp",
                cppc_online.CPPC_MAXCPUS_PARAM_VALUE: "maxcpus=4",
            }
        }]
        status = self.mock.Mock(active=False, configured=False, added_by_glt=False)

        with self.mock.patch.object(cppc_online.glt_utils, "check_kernel_parameter",
                                    return_value=status):
            complete, reason = cppc_online._finish_restore_reboot_if_needed(
                "test_case", self.mock_sd
            )

        self.assertTrue(complete)
        self.assertEqual(reason, "original failure")
        self.mock_sd.progress_delete_time.assert_has_calls([
            self.mock.call("test_case", cppc_online.CPPC_RESTORE_REBOOT_MARKER),
            self.mock.call("test_case", cppc_online.CPPC_RESTORE_RESULT_REASON),
            self.mock.call("test_case", cppc_online.CPPC_MAXCPUS_PARAM_MARKER),
            self.mock.call("test_case", cppc_online.CPPC_MAXCPUS_PARAM_VALUE),
        ])
        self.assertEqual(self.mock_sd.progress_delete_time.call_count, 4)

    def test_policy_level_cppc_observability_accepts_valid_online_cpus(self):
        """DUTs may expose cpufreq under policy directories instead of cpu directories."""
        files = {
            "/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf": "386\n",
            "/sys/devices/system/cpu/cpu1/acpi_cppc/highest_perf": "386\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpufreq/policy0/related_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/affected_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq": "3465000\n",
            "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq": "3465000\n",
            "/sys/devices/system/cpu/cpufreq/policy1/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpufreq/policy1/related_cpus": "1\n",
            "/sys/devices/system/cpu/cpufreq/policy1/affected_cpus": "1\n",
            "/sys/devices/system/cpu/cpufreq/policy1/scaling_cur_freq": "3474000\n",
            "/sys/devices/system/cpu/cpufreq/policy1/cpuinfo_max_freq": "3474000\n",
        }

        ok, reason = cppc_online._validate_cppc_observability(
            online_cpus={0, 1},
            policy_dirs=[
                "/sys/devices/system/cpu/cpufreq/policy0",
                "/sys/devices/system/cpu/cpufreq/policy1",
            ],
            read_file=files.__getitem__,
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_policy_level_cppc_observability_fails_on_missing_online_policy(self):
        """Every online CPU needs CPPC cpufreq policy evidence."""
        files = {
            "/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf": "386\n",
            "/sys/devices/system/cpu/cpu1/acpi_cppc/highest_perf": "386\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpufreq/policy0/related_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/affected_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq": "3465000\n",
            "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq": "3465000\n",
        }

        ok, reason = cppc_online._validate_cppc_observability(
            online_cpus={0, 1},
            policy_dirs=["/sys/devices/system/cpu/cpufreq/policy0"],
            read_file=files.__getitem__,
        )

        self.assertFalse(ok)
        self.assertIn("No cppc_cpufreq policy covers online CPU1", reason)

    def test_policy_level_cppc_observability_fails_when_policy_dirs_are_missing(self):
        """CPPC observability requires cpufreq policy directories."""
        files = {
            "/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf": "386\n",
        }

        ok, reason = cppc_online._validate_cppc_observability(
            online_cpus={0},
            policy_dirs=[],
            read_file=files.__getitem__,
        )

        self.assertFalse(ok)
        self.assertIn("not available", reason)

    def test_policy_level_cppc_observability_fails_when_affected_has_offline_cpu(self):
        """Policy affected CPUs must not include CPUs that never came online."""
        files = {
            "/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf": "386\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpufreq/policy0/related_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/affected_cpus": "0-1\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq": "3465000\n",
            "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq": "3465000\n",
        }

        ok, reason = cppc_online._validate_cppc_observability(
            online_cpus={0},
            policy_dirs=["/sys/devices/system/cpu/cpufreq/policy0"],
            read_file=files.__getitem__,
        )

        self.assertFalse(ok)
        self.assertIn("affected_cpus includes offline CPUs", reason)

    def test_policy_level_cppc_observability_fails_when_related_has_offline_cpu(self):
        """Policy masks must not include CPUs that never came online."""
        files = {
            "/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf": "386\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpufreq/policy0/related_cpus": "0-1\n",
            "/sys/devices/system/cpu/cpufreq/policy0/affected_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq": "3465000\n",
            "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq": "3465000\n",
        }

        ok, reason = cppc_online._validate_cppc_observability(
            online_cpus={0},
            policy_dirs=["/sys/devices/system/cpu/cpufreq/policy0"],
            read_file=files.__getitem__,
        )

        self.assertFalse(ok)
        self.assertIn("related_cpus includes offline CPUs", reason)

    def test_policy_level_cppc_observability_fails_when_driver_is_not_cppc(self):
        """The test must reject non-CPPC cpufreq policy evidence."""
        files = {
            "/sys/devices/system/cpu/cpu0/acpi_cppc/highest_perf": "386\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_driver": "acpi-cpufreq\n",
            "/sys/devices/system/cpu/cpufreq/policy0/related_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/affected_cpus": "0\n",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq": "3465000\n",
            "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq": "3465000\n",
        }

        ok, reason = cppc_online._validate_cppc_observability(
            online_cpus={0},
            policy_dirs=["/sys/devices/system/cpu/cpufreq/policy0"],
            read_file=files.__getitem__,
        )

        self.assertFalse(ok)
        self.assertIn("expected cppc_cpufreq", reason)

    def test_cppc_online_cpu_coverage_prefers_first_matching_signature(self):
        """Stop at the first matching CPPC signature instead of checking the rest."""
        self.mock_sd.progress_check.return_value = False

        with self.mock.patch.object(
            cppc_online, "_finish_restore_reboot_if_needed", return_value=(False, "")
        ), self.mock.patch.object(
            cppc_online.platform, "machine", return_value="aarch64"
        ), self.mock.patch.object(
            cppc_online, "_read_cpu_mask", side_effect=[{0, 1, 2, 3}, {0, 1}]
        ), self.mock.patch.object(
            cppc_online, "_read_file", return_value="maxcpus=2"
        ), self.mock.patch.object(
            cppc_online, "_prepare_maxcpus_boot_parameter", return_value=(True, "")
        ), self.mock.patch.object(
            cppc_online, "_validate_cppc_observability", return_value=(True, "")
        ), self.mock.patch.object(
            cppc_online.glt_utils,
            "check_kernel_log",
            side_effect=[True, RuntimeError("journal lost")],
        ) as check_log, self.mock.patch.object(
            cppc_online, "_restore_maxcpus_boot_parameter", return_value=(True, "")
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                cppc_online.test_cppc_online_cpu_coverage(
                    self.mock_sd, self.mock_logger
                )

        self.assertIn("cppc_allow_fast_switch", str(ctx.exception))
        self.assertNotIn("Failed to collect kernel logs", str(ctx.exception))
        check_log.assert_called_once_with("cppc_allow_fast_switch")


if __name__ == "__main__":
    GLTTestBase.unittest.main()
