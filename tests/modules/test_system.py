"""Unit tests for system-state readiness checks."""
# pylint: disable=wrong-import-position

import subprocess
import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from modules.system import glt_test_system as system_test


STATE_COMMAND = ["systemctl", "is-system-running", "--wait"]
FAILED_COMMAND = ["systemctl", "--failed"]
RUN_KWARGS = {
    "capture_output": True,
    "text": True,
    "check": False,
    "timeout": 30,
}


class TestSystemStateIsRunning(GLTTestBase):
    """Validate systemd readiness handling for DGX-17387."""

    def _prerequisite_patches(self, systemd_running=True):
        """Return patches for a host with systemctl and configurable systemd."""
        return (
            self.mock.patch.object(
                system_test.glt_utils,
                "is_cli_available",
                return_value=True,
            ),
            self.mock.patch.object(
                system_test.glt_utils,
                "is_systemd_running",
                return_value=systemd_running,
            ),
        )

    def test_service_uses_simple_type_for_boot_readiness_wait(self):
        """The boot service must not keep a oneshot job pending during the wait."""
        service_path = (
            Path(__file__).resolve().parents[2] / "confs" / "greenlit.service"
        )
        directives = [
            line.strip()
            for line in service_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertIn("Type=simple", directives)
        self.assertNotIn("Type=oneshot", directives)

    def test_system_state_degraded_reports_failed_units(self):
        """A settled degraded state reports the failed systemd units."""
        state_result = subprocess.CompletedProcess(
            STATE_COMMAND,
            1,
            stdout="degraded\n",
            stderr="",
        )
        failed_output = (
            "UNIT LOAD ACTIVE SUB DESCRIPTION\n"
            "openibd.service loaded failed failed OpenIB Drivers\n"
        )
        failed_result = subprocess.CompletedProcess(
            FAILED_COMMAND,
            0,
            stdout=failed_output,
            stderr="",
        )
        cli_patch, systemd_patch = self._prerequisite_patches()

        with cli_patch, systemd_patch, self.mock.patch.object(
            system_test.subprocess,
            "run",
            side_effect=[state_result, failed_result],
        ) as mock_run, self.assertRaises(GLTCore.GLTTestFail) as context:
            system_test.test_system_state_is_running(None, None)

        self.assertIn("System state is 'degraded'", str(context.exception))
        self.assertIn("openibd.service", str(context.exception))
        self.assertEqual(
            mock_run.call_args_list,
            [
                self.mock.call(STATE_COMMAND, **RUN_KWARGS),
                self.mock.call(FAILED_COMMAND, **RUN_KWARGS),
            ],
        )

    def test_system_state_missing_systemctl_skips(self):
        """A host without systemctl is not applicable to this test."""
        with self.mock.patch.object(
            system_test.glt_utils,
            "is_cli_available",
            return_value=False,
        ), self.mock.patch.object(
            system_test.glt_utils,
            "is_systemd_running",
        ) as mock_systemd, self.mock.patch.object(
            system_test.subprocess,
            "run",
        ) as mock_run, self.assertRaises(GLTCore.GLTTestSkip) as context:
            system_test.test_system_state_is_running(None, None)

        self.assertIn("systemctl is not available", str(context.exception))
        mock_systemd.assert_not_called()
        mock_run.assert_not_called()

    def test_system_state_non_systemd_skips(self):
        """A host whose PID 1 is not systemd is not applicable."""
        cli_patch, systemd_patch = self._prerequisite_patches(
            systemd_running=False
        )

        with cli_patch, systemd_patch, self.mock.patch.object(
            system_test.subprocess,
            "run",
        ) as mock_run, self.assertRaises(GLTCore.GLTTestSkip) as context:
            system_test.test_system_state_is_running(None, None)

        self.assertIn("not booted with systemd", str(context.exception))
        mock_run.assert_not_called()

    def test_system_state_running_passes_after_wait(self):
        """A settled running state passes without querying failed units."""
        state_result = subprocess.CompletedProcess(
            STATE_COMMAND,
            0,
            stdout="running\n",
            stderr="",
        )
        cli_patch, systemd_patch = self._prerequisite_patches()

        with cli_patch, systemd_patch, self.mock.patch.object(
            system_test.subprocess,
            "run",
            return_value=state_result,
        ) as mock_run:
            system_test.test_system_state_is_running(None, None)

        mock_run.assert_called_once_with(STATE_COMMAND, **RUN_KWARGS)

    def test_system_state_wait_timeout_fails_without_unit_query(self):
        """An unsettled boot fails after the bounded wait and stops diagnostics."""
        timeout = subprocess.TimeoutExpired(STATE_COMMAND, 30)
        cli_patch, systemd_patch = self._prerequisite_patches()

        with cli_patch, systemd_patch, self.mock.patch.object(
            system_test.subprocess,
            "run",
            side_effect=timeout,
        ) as mock_run, self.assertRaises(GLTCore.GLTTestFail) as context:
            system_test.test_system_state_is_running(None, None)

        self.assertIn("timed out after 30 seconds", str(context.exception))
        mock_run.assert_called_once_with(STATE_COMMAND, **RUN_KWARGS)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
