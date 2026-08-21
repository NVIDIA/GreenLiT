"""Unit tests for NVIDIA Olympus PMCCNTR_EL0 coverage."""
# pylint: disable=duplicate-code,protected-access,too-many-public-methods

import subprocess
import sys
import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase

from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_olympus_pmccntr as olympus_pmccntr


MODULE_PATH = "src.modules.arm64.glt_test_olympus_pmccntr"


class TestOlympusPmccntrHelpers(GLTTestBase):
    """Test Olympus PMCCNTR parsing, execution, and precondition helpers."""

    def _start_patch(self, target, **kwargs):
        """Start a patch that is automatically stopped after each test."""
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def _write_topology(root, control="notsupported", online="0-1", siblings=None):
        """Create a minimal CPU topology sysfs fixture."""
        sibling_values = siblings or {0: "0", 1: "1"}
        smt_dir = root / "smt"
        smt_dir.mkdir(parents=True)
        (smt_dir / "control").write_text(control, encoding="utf-8")
        (root / "online").write_text(online, encoding="utf-8")
        for cpu_id, value in sibling_values.items():
            topology_dir = root / f"cpu{cpu_id}" / "topology"
            topology_dir.mkdir(parents=True)
            (topology_dir / "thread_siblings_list").write_text(
                value, encoding="utf-8"
            )

    # --- Firmware and watchdog preconditions ---

    def test_firmware_single_thread_accepts_notsupported_singletons(self):
        """UEFI-disabled SMT with singleton sibling masks is eligible."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_topology(root)

            self.assertEqual(
                olympus_pmccntr._firmware_single_thread_reason(root), ""
            )

    def test_firmware_single_thread_rejects_empty_online_mask(self):
        """An empty online mask cannot establish singleton topology."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_topology(root, online="")

            with self.assertRaisesRegex(ValueError, "is empty"):
                olympus_pmccntr._firmware_single_thread_reason(root)

    def test_firmware_single_thread_skips_known_unsupported_states(self):
        """Known non-UEFI-disabled SMT states are unsupported, not failures."""
        expected_controls = {"forceoff", "notimplemented", "off", "on"}
        for control in sorted(expected_controls):
            with self.subTest(control=control), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                self._write_topology(root, control=control)

                reason = olympus_pmccntr._firmware_single_thread_reason(root)

            self.assertIn("SMT disabled in UEFI", reason)
            self.assertIn("nosmt", reason)

    def test_firmware_single_thread_rejects_exposed_siblings(self):
        """Non-singleton sibling topology makes the patch indistinguishable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_topology(
                root,
                siblings={0: "0-1", 1: "0-1"},
            )

            reason = olympus_pmccntr._firmware_single_thread_reason(root)

        self.assertIn("exposes SMT siblings", reason)

    def test_firmware_single_thread_rejects_inconsistent_sibling_mask(self):
        """A sibling mask that omits its CPU is malformed topology."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_topology(
                root,
                siblings={0: "1", 1: "1"},
            )

            with self.assertRaisesRegex(ValueError, "does not include CPU 0"):
                olympus_pmccntr._firmware_single_thread_reason(root)

    def test_firmware_single_thread_rejects_invalid_online_data(self):
        """Missing or malformed online CPU data is an inspection failure."""
        for value in (None, "invalid"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                self._write_topology(root)
                online_path = root / "online"
                if value is None:
                    online_path.unlink()
                else:
                    online_path.write_text(value, encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "Cannot validate online CPU"):
                    olympus_pmccntr._firmware_single_thread_reason(root)

    def test_firmware_single_thread_rejects_invalid_sibling_data(self):
        """Missing or malformed sibling data is an inspection failure."""
        for value in (None, "invalid"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                self._write_topology(root)
                siblings_path = root / "cpu0" / "topology" / "thread_siblings_list"
                if value is None:
                    siblings_path.unlink()
                else:
                    siblings_path.write_text(value, encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "Cannot validate SMT topology"):
                    olympus_pmccntr._firmware_single_thread_reason(root)

    def test_firmware_single_thread_rejects_unknown_control_value(self):
        """An unknown SMT control value is an inspection failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_topology(root, control="unexpected")

            with self.assertRaisesRegex(ValueError, "Cannot interpret SMT control"):
                olympus_pmccntr._firmware_single_thread_reason(root)

    def test_firmware_single_thread_rejects_unreadable_control(self):
        """Missing SMT control data cannot produce a patch-presence result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "unreadable"):
                olympus_pmccntr._firmware_single_thread_reason(temp_dir)

    def test_nmi_watchdog_disabled_is_eligible(self):
        """A disabled PMU hard-lockup watchdog leaves PMCCNTR available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nmi_watchdog"
            path.write_text("0\n", encoding="utf-8")

            self.assertEqual(
                olympus_pmccntr._nmi_watchdog_reason(path), ""
            )

    def test_nmi_watchdog_missing_is_failure(self):
        """A missing watchdog sysctl leaves PMU counter ownership unknown."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing_nmi_watchdog"

            with self.assertRaisesRegex(ValueError, "Cannot confirm"):
                olympus_pmccntr._nmi_watchdog_reason(path)

    def test_nmi_watchdog_enabled_is_ineligible(self):
        """A pinned watchdog cycle event can hide a missing Olympus fix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nmi_watchdog"
            path.write_text("1\n", encoding="utf-8")

            reason = olympus_pmccntr._nmi_watchdog_reason(path)

        self.assertIn("can occupy PMCCNTR_EL0", reason)

    def test_nmi_watchdog_invalid_value_is_failure(self):
        """Unknown watchdog state fails closed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nmi_watchdog"
            path.write_text("unexpected\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Cannot interpret"):
                olympus_pmccntr._nmi_watchdog_reason(path)

    # --- perf stat parsing and execution ---

    def test_parse_cycle_rows_preserves_repeated_counts(self):
        """Repeated cycle events remain ordered for every CPU."""
        stderr = (
            "CPU2,88974508,,cycles,1000419768,100.00,,\n"
            "CPU2,446913,,cpu_cycles:u,1000419768,100.00,,\n"
            "3,125901,,cycles,1000419768,100.00,,\n"
            "CPU3,125901,,cycles:k,1000419768,100.00,,\n"
            "CPU4,999,,cpu-cycles,1000419768,100.00,,\n"
            "CPU3,42,,instructions,1000419768,100.00,,\n"
        )

        self.assertEqual(
            olympus_pmccntr._parse_perf_stat_cycle_rows(stderr),
            {
                2: [88_974_508, 446_913],
                3: [125_901, 125_901],
                4: [999],
            },
        )

    def test_parse_cycle_rows_rejects_unavailable_count(self):
        """Unavailable grouped counters cannot be mistaken for zero."""
        stderr = "CPU0,<not counted>,,cycles,0,0.00,,\n"

        with self.assertRaisesRegex(ValueError, "unavailable"):
            olympus_pmccntr._parse_perf_stat_cycle_rows(stderr)

    def test_parse_cycle_rows_rejects_invalid_counts(self):
        """Malformed and negative cycle values fail closed."""
        cases = (
            ("CPU0,invalid,,cycles,0,0.00,,\n", "invalid cycle count"),
            ("CPU0,-1,,cycles,0,0.00,,\n", "negative cycle count"),
        )
        for stderr, message in cases:
            with self.subTest(stderr=stderr):
                with self.assertRaisesRegex(ValueError, message):
                    olympus_pmccntr._parse_perf_stat_cycle_rows(stderr)

    def test_parse_repeated_cycles_requires_rows(self):
        """Diagnostics without cycle rows make the probe invalid."""
        with self.assertRaisesRegex(ValueError, "No per-CPU cycle counts"):
            olympus_pmccntr._parse_perf_stat_repeated_cycles(
                "WARNING: no events\n"
            )

    def test_parse_repeated_cycles_requires_complete_pairs(self):
        """Every reported CPU must have two grouped cycle counts."""
        stderr = (
            "CPU0,125901,,cycles,1000419768,100.00,,\n"
            "CPU0,125901,,cycles,1000419768,100.00,,\n"
            "CPU1,125901,,cycles,1000419768,100.00,,\n"
        )

        with self.assertRaisesRegex(ValueError, "CPU 1 reported 1 cycle counts"):
            olympus_pmccntr._parse_perf_stat_repeated_cycles(stderr)

    def test_probe_uses_grouped_system_wide_cycle_events(self):
        """The probe compares simultaneous per-CPU cycle events."""
        stderr = (
            "CPU0,125901,,cycles,1000419768,100.00,,\n"
            "CPU0,125901,,cycles,1000419768,100.00,,\n"
        )
        mock_run = self._start_patch(f"{MODULE_PATH}.subprocess.run")
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/perf"], returncode=0, stdout="", stderr=stderr
        )

        self.assertEqual(
            olympus_pmccntr._probe_olympus_pmccntr(
                self.mock_logger, "/perf"
            ),
            {0: [125_901, 125_901]},
        )
        mock_run.assert_called_once_with(
            [
                "/perf",
                "stat",
                "-x,",
                "-a",
                "-A",
                "-e",
                "{cycles,cycles}:e",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=olympus_pmccntr.PROBE_TIMEOUT_SECONDS,
        )

    def test_probe_rejects_nonzero_status(self):
        """A failed perf command cannot be mistaken for counter results."""
        mock_run = self._start_patch(f"{MODULE_PATH}.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["/perf"],
            stderr="permission denied",
        )

        with self.assertRaisesRegex(ValueError, "status 1"):
            olympus_pmccntr._probe_olympus_pmccntr(
                self.mock_logger, "/perf"
            )

    def test_probe_reports_timeout_and_execution_errors(self):
        """A hung or unexecutable perf command produces a bounded error."""
        mock_run = self._start_patch(f"{MODULE_PATH}.subprocess.run")
        errors = (
            (subprocess.TimeoutExpired(["/perf"], 10), "timed out"),
            (OSError("exec failed"), "Unable to execute"),
        )
        for error, message in errors:
            with self.subTest(error=error):
                mock_run.side_effect = error
                with self.assertRaisesRegex(ValueError, message):
                    olympus_pmccntr._probe_olympus_pmccntr(
                        self.mock_logger, "/perf"
                    )


class TestOlympusPmccntrRuntime(GLTTestBase):
    """Test the decorated Olympus PMCCNTR runtime result."""

    def setUp(self):
        """Set up a probe-eligible patched Vera system by default."""
        super().setUp()
        self.mock_nvdev = self._start_patch(f"{MODULE_PATH}.NvDev")
        self.mock_nvdev.return_value.is_cpu_type.return_value = True
        self.mock_topology_reason = self._start_patch(
            f"{MODULE_PATH}._firmware_single_thread_reason",
            return_value="",
        )
        self.mock_watchdog_reason = self._start_patch(
            f"{MODULE_PATH}._nmi_watchdog_reason",
            return_value="",
        )
        self.mock_resolve_perf = self._start_patch(
            f"{MODULE_PATH}.perf_utils.resolve_perf_command",
            return_value=("/perf", ""),
        )
        self.mock_probe = self._start_patch(
            f"{MODULE_PATH}._probe_olympus_pmccntr",
            return_value={0: [125_901, 125_901]},
        )

    def _start_patch(self, target, **kwargs):
        """Start a patch that is automatically stopped after each test."""
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    # --- Decorated runtime test ---

    def test_runtime_skips_non_vera(self):
        """Non-Vera systems skip before checking runtime preconditions."""
        self.mock_nvdev.return_value.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.mock_topology_reason.assert_not_called()
        self.mock_probe.assert_not_called()

    def test_runtime_fails_when_firmware_topology_inspection_fails(self):
        """Unreadable topology is a test failure, not an unsupported skip."""
        self.mock_topology_reason.side_effect = ValueError(
            "Cannot inspect topology"
        )

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "Cannot inspect"):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.mock_watchdog_reason.assert_not_called()
        self.mock_probe.assert_not_called()

    def test_runtime_skips_without_firmware_single_thread_topology(self):
        """Exposed SMT skips before checking perf or counters."""
        self.mock_topology_reason.return_value = "SMT is exposed"

        with self.assertRaisesRegex(GLTCore.GLTTestSkip, "SMT is exposed"):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.mock_watchdog_reason.assert_not_called()
        self.mock_probe.assert_not_called()

    def test_runtime_fails_when_watchdog_inspection_fails(self):
        """Unreadable watchdog state is a failure, not an unsupported skip."""
        self.mock_watchdog_reason.side_effect = ValueError(
            "Cannot inspect watchdog"
        )

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "Cannot inspect"):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.mock_resolve_perf.assert_not_called()
        self.mock_probe.assert_not_called()

    def test_runtime_skips_with_pmu_watchdog(self):
        """An active PMU watchdog skips before resolving perf."""
        self.mock_watchdog_reason.return_value = "watchdog occupies PMCCNTR"

        with self.assertRaisesRegex(GLTCore.GLTTestSkip, "occupies PMCCNTR"):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.mock_resolve_perf.assert_not_called()
        self.mock_probe.assert_not_called()

    def test_runtime_fails_without_perf(self):
        """A probe-eligible Vera system still requires a usable perf."""
        self.mock_resolve_perf.return_value = (None, "no usable perf")

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "no usable perf"):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.mock_probe.assert_not_called()

    def test_runtime_fails_on_probe_error(self):
        """Probe and parser errors become framework failures."""
        self.mock_probe.side_effect = ValueError("missing grouped counters")

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "missing grouped counters"):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

    def test_runtime_skips_without_positive_idle_reference(self):
        """Busy and fully stopped CPUs cannot prove patch presence."""
        self.mock_probe.return_value = {
            0: [500_000_000, 490_000_000],
            1: [0, 0],
        }

        with self.assertRaisesRegex(GLTCore.GLTTestSkip, "No CPU"):
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

    def test_runtime_passes_matching_idle_counts(self):
        """Matching programmable cycle events pass on idle CPUs."""
        self.mock_probe.return_value = {
            0: [500_000_000, 490_000_000],
            1: [0, 0],
            2: [125_901, 125_901],
            3: [50_000_000, 49_000_000],
        }

        olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
            None, self.mock_logger
        )

        self.mock_logger.info.assert_called_once_with(
            "Olympus CPU_CYCLES counters matched on %d sufficiently idle CPUs",
            2,
        )

    def test_runtime_fails_divergent_idle_counts(self):
        """The known PMCCNTR-versus-CPU_CYCLES signature fails actionably."""
        self.mock_probe.return_value = {2: [88_974_508, 446_913]}

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.assertIn("first=88974508", str(context.exception))
        self.assertIn("second=446913", str(context.exception))
        self.assertIn(olympus_pmccntr.FIX_COMMIT, str(context.exception))

    def test_runtime_fails_pmccntr_signature_above_idle_limit(self):
        """A busy reference cannot hide a directional PMCCNTR signature."""
        self.mock_probe.return_value = {
            0: [2_000_000_000, 150_000_000],
            1: [125_901, 125_901],
        }

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        reason = str(context.exception)
        self.assertIn("CPU 0", reason)
        self.assertIn("first=2000000000", reason)
        self.assertIn("second=150000000", reason)
        self.assertIn(olympus_pmccntr.FIX_COMMIT, reason)

    def test_runtime_fails_positive_pmccntr_with_zero_reference(self):
        """A stopped programmable counter remains an actionable divergence."""
        self.mock_probe.return_value = {2: [88_974_508, 0]}

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        self.assertIn("first=88974508", str(context.exception))
        self.assertIn("second=0", str(context.exception))

    def test_runtime_prefers_pmccntr_signature_over_reverse_divergence(self):
        """An Olympus signature remains actionable alongside reverse noise."""
        self.mock_probe.return_value = {
            2: [88_974_508, 446_913],
            3: [0, 125_901],
        }

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        reason = str(context.exception)
        self.assertIn("CPU 2", reason)
        self.assertIn(olympus_pmccntr.FIX_COMMIT, reason)

    def test_runtime_warns_reverse_divergence_as_inconclusive(self):
        """Reverse divergence warns without blaming the Olympus regression."""
        self.mock_probe.return_value = {2: [0, 125_901]}

        with self.assertRaises(GLTCore.GLTTestWarn) as context:
            olympus_pmccntr.test_olympus_pmccntr_wfi_handling(
                None, self.mock_logger
            )

        reason = str(context.exception)
        self.assertIn("unexpected direction", reason)
        self.assertIn("inconclusive", reason)
        self.assertNotIn(olympus_pmccntr.FIX_COMMIT, reason)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
