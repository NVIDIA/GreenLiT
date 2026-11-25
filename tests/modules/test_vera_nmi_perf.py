"""Offline tests for Vera pseudo-NMI PMU overflow validation."""
# pylint: disable=protected-access,too-many-public-methods

import errno
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_vera_nmi_perf as nmi_perf

MODULE_PATH = "src.modules.arm64.glt_test_vera_nmi_perf"
PSEUDO_LOG = "Pseudo-NMIs enabled using forced ICC_PMR_EL1 synchronisation\n"
VALID_CONFIG = (
    "CONFIG_MODULES=y\nCONFIG_MODULE_UNLOAD=y\n"
    "CONFIG_ARM64_PSEUDO_NMI=y\nCONFIG_PERF_EVENTS=y\n"
    "CONFIG_HW_PERF_EVENTS=y\nCONFIG_ARM_PMU=y\nCONFIG_ARM_PMUV3=y\n"
)


def valid_trigger(**changes):
    values = {
        "cpu": 1,
        "duration_ms": 1000,
        "armed": 1,
        "overflows": 10,
        "nmi_overflows": 10,
        "irqoff_overflows": 8,
        "irqoff_nmi_overflows": 8,
        "module_irqoff_nmi_overflows": 2,
    }
    values.update(changes)
    return nmi_perf.NmiPerfTrigger(**values)


class TestNmiPerfHelpers(unittest.TestCase):
    def test_ioctl_abi_matches_u32_layout(self):
        self.assertEqual(nmi_perf.ctypes.sizeof(nmi_perf.NmiPerfTrigger), 36)
        self.assertEqual(nmi_perf.NMI_PERF_IOCTL_TRIGGER, 0xC0244771)

    def test_kernel_bool_matches_kernel_semantics(self):
        cases = (
            ("irqchip.gicv3_pseudo_nmi=1", True),
            ("irqchip.gicv3_pseudo_nmi=yup", True),
            ("irqchip.gicv3-pseudo-nmi=on", True),
            ("irqchip.gicv3_pseudo_nmi", False),
            ("irqchip.gicv3_pseudo_nmi=0", False),
            ("irqchip.gicv3_pseudo_nmi=0 irqchip.gicv3_pseudo_nmi=y", True),
            ("irqchip.gicv3_pseudo_nmi=1 -- irqchip.gicv3_pseudo_nmi=0", True),
            ("-- irqchip.gicv3_pseudo_nmi=1", False),
        )
        for cmdline, expected in cases:
            with self.subTest(cmdline=cmdline):
                self.assertEqual(
                    nmi_perf._kernel_bool_param(
                        cmdline, "irqchip.gicv3_pseudo_nmi"
                    ),
                    expected,
                )

    def test_pseudo_nmi_requires_config_cmdline_and_boot_evidence(self):
        cases = (
            (VALID_CONFIG, "irqchip.gicv3_pseudo_nmi=1", PSEUDO_LOG, ""),
            (VALID_CONFIG.replace("CONFIG_ARM64_PSEUDO_NMI=y\n", ""),
             "irqchip.gicv3_pseudo_nmi=1", PSEUDO_LOG, "CONFIG_ARM64_PSEUDO_NMI"),
            (VALID_CONFIG, "", PSEUDO_LOG, "not enabled"),
            (VALID_CONFIG, "irqchip.gicv3_pseudo_nmi=1", "", "not observed"),
        )
        for config, cmdline, log, reason in cases:
            with self.subTest(reason=reason):
                self.assertIn(
                    reason, nmi_perf._pseudo_nmi_reason(config, cmdline, log)
                )

    def test_duration_is_bounded_and_has_a_reliable_floor(self):
        self.assertEqual(nmi_perf._trigger_duration_ms(1000), 1000)
        self.assertEqual(nmi_perf._trigger_duration_ms(100), 100)
        with self.assertRaisesRegex(ValueError, "at least 100ms"):
            nmi_perf._trigger_duration_ms(99)

    def test_evaluate_accepts_only_joint_nmi_irqoff_module_evidence(self):
        outcome, reason = nmi_perf._evaluate_trigger(valid_trigger(), 1, 1000)
        self.assertEqual(outcome, "pass")
        self.assertIn("module_irqoff_nmi=2", reason)

    def test_evaluate_rejects_false_pass_and_inconsistent_results(self):
        cases = (
            ({"overflows": 0, "nmi_overflows": 0, "irqoff_overflows": 0,
              "irqoff_nmi_overflows": 0, "module_irqoff_nmi_overflows": 0},
             "No PMU overflow"),
            ({"nmi_overflows": 0, "irqoff_overflows": 0,
              "irqoff_nmi_overflows": 0, "module_irqoff_nmi_overflows": 0},
             "Not every"),
            ({"nmi_overflows": 9}, "Not every"),
            ({"irqoff_nmi_overflows": 7}, "Inconsistent"),
            ({"irqoff_overflows": 0, "irqoff_nmi_overflows": 0,
              "module_irqoff_nmi_overflows": 0}, "No PMU overflow interrupted"),
            ({"module_irqoff_nmi_overflows": 0}, "inside the probe"),
            ({"module_irqoff_nmi_overflows": 9}, "Invalid probe-module"),
            ({"irqoff_overflows": 11, "irqoff_nmi_overflows": 11},
             "Inconsistent"),
            ({"cpu": 2}, "requested CPU"),
            ({"duration_ms": 100}, "requested CPU"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                outcome, actual_reason = nmi_perf._evaluate_trigger(
                    valid_trigger(**changes), 1, 1000
                )
                self.assertEqual(outcome, "fail")
                self.assertIn(reason, actual_reason)

    def test_c_lifecycle_and_joint_context_accounting(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src/modules/arm64/vera_nmi_perf_probe/vera_nmi_perf_probe.c"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("cpu_hotplug_disable();"), 1)
        self.assertEqual(source.count("cpu_hotplug_enable();"), 1)
        self.assertEqual(source.count("local_irq_save(flags);"), 1)
        self.assertEqual(source.count("local_irq_restore(flags);"), 1)
        self.assertEqual(source.count("perf_event_release_kernel(request->event)"), 1)
        self.assertIn("bool nmi = in_nmi();", source)
        self.assertIn("bool irqoff = !interrupts_enabled(regs);", source)
        self.assertIn("within_module(pc, THIS_MODULE)", source)
        self.assertIn("read_sysreg(cntvct_el0)", source)
        self.assertIn("trigger.armed != 1", source)
        self.assertIn("!trigger.duration_ms", source)
        self.assertIn("trigger.cpu >= nr_cpu_ids", source)


class TestNmiPerfRuntime(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "manifest.json"
        self.serial_path = self.root / "serial"
        self.cpu_root = self.root / "cpus"
        self.proc_root = self.root / "proc"
        self.module_dir = self.root / "module"
        self.headers_root = self.root / "headers"
        self.module_state = self.root / "loaded"
        self.module_dir.mkdir()
        (self.headers_root / "release" / "build").mkdir(parents=True)
        self.cpu_root.mkdir()
        (self.cpu_root / "online").write_text("0-1\n", encoding="utf-8")
        (self.proc_root / "sys" / "kernel").mkdir(parents=True)
        for name, value in (("nmi_watchdog", "0"), ("hardlockup_panic", "0")):
            (self.proc_root / "sys" / "kernel" / name).write_text(
                value + "\n", encoding="utf-8"
            )
        self.write_manifest()
        self.serial_path.write_text("board-1\n", encoding="utf-8")
        self.positive_changes = {}
        self.positive_error = None

        self._start_patch(
            f"{MODULE_PATH}._read_kernel_config_text", return_value=VALID_CONFIG
        )
        self._start_patch(
            f"{MODULE_PATH}._read_cmdline",
            return_value="irqchip.gicv3_pseudo_nmi=1",
        )
        self._start_patch(f"{MODULE_PATH}.glt_utils.is_cli_available", return_value=True)
        self._start_patch(f"{MODULE_PATH}.glt_utils.sign_kernel_module", return_value=True)
        self._start_patch(
            f"{MODULE_PATH}.glt_utils.capture_kernel_log_checkpoint",
            side_effect=["cp1", "cp2"],
        )
        self._start_patch(
            f"{MODULE_PATH}.glt_utils.get_kernel_log",
            side_effect=[PSEUDO_LOG, "", ""],
        )
        self._start_patch(
            f"{MODULE_PATH}.glt_utils.chdir_managed",
            side_effect=lambda _path: unittest.mock.MagicMock(),
        )
        self._start_patch(f"{MODULE_PATH}.os.open", return_value=99)
        self._start_patch(f"{MODULE_PATH}.os.close", return_value=0)
        self._start_patch(f"{MODULE_PATH}.os.sched_getaffinity", return_value={0, 1})
        self.mock_run = self._start_patch(f"{MODULE_PATH}.glt_utils.run_command")
        self.mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
        self.mock_ioctl = self._start_patch(f"{MODULE_PATH}.fcntl.ioctl")
        self.mock_ioctl.side_effect = self.fake_ioctl

    def _start_patch(self, target, **kwargs):
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def write_manifest(self, **changes):
        manifest = {
            "dut_id": "board-1",
            "recovery": "ready",
            "backend": "pmu_pseudo_nmi",
            "cpu": 1,
            "max_irqoff_ms": 1000,
        }
        manifest.update(changes)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_ioctl(self, _fd, _request, trigger, _mutate):
        if not trigger.armed or not trigger.duration_ms or trigger.cpu == 0xFFFFFFFF:
            raise OSError(errno.EINVAL, "invalid")
        self.assertEqual(trigger.cpu, 1)
        self.assertEqual(trigger.duration_ms, 1000)
        if self.positive_error is not None:
            raise self.positive_error
        result = valid_trigger(**self.positive_changes)
        for name in (
            "overflows", "nmi_overflows", "irqoff_overflows",
            "irqoff_nmi_overflows", "module_irqoff_nmi_overflows",
        ):
            setattr(trigger, name, getattr(result, name))
        return 0

    def run_runtime(self):
        return nmi_perf._run_nmi_perf_probe(
            self.mock_logger,
            manifest_path=self.manifest_path,
            serial_path=self.serial_path,
            cpu_root=self.cpu_root,
            module_dir=self.module_dir,
            module_state_path=self.module_state,
            headers_root=self.headers_root,
            kernel_release="release",
            device_path="/fake/device",
            proc_root=self.proc_root,
        )

    def test_positive_path_uses_joint_evidence_and_cleans_up(self):
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "pass")
        self.assertIn("module_irqoff_nmi=2", reason)
        self.assertEqual(self.mock_ioctl.call_count, 4)
        controls = [call.args[2] for call in self.mock_ioctl.call_args_list]
        self.assertEqual([trigger.armed for trigger in controls], [0, 1, 1, 1])
        self.assertEqual(controls[0].duration_ms, 1000)
        self.assertEqual(controls[1].duration_ms, 0)
        self.assertEqual(controls[2].cpu, 0xFFFFFFFF)
        self.assertEqual(controls[3].cpu, 1)
        self.assertEqual(controls[3].duration_ms, 1000)
        self.assertEqual(
            self.mock_run.call_args_list[-2].args[0],
            ["rmmod", "vera_nmi_perf_probe"],
        )

    def test_ordinary_irq_samples_cannot_pass(self):
        self.positive_changes = {
            "nmi_overflows": 0,
            "irqoff_overflows": 0,
            "irqoff_nmi_overflows": 0,
            "module_irqoff_nmi_overflows": 0,
        }
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("Not every", reason)

    def test_zero_overflows_fail(self):
        self.positive_changes = {
            "overflows": 0,
            "nmi_overflows": 0,
            "irqoff_overflows": 0,
            "irqoff_nmi_overflows": 0,
            "module_irqoff_nmi_overflows": 0,
        }
        self.assertEqual(self.run_runtime()[0], "fail")

    def test_missing_probe_pc_fails(self):
        self.positive_changes = {"module_irqoff_nmi_overflows": 0}
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("inside the probe", reason)

    def test_counter_pressure_is_inconclusive(self):
        self.positive_error = OSError(errno.ENOSPC, "no counter")
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("No PMU counter", reason)

    def test_unarmed_accepted_is_failure(self):
        self.mock_ioctl.return_value = 0
        self.mock_ioctl.side_effect = None
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("Unarmed", reason)
        self.assertEqual(self.mock_ioctl.call_count, 1)

    def test_preexisting_hardlockup_is_failure(self):
        self._start_patch(
            f"{MODULE_PATH}.glt_utils.get_kernel_log",
            side_effect=[PSEUDO_LOG, "Watchdog detected hard LOCKUP on cpu 1\n"],
        )
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("before", reason)

    def test_probe_hardlockup_is_failure(self):
        self._start_patch(
            f"{MODULE_PATH}.glt_utils.get_kernel_log",
            side_effect=[PSEUDO_LOG, "", "Watchdog detected hard LOCKUP on cpu 1\n"],
        )
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("caused a hardlockup", reason)

    def test_manifest_mismatch_skips_before_build(self):
        self.serial_path.write_text("other-board\n", encoding="utf-8")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_manifest_requires_explicit_irqoff_limit(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del manifest["max_irqoff_ms"]
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("explicit max_irqoff_ms", reason)
        self.mock_run.assert_not_called()

    def test_recovery_not_ready_skips(self):
        self.write_manifest(recovery="not-ready")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_sdei_manifest_cannot_select_perf_sampling(self):
        self.write_manifest(backend="sdei")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_watchdog_enabled_skips_to_isolate_counter_ownership(self):
        (self.proc_root / "sys" / "kernel" / "nmi_watchdog").write_text(
            "1\n", encoding="utf-8"
        )
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("nmi_watchdog=0", reason)
        self.mock_run.assert_not_called()

    def test_panic_enabled_skips(self):
        (self.proc_root / "sys" / "kernel" / "hardlockup_panic").write_text(
            "1\n", encoding="utf-8"
        )
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_duration_limit_below_floor_skips(self):
        self.write_manifest(max_irqoff_ms=99)
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_cpu_not_in_affinity_skips(self):
        self._start_patch(f"{MODULE_PATH}.os.sched_getaffinity", return_value={0})
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_existing_module_skips_before_build(self):
        self.module_state.mkdir()
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_unknown_module_after_insmod_failure_is_not_unloaded(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "insmod":
                self.module_state.mkdir()
                raise RuntimeError("load timeout")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.mock_run.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "fail")
        commands = [call.args[0] for call in self.mock_run.call_args_list]
        self.assertNotIn(["rmmod", "vera_nmi_perf_probe"], commands)

    def test_rmmod_failure_blocks_pass(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "rmmod":
                raise RuntimeError("unload failed")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.mock_run.side_effect = fake_run
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("cleanup", reason)

    def test_checkpoint_and_cleanup_failures_are_both_reported(self):
        self._start_patch(
            f"{MODULE_PATH}.glt_utils.capture_kernel_log_checkpoint",
            side_effect=RuntimeError("checkpoint failed"),
        )

        def fake_run(command, *_args, **_kwargs):
            if command[0] == "rmmod":
                raise RuntimeError("unload failed")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.mock_run.side_effect = fake_run
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("checkpoint failed", reason)
        self.assertIn("unload failed", reason)


class TestNmiPerfNativeResult(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.mock_nv_dev = self._start_patch(f"{MODULE_PATH}.NvDev")
        self.mock_nv_dev.return_value.is_cpu_type.return_value = True
        self._start_patch(f"{MODULE_PATH}.platform.machine", return_value="aarch64")
        self.mock_runner = self._start_patch(
            f"{MODULE_PATH}._run_nmi_perf_probe", return_value=("pass", "passed")
        )

    def _start_patch(self, target, **kwargs):
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_non_vera_skips_before_probe(self):
        self.mock_nv_dev.return_value.is_cpu_type.return_value = False
        with self.assertRaises(GLTCore.GLTTestSkip):
            nmi_perf.test_vera_nmi_perf_sampling(None, self.mock_logger)
        self.mock_runner.assert_not_called()

    def test_non_arm64_skips_before_probe(self):
        self._start_patch(f"{MODULE_PATH}.platform.machine", return_value="x86_64")
        with self.assertRaises(GLTCore.GLTTestSkip):
            nmi_perf.test_vera_nmi_perf_sampling(None, self.mock_logger)
        self.mock_runner.assert_not_called()

    def test_result_mapping(self):
        nmi_perf.test_vera_nmi_perf_sampling(None, self.mock_logger)
        for outcome, exception in (
            ("fail", GLTCore.GLTTestFail),
            ("skip", GLTCore.GLTTestSkip),
        ):
            with self.subTest(outcome=outcome):
                self.mock_runner.return_value = (outcome, "reason")
                with self.assertRaises(exception):
                    nmi_perf.test_vera_nmi_perf_sampling(None, self.mock_logger)


if __name__ == "__main__":
    unittest.main()
