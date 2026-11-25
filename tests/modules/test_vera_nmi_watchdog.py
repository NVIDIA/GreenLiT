"""Offline tests for bounded Vera NMI hardlockup validation."""
# pylint: disable=protected-access,too-many-public-methods

import errno
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_vera_nmi_watchdog as nmi

MODULE_PATH = "src.modules.arm64.glt_test_vera_nmi_watchdog"
SDEI_LOG = "SDEI Watchdog registered successfully\n"
PMU_LOG = (
    "Pseudo-NMIs enabled using GIC CPU interface synchronisation\n"
    "NMI watchdog: Enabled. Permanently consumes one hw-PMU counter.\n"
)
POSITIVE_LOG = (
    "Watchdog detected hard LOCKUP on cpu 1\n"
    "Modules linked in:\n"
    "Call trace:\n"
    "glt_nmi_spin_thread+0x10/0x40 [vera_nmi_watchdog]\n"
)
OTHER_CPU_LOG = "Watchdog detected hard LOCKUP on cpu 2\n"
VALID_CONFIG = (
    "CONFIG_MODULES=y\nCONFIG_MODULE_UNLOAD=y\n"
    "CONFIG_HARDLOCKUP_DETECTOR=y\nCONFIG_SDEI_WATCHDOG=y\n"
)


class TestNmiHelpers(unittest.TestCase):
    def test_helper_must_be_in_call_trace_not_only_comm(self):
        self.assertTrue(nmi._has_helper_callframe({"text": POSITIVE_LOG}))
        self.assertFalse(nmi._has_helper_callframe({"text": "Comm: glt_nmi_spin_thread\n"}))

    def test_await_rejects_multiple_blocks_and_polls_until_complete(self):
        with unittest.mock.patch.object(
            nmi.glt_utils, "get_kernel_log", return_value=OTHER_CPU_LOG + POSITIVE_LOG
        ), self.assertRaisesRegex(ValueError, "additional hardlockup"):
            nmi._await_hardlockup("checkpoint", 1)
        log_calls = {"count": 0}
        def delayed_positive_log(*_args):
            log_calls["count"] += 1
            return "" if log_calls["count"] == 1 else POSITIVE_LOG
        with unittest.mock.patch.object(
            nmi.glt_utils, "get_kernel_log", side_effect=delayed_positive_log
        ), unittest.mock.patch.object(nmi.time, "sleep", return_value=None):
            block = nmi._await_hardlockup("checkpoint", 1)
        self.assertEqual(block["cpu"], 1)

    def test_late_additional_block_after_valid_candidate_is_rejected(self):
        with unittest.mock.patch.object(
            nmi.glt_utils, "get_kernel_log",
            side_effect=[POSITIVE_LOG, OTHER_CPU_LOG + POSITIVE_LOG],
        ), unittest.mock.patch.object(nmi.time, "monotonic", side_effect=[0.0, 0.3, 0.4]):
            with self.assertRaisesRegex(ValueError, "additional hardlockup"):
                nmi._await_hardlockup("checkpoint", 1)

    def test_cpu_hotplug_disable_enable_are_balanced(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src/modules/arm64/vera_nmi_watchdog/vera_nmi_watchdog.c"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("cpu_hotplug_disable();"), 1)
        self.assertEqual(source.count("cpu_hotplug_enable();"), 3)

    def test_hardlockup_blocks_require_cpu_and_helper_in_same_block(self):
        log = OTHER_CPU_LOG + "unrelated frame\n" + POSITIVE_LOG
        blocks = nmi._hardlockup_blocks(log)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["cpu"], 2)
        self.assertEqual(blocks[1]["cpu"], 1)
        self.assertIn("glt_nmi_spin_thread", blocks[1]["text"])

    def test_kernel_bool_matches_kernel_semantics(self):
        cases = (
            ("irqchip.gicv3_pseudo_nmi=1", True),
            ("irqchip.gicv3_pseudo_nmi=yup", True),
            ("irqchip.gicv3-pseudo-nmi=on", True),
            ("irqchip.gicv3_pseudo_nmi", False),
            ("irqchip.gicv3_pseudo_nmi=0", False),
            ("irqchip.gicv3_pseudo_nmi=1 -- irqchip.gicv3_pseudo_nmi=0", True),
            ("-- irqchip.gicv3_pseudo_nmi=1", False),
        )
        for cmdline, expected in cases:
            with self.subTest(cmdline=cmdline):
                self.assertEqual(nmi._kernel_bool_param(cmdline, "irqchip.gicv3_pseudo_nmi"), expected)

    def test_trigger_duration_uses_threshold_plus_two_and_limit(self):
        self.assertEqual(nmi._trigger_duration_ms(10, 25000), 22000)
        self.assertEqual(nmi._trigger_duration_ms(1, 4000), 4000)
        with self.assertRaises(ValueError):
            nmi._trigger_duration_ms(10, 1000)


class TestNmiRuntime(GLTTestBase):
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
        for name, value in (("nmi_watchdog", "1"), ("hardlockup_panic", "0"), ("watchdog_thresh", "10")):
            (self.proc_root / "sys" / "kernel" / name).write_text(value + "\n", encoding="utf-8")
        self.write_manifest()
        self.serial_path.write_text("board-1\n", encoding="utf-8")

        self._start_patch(f"{MODULE_PATH}._read_kernel_config_text", return_value=VALID_CONFIG)
        self._start_patch(f"{MODULE_PATH}._read_cmdline", return_value="")
        self._start_patch(f"{MODULE_PATH}.glt_utils.is_cli_available", return_value=True)
        self._start_patch(f"{MODULE_PATH}.glt_utils.sign_kernel_module", return_value=True)
        self._start_patch(f"{MODULE_PATH}.glt_utils.capture_kernel_log_checkpoint", side_effect=["cp1", "cp2"])
        self._start_patch(f"{MODULE_PATH}.glt_utils.get_kernel_log", side_effect=[SDEI_LOG, "", "", POSITIVE_LOG, POSITIVE_LOG])
        self._start_patch(f"{MODULE_PATH}.glt_utils.chdir_managed", side_effect=lambda path: unittest.mock.MagicMock())
        self._start_patch(f"{MODULE_PATH}.os.open", return_value=99)
        self._start_patch(f"{MODULE_PATH}.os.close", return_value=0)
        self._start_patch(f"{MODULE_PATH}.os.sched_getaffinity", return_value={0, 1})
        self._start_patch(f"{MODULE_PATH}.os.sched_setaffinity", return_value=0)
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
            "dut_id": "board-1", "recovery": "ready", "backend": "sdei",
            "cpu": 1, "max_irqoff_ms": 25000,
        }
        manifest.update(changes)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_ioctl(self, _fd, _request, trigger):
        if trigger.duration_ms == 0:
            raise OSError(errno.EINVAL, "invalid")
        return 0

    def run_runtime(self):
        return nmi._run_nmi_probe(
            self.mock_logger, manifest_path=self.manifest_path,
            serial_path=self.serial_path, cpu_root=self.cpu_root,
            module_dir=self.module_dir, module_state_path=self.module_state,
            headers_root=self.headers_root, kernel_release="release",
            device_path="/fake/device", proc_root=self.proc_root,
        )

    def test_sdei_positive_path_cleans_up(self):
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "pass")
        self.assertEqual(self.mock_ioctl.call_args_list[0].args[2].duration_ms, 0)
        self.assertEqual(self.mock_ioctl.call_args_list[1].args[2].duration_ms, 22000)
        self.assertEqual(self.mock_run.call_args_list[-2].args[0][0], "rmmod")

    def test_pmu_positive_path(self):
        self.write_manifest(backend="pmu_pseudo_nmi")
        config = VALID_CONFIG.replace("CONFIG_SDEI_WATCHDOG=y", "CONFIG_ARM64_PSEUDO_NMI=y\nCONFIG_HARDLOCKUP_DETECTOR_PERF=y")
        self._start_patch(f"{MODULE_PATH}._read_kernel_config_text", return_value=config)
        self._start_patch(f"{MODULE_PATH}._read_cmdline", return_value="irqchip.gicv3_pseudo_nmi=1")
        self._start_patch(f"{MODULE_PATH}.glt_utils.get_kernel_log", side_effect=[PMU_LOG, "", POSITIVE_LOG, POSITIVE_LOG])
        self.assertEqual(self.run_runtime()[0], "pass")

    def test_disable_sdei_allows_pmu_backend(self):
        self.write_manifest(backend="pmu_pseudo_nmi")
        config = VALID_CONFIG.replace("CONFIG_SDEI_WATCHDOG=y", "CONFIG_ARM64_PSEUDO_NMI=y\nCONFIG_HARDLOCKUP_DETECTOR_PERF=y")
        self._start_patch(f"{MODULE_PATH}._read_kernel_config_text", return_value=config)
        self._start_patch(f"{MODULE_PATH}._read_cmdline", return_value="disable_sdei_nmi_watchdog irqchip.gicv3_pseudo_nmi=1")
        self._start_patch(f"{MODULE_PATH}.glt_utils.get_kernel_log", side_effect=[PMU_LOG, "", POSITIVE_LOG, POSITIVE_LOG])
        self.assertEqual(self.run_runtime()[0], "pass")

    def test_manifest_mismatch_skips_before_build(self):
        self.serial_path.write_text("other-board\n", encoding="utf-8")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_recovery_missing_skips(self):
        self.write_manifest(recovery="not-ready")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_sdei_disabled_skips(self):
        self._start_patch(f"{MODULE_PATH}._read_cmdline", return_value="disable_sdei_nmi_watchdog=0")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_sdei_active_conflicts_with_pmu_backend(self):
        self.write_manifest(backend="pmu_pseudo_nmi")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_panic_enabled_skips(self):
        (self.proc_root / "sys" / "kernel" / "hardlockup_panic").write_text("1\n", encoding="utf-8")
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_cpu_not_in_affinity_skips(self):
        self._start_patch(f"{MODULE_PATH}.os.sched_getaffinity", return_value={0})
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_invalid_watchdog_threshold_is_a_controlled_skip(self):
        (self.proc_root / "sys" / "kernel" / "watchdog_thresh").write_text("invalid\n", encoding="utf-8")
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("preconditions", reason)
        self.mock_run.assert_not_called()

    def test_duration_limit_skips(self):
        self.write_manifest(max_irqoff_ms=1000)
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_unarmed_accepted_is_failure(self):
        self.mock_ioctl.return_value = 0
        self.mock_ioctl.side_effect = None
        self.assertEqual(self.run_runtime()[0], "fail")

    def test_preexisting_hardlockup_is_failure(self):
        self._start_patch(f"{MODULE_PATH}.glt_utils.get_kernel_log", side_effect=[SDEI_LOG, POSITIVE_LOG, POSITIVE_LOG])
        self.assertEqual(self.run_runtime()[0], "fail")

    def test_additional_hardlockup_block_is_failure(self):
        self._start_patch(f"{MODULE_PATH}.glt_utils.get_kernel_log", side_effect=[SDEI_LOG, "", OTHER_CPU_LOG + POSITIVE_LOG])
        self.assertEqual(self.run_runtime()[0], "fail")

    def test_wrong_cpu_or_missing_helper_is_failure(self):
        self._start_patch(f"{MODULE_PATH}.glt_utils.get_kernel_log", side_effect=[SDEI_LOG, "", OTHER_CPU_LOG])
        self.assertEqual(self.run_runtime()[0], "fail")

    def test_unknown_module_after_insmod_failure_is_not_unloaded(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "insmod":
                self.module_state.mkdir()
                raise RuntimeError("load timeout")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "fail")
        self.assertNotIn(["rmmod", "vera_nmi_watchdog"], [call.args[0] for call in self.mock_run.call_args_list])

    def test_rmmod_failure_blocks_pass(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "rmmod":
                raise RuntimeError("unload failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "fail")


class TestNmiNativeResult(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.mock_nv_dev = self._start_patch(f"{MODULE_PATH}.NvDev")
        self.mock_nv_dev.return_value.is_cpu_type.return_value = True
        self.mock_runner = self._start_patch(f"{MODULE_PATH}._run_nmi_probe", return_value=("pass", "passed"))

    def _start_patch(self, target, **kwargs):
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_non_vera_skips_before_probe(self):
        self.mock_nv_dev.return_value.is_cpu_type.return_value = False
        with self.assertRaises(GLTCore.GLTTestSkip):
            nmi.test_vera_nmi_hardlockup(None, self.mock_logger)
        self.mock_runner.assert_not_called()

    def test_result_mapping(self):
        nmi.test_vera_nmi_hardlockup(None, self.mock_logger)
        for outcome, exception in (("fail", GLTCore.GLTTestFail), ("skip", GLTCore.GLTTestSkip)):
            with self.subTest(outcome=outcome):
                self.mock_runner.return_value = (outcome, "reason")
                with self.assertRaises(exception):
                    nmi.test_vera_nmi_hardlockup(None, self.mock_logger)


if __name__ == "__main__":
    unittest.main()
