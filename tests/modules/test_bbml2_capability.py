"""Offline tests for Vera BBML2 capability validation."""
# pylint: disable=protected-access,too-many-public-methods

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_bbml2 as bbml2

MODULE_PATH = "src.modules.arm64.glt_test_bbml2"
OLYMPUS_MIDR = 0x4E000100


def write_midr(root, cpu_id, value=OLYMPUS_MIDR):
    path = root / f"cpu{cpu_id}" / "regs" / "identification"
    path.mkdir(parents=True, exist_ok=True)
    (path / "midr_el1").write_text(hex(value) + "\n", encoding="utf-8")


class TestBbml2Helpers(unittest.TestCase):
    def test_present_midr_accepts_unsorted_all_olympus_topology(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "present").write_text("2,0-1\n", encoding="utf-8")
            for cpu_id in (0, 1, 2):
                write_midr(root, cpu_id)
            self.assertEqual(bbml2._non_olympus_present_cpus(root), [])

    def test_present_midr_reports_mixed_topology(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "present").write_text("0-2\n", encoding="utf-8")
            write_midr(root, 0)
            write_midr(root, 1, 0x41000000)
            write_midr(root, 2)
            self.assertEqual(bbml2._non_olympus_present_cpus(root), [1])

    def test_missing_or_empty_midr_data_fails_inspection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "present").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                bbml2._non_olympus_present_cpus(root)
            (root / "present").write_text("0\n", encoding="utf-8")
            with self.assertRaises(OSError):
                bbml2._non_olympus_present_cpus(root)

    def test_probe_log_requires_one_exact_result(self):
        compiled, enabled = bbml2._parse_probe_log(
            "glt_bbml2_capability: compiled=1 enabled=1\n"
        )
        self.assertEqual((compiled, enabled), (1, 1))
        for text in (
            "",
            "glt_bbml2_capability: compiled=0 enabled=0\n" * 2,
            "glt_bbml2_capability: compiled=x enabled=1\n",
            "glt_bbml2_capability: compiled=0 enabled=1\n",
            "glt_bbml2_capability: compiled=1 enabled=1\nglt_bbml2_capability: malformed\n",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    bbml2._parse_probe_log(text)

    def test_result_mapping_distinguishes_skip_fail_and_pass(self):
        self.assertEqual(bbml2._evaluate_probe_result(0, 0)[0], "skip")
        self.assertEqual(bbml2._evaluate_probe_result(1, 0)[0], "fail")
        self.assertEqual(bbml2._evaluate_probe_result(1, 1)[0], "pass")


class TestBbml2Runtime(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.cpu_root = self.root / "cpus"
        self.cpu_root.mkdir()
        (self.cpu_root / "present").write_text("0-1\n", encoding="utf-8")
        write_midr(self.cpu_root, 0)
        write_midr(self.cpu_root, 1)
        self.module_dir = self.root / "module"
        self.module_dir.mkdir()
        self.headers_root = self.root / "headers"
        (self.headers_root / "test-release" / "build").mkdir(parents=True)
        self.module_state = self.root / "loaded"

        self.mock_config = self._start_patch(
            f"{MODULE_PATH}._read_kernel_config_text",
            return_value="CONFIG_MODULES=y\nCONFIG_MODULE_UNLOAD=y\n",
        )
        self._start_patch(f"{MODULE_PATH}.glt_utils.is_cli_available", return_value=True)
        self._start_patch(f"{MODULE_PATH}.glt_utils.sign_kernel_module", return_value=True)
        self._start_patch(f"{MODULE_PATH}.glt_utils.capture_kernel_log_checkpoint", return_value="checkpoint")
        self.mock_get_log = self._start_patch(
            f"{MODULE_PATH}.glt_utils.get_kernel_log",
            return_value="glt_bbml2_capability: compiled=1 enabled=1\n",
        )
        self.mock_run = self._start_patch(f"{MODULE_PATH}.glt_utils.run_command")
        self.mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

    def _start_patch(self, target, **kwargs):
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def run_runtime(self):
        return bbml2._run_bbml2_probe(
            self.mock_logger,
            cpu_root=self.cpu_root,
            module_dir=self.module_dir,
            module_state_path=self.module_state,
            headers_root=self.headers_root,
            kernel_release="test-release",
        )

    def test_positive_capability_loads_and_cleans_up(self):
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "pass")
        commands = [call.args[0] for call in self.mock_run.call_args_list]
        self.assertEqual(commands, [["make"], ["insmod", "bbml2_capability_probe.ko"], ["rmmod", "bbml2_capability_probe"], ["make", "clean"]])

    def test_uncompiled_capability_skips(self):
        self.mock_get_log.return_value = "glt_bbml2_capability: compiled=0 enabled=0\n"
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("early capability", reason)

    def test_disabled_capability_fails(self):
        self.mock_get_log.return_value = "glt_bbml2_capability: compiled=1 enabled=0\n"
        self.assertEqual(self.run_runtime()[0], "fail")

    def test_non_unloadable_module_kernel_skips_before_build(self):
        self.mock_config.return_value = "CONFIG_MODULES=y\n# CONFIG_MODULE_UNLOAD is not set\n"
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("CONFIG_MODULE_UNLOAD=y", reason)
        self.mock_run.assert_not_called()

    def test_mixed_midr_skips_before_build(self):
        write_midr(self.cpu_root, 1, 0x41000000)
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.mock_run.assert_not_called()

    def test_preloaded_module_skips_without_build(self):
        self.module_state.mkdir()
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_make_failure_skips_and_still_cleans(self):
        def fake_run(command, *_args, **_kwargs):
            if command == ["make"]:
                raise RuntimeError("build failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertEqual(self.mock_run.call_args_list[-1].args[0], ["make", "clean"])

    def test_insmod_failure_without_module_does_not_unload(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "insmod":
                raise RuntimeError("load failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "skip")
        self.assertNotIn(["rmmod", "bbml2_capability_probe"], [call.args[0] for call in self.mock_run.call_args_list])

    def test_uncertain_insmod_fails_without_unloading_unknown_module(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "insmod":
                self.module_state.mkdir()
                raise RuntimeError("load timed out")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("ownership is unknown", reason)
        self.assertNotIn(["rmmod", "bbml2_capability_probe"], [call.args[0] for call in self.mock_run.call_args_list])

    def test_invalid_log_fails_and_cleans_up(self):
        self.mock_get_log.return_value = ""
        self.assertEqual(self.run_runtime()[0], "fail")
        self.assertEqual(self.mock_run.call_args_list[-1].args[0], ["make", "clean"])

    def test_rmmod_failure_blocks_pass(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "rmmod":
                raise RuntimeError("unload failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("cleanup failed", reason)

    def test_make_clean_failure_blocks_pass(self):
        def fake_run(command, *_args, **_kwargs):
            if command == ["make", "clean"]:
                raise RuntimeError("clean failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "fail")


class TestBbml2NativeResult(GLTTestBase):
    def setUp(self):
        super().setUp()
        self._patchers = []
        self.mock_nv_dev = self._start_patch(f"{MODULE_PATH}.NvDev")
        self.mock_nv_dev.return_value.is_cpu_type.return_value = True
        self._start_patch(f"{MODULE_PATH}.platform.machine", return_value="aarch64")
        self.mock_runner = self._start_patch(f"{MODULE_PATH}._run_bbml2_probe", return_value=("pass", "passed"))

    def _start_patch(self, target, **kwargs):
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_non_vera_skips_before_probe(self):
        self.mock_nv_dev.return_value.is_cpu_type.return_value = False
        with self.assertRaises(GLTCore.GLTTestSkip):
            bbml2.test_vera_bbml2_capability(None, self.mock_logger)
        self.mock_runner.assert_not_called()

    def test_native_result_mapping(self):
        bbml2.test_vera_bbml2_capability(None, self.mock_logger)
        for outcome, exception in (("fail", GLTCore.GLTTestFail), ("skip", GLTCore.GLTTestSkip)):
            with self.subTest(outcome=outcome):
                self.mock_runner.return_value = (outcome, "reason")
                with self.assertRaises(exception):
                    bbml2.test_vera_bbml2_capability(None, self.mock_logger)


if __name__ == "__main__":
    unittest.main()
