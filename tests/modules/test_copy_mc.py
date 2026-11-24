"""Offline tests for Vera copy_mc memcpy KUnit integration."""
# pylint: disable=protected-access,too-many-public-methods

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_copy_mc as copy_mc

MODULE_PATH = "src.modules.arm64.glt_test_copy_mc"
VALID_CONFIG = """CONFIG_ARCH_HAS_COPY_MC=y
CONFIG_KUNIT=y
CONFIG_KUNIT_DEBUGFS=y
CONFIG_MEMCPY_KUNIT_TEST={memcpy_mode}
"""


def make_results(cases=None, suite_status="ok"):
    if cases is None:
        cases = [(name, "ok", "") for name in copy_mc.REQUIRED_KUNIT_CASES]
    lines = [
        "KTAP version 1",
        "1..1",
        "    KTAP version 1",
        "    # Subtest: memcpy",
        f"    1..{len(cases)}",
    ]
    for number, (name, status, directive) in enumerate(cases, start=1):
        lines.append(f"    {status} {number} {name}{directive}")
    lines.append(f"{suite_status} 1 memcpy")
    return "\n".join(lines) + "\n"


class TestCopyMcKunitParser(unittest.TestCase):
    def test_all_required_cases_pass(self):
        failed, skipped, missing = copy_mc._parse_memcpy_results(make_results())
        self.assertEqual((failed, skipped, missing), ([], [], []))

    def test_failures_include_case_and_suite(self):
        cases = [(name, "ok", "") for name in copy_mc.REQUIRED_KUNIT_CASES]
        cases[7] = ("copy_mc_page_test", "not ok", "")
        failed, skipped, missing = copy_mc._parse_memcpy_results(
            make_results(cases, suite_status="not ok")
        )
        self.assertEqual(failed, ["copy_mc_page_test", "memcpy (suite)"])
        self.assertEqual((skipped, missing), ([], []))

    def test_skips_and_missing_cases_are_reported(self):
        cases = [(name, "ok", "") for name in copy_mc.REQUIRED_KUNIT_CASES[:7]]
        cases[2] = ("memcpy_large_test", "ok", " # SKIP slow disabled")
        failed, skipped, missing = copy_mc._parse_memcpy_results(make_results(cases))
        self.assertEqual(failed, [])
        self.assertEqual(skipped, ["memcpy_large_test"])
        self.assertEqual(
            missing,
            ["copy_mc_page_test", "memcpy_mc_test", "memcpy_mc_large_test"],
        )

    def test_comments_do_not_create_test_points(self):
        text = make_results().replace(
            "    ok 1 memset_test",
            "    ok 1 memset_test\n# diagnostic: not ok 99 fake_case",
        )
        failed, skipped, missing = copy_mc._parse_memcpy_results(text)
        self.assertEqual((failed, skipped, missing), ([], [], []))

    def test_invalid_results_are_rejected(self):
        duplicate_names = make_results().replace("ok 2 memcpy_test", "ok 2 memset_test")
        nonconsecutive = make_results().replace("ok 2 memcpy_test", "ok 3 memcpy_test")
        count_mismatch = make_results().replace("    1..10", "    1..9")
        no_final = make_results().replace("ok 1 memcpy\n", "")
        wrong_suite = make_results().replace("# Subtest: memcpy", "# Subtest: other")
        duplicate_marker = make_results() + "    # Subtest: memcpy\n"
        no_outer_header = make_results().replace("KTAP version 1\n", "", 1)
        no_outer_plan = make_results().replace("1..1\n", "", 1)
        no_nested_header = make_results().replace("    KTAP version 1\n", "", 1)
        trailing_content = make_results() + "unexpected trailing content\n"
        bail_out = make_results().replace("ok 1 memcpy\n", "Bail out! injected abort\nok 1 memcpy\n")
        unexpected_outer = make_results().replace(
            "ok 1 memcpy\n", "not ok 2 unexpected_outer_suite\nok 1 memcpy\n"
        )
        malformed_case = make_results().replace("ok 1 memcpy\n", "    not ok\nok 1 memcpy\n")
        extra_plan = make_results().replace("    1..10\n", "    1..10\n    1..10\n")
        outer_plan = make_results().replace("    1..10", "1..10")
        wrong_case_indent = make_results().replace("    ok 1 memset_test", "        ok 1 memset_test")
        for text in (
            "",
            duplicate_names,
            nonconsecutive,
            count_mismatch,
            no_final,
            wrong_suite,
            duplicate_marker,
            no_outer_header,
            no_outer_plan,
            no_nested_header,
            trailing_content,
            bail_out,
            unexpected_outer,
            malformed_case,
            extra_plan,
            outer_plan,
            wrong_case_indent,
            "    # Subtest: memcpy\n",
        ):
            with self.subTest(text=text[:60]):
                with self.assertRaises(ValueError):
                    copy_mc._parse_memcpy_results(text)


class TestCopyMcKunitRuntime(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.kconfig_path = self.root / "config"
        self.results_path = self.root / "results"
        self.module_state_path = self.root / "module"
        self.write_config()
        self.results_path.write_text(make_results(), encoding="utf-8")
        self.mock_run = self.mock.patch(f"{MODULE_PATH}.glt_utils.run_command")
        self.run_command = self.mock_run.start()
        self.addCleanup(self.mock_run.stop)
        self.run_command.return_value = subprocess.CompletedProcess(
            args=["mock"], returncode=0, stdout="", stderr=""
        )

    def write_config(self, memcpy_mode="y", overrides=None):
        text = VALID_CONFIG.format(memcpy_mode=memcpy_mode)
        if overrides:
            for key, value in overrides.items():
                text = text.replace(key, value)
        self.kconfig_path.write_text(text, encoding="utf-8")

    def run_runtime(self):
        return copy_mc._run_memcpy_kunit(
            kconfig_path=self.kconfig_path,
            results_path=self.results_path,
            module_state_path=self.module_state_path,
        )

    def test_builtin_results_pass_without_module_commands(self):
        self.assertEqual(self.run_runtime()[0], "pass")
        self.run_command.assert_not_called()

    def test_module_is_loaded_and_unloaded_by_test(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()

        def fake_run(command, _failure_message="", **_):
            if command[0] == "modprobe":
                self.results_path.write_text(make_results(), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.run_command.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "pass")
        self.assertEqual(
            [call.args[0][0] for call in self.run_command.call_args_list],
            ["modprobe", "rmmod"],
        )

    def test_preloaded_module_is_not_unloaded(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()
        self.module_state_path.mkdir()

        def fake_run(command, _failure_message="", **_):
            if command[0] == "modprobe":
                self.results_path.write_text(make_results(), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.run_command.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "pass")
        self.run_command.assert_called_once()
        self.assertEqual(self.run_command.call_args.args[0][0], "modprobe")

    def test_modprobe_failure_skips_without_unload(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()
        self.run_command.side_effect = RuntimeError("load failed")

        outcome, reason = self.run_runtime()

        self.assertEqual(outcome, "skip")
        self.assertIn("load failed", reason)
        self.run_command.assert_called_once()

    def test_uncertain_module_load_is_cleaned_up(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()

        def fake_run(command, _failure_message="", **_):
            if command[0] == "modprobe":
                self.module_state_path.mkdir()
                raise RuntimeError("load timed out")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.run_command.side_effect = fake_run
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertEqual(
            [call.args[0][0] for call in self.run_command.call_args_list],
            ["modprobe", "rmmod"],
        )

    def test_modprobe_failure_does_not_hide_kunit_failure(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()
        cases = [(name, "ok", "") for name in copy_mc.REQUIRED_KUNIT_CASES]
        cases[7] = ("copy_mc_page_test", "not ok", "")

        def fake_run(command, _failure_message="", **_):
            if command[0] == "modprobe":
                self.results_path.write_text(make_results(cases), encoding="utf-8")
                self.module_state_path.mkdir()
                raise RuntimeError("load timed out")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.run_command.side_effect = fake_run
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("copy_mc_page_test", reason)
        self.assertEqual(self.run_command.call_args_list[-1].args[0][0], "rmmod")

    def test_non_utf8_results_fail(self):
        self.results_path.write_bytes(b"\xff\xfe")

        outcome, reason = self.run_runtime()

        self.assertEqual(outcome, "fail")
        self.assertIn("not valid UTF-8", reason)

    def test_builtin_missing_results_skips(self):
        self.results_path.unlink()
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("built-in suite", reason)
        self.run_command.assert_not_called()

    def test_missing_module_tools_skip(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()
        with self.mock.patch(f"{MODULE_PATH}.glt_utils.is_cli_available", return_value=False):
            outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.run_command.assert_not_called()

    def test_invalid_results_fail_and_cleanup(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()

        def fake_run(command, _failure_message="", **_):
            if command[0] == "modprobe":
                self.results_path.write_text("not ktap", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.run_command.side_effect = fake_run
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("Invalid memcpy KUnit results", reason)
        self.assertEqual(self.run_command.call_args_list[-1].args[0][0], "rmmod")

    def test_missing_v14_cases_skip(self):
        cases = [(name, "ok", "") for name in copy_mc.REQUIRED_KUNIT_CASES[:7]]
        self.results_path.write_text(make_results(cases), encoding="utf-8")
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.assertIn("missing", reason)

    def test_cleanup_failure_blocks_pass(self):
        self.write_config(memcpy_mode="m")
        self.results_path.unlink()

        def fake_run(command, _failure_message="", **_):
            if command[0] == "modprobe":
                self.results_path.write_text(make_results(), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")
            raise RuntimeError("unload failed")

        self.run_command.side_effect = fake_run
        outcome, reason = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertIn("cleanup failed", reason)

    def test_missing_config_skips_before_commands(self):
        self.kconfig_path.write_text("CONFIG_KUNIT=y\n", encoding="utf-8")
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "skip")
        self.run_command.assert_not_called()


class TestCopyMcNativeResult(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.nv_dev = self.mock.patch(f"{MODULE_PATH}.NvDev")
        self.mock_nv_dev = self.nv_dev.start()
        self.addCleanup(self.nv_dev.stop)
        self.mock_nv_dev.return_value.is_cpu_type.return_value = True
        self.runner = self.mock.patch(f"{MODULE_PATH}._run_memcpy_kunit")
        self.mock_runner = self.runner.start()
        self.addCleanup(self.runner.stop)

    def test_non_vera_skips_before_runtime(self):
        self.mock_nv_dev.return_value.is_cpu_type.return_value = False
        with self.assertRaises(GLTCore.GLTTestSkip):
            copy_mc.test_copy_mc_memcpy_kunit(None, self.mock_logger)
        self.mock_runner.assert_not_called()

    def test_pass_is_logged(self):
        self.mock_runner.return_value = ("pass", "all passed")
        copy_mc.test_copy_mc_memcpy_kunit(None, self.mock_logger)
        self.mock_logger.info.assert_called_once_with("%s", "all passed")

    def test_failure_and_skip_are_mapped(self):
        for outcome, exception in (
            ("fail", GLTCore.GLTTestFail),
            ("skip", GLTCore.GLTTestSkip),
        ):
            with self.subTest(outcome=outcome):
                self.mock_runner.return_value = (outcome, "reason")
                with self.assertRaises(exception):
                    copy_mc.test_copy_mc_memcpy_kunit(None, self.mock_logger)


if __name__ == "__main__":
    unittest.main()
