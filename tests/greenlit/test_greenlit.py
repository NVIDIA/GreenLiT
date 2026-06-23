"""Unit tests for greenlit/greenlit.py helper functions."""

import argparse
import io
import inspect
import json
import os
import tempfile
from unittest import mock

from tests.test_base import GLTTestBase
from src.common import glt_core as GLTCore
from src.greenlit import greenlit


def _args(categories, test_cases=None, list_tests=False, resume=None):
    values = {category: False for category in categories}
    values.update({
        "test_cases": test_cases,
        "list": list_tests,
        "resume": resume,
        "verbose": False,
    })
    return argparse.Namespace(**values)


class TestGreenlitHelpers(GLTTestBase):  # pylint: disable=too-many-public-methods
    """Test pure helper functions used by glt-test-suite."""

    def test_parse_test_selector_accepts_category_test_name(self):
        """Valid selectors are returned unchanged."""
        self.assertEqual(
            greenlit.parse_test_selector("cxl:test_cxl_link"),
            "cxl:test_cxl_link")

    def test_parse_test_selector_rejects_malformed_values(self):
        """Malformed selectors include the offending value in the error."""
        for value in ["test_only", "cxl:", ":test_name", "cxl:test:extra"]:
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError) as ctx:
                    greenlit.parse_test_selector(value)
                self.assertIn(value, str(ctx.exception))
                self.assertIn(greenlit.TEST_ID_HELP, str(ctx.exception))

    def test_get_target_categories_uses_test_case_categories(self):
        """Selected test case IDs drive category discovery."""
        supported = ["cxl", "pci", "tpm"]
        args = _args(
            supported,
            test_cases=["cxl:test_a", "pci:test_b"])

        self.assertEqual(
            greenlit.get_target_categories(args, supported),
            ["cxl", "pci"])

    def test_get_target_categories_ignores_category_flags_with_test_case(self):
        """Conflict validation handles category flags before target resolution."""
        supported = ["cxl", "pci", "tpm"]
        args = _args(supported, test_cases=["cxl:test_a"])
        args.tpm = True

        self.assertEqual(
            greenlit.get_target_categories(args, supported),
            ["cxl"])

    def test_apply_glt_configs_skips_non_runnable_tests(self):
        """Config entries outside scheduler.test_list are ignored."""

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_prereq():
            return "prereq"

        scheduler = GLTCore.GLTScheduler([test_prereq])
        scheduler.sort()
        scheduler.test_categories = ["arm64", "hwtracing"]
        yaml_data = [
            {
                "hwtracing:test_unrelated": {
                    "conditional-skip": [{
                        "condition": {"name": "skip unrelated", "script": "true"}
                    }]
                },
            },
            {
                "hwtracing:test_prereq": {
                    "conditional-skip": [{
                        "condition": {"name": "skip prereq", "script": "true"}
                    }]
                },
            },
        ]

        with mock.patch.object(
                greenlit.GLTConf, "load_glt_yaml_files",
                return_value=yaml_data), \
                mock.patch.object(
                    greenlit.GLTConf, "cond_to_func",
                    return_value=lambda: True):
            greenlit.apply_glt_configs(scheduler, self.mock_logger)

        self.assertEqual(
            len(test_prereq.__wrapped__.attr.conditional_skip.any_true), 1)

    def test_apply_glt_configs_later_yaml_overwrites_test_definition(self):
        """Later YAML files replace earlier definitions for the same test."""

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_prereq():
            return "prereq"

        scheduler = GLTCore.GLTScheduler([test_prereq])
        scheduler.sort()
        yaml_data = [
            {
                "hwtracing:test_prereq": {
                    "conditional-skip": [{
                        "condition": {"name": "old skip", "script": "true"}
                    }]
                },
            },
            {
                "hwtracing:test_prereq": {
                    "allow-failure": [{
                        "condition": {"name": "new allow", "script": "true"}
                    }]
                },
            },
        ]

        with mock.patch.object(
                greenlit.GLTConf, "load_glt_yaml_files",
                return_value=yaml_data), \
                mock.patch.object(
                    greenlit.GLTConf, "cond_to_func",
                    return_value=lambda: True):
            greenlit.apply_glt_configs(scheduler, self.mock_logger)

        self.assertEqual(
            len(test_prereq.__wrapped__.attr.conditional_skip.any_true), 0)
        self.assertEqual(
            len(test_prereq.__wrapped__.attr.allow_failure.any_true), 1)

    def test_validate_test_case_args_rejects_list(self):
        """--list cannot be combined with parsed --test-case selectors."""
        args = _args(["cxl"], test_cases=["cxl:test_a"], list_tests=True)

        valid, error = greenlit.validate_test_case_args_conflicts(args)

        self.assertFalse(valid)
        self.assertIn("--test-case cannot be combined with --list", error)

    def test_validate_test_case_args_rejects_deferred_list_selector(self):
        """The pre-parser catches --list --resume --test-case before resume."""
        args = _args(["cxl"], list_tests=True)

        valid, error = greenlit.validate_test_case_args_conflicts(
            args, unknown_args=["--test-case", "cxl:test_a"])
        self.assertFalse(valid)
        self.assertIn("--list", error)

        valid, error = greenlit.validate_test_case_args_conflicts(
            args, unknown_args=["--test-case=cxl:test_a"])
        self.assertFalse(valid)
        self.assertIn("--list", error)

    def test_validate_test_case_args_rejects_resume(self):
        """--resume cannot be combined with --test-case."""
        args = _args(["cxl"], test_cases=["cxl:test_a"], resume=-1)

        valid, error = greenlit.validate_test_case_args_conflicts(args)

        self.assertFalse(valid)
        self.assertIn("--resume", error)

    def test_validate_test_case_args_rejects_category_flags(self):
        """Category flags cannot be combined with --test-case."""
        supported = ["cxl", "pci", "tpm"]
        args = _args(supported, test_cases=["cxl:test_a"])
        args.tpm = True

        valid, error = greenlit.validate_test_case_args_conflicts(
            args, supported)

        self.assertFalse(valid)
        self.assertIn("--test-case cannot be combined with category flags", error)
        self.assertIn("--tpm", error)

    def test_validate_test_case_args_allows_plan_only(self):
        """Plan flags are compatible with --test-case."""
        args = _args(["cxl"], test_cases=["cxl:test_a"])

        valid, error = greenlit.validate_test_case_args_conflicts(
            args, ["cxl"])

        self.assertTrue(valid, error)

    def test_discover_target_test_cases_rejects_unknown_test(self):
        """Selected test IDs must exist."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_known")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_known():
            return "known"

        with mock.patch.object(
                greenlit, "test_case_discover",
                return_value=[test_known]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["cxl"], ["cxl:test_missing"],
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(test_objs, [])
        self.assertIn("Unknown test ID: cxl:test_missing", error)
        self.assertEqual(exit_code, greenlit.CLI_USAGE_ERROR)

    def test_discover_target_test_cases_rejects_missing_dependency(self):
        """Missing @GLTRequires targets are reported before scheduler mutation."""

        @GLTCore.GLTRequires.must_pass("cxl", "test_missing")
        @GLTCore.GLTTestAttr(cat="cxl", name="test_selected")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_selected():
            return "selected"

        with mock.patch.object(
                greenlit, "test_case_discover",
                return_value=[test_selected]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["cxl"], ["cxl:test_selected"],
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(test_objs, [])
        self.assertIn("cxl:test_missing", error)
        self.assertEqual(exit_code, greenlit.CLI_RUNTIME_ERROR)

    def test_discover_target_test_cases_rejects_plan_mismatch(self):
        """Selected tests must be part of the active plan."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_stress")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STRESS.value)
        def test_stress():
            return "stress"

        with mock.patch.object(
                greenlit, "test_case_discover",
                return_value=[test_stress]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["cxl"], ["cxl:test_stress"],
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(test_objs, [])
        self.assertIn("cxl:test_stress", error)
        self.assertIn("not selected by plan", error)
        self.assertEqual(exit_code, greenlit.CLI_USAGE_ERROR)

    def test_discover_target_test_cases_rejects_dependency_plan_mismatch(self):
        """Prerequisites must also match the active plan."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STRESS.value)
        def test_prereq():
            return "prereq"

        @GLTCore.GLTRequires.must_pass("cxl", "test_prereq")
        @GLTCore.GLTTestAttr(cat="cxl", name="test_selected")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_selected():
            return "selected"

        with mock.patch.object(
                greenlit, "test_case_discover",
                return_value=[test_selected, test_prereq]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["cxl"], ["cxl:test_selected"],
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(test_objs, [])
        self.assertIn("Required test ID cxl:test_prereq", error)
        self.assertIn("not selected by plan", error)
        self.assertEqual(exit_code, greenlit.CLI_RUNTIME_ERROR)

    def test_discover_target_test_cases_filters_normal_roots_by_plan(self):
        """Category discovery ignores out-of-plan tests that are not prerequisites."""

        @GLTCore.GLTTestAttr(cat="pci", name="test_sanity")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_sanity():
            return "sanity"

        @GLTCore.GLTTestAttr(cat="pci", name="test_stress")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STRESS.value)
        def test_stress():
            return "stress"

        with mock.patch.object(
                greenlit, "test_case_discover",
                return_value=[test_sanity, test_stress]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["pci"], None, {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(error, "")
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [f"{obj.__wrapped__.attr.category}:{obj.__wrapped__.attr.name}"
             for obj in test_objs],
            ["pci:test_sanity"])

    def test_discover_target_test_cases_rejects_empty_selection(self):
        """Discovery reports when plan/category selection finds no tests."""
        with mock.patch.object(greenlit, "test_case_discover",
                               return_value=[]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["cxl"], None, {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(test_objs, [])
        self.assertEqual(error, "")
        self.assertEqual(exit_code, 0)

    def test_discover_target_test_cases_uses_explicit_target_categories(self):
        """Resume discovery can use saved categories instead of current CLI."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_standard")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STANDARD.value)
        def test_standard():
            return "standard"

        @GLTCore.GLTTestAttr(cat="pci", name="test_pci_standard")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STANDARD.value)
        def test_pci_standard():
            return "pci"

        discovered = {
            ("cxl",): [test_standard],
            ("pci",): [test_pci_standard],
        }

        with mock.patch.object(
                greenlit, "test_case_discover",
                side_effect=lambda cats: discovered[tuple(cats)]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["cxl"], None, {GLTCore.GLTPlanType.STANDARD.value})

        self.assertEqual(error, "")
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [f"{obj.__wrapped__.attr.category}:{obj.__wrapped__.attr.name}"
             for obj in test_objs],
            ["cxl:test_standard"])

    def test_discover_target_test_cases_rejects_normal_cross_plan_prereq(self):
        """Runnable category tests cannot depend on out-of-plan prerequisites."""

        @GLTCore.GLTRequires.must_pass("hwtracing", "test_stress_prereq")
        @GLTCore.GLTTestAttr(cat="pci", name="test_sanity")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_sanity():
            return "sanity"

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_stress_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STRESS.value)
        def test_stress_prereq():
            return "stress"

        discovered = {
            ("pci",): [test_sanity],
            ("hwtracing",): [test_stress_prereq],
        }

        with mock.patch.object(
                greenlit, "test_case_discover",
                side_effect=lambda cats: discovered[tuple(cats)]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["pci"], None,
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(test_objs, [])
        self.assertIn("Required test ID hwtracing:test_stress_prereq", error)
        self.assertIn("not selected by plan", error)
        self.assertEqual(exit_code, greenlit.CLI_RUNTIME_ERROR)

    def test_discover_target_test_cases_limits_cross_category_prereqs(self):
        """Category discovery includes required tests, not whole required categories."""

        @GLTCore.GLTRequires.must_pass("hwtracing", "test_prereq")
        @GLTCore.GLTTestAttr(cat="arm64", name="test_main")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_main():
            return "main"

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_prereq():
            return "prereq"

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_unrelated")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_unrelated():
            return "unrelated"

        discovered = {
            ("arm64",): [test_main],
            ("hwtracing",): [test_prereq, test_unrelated],
        }

        with mock.patch.object(
                greenlit, "test_case_discover",
                side_effect=lambda cats: discovered[tuple(cats)]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["arm64"], None,
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(error, "")
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [f"{obj.__wrapped__.attr.category}:{obj.__wrapped__.attr.name}"
             for obj in test_objs],
            ["arm64:test_main", "hwtracing:test_prereq"])

    def test_discover_target_test_cases_limits_test_case_prereqs(self):
        """--test-case discovery returns selected tests plus recursive prerequisites."""

        @GLTCore.GLTRequires.must_pass("hwtracing", "test_prereq")
        @GLTCore.GLTTestAttr(cat="arm64", name="test_selected")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_selected():
            return "selected"

        @GLTCore.GLTTestAttr(cat="arm64", name="test_unrelated")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_unrelated():
            return "unrelated"

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_prereq():
            return "prereq"

        discovered = {
            ("arm64",): [test_selected, test_unrelated],
            ("hwtracing",): [test_prereq],
        }

        with mock.patch.object(
                greenlit, "test_case_discover",
                side_effect=lambda cats: discovered[tuple(cats)]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["arm64"], ["arm64:test_selected"],
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(error, "")
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [f"{obj.__wrapped__.attr.category}:{obj.__wrapped__.attr.name}"
             for obj in test_objs],
            ["arm64:test_selected", "hwtracing:test_prereq"])

    def test_discover_target_test_cases_uses_saved_test_cases_for_resume(self):
        """Saved resume test IDs do not broaden dependency categories."""

        @GLTCore.GLTRequires.must_pass("hwtracing", "test_prereq")
        @GLTCore.GLTTestAttr(cat="arm64", name="test_main")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_main():
            return "main"

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_prereq():
            return "prereq"

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_unrelated")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_unrelated():
            return "unrelated"

        discovered = {
            ("arm64",): [test_main],
            ("hwtracing",): [test_prereq, test_unrelated],
        }

        with mock.patch.object(
                greenlit, "test_case_discover",
                side_effect=lambda cats: discovered[tuple(cats)]):
            test_objs, error, exit_code = greenlit.discover_target_test_cases(
                ["arm64", "hwtracing"],
                ["arm64:test_main", "hwtracing:test_prereq"],
                {GLTCore.GLTPlanType.SANITY.value})

        self.assertEqual(error, "")
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [f"{obj.__wrapped__.attr.category}:{obj.__wrapped__.attr.name}"
             for obj in test_objs],
            ["arm64:test_main", "hwtracing:test_prereq"])


class TestGreenlitResumeHelpers(GLTTestBase):
    """Test resume-specific helper behavior."""

    def test_setup_session_requires_resolved_resume_session_argument(self):
        """Resume session resolution belongs to main, not setup_session."""
        session_parameter = inspect.signature(
            greenlit.setup_session).parameters["resume_session"]

        self.assertIs(session_parameter.default, inspect.Parameter.empty)

    def test_read_session_summary_returns_saved_plan_and_categories(self):
        """Resume setup reads the saved plan before test discovery."""
        with tempfile.TemporaryDirectory() as session:
            log_dir = os.path.join(session, "glt-test-suite")
            os.makedirs(log_dir)
            with open(os.path.join(log_dir, "test_progress.json"),
                      "w", encoding="UTF-8") as progress_file:
                json.dump([{
                    "Summary": {
                        "TestTags": [GLTCore.GLTPlanType.STANDARD.value],
                        "Categories": ["cxl"],
                    }
                }, {
                    "test_standard": {
                        "Category": "cxl",
                    }
                }], progress_file)

            ok, message, test_tags, categories, test_cases = (
                greenlit.read_session_summary(session))

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertEqual(test_tags, {GLTCore.GLTPlanType.STANDARD.value})
        self.assertEqual(categories, ["cxl"])
        self.assertEqual(test_cases, ["cxl:test_standard"])

    def test_read_session_summary_requires_registered_test_cases(self):
        """Resume metadata must include registered per-test categories."""
        with tempfile.TemporaryDirectory() as session:
            log_dir = os.path.join(session, "glt-test-suite")
            os.makedirs(log_dir)
            with open(os.path.join(log_dir, "test_progress.json"),
                      "w", encoding="UTF-8") as progress_file:
                json.dump([{
                    "Summary": {
                        "TestTags": [GLTCore.GLTPlanType.STANDARD.value],
                        "Categories": ["cxl"],
                    }
                }], progress_file)

            ok, message, test_tags, categories, test_cases = (
                greenlit.read_session_summary(session))

        self.assertFalse(ok)
        self.assertIn("registered test cases", message)
        self.assertEqual(test_tags, set())
        self.assertIsNone(categories)
        self.assertIsNone(test_cases)

    def test_read_session_summary_ignores_unregistered_progress_entries(self):
        """Only entries with Category are used to rebuild exact test IDs."""
        with tempfile.TemporaryDirectory() as session:
            log_dir = os.path.join(session, "glt-test-suite")
            os.makedirs(log_dir)
            with open(os.path.join(log_dir, "test_progress.json"),
                      "w", encoding="UTF-8") as progress_file:
                json.dump([{
                    "Summary": {
                        "TestTags": [GLTCore.GLTPlanType.STANDARD.value],
                        "Categories": ["cxl"],
                    }
                }, {
                    "Noise": "not a test progress entry"
                }, {
                    "test_standard": {
                        "Category": "cxl",
                    }
                }], progress_file)

            ok, message, test_tags, categories, test_cases = (
                greenlit.read_session_summary(session))

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertEqual(test_tags, {GLTCore.GLTPlanType.STANDARD.value})
        self.assertEqual(categories, ["cxl"])
        self.assertEqual(test_cases, ["cxl:test_standard"])

    def test_read_session_summary_reports_invalid_json(self):
        """Malformed progress JSON is returned as an error message."""
        with tempfile.TemporaryDirectory() as session:
            log_dir = os.path.join(session, "glt-test-suite")
            os.makedirs(log_dir)
            with open(os.path.join(log_dir, "test_progress.json"),
                      "w", encoding="UTF-8") as progress_file:
                progress_file.write("{bad json")

            ok, message, test_tags, categories, test_cases = (
                greenlit.read_session_summary(session))

        self.assertFalse(ok)
        self.assertIn("Unable to parse resume progress", message)
        self.assertEqual(test_tags, set())
        self.assertIsNone(categories)
        self.assertIsNone(test_cases)

    def test_main_rejects_empty_test_selection(self):
        """CLI reports an empty runnable selection instead of exiting silently."""
        initial_args = argparse.Namespace(
            list=False,
            resume=None,
            verbose=False,
            ext_dir="/missing_ext_dir")
        args = _args(["cxl"])
        for plan in GLTCore.GLTPlanType:
            setattr(args, plan.value, False)

        initial_parser = mock.MagicMock()
        initial_parser.parse_known_args.return_value = (initial_args, [])

        with mock.patch.object(greenlit, "check_root_privileges"), \
                mock.patch.object(greenlit, "create_initial_parser",
                                  return_value=initial_parser), \
                mock.patch.object(greenlit, "test_category_discover",
                                  return_value=["cxl"]), \
                mock.patch.object(greenlit, "parse_arguments",
                                  return_value=args), \
                mock.patch.object(greenlit, "test_case_discover",
                                  return_value=[]), \
                mock.patch.object(greenlit, "setup_session") as setup_session, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                self.assertRaises(SystemExit) as ctx:
            greenlit.main()

        self.assertEqual(ctx.exception.code, greenlit.CLI_USAGE_ERROR)
        self.assertIn("No test cases selected", stderr.getvalue())
        setup_session.assert_not_called()

    def test_resolve_target_test_objects_reports_empty_normal_run(self):
        """Empty normal run selection is handled before main builds a scheduler."""
        initial_args = argparse.Namespace(list=False)
        args = _args(["cxl"])
        active_plan_tags = {GLTCore.GLTPlanType.SANITY.value}

        with mock.patch.object(
                greenlit, "resolve_run_selection",
                return_value=(None, active_plan_tags, ["cxl"], None, "")), \
                mock.patch.object(greenlit, "discover_target_test_cases",
                                  return_value=([], "", 0)):
            resume_session, test_tags, test_objs, error, exit_code = (
                greenlit.resolve_target_test_objects(
                    initial_args, args, ["cxl"], []))

        self.assertIsNone(resume_session)
        self.assertEqual(test_tags, active_plan_tags)
        self.assertEqual(test_objs, [])
        self.assertIn("No test cases selected", error)
        self.assertEqual(exit_code, greenlit.CLI_USAGE_ERROR)

    def test_resolve_target_test_objects_reports_resume_errors_as_runtime(self):
        """Resume metadata read errors are not CLI usage errors."""
        initial_args = argparse.Namespace(list=False, resume=0)
        args = _args(["cxl"])

        with mock.patch.object(
                greenlit, "resolve_run_selection",
                return_value=(
                    "/tmp/session",
                    set(),
                    None,
                    None,
                    "Unable to parse resume progress: bad json")):
            resume_session, test_tags, test_objs, error, exit_code = (
                greenlit.resolve_target_test_objects(
                    initial_args, args, ["cxl"], []))

        self.assertEqual(resume_session, "/tmp/session")
        self.assertEqual(test_tags, set())
        self.assertEqual(test_objs, [])
        self.assertIn("Unable to parse resume progress", error)
        self.assertEqual(exit_code, greenlit.CLI_RUNTIME_ERROR)

    def test_resolve_target_test_objects_allows_empty_list(self):
        """Empty --list selection is not an error."""
        initial_args = argparse.Namespace(list=True)
        args = _args(["cxl"], list_tests=True)
        active_plan_tags = {GLTCore.GLTPlanType.SANITY.value}

        with mock.patch.object(
                greenlit, "resolve_run_selection",
                return_value=(None, active_plan_tags, ["cxl"], None, "")), \
                mock.patch.object(greenlit, "discover_target_test_cases",
                                  return_value=([], "", 0)):
            resume_session, test_tags, test_objs, error, exit_code = (
                greenlit.resolve_target_test_objects(
                    initial_args, args, ["cxl"], []))

        self.assertIsNone(resume_session)
        self.assertEqual(test_tags, active_plan_tags)
        self.assertEqual(test_objs, [])
        self.assertEqual(error, "")
        self.assertEqual(exit_code, 0)

    def test_resolve_target_test_objects_forwards_discovery_error(self):
        """Discovery decides the exit code for discovery-specific failures."""
        initial_args = argparse.Namespace(list=False)
        args = _args(["cxl"])
        active_plan_tags = {GLTCore.GLTPlanType.SANITY.value}

        with mock.patch.object(
                greenlit, "resolve_run_selection",
                return_value=(None, active_plan_tags, ["cxl"], None, "")), \
                mock.patch.object(
                    greenlit, "discover_target_test_cases",
                    return_value=(
                        [],
                        "Required test ID cxl:test_prereq referenced by "
                        "cxl:test_selected does not exist",
                        greenlit.CLI_RUNTIME_ERROR)):
            resume_session, test_tags, test_objs, error, exit_code = (
                greenlit.resolve_target_test_objects(
                    initial_args, args, ["cxl"], []))

        self.assertIsNone(resume_session)
        self.assertEqual(test_tags, active_plan_tags)
        self.assertEqual(test_objs, [])
        self.assertIn("Required test ID cxl:test_prereq", error)
        self.assertEqual(exit_code, greenlit.CLI_RUNTIME_ERROR)

    def test_main_reports_missing_required_test_as_runtime_error(self):
        """Broken test metadata is not a CLI usage error."""
        initial_args = argparse.Namespace(
            list=False,
            resume=None,
            verbose=False,
            ext_dir="/missing_ext_dir")
        args = _args(["cxl"])
        for plan in GLTCore.GLTPlanType:
            setattr(args, plan.value, False)

        initial_parser = mock.MagicMock()
        initial_parser.parse_known_args.return_value = (initial_args, [])

        with mock.patch.object(greenlit, "check_root_privileges"), \
                mock.patch.object(greenlit, "create_initial_parser",
                                  return_value=initial_parser), \
                mock.patch.object(greenlit, "test_category_discover",
                                  return_value=["cxl"]), \
                mock.patch.object(greenlit, "parse_arguments",
                                  return_value=args), \
                mock.patch.object(greenlit, "discover_target_test_cases",
                                  return_value=(
                                      [],
                                      "Required test ID cxl:test_prereq "
                                      "referenced by cxl:test_selected "
                                      "does not exist",
                                      greenlit.CLI_RUNTIME_ERROR)), \
                mock.patch.object(greenlit, "setup_session") as setup_session, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                self.assertRaises(SystemExit) as ctx:
            greenlit.main()

        self.assertEqual(ctx.exception.code, greenlit.CLI_RUNTIME_ERROR)
        self.assertIn("Required test ID cxl:test_prereq", stderr.getvalue())
        setup_session.assert_not_called()

    def test_main_list_allows_empty_test_selection(self):
        """--list can print an empty result without creating a session."""
        initial_args = argparse.Namespace(
            list=True,
            resume=None,
            verbose=False,
            ext_dir="/missing_ext_dir")
        args = _args(["cxl"], list_tests=True)
        for plan in GLTCore.GLTPlanType:
            setattr(args, plan.value, False)

        initial_parser = mock.MagicMock()
        initial_parser.parse_known_args.return_value = (initial_args, [])

        with mock.patch.object(greenlit, "check_root_privileges"), \
                mock.patch.object(greenlit, "create_initial_parser",
                                  return_value=initial_parser), \
                mock.patch.object(greenlit, "test_category_discover",
                                  return_value=["cxl"]), \
                mock.patch.object(greenlit, "parse_arguments",
                                  return_value=args), \
                mock.patch.object(greenlit, "test_case_discover",
                                  return_value=[]), \
                mock.patch.object(greenlit, "setup_session") as setup_session, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            greenlit.main()

        self.assertEqual(stdout.getvalue(), "")
        setup_session.assert_not_called()

    def test_main_resume_discovers_with_saved_plan(self):
        """Resume without a plan flag discovers tests from saved Summary."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_standard")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STANDARD.value)
        def test_standard():
            return "standard"

        with tempfile.TemporaryDirectory() as session:
            log_dir = os.path.join(session, "glt-test-suite")
            os.makedirs(log_dir)
            with open(os.path.join(log_dir, "test_progress.json"),
                      "w", encoding="UTF-8") as progress_file:
                json.dump([{
                    "Summary": {
                        "TestTags": [GLTCore.GLTPlanType.STANDARD.value],
                        "Categories": ["cxl"],
                    }
                }, {
                    "test_standard": {
                        "Category": "cxl",
                    }
                }], progress_file)

            initial_args = argparse.Namespace(
                list=False,
                resume=-1,
                verbose=False,
                ext_dir=os.path.join(session, "missing_ext_dir"))
            args = _args(["cxl"], resume=-1)
            for plan in GLTCore.GLTPlanType:
                setattr(args, plan.value, False)

            initial_parser = mock.MagicMock()
            initial_parser.parse_known_args.return_value = (initial_args, [])

            with mock.patch.object(greenlit, "check_root_privileges"), \
                    mock.patch.object(greenlit, "create_initial_parser",
                                      return_value=initial_parser), \
                    mock.patch.object(greenlit, "test_category_discover",
                                      return_value=["cxl"]), \
                    mock.patch.object(greenlit, "parse_arguments",
                                      return_value=args), \
                    mock.patch.object(greenlit, "get_cached_session_path",
                                      return_value=session), \
                    mock.patch.object(greenlit, "discover_target_test_cases",
                                      return_value=([test_standard], "",
                                                    0)) as discover, \
                    mock.patch.object(greenlit, "setup_session",
                                      return_value=(session, self.mock_logger)), \
                    mock.patch.object(greenlit, "apply_glt_configs"), \
                    mock.patch.object(greenlit.signal, "signal"), \
                    mock.patch.object(greenlit.glt_utils, "log_system_info"), \
                    mock.patch.object(greenlit.glt_utils, "is_systemd_running",
                                      return_value=True), \
                    mock.patch.object(greenlit, "check_required_commands"), \
                    mock.patch.object(greenlit, "check_required_files"), \
                    mock.patch.object(greenlit, "resume_from_session"), \
                    mock.patch.object(greenlit, "perform_tests"):
                greenlit.main()

        self.assertEqual(
            discover.call_args.args[2],
            {GLTCore.GLTPlanType.STANDARD.value})
        self.assertEqual(discover.call_args.args[0], ["cxl"])
        self.assertEqual(discover.call_args.args[1], ["cxl:test_standard"])

    def test_main_resume_discovers_with_saved_test_cases(self):
        """Resume passes exact saved test IDs back to discovery."""

        @GLTCore.GLTTestAttr(cat="arm64", name="test_main")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_main():
            return "main"

        @GLTCore.GLTTestAttr(cat="hwtracing", name="test_prereq")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.SANITY.value)
        def test_prereq():
            return "prereq"

        with tempfile.TemporaryDirectory() as session:
            log_dir = os.path.join(session, "glt-test-suite")
            os.makedirs(log_dir)
            with open(os.path.join(log_dir, "test_progress.json"),
                      "w", encoding="UTF-8") as progress_file:
                json.dump([{
                    "Summary": {
                        "TestTags": [GLTCore.GLTPlanType.SANITY.value],
                        "Categories": ["arm64", "hwtracing"],
                    }
                }, {
                    "test_main": {
                        "Category": "arm64",
                    }
                }, {
                    "test_prereq": {
                        "Category": "hwtracing",
                    }
                }], progress_file)

            initial_args = argparse.Namespace(
                list=False,
                resume=-1,
                verbose=False,
                ext_dir=os.path.join(session, "missing_ext_dir"))
            args = _args(["arm64", "hwtracing"], resume=-1)
            for plan in GLTCore.GLTPlanType:
                setattr(args, plan.value, False)

            initial_parser = mock.MagicMock()
            initial_parser.parse_known_args.return_value = (initial_args, [])

            with mock.patch.object(greenlit, "check_root_privileges"), \
                    mock.patch.object(greenlit, "create_initial_parser",
                                      return_value=initial_parser), \
                    mock.patch.object(greenlit, "test_category_discover",
                                      return_value=["arm64", "hwtracing"]), \
                    mock.patch.object(greenlit, "parse_arguments",
                                      return_value=args), \
                    mock.patch.object(greenlit, "get_cached_session_path",
                                      return_value=session), \
                    mock.patch.object(greenlit, "discover_target_test_cases",
                                      return_value=([test_main, test_prereq],
                                                    "", 0)) as discover, \
                    mock.patch.object(greenlit, "setup_session",
                                      return_value=(session, self.mock_logger)), \
                    mock.patch.object(greenlit, "apply_glt_configs"), \
                    mock.patch.object(greenlit.signal, "signal"), \
                    mock.patch.object(greenlit.glt_utils, "log_system_info"), \
                    mock.patch.object(greenlit.glt_utils, "is_systemd_running",
                                      return_value=True), \
                    mock.patch.object(greenlit, "check_required_commands"), \
                    mock.patch.object(greenlit, "check_required_files"), \
                    mock.patch.object(greenlit, "resume_from_session"), \
                    mock.patch.object(greenlit, "perform_tests"):
                greenlit.main()

        self.assertEqual(discover.call_args.args[0], ["arm64", "hwtracing"])
        self.assertEqual(
            discover.call_args.args[1],
            ["arm64:test_main", "hwtracing:test_prereq"])
