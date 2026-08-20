# pylint: disable=duplicate-code
"""Unit tests for GreenLiT resume verbose logging behavior."""

import argparse
import io
import json
import os
import tempfile
from unittest import mock

from tests.test_base import GLTTestBase
from src.common import glt_core as GLTCore
from src.greenlit import greenlit


def _write_session_progress(session_dir, summary):
    progress_path = greenlit.glt_session.get_session_progress_path(session_dir)
    log_dir = os.path.dirname(progress_path)
    os.makedirs(log_dir)
    with open(progress_path, "w", encoding="UTF-8") as progress_file:
        json.dump([{
            "Summary": summary,
        }, {
            "test_standard": {
                "Category": "cxl",
            }
        }], progress_file)
    return log_dir


class TestGreenlitResumeVerbose(GLTTestBase):
    """Test resume-specific verbose logging behavior."""

    def test_read_session_summary_defaults_verbose_false_for_old_sessions(self):
        """Resume remains compatible with sessions saved before Verbose existed."""
        with tempfile.TemporaryDirectory() as session_dir:
            _write_session_progress(session_dir, {
                "TestTags": [GLTCore.GLTPlanType.STANDARD.value],
                "Categories": ["cxl"],
            })

            ok, message, summary = greenlit.read_session_summary(session_dir)

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertFalse(summary.verbose)

    def test_read_session_summary_returns_saved_verbose_flag(self):
        """Saved resume metadata records whether the original run was verbose."""
        with tempfile.TemporaryDirectory() as session_dir:
            _write_session_progress(session_dir, {
                "TestTags": [GLTCore.GLTPlanType.STANDARD.value],
                "Categories": ["cxl"],
                "Verbose": True,
            })

            ok, message, summary = greenlit.read_session_summary(session_dir)

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertTrue(summary.verbose)

    def test_registry_test_cases_records_verbose_option(self):
        """New sessions persist whether they were started with --verbose."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_standard")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STANDARD.value)
        def test_standard():
            return "standard"

        with tempfile.TemporaryDirectory() as session_dir:
            progress_path = greenlit.glt_session.get_session_progress_path(
                session_dir)
            log_dir = os.path.dirname(progress_path)
            os.makedirs(log_dir)
            scheduler = GLTCore.GLTScheduler([test_standard], logdir=log_dir)
            scheduler.test_tags = {GLTCore.GLTPlanType.STANDARD.value}
            scheduler.test_categories = ["cxl"]

            greenlit.registry_test_cases(self.mock_logger, scheduler, verbose=True)

            with open(progress_path, encoding="UTF-8") as progress_file:
                progress = json.load(progress_file)

        summary = next(item["Summary"] for item in progress if "Summary" in item)
        self.assertTrue(summary["Verbose"])

    def test_resume_selection_uses_saved_verbose_for_session_setup(self):
        """Resuming a verbose session passes effective verbose into setup_session."""
        args = argparse.Namespace(
            list=False,
            resume=-1,
            verbose=False,
            test_cases=None,
            cxl=False)

        with tempfile.TemporaryDirectory() as session_dir:
            _write_session_progress(session_dir, {
                "TestTags": [GLTCore.GLTPlanType.STANDARD.value],
                "Categories": ["cxl"],
                "Verbose": True,
            })

            with mock.patch.object(greenlit, "get_cached_session_path",
                                   return_value=session_dir):
                resume_session_dir, summary, error = greenlit.resolve_run_selection(
                    args, ["cxl"])

            with mock.patch.object(greenlit.glt_logger, "get_logger",
                                   return_value=self.mock_logger), \
                    mock.patch.object(greenlit.glt_logger, "verbose") as set_verbose:
                session_path, logger = greenlit.setup_session(
                    summary.verbose, resume_session_dir)

        self.assertEqual(error, "")
        self.assertEqual(resume_session_dir, session_dir)
        self.assertEqual(summary.test_tags, {GLTCore.GLTPlanType.STANDARD.value})
        self.assertEqual(summary.categories, ["cxl"])
        self.assertEqual(summary.test_cases, ["cxl:test_standard"])
        self.assertEqual(session_path, session_dir)
        self.assertIs(logger, self.mock_logger)
        set_verbose.assert_called_once_with(self.mock_logger)

    def test_main_reports_resume_apply_error_as_runtime_error(self):
        """Resume apply failures are printed and exit as runtime errors."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_standard")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STANDARD.value)
        def test_standard():
            return "standard"

        initial_args = argparse.Namespace(
            list=False,
            resume=-1,
            verbose=False,
            ext_dir="/missing_ext_dir")
        args = argparse.Namespace(
            list=False,
            resume=-1,
            verbose=False,
            test_cases=None,
            cxl=False)
        initial_parser = mock.MagicMock()
        initial_parser.parse_known_args.return_value = (initial_args, [])
        resume_error = "Unable to read resume progress: missing file"

        with tempfile.TemporaryDirectory() as session_dir, \
                mock.patch.object(greenlit, "check_root_privileges"), \
                mock.patch.object(greenlit, "create_initial_parser",
                                  return_value=initial_parser), \
                mock.patch.object(greenlit, "test_category_discover",
                                  return_value=["cxl"]), \
                mock.patch.object(greenlit, "parse_arguments",
                                  return_value=args), \
                mock.patch.object(
                    greenlit, "resolve_target_test_objects",
                    return_value=(
                        session_dir,
                        {GLTCore.GLTPlanType.STANDARD.value},
                        [test_standard], False, "", 0)), \
                mock.patch.object(greenlit, "setup_session",
                                  return_value=(session_dir,
                                                self.mock_logger)), \
                mock.patch.object(greenlit, "apply_glt_configs"), \
                mock.patch.object(greenlit.signal, "signal"), \
                mock.patch.object(greenlit.glt_utils, "log_system_info"), \
                mock.patch.object(greenlit.glt_utils, "is_systemd_running",
                                  return_value=True), \
                mock.patch.object(greenlit, "check_required_commands"), \
                mock.patch.object(greenlit, "check_required_files"), \
                mock.patch.object(greenlit, "resume_from_session",
                                  return_value=resume_error), \
                mock.patch.object(greenlit, "perform_tests") as perform_tests, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                self.assertRaises(SystemExit) as ctx:
            greenlit.main()

        self.assertEqual(ctx.exception.code, greenlit.CLI_RUNTIME_ERROR)
        self.assertIn(resume_error, stderr.getvalue())
        perform_tests.assert_not_called()
