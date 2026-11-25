# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=duplicate-code
"""Unit tests for GreenLiT resume verbose logging behavior."""

import argparse
import io
import json
import logging
import os
import tempfile
from unittest import mock

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
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


class TestGreenlitBugReportCollection(GLTTestBase):
    """nvidia-bug-report.sh runs only when --collect-nv-bug-report is requested."""

    def setUp(self):
        super().setUp()
        self.logger = logging.Logger(__name__, level=logging.INFO)
        self.calls = []

    def _run_main(self, collect_bug_report, collect_result=(True, ""), verbose=False,
                  perform_tests_error=None):
        """Run main() for a fresh session; return (collect_logs mock, call order)."""

        @GLTCore.GLTTestAttr(cat="cxl", name="test_standard")
        @GLTCore.GLTPlan(GLTCore.GLTPlanType.STANDARD.value)
        def test_standard():
            return "standard"

        initial_args = argparse.Namespace(
            list=False, resume=None, verbose=verbose, ext_dir="/missing_ext_dir")
        args = argparse.Namespace(
            list=False, resume=None, verbose=verbose, test_cases=None, cxl=False,
            collect_nv_bug_report=collect_bug_report)
        initial_parser = mock.MagicMock()
        initial_parser.parse_known_args.return_value = (initial_args, [])
        order = mock.Mock()
        order.perform_tests.side_effect = perform_tests_error
        real_resolve = greenlit.resolve_collect_bug_report
        order.resolve_collect_bug_report.side_effect = real_resolve

        with tempfile.TemporaryDirectory() as session_dir, \
                mock.patch.object(greenlit, "check_root_privileges"), \
                mock.patch.object(greenlit, "create_initial_parser",
                                  return_value=initial_parser), \
                mock.patch.object(greenlit, "test_category_discover",
                                  return_value=["cxl"]), \
                mock.patch.object(greenlit, "parse_arguments", return_value=args), \
                mock.patch.object(
                    greenlit, "resolve_target_test_objects",
                    return_value=(
                        None, {GLTCore.GLTPlanType.STANDARD.value},
                        [test_standard], verbose, "", 0)), \
                mock.patch.object(greenlit, "setup_session",
                                  return_value=(session_dir, self.logger)), \
                mock.patch.object(greenlit, "apply_glt_configs"), \
                mock.patch.object(greenlit.signal, "signal"), \
                mock.patch.object(greenlit.glt_utils, "log_system_info"), \
                mock.patch.object(greenlit.glt_utils, "is_systemd_running",
                                  return_value=True), \
                mock.patch.object(greenlit, "check_required_commands"), \
                mock.patch.object(greenlit, "check_required_files"), \
                mock.patch.object(greenlit.GLTCore.GLTScheduler, "progress_save"), \
                mock.patch.object(greenlit, "resolve_collect_bug_report",
                                  order.resolve_collect_bug_report), \
                mock.patch.object(greenlit, "perform_tests",
                                  order.perform_tests), \
                mock.patch.object(greenlit.glt_utils, "collect_logs",
                                  order.collect_logs) as collect:
            order.collect_logs.return_value = collect_result
            try:
                greenlit.main()
            finally:
                self.calls = [call[0] for call in order.mock_calls]
        return collect, self.calls

    def test_logs_not_collected_by_default(self):
        """Without the flag, nvidia-bug-report.sh is not run."""
        collect, _ = self._run_main(collect_bug_report=False)
        collect.assert_not_called()

    def test_logs_not_collected_by_verbose_alone(self):
        """DEBUG logging does not imply bug-report collection."""
        collect, _ = self._run_main(collect_bug_report=False, verbose=True)
        collect.assert_not_called()

    def test_logs_collected_after_tests_when_requested(self):
        """With the flag, logs are collected after perform_tests (and its report)."""
        collect, calls = self._run_main(collect_bug_report=True)
        collect.assert_called_once()
        self.assertEqual(
            calls, ["resolve_collect_bug_report", "perform_tests", "collect_logs"])

    def test_choice_resolved_before_tests_run(self):
        """The request is saved before perform_tests, which a reboot hook may not leave."""
        with mock.patch.object(greenlit.GLTCore.GLTScheduler, "progress_update") as update:
            with self.assertRaises(SystemExit):
                self._run_main(
                    collect_bug_report=True, perform_tests_error=SystemExit)
        self.assertEqual(
            self.calls, ["resolve_collect_bug_report", "perform_tests"])
        update.assert_any_call("Summary", "CollectBugReport", True)

    def test_collect_failure_only_warns(self):
        """A collection failure is reported as a warning, not an error."""
        with self.assertLogs(self.logger, level=logging.WARNING) as logs:
            self._run_main(collect_bug_report=True, collect_result=(False, "boom"))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelno, logging.WARNING)
        self.assertEqual(logs.records[0].getMessage(), "Failed to collect logs: boom")

    def test_registry_test_cases_records_collect_option(self):
        """A fresh run saves the choice so a later --resume keeps it."""
        scheduler = mock.MagicMock()
        scheduler.test_categories = ["cxl"]
        scheduler.test_tags = {GLTCore.GLTPlanType.STANDARD.value}
        scheduler.test_list = []
        greenlit.registry_test_cases(
            self.logger, scheduler, collect_bug_report=True)
        scheduler.progress_update.assert_any_call(
            "Summary", "CollectBugReport", True)

    def test_resolve_uses_flag_and_persists_it(self):
        """--collect-nv-bug-report on a resume is saved for later resumes."""
        scheduler = mock.MagicMock()
        scheduler.progress = [{"Summary": {"Verbose": False}}]
        args = argparse.Namespace(collect_nv_bug_report=True)
        self.assertTrue(greenlit.resolve_collect_bug_report(args, scheduler))
        scheduler.progress_update.assert_called_once_with(
            "Summary", "CollectBugReport", True)

    def test_resolve_keeps_saved_choice_on_resume(self):
        """A resume without the flag still collects if the session asked for it."""
        scheduler = mock.MagicMock()
        scheduler.progress = [{"Summary": {"CollectBugReport": True}}]
        args = argparse.Namespace(collect_nv_bug_report=False)
        self.assertTrue(greenlit.resolve_collect_bug_report(args, scheduler))
        scheduler.progress_update.assert_not_called()

    def test_resolve_defaults_to_off(self):
        """Old sessions and runs without the flag do not collect."""
        scheduler = mock.MagicMock()
        scheduler.progress = [{"Summary": {"Verbose": True}}]
        args = argparse.Namespace(collect_nv_bug_report=False)
        self.assertFalse(greenlit.resolve_collect_bug_report(args, scheduler))
