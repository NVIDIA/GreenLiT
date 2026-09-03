"""Unit tests for GLTScheduler resume behavior."""

import json
import os
import tempfile
from unittest import mock

from tests.test_base import GLTTestBase
from common import glt_session
from src.common.glt_core import (
    GLTPlan,
    GLTPlanType,
    GLTTestAttr,
    GLTTestResult,
    GLTScheduler,
)


def _write_progress_file(progress_path, progress):
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, "w", encoding="UTF-8") as progress_file:
        json.dump(progress, progress_file)


def _write_session_progress(session_dir, progress):
    _write_progress_file(
        glt_session.get_session_progress_path(session_dir), progress)


class TestGLTSchedulerResume(GLTTestBase):
    """Test GLTScheduler resume behavior."""

    def test_progress_read_restores_external_test_names(self):
        """Resume keeps registered external tests whose names do not start with test_."""

        @GLTTestAttr(cat="external", name="cpu_feat_ls64_test_test_ls64_sh")
        @GLTPlan(GLTPlanType.EXT_EXECUTABLE.value)
        def external_test():
            return "external"

        scheduler = GLTScheduler([external_test])
        scheduler.sort()

        with tempfile.TemporaryDirectory() as session_dir:
            _write_session_progress(session_dir, [{
                "Summary": {
                    "TestTags": [GLTPlanType.EXT_EXECUTABLE.value],
                    "Categories": ["external"],
                }
            }, {
                "cpu_feat_ls64_test_test_ls64_sh": {
                    "Registry": "2026-06-25 15:35:54",
                    "Category": "external",
                }
            }])

            progress_error = scheduler.progress_read(session_dir)

        self.assertEqual(progress_error, "")
        self.assertEqual(scheduler.test_tags,
                         {GLTPlanType.EXT_EXECUTABLE.value})
        self.assertEqual(scheduler.test_categories, ["external"])
        self.assertEqual(scheduler.test_list, [external_test])

    def test_progress_read_preserves_test_list_without_resumable_entries(self):
        """Malformed progress should not silently erase the runnable test list."""

        @GLTTestAttr(cat="external", name="cpu_feat_ls64_test_test_ls64_sh")
        @GLTPlan(GLTPlanType.EXT_EXECUTABLE.value)
        def external_test():
            return "external"

        scheduler = GLTScheduler([external_test])
        scheduler.sort()
        initial_progress = [{"Current": "progress"}]
        scheduler.progress = initial_progress

        with tempfile.TemporaryDirectory() as session_dir:
            progress = [{"Noise": "not a test progress entry"}]
            _write_session_progress(session_dir, progress)

            progress_error = scheduler.progress_read(session_dir)

        self.assertIn("TestTags or Categories", progress_error)
        self.assertIs(scheduler.progress, initial_progress)
        self.assertEqual(scheduler.test_categories, [])
        self.assertEqual(scheduler.test_list, [external_test])

    def test_progress_read_reports_unknown_resumed_test_case(self):
        """Stale progress is rejected without partially changing scheduler state."""

        @GLTTestAttr(cat="external", name="cpu_feat_ls64_test_test_ls64_sh")
        @GLTPlan(GLTPlanType.EXT_EXECUTABLE.value)
        def external_test():
            return "external"

        scheduler = GLTScheduler([external_test])
        scheduler.sort()
        initial_progress = [{"Current": "progress"}]
        scheduler.progress = initial_progress
        scheduler.test_tags = {GLTPlanType.STANDARD.value}
        scheduler.test_categories = ["current"]

        with tempfile.TemporaryDirectory() as session_dir:
            _write_session_progress(session_dir, [{
                "Summary": {
                    "TestTags": [GLTPlanType.EXT_EXECUTABLE.value],
                    "Categories": ["external"],
                }
            }, {
                "cpu_feat_ls64_test_test_ls64_sh": {
                    "Category": "external",
                    "Result": GLTTestResult.PASS.value,
                    "ResultReason": "completed",
                }
            }, {
                "missing_sh": {
                    "Registry": "2026-06-25 15:35:54",
                    "Category": "external",
                }
            }])

            progress_error = scheduler.progress_read(session_dir)

        self.assertIn(
            "unknown test case external:missing_sh", progress_error)
        self.assertIs(scheduler.progress, initial_progress)
        self.assertEqual(scheduler.test_tags, {GLTPlanType.STANDARD.value})
        self.assertEqual(scheduler.test_categories, ["current"])
        self.assertEqual(scheduler.test_list, [external_test])
        self.assertEqual(
            external_test.__wrapped__.attr.result,
            (GLTTestResult.NA, ""))

    def test_progress_read_reports_missing_progress_file(self):
        """An I/O failure is returned without changing scheduler state."""

        @GLTTestAttr(cat="external", name="test_external")
        @GLTPlan(GLTPlanType.EXT_EXECUTABLE.value)
        def external_test():
            return "external"

        scheduler = GLTScheduler([external_test])
        scheduler.sort()

        with tempfile.TemporaryDirectory() as session_dir:
            progress_error = scheduler.progress_read(session_dir)

        self.assertIn("Unable to read resume progress", progress_error)
        self.assertEqual(scheduler.progress, [])
        self.assertEqual(scheduler.test_list, [external_test])

    def test_progress_read_reports_invalid_json(self):
        """Invalid JSON is returned without changing scheduler state."""

        @GLTTestAttr(cat="external", name="test_external")
        @GLTPlan(GLTPlanType.EXT_EXECUTABLE.value)
        def external_test():
            return "external"

        scheduler = GLTScheduler([external_test])
        scheduler.sort()

        decode_error = json.JSONDecodeError("bad json", "{bad json", 1)
        with tempfile.TemporaryDirectory() as session_dir, \
                mock.patch.object(
                    glt_session, "read_progress_file",
                    side_effect=decode_error):
            progress_error = scheduler.progress_read(session_dir)

        self.assertIn("Unable to parse resume progress", progress_error)
        self.assertEqual(scheduler.progress, [])
        self.assertEqual(scheduler.test_list, [external_test])

    def test_progress_read_applies_test_beside_non_dict_summary(self):
        """Summary-like metadata does not hide a real test in the same entry."""

        @GLTTestAttr(cat="external", name="test_external")
        @GLTPlan(GLTPlanType.EXT_EXECUTABLE.value)
        def external_test():
            return "external"

        scheduler = GLTScheduler([external_test])
        scheduler.sort()

        with tempfile.TemporaryDirectory() as session_dir:
            _write_session_progress(session_dir, [{
                "Summary": {
                    "TestTags": [GLTPlanType.EXT_EXECUTABLE.value],
                    "Categories": ["external"],
                }
            }, {
                "Summary": "legacy marker",
                "test_external": {
                    "Category": "external",
                    "Result": GLTTestResult.PASS.value,
                    "ResultReason": "completed",
                },
            }])

            progress_error = scheduler.progress_read(session_dir)

        self.assertEqual(progress_error, "")
        self.assertEqual(scheduler.test_list, [external_test])
        self.assertEqual(
            external_test.__wrapped__.attr.result,
            (GLTTestResult.PASS, "completed"))

    def test_progress_read_without_session_path_uses_shared_progress_reader(self):
        """Current-session reads use the same progress reader as resume reads."""

        @GLTTestAttr(cat="external", name="cpu_feat_ls64_test_test_ls64_sh")
        @GLTPlan(GLTPlanType.EXT_EXECUTABLE.value)
        def external_test():
            return "external"

        scheduler = GLTScheduler([external_test])
        scheduler.sort()

        with tempfile.TemporaryDirectory() as log_dir:
            scheduler.log_dir = log_dir
            _write_progress_file(os.path.join(
                log_dir, glt_session.SESSION_PROGRESS_FILE), [{
                "Summary": {
                    "TestTags": [GLTPlanType.EXT_EXECUTABLE.value],
                    "Categories": ["external"],
                }
            }, {
                "cpu_feat_ls64_test_test_ls64_sh": {
                    "Registry": "2026-06-25 15:35:54",
                    "Category": "external",
                }
            }])

            progress_error = scheduler.progress_read()

        self.assertEqual(progress_error, "")
        self.assertEqual(scheduler.test_tags,
                         {GLTPlanType.EXT_EXECUTABLE.value})
        self.assertEqual(scheduler.test_categories, ["external"])
        self.assertEqual(scheduler.test_list, [external_test])
