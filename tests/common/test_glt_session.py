"""Unit tests for cached GreenLiT session helpers."""

import json
import os
import tempfile

from tests.test_base import GLTTestBase
from src.common import glt_session


class TestGLTSession(GLTTestBase):
    """Test cached session progress parsing."""

    def test_parse_session_summary_returns_stable_metadata(self):
        """Summary metadata is parsed while per-test progress stays raw."""
        progress = [{
            "Summary": {
                "TestTags": ["sanity"],
                "Categories": ["cxl"],
                "Verbose": True,
            }
        }, {
            "test_standard": {
                "Category": "cxl",
                "TriggerReboot": "2026-07-08 10:00:00",
            }
        }]

        summary = glt_session.parse_session_summary(progress)

        self.assertEqual(summary.test_tags, {"sanity"})
        self.assertEqual(summary.categories, ["cxl"])
        self.assertEqual(summary.test_cases, ["cxl:test_standard"])
        self.assertTrue(summary.verbose)

    def test_get_session_progress_path_builds_progress_file_path(self):
        """Session directories resolve to the cached progress file path."""
        progress = [{
            "Summary": {
                "TestTags": ["sanity"],
                "Categories": ["cxl"],
            }
        }, {
            "test_standard": {
                "Category": "cxl",
            }
        }]

        with tempfile.TemporaryDirectory() as session_dir:
            progress_path = glt_session.get_session_progress_path(session_dir)
            os.makedirs(os.path.dirname(progress_path))
            with open(progress_path, "w", encoding="UTF-8") as progress_file:
                json.dump(progress, progress_file)

            session_progress = glt_session.read_progress_file(progress_path)

        self.assertEqual(session_progress.progress, progress)
        self.assertEqual(session_progress.summary_error(), "")
        self.assertFalse(session_progress.summary.verbose)
        self.assertEqual(session_progress.summary.test_cases, ["cxl:test_standard"])

    def test_read_progress_file_returns_raw_progress_and_summary(self):
        """A direct progress file read returns raw progress plus parsed metadata."""
        progress = [{
            "Summary": {
                "TestTags": ["sanity"],
                "Categories": ["cxl"],
            }
        }, {
            "test_standard": {
                "Category": "cxl",
            }
        }]

        with tempfile.TemporaryDirectory() as log_dir:
            progress_path = os.path.join(log_dir, glt_session.SESSION_PROGRESS_FILE)
            with open(progress_path, "w", encoding="UTF-8") as progress_file:
                json.dump(progress, progress_file)

            session_progress = glt_session.read_progress_file(progress_path)

        self.assertEqual(session_progress.progress, progress)
        self.assertEqual(session_progress.summary_error(), "")
        self.assertFalse(session_progress.summary.verbose)
        self.assertEqual(session_progress.summary.test_cases, ["cxl:test_standard"])

    def test_read_progress_file_reports_summary_error(self):
        """A malformed summary is reported before callers use metadata."""
        progress = [{
            "Summary": {
                "TestTags": ["sanity"],
                "Categories": ["cxl"],
            }
        }]

        with tempfile.TemporaryDirectory() as log_dir:
            progress_path = os.path.join(log_dir, glt_session.SESSION_PROGRESS_FILE)
            with open(progress_path, "w", encoding="UTF-8") as progress_file:
                json.dump(progress, progress_file)

            session_progress = glt_session.read_progress_file(progress_path)

        self.assertIsNone(session_progress.summary)
        self.assertIn("registered test cases", session_progress.summary_error())

    def test_read_progress_file_reports_invalid_format_for_non_array(self):
        """A non-array JSON payload is reported instead of crashing resume."""
        progress = {
            "Summary": {
                "TestTags": ["sanity"],
                "Categories": ["cxl"],
            }
        }

        with tempfile.TemporaryDirectory() as log_dir:
            progress_path = os.path.join(log_dir, glt_session.SESSION_PROGRESS_FILE)
            with open(progress_path, "w", encoding="UTF-8") as progress_file:
                json.dump(progress, progress_file)

            session_progress = glt_session.read_progress_file(progress_path)

        self.assertEqual(session_progress.progress, progress)
        self.assertIsNone(session_progress.summary)
        self.assertIn("expected a JSON array", session_progress.summary_error())
