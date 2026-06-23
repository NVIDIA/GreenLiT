"""Unit tests for GLTScheduler resume behavior."""

import json
import os
import tempfile

from tests.test_base import GLTTestBase
from src.common.glt_core import (
    GLTPlan,
    GLTPlanType,
    GLTTestAttr,
    GLTScheduler,
)


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

        with tempfile.TemporaryDirectory() as session:
            log_dir = os.path.join(session, "glt-test-suite")
            os.makedirs(log_dir)
            with open(os.path.join(log_dir, "test_progress.json"),
                      "w", encoding="UTF-8") as progress_file:
                json.dump([{
                    "Summary": {
                        "TestTags": [GLTPlanType.EXT_EXECUTABLE.value],
                        "Categories": ["external"],
                    }
                }, {
                    "cpu_feat_ls64_test_test_ls64_sh": {
                        "Registry": "2026-06-25 15:35:54",
                        "Category": "external",
                    }
                }], progress_file)

            scheduler.progress_read(session)

        self.assertEqual(scheduler.test_tags,
                         {GLTPlanType.EXT_EXECUTABLE.value})
        self.assertEqual(scheduler.test_categories, ["external"])
        self.assertEqual(scheduler.test_list, [external_test])
