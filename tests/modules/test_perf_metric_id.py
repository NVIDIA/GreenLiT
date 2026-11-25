# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the paired Arm IPC metric-ID oracle and applicability."""
# pylint: disable=wrong-import-position,duplicate-code

import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.hwtracing import glt_test_metric_id


class TestPerfMetricId(GLTTestBase):
    """Require the Arm pair on Arm64 and skip other architectures early."""

    def test_skips_non_arm_before_resolving_or_running_perf(self):
        """Non-Arm hosts skip before strict resolution or any stat call."""
        with self.mock.patch.object(
            glt_test_metric_id.platform, "machine", return_value="x86_64"
        ), self.mock.patch.object(
            glt_test_metric_id.perf_utils, "resolve_perf_command"
        ) as mock_resolver, self.mock.patch.object(
            glt_test_metric_id.subprocess, "run"
        ) as mock_run:
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                glt_test_metric_id.test_perf_metric_id(self.mock_sd, self.mock_logger)
        self.assertIn("Arm64", str(context.exception))
        mock_resolver.assert_not_called()
        mock_run.assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
