# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual C run loop with mocked KVM, signals, and time."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestKvmMicrovmTimeout(unittest.TestCase):
    """Verify the timeout contract without root or KVM hardware."""

    def test_run_deadline_and_caller_alarm(self):
        """Retries share one deadline and caller signal state survives."""
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("A C compiler is required for timeout tests")
        source = Path(__file__).with_name("kvm_run_timeout.c")
        with tempfile.TemporaryDirectory(prefix="glt-kvm-timeout-") as directory:
            probe = Path(directory) / "timeout-test"
            subprocess.run(
                [
                    compiler, "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(probe),
                    "-Wl,--wrap=ioctl", "-Wl,--wrap=alarm",
                    "-Wl,--wrap=sigaction", "-Wl,--wrap=clock_gettime",
                    "-Wl,--wrap=raise",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            for scenario in (
                "normal", "eintr", "exit_intr", "entry_alarm", "caller_alarm",
                "expired_alarm", "zero_timeout", "ioctl_error",
                "caller_alarm_setup_delay", "clock_error",
            ):
                with self.subTest(scenario=scenario):
                    result = subprocess.run(
                        [str(probe), scenario],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
