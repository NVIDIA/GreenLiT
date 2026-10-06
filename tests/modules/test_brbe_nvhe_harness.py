# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual C sampling oracle without KVM or perf hardware."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestBrbeNvheHarness(unittest.TestCase):
    """Compile and run the production sampling checks with mocked counters."""

    def test_guest_window_sampling_oracle(self):
        """Cover absent, misplaced, short and valid guest-window evidence."""
        compilers = [compiler for compiler in ("gcc", "clang") if shutil.which(compiler)]
        if not compilers:
            self.skipTest("A C compiler is required for harness oracle tests")
        root = Path(__file__).resolve().parents[2]
        source = (root / "src/modules/arm64/brbe_nvhe/glt_brbe_nvhe_harness.c").read_text()
        declarations = []
        for name in ("options", "guest_perf_window", "evidence"):
            match = re.search(rf"^struct {name} \{{.*?^\}};", source, re.M | re.S)
            self.assertIsNotNone(match, name)
            declarations.append(match.group())
        for name in (
            "address_in_range", "branch_hits_range", "json_write_string",
            "write_result_json", "set_error", "pull_bytes", "parse_sample_record",
            "require_one_new_sample", "check_guest_perf_window",
        ):
            match = re.search(rf"^static [^;{{]*\b{name}\(", source, re.M)
            self.assertIsNotNone(match, name)
            end = source.index("\nstatic ", match.start() + 1)
            declarations.append(source[match.start():end])
        fixture = Path(__file__).with_name("brbe_nvhe_oracle.c")
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "brbe_nvhe_oracle.inc").write_text("\n\n".join(declarations))
            for compiler in compilers:
                with self.subTest(compiler=compiler):
                    executable = directory / compiler
                    build = subprocess.run(
                        [compiler, "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                         "-I", str(directory), str(fixture), "-o", str(executable)],
                        check=False, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(build.returncode, 0, build.stderr)
                    result_path = directory / "result.json"
                    result = subprocess.run(
                        [str(executable), str(result_path)], check=False, capture_output=True,
                        text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    evidence = json.loads(result_path.read_text())
                    self.assertEqual(evidence["schema_version"], 3)
                    self.assertEqual(evidence["guest_samples"], 1)
                    self.assertEqual(evidence["guest_windows_verified"], 1)
                    self.assertEqual(evidence["min_guest_instruction_delta"], 90)
                    self.assertEqual(evidence["context_switches"], 6)


if __name__ == "__main__":
    unittest.main()
