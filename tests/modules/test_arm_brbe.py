# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Arm Branch Record Buffer Extension validation."""
# pylint: disable=duplicate-code,protected-access

import subprocess

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from modules.arm64 import glt_test_brbe


class TestArmBrbe(GLTTestBase):
    """Test BRBE capability and runtime validation."""

    @staticmethod
    def _completed_process(returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(
            ["perf"], returncode, stdout=stdout, stderr=stderr)

    def _run_runtime_case(
        self, *, record_result=None, record_error=None,
        decode_result=None, decode_error=None,
    ):
        """Run the decorated Vera path with controlled subprocess results."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.return_value = True
        self.mock_sd.log_dir = "/var/cache/greenlit/session"
        if record_result is None:
            record_result = self._completed_process()
        if decode_result is None:
            decode_result = self._completed_process(
                stdout="0x1000/0x2000/P/-/-/0/CALL/-\n")

        def run_perf(command, **_kwargs):
            if command[1] == "record":
                if record_error:
                    raise record_error
                return record_result
            if decode_error:
                raise decode_error
            return decode_result

        with self.mock.patch.object(
            glt_test_brbe, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_brbe.check_kernel_config, "parse_kconfig",
            return_value={"CONFIG_ARM64_BRBE": "y"}
        ), self.mock.patch.object(
            glt_test_brbe.perf_utils, "read_brbe_capabilities",
            return_value={"armv8_pmuv3_0": 64}
        ), self.mock.patch.object(
            glt_test_brbe.perf_utils, "resolve_perf_command",
            return_value=("/perf", "")
        ), self.mock.patch.object(
            glt_test_brbe.subprocess, "run", side_effect=run_perf,
        ) as mock_run:
            glt_test_brbe.test_brbe_branch_stack(
                self.mock_sd, self.mock_logger)
        return mock_run


    def test_parse_stack_entries_returns_typed_address_records(self):
        """Return only decoded source, target, and branch-type records."""
        output = (
            "0x1000/0x2000/P/-/-/0/CALL/- "
            "0x3000/0x4000/M/-/-/4/RET/-\n"
            "not a branch stack\n"
        )
        self.assertEqual(
            glt_test_brbe._parse_brbe_stack_entries(output),
            [(0x1000, 0x2000, "CALL"), (0x3000, 0x4000, "RET")],
        )

    def test_parse_stack_entries_requires_decoded_branch_type(self):
        """Do not accept address pairs whose saved branch type is absent."""
        output = "0x1000/0x2000/P/-/-/0/-/-\n"
        self.assertEqual(glt_test_brbe._parse_brbe_stack_entries(output), [])

    def test_parse_stack_entries_rejects_kernel_addresses(self):
        """User-only branch recording must not expose kernel addresses."""
        for output in (
            "0xffff000000001000/0x2000/P/-/-/0/CALL/-\n",
            "0x1000/0xffff000000002000/P/-/-/0/CALL/-\n",
        ):
            with self.subTest(output=output), self.assertRaisesRegex(
                ValueError, "kernel-space address"
            ):
                glt_test_brbe._parse_brbe_stack_entries(output)

    def test_host_retains_standard_and_stress_plans(self):
        """The host test remains non-disruptive and independently selectable."""
        self.assertCountEqual(
            glt_test_brbe.test_brbe_branch_stack.plans, ["standard", "stress"]
        )

    def test_branch_stack_skips_non_vera(self):
        """Skip before probing non-Vera systems."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.return_value = False

        with self.mock.patch.object(
            glt_test_brbe, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_brbe.check_kernel_config, "parse_kconfig"
        ) as mock_parse, self.assertRaises(GLTCore.GLTTestSkip):
            glt_test_brbe.test_brbe_branch_stack(None, self.mock_logger)

        mock_parse.assert_not_called()

    def test_branch_stack_fails_without_kernel_config(self):
        """Vera fails when the required CONFIG_ARM64_BRBE setting is absent."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.return_value = True

        with self.mock.patch.object(
            glt_test_brbe, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_brbe.check_kernel_config, "parse_kconfig", return_value={}
        ), self.mock.patch.object(
            glt_test_brbe.perf_utils, "read_brbe_capabilities"
        ) as mock_capabilities, self.assertRaisesRegex(
            GLTCore.GLTTestFail, "CONFIG_ARM64_BRBE"
        ):
            glt_test_brbe.test_brbe_branch_stack(None, self.mock_logger)

        mock_capabilities.assert_not_called()

    def test_branch_stack_fails_when_kernel_config_is_unreadable(self):
        """An unreadable running-kernel config is an inspection failure."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.return_value = True

        with self.mock.patch.object(
            glt_test_brbe, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_brbe.check_kernel_config, "parse_kconfig",
            side_effect=OSError("permission denied")
        ), self.assertRaisesRegex(
            GLTCore.GLTTestFail, "Unable to inspect kernel config"
        ):
            glt_test_brbe.test_brbe_branch_stack(None, self.mock_logger)

    def test_branch_stack_fails_when_capability_is_missing(self):
        """Missing sysfs capability fails without claiming kernel provenance."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.return_value = True

        with self.mock.patch.object(
            glt_test_brbe, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_brbe.check_kernel_config, "parse_kconfig",
            return_value={"CONFIG_ARM64_BRBE": "y"}
        ), self.mock.patch.object(
            glt_test_brbe.perf_utils, "read_brbe_capabilities",
            side_effect=ValueError("BRBE caps/branches is missing")
        ), self.assertRaisesRegex(GLTCore.GLTTestFail, "BRBE caps/branches"):
            glt_test_brbe.test_brbe_branch_stack(None, self.mock_logger)

    def test_branch_stack_fails_without_matching_perf(self):
        """Missing matching perf fails with its package-specific reason."""
        fake_nvdev = self.mock.Mock()
        fake_nvdev.is_cpu_type.return_value = True

        with self.mock.patch.object(
            glt_test_brbe, "NvDev", return_value=fake_nvdev
        ), self.mock.patch.object(
            glt_test_brbe.check_kernel_config, "parse_kconfig",
            return_value={"CONFIG_ARM64_BRBE": "y"}
        ), self.mock.patch.object(
            glt_test_brbe.perf_utils, "read_brbe_capabilities",
            return_value={"armv8_pmuv3_0": 64}
        ), self.mock.patch.object(
            glt_test_brbe.perf_utils, "resolve_perf_command",
            return_value=(None, "No usable perf found")
        ), self.assertRaisesRegex(GLTCore.GLTTestFail, "No usable perf"):
            glt_test_brbe.test_brbe_branch_stack(None, self.mock_logger)

    def test_branch_stack_records_and_decodes_runtime_entries(self):
        """Run the exact record/decode commands and accept typed entries."""
        perf_data_path = (
            "/var/cache/greenlit/session/test_brbe_branch_stack.perf.data")
        mock_run = self._run_runtime_case(
            decode_result=self._completed_process(
                stdout=(
                    "0x1000/0x2000/P/-/-/0/CALL/- "
                    "0x3000/0x4000/M/-/-/4/RET/-\n"
                )
            )
        )

        common_run_args = {
            "check": False,
            "text": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        self.assertEqual(
            mock_run.call_args_list,
            [
                self.mock.call(
                    [
                        "/perf", "record", "-o", perf_data_path,
                        "--no-buildid", "-e", "cycles:u", "-c", "10000",
                        "--branch-filter", "any,save_type,u", "--",
                        "dd", "if=/dev/zero", "of=/dev/null",
                        "bs=64M", "count=8",
                    ],
                    timeout=glt_test_brbe.BRBE_RECORD_TIMEOUT_SECONDS,
                    **common_run_args,
                ),
                self.mock.call(
                    [
                        "/perf", "script", "-i", perf_data_path,
                        "--fields", "brstack",
                    ],
                    timeout=glt_test_brbe.BRBE_DECODE_TIMEOUT_SECONDS,
                    **common_run_args,
                ),
            ],
        )
        self.mock_logger.info.assert_any_call(
            "Decoded %d BRBE branch-stack entries from %s",
            2, perf_data_path,
        )

    def test_branch_stack_fails_when_perf_record_is_rejected(self):
        """Treat a rejected branch-filter record as a Vera failure."""
        with self.assertRaisesRegex(
            GLTCore.GLTTestFail, "perf BRBE record failed"
        ):
            self._run_runtime_case(
                record_result=self._completed_process(
                    returncode=1,
                    stderr="branch stack sampling is not supported",
                )
            )

    def test_branch_stack_fails_when_decode_has_no_entries(self):
        """A successful decode with no branch records still fails."""
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "no typed user-space branch-stack"):
            self._run_runtime_case(
                decode_result=self._completed_process(stdout=""))

    def test_branch_stack_fails_when_decode_exposes_kernel_address(self):
        """Translate a user-only privilege-filter violation into failure."""
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "kernel-space address"):
            self._run_runtime_case(
                decode_result=self._completed_process(
                    stdout=(
                        "0xffff000000001000/0x2000/P/-/-/0/CALL/-\n"
                    )
                )
            )

    def test_branch_stack_fails_on_perf_execution_errors(self):
        """Translate perf launch, timeout, and decode errors into failures."""
        cases = (
            (
                {"record_error": subprocess.TimeoutExpired("perf", 30)},
                "record timed out",
            ),
            ({"record_error": OSError("not executable")},
             "record failed to start"),
            (
                {"decode_error": subprocess.TimeoutExpired("perf", 30)},
                "decode timed out",
            ),
            ({"decode_error": OSError("not executable")},
             "decode failed to start"),
            (
                {
                    "decode_result": self._completed_process(
                        returncode=1, stderr="bad perf.data")
                },
                "decode failed",
            ),
        )

        for arguments, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason), \
                    self.assertRaisesRegex(
                        GLTCore.GLTTestFail, expected_reason):
                self._run_runtime_case(**arguments)
