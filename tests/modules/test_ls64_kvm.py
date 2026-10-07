# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Vera LS64 userspace ABI checks."""
# pylint: disable=wrong-import-position,protected-access

import sys
import subprocess
import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_ls64_kvm as ls64_kvm


class TestVeraLs64UserspaceAbi(GLTTestBase):
    """Validate DGX-16333 helper and runtime decision logic."""

    def test_read_cpu_features_merges_feature_lines(self):
        """CPU feature parsing is case-insensitive and accumulates all CPUs."""
        mock_open = self.mock.mock_open(
            read_data=(
                "processor\t: 0\n"
                "Features\t: fp asimd LS64\n\n"
                "processor\t: 1\n"
                "Features\t: fp aes\n"
            )
        )
        with self.mock.patch("builtins.open", mock_open):
            features = ls64_kvm._read_cpu_features()

        self.assertIn("ls64", features)
        self.assertIn("aes", features)

    def test_vera_ls64_userspace_abi_has_no_broad_kselftest_dependency(self):
        """The LS64 gate is not blocked by unrelated broad KVM selftest setup."""
        test_func = ls64_kvm.test_vera_ls64_userspace_abi
        while hasattr(test_func, "__wrapped__"):
            self.assertFalse(hasattr(test_func, "requires"))
            test_func = test_func.__wrapped__
        self.assertFalse(hasattr(test_func, "requires"))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_skips_non_vera(self, mock_nvdev):
        """The runtime test only applies to Vera."""
        mock_nvdev.return_value.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip):
            ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_fails_missing_cpuinfo_ls64(self, mock_nvdev):
        """Vera must expose ls64 in /proc/cpuinfo."""
        mock_nvdev.return_value.is_cpu_type.return_value = True

        with self.mock.patch.object(ls64_kvm, "_read_cpu_features", return_value=set()):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)

        self.assertIn("does not expose", str(context.exception))
        self.assertIn(ls64_kvm.LS64_PATCH_SERIES, str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_fails_missing_hwcap(self, mock_nvdev):
        """Vera must expose the LS64 HWCAP bit."""
        mock_nvdev.return_value.is_cpu_type.return_value = True

        with self.mock.patch.object(
            ls64_kvm,
            "_read_cpu_features",
            return_value={ls64_kvm.LS64_CPUINFO_FEATURE},
        ), self.mock.patch.object(ls64_kvm, "_getauxval", return_value=0):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)

        self.assertIn("HWCAP3_LS64", str(context.exception))
        self.assertIn(ls64_kvm.LS64_PATCH_SERIES, str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm._getauxval")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.arm64.glt_test_ls64_kvm._read_cpu_features"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_passes_when_capabilities_are_present(
        self, mock_nvdev, mock_features, mock_getauxval
    ):
        """The pass path is source-tree independent."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_features.return_value = {ls64_kvm.LS64_CPUINFO_FEATURE}
        mock_getauxval.return_value = ls64_kvm.HWCAP3_LS64

        ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)
        self.mock_logger.info.assert_called_once_with(
            "Vera LS64 userspace ABI checks passed."
        )


class TestLs64NativeMetadata(GLTTestBase):
    """Verify plans and dependency metadata for native LS64 runtime tests."""

    def test_runtime_case_plans_and_userspace_abi_dependency(self):
        """Runtime cases use standard/stress plans and require the ABI check."""
        cases = (
            ls64_kvm.test_vera_ls64_normal_memory,
            ls64_kvm.test_vera_ls64_alignment,
            ls64_kvm.test_vera_ls64_kvm_exit,
        )
        for test_case in cases:
            with self.subTest(test_case=test_case.__name__):
                self.assertEqual(
                    set(test_case.plans), {"standard", "stress"}
                )
                self.assertIn(
                    "arm64:test_vera_ls64_userspace_abi",
                    test_case.__wrapped__.requires.all_true,
                )


class TestLs64NativeProbe(GLTTestBase):
    """Verify native probe build, invocation, and cleanup."""

    def test_execute_probe_uses_common_kvm_source_and_removes_binary(self):
        """Build one mode in the session without leaving a native artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = self.mock.Mock(log_dir=temp_dir)
            probe_path = Path(temp_dir) / "test_case.probe"

            def complete_build(command, *_args, **_kwargs):
                probe_path.write_text("binary", encoding="ascii")
                probe_path.chmod(0o755)
                return subprocess.CompletedProcess(command, 0)

            with self.mock.patch.object(
                ls64_kvm.glt_utils, "is_cli_available", return_value=True
            ), self.mock.patch.object(
                ls64_kvm.glt_utils, "run_command", side_effect=complete_build
            ) as mock_build, self.mock.patch.object(
                ls64_kvm.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [str(probe_path), "normal-memory"],
                    ls64_kvm.LS64_PROBE_SIGBUS,
                    stderr="SIGBUS",
                ),
            ) as mock_run:
                result, reason = ls64_kvm._execute_ls64_probe(
                    scheduler, "test_case", "normal-memory", self.mock_logger
                )

            self.assertEqual(reason, "")
            self.assertEqual(result.returncode, ls64_kvm.LS64_PROBE_SIGBUS)
            self.assertFalse(probe_path.exists())
            self.assertEqual(
                mock_build.call_args.args[0],
                [
                    "make", "-C", str(ls64_kvm.LS64_PROBE_SOURCE_DIR),
                    f"OUTPUT={probe_path}",
                    f"KVM_MICROVM_DIR={ls64_kvm.KVM_MICROVM_SOURCE_DIR}",
                ],
            )
            self.assertEqual(
                mock_run.call_args.args[0],
                [str(probe_path), "normal-memory"],
            )

    def test_execute_probe_reports_build_and_execution_errors(self):
        """Native failures return reasons for decorated tests to report."""
        scheduler = self.mock.Mock(log_dir="/session")
        for expected in ("compiler failed", "timed out"):
            with self.subTest(expected=expected), self.mock.patch.object(
                ls64_kvm.glt_utils, "is_cli_available", return_value=True
            ), self.mock.patch.object(
                ls64_kvm.Path, "is_file", return_value=True
            ), self.mock.patch.object(
                ls64_kvm.glt_utils, "run_command",
                side_effect=RuntimeError(expected),
            ):
                result, reason = ls64_kvm._execute_ls64_probe(
                    scheduler, "test_case", "normal-memory", self.mock_logger
                )

            self.assertIsNone(result)
            self.assertIn(expected, reason)

        with self.mock.patch.object(
            ls64_kvm.glt_utils, "is_cli_available", return_value=True
        ), self.mock.patch.object(
            ls64_kvm.Path, "is_file", return_value=True
        ), self.mock.patch.object(
            ls64_kvm.Path, "unlink"
        ), self.mock.patch.object(
            ls64_kvm.glt_utils,
            "run_command",
            return_value=subprocess.CompletedProcess(["make"], 0),
        ), self.mock.patch.object(
            ls64_kvm.os, "access", return_value=True
        ), self.mock.patch.object(
            ls64_kvm.subprocess, "run", side_effect=OSError("permission denied")
        ):
            result, reason = ls64_kvm._execute_ls64_probe(
                scheduler, "test_case", "normal-memory", self.mock_logger
            )

        self.assertIsNone(result)
        self.assertIn("permission denied", reason)


class TestLs64NativeResults(GLTTestBase):
    """Verify native GreenLiT result mapping for every LS64 mode."""

    def setUp(self):
        super().setUp()
        self.mock_sd.log_dir = "/session"

    def test_normal_memory_accepts_sigbus_or_success(self):
        """Both base LS64 and optional LS64_WB behavior are valid."""
        for returncode in (0, ls64_kvm.LS64_PROBE_SIGBUS):
            result = subprocess.CompletedProcess(["probe"], returncode)
            with self.subTest(returncode=returncode), self.mock.patch.object(
                ls64_kvm, "_execute_ls64_probe", return_value=(result, "")
            ):
                ls64_kvm.test_vera_ls64_normal_memory(
                    self.mock_sd, self.mock_logger
                )

    def test_normal_memory_rejects_sigill_and_probe_errors(self):
        """Reported LS64 support cannot coexist with SIGILL or setup failure."""
        cases = (
            (ls64_kvm.LS64_PROBE_SIGILL, "SIGILL"),
            (1, "unexpected status"),
        )
        for returncode, expected in cases:
            result = subprocess.CompletedProcess(["probe"], returncode)
            with self.subTest(returncode=returncode), self.mock.patch.object(
                ls64_kvm, "_execute_ls64_probe", return_value=(result, "")
            ), self.assertRaisesRegex(GLTCore.GLTTestFail, expected):
                ls64_kvm.test_vera_ls64_normal_memory(
                    self.mock_sd, self.mock_logger
                )

    def test_alignment_skips_without_ls64_wb(self):
        """An aligned SIGBUS makes the misalignment oracle unobservable."""
        normal = subprocess.CompletedProcess(
            ["probe"], ls64_kvm.LS64_PROBE_SIGBUS
        )
        with self.mock.patch.object(
            ls64_kvm, "_execute_ls64_probe", return_value=(normal, "")
        ), self.assertRaisesRegex(GLTCore.GLTTestSkip, "LS64_WB"):
            ls64_kvm.test_vera_ls64_alignment(
                self.mock_sd, self.mock_logger
            )

    def test_alignment_requires_misaligned_sigbus(self):
        """When normal memory works, only SIGBUS proves alignment checking."""
        normal = subprocess.CompletedProcess(["probe"], 0)
        misaligned = subprocess.CompletedProcess(
            ["probe"], ls64_kvm.LS64_PROBE_SIGBUS
        )
        with self.mock.patch.object(
            ls64_kvm, "_execute_ls64_probe",
            side_effect=((normal, ""), (misaligned, "")),
        ):
            ls64_kvm.test_vera_ls64_alignment(
                self.mock_sd, self.mock_logger
            )

        with self.mock.patch.object(
            ls64_kvm, "_execute_ls64_probe",
            side_effect=((normal, ""), (normal, "")),
        ), self.assertRaisesRegex(GLTCore.GLTTestFail, "did not raise SIGBUS"):
            ls64_kvm.test_vera_ls64_alignment(
                self.mock_sd, self.mock_logger
            )

    def test_kvm_exit_requires_access_and_success(self):
        """The KVM case fails closed on access or harness errors."""
        result = subprocess.CompletedProcess(["probe"], 0)
        with self.mock.patch.object(
            ls64_kvm.os, "access", return_value=True
        ), self.mock.patch.object(
            ls64_kvm, "_execute_ls64_probe", return_value=(result, "")
        ):
            ls64_kvm.test_vera_ls64_kvm_exit(
                self.mock_sd, self.mock_logger
            )

        with self.mock.patch.object(
            ls64_kvm.os, "access", return_value=False
        ), self.assertRaisesRegex(GLTCore.GLTTestFail, "/dev/kvm"):
            ls64_kvm.test_vera_ls64_kvm_exit(
                self.mock_sd, self.mock_logger
            )

        failed = subprocess.CompletedProcess(["probe"], 1, stderr="bad exit")
        with self.mock.patch.object(
            ls64_kvm.os, "access", return_value=True
        ), self.mock.patch.object(
            ls64_kvm, "_execute_ls64_probe", return_value=(failed, "")
        ), self.assertRaisesRegex(GLTCore.GLTTestFail, "bad exit"):
            ls64_kvm.test_vera_ls64_kvm_exit(
                self.mock_sd, self.mock_logger
            )

    def test_kvm_setup_and_run_errors_do_not_claim_missing_patches(self):
        """Generic nonzero probe statuses carry diagnostics, not provenance."""
        errors = (
            "sysconf(_SC_PAGESIZE) failed",
            "mmap(guest memory) failed",
            "KVM_CREATE_VM failed",
            "KVM_RUN timed out after 5 seconds",
        )
        for error in errors:
            with self.subTest(error=error), self.mock.patch.object(
                ls64_kvm.os, "access", return_value=True
            ), self.mock.patch.object(
                ls64_kvm, "_execute_ls64_probe",
                return_value=(
                    subprocess.CompletedProcess(["probe"], 1, "", error), ""
                ),
            ):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    ls64_kvm.test_vera_ls64_kvm_exit(
                        self.mock.Mock(), self.mock_logger
                    )
                reason = str(context.exception)
                self.assertIn(error, reason)
                self.assertNotIn("f174a9ffcd48", reason)
                self.assertNotIn("151b92c92a45", reason)

    def test_kvm_preserves_the_probes_observed_exit_patch_guidance(self):
        """An exit-specific diagnostic from the C probe remains actionable."""
        diagnostic = (
            "Expected KVM_EXIT_ARM_LDST64B (42); check kernel patches "
            "f174a9ffcd48 and 151b92c92a45"
        )
        with self.mock.patch.object(
            ls64_kvm.os, "access", return_value=True
        ), self.mock.patch.object(
            ls64_kvm, "_execute_ls64_probe",
            return_value=(
                subprocess.CompletedProcess(["probe"], 1, "", diagnostic), ""
            ),
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                ls64_kvm.test_vera_ls64_kvm_exit(self.mock.Mock(), self.mock_logger)
        reason = str(context.exception)
        self.assertIn(diagnostic, reason)
        self.assertNotIn("consider checking kernel patches", reason)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
