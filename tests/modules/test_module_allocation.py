# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the arm64 module-allocation observer's applicability."""

from types import SimpleNamespace
from unittest import mock

from tests.test_base import GLTTestBase
from src.modules.mm import glt_test_module as sut


class TestModuleAllocation(GLTTestBase):
    """Keep eligibility before effects and preserve the supported Arm contract."""

    def setUp(self):
        """Provide healthy prerequisites without running any host operations."""
        super().setUp()
        attr = sut.test_module_allocation.attr
        self.addCleanup(setattr, attr, "result", attr.result)
        patchers = {
            "machine": mock.patch.object(sut.platform, "machine", return_value="aarch64"),
            "release": mock.patch.object(sut.platform, "release", return_value="6.8.0"),
            "os_info": mock.patch.object(
                sut.NvDev, "get_release_info", return_value={"NAME": "Ubuntu"}
            ),
            "kernel_log": mock.patch.object(
                sut.glt_utils,
                "get_kernel_log",
                return_value=(
                    "131072 pages in range for non-PLT usage\n"
                    "131072 pages in range for PLT usage\n"
                ),
            ),
            "page_size": mock.patch.object(
                sut.glt_utils, "run_command", return_value=SimpleNamespace(stdout="4096")
            ),
            "chdir": mock.patch.object(sut.glt_utils, "chdir_managed"),
            "run": mock.patch.object(sut.subprocess, "run"),
            "sign": mock.patch.object(
                sut.glt_utils, "sign_kernel_module", return_value=True
            ),
        }
        self.boundaries = {}
        for name, patcher in patchers.items():
            self.boundaries[name] = patcher.start()
            self.addCleanup(patcher.stop)
        lifecycle = mock.Mock()
        lifecycle.attach_mock(self.boundaries["run"], "run")
        lifecycle.attach_mock(self.boundaries["sign"], "sign")
        self.boundaries["lifecycle"] = lifecycle

    def assert_module_not_started(self):
        """A prerequisite outcome must not start or clean a kernel module."""
        for name in ("chdir", "run", "sign"):
            self.boundaries[name].assert_not_called()

    def test_non_arm64_skips_before_prerequisites(self):
        """Unsupported hosts must not reach even OS, log or build prerequisites."""
        self.boundaries["machine"].return_value = "x86_64"

        with self.assertRaises(sut.GLTCore.GLTTestSkip) as skipped:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertEqual(
            str(skipped.exception),
            "Requires arm64 module-VA allocation support from kernel patch 3e35d303ab7d.",
        )
        for name in ("release", "os_info", "kernel_log", "page_size"):
            self.boundaries[name].assert_not_called()
        self.assert_module_not_started()

    def test_arm64_uname_aliases_run_module_va_check(self):
        """arm64, ARM64, and aarch64_be use the same module-VA check as aarch64."""
        for machine in ("arm64", "ARM64", "aarch64_be"):
            with self.subTest(machine=machine):
                self.boundaries["machine"].return_value = machine
                self.boundaries["kernel_log"].reset_mock()
                sut.test_module_allocation(None, self.mock_logger)
                self.boundaries["kernel_log"].assert_called_once_with()

    def test_arm64_eligible_runs_module_lifecycle(self):
        """Exactly one GiB of observed range must retain the signed lifecycle."""
        sut.test_module_allocation(None, self.mock_logger)

        self.boundaries["kernel_log"].assert_called_once_with()
        self.boundaries["page_size"].assert_called_once_with(
            ["getconf", "PAGE_SIZE"], "Failed to retrieve the system page size"
        )
        self.boundaries["chdir"].assert_called_once()
        self.assertEqual(
            self.boundaries["lifecycle"].mock_calls,
            [
                mock.call.run(["make"], check=True, shell=False),
                mock.call.sign("bingo.ko", self.mock_logger),
                mock.call.run(["insmod", "bingo.ko"], check=True, shell=False),
                mock.call.run(["rmmod", "bingo"], check=True, shell=False),
                mock.call.run(["make", "clean"], check=True, shell=False),
            ],
        )

    def test_arm64_zero_direct_window_completes_module_lifecycle(self):
        """Enough PLT range permits normal completion with no direct window."""
        self.boundaries["kernel_log"].return_value = (
            "Booting Linux\n"
            "0 pages in range for non-PLT usage\n"
            "Initializing kernel subsystems\n"
            "262144 pages in range for PLT usage\n"
            "Kernel initialization complete\n"
        )

        result = sut.test_module_allocation(None, self.mock_logger)

        self.assertIsNone(result)
        self.boundaries["kernel_log"].assert_called_once_with()
        self.boundaries["page_size"].assert_called_once_with(
            ["getconf", "PAGE_SIZE"], "Failed to retrieve the system page size"
        )
        self.boundaries["chdir"].assert_called_once()
        self.assertEqual(
            self.boundaries["lifecycle"].mock_calls,
            [
                mock.call.run(["make"], check=True, shell=False),
                mock.call.sign("bingo.ko", self.mock_logger),
                mock.call.run(["insmod", "bingo.ko"], check=True, shell=False),
                mock.call.run(["rmmod", "bingo"], check=True, shell=False),
                mock.call.run(["make", "clean"], check=True, shell=False),
            ],
        )

    def test_arm64_unsupported_os_skips_before_module_operations(self):
        """The existing Ubuntu restriction remains a prerequisite SKIP."""
        self.boundaries["os_info"].return_value = {"NAME": "Fedora"}

        with self.assertRaises(sut.GLTCore.GLTTestSkip) as skipped:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertEqual(
            str(skipped.exception), "Test requires Ubuntu with kernel version 6.5 or later."
        )
        self.boundaries["kernel_log"].assert_not_called()
        self.assert_module_not_started()

    def test_arm64_old_kernel_skips_before_module_operations(self):
        """A kernel older than the patch still skips without a build or load."""
        self.boundaries["release"].return_value = "6.4.0"

        with self.assertRaises(sut.GLTCore.GLTTestSkip) as skipped:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertEqual(
            str(skipped.exception), "Test requires Ubuntu with kernel version 6.5 or later."
        )
        self.boundaries["kernel_log"].assert_not_called()
        self.assert_module_not_started()

    def test_arm64_missing_non_plt_telemetry_warns(self):
        """An unobserved non-PLT window must warn before module operations."""
        self.boundaries["kernel_log"].return_value = "262144 pages in range for PLT usage\n"

        with self.assertRaises(sut.GLTCore.GLTTestWarn) as warned:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertIs(type(warned.exception), sut.GLTCore.GLTTestWarn)
        self.assertEqual(
            str(warned.exception),
            "Non-PLT module allocation telemetry not found in kernel log.",
        )
        self.boundaries["page_size"].assert_not_called()
        self.assert_module_not_started()

    def test_arm64_missing_plt_telemetry_warns(self):
        """An unobserved PLT window must warn before module operations."""
        self.boundaries["kernel_log"].return_value = "262144 pages in range for non-PLT usage\n"

        with self.assertRaises(sut.GLTCore.GLTTestWarn) as warned:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertIs(type(warned.exception), sut.GLTCore.GLTTestWarn)
        self.assertEqual(
            str(warned.exception),
            "PLT module allocation telemetry not found in kernel log.",
        )
        self.boundaries["page_size"].assert_not_called()
        self.assert_module_not_started()

    def test_arm64_region_below_one_gib_fails(self):
        """A region one page below one GiB must fail without loading a module."""
        self.boundaries["kernel_log"].return_value = (
            "131071 pages in range for non-PLT usage\n"
            "131072 pages in range for PLT usage\n"
        )

        with self.assertRaises(sut.GLTCore.GLTTestFail) as failed:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertEqual(
            str(failed.exception),
            "Module allocation region is less than 1GiB: "
            "non-PLT pages=131071, PLT pages=131072, page size=4096 bytes.",
        )
        self.assert_module_not_started()

    def test_arm64_kernel_log_error_remains_test_failure(self):
        """Kernel-log errors must become GLT failures with their existing reason."""
        self.boundaries["kernel_log"].side_effect = RuntimeError("kernel log unavailable")

        with self.assertRaises(sut.GLTCore.GLTTestFail) as failed:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertEqual(
            str(failed.exception), "Failed to collect kernel logs: kernel log unavailable"
        )
        self.boundaries["page_size"].assert_not_called()
        self.assert_module_not_started()

    def test_arm64_signing_rejection_remains_test_failure(self):
        """A signing failure must fail after make, before load or cleanup."""
        self.boundaries["sign"].return_value = False

        with self.assertRaises(sut.GLTCore.GLTTestFail) as failed:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertEqual(
            str(failed.exception),
            "Cannot sign bingo.ko for Secure Boot. "
            "See docs/user-guides/secure-boot.md for setup.",
        )
        self.assertEqual(
            self.boundaries["lifecycle"].mock_calls,
            [
                mock.call.run(["make"], check=True, shell=False),
                mock.call.sign("bingo.ko", self.mock_logger),
            ],
        )

    def test_arm64_insmod_error_remains_test_failure(self):
        """A load error must fail without inventing unload or cleanup success."""
        error = sut.subprocess.CalledProcessError(1, ["insmod", "bingo.ko"])
        self.boundaries["run"].side_effect = [None, error]

        with self.assertRaises(sut.GLTCore.GLTTestFail) as failed:
            sut.test_module_allocation(None, self.mock_logger)

        self.assertEqual(str(failed.exception), f"Error executing command: {error}")
        self.assertEqual(
            self.boundaries["lifecycle"].mock_calls,
            [
                mock.call.run(["make"], check=True, shell=False),
                mock.call.sign("bingo.ko", self.mock_logger),
                mock.call.run(["insmod", "bingo.ko"], check=True, shell=False),
            ],
        )
