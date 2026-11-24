# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PCI ASPM saved-state validation."""
# pylint: disable=protected-access,wrong-import-position

import errno
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import mock_open

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.glt_core import GLTTestFail, GLTTestSkip
from src.modules.pci import glt_test_pci


class TestPciAspmSavedState(GLTTestBase):
    """Test PCI ASPM saved-state helper behavior."""

    def _assert_policy_readback_failure_restores(self, readback_result, message):
        """A completed write requires restoration even when its readback fails."""
        policies = ["performance", "powersave"]
        policy_file = mock_open()
        with self._mock_aspm_policy_test(policy_file, [
            (policies, "performance"), readback_result, (policies, "performance"),
        ]) as (cleanup, read_policy):
            with self.assertRaisesRegex(GLTTestFail, message):
                glt_test_pci.test_pci_aspm_saved_state_update(
                    self.mock_sd, self.mock_logger,
                )
            self.assertEqual({
                "cleanup": cleanup.call_args_list,
                "writes": policy_file.return_value.write.call_args_list,
                "reads": read_policy.call_count,
            }, {
                "cleanup": [self.mock.call(True, True, "performance", self.mock_logger)],
                "writes": [self.mock.call("powersave"), self.mock.call("performance")],
                "reads": 3,
            })
            read_policy.assert_called_with(glt_test_pci.ASPM_POLICY_PATH)

    def _assert_policy_write_preserves_eperm(self, operation):
        """Raw-write failures retain the original operating-system error."""
        error = OSError(
            errno.EPERM, "Operation not permitted", glt_test_pci.ASPM_POLICY_PATH,
        )
        policy_file = mock_open()
        if operation == "open":
            policy_file.side_effect = error
        else:
            getattr(policy_file.return_value, operation).side_effect = error
        with self.mock.patch.object(
            glt_test_pci, "open", policy_file, create=True,
        ):
            with self.assertRaises(OSError) as caught:
                glt_test_pci._write_aspm_policy("powersave")

        self.assertIs(caught.exception, error)
        self.assertEqual(caught.exception.errno, errno.EPERM)
        self.assertEqual(caught.exception.filename, glt_test_pci.ASPM_POLICY_PATH)

    @contextmanager
    def _mock_aspm_cleanup(self, read_result):
        """Track real restore and verification before mocked hardware cleanup."""
        policy_file = mock_open()
        run_result = self.mock.Mock(returncode=0, stdout="", stderr="")
        events = self.mock.Mock()
        events.attach_mock(policy_file.return_value.write, "write")
        with self.mock.patch.object(
            glt_test_pci, "open", policy_file, create=True,
        ), self.mock.patch.object(
            glt_test_pci, "_read_aspm_policy", side_effect=[read_result],
        ) as read_policy, self.mock.patch.object(
            glt_test_pci.glt_utils, "is_cli_available", return_value=True,
        ), self.mock.patch.object(
            glt_test_pci.subprocess, "run", return_value=run_result,
        ) as run:
            events.attach_mock(read_policy, "read")
            events.attach_mock(run, "run")
            yield

        self.assertEqual(events.mock_calls, [
            self.mock.call.write("performance"),
            self.mock.call.read(glt_test_pci.ASPM_POLICY_PATH),
            self.mock.call.run(
                ["rmmod", glt_test_pci.ASPM_SAVED_STATE_MODULE],
                capture_output=True, check=False, shell=False, text=True,
            ),
            self.mock.call.run(["make", "clean"], check=False, shell=False),
        ])

    @contextmanager
    def _mock_aspm_policy_test(self, policy_file, read_results):
        """Mock hardware operations while retaining policy writes and cleanup."""
        original_cwd = Path.cwd()
        run_result = self.mock.Mock(returncode=0, stdout="", stderr="")
        with self.mock.patch.multiple(
            glt_test_pci,
            _check_aspm_saved_state_kernel_config=self.mock.Mock(return_value=[]),
            _read_aspm_probe_hits=self.mock.Mock(return_value={}),
        ), self.mock.patch.object(
            glt_test_pci.os.path, "exists", return_value=True,
        ), self.mock.patch.object(
            glt_test_pci.glt_utils, "is_cli_available", return_value=True,
        ), self.mock.patch.object(
            glt_test_pci.glt_utils, "sign_kernel_module", return_value=True,
        ), self.mock.patch.object(
            glt_test_pci.subprocess, "run", return_value=run_result,
        ) as run, self.mock.patch.object(
            glt_test_pci, "open", policy_file, create=True,
        ), self.mock.patch.object(
            glt_test_pci, "_read_aspm_policy",
            side_effect=read_results,
        ) as read_policy, self.mock.patch.object(
            glt_test_pci, "_cleanup_aspm_saved_state_probe",
            wraps=glt_test_pci._cleanup_aspm_saved_state_probe,
        ) as cleanup:
            yield cleanup, read_policy

        self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(run.call_args_list, [
            self.mock.call(["make"], check=True, shell=False),
            self.mock.call(
                ["insmod", f"{glt_test_pci.ASPM_SAVED_STATE_MODULE}.ko"],
                capture_output=True, shell=False, text=True, check=False,
            ),
            self.mock.call(
                ["rmmod", glt_test_pci.ASPM_SAVED_STATE_MODULE],
                capture_output=True, check=False, shell=False, text=True,
            ),
            self.mock.call(["make", "clean"], check=False, shell=False),
        ])

    def test_outbound_policy_write_eperm_skips_and_cleans(self):
        """A rejected outbound write cannot exercise saved state or need restore."""
        for operation in ("open", "write", "__exit__"):
            with self.subTest(operation=operation):
                policy_file = mock_open()
                error = OSError(
                    errno.EPERM, "Operation not permitted", glt_test_pci.ASPM_POLICY_PATH,
                )
                if operation == "open":
                    policy_file.side_effect = error
                else:
                    getattr(policy_file.return_value, operation).side_effect = error
                with self._mock_aspm_policy_test(policy_file, [
                    (["performance", "powersave"], "performance"),
                ]) as (cleanup, read_policy):
                    with self.assertRaisesRegex(GLTTestSkip, "EPERM.*cannot exercise"):
                        glt_test_pci.test_pci_aspm_saved_state_update(
                            self.mock_sd, self.mock_logger,
                        )
                    cleanup.assert_called_once_with(
                        True, False, "performance", self.mock_logger,
                    )
                    read_policy.assert_called_once_with()

                policy_file.assert_called_once_with(
                    glt_test_pci.ASPM_POLICY_PATH, "w", encoding="utf-8",
                )
                if operation == "open":
                    policy_file.return_value.write.assert_not_called()
                else:
                    policy_file.return_value.write.assert_called_once_with("powersave")

    def test_outbound_policy_write_other_errors_fail_and_clean(self):
        """EACCES and I/O errors must fail rather than mask a broken write as SKIP."""
        for error_number in (errno.EACCES, errno.EIO):
            with self.subTest(errno=error_number):
                policy_file = mock_open()
                policy_file.return_value.write.side_effect = OSError(
                    error_number, "Write failed", glt_test_pci.ASPM_POLICY_PATH,
                )
                with self._mock_aspm_policy_test(policy_file, [
                    (["performance", "powersave"], "performance"),
                ]) as (cleanup, read_policy):
                    with self.assertRaisesRegex(GLTTestFail, "Write failed"):
                        glt_test_pci.test_pci_aspm_saved_state_update(
                            self.mock_sd, self.mock_logger,
                        )
                    cleanup.assert_called_once_with(
                        True, False, "performance", self.mock_logger,
                    )
                    read_policy.assert_called_once_with()
                policy_file.return_value.write.assert_called_once_with("powersave")

    def test_policy_readback_eperm_fails_and_restores(self):
        """Readback failure must remain an error after a successful raw write."""
        error = OSError(errno.EPERM, "Readback denied", glt_test_pci.ASPM_POLICY_PATH)
        self._assert_policy_readback_failure_restores(error, "Readback denied")

    def test_policy_readback_mismatch_fails_and_restores(self):
        """A stale alternate readback still requires restoring the original policy."""
        self._assert_policy_readback_failure_restores(
            (["performance", "powersave"], "performance"), "did not take effect",
        )

    def test_policy_restore_eperm_fails_and_cleanup_retries(self):
        """Once policy changed, a rejected restore is a failure and is retried."""
        error = OSError(errno.EPERM, "Restore denied", glt_test_pci.ASPM_POLICY_PATH)
        policy_file = mock_open()
        policy_file.return_value.write.side_effect = [None, error, error]
        policies = ["performance", "powersave"]
        with self._mock_aspm_policy_test(policy_file, [
            (policies, "performance"), (policies, "powersave"),
        ]) as (cleanup, read_policy):
            with self.assertRaisesRegex(GLTTestFail, "Restore denied"):
                glt_test_pci.test_pci_aspm_saved_state_update(
                    self.mock_sd, self.mock_logger,
                )
            cleanup.assert_called_once_with(
                True, True, "performance", self.mock_logger,
            )
            self.assertEqual(read_policy.call_count, 2)

        self.assertEqual(policy_file.return_value.write.call_args_list, [
            self.mock.call("powersave"),
            self.mock.call("performance"),
            self.mock.call("performance"),
        ])
        self.mock_logger.error.assert_called_once()
        self.assertIn("Failed to restore", self.mock_logger.error.call_args.args[0])

    def test_policy_restore_readback_mismatch_fails_and_cleanup_retries(self):
        """A restore is incomplete until its readback confirms the original policy."""
        policies = ["performance", "powersave"]
        policy_file = mock_open()
        with self._mock_aspm_policy_test(policy_file, [
            (policies, "performance"), (policies, "powersave"),
            (policies, "powersave"), (policies, "performance"),
        ]) as (cleanup, read_policy):
            with self.assertRaisesRegex(GLTTestFail, "did not take effect"):
                glt_test_pci.test_pci_aspm_saved_state_update(
                    self.mock_sd, self.mock_logger,
                )
            cleanup.assert_called_once_with(
                True, True, "performance", self.mock_logger,
            )
            self.assertEqual(read_policy.call_count, 4)
        self.assertEqual(policy_file.return_value.write.call_args_list, [
            self.mock.call("powersave"), self.mock.call("performance"),
            self.mock.call("performance"),
        ])

    def test_policy_write_eperm_propagates_unchanged(self):
        """Open, write and buffered close preserve the original EPERM error."""
        for operation in ("open", "write", "__exit__"):
            with self.subTest(operation=operation):
                self._assert_policy_write_preserves_eperm(operation)

    def test_policy_write_other_errors_propagate_unchanged(self):
        """PermissionError alone and I/O failures do not imply inapplicability."""
        for error_number in (errno.EACCES, errno.EIO):
            with self.subTest(errno=error_number):
                error = OSError(error_number, "Write failed", glt_test_pci.ASPM_POLICY_PATH)
                policy_file = mock_open()
                policy_file.return_value.write.side_effect = error
                with self.mock.patch.object(
                    glt_test_pci, "open", policy_file, create=True,
                ):
                    with self.assertRaises(OSError) as caught:
                        glt_test_pci._write_aspm_policy("powersave")

                self.assertIs(caught.exception, error)

    def test_parse_aspm_policy_content(self):
        """ASPM policy parser returns available policies and current policy."""
        policies, current_policy = glt_test_pci._parse_aspm_policy_content(
            "default [performance] powersave powersupersave\n"
        )

        self.assertEqual(
            policies,
            ["default", "performance", "powersave", "powersupersave"],
        )
        self.assertEqual(current_policy, "performance")

    def test_check_aspm_saved_state_kernel_config_returns_missing_configs(self):
        """Kernel config checker reports missing ASPM saved-state requirements."""
        uname_result = self.mock.Mock(release="test-kernel")
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.platform.uname",
            return_value=uname_result,
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.check_kernel_config.parse_kconfig",
            return_value={"CONFIG_PCIEASPM": "y", "CONFIG_KPROBES": "y"},
        ) as parse_kconfig:
            result = glt_test_pci._check_aspm_saved_state_kernel_config()

        parse_kconfig.assert_called_once_with("/boot/config-test-kernel")
        self.assertEqual(result, ["CONFIG_KRETPROBES"])

    def test_verify_aspm_policy_accepts_matching_readback(self):
        """Verification reads the requested policy path and accepts a matching policy."""
        policy_path = "/test/aspm/policy"
        with self.mock.patch.object(
            glt_test_pci, "_read_aspm_policy",
            return_value=(["powersave", "performance"], "powersave"),
        ) as read_policy:
            glt_test_pci._verify_aspm_policy("powersave", policy_path)

        read_policy.assert_called_once_with(policy_path)

    def test_verify_aspm_policy_fails_when_readback_does_not_match(self):
        """Verification fails if sysfs readback shows a stale policy."""
        with self.mock.patch.object(
            glt_test_pci, "_read_aspm_policy",
            return_value=(["powersave", "performance"], "performance"),
        ) as read_policy:
            with self.assertRaisesRegex(OSError, "did not take effect"):
                glt_test_pci._verify_aspm_policy("powersave")

        read_policy.assert_called_once_with(glt_test_pci.ASPM_POLICY_PATH)

    def test_write_aspm_policy_only_writes_and_closes(self):
        """Raw writes finish before callers decide how to handle verification errors."""
        policy_path = "/test/aspm/policy"
        policy_file = mock_open()
        with self.mock.patch.object(
            glt_test_pci, "open", policy_file, create=True,
        ), self.mock.patch.object(
            glt_test_pci, "_read_aspm_policy",
        ) as read_policy:
            glt_test_pci._write_aspm_policy("powersave", policy_path)

        policy_file.assert_called_once_with(policy_path, "w", encoding="utf-8")
        policy_file.return_value.write.assert_called_once_with("powersave")
        policy_file.return_value.__exit__.assert_called_once_with(None, None, None)
        read_policy.assert_not_called()

    def test_read_aspm_probe_hits_parses_key_value_counters(self):
        """Probe counter parser returns integer key-value data."""
        proc_file = mock_open(
            read_data="config_hits=4\nsaved_state_hits=2\nsaved_state_probe_ret=0\n"
        )
        with self.mock.patch("src.modules.pci.glt_test_pci.open", proc_file, create=True):
            result = glt_test_pci._read_aspm_probe_hits()

        self.assertEqual(
            result,
            {
                "config_hits": 4,
                "saved_state_hits": 2,
                "saved_state_probe_ret": 0,
            },
        )

    def test_cleanup_restores_policy_unloads_module_and_cleans(self):
        """Cleanup restores policy, unloads the module, and cleans build artifacts."""
        with self._mock_aspm_cleanup((["performance", "powersave"], "performance")):
            glt_test_pci._cleanup_aspm_saved_state_probe(
                module_loaded=True,
                policy_changed=True,
                original_policy="performance",
                logger=self.mock_logger,
            )

        self.mock_logger.error.assert_not_called()

    def test_cleanup_logs_policy_verification_failure_and_still_cleans(self):
        """Cleanup must verify restoration and unload/clean even if verification fails."""
        error = OSError(errno.EPERM, "Readback denied", glt_test_pci.ASPM_POLICY_PATH)
        for read_result in (error, (["performance", "powersave"], "powersave")):
            with self.subTest(read_result=read_result):
                self.mock_logger.reset_mock()
                with self._mock_aspm_cleanup(read_result):
                    glt_test_pci._cleanup_aspm_saved_state_probe(
                        module_loaded=True,
                        policy_changed=True,
                        original_policy="performance",
                        logger=self.mock_logger,
                    )
                self.mock_logger.error.assert_called_once()
                message, original_policy, logged_error = self.mock_logger.error.call_args.args
                self.assertEqual(message, "Failed to restore PCIe ASPM policy %s: %s")
                self.assertEqual(original_policy, "performance")
                if isinstance(read_result, OSError):
                    self.assertIs(logged_error, error)
                else:
                    self.assertIn("did not take effect", str(logged_error))

    def test_cleanup_skips_make_clean_when_make_is_missing(self):
        """Cleanup does not fail if the build tool is absent."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.glt_utils.is_cli_available",
            return_value=False,
        ), self.mock.patch("src.modules.pci.glt_test_pci.subprocess.run") as run:
            glt_test_pci._cleanup_aspm_saved_state_probe(
                module_loaded=False,
                policy_changed=False,
                original_policy="performance",
                logger=self.mock_logger,
            )

        run.assert_not_called()

    def test_cleanup_logs_unload_failure(self):
        """Cleanup logs unload failures so stale probes are visible."""
        run_result = self.mock.Mock(returncode=1, stdout="", stderr="module busy")
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.subprocess.run",
            return_value=run_result,
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.glt_utils.is_cli_available",
            return_value=False,
        ):
            glt_test_pci._cleanup_aspm_saved_state_probe(
                module_loaded=True,
                policy_changed=False,
                original_policy="performance",
                logger=self.mock_logger,
            )

        self.mock_logger.error.assert_called_with(
            "Failed to unload %s: %s",
            glt_test_pci.ASPM_SAVED_STATE_MODULE,
            "module busy",
        )

    def test_aspm_saved_state_skips_when_make_is_missing(self):
        """The ASPM saved-state test skips when helper build tools are missing."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.os.path.exists",
            return_value=True,
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci._check_aspm_saved_state_kernel_config",
            return_value=[],
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci._read_aspm_policy",
            return_value=(["default", "performance"], "default"),
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.glt_utils.is_cli_available",
            side_effect=lambda command: command != "make",
        ):
            with self.assertRaisesRegex(GLTTestSkip, "make is unavailable"):
                glt_test_pci.test_pci_aspm_saved_state_update(
                    self.mock_sd,
                    self.mock_logger,
                )

    def test_aspm_saved_state_skips_when_required_kernel_config_is_missing(self):
        """The ASPM saved-state test skips when kernel config prerequisites are absent."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.os.path.exists",
            return_value=True,
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci._check_aspm_saved_state_kernel_config",
            return_value=["CONFIG_KPROBES"],
        ):
            with self.assertRaisesRegex(GLTTestSkip, "CONFIG_KPROBES"):
                glt_test_pci.test_pci_aspm_saved_state_update(
                    self.mock_sd,
                    self.mock_logger,
                )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
