"""Unit tests for PCI ASPM saved-state validation."""
# pylint: disable=protected-access,wrong-import-position

import sys
from pathlib import Path
from unittest.mock import mock_open

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.glt_core import GLTTestSkip
from src.modules.pci import glt_test_pci


class TestPciAspmSavedState(GLTTestBase):
    """Test PCI ASPM saved-state helper behavior."""

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

    def test_write_aspm_policy_verifies_readback(self):
        """ASPM policy writer verifies that the sysfs write takes effect."""
        policy_file = mock_open(read_data="[powersave] performance\n")
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.open",
            policy_file,
            create=True,
        ):
            glt_test_pci._write_aspm_policy("powersave")

        policy_file.assert_any_call(
            glt_test_pci.ASPM_POLICY_PATH,
            "w",
            encoding="utf-8",
        )
        policy_file.assert_any_call(
            glt_test_pci.ASPM_POLICY_PATH,
            "r",
            encoding="utf-8",
        )

    def test_write_aspm_policy_fails_when_readback_does_not_match(self):
        """ASPM policy writer fails if sysfs readback shows a stale policy."""
        policy_file = mock_open(read_data="powersave [performance]\n")
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.open",
            policy_file,
            create=True,
        ):
            with self.assertRaisesRegex(OSError, "did not take effect"):
                glt_test_pci._write_aspm_policy("powersave")

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
        run_result = self.mock.Mock(returncode=0, stdout="", stderr="")
        with self.mock.patch(
            "src.modules.pci.glt_test_pci._write_aspm_policy"
        ) as write_policy, self.mock.patch(
            "src.modules.pci.glt_test_pci.glt_utils.is_cli_available",
            return_value=True,
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.subprocess.run",
            return_value=run_result,
        ) as run:
            glt_test_pci._cleanup_aspm_saved_state_probe(
                module_loaded=True,
                policy_changed=True,
                original_policy="performance",
                logger=self.mock_logger,
            )

        write_policy.assert_called_once_with("performance")
        run.assert_any_call(
            ["rmmod", glt_test_pci.ASPM_SAVED_STATE_MODULE],
            capture_output=True,
            check=False,
            shell=False,
            text=True,
        )
        run.assert_any_call(["make", "clean"], check=False, shell=False)

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
