"""Unit tests for Arm CCA firmware-data running-kernel checks."""

# pylint: disable=protected-access

from common import glt_core as GLTCore
from src.modules.virt import glt_test_arm_cca_firmware_data as firmware_data
from tests.test_base import GLTTestBase


class _ArmCcaFirmwareDataTestBase(GLTTestBase):
    """Provide patch cleanup shared by this module's test classes."""

    def _start_patch(self, target, **kwargs):
        """Start a patch that is automatically stopped after each test."""
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()


class TestArmCcaEfiSecretKernelConfig(_ArmCcaFirmwareDataTestBase):
    """Test the EFI secret-area Kconfig qualification."""

    def setUp(self):
        """Set up a Vera kernel with the required config values."""
        super().setUp()
        self.mock_nvdev = self._start_patch(
            "src.modules.virt.glt_test_arm_cca_firmware_data.NvDev"
        )
        self.mock_nvdev.return_value.is_cpu_type.return_value = True
        self.mock_parse_kconfig = self._start_patch(
            "src.modules.virt.glt_test_arm_cca_firmware_data."
            "check_kernel_config.parse_kconfig",
            return_value={
                firmware_data.EFI_SECRET_CONFIG: "y",
                firmware_data.EFI_COCO_SECRET_CONFIG: "y",
                firmware_data.SECURITYFS_CONFIG: "y",
            },
        )
        self.mock_platform = self._start_patch(
            "src.modules.virt.glt_test_arm_cca_firmware_data.platform"
        )
        self.mock_platform.uname.return_value.release = "test-kernel"

    def test_accepts_builtin_or_modular_efi_secret_config(self):
        """EFI secret support may be configured built-in or as a module."""
        for value in ("y", "m"):
            with self.subTest(value=value):
                self.mock_parse_kconfig.return_value[
                    firmware_data.EFI_SECRET_CONFIG
                ] = value

                firmware_data.test_arm_cca_efi_secret_kernel_config(
                    None, self.mock_logger
                )

    def test_fails_when_efi_secret_configs_are_incomplete(self):
        """Every config needed by the EFI secret driver is required."""
        cases = (
            (firmware_data.EFI_SECRET_CONFIG, None),
            (firmware_data.EFI_SECRET_CONFIG, "n"),
            (firmware_data.EFI_COCO_SECRET_CONFIG, None),
            (firmware_data.EFI_COCO_SECRET_CONFIG, "m"),
            (firmware_data.SECURITYFS_CONFIG, None),
            (firmware_data.SECURITYFS_CONFIG, "n"),
        )
        valid_config = self.mock_parse_kconfig.return_value.copy()

        for name, value in cases:
            with self.subTest(name=name, value=value):
                config = valid_config.copy()
                if value is None:
                    config.pop(name)
                else:
                    config[name] = value
                self.mock_parse_kconfig.return_value = config

                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    firmware_data.test_arm_cca_efi_secret_kernel_config(
                        None, self.mock_logger
                    )

                failure = str(context.exception)
                self.assertIn(name, failure)
                self.assertIn(firmware_data.ARM64_EFI_SECRET_COMMIT, failure)

    def test_fails_when_kernel_config_is_unreadable(self):
        """An unreadable exact-release config prevents qualification."""
        self.mock_parse_kconfig.side_effect = PermissionError("denied")

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            firmware_data.test_arm_cca_efi_secret_kernel_config(
                None, self.mock_logger
            )

        failure = str(context.exception)
        self.assertIn("/boot/config-test-kernel", failure)
        self.assertIn("denied", failure)
        self.mock_parse_kconfig.assert_called_once_with(
            "/boot/config-test-kernel"
        )

    def test_passes_with_required_configs(self):
        """A Vera exact-release config with all required values passes."""
        firmware_data.test_arm_cca_efi_secret_kernel_config(
            None, self.mock_logger
        )

        self.mock_parse_kconfig.assert_called_once_with(
            "/boot/config-test-kernel"
        )
        self.mock_logger.info.assert_called_once()

    def test_skips_non_vera_before_config_inspection(self):
        """The config qualification applies only to NVIDIA Vera systems."""
        self.mock_nvdev.return_value.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            firmware_data.test_arm_cca_efi_secret_kernel_config(
                None, self.mock_logger
            )

        self.assertIn("not Vera", str(context.exception))
        self.mock_parse_kconfig.assert_not_called()


class TestArmCcaRealmMappingKernelFingerprint(_ArmCcaFirmwareDataTestBase):
    """Test the running-kernel RSI symbol fingerprint."""

    def setUp(self):
        """Set up a Vera kernel exposing the patched RSI symbol."""
        super().setUp()
        self.mock_nvdev = self._start_patch(
            "src.modules.virt.glt_test_arm_cca_firmware_data.NvDev"
        )
        self.mock_nvdev.return_value.is_cpu_type.return_value = True
        self.mock_symbols = self._start_patch(
            "src.modules.virt.glt_test_arm_cca_firmware_data."
            "_read_kernel_text_symbols",
            return_value={firmware_data.PATCHED_RSI_SYMBOL},
        )

    def test_fails_when_required_rsi_symbol_is_absent(self):
        """A readable image without the patched symbol is not qualified."""
        self.mock_symbols.return_value = set()

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            firmware_data.test_arm_cca_realm_encrypted_mapping_kernel_fingerprint(
                None, self.mock_logger
            )

        failure = str(context.exception)
        self.assertIn(firmware_data.PATCHED_RSI_SYMBOL, failure)
        self.assertIn(firmware_data.REALM_ENCRYPTED_MAPPING_COMMIT, failure)

    def test_fails_with_prepatch_rsi_diagnosis(self):
        """The legacy-only export identifies the known prepatch state."""
        self.mock_symbols.return_value = {firmware_data.PREPATCH_RSI_SYMBOL}

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            firmware_data.test_arm_cca_realm_encrypted_mapping_kernel_fingerprint(
                None, self.mock_logger
            )

        failure = str(context.exception)
        self.assertIn("prepatch symbol", failure)
        self.assertIn(firmware_data.PREPATCH_RSI_SYMBOL, failure)

    def test_passes_when_both_symbols_are_exported(self):
        """The new fingerprint remains valid when a legacy export is retained."""
        self.mock_symbols.return_value = {
            firmware_data.PATCHED_RSI_SYMBOL,
            firmware_data.PREPATCH_RSI_SYMBOL,
        }

        firmware_data.test_arm_cca_realm_encrypted_mapping_kernel_fingerprint(
            None, self.mock_logger
        )

        self.mock_logger.warning.assert_called_once()
        self.assertIn(
            "cannot determine",
            self.mock_logger.warning.call_args.args[0],
        )

    def test_passes_with_required_rsi_symbol(self):
        """The patched running-kernel fingerprint passes."""
        firmware_data.test_arm_cca_realm_encrypted_mapping_kernel_fingerprint(
            None, self.mock_logger
        )

        self.mock_symbols.assert_called_once_with()
        self.mock_logger.info.assert_called_once_with(
            "Running kernel exposes the Arm CCA RSI symbol: %s",
            firmware_data.PATCHED_RSI_SYMBOL,
        )

    def test_skips_non_vera_before_symbol_inspection(self):
        """The RSI fingerprint applies only to NVIDIA Vera systems."""
        self.mock_nvdev.return_value.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            firmware_data.test_arm_cca_realm_encrypted_mapping_kernel_fingerprint(
                None, self.mock_logger
            )

        self.assertIn("not Vera", str(context.exception))
        self.mock_symbols.assert_not_called()

    def test_skips_when_kallsyms_is_unobservable(self):
        """Unavailable or malformed kallsyms is an observability gap."""
        for error in (PermissionError("denied"), ValueError("malformed")):
            with self.subTest(error=type(error).__name__):
                self.mock_symbols.side_effect = error

                with self.assertRaises(GLTCore.GLTTestSkip) as context:
                    firmware_data.test_arm_cca_realm_encrypted_mapping_kernel_fingerprint(
                        None, self.mock_logger
                    )

                self.assertIn(
                    "Cannot inspect running-kernel symbols",
                    str(context.exception),
                )
                self.mock_symbols.reset_mock(side_effect=True)


class TestKallsymsReader(GLTTestBase):
    """Test exact parsing of running-kernel text symbols."""

    def test_extracts_only_exact_text_symbol_names(self):
        """Addresses and similarly named metadata cannot satisfy the check."""
        kallsyms = (
            "0000000000000000 T arm64_rsi_is_protected\n"
            "0000000000000000 t local_text\n"
            "0000000000000000 T module_text\t[arm_cca]\n"
            "0000000000000000 R __ksymtab_arm64_rsi_is_protected\n"
            "not-an-address T misleading_symbol\n"
            "malformed\n"
        )
        path = self.mock.MagicMock()
        path.open.return_value.__enter__.return_value = kallsyms.splitlines(True)

        symbols = firmware_data._read_kernel_text_symbols(path)

        self.assertEqual(
            symbols,
            {"arm64_rsi_is_protected", "local_text", "module_text"},
        )
        path.open.assert_called_once_with(encoding="utf-8")

    def test_rejects_a_file_without_symbol_records(self):
        """An empty or malformed file cannot prove kernel observability."""
        path = self.mock.MagicMock()
        path.open.return_value.__enter__.return_value = iter(("", "malformed\n"))

        with self.assertRaises(ValueError):
            firmware_data._read_kernel_text_symbols(path)


class TestTestCaseDiscovery(GLTTestBase):
    """Test test-case identity and plans."""

    def test_test_case_identities_and_plans(self):
        """Discovery exposes two focused virt tests in every standard plan."""
        expected_names = {
            firmware_data.test_arm_cca_efi_secret_kernel_config:
                "virt:test_arm_cca_efi_secret_kernel_config",
            firmware_data.test_arm_cca_realm_encrypted_mapping_kernel_fingerprint:
                "virt:test_arm_cca_realm_encrypted_mapping_kernel_fingerprint",
        }

        for test_case, expected_name in expected_names.items():
            with self.subTest(test_case=test_case.__name__):
                self.assertEqual(
                    set(test_case.plans), {"standard", "stress"}
                )
                self.assertEqual(
                    GLTCore.GLTScheduler.get_test_case_name(test_case),
                    expected_name,
                )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
