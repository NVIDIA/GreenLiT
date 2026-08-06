"""Unit tests for NVIDIA Olympus T410-OLY-1027 erratum coverage."""

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_erratum_olympus_1027 as olympus_1027


class TestNvidiaOlympus1027Erratum(GLTTestBase):
    """Test NVIDIA Olympus T410-OLY-1027 erratum checks."""

    def setUp(self):
        """Set up a patched Vera kernel by default."""
        super().setUp()
        self.mock_nvdev = self._start_patch(
            "src.modules.arm64.glt_test_erratum_olympus_1027.NvDev"
        )
        self.mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        self.mock_parse_kconfig = self._start_patch(
            "src.modules.arm64.glt_test_erratum_olympus_1027."
            "check_kernel_config.parse_kconfig",
            return_value={olympus_1027.ERRATUM_CONFIG: "y"},
        )
        self.mock_kernel_log = self._start_patch(
            "src.modules.arm64.glt_test_erratum_olympus_1027.glt_utils.check_kernel_log",
            return_value=True,
        )
        self.mock_platform = self._start_patch(
            "src.modules.arm64.glt_test_erratum_olympus_1027.platform"
        )
        self.mock_platform.uname.return_value.release = "test-kernel"

    def _start_patch(self, target, **kwargs):
        """Start a patch that is automatically stopped after each test."""
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_fails_when_kernel_config_is_missing(self):
        """A missing boot config is a hard failure with patch guidance."""
        self.mock_parse_kconfig.side_effect = FileNotFoundError(
            "/boot/config-test-kernel"
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            olympus_1027.test_nvidia_olympus_1027_erratum(None, self.mock_logger)

        self.assertIn("Kernel config not found", str(context.exception))
        self.assertIn(olympus_1027.FIX_COMMIT, str(context.exception))
        self.assertIn("runtime cpucap marker", str(context.exception))
        self.mock_kernel_log.assert_not_called()

    def test_fails_when_kernel_config_is_not_enabled(self):
        """CONFIG_NVIDIA_OLYMPUS_1027_ERRATUM must be built in."""
        for value in (None, "n", "m"):
            with self.subTest(value=value):
                self.mock_parse_kconfig.return_value = (
                    {} if value is None else {olympus_1027.ERRATUM_CONFIG: value}
                )

                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    olympus_1027.test_nvidia_olympus_1027_erratum(
                        None, self.mock_logger
                    )

                self.assertIn(olympus_1027.ERRATUM_CONFIG, str(context.exception))
                self.assertIn(olympus_1027.FIX_COMMIT, str(context.exception))
                self.assertIn("runtime cpucap marker", str(context.exception))

    def test_fails_when_runtime_erratum_log_is_missing(self):
        """The detected CPU capability should be visible in current boot logs."""
        self.mock_kernel_log.return_value = False

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            olympus_1027.test_nvidia_olympus_1027_erratum(None, self.mock_logger)

        self.assertIn("Runtime erratum print not found", str(context.exception))
        self.assertIn(olympus_1027.RUNTIME_MSG, str(context.exception))
        self.assertIn(olympus_1027.FIX_COMMIT, str(context.exception))
        self.assertIn("runtime cpucap marker", str(context.exception))

    def test_fails_when_kernel_log_retrieval_fails(self):
        """Unreadable kernel logs are reported separately from missing signatures."""
        self.mock_kernel_log.side_effect = RuntimeError("dmesg access denied")

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            olympus_1027.test_nvidia_olympus_1027_erratum(None, self.mock_logger)

        self.assertIn("Failed to retrieve Olympus 1027", str(context.exception))
        self.assertIn("dmesg access denied", str(context.exception))

    def test_passes_when_config_and_runtime_log_are_present(self):
        """A Vera system with config and runtime detection passes."""
        olympus_1027.test_nvidia_olympus_1027_erratum(None, self.mock_logger)

        self.mock_parse_kconfig.assert_called_once_with("/boot/config-test-kernel")
        self.mock_kernel_log.assert_called_once_with(olympus_1027.RUNTIME_MSG)

    def test_skips_non_vera_systems(self):
        """The erratum is scoped to NVIDIA Vera/Olympus systems."""
        self.mock_nvdev.return_value.is_cpu_type.side_effect = None
        self.mock_nvdev.return_value.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            olympus_1027.test_nvidia_olympus_1027_erratum(None, self.mock_logger)

        self.assertIn("not an NVIDIA Vera CPU", str(context.exception))
        self.mock_parse_kconfig.assert_not_called()
        self.mock_kernel_log.assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
