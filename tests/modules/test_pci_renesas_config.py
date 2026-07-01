"""Unit tests for the Renesas xHCI PCI config testcase."""
# pylint: disable=wrong-import-position

import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.pci import glt_test_pci


class TestRenesasXhciPciConfig(GLTTestBase):
    """Test DGX-11145 Renesas xHCI PCI config behavior."""

    def setUp(self):
        """Set up common mocks."""
        super().setUp()
        self.nvdev_patcher = self.mock.patch("src.modules.pci.glt_test_pci.NvDev")
        self.mock_nvdev = self.nvdev_patcher.start()
        self.mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"

    def tearDown(self):
        """Clean up common mocks."""
        self.nvdev_patcher.stop()
        super().tearDown()

    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.platform")
    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.check_kernel_config")
    def test_accepts_builtin_or_module(self, mock_check_kernel_config, mock_platform):
        """CONFIG_USB_XHCI_PCI_RENESAS=y or =m passes."""
        mock_platform.release.return_value = "test-kernel"

        for value in ("y", "m"):
            with self.subTest(value=value):
                mock_check_kernel_config.parse_kconfig.return_value = {
                    "CONFIG_USB_XHCI_PCI_RENESAS": value
                }

                glt_test_pci.test_renesas_xhci_pci_config(None, None)

                mock_check_kernel_config.parse_kconfig.assert_called_once_with(
                    "/boot/config-test-kernel"
                )
                mock_check_kernel_config.parse_kconfig.reset_mock()

    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.platform")
    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.check_kernel_config")
    def test_fails_when_disabled_or_missing(self, mock_check_kernel_config, mock_platform):
        """Missing or disabled Renesas xHCI PCI support fails validation."""
        mock_platform.release.return_value = "test-kernel"
        cases = [
            ({}, "found None"),
            ({"CONFIG_USB_XHCI_PCI_RENESAS": "n"}, "found 'n'"),
        ]

        for kconfig, expected in cases:
            with self.subTest(kconfig=kconfig):
                mock_check_kernel_config.parse_kconfig.return_value = kconfig

                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    glt_test_pci.test_renesas_xhci_pci_config(None, None)

                self.assertIn("CONFIG_USB_XHCI_PCI_RENESAS", str(context.exception))
                self.assertIn(expected, str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.platform")
    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.check_kernel_config")
    def test_fails_when_kernel_config_unreadable(
        self, mock_check_kernel_config, mock_platform
    ):
        """Unreadable kernel config is a hard failure for this config testcase."""
        mock_platform.release.return_value = "test-kernel"
        mock_check_kernel_config.parse_kconfig.side_effect = FileNotFoundError(
            "/boot/config-test-kernel"
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_pci.test_renesas_xhci_pci_config(None, None)

        self.assertIn("Cannot read kernel config", str(context.exception))
        self.assertIn("/boot/config-test-kernel", str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.platform")
    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.check_kernel_config")
    def test_fails_when_kernel_config_parse_fails(
        self, mock_check_kernel_config, mock_platform
    ):
        """Malformed kernel config data is a hard failure for this config testcase."""
        mock_platform.release.return_value = "test-kernel"
        mock_check_kernel_config.parse_kconfig.side_effect = ValueError(
            "malformed kernel config"
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_pci.test_renesas_xhci_pci_config(None, None)

        self.assertIn("Cannot parse kernel config", str(context.exception))
        self.assertIn("/boot/config-test-kernel", str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.platform")
    @GLTTestBase.unittest.mock.patch("src.modules.pci.glt_test_pci.check_kernel_config")
    def test_skips_non_grace_systems(self, mock_check_kernel_config, mock_platform):
        """The testcase is scoped to Grace systems."""
        mock_platform.release.return_value = "test-kernel"
        self.mock_nvdev.return_value.is_cpu_type.return_value = False
        self.mock_nvdev.return_value.is_cpu_type.side_effect = None

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_pci.test_renesas_xhci_pci_config(None, None)

        self.assertIn("not an NVIDIA Grace CPU", str(context.exception))
        mock_check_kernel_config.parse_kconfig.assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
