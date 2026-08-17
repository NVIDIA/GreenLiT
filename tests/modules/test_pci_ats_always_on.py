"""Unit tests for the CXL.cache ATS always-on testcase."""
# pylint: disable=wrong-import-position

import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.pci import glt_test_pci


class TestCxlCacheAtsAlwaysOn(GLTTestBase):
    """Test ATS always-on validation under active IOMMU domain modes."""

    BDF = "0000:02:00.0"

    def setUp(self):
        """Mock target-device discovery for each testcase."""
        super().setUp()
        cli_patcher = self.mock.patch(
            "src.modules.pci.glt_test_pci.glt_utils.is_cli_available",
            return_value=True,
        )
        pci_devices_patcher = self.mock.patch(
            "src.modules.pci.glt_test_pci.os.listdir",
            return_value=[self.BDF],
        )
        dvsec_patcher = self.mock.patch(
            "src.modules.pci.glt_test_pci.cxl_utils.find_cxl_device_dvsec_base",
            return_value=0x10C,
        )
        cxl_cap_patcher = self.mock.patch(
            "src.modules.pci.glt_test_pci.cxl_utils.read_cxl_device_dvsec_cap_bits",
            return_value={"cache_capable": True},
        )
        nvdev_patcher = self.mock.patch(
            "src.modules.pci.glt_test_pci.NvDev",
        )
        self.cli_mock = cli_patcher.start()
        self.pci_devices_mock = pci_devices_patcher.start()
        self.dvsec_mock = dvsec_patcher.start()
        self.cxl_cap_mock = cxl_cap_patcher.start()
        self.nvdev_mock = nvdev_patcher.start().return_value
        self.nvdev_mock.is_cpu_type.return_value = True
        self.addCleanup(nvdev_patcher.stop)
        self.addCleanup(cxl_cap_patcher.stop)
        self.addCleanup(dvsec_patcher.stop)
        self.addCleanup(pci_devices_patcher.stop)
        self.addCleanup(cli_patcher.stop)

    def test_identity_domain_with_ats_disabled_fails(self):
        """ATS disabled in the always-on identity path is a failure."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.get_iommu_group_type",
            return_value="identity",
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.is_ats_enabled",
            return_value=False,
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                glt_test_pci.test_cxl_cache_ats_always_on(
                    None, self.mock_logger
                )
        self.assertIn(self.BDF, str(context.exception))

    def test_failure_logs_unvalidated_devices(self):
        """A mixed failure logs inconclusive CXL devices separately."""
        other_bdf = "0000:03:00.0"
        self.pci_devices_mock.return_value = [self.BDF, other_bdf]

        with self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.get_iommu_group_type",
            side_effect=("identity", "DMA"),
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.is_ats_enabled",
            side_effect=(False, True),
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                glt_test_pci.test_cxl_cache_ats_always_on(
                    None, self.mock_logger
                )

        self.assertIn(self.BDF, str(context.exception))
        self.assertNotIn(other_bdf, str(context.exception))
        self.mock_logger.warning.assert_called_once_with(
            "CXL.cache device %s is not in an identity IOMMU domain "
            "(type=%s); its ATS state does not validate the always-on bypass path.",
            other_bdf, "DMA",
        )

    def test_identity_domain_with_ats_enabled_passes(self):
        """ATS enabled in the identity path validates always-on behavior."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.get_iommu_group_type",
            return_value="identity",
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.is_ats_enabled",
            return_value=True,
        ):
            glt_test_pci.test_cxl_cache_ats_always_on(
                None, self.mock_logger
            )
        self.nvdev_mock.is_cpu_type.assert_called_once_with("vera")
        self.pci_devices_mock.assert_called_once_with("/sys/bus/pci/devices")
        self.dvsec_mock.assert_called_once_with(self.BDF)
        self.cxl_cap_mock.assert_called_once_with(self.BDF, 0x10C)
        self.mock_logger.info.assert_called_once()

    def test_non_cxl_device_is_ignored(self):
        """A non-CXL PCI function does not reach the ATS observer."""
        cxl_bdf = "0000:03:00.0"
        self.pci_devices_mock.return_value = [self.BDF, cxl_bdf]
        self.dvsec_mock.side_effect = (None, 0x10C)

        with self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.get_iommu_group_type",
            return_value="identity",
        ) as group_type_mock, self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.is_ats_enabled",
            return_value=True,
        ) as ats_mock:
            glt_test_pci.test_cxl_cache_ats_always_on(
                None, self.mock_logger
            )

        self.cxl_cap_mock.assert_called_once_with(cxl_bdf, 0x10C)
        group_type_mock.assert_called_once_with(cxl_bdf)
        ats_mock.assert_called_once_with(cxl_bdf)

    def test_identity_domain_with_unknown_ats_warns(self):
        """An inconclusive ATSCtl observer result cannot prove failure."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.get_iommu_group_type",
            return_value="identity",
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.is_ats_enabled",
            return_value=None,
        ):
            with self.assertRaises(GLTCore.GLTTestWarn) as context:
                glt_test_pci.test_cxl_cache_ats_always_on(
                    None, self.mock_logger
                )
        self.assertIn(self.BDF, str(context.exception))
        self.mock_logger.warning.assert_called_once_with(
            "Unable to determine ATS state for CXL.cache device %s; lspci "
            "failed or did not expose a conclusive ATSCtl value.",
            self.BDF,
        )

    def test_unreadable_iommu_group_type_warns(self):
        """An unreadable IOMMU group type is inconclusive, not a domain mismatch."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.get_iommu_group_type",
            return_value=None,
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.is_ats_enabled",
            return_value=True,
        ):
            with self.assertRaises(GLTCore.GLTTestWarn) as context:
                glt_test_pci.test_cxl_cache_ats_always_on(
                    None, self.mock_logger
                )
        self.assertIn(self.BDF, str(context.exception))
        self.mock_logger.warning.assert_called_once_with(
            "CXL.cache device %s: IOMMU group type is unreadable; cannot "
            "validate the always-on bypass path.",
            self.BDF,
        )

    def test_missing_lspci_skips(self):
        """Missing lspci skips before PCI device discovery."""
        self.cli_mock.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_pci.test_cxl_cache_ats_always_on(
                None, self.mock_logger
            )

        self.assertIn("lspci not available", str(context.exception))
        self.pci_devices_mock.assert_not_called()
        self.dvsec_mock.assert_not_called()

    def test_pci_enumeration_failure_skips(self):
        """An unreadable PCI device directory cannot be validated."""
        self.pci_devices_mock.side_effect = OSError("unavailable")

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_pci.test_cxl_cache_ats_always_on(
                None, self.mock_logger
            )

        self.assertIn("Unable to enumerate PCI devices", str(context.exception))
        self.dvsec_mock.assert_not_called()

    def test_non_vera_system_skips(self):
        """The ATS always-on coverage is scoped to Vera platforms."""
        self.nvdev_mock.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_pci.test_cxl_cache_ats_always_on(
                None, self.mock_logger
            )

        self.assertIn("Vera-based", str(context.exception))
        self.pci_devices_mock.assert_not_called()

    def test_no_cxl_cache_devices_skips(self):
        """A Vera host without CXL.cache devices is not applicable."""
        self.pci_devices_mock.return_value = []

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_pci.test_cxl_cache_ats_always_on(
                None, self.mock_logger
            )

        self.assertIn("No PCI device with readable CXL.cache", str(context.exception))
        self.dvsec_mock.assert_not_called()

    def test_pci_device_without_cxl_cache_skips(self):
        """A PCI function requires readable Cache_Capable to be applicable."""
        cases = (
            (None, None),
            (0x10C, None),
            (0x10C, {"cache_capable": False}),
        )

        for dvsec_off, cap_bits in cases:
            with self.subTest(dvsec_off=dvsec_off, cap_bits=cap_bits):
                self.dvsec_mock.return_value = dvsec_off
                self.cxl_cap_mock.return_value = cap_bits

                with self.assertRaises(GLTCore.GLTTestSkip) as context:
                    glt_test_pci.test_cxl_cache_ats_always_on(
                        None, self.mock_logger
                    )

                self.assertIn("readable CXL.cache capability", str(context.exception))
                self.dvsec_mock.reset_mock()
                self.cxl_cap_mock.reset_mock()

    def test_translated_domain_warns_instead_of_passing(self):
        """ATS enabled under translation does not prove always-on behavior."""
        with self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.get_iommu_group_type",
            return_value="DMA",
        ), self.mock.patch(
            "src.modules.pci.glt_test_pci.pci_utils.is_ats_enabled",
            return_value=True,
        ):
            with self.assertRaises(GLTCore.GLTTestWarn) as context:
                glt_test_pci.test_cxl_cache_ats_always_on(
                    None, self.mock_logger
                )
        self.assertIn("identity", str(context.exception))
        self.assertIn(self.BDF, str(context.exception))
        self.mock_logger.warning.assert_called_once_with(
            "CXL.cache device %s is not in an identity IOMMU domain "
            "(type=%s); its ATS state does not validate the always-on bypass path.",
            self.BDF, "DMA",
        )
