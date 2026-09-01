"""Unit tests for ARM64 PMU helper functions."""
# pylint: disable=duplicate-code,protected-access,too-many-public-methods

import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from modules.arm64 import glt_test_pmu


class TestPmuHelpers(GLTTestBase):
    """Test PMU helper functions."""

    def test_normalize_nvidia_pmu_name(self):
        """Test PMU instance suffix normalization."""
        cases = {
            "nvidia_pcie_pmu_0": "nvidia_pcie_pmu",
            "nvidia_pcie_pmu_0_rc_1": "nvidia_pcie_pmu",
            "nvidia_pcie_pmu_0_chiplet_5": "nvidia_pcie_pmu",
            "nvidia_pcie_tgt_pmu_1_rc_5": "nvidia_pcie_tgt_pmu",
            "nvidia_pcie_tgt_pmu_1_chiplet_4": "nvidia_pcie_tgt_pmu",
            "nvidia_nvlink_c2c0_pmu_0": "nvidia_nvlink_c2c0_pmu",
            "nvidia_nvlink_c2c_pmu_0": "nvidia_nvlink_c2c_pmu",
            "nvidia_cmem_latency_pmu_1": "nvidia_cmem_latency_pmu",
            "nvidia_uncore_pmu_10": "nvidia_uncore_pmu",
            "nvidia_PCIE_PMU_0_RC_1": "nvidia_pcie_pmu",
            "nvidia_pcie_pmu_0_rc_1_extra": "nvidia_pcie_pmu",
            "cpu-cycles": "cpu-cycles",
        }

        for pmu_name, expected in cases.items():
            with self.subTest(pmu_name=pmu_name):
                self.assertEqual(
                    glt_test_pmu._normalize_nvidia_pmu_name(pmu_name),
                    expected)

    def test_parse_nvidia_perf_units(self):
        """Test parsing NVIDIA PMU families from perf list JSON entries."""
        perf_json = [
            {"EventName": "cycles"},
            {"Unit": "nvidia_ucf_pmu_0", "EventName": "slc_access_rd"},
            {"Unit": "nvidia_pcie_pmu_0_rc_1", "EventName": "rd_req"},
            {"Unit": "nvidia_pcie_tgt_pmu_0_rc_2", "EventName": "wr_req"},
            {"Unit": "nvidia_PCIE_PMU_0_RC_3", "EventName": "rd_req"},
            {"Unit": "armv8_pmuv3_0", "EventName": "cpu_cycles"},
            {"Unit": "nvidia_uncore_pmu_0", "EventName": "event=0x0"},
            {"Unit": None, "EventName": "bad_unit"},
            "not-a-dict",
        ]

        self.assertEqual(
            glt_test_pmu._parse_nvidia_perf_units(perf_json),
            {
                "nvidia_ucf_pmu",
                "nvidia_pcie_pmu",
                "nvidia_pcie_tgt_pmu",
                "nvidia_uncore_pmu",
            },
        )

    def test_get_nvidia_perf_list_uses_resolved_perf_command(self):
        """The caller-supplied perf binary is forwarded to the JSON helper."""
        with self.mock.patch.object(
            glt_test_pmu.perf_utils,
            "get_perf_list_json",
            return_value=[{"Unit": "nvidia_ucf_pmu_0"}],
        ) as mock_get_perf_list:
            self.assertEqual(
                glt_test_pmu._get_nvidia_perf_list("/perf", self.mock_logger),
                {"nvidia_ucf_pmu"},
            )

        mock_get_perf_list.assert_called_once_with(
            "/perf", self.mock_logger
        )

    def test_get_nvidia_perf_list_times_out(self):
        """A timeout reported by the shared JSON helper is propagated."""
        with self.mock.patch.object(
            glt_test_pmu.perf_utils,
            "get_perf_list_json",
            side_effect=ValueError("perf list timed out after 30 seconds"),
        ) as mock_get_perf_list:
            with self.assertRaisesRegex(ValueError, "perf list timed out"):
                glt_test_pmu._get_nvidia_perf_list("/perf", self.mock_logger)

        mock_get_perf_list.assert_called_once_with(
            "/perf", self.mock_logger
        )

    def test_classify_pmu_devices(self):
        """Test classifying named PMU devices versus fallback devices."""
        named, fallback = glt_test_pmu._classify_pmu_devices([
            "nvidia_uncore_pmu_0",
            "nvidia_ucf_pmu_0",
            "nvidia_pcie_pmu_0_rc_1",
            "nvidia_uncore_pmu_10",
        ])

        self.assertEqual(named, ["nvidia_ucf_pmu_0", "nvidia_pcie_pmu_0_rc_1"])
        self.assertEqual(
            fallback,
            ["nvidia_uncore_pmu_0", "nvidia_uncore_pmu_10"])

    def test_get_nvidia_pmu_devices(self):
        """Test reading NVIDIA PMU device names from sysfs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pmu_path = Path(temp_dir)
            for name in (
                    "nvidia_pcie_pmu_0_rc_1",
                    "nvidia_ucf_pmu_0",
                    "nvidia_PCIE_TGT_PMU_0_RC_2",
                    "armv8_pmuv3_0",
                    "nvidia_not_a_perf_source"):
                (pmu_path / name).mkdir()
            (pmu_path / "nvidia_fake_pmu_0").write_text(
                "not a directory\n",
                encoding="utf-8",
            )

            self.assertEqual(
                glt_test_pmu._get_nvidia_pmu_devices(pmu_path),
                [
                    "nvidia_PCIE_TGT_PMU_0_RC_2",
                    "nvidia_pcie_pmu_0_rc_1",
                    "nvidia_ucf_pmu_0",
                ])

    def test_get_nvidia_pmu_devices_ignores_non_directory(self):
        """Test reading PMU devices ignores non-directory sysfs paths."""
        with tempfile.NamedTemporaryFile() as temp_file:
            self.assertEqual(
                glt_test_pmu._get_nvidia_pmu_devices(temp_file.name), [])

    def test_get_nvidia_pmu_devices_handles_iteration_error(self):
        """Test sysfs iteration errors report no visible NVIDIA PMU devices."""
        with tempfile.TemporaryDirectory() as temp_dir, \
                self.mock.patch.object(
                    Path, "iterdir", side_effect=PermissionError):
            self.assertEqual(
                glt_test_pmu._get_nvidia_pmu_devices(temp_dir), [])


class TestPmuValidation(GLTTestBase):
    """Test PMU validation helpers."""

    def test_validate_common_nvidia_pmus(self):
        """Test common NVIDIA PMU validation allows mixed named and fallback PMUs."""
        self.assertIsNone(
            glt_test_pmu._validate_common_nvidia_pmus(
                self.mock_logger,
                {"nvidia_pcie_pmu", "nvidia_uncore_pmu"},
                ["nvidia_uncore_pmu_0", "nvidia_pcie_pmu_0_rc_1"],
            ))

        self.mock_logger.warning.assert_not_called()

    def test_validate_common_nvidia_pmus_fails_without_perf_events(self):
        """Test missing perf-visible PMU families fail common PMU validation."""
        self.assertRegex(
            glt_test_pmu._validate_common_nvidia_pmus(
                self.mock_logger,
                set(),
                ["nvidia_pcie_pmu_0_rc_1"],
            ),
            "No NVIDIA perf events")

    def test_validate_common_nvidia_pmus_fails_without_sysfs_devices(self):
        """Test missing sysfs PMU devices fail common PMU validation."""
        self.assertRegex(
            glt_test_pmu._validate_common_nvidia_pmus(
                self.mock_logger,
                {"nvidia_pcie_pmu"},
                [],
            ),
            "No NVIDIA PMU devices")

    def test_validate_common_nvidia_pmus_fails_on_fallback_only(self):
        """Test fallback-only PMU devices fail common PMU validation."""
        failure_reason = glt_test_pmu._validate_common_nvidia_pmus(
            self.mock_logger,
            {"nvidia_uncore_pmu"},
            ["nvidia_uncore_pmu_0", "nvidia_uncore_pmu_1"],
        )

        self.assertRegex(
            failure_reason, "Only generic fallback NVIDIA PMU devices")
        self.assertEqual(failure_reason, failure_reason.strip())

    def test_validate_grace_pmus_accepts_expected_families(self):
        """Test Grace validation accepts all expected PMU families."""
        self.assertIsNone(
            glt_test_pmu.validate_grace_pmus(
                self.mock_logger,
                set(glt_test_pmu.GRACE_PMU_FAMILIES),
                [
                    "nvidia_cnvlink_pmu_0",
                    "nvidia_nvlink_c2c0_pmu_0",
                    "nvidia_nvlink_c2c1_pmu_0",
                    "nvidia_pcie_pmu_0",
                    "nvidia_scf_pmu_0",
                ],
            ))

    def test_validate_grace_pmus_accepts_mixed_fallback_families(self):
        """Test Grace validation allows fallback PMUs with named PMUs."""
        self.assertIsNone(
            glt_test_pmu.validate_grace_pmus(
                self.mock_logger,
                set(glt_test_pmu.GRACE_PMU_FAMILIES) | {"nvidia_uncore_pmu"},
                [
                    "nvidia_cnvlink_pmu_0",
                    "nvidia_nvlink_c2c0_pmu_0",
                    "nvidia_nvlink_c2c1_pmu_0",
                    "nvidia_pcie_pmu_0",
                    "nvidia_scf_pmu_0",
                    "nvidia_uncore_pmu_0",
                ],
            ))

    def test_validate_grace_pmus_fails_on_missing_families(self):
        """Test Grace validation reports expected and detected PMU families."""
        self.assertRegex(
            glt_test_pmu.validate_grace_pmus(
                self.mock_logger,
                {"nvidia_pcie_pmu"},
                ["nvidia_pcie_pmu_0"],
            ),
            "detected perf")

    def test_validate_grace_pmus_fails_on_extra_families(self):
        """Test Grace validation rejects unexpected PMU families."""
        failure_reason = (
            glt_test_pmu.validate_grace_pmus(
                self.mock_logger,
                set(glt_test_pmu.GRACE_PMU_FAMILIES) | {"nvidia_future_pmu"},
                [
                    "nvidia_cnvlink_pmu_0",
                    "nvidia_nvlink_c2c0_pmu_0",
                    "nvidia_nvlink_c2c1_pmu_0",
                    "nvidia_pcie_pmu_0",
                    "nvidia_scf_pmu_0",
                    "nvidia_future_pmu_0",
                ],
            ))

        self.assertRegex(failure_reason, "unexpected Grace NVIDIA PMU")
        self.assertRegex(failure_reason, "detected perf")
        self.assertRegex(failure_reason, "extra perf")
        self.mock_logger.warning.assert_not_called()

    def test_missing_required_vera_pmu_families(self):
        """Test Vera required PMU family detection."""
        pmu_devices = [
            "nvidia_ucf_pmu_0",
            "nvidia_pcie_pmu_0_rc_0",
            "nvidia_cmem_latency_pmu_0",
        ]

        self.assertEqual(
            glt_test_pmu._missing_pmu_families(
                glt_test_pmu._get_pmu_families(pmu_devices),
                glt_test_pmu.VERA_REQUIRED_CS_PMU_FAMILIES),
            ["nvidia_pcie_tgt_pmu"],
        )

    def test_all_required_vera_pmu_families_present(self):
        """Test Vera required PMU families pass when all are visible."""
        pmu_devices = [
            "nvidia_ucf_pmu_0",
            "nvidia_pcie_pmu_0_rc_0",
            "nvidia_pcie_tgt_pmu_0_rc_0",
            "nvidia_nvlink_c2c_pmu_0",
        ]

        self.assertEqual(
            glt_test_pmu._missing_pmu_families(
                glt_test_pmu._get_pmu_families(pmu_devices),
                glt_test_pmu.VERA_REQUIRED_CS_PMU_FAMILIES),
            [],
        )

    def test_validate_vera_pmus_requires_perf_families(self):
        """Test Vera validation requires PMU visibility from perf and sysfs."""
        self.assertRegex(
            glt_test_pmu.validate_vera_pmus(
                self.mock_logger,
                {"nvidia_ucf_pmu", "nvidia_pcie_pmu"},
                [
                    "nvidia_ucf_pmu_0",
                    "nvidia_pcie_pmu_0_rc_0",
                    "nvidia_pcie_tgt_pmu_0_rc_0",
                ],
            ),
            "from perf")

    def test_validate_vera_pmus_requires_sysfs_families(self):
        """Test Vera validation reports missing PMU families from sysfs."""
        self.assertRegex(
            glt_test_pmu.validate_vera_pmus(
                self.mock_logger,
                {
                    "nvidia_ucf_pmu",
                    "nvidia_pcie_pmu",
                    "nvidia_pcie_tgt_pmu",
                },
                [
                    "nvidia_ucf_pmu_0",
                    "nvidia_pcie_pmu_0_rc_0",
                ],
            ),
            "from sysfs")

    def test_test_nvidia_perf_events_records_framework_failure(self):
        """Test decorated PMU test records a framework failure result."""
        fake_nvdev = self.mock.MagicMock()
        fake_nvdev.is_cpu_type.side_effect = lambda cpu: cpu == "vera"
        glt_test_pmu.test_nvidia_perf_events.attr.result = (
            GLTCore.GLTTestResult.NA, "")

        with self.mock.patch.object(glt_test_pmu, "NvDev") as nvdev_cls, \
                self.mock.patch.object(
                    glt_test_pmu.perf_utils, "resolve_perf_command",
                    return_value=("/perf", "")), \
                self.mock.patch.object(
                    glt_test_pmu, "_get_nvidia_perf_list",
                    return_value={"nvidia_uncore_pmu"}) as mock_perf_list, \
                self.mock.patch.object(
                    glt_test_pmu, "_get_nvidia_pmu_devices",
                    return_value=["nvidia_uncore_pmu_0"]):
            nvdev_cls.return_value = fake_nvdev
            nvdev_cls.get_release_info.return_value = {
                "NAME": "Ubuntu",
                "VERSION_ID": "24.04",
            }
            fake_nvdev.get_release_info.return_value = (
                nvdev_cls.get_release_info.return_value)

            with self.assertRaisesRegex(
                    GLTCore.GLTTestFail,
                    "Only generic fallback NVIDIA PMU devices"):
                glt_test_pmu.test_nvidia_perf_events(None, self.mock_logger)

        self.assertEqual(
            glt_test_pmu.test_nvidia_perf_events.attr.result[0],
            GLTCore.GLTTestResult.FAIL)
        mock_perf_list.assert_called_once_with("/perf", self.mock_logger)

    def test_test_nvidia_perf_events_fails_on_perf_list_timeout(self):
        """Decorated PMU test fails when perf list times out."""
        fake_nvdev = self.mock.MagicMock()
        fake_nvdev.is_cpu_type.side_effect = lambda cpu: cpu == "vera"
        glt_test_pmu.test_nvidia_perf_events.attr.result = (
            GLTCore.GLTTestResult.NA, "")

        with self.mock.patch.object(glt_test_pmu, "NvDev") as nvdev_cls, \
                self.mock.patch.object(
                    glt_test_pmu.perf_utils, "resolve_perf_command",
                    return_value=("/perf", "")), \
                self.mock.patch.object(
                    glt_test_pmu, "_get_nvidia_perf_list",
                    side_effect=ValueError("perf list timed out after 30 seconds")):
            nvdev_cls.return_value = fake_nvdev
            nvdev_cls.get_release_info.return_value = {
                "NAME": "Ubuntu",
                "VERSION_ID": "24.04",
            }
            fake_nvdev.get_release_info.return_value = (
                nvdev_cls.get_release_info.return_value)

            with self.assertRaisesRegex(
                    GLTCore.GLTTestFail,
                    "perf list timed out"):
                glt_test_pmu.test_nvidia_perf_events(None, self.mock_logger)
