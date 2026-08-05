"""Unit tests for ARM FF-A test cases."""
# pylint: disable=protected-access

import tempfile
import uuid
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_arm_ffa as arm_ffa


class TestArmFfaAvailability(GLTTestBase):
    """Test ARM FF-A availability and RXTX_MAP regression coverage."""

    def setUp(self):
        """Set up a healthy built-in FF-A transport by default."""
        super().setUp()
        self.rxtx_failure = False
        self.ffa_init_logged = True
        self._start_patch(
            "src.modules.arm64.glt_test_arm_ffa."
            "check_kernel_config.parse_kconfig",
            return_value={
                arm_ffa.ARM_FFA_CONFIG: "y",
                arm_ffa.ARM_FFA_SMCCC_CONFIG: "y",
            },
        )
        self._start_patch(
            "src.modules.arm64.glt_test_arm_ffa._query_ffa_modinfo",
            return_value=("ffa_module", True),
        )
        self.mock_check_kernel_log = self._start_patch(
            "src.modules.arm64.glt_test_arm_ffa.glt_utils.check_kernel_log",
            side_effect=self._kernel_log_result,
        )
        self.mock_isdir = self._start_patch(
            "src.modules.arm64.glt_test_arm_ffa.os.path.isdir",
            return_value=True,
        )
        self.mock_listdir = self._start_patch(
            "src.modules.arm64.glt_test_arm_ffa.os.listdir",
            return_value=["arm-ffa-0"],
        )
        self.mock_validate_attrs = self._start_patch(
            "src.modules.arm64.glt_test_arm_ffa._validate_ffa_partition_attrs",
            return_value=(True, ""),
        )

    def _start_patch(self, target, **kwargs):
        """Start a patch that is automatically stopped after each test."""
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _kernel_log_result(self, pattern, checkpoint=None):
        """Return configured results for the two kernel-log probes."""
        del checkpoint
        if pattern == arm_ffa.ARM_FFA_RXTX_MAP_FAILURE_PATTERN:
            return self.rxtx_failure
        if pattern == arm_ffa.ARM_FFA_DMESG_PATTERN:
            return self.ffa_init_logged
        return False

    def test_fails_on_rxtx_map_kernel_log_error(self):
        """Fail with the patch-specific diagnosis on the RXTX_MAP error."""
        self.rxtx_failure = True

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            arm_ffa.test_arm_ffa_availability(None, self.mock_logger)

        self.assertIn("RXTX_MAP registration failed", str(context.exception))
        self.mock_isdir.assert_not_called()

    def test_fails_when_ffa_bus_is_missing(self):
        """Fail when the FF-A bus is absent after transport probing."""
        self.mock_isdir.return_value = False

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            arm_ffa.test_arm_ffa_availability(None, self.mock_logger)

        self.assertIn("ARM FF-A bus not registered", str(context.exception))
        self.mock_listdir.assert_not_called()

    def test_fails_when_no_partitions_are_discovered(self):
        """Fail when the bus exists but no FF-A partitions were discovered."""
        self.mock_listdir.return_value = []

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            arm_ffa.test_arm_ffa_availability(None, self.mock_logger)

        self.assertIn("no partitions were discovered", str(context.exception))

    def test_fails_when_partitions_cannot_be_listed(self):
        """Fail cleanly when the FF-A partition directory cannot be read."""
        self.mock_listdir.side_effect = OSError("permission denied")

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            arm_ffa.test_arm_ffa_availability(None, self.mock_logger)

        self.assertIn("Failed to list ARM FF-A partitions", str(context.exception))

    def test_fails_when_partition_sysfs_is_incomplete(self):
        """Fail when discovered FF-A partitions do not expose required attrs."""
        self.mock_validate_attrs.return_value = (False, "missing uuid")

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            arm_ffa.test_arm_ffa_availability(None, self.mock_logger)

        self.assertIn("missing uuid", str(context.exception))

    def test_passes_when_rxtx_map_and_partition_sysfs_are_healthy(self):
        """Pass when RXTX_MAP has no error and partition sysfs is usable."""
        arm_ffa.test_arm_ffa_availability(None, self.mock_logger)

        self.mock_validate_attrs.assert_called_once_with(["arm-ffa-0"])


class TestArmFfaPartitionAttrs(GLTTestBase):
    """Test validation of required FF-A partition attributes."""

    def test_partition_attrs_are_readable(self):
        """Accept non-empty required attributes for every partition."""
        with self.mock.patch(
            "builtins.open", self.mock.mock_open(read_data="value")
        ) as mock_attr:
            valid, reason = arm_ffa._validate_ffa_partition_attrs(["arm-ffa-0"])

        self.assertTrue(valid)
        self.assertEqual(reason, "")
        self.assertEqual(mock_attr.call_count, 3)

    def test_partition_attrs_fail_when_unreadable(self):
        """Reject a partition attribute that cannot be read."""
        with self.mock.patch("builtins.open", side_effect=OSError("denied")):
            valid, reason = arm_ffa._validate_ffa_partition_attrs(["arm-ffa-0"])

        self.assertFalse(valid)
        self.assertIn("Failed to read", reason)

    def test_partition_attrs_fail_when_empty(self):
        """Reject an empty required partition attribute."""
        with self.mock.patch(
            "builtins.open", self.mock.mock_open(read_data="")
        ):
            valid, reason = arm_ffa._validate_ffa_partition_attrs(["arm-ffa-0"])

        self.assertFalse(valid)
        self.assertIn("empty partition_id", reason)


class TestArmFfaPartitionHelpers(GLTTestBase):
    """Test hardware-independent FF-A sysfs validation."""

    @staticmethod
    def _write_partition(devices_path, name, partition_id, partition_uuid,
                         modalias=None):
        """Create one synthetic FF-A sysfs device."""
        device_path = devices_path / name
        device_path.mkdir()
        (device_path / "partition_id").write_text(partition_id + "\n", encoding="ascii")
        (device_path / "uuid").write_text(partition_uuid + "\n", encoding="ascii")
        if modalias is None:
            try:
                modalias_uuid = str(uuid.UUID(partition_uuid))
            except ValueError:
                modalias_uuid = partition_uuid
            modalias = f"arm_ffa:{partition_id[2:]}:{modalias_uuid}"
        (device_path / "modalias").write_text(modalias + "\n", encoding="ascii")

    def test_collects_two_coherent_partitions(self):
        """Valid multi-partition sysfs data should pass intact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x8001",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )
            self._write_partition(
                devices_path,
                "arm-ffa-2",
                "0x8002",
                "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(error, "")
        self.assertEqual(
            partitions,
            [
                ("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                ("0x8002", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            ],
        )

    def test_accepts_uppercase_uuid_with_canonical_modalias(self):
        """Synthetic sysfs data should model the kernel's canonical UUID alias."""
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x8001",
                "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(error, "")
        self.assertEqual(
            partitions,
            [("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")],
        )

    def test_reports_missing_attribute(self):
        """A partially exposed partition should identify its missing attribute."""
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            device_path = devices_path / "arm-ffa-1"
            device_path.mkdir()
            (device_path / "partition_id").write_text("0x8001\n", encoding="ascii")
            (device_path / "modalias").write_text(
                "arm_ffa:8001:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\n",
                encoding="ascii",
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(partitions, [])
        self.assertIn("arm-ffa-1/uuid", error)

    def test_rejects_malformed_partition_id(self):
        """Partition IDs must use the kernel's four-digit hexadecimal format."""
        for partition_id in ("8001", "0xGGGG"):
            with self.subTest(partition_id=partition_id), \
                    tempfile.TemporaryDirectory() as temp_dir:
                devices_path = Path(temp_dir)
                self._write_partition(
                    devices_path,
                    "arm-ffa-1",
                    partition_id,
                    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                )

                partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

                self.assertEqual(partitions, [])
                self.assertIn("partition ID", error)

    def test_rejects_malformed_uuid(self):
        """Partition UUIDs must use a valid UUID representation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x8001",
                "not-a-uuid",
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(partitions, [])
        self.assertIn("UUID", error)

    def test_accepts_non_secure_host_with_null_uuid(self):
        """The driver-created host partition legitimately has ID zero and a null UUID."""
        null_uuid = "00000000-0000-0000-0000-000000000000"
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x0000",
                null_uuid,
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(error, "")
        self.assertEqual(partitions, [("0x0000", null_uuid)])

    def test_accepts_secure_partition_with_null_uuid(self):
        """FF-A v1.0 secure partitions may not expose service UUIDs."""
        null_uuid = "00000000-0000-0000-0000-000000000000"
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x8001",
                null_uuid,
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(error, "")
        self.assertEqual(partitions, [("0x8001", null_uuid)])

    def test_rejects_modalias_mismatch(self):
        """The modalias must encode the same ID and UUID as sibling attributes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x8001",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                modalias="arm_ffa:8002:bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(partitions, [])
        self.assertIn("modalias", error)

    def test_accepts_one_partition_id_with_multiple_service_uuids(self):
        """One FF-A endpoint may advertise more than one service UUID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x8001",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )
            self._write_partition(
                devices_path,
                "arm-ffa-2",
                "0x8001",
                "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(error, "")
        self.assertEqual(len(partitions), 2)

    def test_accepts_one_service_uuid_from_multiple_partitions(self):
        """Multiple FF-A endpoints may implement the same service UUID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            self._write_partition(
                devices_path,
                "arm-ffa-1",
                "0x8001",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )
            self._write_partition(
                devices_path,
                "arm-ffa-2",
                "0x8002",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )

            partitions, error = arm_ffa._collect_ffa_partitions(devices_path)

        self.assertEqual(error, "")
        self.assertEqual(len(partitions), 2)


class TestArmFfaPartitionRuntime(GLTTestBase):
    """Test GreenLiT result mapping for partition descriptor coverage."""

    def test_fails_on_self_consistent_misparsed_vera_descriptors(self):
        """The duplicate pair observed with the bad descriptor stride must fail."""
        partitions = [
            ("0x8001", "9b12fb8a-6047-ca64-88c8-18864a6caa4c"),
            ("0xc50e", "bd5e4112-19ee-6a3a-0300-010000000000"),
            ("0x8001", "17b862a4-1806-4faf-86b3-089a58353861"),
            ("0xc50e", "bd5e4112-19ee-6a3a-0300-010000000000"),
            ("0x8001", "257e0cd2-e364-48f6-b0c6-5260fd3a8d44"),
            ("0x0000", "00000000-0000-0000-0000-000000000000"),
        ]
        with self.mock.patch.object(
                arm_ffa,
                "_collect_ffa_partitions",
                return_value=(partitions, "")), \
                self.assertRaisesRegex(
                    GLTCore.GLTTestFail,
                    r"Duplicate.*apply or backport.*01b9cae70616",
                ):
            arm_ffa.test_arm_ffa_partition_info_descriptors(
                self.mock_sd, self.mock_logger
            )

    def test_accepts_same_uuid_from_different_partition_ids(self):
        """A service UUID may legitimately be implemented by multiple endpoints."""
        shared_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        partitions = [
            ("0x8001", shared_uuid),
            ("0x8002", shared_uuid),
            ("0x0000", "00000000-0000-0000-0000-000000000000"),
        ]
        with self.mock.patch.object(
                arm_ffa,
                "_collect_ffa_partitions",
                return_value=(partitions, "")):
            arm_ffa.test_arm_ffa_partition_info_descriptors(
                self.mock_sd, self.mock_logger
            )

        self.assertEqual(self.mock_logger.info.call_count, 1)

    def test_reports_partition_validation_error(self):
        """Malformed discovery data should report a contextual validation error."""
        with self.mock.patch.object(
                arm_ffa,
                "_collect_ffa_partitions",
                return_value=([], "Malformed FF-A partition")), \
                self.assertRaisesRegex(
                    GLTCore.GLTTestFail,
                    r"^Unable to validate ARM FF-A partition descriptors: "
                    r"Malformed FF-A partition$",
                ):
            arm_ffa.test_arm_ffa_partition_info_descriptors(
                self.mock_sd, self.mock_logger
            )

    def test_passes_with_unique_secure_partition_records(self):
        """Distinct secure partition records should complete normally."""
        partitions = [
            ("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            ("0x8001", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            ("0x0000", "00000000-0000-0000-0000-000000000000"),
        ]
        with self.mock.patch.object(
                arm_ffa,
                "_collect_ffa_partitions",
                return_value=(partitions, "")):
            arm_ffa.test_arm_ffa_partition_info_descriptors(
                self.mock_sd, self.mock_logger
            )

        self.assertEqual(self.mock_logger.info.call_count, 1)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
