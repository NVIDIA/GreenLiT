"""Unit tests for ARM FF-A test cases."""
# pylint: disable=protected-access

import os
import struct
import tempfile
from types import SimpleNamespace
from unittest import mock
import uuid
from pathlib import Path

import pytest

from tests.test_base import GLTTestBase
from common.tpm import acpi as acpi_tpm
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

    def setUp(self):
        """Keep unrelated host ACPI state out of synthetic descriptor tests."""
        super().setUp()
        patcher = self.mock.patch.object(arm_ffa, "read_tpm2_ffa_partition", return_value=None)
        self.addCleanup(patcher.stop)
        self.mock_acpi_partition = patcher.start()

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
                    r"Duplicate.*apply or backport.*01b9cae70616.*0da72a88dbba",
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
        """Unreadable/malformed inventory is not a patch-specific signature."""
        for byteorder in ("little", "big"):
            with self.subTest(byteorder=byteorder), \
                    self.mock.patch("sys.byteorder", byteorder), \
                    self.mock.patch.object(
                        arm_ffa, "_collect_ffa_partitions", return_value=([], "Malformed FF-A partition")), \
                    self.assertRaises(GLTCore.GLTTestFail) as failure:
                arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
            message = str(failure.exception)
            self.assertIn("Malformed FF-A partition", message)
            self.assertIn("FF-A initialization", message)
            for commit in ("f94ebb724067", "7bc0f589c81d", "be61da938576"):
                self.assertNotIn(commit, message)

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


    def test_fails_when_no_partition_records_remain(self):
        """An empty inventory cannot pass even if transport dependency once passed."""
        with self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=([], "")), \
                self.assertRaisesRegex(GLTCore.GLTTestFail, "No FF-A partition records"):
            arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)

    def test_validates_acpi_partition_and_architected_uuid(self):
        """An independent ACPI identity strengthens format-only discovery checks."""
        self.mock_acpi_partition.return_value = 0x8123
        partitions = [("0x8123", str(arm_ffa.ARM_FFA_TPM_UUID))]
        with self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=(partitions, "")):
            arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
        self.assertIn("Validated independent ACPI", self.mock_logger.info.call_args.args[-1])

    def test_fails_on_self_consistent_swapped_firmware_identity(self):
        """Identity mismatches fail with host-appropriate, conditional guidance."""
        self.mock_acpi_partition.return_value = 0x8123
        cases = [
            [("0x2381", str(arm_ffa.ARM_FFA_TPM_UUID))],
            [("0x8123", "a462b817-0618-af4f-b386-9a0835586138")],
        ]
        for byteorder in ("little", "big"):
            for partitions in cases:
                with self.subTest(byteorder=byteorder, partitions=partitions), \
                        self.mock.patch("sys.byteorder", byteorder), \
                        self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=(partitions, "")), \
                        self.assertRaisesRegex(GLTCore.GLTTestFail, "no matching ID/UUID pair") as failure:
                    arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
                message = str(failure.exception)
                self.assertIn("If this partition exposes multiple service UUIDs", message)
                self.assertIn("be61da938576", message)
                for commit in ("f94ebb724067", "7bc0f589c81d"):
                    self.assertEqual(commit in message, byteorder == "big")

    def test_logs_consistency_only_with_malformed_optional_acpi(self):
        """Corrupt TPM tables cannot fail otherwise healthy FF-A descriptors."""
        self.mock_acpi_partition.side_effect = acpi_tpm.read_tpm2_ffa_partition
        partitions = [("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")]
        for start_method in (7, 15):
            table = bytearray(64)
            table[:4] = b"TPM2"
            struct.pack_into("<I", table, 4, len(table))
            struct.pack_into("<I", table, 48, start_method)
            table[9] = (-sum(table) + 1) % 256  # Deliberately invalid checksum.
            with self.subTest(start_method=start_method), \
                    self.mock.patch("common.tpm.acpi.Path.read_bytes", return_value=bytes(table)), \
                    self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=(partitions, "")):
                arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
            evidence = self.mock_logger.info.call_args.args[-1]
            self.assertIn("checksum is invalid", evidence)
            self.assertIn("consistency only", evidence)
            self.assertIn("without an independent identity witness", evidence)

    def test_bad_descriptors_fail_even_with_invalid_optional_acpi(self):
        """Optional ACPI failures must not hide the duplicate-record regression."""
        self.mock_acpi_partition.side_effect = ValueError("invalid optional TPM2 table")
        partitions = [("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")] * 2
        with self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=(partitions, "")), \
                self.assertRaisesRegex(GLTCore.GLTTestFail, "Duplicate.*01b9cae70616.*0da72a88dbba"):
            arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
        self.mock_acpi_partition.assert_not_called()

    def test_logs_consistency_only_without_independent_identity(self):
        """Non-FF-A TPM firmware does not establish endian/discovery patch proof."""
        partitions = [("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")]
        with self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=(partitions, "")):
            arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
        self.assertIn("consistency only", self.mock_logger.info.call_args.args[-1])

    def test_logs_consistency_only_with_no_acpi_table(self):
        """DT-only systems retain generic checks without pretending an identity match."""
        self.mock_acpi_partition.side_effect = FileNotFoundError("no TPM2 table")
        partitions = [("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")]
        with self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=(partitions, "")):
            arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
        self.assertIn("No TPM2 ACPI table", self.mock_logger.info.call_args.args[-1])

    def test_logs_consistency_only_when_optional_acpi_cannot_be_read(self):
        """Report unavailable independent evidence without failing generic FF-A."""
        partitions = [("0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")]
        for error in (PermissionError("denied"), OSError("ACPI read failed")):
            self.mock_acpi_partition.side_effect = error
            with self.subTest(error=error), \
                    self.mock.patch.object(arm_ffa, "_collect_ffa_partitions", return_value=(partitions, "")):
                arm_ffa.test_arm_ffa_partition_info_descriptors(self.mock_sd, self.mock_logger)
            evidence = self.mock_logger.info.call_args.args[-1]
            self.assertIn(str(error), evidence)
            self.assertIn("consistency only", evidence)
            self.assertIn("without an independent identity witness", evidence)


def _dma_report(**overrides):
    """Create structured running-kernel evidence without loading a module."""
    values = {
        "schema_version": "2", "evidence_kind": "running_kernel_dma",
        "run_id": "a" * 32, "kernel_release": "6.14-unit", "device": "arm-ffa-7",
        "stage": "complete", "mask_present": "1", "mask_set_rc": "0",
        "allocation_bytes": str(os.sysconf("SC_PAGE_SIZE")), "data_checked": "1", "state_restored": "1",
    }
    values.update(overrides)
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _dma_parse(output):
    """Validate evidence against its particular module invocation."""
    return arm_ffa._parse_ffa_dma_result(output, "a" * 32, "6.14-unit", "arm-ffa-7")


def test_dma_candidates_prefer_host_and_exclude_tpm_bound_devices(tmp_path, monkeypatch):
    """Only existing unbound non-TPM devices may be selected; prefer the host."""
    records = (
        ("arm-ffa-1", "0x8001", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", True),
        ("arm-ffa-3", "0x8001", str(arm_ffa.ARM_FFA_TPM_UUID), False),
        ("arm-ffa-5", "0x8001", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", False),
        ("arm-ffa-9", "0x0000", str(uuid.UUID(int=0)), False),
    )
    for name, identifier, service, bound in records:
        path = tmp_path / name
        path.mkdir()
        (path / "partition_id").write_text(identifier, encoding="ascii")
        (path / "uuid").write_text(service, encoding="ascii")
        (path / "modalias").write_text(f"arm_ffa:{identifier[2:]}:{service}", encoding="ascii")
        if bound:
            (path / "driver").symlink_to(tmp_path / "missing-driver")
    monkeypatch.setattr(arm_ffa, "ARM_FFA_SYSFS_DEVICES_PATH", str(tmp_path))
    assert arm_ffa._ffa_dma_candidates() == ["arm-ffa-9", "arm-ffa-5"]
    (tmp_path / "arm-ffa-5/uuid").write_text("invalid", encoding="ascii")
    with pytest.raises(RuntimeError, match="Invalid FF-A partition UUID"):
        arm_ffa._ffa_dma_candidates()


def test_dma_missing_bus_has_no_candidates(tmp_path, monkeypatch):
    """No FF-A bus is a platform prerequisite gap."""
    monkeypatch.setattr(arm_ffa, "ARM_FFA_SYSFS_DEVICES_PATH", str(tmp_path / "absent"))
    assert arm_ffa._ffa_dma_candidates() == []


@pytest.mark.parametrize("field", [line.split("=", 1)[0] for line in _dma_report().splitlines()])
def test_dma_evidence_requires_every_field(field):
    """An insmod success or a stage alone cannot replace complete raw evidence."""
    output = "\n".join(line for line in _dma_report().splitlines() if not line.startswith(field + "="))
    with pytest.raises(RuntimeError):
        _dma_parse(output)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "1"), ("evidence_kind", "isolated_kernel_guest"),
    ("run_id", "b" * 32), ("kernel_release", "another-kernel"), ("device", "arm-ffa-1"),
    ("mask_present", "0"), ("mask_set_rc", "-5"), ("allocation_bytes", "0"),
    ("data_checked", "0"), ("state_restored", "0"), ("status", "PASS"), ("stage", "unknown"),
])
def test_dma_evidence_rejects_identity_and_operation_mismatches(field, value):
    """Only fresh, successful running-kernel DMA evidence can pass."""
    with pytest.raises(RuntimeError):
        _dma_parse(_dma_report(**{field: value}))


@pytest.mark.parametrize("output", [
    "", "PASS", _dma_report() + "status=PASS\n", _dma_report() + "stage=complete\n",
])
def test_dma_malformed_or_duplicate_result_is_rejected(output):
    """Malformed and ambiguous result protocols fail closed."""
    with pytest.raises(RuntimeError):
        _dma_parse(output)


@pytest.mark.parametrize("stage,fields,patch_hint", [
    ("dma_mask_missing", {"mask_present": "0", "mask_set_rc": "0", "allocation_bytes": "0", "data_checked": "0"}, True),
    ("dma_mask_set", {"mask_set_rc": "-5", "allocation_bytes": "0", "data_checked": "0"}, False),
    ("dma_allocation", {"allocation_bytes": "0", "data_checked": "0"}, False),
    ("dma_buffer_check", {"data_checked": "0"}, False),
])
def test_dma_semantic_failure_has_stage_appropriate_guidance(stage, fields, patch_hint):
    """Only a consistent missing mask warrants the mask-initialization patch hint."""
    status, reason = _dma_parse(_dma_report(stage=stage, **fields))
    assert status == "FAIL"
    assert stage in reason
    assert f"mask_set_rc={fields.get('mask_set_rc', '0')}" in reason
    assert ("cc0aac7ca17e" in reason) == patch_hint
    if not patch_hint:
        assert "device DMA configuration" in reason


@pytest.mark.parametrize("stage,fields", [
    ("dma_mask_set", {"allocation_bytes": "0", "data_checked": "0"}),
    ("dma_mask_set", {"mask_set_rc": "-5", "data_checked": "0"}),
    ("dma_allocation", {"mask_set_rc": "-5", "allocation_bytes": "0", "data_checked": "0"}),
    ("dma_allocation", {"mask_present": "0", "allocation_bytes": "0", "data_checked": "0"}),
    ("dma_buffer_check", {"allocation_bytes": "0", "data_checked": "0"}),
    ("dma_buffer_check", {"mask_set_rc": "-5", "data_checked": "0"}),
])
def test_dma_stage_rejects_contradictory_raw_progress(stage, fields):
    """A terminal stage must agree with the raw operation evidence."""
    with pytest.raises(RuntimeError, match="contradictory"):
        _dma_parse(_dma_report(stage=stage, **fields))


@pytest.mark.parametrize("field,value", [
    ("mask_present", "1"), ("mask_set_rc", "-5"),
    ("allocation_bytes", str(os.sysconf("SC_PAGE_SIZE"))), ("data_checked", "1"),
])
def test_dma_missing_mask_rejects_contradictory_evidence(field, value):
    """Do not attribute inconsistent probe metadata to the DMA-mask patch."""
    fields = {"mask_present": "0", "mask_set_rc": "0", "allocation_bytes": "0", "data_checked": "0"}
    fields[field] = value
    with pytest.raises(RuntimeError, match="missing-mask"):
        _dma_parse(_dma_report(stage="dma_mask_missing", **fields))


@pytest.mark.parametrize("restored", ["0", "1"])
def test_dma_restoration_failure_has_no_patch_attribution(restored):
    """A restoration problem or contradictory stage is a probe failure."""
    with pytest.raises(RuntimeError) as failure:
        _dma_parse(_dma_report(stage="restore_failed", state_restored=restored))
    assert "cc0aac7ca17e" not in str(failure.value)


def test_dma_skip_requires_no_dma_execution():
    """Only devices ineligible before DMA may skip; restoration is mandatory."""
    fields = {"mask_present": "0", "mask_set_rc": "0", "allocation_bytes": "0", "data_checked": "0"}
    for stage in ("device_missing", "device_bound", "tpm_device"):
        assert _dma_parse(_dma_report(stage=stage, **fields))[0] == "SKIP"
    with pytest.raises(RuntimeError, match="unexpected DMA execution"):
        _dma_parse(_dma_report(stage="device_bound"))
    assert _dma_parse(_dma_report())[0] == "PASS"


@pytest.fixture(name="dma_runtime")
def dma_runtime_fixture(tmp_path, monkeypatch):
    """Mock all kernel build/load calls; filesystem fixtures are ordinary files."""
    headers = tmp_path / "headers/6.14-unit/build"
    headers.mkdir(parents=True)
    (headers / "Makefile").write_text("# mocked headers", encoding="ascii")
    cached_release = headers / "include/config/kernel.release"
    cached_release.parent.mkdir(parents=True)
    cached_release.write_text("6.14-unit\n", encoding="ascii")
    result = tmp_path / "result"
    module = tmp_path / "module"
    state = {
        "headers": headers, "result": result, "module": module, "builds": [], "failure": None,
        "fields": {}, "raw": None, "load_error_after_init": False, "leave_resources": False,
        "header_release": "6.14-unit", "queries": [], "logger": mock.Mock(),
        "header_files": {str(path.relative_to(headers)): path.read_bytes()
                         for path in headers.rglob("*") if path.is_file()},
    }
    monkeypatch.setattr(arm_ffa, "ARM_FFA_DMA_HEADERS_ROOT", tmp_path / "headers")
    monkeypatch.setattr(arm_ffa, "ARM_FFA_DMA_RESULT", result)
    monkeypatch.setattr(arm_ffa, "ARM_FFA_DMA_MODULE_STATE", module)
    monkeypatch.setattr(arm_ffa.platform, "release", lambda: "6.14-unit")
    monkeypatch.setattr(arm_ffa.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(arm_ffa.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    config = mock.Mock(return_value={"CONFIG_MODULES": "y", "CONFIG_MODULE_UNLOAD": "y"})
    monkeypatch.setattr(arm_ffa.check_kernel_config, "parse_kconfig", config)
    tools = mock.Mock(return_value=True)
    sign = mock.Mock(return_value=True)
    candidates = mock.Mock(return_value=["arm-ffa-7"])
    monkeypatch.setattr(arm_ffa.glt_utils, "is_cli_available", tools)
    monkeypatch.setattr(arm_ffa.glt_utils, "sign_kernel_module", sign)
    monkeypatch.setattr(arm_ffa, "_ffa_dma_candidates", candidates)
    state.update(tools=tools, sign=sign, candidates=candidates, config=config)

    def command(arguments, *_, **__):
        """Produce only the owned filesystem effects of a module invocation."""
        if arguments[-1] == "kernelrelease":
            external = next((item[2:] for item in arguments if item.startswith("M=")), None)
            if external is None:
                return SimpleNamespace(stdout="6.14.99\n")
            directory = Path(external)
            assert directory.is_dir() and (directory / "Makefile").is_file()
            state["queries"].append(directory)
            return SimpleNamespace(stdout=state["header_release"] + "\n")
        if arguments[0] == "make":
            directory = Path(next(item[2:] for item in arguments if item.startswith("M=")))
            state["builds"].append(directory)
            if state["failure"] == "build":
                raise RuntimeError("compiler failed")
            (directory / (arm_ffa.ARM_FFA_DMA_MODULE + ".ko")).write_bytes(b"unit fixture")
        elif arguments[0] == "insmod":
            if state["failure"] == "load":
                raise RuntimeError("loader failed")
            module.mkdir()
            result.write_text(state["raw"] if state["raw"] is not None else _dma_report(**state["fields"]),
                              encoding="ascii")
            if state["load_error_after_init"]:
                raise RuntimeError("loader timed out after initialization")
        elif arguments[0] == "rmmod":
            if state["failure"] == "unload":
                raise RuntimeError("module still in use")
            if not state["leave_resources"]:
                result.unlink()
                module.rmdir()
        return SimpleNamespace(stdout="")

    commands = mock.Mock(side_effect=command)
    monkeypatch.setattr(arm_ffa.glt_utils, "run_command", commands)
    state["commands"] = commands
    return state


def _dma_invoke(state):
    """Invoke only the native wrapper whose hardware operations are mocked."""
    arm_ffa.test_arm_ffa_coherent_dma(None, state["logger"])


def test_dma_runtime_pass_needs_no_full_source_and_cleans_all_resources(dma_runtime, monkeypatch, tmp_path):
    """Installed headers suffice even when the old source variable points nowhere."""
    monkeypatch.setenv("GLT_FFA_KERNEL_SOURCE", str(tmp_path / "absent-full-source"))
    _dma_invoke(dma_runtime)
    commands = [call.args[0] for call in dma_runtime["commands"].call_args_list]
    assert commands[0] == ["make", "-s", "-C", str(dma_runtime["headers"]),
                           f"M={dma_runtime['builds'][0]}", "kernelrelease"]
    assert dma_runtime["queries"] == dma_runtime["builds"]
    assert {str(path.relative_to(dma_runtime["headers"])): path.read_bytes()
            for path in dma_runtime["headers"].rglob("*") if path.is_file()} == dma_runtime["header_files"]
    assert commands[-1] == ["rmmod", arm_ffa.ARM_FFA_DMA_MODULE]
    loader = next(command for command in commands if command[0] == "insmod")
    assert "device_name=arm-ffa-7" in loader
    assert "run_id=" + "a" * 32 in loader
    assert all(not path.exists() for path in dma_runtime["builds"])
    assert not dma_runtime["module"].exists()
    assert not dma_runtime["result"].exists()


@pytest.mark.parametrize("failure,message", [("build", "compiler failed"), ("load", "loader failed")])
def test_dma_build_or_load_failure_cleans_temp_files_without_blind_unload(dma_runtime, failure, message):
    """A probe that never loaded cannot cause an unconditional rmmod."""
    dma_runtime["failure"] = failure
    with pytest.raises(GLTCore.GLTTestFail, match=message):
        _dma_invoke(dma_runtime)
    assert all(not path.exists() for path in dma_runtime["builds"])
    assert not any(call.args[0][0] == "rmmod" for call in dma_runtime["commands"].call_args_list)


@pytest.mark.parametrize("raw,fields", [
    ("status=PASS\n", {}),
    (None, {"stage": "dma_mask_missing", "mask_present": "0", "allocation_bytes": "0",
            "data_checked": "0"}),
])
def test_dma_bad_or_failing_result_still_unloads(dma_runtime, raw, fields):
    """Semantic failures and malformed output cannot leave a loaded probe."""
    dma_runtime.update(raw=raw, fields=fields)
    with pytest.raises(GLTCore.GLTTestFail):
        _dma_invoke(dma_runtime)
    assert not dma_runtime["module"].exists()
    assert not dma_runtime["result"].exists()


@pytest.mark.parametrize("foreign", [False, True])
def test_dma_uncertain_loader_cleanup_requires_matching_run_identity(dma_runtime, foreign):
    """Clean up a timed-out owned load while preserving another invocation."""
    dma_runtime["load_error_after_init"] = True
    if foreign:
        dma_runtime["fields"]["run_id"] = "b" * 32
    with pytest.raises(GLTCore.GLTTestFail, match="loader timed out"):
        _dma_invoke(dma_runtime)
    unloaded = any(call.args[0][0] == "rmmod" for call in dma_runtime["commands"].call_args_list)
    assert unloaded == (not foreign)
    assert dma_runtime["module"].exists() == foreign


@pytest.mark.parametrize("leave_resources", [False, True])
def test_dma_cleanup_failure_or_leftovers_override_pass(dma_runtime, leave_resources):
    """A good DMA result cannot conceal a module or proc entry left behind."""
    if leave_resources:
        dma_runtime["leave_resources"] = True
    else:
        dma_runtime["failure"] = "unload"
    with pytest.raises(GLTCore.GLTTestFail, match="cleanup failed"):
        _dma_invoke(dma_runtime)
    assert all(not path.exists() for path in dma_runtime["builds"])


@pytest.mark.parametrize("prerequisite", ["headers", "wrong_headers", "tools", "device", "busy", "sign"])
def test_dma_prerequisite_gaps_skip_without_unsafe_cleanup(dma_runtime, prerequisite):
    """Missing runtime prerequisites remain explicit, non-passing skips."""
    if prerequisite == "headers":
        (dma_runtime["headers"] / "Makefile").unlink()
    elif prerequisite == "wrong_headers":
        dma_runtime["header_release"] = "wrong-kernel"
    elif prerequisite == "tools":
        dma_runtime["tools"].return_value = False
    elif prerequisite == "device":
        dma_runtime["candidates"].return_value = []
    elif prerequisite == "busy":
        dma_runtime["result"].write_text("another owner", encoding="ascii")
    else:
        dma_runtime["sign"].return_value = False
    with pytest.raises(GLTCore.GLTTestSkip):
        _dma_invoke(dma_runtime)
    assert not any(call.args[0][0] == "insmod" for call in dma_runtime["commands"].call_args_list)
    assert all(not path.exists() for path in dma_runtime["builds"] + dma_runtime["queries"])


@pytest.mark.parametrize("case", ["disabled", "missing", "unreadable"])
def test_dma_unavailable_unloading_skips_before_load(dma_runtime, case):
    """No DMA build or module load occurs without confirmed kernel unloading."""
    if case == "disabled":
        dma_runtime["config"].return_value["CONFIG_MODULE_UNLOAD"] = "n"
    elif case == "missing":
        dma_runtime["config"].return_value.pop("CONFIG_MODULE_UNLOAD")
    else:
        dma_runtime["config"].side_effect = PermissionError("unreadable config")
    with pytest.raises(GLTCore.GLTTestSkip) as failure:
        _dma_invoke(dma_runtime)
    assert "unload" in str(failure.value).lower()
    dma_runtime["commands"].assert_not_called()
    dma_runtime["candidates"].assert_not_called()
    assert not dma_runtime["module"].exists()
    assert not dma_runtime["result"].exists()


def test_dma_binding_race_skips_after_unloading(dma_runtime):
    """Device ownership is rechecked by the kernel before any DMA mutation."""
    dma_runtime["fields"] = {"stage": "device_bound", "mask_present": "0",
                             "allocation_bytes": "0", "data_checked": "0"}
    with pytest.raises(GLTCore.GLTTestSkip, match="device_bound"):
        _dma_invoke(dma_runtime)
    assert not dma_runtime["module"].exists()


def test_dma_missing_payload_fails_instead_of_skipping(dma_runtime, monkeypatch, tmp_path):
    """A broken packaged payload is not an unsupported platform."""
    monkeypatch.setattr(arm_ffa, "ARM_FFA_DMA_PAYLOAD", tmp_path / "missing-payload")
    with pytest.raises(GLTCore.GLTTestFail, match="payload is missing"):
        _dma_invoke(dma_runtime)


def test_dma_wrong_architecture_skips(dma_runtime, monkeypatch):
    """Unsupported architectures never attempt module operations."""
    monkeypatch.setattr(arm_ffa.platform, "machine", lambda: "x86_64")
    with pytest.raises(GLTCore.GLTTestSkip):
        _dma_invoke(dma_runtime)
    dma_runtime["commands"].assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
