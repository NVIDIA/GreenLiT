"""Unit tests for ARM FF-A test cases."""
# pylint: disable=protected-access

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

    def _kernel_log_result(self, pattern):
        """Return configured results for the two kernel-log probes."""
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


if __name__ == "__main__":
    GLTTestBase.unittest.main()
