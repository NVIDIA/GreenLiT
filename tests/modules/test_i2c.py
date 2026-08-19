"""Unit tests for I2C GreenLiT helpers."""
# pylint: disable=protected-access,too-many-arguments,too-many-lines,too-many-public-methods,wrong-import-position

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.i2c import glt_test_i2c as i2c_tests
from src.modules.i2c import glt_test_i2c_error_reset as i2c_error_reset
from src.modules.i2c.glt_test_i2c import (
    _restore_tegra_i2c_binding,
    _tegra_i2c_devices,
    _tegra_i2c_kernel_error_messages,
)


class TestI2C(GLTTestBase):
    """Test Class for I2C module helpers."""

    def test_tegra_i2c_devices_returns_sorted_bound_devices(self):
        """Only matching driver symlinks should be returned."""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            driver = root / "drivers" / "tegra-i2c"
            target = root / "devices" / "NVDA2017:00"
            driver.mkdir(parents=True)
            target.mkdir(parents=True)

            (driver / "NVDA2017:01").symlink_to(target, target_is_directory=True)
            (driver / "NVDA2017:00").symlink_to(target, target_is_directory=True)
            (driver / "NVDA0301:00").symlink_to(target, target_is_directory=True)
            (driver / "NVDA2017:03").symlink_to(
                root / "devices" / "missing",
                target_is_directory=True,
            )
            (driver / "NVDA2017:02").mkdir()

            self.assertEqual(
                _tegra_i2c_devices(drv=driver),
                ["NVDA2017:00", "NVDA2017:01"],
            )

    def test_tegra_i2c_devices_missing_driver_returns_empty(self):
        """Missing tegra-i2c driver sysfs directory means no devices."""
        with TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing-i2c-driver"

            self.assertEqual(_tegra_i2c_devices(drv=missing), [])

    def test_tegra_i2c_devices_unreadable_driver_returns_empty(self):
        """Sysfs enumeration errors should make discovery fail closed."""
        mock = GLTTestBase.unittest.mock

        with TemporaryDirectory() as temp_dir:
            driver = Path(temp_dir) / "drivers" / "tegra-i2c"
            driver.mkdir(parents=True)

            with mock.patch.object(Path, "iterdir", side_effect=OSError("permission denied")):
                self.assertEqual(_tegra_i2c_devices(drv=driver), [])

    def test_tegra_i2c_kernel_error_messages_include_device(self):
        """Kernel log fragments must be specific to the selected device."""
        messages = _tegra_i2c_kernel_error_messages("NVDA2017:00")

        self.assertTrue(messages)
        self.assertTrue(all("NVDA2017:00" in message for message in messages))
        self.assertIn(
            "tegra-i2c NVDA2017:00: i2c transfer timed out",
            messages,
        )

    def test_tegra410_i2c_device_rebinds_each_detected_controller(self):
        """The Tegra410 test should exercise every detected NVDA2017 controller."""
        mock = GLTTestBase.unittest.mock
        sentinel = object()
        checkpoint = object()
        original_rebind_devices = getattr(
            i2c_tests.test_tegra410_i2c_device.attr,
            "rebind_devices",
            sentinel,
        )

        i2c_tests.test_tegra410_i2c_device.attr.rebind_devices = ("stale",)

        try:
            with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
                 mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
                 mock.patch.object(
                     i2c_tests, "_tegra_i2c_devices",
                     return_value=["NVDA2017:00", "NVDA2017:01"],
                 ), \
                 mock.patch.object(i2c_tests, "_dev_unbind_bind", return_value=(True, "")) as rebind, \
                 mock.patch.object(
                     i2c_tests.glt_utils, "capture_kernel_log_checkpoint",
                     return_value=checkpoint,
                 ) as capture, \
                 mock.patch.object(
                     i2c_tests.glt_utils, "get_kernel_log", return_value="",
                 ) as get_log:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

                self.assertEqual(rebind.call_count, 2)
                rebind.assert_any_call("NVDA2017:00")
                rebind.assert_any_call("NVDA2017:01")
                self.assertEqual(capture.call_count, 2)
                self.assertEqual(get_log.call_count, 2)
                get_log.assert_called_with(checkpoint)
                self.assertEqual(
                    i2c_tests.test_tegra410_i2c_device.attr.rebind_devices,
                    ("NVDA2017:00", "NVDA2017:01"),
                )
        finally:
            if original_rebind_devices is sentinel:
                delattr(i2c_tests.test_tegra410_i2c_device.attr, "rebind_devices")
            else:
                i2c_tests.test_tegra410_i2c_device.attr.rebind_devices = original_rebind_devices

    def test_tegra410_i2c_device_fails_when_device_is_not_bound_after_rebind(self):
        """A successful bind write must leave the device visible in the driver directory."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(
                 i2c_tests, "_tegra_i2c_devices",
                 side_effect=[["NVDA2017:00"], []],
             ), \
             mock.patch.object(i2c_tests, "_dev_unbind_bind", return_value=(True, "")), \
             mock.patch.object(
                 i2c_tests.glt_utils, "capture_kernel_log_checkpoint",
                 return_value=object(),
             ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn("not bound to tegra-i2c after rebind", str(ctx.exception))

    def test_tegra410_i2c_device_fails_when_checkpoint_capture_fails(self):
        """Checkpoint capture must succeed before the destructive unbind/bind cycle."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(i2c_tests, "_tegra_i2c_devices", return_value=["NVDA2017:00"]), \
             mock.patch.object(i2c_tests, "_dev_unbind_bind") as rebind, \
             mock.patch.object(
                 i2c_tests.glt_utils, "capture_kernel_log_checkpoint",
                 side_effect=RuntimeError("journal unavailable"),
             ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn(
            "Failed to capture kernel log checkpoint before rebinding",
            str(ctx.exception),
        )
        rebind.assert_not_called()

    def test_tegra410_i2c_device_fails_when_kernel_log_retrieval_fails(self):
        """Retrieval failures after rebind must not be treated as a clean pass."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(i2c_tests, "_tegra_i2c_devices", return_value=["NVDA2017:00"]), \
             mock.patch.object(i2c_tests, "_dev_unbind_bind", return_value=(True, "")), \
             mock.patch.object(
                 i2c_tests.glt_utils, "capture_kernel_log_checkpoint",
                 return_value=object(),
             ), \
             mock.patch.object(
                 i2c_tests.glt_utils, "get_kernel_log",
                 side_effect=RuntimeError("journal truncated"),
             ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn(
            "Failed to retrieve kernel log after rebinding",
            str(ctx.exception),
        )

    def test_tegra410_i2c_device_fails_on_new_kernel_error(self):
        """Only kernel errors emitted after the rebind cycle should fail the test."""
        mock = GLTTestBase.unittest.mock
        error = "tegra-i2c NVDA2017:00: i2c transfer timed out"

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(i2c_tests, "_tegra_i2c_devices", return_value=["NVDA2017:00"]), \
             mock.patch.object(i2c_tests, "_dev_unbind_bind", return_value=(True, "")), \
             mock.patch.object(
                 i2c_tests.glt_utils, "capture_kernel_log_checkpoint",
                 return_value=object(),
             ), \
             mock.patch.object(
                 i2c_tests.glt_utils, "get_kernel_log", return_value=error,
             ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn(error, str(ctx.exception))

    def test_restore_tegra_i2c_binding_rebinds_all_recorded_devices(self):
        """The post hook should restore every controller recorded by the test."""
        mock = GLTTestBase.unittest.mock
        attr = SimpleNamespace(rebind_devices=("NVDA2017:00", "NVDA2017:01"))
        writes = []

        class FakeOpen:
            """Fake open context manager that records writes."""

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def write(self, value):
                """Record a bind write."""
                writes.append(value)

        with mock.patch("builtins.open", return_value=FakeOpen()) as open_mock:
            _restore_tegra_i2c_binding(None, attr, self.mock_logger)

        self.assertEqual(open_mock.call_count, 2)
        self.assertEqual(writes, ["NVDA2017:00", "NVDA2017:01"])

    def test_restore_tegra_i2c_binding_accepts_single_rebind_device(self):
        """The post hook should not iterate over characters in a device name."""
        mock = GLTTestBase.unittest.mock
        attr = SimpleNamespace(rebind_devices="NVDA2017:00")
        writes = []

        class FakeOpen:
            """Fake open context manager that records writes."""

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def write(self, value):
                """Record a bind write."""
                writes.append(value)

        with mock.patch("builtins.open", return_value=FakeOpen()) as open_mock:
            _restore_tegra_i2c_binding(None, attr, self.mock_logger)

        self.assertEqual(open_mock.call_count, 1)
        self.assertEqual(writes, ["NVDA2017:00"])


class TestI2cAcpiRuntimePm(GLTTestBase):
    """Test Tegra I2C ACPI runtime-PM sleeping-while-atomic coverage logic."""

    _ATOMIC_SLEEP_BUG = (
        "BUG: sleeping function called from invalid context\n"
        "mutex_lock\n"
        "acpi_subsys_runtime_resume\n"
        "rpm_resume\n"
        "tegra_i2c_xfer\n"
    )

    def setUp(self):
        """Reset persistent GLT test result state."""
        super().setUp()
        i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result = (
            GLTCore.GLTTestResult.NA, ""
        )

    def _matching_topology(self):
        """Return a matching topology tuple for runtime-test mocks."""
        return ("match", "NVDA2017:01", Path("/sys/devices/platform/NVDA2017:01"))

    def _non_matching_topology(self):
        """Return a non-matching topology tuple for runtime-test mocks."""
        return ("no_match", "no topology", None)

    def test_tegra_i2c_devices_with_nvda_prefix_includes_grace_and_vera(self):
        """Existing Tegra I2C sysfs helper can discover Grace and Vera HIDs."""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            driver = root / "drivers" / "tegra-i2c"
            target = root / "devices" / "tegra"
            driver.mkdir(parents=True)
            target.mkdir(parents=True)

            (driver / "NVDA0301:00").symlink_to(target, target_is_directory=True)
            (driver / "NVDA2017:01").symlink_to(target, target_is_directory=True)
            (driver / "bind").write_text("", encoding="utf-8")

            self.assertEqual(
                i2c_tests._tegra_i2c_devices(prefix="NVDA", drv=driver),  # pylint: disable=protected-access
                ["NVDA0301:00", "NVDA2017:01"],
            )

    def test_accepts_grace_ipmi_over_tegra_i2c_path(self):
        """Grace-style NVDA0301/IPI0001 topology is in scope."""
        udev_output = (
            "P: /devices/platform/NVDA0301:00/i2c-0/"
            "i2c-IPI0001:00/ipmi/ipmi0\n"
        )

        self.assertEqual(
            i2c_tests._tegra_i2c_parent_from_ipmi_devpath(  # pylint: disable=protected-access
                udev_output,
                {"NVDA0301:00"},
            ),
            "NVDA0301:00",
        )

    def test_accepts_vera_ipmi_over_tegra_i2c_path(self):
        """Vera-style NVDA2017/IPI0001 topology is in scope."""
        udev_output = (
            "P: /devices/platform/NVDA2017:01/i2c-1/"
            "i2c-IPI0001:00/ipmi/ipmi0\n"
        )

        self.assertEqual(
            i2c_tests._tegra_i2c_parent_from_ipmi_devpath(  # pylint: disable=protected-access
                udev_output,
                {"NVDA2017:00", "NVDA2017:01"},
            ),
            "NVDA2017:01",
        )

    def test_rejects_ipmi_not_under_tegra_i2c(self):
        """IPI0001 alone is not enough without a bound Tegra I2C controller."""
        udev_output = (
            "P: /devices/platform/OTHER0001:00/i2c-1/"
            "i2c-IPI0001:00/ipmi/ipmi0\n"
        )

        self.assertIsNone(
            i2c_tests._tegra_i2c_parent_from_ipmi_devpath(  # pylint: disable=protected-access
                udev_output,
                {"NVDA2017:01"},
            )
        )

    def test_detects_atomic_sleep_signature_with_mutex_lock(self):
        """Atomic-sleep splat may contain mutex_lock instead of mutex_lock_nested."""
        self.assertTrue(
            i2c_tests._has_tegra_i2c_acpi_irq_safe_bug(  # pylint: disable=protected-access
                self._ATOMIC_SLEEP_BUG
            )
        )

    def test_detects_atomic_sleep_signature_with_prefixed_kernel_log_lines(self):
        """Journal/dmesg prefixes do not hide the mutex frame."""
        log = """
        Jul 07 16:10:07 host kernel: BUG: sleeping function called from invalid context
        Jul 07 16:10:07 host kernel:  __might_sleep+0x68/0xf0
        Jul 07 16:10:07 host kernel:  mutex_lock+0x30/0xd0
        Jul 07 16:10:07 host kernel:  acpi_subsys_runtime_resume+0x24/0x70
        Jul 07 16:10:07 host kernel:  rpm_resume+0x484/0x668
        Jul 07 16:10:07 host kernel:  tegra_i2c_xfer+0x40/0x460 [i2c_tegra]
        """

        self.assertTrue(
            i2c_tests._has_tegra_i2c_acpi_irq_safe_bug(log)  # pylint: disable=protected-access
        )

    def test_detects_atomic_sleep_signature_with_mutex_lock_nested(self):
        """Older stacks may show mutex_lock_nested."""
        log = """
        BUG: sleeping function called from invalid context at kernel/locking/mutex.c:585
        __might_sleep
        mutex_lock_nested
        acpi_subsys_runtime_resume
        rpm_resume
        tegra_i2c_xfer
        """

        self.assertTrue(
            i2c_tests._has_tegra_i2c_acpi_irq_safe_bug(log)  # pylint: disable=protected-access
        )

    def test_rejects_partial_atomic_sleep_signature(self):
        """The log matcher rejects unrelated sleeping-function splats."""
        log = """
        BUG: sleeping function called from invalid context at kernel/locking/mutex.c:271
        __might_sleep
        mutex_lock
        unrelated_driver_xfer
        """

        self.assertFalse(
            i2c_tests._has_tegra_i2c_acpi_irq_safe_bug(log)  # pylint: disable=protected-access
        )

    def test_rejects_signature_markers_split_across_splats(self):
        """Markers must appear in the same BUG block."""
        log = """
        BUG: sleeping function called from invalid context at kernel/locking/mutex.c:271
        __might_sleep
        mutex_lock
        acpi_subsys_runtime_resume
        rpm_resume
        unrelated_driver
        BUG: sleeping function called from invalid context at kernel/locking/other.c:1
        mutex_lock
        tegra_i2c_xfer
        """

        self.assertFalse(
            i2c_tests._has_tegra_i2c_acpi_irq_safe_bug(log)  # pylint: disable=protected-access
        )

    def _run_runtime_test(
        self,
        *,
        acpi=True,
        config="debug",
        ipmitool_available=True,
        baseline_log="baseline log\n",
        new_log="",
        baseline_error=None,
        checkpoint_error=None,
        retrieval_error=None,
        run_returncode=0,
        run_stderr="",
        run_stdout="Device Available          : yes\n",
        run_side_effect=None,
        runtime_status="suspended",
        topology=None,
    ):
        """Run the Tegra I2C ACPI runtime-PM test with patched dependencies."""
        mock = GLTTestBase.unittest.mock
        if config == "debug":
            config = {
                "CONFIG_DEBUG_ATOMIC_SLEEP": "y",
                "CONFIG_KALLSYMS": "y",
            }
        topology = self._matching_topology() if topology is None else topology
        run_result = mock.Mock(
            returncode=run_returncode,
            stderr=run_stderr,
            stdout=run_stdout,
        )
        checkpoint = object()

        def get_kernel_log(checkpoint_arg=None):
            if checkpoint_arg is None:
                if baseline_error is not None:
                    raise baseline_error
                return "" if baseline_log is None else baseline_log
            if retrieval_error is not None:
                raise retrieval_error
            return new_log

        with mock.patch("src.modules.i2c.glt_test_i2c.Path") as mock_path, \
             mock.patch("src.modules.i2c.glt_test_i2c.time.sleep"), \
             mock.patch(
                 "src.modules.i2c.glt_test_i2c._tegra_i2c_runtime_status",
             ) as mock_runtime_status, \
             mock.patch(
                 "src.modules.i2c.glt_test_i2c._ipmi_over_tegra_i2c_acpi_topology",
                 return_value=topology,
             ), \
             mock.patch("src.modules.i2c.glt_test_i2c.glt_utils") as mock_glt_utils, \
             mock.patch("src.modules.i2c.glt_test_i2c.subprocess.run") as mock_run, \
             mock.patch(
                 "src.modules.i2c.glt_test_i2c.check_kernel_config.parse_kconfig",
                 return_value=config,
             ):
            mock_path.return_value.is_dir.return_value = acpi
            mock_glt_utils.is_cli_available.return_value = ipmitool_available
            mock_glt_utils.get_kernel_log.side_effect = get_kernel_log
            if checkpoint_error is not None:
                mock_glt_utils.capture_kernel_log_checkpoint.side_effect = (
                    checkpoint_error
                )
            else:
                mock_glt_utils.capture_kernel_log_checkpoint.return_value = checkpoint
            if isinstance(runtime_status, (list, tuple)):
                mock_runtime_status.side_effect = list(runtime_status)
            else:
                mock_runtime_status.return_value = runtime_status
            if run_side_effect is not None:
                mock_run.side_effect = run_side_effect
            else:
                mock_run.return_value = run_result

            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe(
                None, self.mock_logger
            )
            return mock.Mock(
                run=mock_run,
                get_kernel_log=mock_glt_utils.get_kernel_log,
                capture_checkpoint=mock_glt_utils.capture_kernel_log_checkpoint,
                runtime_status=mock_runtime_status,
                checkpoint=checkpoint,
            )

    def test_runtime_test_skips_when_debug_config_disabled(self):
        """Runtime test skips when logs cannot prove fix presence."""
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_runtime_test(config={})
        self.assertIn("CONFIG_DEBUG_ATOMIC_SLEEP", str(context.exception))
        self.assertIn("non-conclusive", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.SKIP,
        )

    def test_runtime_test_fails_when_kernel_log_unavailable(self):
        """Runtime test fails if kernel logs cannot be collected."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(baseline_log=None)
        self.assertIn("Cannot read kernel log", str(context.exception))
        self.assertIn("sleeping-while-atomic", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_runtime_test_fails_when_baseline_log_retrieval_raises(self):
        """Baseline retrieval RuntimeError must fail before triggering IPMI."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(
                baseline_error=RuntimeError("journal unavailable"),
            )
        self.assertIn("Cannot read kernel log", str(context.exception))
        self.assertIn("journal unavailable", str(context.exception))

    def test_runtime_test_skips_when_kernel_config_unavailable(self):
        """Missing kernel config makes a quiet run non-conclusive."""
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_runtime_test(config=None)
        self.assertIn("Cannot read kernel config", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.SKIP,
        )

    def test_runtime_test_fails_unconfirmed_resume_before_config_skip(self):
        """Unconfirmed runtime resume is conclusive before config observability skips."""
        for config in (None, {}):
            with self.subTest(config=config), \
                 self.assertRaises(GLTCore.GLTTestFail) as context:
                self._run_runtime_test(
                    config=config,
                    runtime_status=["active", "active", "active", "active", "active"],
                )
            self.assertIn("runtime_status at trigger was active", str(context.exception))
            self.assertIn("runtime resume was not confirmed", str(context.exception))
            self.assertEqual(
                i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
                GLTCore.GLTTestResult.FAIL,
            )

    def test_runtime_test_samples_runtime_status_at_each_trigger(self):
        """runtime_status is sampled per trigger attempt, not only before quiet."""
        mocks = self._run_runtime_test()
        self.assertEqual(
            mocks.runtime_status.call_count,
            i2c_tests._I2C_ACPI_RUNTIME_PM_IPMI_TRIGGER_ATTEMPTS,  # pylint: disable=protected-access
        )
        mocks.capture_checkpoint.assert_called_once_with()
        self.assertEqual(mocks.get_kernel_log.call_count, 2)
        mocks.get_kernel_log.assert_any_call()
        mocks.get_kernel_log.assert_any_call(mocks.checkpoint)

    def test_runtime_test_fails_when_topology_inspection_errors(self):
        """Topology inspection errors fail instead of being treated as unsupported HW."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(topology=("error", "udevadm failed", None))
        self.assertIn("udevadm failed", str(context.exception))

    def test_runtime_test_skips_when_topology_does_not_match(self):
        """Unsupported IPMI topology is skipped, not failed."""
        with self.assertRaises(GLTCore.GLTTestSkip):
            self._run_runtime_test(topology=self._non_matching_topology())

    def test_runtime_test_fails_when_checkpoint_capture_fails(self):
        """Checkpoint capture must succeed before exercising IPMI."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(
                checkpoint_error=RuntimeError("no backend"),
            )
        self.assertIn(
            "Failed to capture kernel log checkpoint before IPMI trigger",
            str(context.exception),
        )

    def test_runtime_test_fails_when_post_trigger_kernel_log_unavailable(self):
        """Runtime test fails if the post-trigger kernel log cannot be collected."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(
                retrieval_error=RuntimeError("The dmesg checkpoint is no longer continuous"),
            )
        self.assertIn("post-trigger kernel log", str(context.exception))
        self.assertIn("no longer continuous", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_runtime_test_accepts_empty_post_checkpoint_log(self):
        """A quiet post-checkpoint window after exercising IPMI is a clean new log."""
        mocks = self._run_runtime_test(new_log="")
        mocks.get_kernel_log.assert_any_call(mocks.checkpoint)

    def test_runtime_test_fails_on_pre_existing_signature(self):
        """Runtime test fails clearly if the current boot already has the splat."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(baseline_log=self._ATOMIC_SLEEP_BUG)
        self.assertIn("already contains sleeping-while-atomic BUG", str(context.exception))

    def test_runtime_test_fails_when_signature_appears(self):
        """Runtime test fails on the Tegra I2C ACPI runtime-PM atomic-sleep signature."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(new_log=self._ATOMIC_SLEEP_BUG)
        self.assertIn("Found sleeping-while-atomic BUG", str(context.exception))
        self.assertIn("tegra_i2c_xfer", str(context.exception))

    def test_runtime_test_fails_on_signature_before_config_skip(self):
        """Concrete BUG evidence must fail before config observability skips."""
        for config in (None, {}):
            with self.subTest(config=config), \
                 self.assertRaises(GLTCore.GLTTestFail) as context:
                self._run_runtime_test(
                    config=config,
                    new_log=self._ATOMIC_SLEEP_BUG,
                )
            self.assertIn("Found sleeping-while-atomic BUG", str(context.exception))
            self.assertEqual(
                i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
                GLTCore.GLTTestResult.FAIL,
            )

    def test_runtime_test_fails_when_ipmitool_command_fails(self):
        """Runtime test fails when the trigger command fails."""
        with self.assertRaises(GLTCore.GLTTestFail):
            self._run_runtime_test(
                run_returncode=1,
                run_stderr="ipmi error",
                run_stdout="",
            )

    def test_runtime_test_skips_when_not_acpi_booted(self):
        """Non-ACPI boots are out of scope for this Tegra I2C ACPI test."""
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_runtime_test(acpi=False)
        self.assertIn("not ACPI-booted", str(context.exception))


class TestI2cAcpiErrorReset(GLTTestBase):
    """Test tracefs-based ACPI Tegra I2C device_reset() proof logic."""

    DEVICE = "NVDA0301:00"
    PID = 1234
    ROOT = Path("/tracefs")
    OK = i2c_error_reset._STATUS_OK
    SKIP = i2c_error_reset._STATUS_SKIP
    WARN = i2c_error_reset._STATUS_WARN
    FAIL = i2c_error_reset._STATUS_FAIL

    def setUp(self):
        """Reset persistent GLT result state."""
        super().setUp()
        i2c_error_reset.test_tegra_i2c_acpi_error_reset.attr.result = (
            GLTCore.GLTTestResult.NA,
            "",
        )

    @classmethod
    def _line(cls, event, body, pid=PID, site=None):
        """Return one ftrace row with the fixed event name."""
        site = site if site is not None else cls._site(event)
        prefix = f"{site} " if site else ""
        return f"python-{pid} [000] .... 1.000000: {event}: {prefix}{body}"

    @staticmethod
    def _site(event):
        """Return the kernel-generated callsite prefix for one fixed event."""
        return {
            i2c_error_reset._EVENT_INIT_ENTRY: "(tegra_i2c_init+0x0/0x428 [i2c_tegra])",
            i2c_error_reset._EVENT_INIT_RETURN: "(tegra_i2c_probe+0x400/0x7a8 [i2c_tegra] <- tegra_i2c_init)",
            i2c_error_reset._EVENT_RESET_ENTRY: "(__device_reset+0x0/0x148)",
            i2c_error_reset._EVENT_RESET_RETURN: "(tegra_i2c_init+0x34/0x428 [i2c_tegra] <- __device_reset)",
            i2c_error_reset._EVENT_WRITEL_ENTRY: "(i2c_writel+0x0/0x2e8 [i2c_tegra])",
        }[event]

    def _valid_trace(
        self,
        *,
        i2c="0x100",
        reset_ret="-2",
        first="0x1",
        first_reg="0xa8",
        second="0x0",
        second_reg="0xa8",
        extra_before_first=(),
        extra_between_pair=(),
        extra_before_return=(),
        init_ret="0",
    ):
        """Return the ordered lifecycle observed on the fixed path."""
        event = i2c_error_reset
        return "\n".join(
            (
                self._line(event._EVENT_INIT_ENTRY, f"i2c={i2c}"),
                self._line(event._EVENT_RESET_ENTRY, "dev=0x200"),
                self._line(event._EVENT_RESET_RETURN, f"retval={reset_ret}"),
                *extra_before_first,
                self._line(event._EVENT_WRITEL_ENTRY, f"i2c={i2c} value={first} reg={first_reg}"),
                *extra_between_pair,
                self._line(event._EVENT_WRITEL_ENTRY, f"i2c={i2c} value={second} reg={second_reg}"),
                *extra_before_return,
                self._line(event._EVENT_INIT_RETURN, f"retval={init_ret}"),
            )
        )

    def _marker_payload(self, **overrides):
        """Return a valid marker payload with optional field overrides."""
        with self.mock.patch.object(
            i2c_error_reset,
            "_boot_id",
            return_value="boot-a",
        ):
            payload = i2c_error_reset._marker_payload(self.DEVICE)
        payload.update(overrides)
        return payload

    def _run_decorated(self, outcome=None, cleanup_errors=(), proof_error=None, cleanup_probe=None):
        """Run the decorated test with host state patched out."""
        root = self.ROOT
        proof_patch = {"side_effect": proof_error} if proof_error is not None else {"return_value": outcome}
        cleanup_mock = cleanup_probe if cleanup_probe is not None else self.mock.Mock(return_value=list(cleanup_errors))
        with self.mock.patch.object(i2c_error_reset, "_tracefs_root", return_value=root), \
             self.mock.patch.object(
                 i2c_error_reset,
                 "_acquire_lock",
                 return_value=(77, (self.OK, "")),
             ), \
             self.mock.patch.object(i2c_error_reset, "_run_tracefs_proof", **proof_patch), \
             self.mock.patch.object(i2c_error_reset, "_cleanup_after_outcome", new=cleanup_mock), \
             self.mock.patch.object(i2c_error_reset, "_release_lock", return_value=None):
            i2c_error_reset.test_tegra_i2c_acpi_error_reset(None, self.mock_logger)

    def _run_proof_case(self, trace_text, options, expected_status, expected_reason, *, context=None):
        """Run one proof case and retain its mocks for caller-specific assertions."""
        if context is None:
            context = {"trace_root": self.ROOT, "selected_device": None, "marker_active": False}
        with self._proof_patches(trace_text, **options) as patches:
            status, reason = i2c_error_reset._run_tracefs_proof(self.ROOT, context, self.mock_logger)
            patches.outcome = (status, reason)
        self.assertEqual(status, expected_status)
        if expected_reason:
            self.assertIn(expected_reason, reason)
        else:
            self.assertEqual(reason, "")
        return patches, context

    def _assert_parser_cases(self, cases, expected_status, expected_reason):
        """Apply one status/reason assertion to parser case tables."""
        for case in cases:
            name, trace = case[:2]
            reason_needle = case[2] if len(case) == 3 else expected_reason
            with self.subTest(case=name):
                status, reason = i2c_error_reset._parse_trace_lifecycle(trace)
                self.assertEqual(status, expected_status)
                if reason_needle:
                    self.assertIn(reason_needle, reason)
                else:
                    self.assertEqual(reason, "")

    def _record_proof_call(self, state, name, result, *, expected, actual):
        self.assertEqual(actual, expected)
        state.order.append(name)
        return result

    @staticmethod
    def _capture_checkpoint(state, options):
        state.order.append("checkpoint")
        if "checkpoint_error" in options:
            raise options["checkpoint_error"]
        return object()

    def test_selector_contract_is_i2c_standard_stress(self):
        """The hardware-mutating proof is not a sanity-plan test."""
        attr = i2c_error_reset.test_tegra_i2c_acpi_error_reset.attr

        self.assertEqual(attr.category, "i2c")
        self.assertEqual(attr.name, "test_tegra_i2c_acpi_error_reset")
        self.assertEqual(set(attr.plans), {"standard", "stress"})
        self.assertEqual(Path(i2c_error_reset.__file__).parent.name, "i2c")

    def test_probe_commands_are_exact_five_fixed_events(self):
        """The tracefs proof uses only the reviewed dynamic event definitions."""
        self.assertEqual(
            i2c_error_reset._DEVICE_RESET_KPROBE_COMMANDS,
            (
                "p:greenlit_tegra_i2c_acpi_reset/greenlit_tegra_i2c_init_entry "
                "tegra_i2c_init i2c=%x0:x64",
                "r:greenlit_tegra_i2c_acpi_reset/greenlit_tegra_i2c_init_return "
                "tegra_i2c_init retval=$retval:s32",
                "p:greenlit_tegra_i2c_acpi_reset/greenlit_device_reset_entry "
                "__device_reset dev=%x0:x64",
                "r:greenlit_tegra_i2c_acpi_reset/greenlit_device_reset_return "
                "__device_reset retval=$retval:s32",
                "p:greenlit_tegra_i2c_acpi_reset/greenlit_i2c_writel_entry "
                "i2c_writel i2c=%x0:x64 value=%x1:x64 reg=%x2:x64",
            ),
        )
        self.assertFalse(any(":s64" in command for command in i2c_error_reset._DEVICE_RESET_KPROBE_COMMANDS))
        self.assertEqual(len(i2c_error_reset._TRACE_EVENTS), 5)

    def test_tracefs_missing_skips_before_lock_and_marker(self):
        """No tracefs mount means no lock, marker, probe, or binding mutation."""
        stat_results = (FileNotFoundError, self.mock.Mock(st_mode=0o100000), self.mock.Mock(st_mode=0o040000))
        with self.mock.patch.object(i2c_error_reset.Path, "stat", autospec=True, side_effect=stat_results):
            self.assertEqual(i2c_error_reset._tracefs_root(), Path("/sys/kernel/debug/tracing"))
        with self.mock.patch.object(i2c_error_reset.Path, "stat", autospec=True, side_effect=OSError("stat denied")), \
             self.assertRaisesRegex(OSError, "stat denied"):
            i2c_error_reset._tracefs_root()

        def stat_with_types(kprobe_mode, instances_mode):
            def stat_path(path):
                if path.name == "kprobe_events":
                    return self.mock.Mock(st_mode=kprobe_mode)
                if path.name == "instances":
                    return self.mock.Mock(st_mode=instances_mode)
                raise AssertionError(f"unexpected stat path {path}")
            return stat_path

        wrong_type_cases = (
            ("kprobe-type", 0o040000, 0o040000, "kprobe_events is not a regular file"),
            ("instances-type", 0o100000, 0o100000, "instances is not a directory"),
        )
        for name, kprobe_mode, instances_mode, expected in wrong_type_cases:
            with self.subTest(case=name), \
                 self.mock.patch.object(
                     i2c_error_reset.Path,
                     "stat",
                     autospec=True,
                     side_effect=stat_with_types(kprobe_mode, instances_mode),
                 ), \
                 self.assertRaisesRegex(RuntimeError, expected):
                i2c_error_reset._tracefs_root()
        with self.mock.patch.object(
            i2c_error_reset.Path,
            "stat",
            autospec=True,
            side_effect=stat_with_types(0o100000, 0o100000),
        ), \
             self.assertRaises(GLTCore.GLTTestFail) as fail_context:
            i2c_error_reset.test_tegra_i2c_acpi_error_reset(None, self.mock_logger)
        self.assertIn("instances is not a directory", str(fail_context.exception))

        with self.mock.patch.object(i2c_error_reset, "_tracefs_root", return_value=None), \
             self.mock.patch.object(i2c_error_reset, "_acquire_lock") as acquire, \
             self.mock.patch.object(i2c_error_reset, "_write_marker") as marker, \
             self.mock.patch.object(i2c_error_reset, "_dev_unbind_bind") as rebind, \
             self.assertRaises(GLTCore.GLTTestSkip) as context:
            i2c_error_reset.test_tegra_i2c_acpi_error_reset(None, self.mock_logger)

        self.assertIn("tracefs is not mounted", str(context.exception))
        acquire.assert_not_called()
        marker.assert_not_called()
        rebind.assert_not_called()

    def test_setup_tracefs_uses_private_instance_controls_and_pid_filters(self):
        """Setup writes global definitions but traces and filters only inside the instance."""
        tracefs_path = Path("/tracefs/kprobe_events")
        with self.mock.patch("builtins.open", self.mock.mock_open()) as open_mock, self.mock.patch.object(
            i2c_error_reset.os, "open", return_value=55
        ) as os_open:
            i2c_error_reset._write_text(tracefs_path, "cmd", "a")
            opener = open_mock.call_args.kwargs["opener"]
            self.assertEqual(opener(str(tracefs_path), 0), 55)
        open_mock.assert_called_once_with(tracefs_path, "w", encoding="ascii", opener=opener)
        os_open.assert_called_once_with(
            str(tracefs_path),
            i2c_error_reset.os.O_WRONLY | i2c_error_reset.os.O_CLOEXEC,
        )

        writes = []
        original_mkdir = Path.mkdir

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "kprobe_events").touch()
            (root / "instances").mkdir()
            instance = root / "instances" / i2c_error_reset._TRACE_INSTANCE

            def mkdir_with_events(path, *args, **kwargs):
                original_mkdir(path, *args, **kwargs)
                if path == instance:
                    for event in i2c_error_reset._TRACE_EVENTS:
                        original_mkdir(
                            path / "events" / i2c_error_reset._TRACE_GROUP / event,
                            parents=True,
                        )

            with self.mock.patch.object(i2c_error_reset.Path, "mkdir", mkdir_with_events), self.mock.patch.object(
                i2c_error_reset.os, "getpid", return_value=self.PID
            ), self.mock.patch.object(
                i2c_error_reset,
                "_write_text",
                side_effect=lambda path, value, mode="w": writes.append((path, value, mode)),
            ):
                self.assertEqual(i2c_error_reset._setup_tracefs(root), (self.OK, ""))

        self.assertEqual(
            [(value, mode) for path, value, mode in writes if path == root / "kprobe_events"],
            [(command, "a") for command in i2c_error_reset._DEVICE_RESET_KPROBE_COMMANDS],
        )
        self.assertNotIn((root / "tracing_on", "1", "w"), writes)
        for event in i2c_error_reset._TRACE_EVENTS:
            event_dir = instance / "events" / i2c_error_reset._TRACE_GROUP / event
            self.assertIn((event_dir / "filter", f"common_pid == {self.PID}", "w"), writes)
            self.assertIn((event_dir / "enable", "1", "w"), writes)
        self.assertIn((instance / "trace", "", "w"), writes)
        self.assertIn((instance / "tracing_on", "1", "w"), writes)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "kprobe_events").touch()
            (root / "instances").mkdir()
            with self.mock.patch.object(i2c_error_reset, "_write_text", side_effect=OSError("kprobe denied")):
                status, reason = i2c_error_reset._setup_tracefs(root)
        self.assertEqual(status, self.SKIP)
        self.assertIn("Cannot configure tracefs proof", reason)

    def test_marker_payload_contains_fixed_identities_and_valid_device(self):
        """The same-boot marker records constants, not cleanup-controlled paths."""
        payload = self._marker_payload()

        self.assertEqual(payload["boot_id"], "boot-a")
        self.assertEqual(payload["selected_device"], self.DEVICE)
        self.assertEqual(payload["trace_instance"], i2c_error_reset._TRACE_INSTANCE)
        self.assertEqual(payload["trace_group"], i2c_error_reset._TRACE_GROUP)
        self.assertEqual(payload["trace_events"], list(i2c_error_reset._TRACE_EVENTS))
        self.assertNotIn("kprobe_commands", payload)

    def test_marker_reader_accepts_only_same_boot_fixed_identity(self):
        """A valid marker authorizes same-boot stale cleanup."""
        with TemporaryDirectory() as temp_dir:
            marker_path = Path(temp_dir) / "marker.json"
            marker_path.write_text(json.dumps(self._marker_payload(kprobe_commands=["prior-build"])), encoding="utf-8")
            marker_path.chmod(0o600)
            with self.mock.patch.object(i2c_error_reset, "_MARKER_PATH", marker_path), \
                 self.mock.patch.object(i2c_error_reset, "_boot_id", return_value="boot-a"):
                marker, reason = i2c_error_reset._read_valid_marker()

        self.assertEqual(reason, "")
        self.assertEqual(marker["selected_device"], self.DEVICE)

    def test_marker_reader_rejects_wrong_boot_device_or_event_identity(self):
        """Stale cleanup requires exact same-boot marker identity and a device token."""
        variants = (
            ("boot_id", {"boot_id": "old-boot"}, "boot_id"),
            ("device", {"selected_device": "../unbind"}, "invalid selected device"),
            ("unsupported-device", {"selected_device": "NVDA2017:00"}, "invalid selected device"),
            ("events", {"trace_events": ["foreign"]}, "trace_events"),
        )

        with TemporaryDirectory() as temp_dir:
            marker_path = Path(temp_dir) / "marker.json"
            with self.mock.patch.object(i2c_error_reset, "_MARKER_PATH", marker_path), \
                 self.mock.patch.object(i2c_error_reset, "_boot_id", return_value="boot-a"):
                for _name, override, expected in variants:
                    with self.subTest(expected=expected):
                        marker_path.write_text(json.dumps(self._marker_payload(**override)), encoding="utf-8")
                        marker_path.chmod(0o600)
                        marker, reason = i2c_error_reset._read_valid_marker()
                        self.assertIsNone(marker)
                        self.assertIn(expected, reason)
                with self.subTest(expected="0600"):
                    marker_path.write_text(json.dumps(self._marker_payload()), encoding="utf-8")
                    marker_path.chmod(0o644)
                    marker, reason = i2c_error_reset._read_valid_marker()
                    self.assertIsNone(marker)
                    self.assertIn("0600", reason)
                with self.subTest(expected="owned"):
                    marker_path.chmod(0o600)
                    fake_stat = self.mock.Mock(st_mode=0o100600, st_uid=i2c_error_reset.os.geteuid() + 1)
                    with self.mock.patch.object(i2c_error_reset.os, "fstat", return_value=fake_stat):
                        marker, reason = i2c_error_reset._read_valid_marker()
                    self.assertIsNone(marker)
                    self.assertIn("owned", reason)
            with self.mock.patch.object(i2c_error_reset, "_MARKER_PATH", Path(temp_dir)), \
                 self.mock.patch.object(i2c_error_reset, "_boot_id", return_value="boot-a"):
                marker, reason = i2c_error_reset._read_valid_marker()
            self.assertIsNone(marker)
            self.assertIn("regular file", reason)

    def test_lock_authenticity_rejects_untrusted_file_before_flock(self):
        """The lifecycle lock is never trusted only because it lives under /run/lock."""
        cases = (
            ("type", 0o040600, i2c_error_reset.os.geteuid(), "regular file"),
            ("owner", 0o100600, i2c_error_reset.os.geteuid() + 1, "owned"),
            ("mode", 0o100644, i2c_error_reset.os.geteuid(), "0600"),
        )

        for _name, mode, uid, expected in cases:
            with self.subTest(expected=expected), \
                 self.mock.patch.object(i2c_error_reset.os, "open", return_value=91), \
                 self.mock.patch.object(
                     i2c_error_reset.os,
                     "fstat",
                     return_value=self.mock.Mock(st_mode=mode, st_uid=uid),
                 ), \
                 self.mock.patch.object(i2c_error_reset.os, "close") as close, \
                 self.mock.patch.object(i2c_error_reset.fcntl, "flock") as flock:
                lock_fd, outcome = i2c_error_reset._acquire_lock()
            self.assertIsNone(lock_fd)
            self.assertEqual(outcome[0], self.FAIL)
            self.assertIn(expected, outcome[1])
            close.assert_called_once_with(91)
            flock.assert_not_called()

        with self.mock.patch.object(i2c_error_reset.fcntl, "flock", side_effect=OSError("unlock failed")), \
             self.mock.patch.object(i2c_error_reset.os, "close", side_effect=OSError("close failed")) as close:
            reason = i2c_error_reset._release_lock(91)
        self.assertIn("unlock failed", reason)
        self.assertIn("close failed", reason)
        close.assert_called_once_with(91)

    def test_recovery_refuses_fixed_artifacts_without_valid_marker(self):
        """Fixed tracefs names without the same-boot marker are treated as foreign."""
        with self.mock.patch.object(i2c_error_reset, "_read_valid_marker", return_value=(None, "")), \
             self.mock.patch.object(i2c_error_reset, "_fixed_artifacts", return_value=["instance"]):
            status, reason = i2c_error_reset._recover_or_refuse_foreign_artifacts(self.ROOT)

        self.assertEqual(status, self.FAIL)
        self.assertIn("without a valid same-boot marker", reason)

    def test_recovery_cleans_only_valid_same_boot_marker(self):
        """Valid marker cleanup uses only constants plus the validated device token."""
        marker = {"selected_device": self.DEVICE}

        with self.mock.patch.object(i2c_error_reset, "_read_valid_marker", return_value=(marker, "")), \
             self.mock.patch.object(i2c_error_reset, "_cleanup_owned_tracefs", return_value=[]) as cleanup:
            status, reason = i2c_error_reset._recover_or_refuse_foreign_artifacts(self.ROOT)

        self.assertEqual((status, reason), (self.OK, ""))
        cleanup.assert_called_once_with(self.ROOT, self.DEVICE)

    def test_bind_restore_is_idempotent_and_bind_only(self):
        """Cleanup restoration never unbinds and writes bind only when the selected device is absent."""
        with self.mock.patch.object(i2c_error_reset, "_tegra_i2c_devices", return_value=[self.DEVICE]), \
             self.mock.patch.object(i2c_error_reset, "_write_text") as write_text, \
             self.mock.patch.object(i2c_error_reset, "_dev_unbind_bind") as unbind_bind:
            self.assertEqual(i2c_error_reset._restore_binding(self.DEVICE), [])
        write_text.assert_not_called()
        unbind_bind.assert_not_called()

        with self.mock.patch.object(i2c_error_reset, "_tegra_i2c_devices", side_effect=[[], [self.DEVICE]]), \
             self.mock.patch.object(i2c_error_reset, "_write_text") as write_text:
            self.assertEqual(i2c_error_reset._restore_binding(self.DEVICE), [])
        write_text.assert_called_once_with(Path("/sys/bus/platform/drivers/tegra-i2c/bind"), self.DEVICE)

    def test_bind_restore_reports_unresolved_absence(self):
        """A bind write is not enough; cleanup requires positive bound-state verification."""
        with self.mock.patch.object(i2c_error_reset, "_tegra_i2c_devices", return_value=[]), \
             self.mock.patch.object(i2c_error_reset, "_write_text"), \
             self.mock.patch.object(i2c_error_reset.time, "sleep"):
            errors = i2c_error_reset._restore_binding(self.DEVICE)

        self.assertIn("still absent", errors[-1])

    def test_cleanup_order_removes_marker_last_after_absence(self):
        """Cleanup order is binding, stop, disable, instance removal, global delete, marker."""
        order = []

        class FakePath:
            """Minimal path fake used to record cleanup order."""

            def __init__(self, label):
                self.label = label

            def __truediv__(self, child):
                return FakePath(f"{self.label}/{child}")

            def __str__(self):
                return self.label

            def lstat(self):
                """Report owned paths as present until cleanup removes them."""
                return self

            def rmdir(self):
                """Record private trace-instance removal."""
                order.append("remove-instance")

            def unlink(self):
                """Record cleanup marker removal."""
                order.append("remove-marker")

        def write_text(path, value, mode="w"):
            order.append(f"write:{path}:{mode}:{value}")

        with self.mock.patch.object(
            i2c_error_reset,
            "_restore_binding",
            side_effect=lambda _dev: order.append("restore") or [],
        ), self.mock.patch.object(
            i2c_error_reset,
            "_stop_instance_tracing",
            side_effect=lambda _root: order.append("stop") or None,
        ), self.mock.patch.object(i2c_error_reset, "_instance_dir", return_value=FakePath("instance")), \
             self.mock.patch.object(
                 i2c_error_reset,
                 "_instance_event_dir",
                 side_effect=lambda _root, event: FakePath(f"instance/{event}"),
             ), self.mock.patch.object(
                 i2c_error_reset,
                 "_event_dir",
                 side_effect=lambda _root, event: FakePath(f"global/{event}"),
             ), self.mock.patch.object(i2c_error_reset, "_write_text", side_effect=write_text), \
             self.mock.patch.object(i2c_error_reset, "_fixed_artifacts", return_value=[]), \
             self.mock.patch.object(i2c_error_reset, "_MARKER_PATH", FakePath("marker")), \
             self.mock.patch.object(
                 i2c_error_reset,
                 "_fsync_parent",
                 side_effect=lambda _path: order.append("fsync-marker"),
             ):
            self.assertEqual(i2c_error_reset._cleanup_owned_tracefs(self.ROOT, self.DEVICE), [])

        self.assertEqual(order[:2], ["restore", "stop"])
        first_global_delete = (
            "write:/tracefs/kprobe_events:a:-:"
            "greenlit_tegra_i2c_acpi_reset/greenlit_tegra_i2c_init_entry"
        )
        self.assertLess(order.index("remove-instance"), order.index(first_global_delete))
        self.assertTrue(all(
            entry.startswith("write:/tracefs/kprobe_events:a:-:")
            for entry in order if "kprobe_events" in entry
        ))
        self.assertTrue(all(":w:" in entry for entry in order if entry.startswith("write:instance/")))
        self.assertEqual(order[-2:], ["remove-marker", "fsync-marker"])

    def test_cleanup_retains_marker_on_restore_failure_or_remaining_artifacts(self):
        """The marker survives until binding is restored and fixed tracefs names are absent."""
        marker = self.mock.Mock()
        absent = self.mock.Mock(lstat=self.mock.Mock(side_effect=FileNotFoundError))
        denied = self.mock.Mock(lstat=self.mock.Mock(side_effect=OSError("lstat denied")))

        self.assertFalse(i2c_error_reset._path_present(absent))
        with self.assertRaisesRegex(OSError, "lstat denied"):
            i2c_error_reset._path_present(denied)

        with self.mock.patch.object(i2c_error_reset, "_restore_binding", return_value=["bind failed"]), \
             self.mock.patch.object(i2c_error_reset, "_stop_instance_tracing", return_value=None), \
             self.mock.patch.object(i2c_error_reset, "_instance_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_instance_event_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_event_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_fixed_artifacts", return_value=[]), \
             self.mock.patch.object(i2c_error_reset, "_MARKER_PATH", marker):
            errors = i2c_error_reset._cleanup_owned_tracefs(self.ROOT, self.DEVICE)
        self.assertIn("bind failed", errors)
        marker.unlink.assert_not_called()

        with self.mock.patch.object(i2c_error_reset, "_restore_binding", return_value=[]), \
             self.mock.patch.object(i2c_error_reset, "_stop_instance_tracing", return_value=None), \
             self.mock.patch.object(i2c_error_reset, "_instance_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_instance_event_dir", return_value=denied), \
             self.mock.patch.object(i2c_error_reset, "_event_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_MARKER_PATH", marker):
            errors = i2c_error_reset._cleanup_owned_tracefs(self.ROOT, self.DEVICE)
        self.assertIn("lstat denied", errors[0])
        marker.unlink.assert_not_called()

        with self.mock.patch.object(i2c_error_reset, "_restore_binding", return_value=[]), \
             self.mock.patch.object(i2c_error_reset, "_stop_instance_tracing", return_value=None), \
             self.mock.patch.object(i2c_error_reset, "_instance_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_instance_event_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_event_dir", return_value=absent), \
             self.mock.patch.object(i2c_error_reset, "_fixed_artifacts", return_value=["event"]), \
             self.mock.patch.object(i2c_error_reset, "_MARKER_PATH", marker):
            errors = i2c_error_reset._cleanup_owned_tracefs(self.ROOT, self.DEVICE)
        self.assertIn("Owned tracefs artifacts remain", errors[-1])

    def test_decorated_result_classifies_pass_skip_fail_and_cleanup_override(self):
        """Final GLT result decisions happen after cleanup inside the decorated test."""
        cases = (
            ((self.OK, ""), (), None, GLTCore.GLTTestResult.NA),
            ((self.SKIP, "non-conclusive"), (), GLTCore.GLTTestSkip, GLTCore.GLTTestResult.SKIP),
            ((self.WARN, "non-conclusive"), (), GLTCore.GLTTestWarn, GLTCore.GLTTestResult.WARN),
            ((self.WARN, "non-conclusive"), ("cleanup busy",), GLTCore.GLTTestFail, GLTCore.GLTTestResult.FAIL),
            ((self.FAIL, "bad reset"), (), GLTCore.GLTTestFail, GLTCore.GLTTestResult.FAIL),
            ((self.OK, ""), ("cleanup busy",), GLTCore.GLTTestFail, GLTCore.GLTTestResult.FAIL),
        )
        for outcome, cleanup_errors, raised, result in cases:
            with self.subTest(outcome=outcome, cleanup=cleanup_errors):
                self.setUp()
                if raised is None:
                    self._run_decorated(outcome, cleanup_errors)
                else:
                    with self.assertRaises(raised):
                        self._run_decorated(outcome, cleanup_errors)
                self.assertEqual(i2c_error_reset.test_tegra_i2c_acpi_error_reset.attr.result[0], result)

        self.setUp()
        captured_context = {}

        def capture_cleanup(context):
            captured_context.update(context)
            return []

        with self.mock.patch.object(i2c_error_reset, "_tracefs_root", return_value=self.ROOT), \
             self.mock.patch.object(
                 i2c_error_reset,
                 "_acquire_lock",
                 return_value=(77, (self.OK, "")),
             ), \
             self.mock.patch.object(
                 i2c_error_reset,
                 "_recover_or_refuse_foreign_artifacts",
                 return_value=(self.OK, ""),
             ), self.mock.patch.object(
                 i2c_error_reset,
                 "_select_device",
                 return_value=((self.OK, ""), self.DEVICE),
             ), \
             self.mock.patch.object(i2c_error_reset, "_run_ipmitool_mc_info", return_value=(True, "")), \
             self.mock.patch.object(i2c_error_reset, "_write_marker", side_effect=OSError("parent fsync failed")), \
             self.mock.patch.object(
                 i2c_error_reset,
                 "_cleanup_after_outcome",
                 side_effect=capture_cleanup,
             ) as cleanup_mock, \
             self.mock.patch.object(i2c_error_reset, "_release_lock", return_value=None), \
             self.assertRaises(GLTCore.GLTTestFail) as context:
            i2c_error_reset.test_tegra_i2c_acpi_error_reset(None, self.mock_logger)
        cleanup_mock.assert_called_once()
        self.assertTrue(captured_context["marker_active"])
        self.assertEqual(captured_context["selected_device"], self.DEVICE)
        self.assertIn("parent fsync failed", str(context.exception))

    def test_decorated_test_preserves_base_exception_after_cleanup(self):
        """KeyboardInterrupt identity is retained while cleanup diagnostics are attached."""
        cleanup = self.mock.Mock(return_value=["cleanup busy"])
        with self.assertRaises(KeyboardInterrupt) as context:
            self._run_decorated(proof_error=KeyboardInterrupt("stop"), cleanup_probe=cleanup)

        self.assertIn("stop", str(context.exception))
        self.assertIn("cleanup busy", " ".join(getattr(context.exception, "__notes__", ())))
        self.mock_logger.error.assert_called_once()

    def test_decorated_test_turns_unexpected_exception_into_failure_after_cleanup(self):
        """Unexpected Python errors still run cleanup and become GLT failures."""
        cleanup = self.mock.Mock(return_value=[])
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_decorated(proof_error=RuntimeError("boom"), cleanup_probe=cleanup)

        cleanup.assert_called_once()
        self.assertIn("Unexpected error", str(context.exception))
        self.assertIn("boom", str(context.exception))

    def test_select_device_classifies_acpi_topology_and_bound_state(self):
        """Preflight separates unsupported topology from attributable discovery failures."""
        cases = (
            (False, None, [], self.SKIP, "not ACPI"),
            (True, (i2c_error_reset._IPMI_TEGRA_TOPOLOGY_ERROR, "udev failed", None), [], self.FAIL, "udev failed"),
            (True, (i2c_error_reset._IPMI_TEGRA_TOPOLOGY_NO_MATCH, "not ipmi", None), [], self.SKIP, "not ipmi"),
            (True, (i2c_error_reset._IPMI_TEGRA_TOPOLOGY_MATCH, "../bad", None), [], self.FAIL, "Unexpected"),
            (
                True, (i2c_error_reset._IPMI_TEGRA_TOPOLOGY_MATCH, "NVDA2017:00", None),
                ["NVDA2017:00"], self.SKIP, "not a supported Grace",
            ),
            (True, self._topology(), [], self.FAIL, "not bound"),
            (True, self._topology(), [self.DEVICE], self.OK, ""),
        )

        for acpi, topology, devices, expected_status, expected_reason in cases:
            with self.subTest(expected=expected_reason), \
                 self.mock.patch.object(i2c_error_reset.Path, "is_dir", return_value=acpi), \
                 self.mock.patch.object(i2c_error_reset, "_ipmi_over_tegra_i2c_acpi_topology", return_value=topology), \
                 self.mock.patch.object(i2c_error_reset, "_tegra_i2c_devices", return_value=devices):
                outcome, device = i2c_error_reset._select_device(self.mock_logger)
            self.assertEqual(outcome[0], expected_status)
            if expected_reason:
                self.assertIn(expected_reason, outcome[1])
            else:
                self.assertEqual(outcome[1], "")
            if expected_status == self.OK:
                self.assertEqual(device, self.DEVICE)

    def test_run_tracefs_proof_happy_path_uses_controlled_rebind_once(self):
        """A clean proof checkpoints before mutation and parses only the private trace."""
        trace_text = self._valid_trace()
        patches, context = self._run_proof_case(trace_text, {}, self.OK, "")

        self.assertEqual(context["selected_device"], self.DEVICE)
        self.assertTrue(context["marker_active"])
        patches.setup.assert_called_once_with(self.ROOT)
        patches.capture.assert_called_once_with()
        patches.rebind.assert_called_once_with(self.DEVICE)
        patches.restore.assert_not_called()
        self.assertEqual(patches.order, ["setup", "checkpoint", "rebind"])
        self.assertEqual(patches.ipmi.call_count, 2)
        patches.stop.assert_called_once_with(self.ROOT)
        patches.get_log.assert_called_once()
        patches.read_trace.assert_called_once_with(self.ROOT)
        patches.parse.assert_called_once_with(trace_text)

    def test_run_tracefs_proof_fails_binding_and_bmc_health(self):
        """Mutation failures restore binding, then health checks can override them."""
        cases = (
            ("unsupported", {"select": ((self.SKIP, "unsupported Grace-only proof"), None)}, self.SKIP, "unsupported"),
            ("pre-bmc", {"ipmi_results": [(False, "pre ipmi failed")]}, self.FAIL, "pre ipmi failed"),
            ("rebind", {"rebind": (False, "bind failed")}, self.FAIL, "Controlled unbind/bind"),
            ("not-bound", {"bound_devices": []}, self.FAIL, "not bound"),
            (
                "rebind-stop",
                {"rebind": (False, "bind failed"), "stop_error": "Cannot stop private trace instance"},
                self.FAIL,
                "Controlled unbind/bind",
            ),
            ("restore", {"rebind": (False, "bind failed"), "restore": ["restore failed"]}, self.FAIL, "restore failed"),
            (
                "rebind-post-bmc",
                {"rebind": (False, "bind failed"), "ipmi_results": [(True, ""), (False, "post ipmi failed")]},
                self.FAIL,
                "post ipmi failed",
            ),
            (
                "rebind-kernel-error",
                {
                    "rebind": (False, "bind failed"),
                    "kernel_log": "tegra-i2c NVDA0301:00: i2c transfer timed out",
                },
                self.FAIL,
                "Kernel reported",
            ),
            ("post-bmc", {"ipmi_results": [(True, ""), (False, "post ipmi failed")]}, self.FAIL, "post ipmi failed"),
        )

        for name, options, expected_status, expected in cases:
            with self.subTest(expected=expected):
                patches, _context = self._run_proof_case(
                    self._valid_trace(), options, expected_status, expected
                )
            if name == "unsupported":
                patches.marker.assert_not_called()
                patches.setup.assert_not_called()
                patches.rebind.assert_not_called()
                patches.restore.assert_not_called()
            if name in ("rebind", "not-bound", "rebind-stop"):
                patches.restore.assert_called_once_with(self.DEVICE)
                self.assertEqual(patches.ipmi.call_count, 2)
                patches.stop.assert_called_once_with(self.ROOT)
                patches.get_log.assert_called_once()
                self.assertIn(expected, patches.outcome[1])
                self.assertNotIn("non-conclusive", patches.outcome[1])
                patches.stats.assert_not_called()
                patches.read_trace.assert_not_called()
                patches.parse.assert_not_called()
            if name == "restore":
                patches.restore.assert_called_once_with(self.DEVICE)
                self.assertEqual(patches.ipmi.call_count, 1)
                patches.get_log.assert_not_called()
            if name in ("rebind-post-bmc", "rebind-kernel-error"):
                patches.restore.assert_called_once_with(self.DEVICE)
                self.assertNotIn("Controlled unbind/bind", patches.outcome[1])
                patches.stats.assert_not_called()
                patches.read_trace.assert_not_called()
                patches.parse.assert_not_called()

    def test_run_tracefs_proof_kernel_log_and_trace_loss_classification(self):
        """Kernel-log failures fail; lost private trace events warn."""
        cases = (
            (
                "checkpoint",
                {"checkpoint_error": RuntimeError("no checkpoint")},
                self.FAIL,
                "checkpoint",
            ),
            ("get-log", {"log_error": RuntimeError("lost")}, self.FAIL, "scoped kernel log"),
            (
                "kernel-error",
                {"kernel_log": "tegra-i2c NVDA0301:00: i2c transfer timed out"},
                self.FAIL,
                "Kernel reported",
            ),
            (
                "tegra-warning",
                {"kernel_log": "WARNING: CPU: 0 PID: 1\nCall Trace:\n tegra_i2c_init+0x10/0x80\n"},
                self.FAIL,
                "tegra_i2c_init warning",
            ),
            (
                "unrelated-warning",
                {"kernel_log": "WARNING: CPU: 0 PID: 1\nCall Trace:\n unrelated_driver+0x10/0x80\n"},
                self.OK,
                "",
            ),
            (
                "warning-with-non-frame-init-text",
                {"kernel_log": "WARNING: CPU: 0 PID: 1\nCall Trace:\n tegra_i2c_init text only\n"},
                self.OK,
                "",
            ),
            (
                "warning-header-prose-init-token",
                {"kernel_log": "WARNING: tegra_i2c_init+0x10/0x80 is only prose\n"},
                self.OK,
                "",
            ),
            (
                "completed-unrelated-warning-before-init-frame",
                {
                    "kernel_log": (
                        "WARNING: CPU: 0 PID: 1\nCall Trace:\n unrelated_driver+0x10/0x80\n"
                        "---[ end trace 0000000000000000 ]---\n tegra_i2c_init+0x10/0x80\n"
                    )
                },
                self.OK,
                "",
            ),
            ("stats", {"stats": ["cpu0: overrun: 1"]}, self.WARN, "lost events"),
            (
                "stop-only",
                {"stop_error": "Cannot stop private trace instance"},
                self.WARN,
                "Cannot stop private trace instance",
            ),
            (
                "stop-post-bmc",
                {
                    "stop_error": "Cannot stop private trace instance",
                    "ipmi_results": [(True, ""), (False, "post ipmi failed")],
                },
                self.FAIL,
                "post ipmi failed",
            ),
            (
                "stop-kernel-error",
                {
                    "stop_error": "Cannot stop private trace instance",
                    "kernel_log": "tegra-i2c NVDA0301:00: i2c transfer timed out",
                },
                self.FAIL,
                "Kernel reported",
            ),
            (
                "trace-read",
                {"read_trace": (self.WARN, "trace read failed")},
                self.WARN,
                "trace read failed",
            ),
        )

        for name, options, expected_status, expected_reason in cases:
            with self.subTest(expected=expected_reason):
                patches, _context = self._run_proof_case(
                    self._valid_trace(), options, expected_status, expected_reason
                )
            if name == "checkpoint":
                patches.rebind.assert_not_called()
                self.assertNotIn("rebind", patches.order)
            if "stop_error" in options:
                patches.stats.assert_not_called()
                patches.read_trace.assert_not_called()

    def test_parser_accepts_ordered_lifecycle_and_signed_retval(self):
        """The parser proves the fallback lifecycle and skips successful ACPI reset."""
        event = i2c_error_reset
        cases = (
            ("signed", self._valid_trace(reset_ret="0xfffffffffffffffe")),
            ("built-in-reset-entry-no-module", self._valid_trace()),
            ("same-pid-multiple-cycles", "\n".join((self._valid_trace(), self._valid_trace()))),
            (
                "post-pair-unrelated-write",
                self._valid_trace(extra_before_return=(
                    self._line(event._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0xff reg=0x20"),
                )),
            ),
            (
                "unrelated-register-writes-around-pair",
                self._valid_trace(
                    extra_before_first=(
                        self._line(event._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0xff reg=0x20"),
                    ),
                    extra_between_pair=(
                        self._line(event._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0xfe reg=0x24"),
                    ),
                ),
            ),
            (
                "post-init-client-write",
                self._valid_trace() + "\n" + self._line(
                    event._EVENT_WRITEL_ENTRY,
                    "i2c=0x100 value=0x1 reg=0xb4",
                ),
            ),
            (
                "pre-proof-non-reset-write",
                self._line(event._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x1 reg=0xb4")
                + "\n"
                + self._valid_trace(),
            ),
            (
                "acpi-reset-before-fallback",
                "\n".join((self._valid_trace(reset_ret="0", first_reg="0x20", second_reg="0x24"),
                           self._valid_trace())),
            ),
        )
        self._assert_parser_cases(cases, self.OK, "")
        acpi_reset = "\n".join((
            self._line(event._EVENT_INIT_ENTRY, "i2c=0x100"),
            self._line(event._EVENT_RESET_ENTRY, "dev=0x200"),
            self._line(event._EVENT_RESET_RETURN, "retval=0"),
            self._line(event._EVENT_INIT_RETURN, "retval=0"),
        ))
        self._assert_parser_cases((("acpi-reset", acpi_reset),), self.SKIP, "fallback was not exercised")

    def test_parser_warns_ambiguous_unmatched_or_truncated_traces(self):
        """Ambiguous parser states warn instead of failing."""
        event = i2c_error_reset
        cases = (
            (
                "nested",
                "\n".join((
                    self._line(event._EVENT_INIT_ENTRY, "i2c=0x100"),
                    self._line(event._EVENT_INIT_ENTRY, "i2c=0x100"),
                )),
            ),
            ("reset-return", self._line(event._EVENT_RESET_RETURN, "retval=-2")),
            ("init-return", self._line(event._EVENT_INIT_RETURN, "retval=0")),
            ("missing-pid", f"bad row: {event._EVENT_INIT_ENTRY}: i2c=0x100"),
            ("truncated", self._line(event._EVENT_INIT_ENTRY, "i2c=0x100")),
            ("acpi-reset-truncated-before-init-return", "\n".join(self._valid_trace(reset_ret="0").splitlines()[:3])),
            ("reset-dev", self._valid_trace().replace("dev=0x200", "")),
            ("reset-retval", self._valid_trace().replace("retval=-2", "")),
            ("init-retval", self._valid_trace().replace("retval=0", "")),
            ("writel-i2c", self._valid_trace().replace("i2c=0x100 value=0x1", "value=0x1")),
            ("writel-value", self._valid_trace().replace(" value=0x1", "")),
            ("writel-reg", self._valid_trace().replace(" reg=0xa8", "", 1)),
            (
                "duplicate-field",
                self._valid_trace().replace("value=0x1 reg=0xa8", "value=0x1 value=0x1 reg=0xa8"),
            ),
            ("garbage-suffix", self._valid_trace().replace("value=0x1", "value=0x1oops", 1)),
            ("second-prefix", self._valid_trace().replace("i2c=0x100", "(extra+0x0/0x4) i2c=0x100", 1)),
            ("missing-site", self._line(event._EVENT_INIT_ENTRY, "i2c=0x100", site="")),
            ("garbage-site", self._line(event._EVENT_INIT_ENTRY, "i2c=0x100", site="tegra_i2c_init")),
            (
                "wrong-entry-site",
                self._line(
                    event._EVENT_WRITEL_ENTRY,
                    "i2c=0x100 value=0x1 reg=0xa8",
                    site="(tegra_i2c_init+0x0/0x428 [i2c_tegra])",
                ),
            ),
            (
                "malformed-module-suffix",
                self._valid_trace().replace(
                    self._site(event._EVENT_WRITEL_ENTRY),
                    "(i2c_writel+0x0/0x2e8 [i2c tegra])",
                    1,
                ),
            ),
            (
                "trailing-module-suffix",
                self._valid_trace().replace(
                    self._site(event._EVENT_WRITEL_ENTRY),
                    "(i2c_writel+0x0/0x2e8 [i2c_tegra] extra)",
                    1,
                ),
            ),
            (
                "wrong-module-suffix",
                self._valid_trace().replace(
                    self._site(event._EVENT_WRITEL_ENTRY),
                    "(i2c_writel+0x0/0x2e8 [i2c_other])",
                    1,
                ),
            ),
            (
                "built-in-reset-entry-module-suffix",
                self._valid_trace().replace(
                    self._site(event._EVENT_RESET_ENTRY),
                    "(__device_reset+0x0/0x148 [i2c_tegra])",
                    1,
                ),
            ),
            (
                "wrong-return-site",
                self._line(
                    event._EVENT_INIT_RETURN,
                    "retval=0",
                    site="(vfs_read+0x0/0x348 <- i2c_writel)",
                ),
            ),
            (
                "mixed-pid",
                self._valid_trace() + "\n" + self._line(event._EVENT_INIT_ENTRY, "i2c=0x100", pid=4321),
            ),
            (
                "mixed-controller",
                "\n".join((self._valid_trace(), self._valid_trace(i2c="0x101"))),
                "multiple I2C controllers",
            ),
            (
                "mismatch",
                self._valid_trace().replace("i2c=0x100 value=0x1", "i2c=0x101 value=0x1"),
            ),
            (
                "post-proof-orphan-reset-write",
                (
                    self._valid_trace()
                    + "\n"
                    + self._line(event._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x0 reg=0xa8")
                ),
            ),
            (
                "acpi-reset-fallback-orphan-reset-write",
                "\n".join((
                    self._valid_trace(reset_ret="0", first_reg="0x20", second_reg="0x24"),
                    self._valid_trace(),
                    self._line(event._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x0 reg=0xa8"),
                )),
            ),
        )

        self._assert_parser_cases(cases, self.WARN, "non-conclusive")

    def test_parser_fails_wrong_attributable_lifecycle_values_and_zero_events(self):
        """Complete but wrong lifecycle evidence is a failure, not non-conclusive."""
        cases = (
            ("reset-ret", self._valid_trace(reset_ret="-1"), "expected 0 or -ENOENT (-2)"),
            ("acpi-reset-register-write", self._valid_trace(reset_ret="0"), "reset-register write after ACPI _RST"),
            ("acpi-reset-init-ret", self._valid_trace(
                reset_ret="0", first_reg="0x20", second_reg="0x24", init_ret="1"), "tegra_i2c_init returned"),
            ("acpi-reset-then-invalid-reset-ret", "\n".join((
                self._valid_trace(reset_ret="0", first_reg="0x20", second_reg="0x24"),
                self._valid_trace(reset_ret="-1"))), "expected 0 or -ENOENT (-2)"),
            (
                "acpi-reset-then-orphan-reset-write",
                self._valid_trace(reset_ret="0", first_reg="0x20", second_reg="0x24") + "\n" + self._line(
                    i2c_error_reset._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x0 reg=0xa8"),
                "reset-register i2c_writel after ACPI _RST",
            ),
            (
                "fallback-acpi-reset-orphan-reset-write",
                "\n".join((
                    self._valid_trace(),
                    self._valid_trace(reset_ret="0", first_reg="0x20", second_reg="0x24"),
                    self._line(i2c_error_reset._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x0 reg=0xa8"),
                )),
                "reset-register i2c_writel after ACPI _RST",
            ),
            ("first", self._valid_trace(first="0x0"), "first reset-register write"),
            ("first-reg", self._valid_trace(first_reg="0x20"), "first reset-register write value"),
            ("second", self._valid_trace(second_reg="0x24"), "assert/deassert"),
            ("second-value", self._valid_trace(second="0x1"), "second reset-register write value"),
            ("init-ret", self._valid_trace(init_ret="1"), "tegra_i2c_init returned"),
            (
                "third-write",
                self._valid_trace(extra_before_return=(
                    self._line(i2c_error_reset._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x0 reg=0xa8"),
                )),
                "unexpected reset-register",
            ),
            (
                "order",
                "\n".join((
                    self._line(i2c_error_reset._EVENT_INIT_ENTRY, "i2c=0x100"),
                    self._line(i2c_error_reset._EVENT_RESET_ENTRY, "dev=0x200"),
                    self._line(i2c_error_reset._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x1 reg=0xa8"),
                    self._line(i2c_error_reset._EVENT_RESET_RETURN, "retval=-2"),
                    self._line(i2c_error_reset._EVENT_INIT_RETURN, "retval=0"),
                )),
                "before __device_reset returned",
            ),
            (
                "no-reset",
                "\n".join((
                    self._line(i2c_error_reset._EVENT_INIT_ENTRY, "i2c=0x100"),
                    self._line(i2c_error_reset._EVENT_INIT_RETURN, "retval=0"),
                )),
                "without an attributable",
            ),
            (
                "no-writes",
                "\n".join((
                    self._line(i2c_error_reset._EVENT_INIT_ENTRY, "i2c=0x100"),
                    self._line(i2c_error_reset._EVENT_RESET_ENTRY, "dev=0x200"),
                    self._line(i2c_error_reset._EVENT_RESET_RETURN, "retval=-2"),
                    self._line(i2c_error_reset._EVENT_INIT_RETURN, "retval=0"),
                )),
                "635bf3c8853359a987c5c909d424df92a0d3016a ACPI-without-_RST fallback discriminator",
            ),
            (
                "continues-after-post-init-client-write",
                "\n".join((
                    self._valid_trace(),
                    self._line(i2c_error_reset._EVENT_WRITEL_ENTRY, "i2c=0x100 value=0x1 reg=0xb4"),
                    self._valid_trace(reset_ret="-1"),
                )),
                "expected 0 or -ENOENT (-2)",
            ),
            ("zero", "", "No relevant tracefs events"),
        )

        self._assert_parser_cases(cases, self.FAIL, "")

    def test_trace_stats_reports_overruns_drops_and_malformed_values(self):
        """Per-CPU trace loss makes the proof non-conclusive."""
        valid = "entries: 4\noverrun: 0\ncommit overrun: 0\ndropped events: 0\n"
        cases = (
            ("valid-zeros", valid, ()),
            ("missing-key", "overrun: 0\ncommit overrun: 0\n", ("missing dropped events",)),
            (
                "duplicate-key",
                valid.replace("overrun: 0", "overrun: 0\noverrun: 0"),
                ("duplicate overrun",),
            ),
            ("missing-separator", valid.replace("overrun: 0", "overrun"), ("missing ':'",)),
            ("empty-value", valid.replace("overrun: 0", "overrun:"), ("empty overrun",)),
            ("nonnumeric-value", valid.replace("overrun: 0", "overrun: x"), ("malformed overrun",)),
            ("negative-value", valid.replace("overrun: 0", "overrun: -1"), ("malformed overrun",)),
            (
                "positive-values",
                "entries: 4\noverrun: 1\ncommit overrun: 2\ndropped events: 3\n",
                ("overrun: 1", "commit overrun: 2", "dropped events: 3"),
            )
        )

        for name, text, expected in cases:
            with self.subTest(case=name), TemporaryDirectory() as temp_dir:
                stats = (
                    Path(temp_dir)
                    / "instances"
                    / i2c_error_reset._TRACE_INSTANCE
                    / "per_cpu"
                    / "cpu0"
                    / "stats"
                )
                stats.parent.mkdir(parents=True)
                stats.write_text(text, encoding="utf-8")

                problems = i2c_error_reset._trace_stats_overrun_or_dropped(Path(temp_dir))

            self.assertEqual(bool(problems), bool(expected))
            for needle in expected:
                self.assertTrue(any(needle in problem for problem in problems))

    @staticmethod
    def _topology():
        """Return a matching IPMI-over-Tegra-I2C topology tuple."""
        return (
            i2c_error_reset._IPMI_TEGRA_TOPOLOGY_MATCH,
            TestI2cAcpiErrorReset.DEVICE,
            Path(f"/sys/devices/platform/{TestI2cAcpiErrorReset.DEVICE}"),
        )

    @contextlib.contextmanager
    def _proof_patches(self, trace_text, **options):
        """Patch host interactions for _run_tracefs_proof."""
        with contextlib.ExitStack() as stack:
            ns = SimpleNamespace()
            ns.order = []
            mocks = stack.enter_context(self.mock.patch.multiple(
                i2c_error_reset,
                _recover_or_refuse_foreign_artifacts=self.mock.DEFAULT,
                _select_device=self.mock.DEFAULT,
                _run_ipmitool_mc_info=self.mock.DEFAULT,
                _write_marker=self.mock.DEFAULT,
                _restore_binding=self.mock.DEFAULT,
                _tegra_i2c_devices=self.mock.DEFAULT,
                _stop_instance_tracing=self.mock.DEFAULT,
                _trace_stats_overrun_or_dropped=self.mock.DEFAULT,
                _read_private_trace=self.mock.DEFAULT,
                _parse_trace_lifecycle=self.mock.DEFAULT,
            ))
            mocks["_recover_or_refuse_foreign_artifacts"].return_value = (
                self.OK, ""
            )
            mocks["_select_device"].return_value = options.get(
                "select", ((self.OK, ""), self.DEVICE)
            )
            ns.ipmi = mocks["_run_ipmitool_mc_info"]
            ns.ipmi.side_effect = options.get(
                "ipmi_results", [(True, ""), (True, "")]
            )
            ns.marker = mocks["_write_marker"]

            ns.setup = stack.enter_context(self.mock.patch.object(
                i2c_error_reset,
                "_setup_tracefs",
                side_effect=lambda root: self._record_proof_call(
                    ns, "setup", (self.OK, ""), expected=self.ROOT, actual=root
                ),
            ))
            ns.capture = stack.enter_context(self.mock.patch.object(
                i2c_error_reset.glt_utils,
                "capture_kernel_log_checkpoint",
                side_effect=lambda: self._capture_checkpoint(ns, options),
            ))

            ns.rebind = stack.enter_context(self.mock.patch.object(
                i2c_error_reset,
                "_dev_unbind_bind",
                side_effect=lambda device: self._record_proof_call(
                    ns, "rebind", options.get("rebind", (True, "")), expected=self.DEVICE, actual=device
                ),
            ))
            ns.restore = mocks["_restore_binding"]
            ns.restore.side_effect = lambda device: self._record_proof_call(
                ns,
                "restore",
                options.get("restore", []),
                expected=self.DEVICE,
                actual=device,
            )
            mocks["_tegra_i2c_devices"].return_value = options.get("bound_devices", [self.DEVICE])
            ns.stop = mocks["_stop_instance_tracing"]
            ns.stop.return_value = options.get("stop_error")
            if "log_error" in options:
                log_patch = self.mock.patch.object(
                    i2c_error_reset.glt_utils,
                    "get_kernel_log",
                    side_effect=options["log_error"],
                )
            else:
                log_patch = self.mock.patch.object(
                    i2c_error_reset.glt_utils,
                    "get_kernel_log",
                    return_value=options.get("kernel_log", ""),
                )
            ns.get_log = stack.enter_context(log_patch)
            ns.stats = mocks["_trace_stats_overrun_or_dropped"]
            ns.stats.return_value = options.get("stats", [])
            read_trace = options.get("read_trace", (self.OK, ""))
            ns.read_trace = mocks["_read_private_trace"]
            ns.read_trace.return_value = (read_trace, trace_text)
            ns.parse = mocks["_parse_trace_lifecycle"]
            ns.parse.return_value = options.get("parse", (self.OK, ""))
            yield ns
