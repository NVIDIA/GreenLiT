"""Unit tests for I2C GreenLiT helpers."""
# pylint: disable=too-many-arguments,too-many-lines,too-many-public-methods,wrong-import-position

import sys
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.i2c import glt_test_i2c as i2c_tests
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
