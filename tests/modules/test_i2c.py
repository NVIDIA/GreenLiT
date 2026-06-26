"""Unit tests for I2C GreenLiT helpers."""

from pathlib import Path
import subprocess
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.i2c import glt_test_i2c as i2c_tests
from src.modules.i2c.glt_test_i2c import (
    _kernel_log_text,
    _new_tegra_i2c_kernel_errors,
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

    def test_new_tegra_i2c_kernel_errors_ignores_non_contiguous_log(self):
        """A log snapshot that does not extend the cursor should not match stale errors."""
        error = "tegra-i2c NVDA2017:00: i2c transfer timed out"

        self.assertEqual(
            _new_tegra_i2c_kernel_errors("NVDA2017:00", "old log", error),
            [],
        )

    def test_new_tegra_i2c_kernel_errors_ignores_empty_baseline(self):
        """An unavailable baseline should not make the whole boot log look new."""
        error = "tegra-i2c NVDA2017:00: i2c transfer timed out"

        self.assertEqual(
            _new_tegra_i2c_kernel_errors("NVDA2017:00", "", error),
            [],
        )

    def test_kernel_log_text_returns_empty_on_timeout(self):
        """Hung kernel-log readers should not block the test indefinitely."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(
                 i2c_tests.subprocess,
                 "run",
                 side_effect=[
                     subprocess.TimeoutExpired(["journalctl", "--no-pager", "-b0", "-k"], 10),
                     subprocess.TimeoutExpired(["dmesg"], 10),
                 ],
             ):
            self.assertEqual(_kernel_log_text(), "")

    def test_kernel_log_text_falls_back_to_dmesg_on_journalctl_timeout(self):
        """A slow journalctl should not prevent using dmesg as a fallback."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(
                 i2c_tests.subprocess,
                 "run",
                 side_effect=[
                     subprocess.TimeoutExpired(["journalctl", "--no-pager", "-b0", "-k"], 10),
                     mock.Mock(returncode=0, stdout="dmesg log"),
                 ],
             ):
            self.assertEqual(_kernel_log_text(), "dmesg log")

    def test_tegra410_i2c_device_rebinds_each_detected_controller(self):
        """The Tegra410 test should exercise every detected NVDA2017 controller."""
        mock = GLTTestBase.unittest.mock
        sentinel = object()
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
                     i2c_tests, "_kernel_log_text",
                     side_effect=[
                         "old tegra-i2c NVDA2017:00: i2c transfer timed out",
                         "old tegra-i2c NVDA2017:00: i2c transfer timed out",
                         "old tegra-i2c NVDA2017:00: i2c transfer timed out",
                     ],
                     create=True,
                 ) as log:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

                self.assertEqual(rebind.call_count, 2)
                rebind.assert_any_call("NVDA2017:00")
                rebind.assert_any_call("NVDA2017:01")
                self.assertEqual(log.call_count, 3)
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
             mock.patch.object(i2c_tests, "_kernel_log_text", return_value="old log"):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn("not bound to tegra-i2c after rebind", str(ctx.exception))

    def test_tegra410_i2c_device_skips_when_initial_kernel_log_is_unavailable(self):
        """The test should not pass if it cannot capture a baseline kernel log."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(i2c_tests, "_tegra_i2c_devices", return_value=["NVDA2017:00"]), \
             mock.patch.object(i2c_tests, "_kernel_log_text", return_value=""):
            with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn("Kernel log is unavailable", str(ctx.exception))

    def test_tegra410_i2c_device_skips_on_non_contiguous_kernel_log(self):
        """The test should not treat unrelated log snapshots as a clean diff."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(i2c_tests, "_tegra_i2c_devices", return_value=["NVDA2017:00"]), \
             mock.patch.object(i2c_tests, "_dev_unbind_bind", return_value=(True, "")), \
             mock.patch.object(
                 i2c_tests,
                 "_kernel_log_text",
                 side_effect=["baseline log", "different log"],
             ):
            with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn("Kernel log cursor is non-contiguous", str(ctx.exception))

    def test_tegra410_i2c_device_fails_on_new_kernel_error(self):
        """Only kernel errors emitted after the rebind cycle should fail the test."""
        mock = GLTTestBase.unittest.mock
        error = "tegra-i2c NVDA2017:00: i2c transfer timed out"

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(i2c_tests, "_tegra_i2c_devices", return_value=["NVDA2017:00"]), \
             mock.patch.object(i2c_tests, "_dev_unbind_bind", return_value=(True, "")), \
             mock.patch.object(
                 i2c_tests, "_kernel_log_text",
                 side_effect=["old log", f"old log\n{error}"],
                 create=True,
             ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn(error, str(ctx.exception))

    def test_restore_tegra_i2c_binding_rebinds_all_recorded_devices(self):
        """The post hook should restore every controller recorded by the test."""
        mock = GLTTestBase.unittest.mock
        attr = SimpleNamespace(rebind_devices=("NVDA2017:00", "NVDA2017:01"))

        with mock.patch("builtins.open", mock.mock_open()) as open_mock:
            _restore_tegra_i2c_binding(None, attr, self.mock_logger)

        self.assertEqual(open_mock.call_count, 2)
        handle = open_mock()
        handle.write.assert_any_call("NVDA2017:00")
        handle.write.assert_any_call("NVDA2017:01")

    def test_restore_tegra_i2c_binding_accepts_single_rebind_device(self):
        """The post hook should not iterate over characters in a device name."""
        mock = GLTTestBase.unittest.mock
        attr = SimpleNamespace(rebind_devices="NVDA2017:00")

        with mock.patch("builtins.open", mock.mock_open()) as open_mock:
            _restore_tegra_i2c_binding(None, attr, self.mock_logger)

        handle = open_mock()
        handle.write.assert_called_once_with("NVDA2017:00")
