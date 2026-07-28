"""Unit tests for I2C GreenLiT helpers."""
# pylint: disable=too-many-arguments,too-many-lines,too-many-public-methods,wrong-import-position

import sys
from pathlib import Path
import subprocess
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.i2c import glt_test_i2c as i2c_tests
from src.modules.i2c.glt_test_i2c import (
    _kernel_log_snapshot,
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

    def test_kernel_log_snapshot_returns_dmesg_failure_on_reader_timeouts(self):
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
            snapshot = _kernel_log_snapshot()

        self.assertEqual(snapshot.text, "")
        self.assertIsNone(snapshot.cursor)
        self.assertFalse(snapshot.query_succeeded)
        self.assertEqual(snapshot.backend, "dmesg")

    def test_kernel_log_snapshot_falls_back_to_dmesg_on_journalctl_timeout(self):
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
            snapshot = _kernel_log_snapshot()

        self.assertEqual(snapshot.text, "dmesg log")
        self.assertIsNone(snapshot.cursor)
        self.assertTrue(snapshot.query_succeeded)
        self.assertEqual(snapshot.backend, "dmesg")

    def test_kernel_log_snapshot_handles_log_reader_oserror(self):
        """Unavailable log executables should produce an empty snapshot."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(
                 i2c_tests.subprocess,
                 "run",
                 side_effect=OSError("executable disappeared"),
             ):
            snapshot = _kernel_log_snapshot()

        self.assertEqual(snapshot.text, "")
        self.assertIsNone(snapshot.cursor)
        self.assertFalse(snapshot.query_succeeded)
        self.assertEqual(snapshot.backend, "dmesg")

    def test_kernel_log_snapshot_preserves_empty_successful_journal_when_no_log_text(self):
        """A successful empty journal read is more informative than empty dmesg."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(
                 i2c_tests.subprocess,
                 "run",
                 side_effect=[
                     mock.Mock(returncode=0, stdout=""),
                     mock.Mock(returncode=0, stdout=""),
                 ],
             ):
            snapshot = _kernel_log_snapshot()

        self.assertEqual(snapshot.text, "")
        self.assertIsNone(snapshot.cursor)
        self.assertTrue(snapshot.query_succeeded)
        self.assertEqual(snapshot.backend, "journal")

    def test_dmesg_kernel_log_snapshot_accepts_empty_success(self):
        """Empty successful dmesg output remains distinct from command failure."""
        mock = GLTTestBase.unittest.mock

        for stdout in ("", " \n\t"):
            with self.subTest(stdout=repr(stdout)), \
                 mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
                 mock.patch.object(
                     i2c_tests.subprocess,
                     "run",
                     return_value=mock.Mock(returncode=0, stdout=stdout),
                 ):
                snapshot = i2c_tests._dmesg_kernel_log_snapshot()  # pylint: disable=protected-access

            self.assertEqual(snapshot.text, "")
            self.assertIsNone(snapshot.cursor)
            self.assertTrue(snapshot.query_succeeded)
            self.assertEqual(snapshot.backend, "dmesg")

    def test_journal_kernel_log_snapshot_returns_typed_contract(self):
        """Journal snapshots expose text, cursor, and command success distinctly."""
        mock = GLTTestBase.unittest.mock
        result = mock.Mock(
            returncode=0,
            stdout="kernel line\n-- cursor: s=abc --\n",
        )

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(i2c_tests.subprocess, "run", return_value=result) as mock_run:
            snapshot = i2c_tests._journal_kernel_log_snapshot()  # pylint: disable=protected-access

            self.assertIsInstance(
                snapshot,
                i2c_tests._KernelLogSnapshot,  # pylint: disable=protected-access
            )
            self.assertEqual(snapshot.text, "kernel line\n")
            self.assertEqual(snapshot.cursor, "s=abc")
            self.assertTrue(snapshot.query_succeeded)
            self.assertEqual(snapshot.backend, "journal")
            self.assertEqual(mock_run.call_args.args[0][-1], "--show-cursor")

    def test_journal_kernel_log_snapshot_uses_after_cursor(self):
        """Post-trigger journal reads must avoid ring-buffer prefix comparisons."""
        mock = GLTTestBase.unittest.mock
        result = mock.Mock(returncode=0, stdout="new kernel line\n-- cursor: s=def --\n")

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(i2c_tests.subprocess, "run", return_value=result) as mock_run:
            i2c_tests._journal_kernel_log_snapshot(after_cursor="s=abc")  # pylint: disable=protected-access

        self.assertIn("--after-cursor=s=abc", mock_run.call_args.args[0])

    def test_journal_kernel_log_snapshot_cursor_query_never_falls_back_to_dmesg(self):
        """A cursor-scoped journal failure must not compare against unrelated dmesg."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True) \
                as mock_cli, \
             mock.patch.object(
                 i2c_tests.subprocess,
                 "run",
                 return_value=mock.Mock(returncode=1, stdout="", stderr="journal error"),
             ) as mock_run:
            snapshot = i2c_tests._journal_kernel_log_snapshot(  # pylint: disable=protected-access
                after_cursor="s=abc"
            )

        self.assertEqual(snapshot.text, "")
        self.assertIsNone(snapshot.cursor)
        self.assertFalse(snapshot.query_succeeded)
        mock_cli.assert_called_once_with("journalctl")
        self.assertEqual(mock_run.call_count, 1)

    def test_journal_kernel_log_snapshot_accepts_empty_success_without_cursor(self):
        """An empty successful journal read remains distinct from command failure."""
        mock = GLTTestBase.unittest.mock
        result = mock.Mock(returncode=0, stdout="")

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(i2c_tests.subprocess, "run", return_value=result):
            snapshot = i2c_tests._journal_kernel_log_snapshot(  # pylint: disable=protected-access
                after_cursor="s=abc"
            )

        self.assertEqual(snapshot.text, "")
        self.assertIsNone(snapshot.cursor)
        self.assertTrue(snapshot.query_succeeded)
        self.assertEqual(snapshot.backend, "journal")

    def test_journal_kernel_log_snapshot_rejects_whitespace_only_output(self):
        """Whitespace-only journal output must not become a usable baseline."""
        mock = GLTTestBase.unittest.mock
        result = mock.Mock(returncode=0, stdout=" \n\t\n-- cursor: s=abc --\n")

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(i2c_tests.subprocess, "run", return_value=result):
            snapshot = i2c_tests._journal_kernel_log_snapshot()  # pylint: disable=protected-access

        self.assertTrue(snapshot.query_succeeded)
        self.assertEqual(snapshot.text, "")
        self.assertTrue(snapshot.query_succeeded)
        self.assertEqual(snapshot.backend, "journal")

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
                     i2c_tests, "_kernel_log_snapshot",
                     side_effect=[
                         SimpleNamespace(
                             text="old tegra-i2c NVDA2017:00: i2c transfer timed out",
                         ),
                         SimpleNamespace(
                             text="old tegra-i2c NVDA2017:00: i2c transfer timed out",
                         ),
                         SimpleNamespace(
                             text="old tegra-i2c NVDA2017:00: i2c transfer timed out",
                         ),
                     ],
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
             mock.patch.object(
                 i2c_tests,
                 "_kernel_log_snapshot",
                 return_value=SimpleNamespace(text="old log"),
             ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                i2c_tests.test_tegra410_i2c_device(self.mock_sd, self.mock_logger)

        self.assertIn("not bound to tegra-i2c after rebind", str(ctx.exception))

    def test_tegra410_i2c_device_skips_when_initial_kernel_log_is_unavailable(self):
        """The test should not pass if it cannot capture a baseline kernel log."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.NvDev, "is_cpu_type", return_value=True), \
             mock.patch.object(i2c_tests.Path, "is_dir", return_value=True), \
             mock.patch.object(i2c_tests, "_tegra_i2c_devices", return_value=["NVDA2017:00"]), \
             mock.patch.object(
                 i2c_tests,
                 "_kernel_log_snapshot",
                 return_value=SimpleNamespace(text=""),
             ):
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
                 "_kernel_log_snapshot",
                 side_effect=[
                     SimpleNamespace(text="baseline log"),
                     SimpleNamespace(text="different log"),
                 ],
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
                 i2c_tests, "_kernel_log_snapshot",
                 side_effect=[
                     SimpleNamespace(text="old log"),
                     SimpleNamespace(text=f"old log\n{error}"),
                 ],
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

        open_mock.assert_called_once()
        self.assertEqual(writes, ["NVDA2017:00"])


class TestI2cAcpiRuntimePm(GLTTestBase):
    """Test Tegra I2C ACPI runtime-PM sleeping-while-atomic coverage logic."""

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

    def _log_snapshot(self, log, cursor=None, query_succeeded=None, backend="journal"):
        """Build the typed kernel-log result used by runtime-test mocks."""
        if isinstance(log, i2c_tests._KernelLogSnapshot):  # pylint: disable=protected-access
            return log
        if query_succeeded is None:
            query_succeeded = log is not None
        return i2c_tests._KernelLogSnapshot(  # pylint: disable=protected-access
            backend=backend,
            text="" if log is None else log,
            cursor=cursor,
            query_succeeded=query_succeeded,
        )

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

    @GLTTestBase.unittest.mock.patch("src.modules.i2c.glt_test_i2c.Path")
    @GLTTestBase.unittest.mock.patch("src.modules.i2c.glt_test_i2c.subprocess.run")
    def test_ipmi_topology_errors_when_udevadm_missing(self, mock_run, mock_path):
        """Missing udevadm is reported as a topology inspection error."""
        mock_path.return_value.exists.return_value = True
        mock_run.side_effect = FileNotFoundError("udevadm")

        status, reason, path = i2c_tests._ipmi_over_tegra_i2c_acpi_topology(  # pylint: disable=protected-access
            self.mock_logger
        )
        self.assertEqual(status, i2c_tests._IPMI_TEGRA_TOPOLOGY_ERROR)  # pylint: disable=protected-access
        self.assertIn("udevadm failed", reason)
        self.assertIsNone(path)

    @GLTTestBase.unittest.mock.patch("src.modules.i2c.glt_test_i2c.Path")
    @GLTTestBase.unittest.mock.patch("src.modules.i2c.glt_test_i2c.subprocess.run")
    def test_ipmi_topology_errors_when_udevadm_times_out(self, mock_run, mock_path):
        """Timed out udevadm is reported as a topology inspection error."""
        mock_path.return_value.exists.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired(["udevadm"], 10)

        status, reason, path = i2c_tests._ipmi_over_tegra_i2c_acpi_topology(  # pylint: disable=protected-access
            self.mock_logger
        )
        self.assertEqual(status, i2c_tests._IPMI_TEGRA_TOPOLOGY_ERROR)  # pylint: disable=protected-access
        self.assertIn("udevadm failed", reason)
        self.assertIsNone(path)

    @GLTTestBase.unittest.mock.patch("src.modules.i2c.glt_test_i2c.Path")
    @GLTTestBase.unittest.mock.patch("src.modules.i2c.glt_test_i2c.subprocess.run")
    def test_ipmi_topology_errors_when_udevadm_fails(self, mock_run, mock_path):
        """udevadm non-zero exit is reported as a topology inspection error."""
        mock_path.return_value.exists.return_value = True
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "missing device"

        status, reason, path = i2c_tests._ipmi_over_tegra_i2c_acpi_topology(  # pylint: disable=protected-access
            self.mock_logger
        )
        self.assertEqual(status, i2c_tests._IPMI_TEGRA_TOPOLOGY_ERROR)  # pylint: disable=protected-access
        self.assertIn("udevadm failed", reason)
        self.assertIsNone(path)

    def test_rejects_tegra_name_outside_ipmi_devpath(self):
        """A Tegra name elsewhere in udev output does not prove parentage."""
        udev_output = (
            "P: /devices/platform/OTHER0001:00/i2c-1/"
            "i2c-IPI0001:00/ipmi/ipmi0\n"
            "E: RANDOM_NOTE=NVDA2017:01\n"
        )

        self.assertIsNone(
            i2c_tests._tegra_i2c_parent_from_ipmi_devpath(  # pylint: disable=protected-access
                udev_output,
                {"NVDA2017:01"},
            )
        )

    def test_detects_atomic_sleep_signature_with_mutex_lock(self):
        """Atomic-sleep splat may contain mutex_lock instead of mutex_lock_nested."""
        log = """
        BUG: sleeping function called from invalid context at kernel/locking/mutex.c:271
        __might_sleep
        mutex_lock
        acpi_subsys_runtime_resume
        rpm_resume
        tegra_i2c_xfer
        ipmi_ssif_thread
        """

        self.assertTrue(
            i2c_tests._has_tegra_i2c_acpi_irq_safe_bug(log)  # pylint: disable=protected-access
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

    def test_kernel_log_snapshot_falls_back_after_empty_journal(self):
        """Empty journal output is not treated as a usable kernel log."""
        mock = GLTTestBase.unittest.mock

        with mock.patch.object(i2c_tests.glt_utils, "is_cli_available", return_value=True), \
             mock.patch.object(
                 i2c_tests.subprocess,
                 "run",
                 side_effect=[
                     mock.Mock(returncode=0, stdout=""),
                     mock.Mock(returncode=0, stdout="dmesg line"),
                 ],
             ):
            snapshot = i2c_tests._kernel_log_snapshot()  # pylint: disable=protected-access

        self.assertEqual(snapshot.text, "dmesg line")
        self.assertIsNone(snapshot.cursor)
        self.assertTrue(snapshot.query_succeeded)
        self.assertEqual(snapshot.backend, "dmesg")

    def _run_runtime_test(
        self,
        *,
        acpi=True,
        config="debug",
        ipmitool_available=True,
        logs="baseline log\n",
        baseline_backend="journal",
        baseline_cursor=None,
        dmesg_side_effect=None,
        post_cursor_log="",
        post_cursor_query_succeeded=True,
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
        baseline_log = logs[0] if isinstance(logs, (list, tuple)) else logs
        post_full_log = logs[-1] if isinstance(logs, (list, tuple)) else logs
        baseline_snapshot = self._log_snapshot(
            baseline_log, cursor=baseline_cursor, backend=baseline_backend
        )
        journal_full_snapshot = self._log_snapshot(post_full_log, backend="journal")
        dmesg_full_snapshot = self._log_snapshot(post_full_log, backend="dmesg")
        post_cursor_snapshot = self._log_snapshot(
            post_cursor_log,
            cursor=None if not post_cursor_log else "s=post",
            query_succeeded=post_cursor_query_succeeded,
            backend="journal",
        )

        def journal_snapshot(after_cursor=None):
            return post_cursor_snapshot if after_cursor else journal_full_snapshot

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
             mock.patch("src.modules.i2c.glt_test_i2c._kernel_log_snapshot") as mock_get_log, \
             mock.patch(
                 "src.modules.i2c.glt_test_i2c._dmesg_kernel_log_snapshot",
                 return_value=dmesg_full_snapshot,
             ) as mock_get_dmesg_log, \
             mock.patch(
                 "src.modules.i2c.glt_test_i2c._journal_kernel_log_snapshot",
                 side_effect=journal_snapshot,
             ) as mock_get_journal_log, \
             mock.patch(
                 "src.modules.i2c.glt_test_i2c.check_kernel_config.parse_kconfig",
                 return_value=config,
             ):
            mock_path.return_value.is_dir.return_value = acpi
            mock_glt_utils.is_cli_available.return_value = ipmitool_available
            if isinstance(runtime_status, (list, tuple)):
                mock_runtime_status.side_effect = list(runtime_status)
            else:
                mock_runtime_status.return_value = runtime_status
            mock_get_log.return_value = baseline_snapshot
            if dmesg_side_effect is not None:
                mock_get_dmesg_log.side_effect = dmesg_side_effect
            if run_side_effect is not None:
                mock_run.side_effect = run_side_effect
            else:
                mock_run.return_value = run_result

            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe(
                None, self.mock_logger
            )
            return mock.Mock(
                run=mock_run,
                kernel_log=mock_get_log,
                dmesg_kernel_log=mock_get_dmesg_log,
                journal_kernel_log=mock_get_journal_log,
                runtime_status=mock_runtime_status,
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
            self._run_runtime_test(logs=None)
        self.assertIn("Cannot read kernel log", str(context.exception))
        self.assertIn("sleeping-while-atomic", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_runtime_test_skips_when_kernel_config_unavailable(self):
        """Runtime test skips distinctly when kernel config cannot be read."""
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

    def test_runtime_test_fails_when_runtime_status_is_active(self):
        """Runtime test fails when runtime resume was not confirmed at trigger."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(runtime_status="active")
        self.assertIn("runtime_status at trigger was active", str(context.exception))
        self.assertIn("runtime resume was not confirmed", str(context.exception))
        self.assertNotIn("before trigger", str(context.exception))

    def test_runtime_test_samples_runtime_status_at_each_trigger(self):
        """runtime_status is sampled per trigger attempt, not only before quiet."""
        mocks = self._run_runtime_test()
        self.assertEqual(
            mocks.runtime_status.call_count,
            i2c_tests._I2C_ACPI_RUNTIME_PM_IPMI_TRIGGER_ATTEMPTS,  # pylint: disable=protected-access
        )

    def test_runtime_test_fails_when_never_suspended_across_triggers(self):
        """If no trigger attempt observes suspended, absence of the splat is a proof failure."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(
                runtime_status=["active", "active", "suspending", "active", "active"]
            )
        self.assertIn("runtime_status at trigger was active", str(context.exception))
        self.assertNotIn("before trigger", str(context.exception))

    def test_runtime_test_fails_when_topology_inspection_errors(self):
        """Topology inspection errors fail instead of being treated as unsupported HW."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(topology=("error", "udevadm failed", None))
        self.assertIn("udevadm failed", str(context.exception))

    def test_runtime_test_fails_on_non_contiguous_kernel_log(self):
        """Non-contiguous post log without the target splat is a proof failure."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(
                logs=("baseline log\n", "unrelated replacement log\n")
            )
        self.assertIn("non-contiguous", str(context.exception))
        self.assertIn("cannot reliably diff", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_runtime_test_dmesg_non_contiguous_log_does_not_reuse_stale_overlap(self):
        """A replaced dmesg log must not combine stale frames into a BUG match."""
        baseline = "old line\n" * 20 + "BUG: sleeping function called from invalid context\n"
        replacement = "replacement log\nmutex_lock\nacpi_subsys_runtime_resume\n"

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(
                logs=(baseline, replacement),
                baseline_backend="dmesg",
            )
        self.assertIn("non-contiguous", str(context.exception))
        self.assertNotIn("Found sleeping-while-atomic BUG", str(context.exception))

    def test_runtime_test_fails_on_non_contiguous_kernel_log_with_signature(self):
        """A dropped cursor must not hide a concrete DGX-17014 splat still in the log."""
        post_log = (
            "replacement log\n"
            "BUG: sleeping function called from invalid context\n"
            "mutex_lock\n"
            "acpi_subsys_runtime_resume\n"
            "rpm_resume\n"
            "tegra_i2c_xfer\n"
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(logs=("baseline log\n", post_log))
        self.assertIn("Found sleeping-while-atomic BUG", str(context.exception))
        self.assertIn("tegra_i2c_xfer", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_runtime_test_fails_when_post_trigger_kernel_log_unavailable(self):
        """Runtime test fails if the post-trigger kernel log cannot be collected."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(logs=("baseline log\n", None))
        self.assertIn("post-trigger kernel log", str(context.exception))
        self.assertEqual(
            i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
            GLTCore.GLTTestResult.FAIL,
        )

    def test_runtime_test_accepts_empty_successful_cursor_delta(self):
        """A quiet journal cursor delta after exercising IPMI is a clean new log."""
        mocks = self._run_runtime_test(
            logs="baseline log\n",
            baseline_cursor="s=baseline",
            post_cursor_log="",
            post_cursor_query_succeeded=True,
        )

        self.assertEqual(mocks.kernel_log.call_count, 1)
        mocks.journal_kernel_log.assert_called_once_with("s=baseline")
        mocks.dmesg_kernel_log.assert_not_called()

    def test_runtime_test_cursorless_journal_baseline_uses_journal_full_snapshot(self):
        """A cursorless journal baseline must not switch the post read to dmesg."""
        mocks = self._run_runtime_test(
            logs=("baseline log\n", "baseline log\nnew journal line\n"),
            baseline_backend="journal",
            baseline_cursor=None,
        )

        mocks.journal_kernel_log.assert_called_once_with(None)
        mocks.dmesg_kernel_log.assert_not_called()

    def test_runtime_test_fails_on_empty_cursorless_journal_snapshot(self):
        """An empty full journal snapshot is a proof failure without a cursor."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(
                logs=("baseline log\n", ""),
                baseline_backend="journal",
                baseline_cursor=None,
            )

        self.assertIn("snapshot is empty", str(context.exception))

    def test_runtime_test_fails_on_pre_existing_signature(self):
        """Runtime test fails clearly if the current boot already has the splat."""
        log = (
            "BUG: sleeping function called from invalid context\n"
            "mutex_lock\n"
            "acpi_subsys_runtime_resume\n"
            "rpm_resume\n"
            "tegra_i2c_xfer\n"
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(logs=log)
        self.assertIn("already contains sleeping-while-atomic BUG", str(context.exception))

    def test_runtime_test_fails_when_signature_appears(self):
        """Runtime test fails on the Tegra I2C ACPI runtime-PM atomic-sleep signature."""
        log = (
            "baseline log\n"
            "BUG: sleeping function called from invalid context\n"
            "mutex_lock\n"
            "acpi_subsys_runtime_resume\n"
            "rpm_resume\n"
            "tegra_i2c_xfer\n"
        )

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_runtime_test(logs=("baseline log\n", log))
        self.assertIn("Found sleeping-while-atomic BUG", str(context.exception))
        self.assertIn("tegra_i2c_xfer", str(context.exception))

    def test_runtime_test_fails_on_signature_before_config_skip(self):
        """Concrete BUG evidence must fail before config observability skips."""
        log = (
            "baseline log\n"
            "BUG: sleeping function called from invalid context\n"
            "mutex_lock\n"
            "acpi_subsys_runtime_resume\n"
            "rpm_resume\n"
            "tegra_i2c_xfer\n"
        )

        for config in (None, {}):
            with self.subTest(config=config), \
                 self.assertRaises(GLTCore.GLTTestFail) as context:
                self._run_runtime_test(config=config, logs=("baseline log\n", log))
            self.assertIn("Found sleeping-while-atomic BUG", str(context.exception))
            self.assertEqual(
                i2c_tests.test_tegra_i2c_acpi_runtime_pm_irq_safe.attr.result[0],
                GLTCore.GLTTestResult.FAIL,
            )

    def test_runtime_test_finds_signature_across_log_boundary(self):
        """Runtime test keeps enough baseline overlap to catch boundary splats."""
        baseline = "old line\n" * 20 + (
            "BUG: sleeping function called from invalid context\n"
            "__might_sleep\n"
            "mutex_lock\n"
            "acpi_subsys_runtime_resume\n"
            "rpm_resume\n"
        )

        with self.assertRaises(GLTCore.GLTTestFail):
            self._run_runtime_test(logs=(baseline, baseline + "tegra_i2c_xfer\n"))

    def test_runtime_test_fails_when_ipmitool_command_fails(self):
        """Runtime test fails when the trigger command fails."""
        with self.assertRaises(GLTCore.GLTTestFail):
            self._run_runtime_test(
                run_returncode=1,
                run_stderr="ipmi error",
                run_stdout="",
            )

    def test_runtime_test_fails_when_ipmitool_times_out(self):
        """Runtime test fails when the trigger command times out."""
        with self.assertRaises(GLTCore.GLTTestFail):
            self._run_runtime_test(
                run_side_effect=subprocess.TimeoutExpired(["ipmitool"], 10)
            )

    def test_runtime_test_fails_when_ipmitool_raises_oserror(self):
        """A missing trigger executable must become a controlled test failure."""
        with self.assertRaises(GLTCore.GLTTestFail):
            self._run_runtime_test(run_side_effect=OSError("ipmitool disappeared"))

    def test_runtime_test_fails_when_ipmi_device_unavailable(self):
        """Runtime test fails when ipmitool reports unavailable IPMI device."""
        with self.assertRaises(GLTCore.GLTTestFail):
            self._run_runtime_test(run_stdout="Device Available          : no\n")

    def test_runtime_test_skips_when_not_acpi(self):
        """Runtime test skips on non-ACPI systems."""
        with self.assertRaises(GLTCore.GLTTestSkip):
            self._run_runtime_test(acpi=False)

    def test_runtime_test_skips_without_ipmi_tegra_topology(self):
        """Runtime test skips with topology_reason when IPMI is not on ACPI Tegra I2C."""
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_runtime_test(topology=self._non_matching_topology())
        self.assertEqual(str(context.exception), "no topology")

    def test_runtime_test_skips_with_missing_ipmi_device_reason(self):
        """NO_MATCH skip preserves the missing-/dev/ipmi0 topology_reason."""
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_runtime_test(
                topology=("no_match", "/dev/ipmi0 is not present.", None)
            )
        self.assertEqual(str(context.exception), "/dev/ipmi0 is not present.")

    def test_runtime_test_passes_when_debug_config_enabled_and_signature_absent(self):
        """Runtime test passes when debug config can observe the issue and logs are clean."""
        mocks = self._run_runtime_test()

        self.assertEqual(
            mocks.run.call_count,
            i2c_tests._I2C_ACPI_RUNTIME_PM_IPMI_TRIGGER_ATTEMPTS,  # pylint: disable=protected-access
        )
        self.assertEqual(
            mocks.runtime_status.call_count,
            i2c_tests._I2C_ACPI_RUNTIME_PM_IPMI_TRIGGER_ATTEMPTS,  # pylint: disable=protected-access
        )
