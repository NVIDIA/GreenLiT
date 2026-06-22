"""Unit tests for CPPC cpufreq sysfs coverage."""

from tests.test_base import GLTTestBase
from src.modules.cpu import glt_test_cpufreq_stress as cpufreq


class TestCppcSysfs(GLTTestBase):
    """Test CPPC sysfs helper validation."""

    def test_validate_cppc_sysfs_helpers_round_trips_limits_and_perf_limited(self):
        """A cppc_cpufreq policy with required sysfs files validates cleanly."""
        files = {
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/perf_limited": "0\n",
        }
        writes = []

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            return files[path].strip()

        def fake_write(path, value):
            writes.append((path, value))
            files[path] = f"{value}\n"

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read), \
             self.mock.patch.object(cpufreq.glt_utils, "write_sysfs_value",
                                    side_effect=fake_write):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertIn(
            ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq", "450000"),
            writes,
        )
        self.assertIn(
            ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "3447000"),
            writes,
        )
        self.assertIn(
            ("/sys/devices/system/cpu/cpu0/cpufreq/perf_limited", "0x3"),
            writes,
        )

    def test_validate_cppc_sysfs_helpers_fails_without_perf_limited(self):
        """Missing perf_limited reports the missing patch-facing ABI."""
        files = {
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq": "3447000\n",
        }

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            if path not in files:
                raise OSError("missing")
            return files[path].strip()

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read), \
             self.mock.patch.object(cpufreq.glt_utils, "write_sysfs_value"):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertFalse(ok)
        self.assertIn("perf_limited", reason)

    def test_validate_cppc_sysfs_helpers_restores_limits_after_failure(self):
        """Original scaling limits are restored when later sysfs validation fails."""
        files = {
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/perf_limited": "0\n",
        }
        writes = []

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            return files[path].strip()

        def fake_write(path, value):
            writes.append((path, value))
            if path.endswith("/perf_limited"):
                raise OSError("write failed")
            files[path] = f"{value}\n"

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read), \
             self.mock.patch.object(cpufreq.glt_utils, "write_sysfs_value",
                                    side_effect=fake_write):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertFalse(ok)
        self.assertIn("write failed", reason)
        self.assertEqual(
            writes[-2:],
            [
                ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "3447000"),
                ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq", "450000"),
            ],
        )

    def test_validate_cppc_sysfs_helpers_fails_without_cppc_driver(self):
        """A non-CPPC policy is not a valid target for this coverage."""

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            if path.endswith("/scaling_driver"):
                return "acpi-cpufreq"
            raise AssertionError(f"unexpected read: {path}")

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertFalse(ok)
        self.assertIn("cppc_cpufreq", reason)
