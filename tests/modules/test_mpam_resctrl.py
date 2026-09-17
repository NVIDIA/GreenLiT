"""Unit tests for ARM64 MPAM resctrl schema and workload policy."""
# pylint: disable=wrong-import-position

from contextlib import contextmanager, ExitStack, nullcontext
from io import StringIO
from itertools import chain, repeat
import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_mpam_resctrl as mpam_resctrl


class TestMpamMbmCounterGrowth(GLTTestBase):
    """Exercise MBM traffic resource decisions without touching hardware."""

    @contextmanager
    def _growth_environment(self, *, free_kib=(2048 * 1024,),
                            returncode=0, samples=(100, 200), poll_sequence=()):
        """Mock MBM hardware and assert resource cleanup after each scenario."""
        group = f"{mpam_resctrl.RESCTRL_MOUNT_POINT}/mon_groups/glt_mbm_growth_123_456"
        states = dict.fromkeys(range(len(free_kib)), "_")
        sample_values = iter(samples)
        mock_files = self.mock.mock_open()
        mock_popen = self.mock.Mock()
        mock_popen.return_value.pid = 789
        mock_popen.return_value.poll.side_effect = chain(poll_sequence, repeat(returncode))
        mock_popen.return_value.wait.return_value = returncode
        mock_rmdir = self.mock.Mock()

        def open_file(path, mode="r", **_kwargs):
            """Provide node memory metadata and the writable assignment file."""
            if path == group + "/mbm_MB_assignments" and mode == "w":
                return mock_files.return_value
            for domain, available in enumerate(free_kib):
                if path == f"/sys/devices/system/node/node{domain}/meminfo":
                    content = f"Node {domain} MemTotal: 8388608 kB\n"
                    if available is not None:
                        content += f"Node {domain} MemFree: {available} kB\n"
                    return StringIO(content)
            raise AssertionError(f"Unexpected file access: {path} ({mode})")

        def read_sysfs(path):
            """Return assignment state, counter availability, and MBM samples."""
            if path.endswith("/mbm_MB_assignments"):
                return "mbm_total_bytes:" + ";".join(
                    f"{domain}={state}" for domain, state in states.items())
            if path.endswith("/available_mbm_cntrs"):
                return ";".join(f"{domain}=1" for domain in states)
            if path.endswith("/mbm_total_bytes"):
                return str(next(sample_values))
            raise AssertionError(f"Unexpected sysfs access: {path}")

        def write_assignment(value):
            """Track counter assignment and release for cleanup assertions."""
            domain, state = value.strip().split(":")[1].split("=")
            if domain == "*":
                states.update(dict.fromkeys(states, state))
            else:
                states[int(domain)] = state

        mock_files.side_effect = open_file
        mock_files.return_value.write.side_effect = write_assignment
        patches = (
            self.mock.patch("builtins.open", mock_files),
            self.mock.patch.object(mpam_resctrl, "_managed_resctrl_mount",
                                   return_value=nullcontext((True, None))),
            self.mock.patch.object(mpam_resctrl, "_cpus_of_node", return_value=[0]),
            self.mock.patch.object(mpam_resctrl.glt_utils, "is_cli_available",
                                   return_value=True),
            self.mock.patch.object(mpam_resctrl.glt_utils, "read_sysfs_value",
                                   side_effect=read_sysfs),
            self.mock.patch.object(mpam_resctrl.os, "getpid", return_value=123),
            self.mock.patch.object(mpam_resctrl.os, "killpg"),
            self.mock.patch.object(mpam_resctrl.os, "mkdir"),
            self.mock.patch.object(mpam_resctrl.os, "rmdir", mock_rmdir),
            self.mock.patch.object(mpam_resctrl.os.path, "isdir", return_value=True),
            self.mock.patch.object(mpam_resctrl.os.path, "isfile",
                                   side_effect=lambda path: not path.endswith("/domain_id")),
            self.mock.patch.object(mpam_resctrl.subprocess, "Popen", mock_popen),
            self.mock.patch.object(mpam_resctrl.time, "monotonic", side_effect=range(20)),
            self.mock.patch.object(mpam_resctrl.time, "sleep"),
            self.mock.patch.object(mpam_resctrl.time, "time_ns", return_value=456),
        )
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            yield mock_popen
            mock_files.return_value.write.assert_any_call("mbm_total_bytes:*=_\n")
            self.assertTrue(all(state == "_" for state in states.values()))
            mock_rmdir.assert_called_once_with(group)

    def test_growth_without_domain_id_at_exact_memory_budget(self):
        """Counter growth works without domain_id and with exactly 2048 MiB free."""
        with self._growth_environment() as mock_popen:
            mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

            mock_popen.assert_called_once()
            self.assertEqual(mock_popen.call_args.args[0][-5:],
                             ["0", "0", "4", "160M", "10s"])

    def test_insufficient_free_memory_skips_before_spawning(self):
        """One KiB below the budget skips and releases the monitor group."""
        with self._growth_environment(free_kib=(2048 * 1024 - 1,)) as mock_popen:
            with self.assertRaisesRegex(GLTCore.GLTTestSkip, "2048 MiB of free memory"):
                mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

            mock_popen.assert_not_called()

    def test_late_counter_decrease_fails_and_stops_workload(self):
        """A dip above baseline fails and terminates the still-running workload."""
        with self._growth_environment(samples=(100, 200, 150, 300),
                                      poll_sequence=(None, None)) as mock_popen:
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "200 -> 150"):
                mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

            mpam_resctrl.os.killpg.assert_called_once_with(789, mpam_resctrl.signal.SIGTERM)
            mock_popen.return_value.wait.assert_called_once_with(timeout=5)

    def test_later_domain_with_free_memory_is_selected(self):
        """A resource-limited first candidate does not hide a usable node."""
        with self._growth_environment(free_kib=(1024, 2048 * 1024)) as mock_popen:
            mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

            self.assertEqual(mock_popen.call_args.args[0][-5:-3], ["1", "1"])

    def test_live_samples_allow_plateau_before_growth(self):
        """Multiple live samples may plateau before eventually increasing."""
        with self._growth_environment(samples=(100, 100, 150, 200),
                                      poll_sequence=(None, None)) as mock_popen:
            mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

            self.assertEqual(mock_popen.return_value.poll.call_count, 4)
            mock_popen.return_value.wait.assert_called_once_with(timeout=5)
            mpam_resctrl.os.killpg.assert_not_called()

    def test_missing_memfree_fails(self):
        """Missing memory metadata remains a failure rather than a skip."""
        with self._growth_environment(free_kib=(None,)) as mock_popen:
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "MemFree"):
                mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

            mock_popen.assert_not_called()

    def test_resource_exit_status_skips(self):
        """Resource and unsupported-workload exits skip even with flat counters."""
        for returncode in (3, 4):
            with self.subTest(returncode=returncode):
                with self._growth_environment(returncode=returncode,
                                              samples=(100, 100)) as mock_popen:
                    with self.assertRaisesRegex(GLTCore.GLTTestSkip,
                                                f"status {returncode}"):
                        mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

                    mock_popen.assert_called_once()

    def test_successful_workload_without_counter_growth_fails(self):
        """A successful stress-ng exit still requires growing MBM counters."""
        with self._growth_environment(samples=(100, 100)):
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "did not grow"):
                mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)

    def test_workload_failure_remains_failure(self):
        """A genuine stress-ng failure cannot be masked by counter growth."""
        with self._growth_environment(returncode=2):
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "status 2"):
                mpam_resctrl.test_mpam_mbm_counter_growth(None, self.mock_logger)


class TestMpamResctrlInfo(GLTTestBase):
    """Validate Grace/Vera-specific MPAM resctrl info behavior."""

    def _patch_mounted_info(self, mb_mon_present):
        def exists(path):
            if path.endswith("MB_MON"):
                return mb_mon_present
            return True

        def isdir(path):
            if path.endswith("MB_MON"):
                return mb_mon_present
            return True

        return (
            self.mock.patch.object(mpam_resctrl, "_resctrl_is_mounted",
                                   return_value=True),
            self.mock.patch.object(mpam_resctrl.os.path, "isdir",
                                   side_effect=isdir),
            self.mock.patch.object(mpam_resctrl.os.path, "exists",
                                   side_effect=exists),
            self.mock.patch.object(mpam_resctrl.os.path, "isfile",
                                   return_value=True),
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_grace_does_not_require_mb_mon(self, mock_nvdev):
        """Grace expects the base schema but not the MB_MON directory."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        patches = self._patch_mounted_info(mb_mon_present=False)
        mock_open = self.mock.mock_open(read_data="mbm_total_bytes\n")

        with patches[0], patches[1], patches[2], patches[3], \
                self.mock.patch("builtins.open", mock_open):
            mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        mock_open.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_vera_missing_mb_mon_fails(self, mock_nvdev):
        """Vera is the full-schema control, so missing MB_MON is a regression."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = self._patch_mounted_info(mb_mon_present=False)

        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        self.assertIn("MB_MON (expected directory) is missing",
                      str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_unknown_cpu_type_skips(self, mock_nvdev):
        """The platform-specific schema is not applicable to other CPUs."""
        mock_nvdev.return_value.is_cpu_type.return_value = False
        patches = self._patch_mounted_info(mb_mon_present=True)

        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        self.assertIn("not an NVIDIA Grace or Vera CPU", str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_vera_mb_mon_requires_mbm_total_bytes(self, mock_nvdev):
        """Vera full-schema pass includes the MBM total-bytes event."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = self._patch_mounted_info(mb_mon_present=True)
        mock_open = self.mock.mock_open(read_data="mbm_total_bytes\n")

        with patches[0], patches[1], patches[2], patches[3], \
                self.mock.patch("builtins.open", mock_open):
            mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        mock_open.assert_called_once_with(
            f"{mpam_resctrl.RESCTRL_MOUNT_POINT}/info/MB_MON/mon_features",
            "r",
            encoding="utf-8",
        )

    def test_info_unmounts_when_post_mount_check_fails(self):
        """A test-owned mount is cleaned up even when validation fails early."""
        with self.mock.patch.object(mpam_resctrl.glt_utils, "run_command") as mock_run, \
                self.mock.patch.object(mpam_resctrl.glt_utils, "is_cli_available",
                                       return_value=True), \
                self.mock.patch.object(mpam_resctrl, "NvDev"), \
                self.mock.patch.object(mpam_resctrl, "_resctrl_is_mounted",
                                       side_effect=(False, False)):
            with self.assertRaises(GLTCore.GLTTestFail):
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(
            mock_run.call_args_list[0].args[0],
            ["mount", "-t", "resctrl", "resctrl", "/sys/fs/resctrl"],
        )
        self.assertEqual(
            mock_run.call_args_list[1].args[0],
            ["umount", "/sys/fs/resctrl"],
        )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
