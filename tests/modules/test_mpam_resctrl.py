"""Unit tests for ARM64 MPAM resctrl schema and workload policy."""
# pylint: disable=wrong-import-position,protected-access,too-many-lines,too-many-public-methods

from contextlib import contextmanager, ExitStack, nullcontext
from functools import partial
import errno
from io import StringIO
from itertools import chain, repeat
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tests.test_base import GLTTestBase
from tests.common.test_glt_proc import proc_task_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from common import nv_device
from common import resctrl_utils
from greenlit import greenlit
from src.modules.arm64 import glt_test_mpam_resctrl as mpam_resctrl


class TestMpamResctrlDependencies(GLTTestBase):
    """Discover real MPAM prerequisites without executing hardware bodies."""

    INIT = "arm64:test_mpam_driver_init_registers"
    MOUNT = "arm64:test_mpam_resctrl_mount"
    INFO = "arm64:test_mpam_resctrl_info"

    def _discover_scheduler(self, cpu, selected):
        """Use fresh CPU-specific metadata and restore discovery's module entry."""
        module_name = "glt_test_mpam_resctrl"
        missing = object()
        previous = sys.modules.get(module_name, missing)
        nvdev = self.mock.Mock()
        nvdev.is_cpu_type.side_effect = lambda name: name == cpu
        try:
            with self.mock.patch.object(nv_device, "NvDev", return_value=nvdev), \
                    self.mock.patch.object(greenlit.glob, "glob",
                                           return_value=[mpam_resctrl.__file__]):
                test_objs, error, exit_code = greenlit.discover_target_test_cases(
                    ["arm64"], [selected], {"standard"})
        finally:
            if previous is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        self.assertEqual((exit_code, error), (0, ""))
        scheduler = GLTCore.GLTScheduler(test_objs)
        scheduler.sort()
        return scheduler

    def _assert_pass_required(self, scheduler, prerequisite, dependent):
        """A skipped, failed, or unexecuted prerequisite must block its dependent."""
        prerequisite_attr = scheduler.test_map[prerequisite].__wrapped__.attr
        requires = scheduler.test_map[dependent].__wrapped__.requires
        for status in (GLTCore.GLTTestResult.PASS, GLTCore.GLTTestResult.FAIL,
                       GLTCore.GLTTestResult.SKIP, GLTCore.GLTTestResult.NA):
            with self.subTest(status=status):
                prerequisite_attr.result = (status, "")
                self.assertEqual(requires.can_start(scheduler)[0],
                                 status == GLTCore.GLTTestResult.PASS)

    def test_grace_info_selection_reaches_shared_mount_and_info(self):
        """Grace info must not pull in the Vera-only initializer that would skip."""
        scheduler = self._discover_scheduler("grace", self.INFO)

        self.assertEqual(list(scheduler.test_map), [self.MOUNT, self.INFO])

    def test_grace_mount_selection_does_not_require_vera_initializer(self):
        """Selecting only the shared mount must remain runnable on Grace."""
        scheduler = self._discover_scheduler("grace", self.MOUNT)

        self.assertEqual(list(scheduler.test_map), [self.MOUNT])

    def test_grace_info_still_requires_mount_pass(self):
        """Removing Grace's Vera prerequisite must preserve info's mount gate."""
        scheduler = self._discover_scheduler("grace", self.INFO)

        self._assert_pass_required(scheduler, self.MOUNT, self.INFO)

    def test_vera_preserves_initializer_selection_order_and_pass_gate(self):
        """Vera must initialize successfully before mounting or inspecting resctrl."""
        for selected, expected in (
                (self.MOUNT, [self.INIT, self.MOUNT]),
                (self.INFO, [self.INIT, self.MOUNT, self.INFO])):
            with self.subTest(selected=selected):
                scheduler = self._discover_scheduler("vera", selected)

                self.assertEqual(list(scheduler.test_map), expected)
                self._assert_pass_required(scheduler, self.INIT, self.MOUNT)

    def test_unknown_cpu_preserves_initializer_selection_order_and_pass_gate(self):
        """Failure to identify Grace must retain the conservative initializer gate."""
        for selected, expected in (
                (self.MOUNT, [self.INIT, self.MOUNT]),
                (self.INFO, [self.INIT, self.MOUNT, self.INFO])):
            with self.subTest(selected=selected):
                scheduler = self._discover_scheduler("unknown", selected)

                self.assertEqual(list(scheduler.test_map), expected)
                self._assert_pass_required(scheduler, self.INIT, self.MOUNT)

    def test_alternating_cpu_discovery_does_not_leak_dependencies(self):
        """Fresh Grace/Vera/Grace discovery must keep each CPU's own prerequisites."""
        selections = []
        for cpu in ("grace", "vera", "grace"):
            scheduler = self._discover_scheduler(cpu, self.INFO)
            selections.append(list(scheduler.test_map))

        self.assertEqual(selections, [
            [self.MOUNT, self.INFO],
            [self.INIT, self.MOUNT, self.INFO],
            [self.MOUNT, self.INFO],
        ])


class TestMpamMbaHardlimExposure(GLTTestBase):
    """Check complete schemata parsing through the public exposure test."""

    SCHEMATA = (
        "\nL3:0=ffff;2=ffff\n\n"
        "MB:0=100;2=100\n\n"
        "MB_NODE:7=100\n"
        "MB_HLIM:2=0;0=1\n\n"
        "L2:0=ff\n\n"
    )

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        resctrl = Path(temp_dir.name)
        mb_info = resctrl / "info" / "MB"
        mb_info.mkdir(parents=True)
        (mb_info / "max_lim").write_text("0\n", encoding="ascii")
        (resctrl / "info" / "MB_HLIM").mkdir()
        self.schemata = resctrl / "schemata"
        nvdev = self.mock.Mock()
        nvdev.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = (
            self.mock.patch.object(mpam_resctrl, "NvDev", return_value=nvdev),
            self.mock.patch.object(resctrl_utils, "RESCTRL_MOUNT_POINT",
                                   str(resctrl)),
            self.mock.patch.object(resctrl_utils, "managed_mount",
                                   return_value=nullcontext((True, None))),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_late_duplicate_or_malformed_line_fails(self):
        """A valid resource pair cannot hide later duplicate or malformed lines."""
        for line in ("MB:0=100;2=100", "MB_HLIM:2=0;0=1", "MB", "MB_HLIM"):
            with self.subTest(line=line):
                self.schemata.write_text(self.SCHEMATA + line + "\n", encoding="ascii")

                with self.assertRaisesRegex(GLTCore.GLTTestFail,
                                            "Malformed or duplicate .* line"):
                    mpam_resctrl.test_mpam_mba_hardlim_exposure(None, self.mock_logger)

    def test_late_malformed_domain_entries_fail(self):
        """A valid MB line cannot hide malformed entries in the later MB_HLIM line."""
        self.schemata.write_text(
            self.SCHEMATA.replace("MB_HLIM:2=0;0=1", "MB_HLIM:2=0;broken"),
            encoding="ascii")

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "Malformed MB_HLIM entry"):
            mpam_resctrl.test_mpam_mba_hardlim_exposure(None, self.mock_logger)

    def test_separated_resources_pass_in_either_order(self):
        """Matching domains pass with unrelated resources and blank lines around them."""
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                lines = self.SCHEMATA.splitlines()
                if reverse:
                    lines.reverse()
                self.schemata.write_text("\n".join(lines) + "\n", encoding="ascii")

                mpam_resctrl.test_mpam_mba_hardlim_exposure(None, self.mock_logger)


class TestMpamMbmCounterGrowth(GLTTestBase):
    """Exercise MBM traffic resource decisions without touching hardware."""

    @contextmanager
    def _growth_environment(self, *, free_kib=(2048 * 1024,),
                            returncode=0, samples=(100, 200), poll_sequence=()):
        """Mock MBM hardware and assert resource cleanup after each scenario."""
        group = f"{resctrl_utils.RESCTRL_MOUNT_POINT}/mon_groups/glt_mbm_growth_123_456"
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
            self.mock.patch.object(resctrl_utils, "managed_mount",
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


class TestMpamMscDomainCompleteness(GLTTestBase):
    """Check domain validation results when an owned mount cannot be removed."""

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        debugfs = root / "mpam"
        for component in (0, 2):
            (debugfs / "class.255" / f"comp.{component}").mkdir(parents=True)
        resctrl = root / "resctrl"
        resctrl.mkdir()
        self.schemata = resctrl / "schemata"
        mounts = root / "mounts"
        mounts.write_text("", encoding="utf-8")

        def run_command(command, failure_message):
            if command == ["mount", "-t", "resctrl", "resctrl", str(resctrl)]:
                mounts.write_text(
                    f"resctrl {resctrl} resctrl rw 0 0\n", encoding="utf-8")
                return ""
            self.assertEqual(command, ["umount", str(resctrl)])
            raise RuntimeError(failure_message)

        patches = (
            self.mock.patch.object(mpam_resctrl, "MPAM_DEBUGFS_PATH", str(debugfs)),
            self.mock.patch.multiple(
                resctrl_utils, RESCTRL_MOUNT_POINT=str(resctrl), PROC_MOUNTS=str(mounts)),
            self.mock.patch.object(mpam_resctrl.glt_utils, "is_cli_available",
                                   return_value=True),
            self.mock.patch.object(mpam_resctrl.glt_utils, "run_command",
                                   side_effect=run_command),
            self.mock.patch.object(mpam_resctrl.subprocess, "run",
                                   side_effect=AssertionError("Unexpected subprocess")),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_cleanup_failure_fails_matching_domains(self):
        """Successful validation must not pass when its resctrl mount leaks."""
        self.schemata.write_text("MB:0=100;2=100\n", encoding="utf-8")

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "Failed to unmount resctrl"):
            mpam_resctrl.test_mpam_msc_domain_completeness(None, self.mock_logger)

        self.mock_logger.warning.assert_called_once_with("Failed to unmount resctrl.")

    def test_cleanup_failure_preserves_domain_mismatch(self):
        """A cleanup error must not replace the original missing-domain failure."""
        self.schemata.write_text("\nL3:0=ffff;2=ffff\n\nMB:0=100\n\n", encoding="utf-8")

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "Check kernel commit 2eacf36cf734"):
            mpam_resctrl.test_mpam_msc_domain_completeness(None, self.mock_logger)

        self.mock_logger.warning.assert_called_once_with("Failed to unmount resctrl.")

    def test_cleanup_failure_preserves_missing_control_skip(self):
        """An unavailable MB control stays inconclusive when cleanup also fails."""
        self.schemata.write_text("L3:0=ffff;2=ffff\n", encoding="utf-8")

        with self.assertRaisesRegex(GLTCore.GLTTestSkip, "resctrl publishes no MB control"):
            mpam_resctrl.test_mpam_msc_domain_completeness(None, self.mock_logger)

        self.mock_logger.warning.assert_called_once_with("Failed to unmount resctrl.")


class TestMpamResctrlInfo(GLTTestBase):
    """Validate Grace/Vera-specific MPAM resctrl info behavior."""

    def _patch_mounted_info(self, mb_mon_present, *, mb_hlim_present=True,
                            mb_hlim_isdir=True, required_entry_defect=(None, None)):
        def exists(path):
            entry = mpam_resctrl.os.path.basename(path)
            if entry == required_entry_defect[0]:
                return required_entry_defect[1] != "missing"
            if entry == "MB_HLIM":
                return mb_hlim_present
            if path.endswith("MB_MON"):
                return mb_mon_present
            return True

        def isdir(path):
            entry = mpam_resctrl.os.path.basename(path)
            if entry == required_entry_defect[0]:
                return False
            if entry == "MB_HLIM":
                return mb_hlim_present and mb_hlim_isdir
            if path.endswith("MB_MON"):
                return mb_mon_present
            return True

        def isfile(path):
            return mpam_resctrl.os.path.basename(path) != required_entry_defect[0]

        return (
            self.mock.patch.object(resctrl_utils, "is_mounted",
                                   return_value=True),
            self.mock.patch.object(mpam_resctrl.os.path, "isdir",
                                   side_effect=isdir),
            self.mock.patch.object(mpam_resctrl.os.path, "exists",
                                   side_effect=exists),
            self.mock.patch.object(mpam_resctrl.os.path, "isfile",
                                   side_effect=isfile),
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_grace_does_not_require_mb_hlim_or_mb_mon(self, mock_nvdev):
        """Grace keeps validating its base schema when optional entries are absent."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        patches = self._patch_mounted_info(mb_mon_present=False, mb_hlim_present=False)
        mock_open = self.mock.mock_open()

        with patches[0], patches[1], patches[2], patches[3], \
                self.mock.patch("builtins.open", mock_open):
            mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        mock_open.assert_not_called()

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
    def test_grace_mb_hlim_non_directory_fails(self, mock_nvdev):
        """An optional MB_HLIM entry must still be a directory when present."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        patches = self._patch_mounted_info(mb_mon_present=False, mb_hlim_isdir=False)

        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(GLTCore.GLTTestFail,
                                        "MB_HLIM exists but is not a directory"):
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_grace_required_entry_defects_fail(self, mock_nvdev):
        """Optional MB_HLIM must not relax the required Grace schema."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        cases = (
            ("L3", "missing", "L3 (expected directory) is missing"),
            ("L3", "wrong type", "L3 exists but is not a directory"),
            ("L3_MON", "missing", "L3_MON (expected directory) is missing"),
            ("L3_MON", "wrong type", "L3_MON exists but is not a directory"),
            ("MB", "missing", "MB (expected directory) is missing"),
            ("MB", "wrong type", "MB exists but is not a directory"),
            ("last_cmd_status", "missing", "last_cmd_status (expected file) is missing"),
            ("last_cmd_status", "wrong type", "last_cmd_status exists but is not a file"),
        )
        for entry, defect, message in cases:
            with self.subTest(entry=entry, defect=defect):
                patches = self._patch_mounted_info(
                    mb_mon_present=False, required_entry_defect=(entry, defect))
                with patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaises(GLTCore.GLTTestFail) as context:
                        mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

                self.assertIn(message, str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_vera_ignores_mb_hlim_non_directory(self, mock_nvdev):
        """Vera MB_HLIM type belongs to the separate hard-limit exposure test."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = self._patch_mounted_info(mb_mon_present=True, mb_hlim_isdir=False)
        mock_open = self.mock.mock_open(read_data="mbm_total_bytes\n")

        with patches[0], patches[1], patches[2], patches[3], \
                self.mock.patch("builtins.open", mock_open):
            mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

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
            f"{resctrl_utils.RESCTRL_MOUNT_POINT}/info/MB_MON/mon_features",
            "r",
            encoding="utf-8",
        )

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_vera_missing_mbm_total_bytes_fails(self, mock_nvdev):
        """Present MB_MON does not replace Vera's required total-bytes feature."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = self._patch_mounted_info(mb_mon_present=True)
        mock_open = self.mock.mock_open(read_data="mbm_local_bytes\n")

        with patches[0], patches[1], patches[2], patches[3], \
                self.mock.patch("builtins.open", mock_open):
            with self.assertRaisesRegex(GLTCore.GLTTestFail,
                                        "missing expected features: mbm_total_bytes"):
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

    def test_info_unmounts_when_post_mount_check_fails(self):
        """A test-owned mount is cleaned up even when validation fails early."""
        with self.mock.patch.object(mpam_resctrl.glt_utils, "run_command") as mock_run, \
                self.mock.patch.object(mpam_resctrl.glt_utils, "is_cli_available",
                                       return_value=True), \
                self.mock.patch.object(mpam_resctrl, "NvDev"), \
                self.mock.patch.object(resctrl_utils, "is_mounted",
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


def _fixture_discovery_error_blocks_live_launch_and_keeps_cleanup_create_group(_resource, _logger=None, **_kwargs):
    return "/fake"


def _fixture_resctrl_monitor_read_tasks(path):
    """Read task IDs from one fixture tasks file."""
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _fixture_resctrl_monitor_write_tasks(path, task_ids):
    """Write task IDs to one fixture tasks file."""
    path.write_text(
        "".join(f"{task_id}\n" for task_id in sorted(task_ids)),
        encoding="utf-8")


def _fixture_resctrl_monitor_remove_task_from_all(fixture, root, task_id):
    """Remove a task from every control and monitor tasks file."""
    read_tasks, write_tasks = fixture
    for tasks_path in root.rglob("tasks"):
        task_ids = read_tasks(tasks_path)
        if task_id in task_ids:
            task_ids.remove(task_id)
            write_tasks(tasks_path, task_ids)


def _fixture_resctrl_monitor_write_sysfs_value(fixture, path, task_id):
    """Model resctrl task writes: CTRL writes clear MON membership."""
    events, read_tasks, write_tasks, remove_task_from_all = fixture
    task_text = str(task_id)
    target = Path(path)
    events.append(("write", str(target), task_text))
    if target.parent.parent.name == "mon_groups":
        parent_control = target.parent.parent.parent
        if task_text not in read_tasks(parent_control / "tasks"):
            raise OSError("monitor write requires parent control")
        for sibling in target.parent.parent.iterdir():
            sibling_tasks = sibling / "tasks"
            if sibling_tasks.exists():
                task_ids = read_tasks(sibling_tasks)
                task_ids.discard(task_text)
                write_tasks(sibling_tasks, task_ids)
        task_ids = read_tasks(target)
        task_ids.add(task_text)
        write_tasks(target, task_ids)
        return
    remove_task_from_all(Path(resctrl_utils.RESCTRL_MOUNT_POINT),
                         task_text)
    task_ids = read_tasks(target)
    task_ids.add(task_text)
    write_tasks(target, task_ids)


def _fixture_reopened_topology_blockage_keeps_output_after_group_absence_absent(fixture, _pgid, sig):
    events, = fixture
    if sig == 0:
        events.append("gone")
        raise ProcessLookupError


def _fixture_reopened_topology_blockage_keeps_output_after_group_absence_output(fixture, pgid, capture):
    events, original_output = fixture
    events.append("output")
    return original_output(pgid, capture)


def _fixture_early_exit_captures_after_straggler_cleanup_killpg(fixture, _pgid, sig):
    group_active, = fixture
    if sig == 0:
        if group_active["value"]:
            return
        raise ProcessLookupError
    if sig == mpam_resctrl.signal.SIGKILL:
        group_active["value"] = False


def _fixture_straggler_output(fixture, pgid, capture):
    read_states, group_active, original_read_output = fixture
    read_states.append(group_active["value"])
    return original_read_output(pgid, capture)


def _fixture_after_first_sample_exit_reports_status_and_output_sum_occupancy(fixture, _occupancy_files):
    occupancy_reads, launched = fixture
    occupancy_reads.append(len(occupancy_reads))
    if len(occupancy_reads) <= 1:
        return 0
    launched[0].exited = True
    return 5 * 1024 * 1024


def _fixture_first_restore_ctrl_write_failure_fails_immediately_write_task(fixture, path, _task_id):
    original_writes, = fixture
    if path == "/original/tasks":
        original_writes.append(path)
        if len(original_writes) == 1:
            raise OSError("root tasks busy")


def _fixture_first_restore_monitor_write_failure_fails_immediately_write_task(fixture, path, _task_id):
    monitor_group, monitor_writes = fixture
    if path == f"{monitor_group}/tasks":
        monitor_writes.append(path)
        if len(monitor_writes) == 1:
            raise OSError("monitor tasks busy")


def _fixture_first_restore_retry_success_remains_failure_write_task(fixture, path, _task_id):
    original_writes, = fixture
    if path == "/original/tasks":
        original_writes.append(path)
        if len(original_writes) == 1:
            raise OSError("first restore failed")


def _fixture_restore_monitor_write_error_fails_cleanup_write_task(fixture, path, _task_id):
    monitor_group, = fixture
    if path == f"{monitor_group}/tasks":
        raise FileNotFoundError("monitor group missing")


def _fixture_expired_exit_still_quiesces_then_blocks_stress_wait(fixture, _proc, _deadline, _argv, _capture):
    clock, self = fixture
    clock.now = 46
    return self._stress_result()


def _fixture_quiesce_overrun_blocks_without_completion_stress_wait(fixture, _proc, _deadline, _argv, _capture):
    clock, self = fixture
    clock.now = 44
    return self._stress_result()


def _fixture_late_nonzero_retains_definitive_failure_late_wait(fixture, process, deadline, argv, capture):
    self, clock = fixture
    self._complete_stress(process, deadline, argv, capture)
    clock.now = 46
    return self._stress_result(7, "late failure output\n" + self._stress_summary())


def _fixture_late_failed_summary_retains_definitive_failure_late_wait(fixture, process, deadline, argv, capture):
    self, clock = fixture
    self._complete_stress(process, deadline, argv, capture)
    clock.now = 46
    return self._stress_result(output=self._stress_summary(failed=1))


def _fixture_late_success_after_wait_is_blocked_late_wait(fixture, process, deadline, argv, capture):
    self, clock = fixture
    result = self._complete_stress(process, deadline, argv, capture)
    clock.now = 46
    return result


def _fixture_clean_exit_with_lingering_group_is_failed_signal_group(fixture, _pid, sig):
    group_alive, = fixture
    if sig == mpam_resctrl.signal.SIGKILL:
        group_alive["value"] = False
    elif not group_alive["value"]:
        raise ProcessLookupError


def _fixture_launch_blocked_keeps_restore_cleanup_failure_write_task(path, _task_id):
    if path == "/original/tasks":
        raise OSError("root tasks busy")


def _fixture_second_discovery(state, error, group, placement):
    """Expose an ordinary discovery fault only after the first completed window."""
    if state.groups:
        raise error
    return state.files(group, placement)


class _LaunchSequence:
    """Record supervisor placement, deadline, launch and prelaunch observations."""

    def __init__(self, test):
        self.test = test
        self.events = []
        self.supervisor_in_group = False
        self.workload_started = False

    def write_task(self, path, task_id):
        """Track assignment and restoration in the real workload context."""
        self.events.append(("write_task", path, task_id))
        self.supervisor_in_group = path == "/fake/tasks"

    def read_group_tasks(self, _group):
        """Expose the supervisor only after explicit assignment."""
        self.events.append(("read_tasks", self.supervisor_in_group))
        return {"777"} if self.supervisor_in_group else set()

    def sum_occupancy(self, _files):
        """Separate empty prelaunch occupancy from assigned and live occupancy."""
        self.events.append(("occupancy", self.supervisor_in_group))
        if self.workload_started:
            return 5 * 1024 * 1024
        return 1330752 if self.supervisor_in_group else 0

    def start(self, argv, **kwargs):
        """Require the supervisor assignment and immediate launch-time deadline."""
        self.test.assertEqual(argv, mpam_resctrl._mpam_l3_max_stress_argv(2))
        self.test.assertIn(("write_task", "/fake/tasks", 777), self.events)
        self.test.assertEqual(self.events[-1], ("deadline",))
        self.events.append(("start", 2))
        self.workload_started = True
        return _FreshProcess(kwargs["stdout"])

    def monotonic(self):
        """Record the last event before the real Popen boundary."""
        self.events.append(("deadline",))
        return 0


class _SampleProofs:
    """Record concrete membership proofs and reads across the sample stream."""

    def __init__(self, failure=None):
        self.events = []
        self.reads = 0
        self.failure = failure

    def proof(self, *_args, **_kwargs):
        """Fail at the next pre-read boundary when persistent drift is requested."""
        phase = "after" if self.reads == 10 else "before"
        index = 9 if phase == "after" else self.reads
        self.events.append(("proof", index, phase))
        if self.failure and self.reads == 1:
            raise mpam_resctrl._MpamBlocked(self.failure)

    def read(self, _files):
        """Append one real observation event after its membership proof."""
        self.events.append("read")
        self.reads += 1
        return 123

    def sleep(self, _seconds):
        """Record the interval that requires a fresh next-sample proof."""
        self.events.append("sleep")


class _FaultyCloseCapture(io.BytesIO):
    """A successful body can still report a real output-capture close failure."""

    def __init__(self, **_kwargs):
        super().__init__()

    def close(self):
        """Close the in-memory capture, then expose the cleanup error once."""
        if not self.closed:
            super().close()
            raise OSError("capture close denied")


class _FreshProcess:
    """One captured stress-ng process, without any host process or signals."""

    def __init__(self, capture):
        self.pid = 4321
        self.returncode = None
        capture.write(TestMpamMaxSchemataHelper._stress_summary().encode())

    def poll(self):
        """Keep all samples live until the explicit wait."""
        return self.returncode

    def wait(self, timeout=None):
        """Complete the fake process within the supplied deadline."""
        if timeout is None or timeout < 0:
            raise AssertionError("A bounded wait is required")
        self.returncode = 0
        return 0


class _FreshWorkload:
    """Kernel/process boundary state for the actual decorated six-window loop."""

    def __init__(self, test, placement):
        self.test = test
        self.placement = placement
        self.supervisor_group = "/original"
        self.percentages = []
        self.groups = []
        self.discoveries = []
        self.sample_reads = []

    def files(self, group, placement):
        """Bind each fresh group once before launch."""
        self.test.assertEqual(placement, self.placement)
        self.discoveries.append(group)
        return (str(Path(group) / "occ"),)

    @property
    def group(self):
        """Return the current once-discovered group."""
        return self.discoveries[-1]

    def write_task(self, path, _task_id):
        """Track the real context's supervisor moves."""
        self.supervisor_group = str(Path(path).parent)

    def read_tasks(self, group):
        """Expose the supervisor only while assigned by the real context."""
        return {"777"} if self.supervisor_group == group else set()

    def launch(self, argv, **kwargs):
        """Observe configured schemata at the actual Popen boundary."""
        self.test.assertEqual(argv, mpam_resctrl._mpam_l3_max_stress_argv(2))
        self.test.assertEqual(self.supervisor_group, self.group)
        values = resctrl_utils.read_schemata(Path(self.group) / "schemata", ("L3_MAX",))["L3_MAX"]
        self.test.assertEqual(len(set(values.values())), 1)
        self.percentages.append(next(iter(values.values())))
        self.groups.append(self.group)
        return _FreshProcess(kwargs["stdout"])

    def occupancy(self, files):
        """Return an empty baseline or one live observation on the bound files."""
        self.test.assertEqual(files, (str(Path(self.group) / "occ"),))
        self.test.assertNotEqual(self.supervisor_group, self.group)
        if len(self.groups) < len(self.discoveries):
            return 0
        self.sample_reads.append((self.group, files))
        return (2 if self.percentages[-1] < 50 else 6) * 1024 * 1024


def _window_group_path(join, group, *parts):
    """Bind isolated entry fixtures to their explicit fake kernel group."""
    if len(parts) == 2 and str(parts[1]).startswith("glt_l3_max_"):
        return group
    return join(*parts)


class _MpamMaxSchemataHelpers(GLTTestBase):
    """Shared fixtures for DGX-17928 MPAM percentage-schema unit tests."""

    def setUp(self):
        """Mock process-group signals for hardware-free MAX unit tests."""
        super().setUp()
        signal_patch = self.mock.patch.object(
            mpam_resctrl.os, "killpg", side_effect=ProcessLookupError)
        signal_patch.start()
        self.addCleanup(signal_patch.stop)
        for patch in (
                self.mock.patch.object(mpam_resctrl, "NvDev"),
                self.mock.patch.object(resctrl_utils, "managed_mount", side_effect=self._mounted)):
            handle = patch.start()
            self.addCleanup(patch.stop)
            if hasattr(handle, "return_value") and patch.attribute == "NvDev":
                self._patch_vera(handle)

    def _run_decorated_window(self, label, placement, group="/fake"):
        """Exercise one window through the decorated calibration/ownership body."""
        join = mpam_resctrl.os.path.join
        with ExitStack() as stack:
            patches = (
                self.mock.patch.object(mpam_resctrl.glt_utils, "is_cli_available", return_value=True),
                self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_require_stress_version"),
                self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_preflight_supervisor_children"),
                self.mock.patch.object(mpam_resctrl, "_mpam_read_percentage_info", return_value=(10, 10)),
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair", return_value=(10, 100)),
                self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_attribution", return_value=placement),
                self.mock.patch.object(mpam_resctrl, "MPAM_L3_MAX_LIVE_PATTERN", (label,)),
                self.mock.patch.object(mpam_resctrl.os, "mkdir"),
                self.mock.patch.object(resctrl_utils, "read_schemata", return_value={"L3_MAX": {0: 50}}),
                self.mock.patch.object(mpam_resctrl.os.path, "join",
                                       side_effect=partial(_window_group_path, join, group)),
            )
            for patch in patches:
                stack.enter_context(patch)
            evaluate = stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_evaluate_l3_max_live_windows", return_value=(0, 0, 0, 0)))
            mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            return evaluate.call_args.args[0][0]

    @staticmethod
    def _ok_schemata_result():
        """Return the success value used by write/readback helpers."""
        return "ok read back exact"

    @staticmethod
    @contextmanager
    def _mounted(_logger=None, *, cleanup_errors=None, cleanup_error=None):
        """Simulate a successful resctrl mount and optional cleanup error."""
        try:
            yield True, None
        finally:
            if cleanup_errors is not None and cleanup_error is not None:
                cleanup_errors.append(cleanup_error)

    @staticmethod
    @contextmanager
    def _mount_failed_with_cleanup(_logger=None, *, cleanup_errors=None):
        """Simulate mount failure and append a cleanup failure."""
        try:
            yield False, None
        finally:
            if cleanup_errors is not None:
                cleanup_errors.append("umount failed")

    def _patch_vera(self, mock_nvdev, *, vera=True):
        """Patch NvDev so only Vera detection follows the test flag."""
        mock_nvdev.return_value.is_cpu_type.side_effect = (
            lambda name: vera and name == "vera")

    def _seed_percentage_resource(
            self, root, resource="L3_MAX", *, info=None):
        """Create percentage resource metadata under a fake resctrl root."""
        if info is None:
            info = ("percentage", "1", "1")
        schema_format, percent_min, percent_gran = info
        info_dir = root / "info" / resource
        info_dir.mkdir(parents=True)
        (info_dir / "schema_format").write_text(
            f"{schema_format}\n", encoding="utf-8")
        (info_dir / "percent_min").write_text(
            f"{percent_min}\n", encoding="utf-8")
        (info_dir / "percent_gran").write_text(
            f"{percent_gran}\n", encoding="utf-8")

    def _run_temp_helper(self, *, resource="L3_MAX", info=None, initial_line=None):
        """Run the decorated schemata policy against real temporary files."""
        with self._schemata_environment(resource, info=info, initial_line=initial_line) as (root, _events):
            entry = (mpam_resctrl.test_mpam_l2_max_schemata if resource == "L2_MAX"
                     else mpam_resctrl.test_mpam_l3_max_schemata)
            entry(None, self.mock_logger)
            groups = list(root.glob("glt_*"))
            messages = [call.args[0] % call.args[1:] for call in self.mock_logger.info.call_args_list]
            return messages[-1], bool(groups)


    @contextmanager
    def _schemata_environment(self, resource="L3_MAX", *, info=None, initial_line=None):
        """Emulate kernel-created group files; retain real parser and restoration."""
        with TemporaryDirectory() as mount:
            root = Path(mount)
            self._seed_percentage_resource(root, resource, info=info)
            mkdir, rmdir = mpam_resctrl.os.mkdir, mpam_resctrl.os.rmdir
            events = []
            if initial_line is None:
                initial_line = f"{resource}:1=100;2=100"

            def create(path, *args, **kwargs):
                mkdir(path, *args, **kwargs)
                if Path(path).parent == root and Path(path).name.startswith("glt_"):
                    (Path(path) / "schemata").write_text(initial_line + "\n", encoding="ascii")
                    events.append(("mkdir", str(path)))

            def remove(path, *args, **kwargs):
                if Path(path).parent == root and Path(path).name.startswith("glt_"):
                    (Path(path) / "schemata").unlink(missing_ok=True)
                    events.append(("rmdir", str(path)))
                rmdir(path, *args, **kwargs)

            with self.mock.patch.object(resctrl_utils, "RESCTRL_MOUNT_POINT", mount), \
                    self.mock.patch.object(resctrl_utils, "managed_mount", side_effect=self._mounted), \
                    self.mock.patch.object(mpam_resctrl, "NvDev") as nvdev, \
                    self.mock.patch.object(mpam_resctrl.os, "mkdir", side_effect=create), \
                    self.mock.patch.object(mpam_resctrl.os, "rmdir", side_effect=remove):
                self._patch_vera(nvdev)
                yield root, events


    def test_existing_resource_missing_metadata_fails_without_group(self):
        """Missing metadata must never become optional-capability absence."""
        for resource in ("L2_MAX", "L3_MAX"):
            for filename in ("schema_format", "percent_min", "percent_gran"):
                with self.subTest(resource=resource, filename=filename):
                    with self._schemata_environment(resource) as (root, events):
                        (root / "info" / resource / filename).unlink()
                        entry = (mpam_resctrl.test_mpam_l2_max_schemata if resource == "L2_MAX"
                                 else mpam_resctrl.test_mpam_l3_max_schemata)
                        with self.assertRaisesRegex(GLTCore.GLTTestFail, filename):
                            entry(None, self.mock_logger)
                        self.assertEqual(events, [])

    def test_missing_schemata_resource_fails_after_owned_removal(self):
        """A readable group missing its requested MAX record is failure evidence."""
        with self._schemata_environment(initial_line="MB:1=100") as (_root, events):
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "L3_MAX resource line is missing"):
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual([event[0] for event in events], ["mkdir", "rmdir"])

    @contextmanager
    def _live_environment(self, *, initial_line="L3_MAX:1=50;2=50", info=("percentage", "10", "10")):
        """Run real orchestration, collector and ownership through fake Popen/I/O."""
        placement = {"node": 2, "cpus": (4, 5), "cache_ids": (9,)}
        state = _FreshWorkload(self, placement)
        with self._schemata_environment(info=info, initial_line=initial_line) as (root, events), \
                ExitStack() as stack:
            patches = (
                self.mock.patch.object(mpam_resctrl.glt_utils, "is_cli_available", return_value=True),
                self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_require_stress_version"),
                self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_preflight_supervisor_children"),
                self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_occupancy_files", side_effect=state.files),
                self.mock.patch.object(resctrl_utils, "sum_l3_occupancy", side_effect=state.occupancy),
                self.mock.patch.object(mpam_resctrl.threading, "get_native_id", return_value=777),
                self.mock.patch.object(mpam_resctrl, "_mpam_resctrl_group_for_task",
                                       return_value=("/original", None)),
                self.mock.patch.object(mpam_resctrl, "_mpam_read_group_task_ids", side_effect=state.read_tasks),
                self.mock.patch.object(mpam_resctrl.glt_utils, "write_sysfs_value", side_effect=state.write_task),
                self.mock.patch.object(mpam_resctrl, "_mpam_wait_for_l3_max_membership"),
                self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot"),
                self.mock.patch.object(mpam_resctrl.time, "monotonic", return_value=0),
                self.mock.patch.object(mpam_resctrl.time, "sleep"),
            )
            # Schemata writes remain real; only task assignment is simulated.
            original_write = stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_write_resource_schemata",
                side_effect=self._write_fixture_schemata))
            for patch in patches:
                stack.enter_context(patch)
            choose = stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_attribution", return_value=placement))
            start = stack.enter_context(self.mock.patch.object(
                mpam_resctrl.subprocess, "Popen", side_effect=state.launch))
            self.assertFalse(original_write.called)
            yield root, events, choose, start, state

    @staticmethod
    def _write_fixture_schemata(path, resource, domains, values):
        """Emulate schemata writes separately from the task-assignment boundary."""
        line = f"{resource}:" + ";".join(f"{domain}={values[domain]}" for domain in domains)
        Path(path).write_text(line, encoding="ascii")
        return line


class TestMpamMaxDecorated(_MpamMaxSchemataHelpers):
    """Validate decorated DGX-17928 MPAM L2_MAX/L3_MAX cases."""

    def setUp(self):
        super().setUp()
        version = self.mock.patch.object(mpam_resctrl.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, "stress-ng, version 0.16.00"))
        version.start()
        self.addCleanup(version.stop)

    def _create_private_group_entry(self, mount, name, tasks_text=""):
        """Create a real private-group fixture under a fake resctrl mount."""
        group = Path(mount) / name
        group.mkdir()
        (group / "tasks").write_text(tasks_text, encoding="utf-8")
        return group

    def test_group_task_id_reader_reads_real_tasks_file(self):
        """Task reader returns nonblank task IDs from a real tasks file."""
        with TemporaryDirectory() as mount:
            group = self._create_private_group_entry(
                mount, "glt_l3_max_1234_deadbeef", "42\n\n  84\n")

            tasks = mpam_resctrl._mpam_read_group_task_ids(str(group))

        self.assertEqual(tasks, {"42", "84"})

    def test_group_task_id_reader_raises_oserror_for_missing_tasks_file(self):
        """Missing tasks file raises unreadable state evidence."""
        with TemporaryDirectory() as mount:
            group = Path(mount) / "glt_l3_max_1234_deadbeef"
            group.mkdir()

            with self.assertRaises(OSError):
                mpam_resctrl._mpam_read_group_task_ids(str(group))


    def test_private_group_name_collision_fails_without_removing_group(self):
        """A failed mkdir never authorizes removal of another invocation's group."""
        with self._schemata_environment() as (_root, events):
            with self.mock.patch.object(mpam_resctrl.os, "mkdir", side_effect=FileExistsError), \
                    self.mock.patch.object(mpam_resctrl.os, "rmdir") as remove:
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "refusing to remove"):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            remove.assert_not_called()
            self.assertEqual(events, [])


    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_decorated_success_calls_matching_resource_helper(self, mock_nvdev):
        """Each decorated MAX case probes and restores only its own resource."""
        self._patch_vera(mock_nvdev)
        for resource, entry in (("L2_MAX", mpam_resctrl.test_mpam_l2_max_schemata),
                                ("L3_MAX", mpam_resctrl.test_mpam_l3_max_schemata)):
            with self.subTest(resource=resource):
                with self._schemata_environment(resource) as (root, events):
                    entry(None, self.mock_logger)
                    self.assertEqual([event[0] for event in events], ["mkdir", "rmdir"])
                    self.assertIn(f"glt_{resource.lower()}_", events[0][1])
                    self.assertEqual(list(root.glob("glt_*")), [])

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_decorated_non_vera_skips_before_mount(self, mock_nvdev):
        """The Vera-only case does not mount resctrl on other CPUs."""
        self._patch_vera(mock_nvdev, vera=False)

        with self.mock.patch.object(resctrl_utils, 'managed_mount') as mount:
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("Vera-specific", str(context.exception))
        mount.assert_not_called()

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l2_max_missing_resource_is_contract_not_exposed_skip(self, mock_nvdev):
        """Only absent optional L2 skips; required L3 fails after mount."""
        self._patch_vera(mock_nvdev)
        with self._schemata_environment("L2_MAX") as (root, events):
            shutil.rmtree(root / "info" / "L2_MAX")
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                mpam_resctrl.test_mpam_l2_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])
        self.assertIn("info/L2_MAX is absent", str(context.exception))
        self.assertIn("no L2_MAX assertion was run", str(context.exception))
        self.assertIn("capability-conditional", str(context.exception))
        self.assertNotIn("feature-acceptance", str(context.exception))
        self.assertNotIn("O04", str(context.exception))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_missing_resource_fails_after_prerequisites(self, mock_nvdev):
        """Only absent optional L2 skips; required L3 fails after mount."""
        self._patch_vera(mock_nvdev)
        with self._schemata_environment("L3_MAX") as (root, events):
            shutil.rmtree(root / "info" / "L3_MAX")
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])
        self.assertIn("info/L3_MAX is absent", str(context.exception))
        self.assertIn("no L3_MAX assertion was run", str(context.exception))


    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_contract_blocked_maps_to_decorated_warn(self, mock_nvdev):
        """Real presence plus a blocked probe retains the technical WARN reason."""
        self._patch_vera(mock_nvdev)
        for resource, entry in (("L2_MAX", mpam_resctrl.test_mpam_l2_max_schemata),
                                ("L3_MAX", mpam_resctrl.test_mpam_l3_max_schemata)):
            with self.subTest(resource=resource), self._schemata_environment(resource) as (_root, events):
                with self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair",
                                            return_value=None):
                    with self.assertRaisesRegex(GLTCore.GLTTestWarn, "no two distinct legal values"):
                        entry(None, self.mock_logger)
                self.assertEqual([event[0] for event in events], ["mkdir", "rmdir"])

    def _run_decorated_real_helper_mismatch(self, case):
        """Drive a decorated L*_MAX case through the real helper cleanup path."""
        events = []
        resource = case["resource"]
        readbacks = list(case["readbacks"])
        group = f"/fake-{resource.lower()}-group"
        cleanup_error = case.get(
            "cleanup_error", f"{resource} unmount cleanup failed")

        @contextmanager
        def mounted_with_cleanup(_logger, *, cleanup_errors=None):
            try:
                yield True, None
            finally:
                events.append(("umount", resource))
                if cleanup_errors is not None and cleanup_error is not None:
                    cleanup_errors.append(cleanup_error)

        def create_group(path):
            nonlocal group
            group = path

        def read(_path, read_resource):
            self.assertEqual(read_resource, (resource,))
            _read_domains, read_values, readback = readbacks.pop(0)
            events.append(("read", readback))
            return {resource: {int(domain): value for domain, value in read_values.items()}}

        def write(_path, write_resource, domains, value_by_domain):
            self.assertEqual(write_resource, resource)
            writeback = f"{resource}:" + ";".join(
                f"{domain}={value_by_domain[domain]}" for domain in domains)
            events.append(("write", writeback))
            return writeback

        def remove_group(path):
            events.append(("rmdir", path))

        original_fail = mpam_resctrl.GLTCore.GLTTestResult.fail

        def fail(message):
            events.append(("fail", message))
            original_fail(message)

        with self.mock.patch.object(mpam_resctrl, "NvDev") as mock_nvdev, \
                self.mock.patch.object(
                    resctrl_utils, 'managed_mount',
                    side_effect=mounted_with_cleanup), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_percentage_info",
                    return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    side_effect=create_group), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=read), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    side_effect=write), \
                self.mock.patch.object(
                    mpam_resctrl.os, "rmdir", side_effect=remove_group), \
                self.mock.patch.object(
                    mpam_resctrl.GLTCore.GLTTestResult, "fail",
                    side_effect=fail):
            self._patch_vera(mock_nvdev)
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                case["test_case"](None, self.mock_logger)

        self.assertEqual(readbacks, [])
        self.assertIn(case["expected_write"], str(context.exception))
        self.assertIn(case["expected_readback"], str(context.exception))
        self.assertIn("exact per-domain readback is required",
                      str(context.exception))
        if cleanup_error is None:
            self.assertNotIn("cleanup:", str(context.exception))
        else:
            self.assertIn(cleanup_error, str(context.exception))

        event_kinds = [event[0] for event in events]
        rmdir_index = events.index(("rmdir", group))
        umount_index = events.index(("umount", resource))
        fail_index = event_kinds.index("fail")
        self.assertEqual(event_kinds[rmdir_index - 2:rmdir_index],
                         ["write", "read"])
        self.assertLess(rmdir_index, umount_index)
        self.assertLess(umount_index, fail_index)

    def test_decorated_real_helper_mismatches_fail_after_cleanup(self):
        """Exact readback mismatches fail only after restore, rmdir, and unmount."""
        cases = (
            {
                "name": "l2-upper-wrong-value",
                "resource": "L2_MAX",
                "test_case": mpam_resctrl.test_mpam_l2_max_schemata,
                "readbacks": (
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                    (("1",), {"1": 99}, "L2_MAX:1=99"),
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                ),
                "expected_write": "wrote 'L2_MAX:1=100'",
                "expected_readback": "read back 'L2_MAX:1=99'",
            },
            {
                "name": "l3-low-wrong-value",
                "resource": "L3_MAX",
                "test_case": mpam_resctrl.test_mpam_l3_max_schemata,
                "readbacks": (
                    (("1",), {"1": 50}, "L3_MAX:1=50"),
                    (("1",), {"1": 100}, "L3_MAX:1=100"),
                    (("1",), {"1": 50}, "L3_MAX:1=50"),
                    (("1",), {"1": 2}, "L3_MAX:1=2"),
                    (("1",), {"1": 50}, "L3_MAX:1=50"),
                ),
                "expected_write": "wrote 'L3_MAX:1=1'",
                "expected_readback": "read back 'L3_MAX:1=2'",
            },
            {
                "name": "l2-high-wrong-value",
                "resource": "L2_MAX",
                "test_case": mpam_resctrl.test_mpam_l2_max_schemata,
                "readbacks": (
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                    (("1",), {"1": 100}, "L2_MAX:1=100"),
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                    (("1",), {"1": 1}, "L2_MAX:1=1"),
                    (("1",), {"1": 99}, "L2_MAX:1=99"),
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                ),
                "expected_write": "wrote 'L2_MAX:1=100'",
                "expected_readback": "read back 'L2_MAX:1=99'",
            },
            {
                "name": "l3-high-extra-domain-clean-unmount",
                "resource": "L3_MAX",
                "test_case": mpam_resctrl.test_mpam_l3_max_schemata,
                "readbacks": (
                    (("1",), {"1": 50}, "L3_MAX:1=50"),
                    (("1",), {"1": 100}, "L3_MAX:1=100"),
                    (("1",), {"1": 50}, "L3_MAX:1=50"),
                    (("1",), {"1": 1}, "L3_MAX:1=1"),
                    (("1", "2"), {"1": 100, "2": 100},
                     "L3_MAX:1=100;2=100"),
                    (("1",), {"1": 50}, "L3_MAX:1=50"),
                ),
                "expected_write": "wrote 'L3_MAX:1=100'",
                "expected_readback": "read back 'L3_MAX:1=100;2=100'",
                "cleanup_error": None,
            },
            {
                "name": "l3-upper-missing-domain",
                "resource": "L3_MAX",
                "test_case": mpam_resctrl.test_mpam_l3_max_schemata,
                "readbacks": (
                    (("1", "2"), {"1": 50, "2": 50},
                     "L3_MAX:1=50;2=50"),
                    (("1",), {"1": 100}, "L3_MAX:1=100"),
                    (("1", "2"), {"1": 50, "2": 50},
                     "L3_MAX:1=50;2=50"),
                ),
                "expected_write": "wrote 'L3_MAX:1=100;2=100'",
                "expected_readback": "read back 'L3_MAX:1=100'",
            },
            {
                "name": "l2-low-extra-domain",
                "resource": "L2_MAX",
                "test_case": mpam_resctrl.test_mpam_l2_max_schemata,
                "readbacks": (
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                    (("1",), {"1": 100}, "L2_MAX:1=100"),
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                    (("1", "2"), {"1": 1, "2": 1},
                     "L2_MAX:1=1;2=1"),
                    (("1",), {"1": 50}, "L2_MAX:1=50"),
                ),
                "expected_write": "wrote 'L2_MAX:1=1'",
                "expected_readback": "read back 'L2_MAX:1=1;2=1'",
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                self._run_decorated_real_helper_mismatch(case)

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_mount_cleanup_failure_wins_over_contract_warning(self, mock_nvdev):
        """Unmount cleanup failure is a decorated fail, not a later warning."""
        with self._schemata_environment():
            self._patch_vera(mock_nvdev)
            blocked = None

            mounted_with_cleanup = partial(self._mounted, cleanup_error="umount failed")

            with self.mock.patch.object(
                    resctrl_utils, 'managed_mount',
                    side_effect=mounted_with_cleanup), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair",
                        return_value=blocked):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

            self.assertIn("umount failed", str(context.exception))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_helper_failure_combines_with_mount_cleanup_failure(self, mock_nvdev):
        """A helper failure and unmount failure are both kept in the fail text."""
        with self._schemata_environment():
            self._patch_vera(mock_nvdev)

            def mounted_with_cleanup(_logger, *, cleanup_errors=None):
                return self._mounted(
                    cleanup_errors=cleanup_errors, cleanup_error="umount failed")

            with self.mock.patch.object(
                    resctrl_utils, 'managed_mount',
                    side_effect=mounted_with_cleanup), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair",
                        side_effect=ValueError(
                            "helper failed")):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

            self.assertIn("helper failed", str(context.exception))
            self.assertIn("umount failed", str(context.exception))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_mount_failure_combines_with_mount_cleanup_failure(self, mock_nvdev):
        """A failed mount waits for context cleanup before decorated fail."""
        with self._schemata_environment():
            self._patch_vera(mock_nvdev)

            with self.mock.patch.object(
                    resctrl_utils, 'managed_mount',
                    side_effect=self._mount_failed_with_cleanup), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair"
                    ) as helper:
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

            self.assertIn("Failed to mount resctrl.", str(context.exception))
            self.assertIn("umount failed", str(context.exception))
            helper.assert_not_called()

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l2_mount_failure_combines_with_mount_cleanup_failure(self, mock_nvdev):
        """L2 uses the same post-context mount-failure reporting path."""
        with self._schemata_environment():
            self._patch_vera(mock_nvdev)

            with self.mock.patch.object(
                    resctrl_utils, 'managed_mount',
                    side_effect=self._mount_failed_with_cleanup), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair"
                    ) as helper:
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l2_max_schemata(None, self.mock_logger)

            self.assertIn("Failed to mount resctrl.", str(context.exception))
            self.assertIn("umount failed", str(context.exception))
            helper.assert_not_called()

    def test_reopened_version_preflight_rejects_before_mutation(self):
        """Unsupported output cannot reach resctrl mutation or workload launch."""
        cases = (
            (subprocess.CompletedProcess([], 0, "stress-ng, version 0.15.06"), "0.15.06"),
            (subprocess.CompletedProcess([], 0, "unknown build"), "unknown build"),
            (subprocess.CompletedProcess([], 1, "version failed"), "version failed"),
            (subprocess.TimeoutExpired(["stress-ng", "--version"], 5), "5"),
            (OSError("version unavailable"), "version unavailable"),
        )
        for response, reason in cases:
            with self.subTest(reason=reason), ExitStack() as stack:
                nvdev = stack.enter_context(self.mock.patch.object(mpam_resctrl, "NvDev"))
                self._patch_vera(nvdev)
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available", return_value=True))
                version_patch = (self.mock.patch.object(
                    mpam_resctrl.subprocess, "run", side_effect=response)
                    if isinstance(response, Exception) else self.mock.patch.object(
                        mpam_resctrl.subprocess, "run", return_value=response))
                version = stack.enter_context(version_patch)
                mount = stack.enter_context(self.mock.patch.object(
                    resctrl_utils, "managed_mount", side_effect=self._mounted))
                helper = stack.enter_context(self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair"))
                with self.assertRaises(GLTCore.GLTTestSkip) as caught:
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
                self.assertIn(reason, str(caught.exception))
                self.assertIn("0.16.00", str(caught.exception))
                self.assertEqual(version.call_args.args[0], ["stress-ng", "--version"])
                self.assertEqual(version.call_args.kwargs["timeout"], 5)
                mount.assert_not_called()
                helper.assert_not_called()

    def test_reopened_scheduler_reaches_live_checks_after_schemata_failure(self):
        """The live probe can explain its own prerequisites on schema WARN/FAIL."""
        requires = mpam_resctrl.test_mpam_l3_max_live_effect.attr.func.requires
        for status in (GLTCore.GLTTestResult.WARN, GLTCore.GLTTestResult.FAIL):
            with self.subTest(status=status):
                scheduler = self.mock.Mock()
                def test_object(name, status=status):
                    obj = self.mock.Mock()
                    obj.__wrapped__ = self.mock.Mock()
                    obj.__wrapped__.attr.result = (
                        GLTCore.GLTTestResult.PASS if name.endswith("resctrl_mount")
                        else status, "prerequisite result")
                    return obj
                scheduler.get_test_obj_by_name.side_effect = test_object
                self.assertTrue(requires.can_start(scheduler)[0])

    def test_reopened_missing_l3_is_consistent_failure(self):
        """Mandatory L3 absence fails both decorated interfaces without mutation."""
        for entry in (mpam_resctrl.test_mpam_l3_max_schemata,
                      mpam_resctrl.test_mpam_l3_max_live_effect):
            with self.subTest(entry=entry.__name__), self._live_environment() as (root, events, _c, window, _state):
                shutil.rmtree(root / "info" / "L3_MAX")
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "info/L3_MAX is absent"):
                    entry(None, self.mock_logger)
                self.assertEqual(events, [])
                window.assert_not_called()

    def test_l3_max_live_effect_requires_resctrl_mount(self):
        """Live-effect reaches independent checks after the mount prerequisite."""
        self.assertIn(
            "arm64:test_mpam_resctrl_mount",
            mpam_resctrl.test_mpam_l3_max_live_effect.attr.func.requires.all_true,
        )

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_success_calls_helper(self, mock_nvdev):
        """Decorated live validation completes all six owned measurement windows."""
        self._patch_vera(mock_nvdev)
        with self._live_environment() as (root, events, choose, window, _state):
            mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            choose.assert_called_once()
            self.assertEqual(window.call_count, 6)
            self.assertEqual(len(events), 14)
            self.assertEqual(list(root.glob("glt_*")), [])

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_blocked_retained_when_unmount_fails(self, mock_nvdev):
        """Cleanup FAIL must retain the original typed observation reason first."""
        with self._live_environment():
            self._patch_vera(mock_nvdev)

            def mounted_with_cleanup(_logger, *, cleanup_errors=None):
                return self._mounted(
                    cleanup_errors=cleanup_errors, cleanup_error="umount failed")

            with self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available", return_value=True), \
                    self.mock.patch.object(
                        resctrl_utils, "managed_mount", side_effect=mounted_with_cleanup), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair",
                        side_effect=mpam_resctrl._MpamBlocked("primary blocked sentinel")):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)

            self.assertIn("primary blocked sentinel", str(context.exception))
            self.assertIn("umount failed", str(context.exception))
            self.assertLess(str(context.exception).index("primary blocked sentinel"),
                            str(context.exception).index("umount failed"))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_not_exposed_retained_when_unmount_fails(self, mock_nvdev):
        """Cleanup FAIL must retain the original typed observation reason first."""
        with self._live_environment():
            self._patch_vera(mock_nvdev)

            def mounted_with_cleanup(_logger, *, cleanup_errors=None):
                return self._mounted(
                    cleanup_errors=cleanup_errors, cleanup_error="umount failed")

            with self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available", return_value=True), \
                    self.mock.patch.object(
                        resctrl_utils, "managed_mount", side_effect=mounted_with_cleanup), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_read_percentage_info",
                        return_value=None):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)

            self.assertIn("info/L3_MAX is absent", str(context.exception))
            self.assertIn("umount failed", str(context.exception))
            self.assertLess(str(context.exception).index("info/L3_MAX is absent"),
                            str(context.exception).index("umount failed"))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_missing_l3_max_fails(self, mock_nvdev):
        """Missing mandatory L3_MAX is a failure after mount prerequisites."""
        with self._live_environment():
            self._patch_vera(mock_nvdev)

            with self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available",
                    return_value=True), \
                    self.mock.patch.object(
                        resctrl_utils, 'managed_mount',
                        side_effect=self._mounted), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_read_percentage_info",
                        return_value=None):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l3_max_live_effect(
                        None, self.mock_logger)

            self.assertIn("info/L3_MAX is absent", str(context.exception))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_blocked_maps_to_warn(self, mock_nvdev):
        """Invalid live observation maps to WARN."""
        with self._live_environment():
            self._patch_vera(mock_nvdev)

            with self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available",
                    return_value=True), \
                    self.mock.patch.object(
                        resctrl_utils, 'managed_mount',
                        side_effect=self._mounted), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair",
                        side_effect=mpam_resctrl._MpamBlocked(
                            "membership proof missing")):
                with self.assertRaises(GLTCore.GLTTestWarn) as context:
                    mpam_resctrl.test_mpam_l3_max_live_effect(
                        None, self.mock_logger)

            self.assertIn("membership proof missing", str(context.exception))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_warns_when_children_interface_missing(
            self, mock_nvdev):
        """Live-effect preflights /proc children before side effects."""
        self._patch_vera(mock_nvdev)
        children_path = "/proc/self/task/777/children"

        def open_children(path, *_args, **_kwargs):
            if path == children_path:
                raise FileNotFoundError(
                    errno.ENOENT, "No such file or directory", path)
            return io.StringIO("")

        with self.mock.patch.object(
                mpam_resctrl.glt_utils, "is_cli_available",
                return_value=True), \
                self.mock.patch.object(
                    mpam_resctrl.threading, "get_native_id",
                    return_value=777), \
                self.mock.patch("builtins.open", side_effect=open_children), \
                self.mock.patch.object(mpam_resctrl.os, "mkdir") as mkdir, \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen") as popen, \
                self.mock.patch.object(
                    resctrl_utils, 'managed_mount',
                    side_effect=self._mounted) as mount, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair") as helper:
            with self.assertRaises(GLTCore.GLTTestWarn) as context:
                mpam_resctrl.test_mpam_l3_max_live_effect(
                    None, self.mock_logger)

        self.assertIn(children_path, str(context.exception))
        self.assertIn(str(errno.ENOENT), str(context.exception))
        mkdir.assert_not_called()
        popen.assert_not_called()
        mount.assert_not_called()
        helper.assert_not_called()

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_warns_when_children_interface_denied(
            self, mock_nvdev):
        """Live-effect reports permission errno from /proc children preflight."""
        self._patch_vera(mock_nvdev)
        children_path = "/proc/self/task/777/children"

        def open_children(path, *_args, **_kwargs):
            if path == children_path:
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return io.StringIO("")

        with self.mock.patch.object(
                mpam_resctrl.glt_utils, "is_cli_available",
                return_value=True), \
                self.mock.patch.object(
                    mpam_resctrl.threading, "get_native_id",
                    return_value=777), \
                self.mock.patch("builtins.open", side_effect=open_children), \
                self.mock.patch.object(mpam_resctrl.os, "mkdir") as mkdir, \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen") as popen, \
                self.mock.patch.object(
                    resctrl_utils, 'managed_mount',
                    side_effect=self._mounted) as mount, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair") as helper:
            with self.assertRaises(GLTCore.GLTTestWarn) as context:
                mpam_resctrl.test_mpam_l3_max_live_effect(
                    None, self.mock_logger)

        self.assertIn(children_path, str(context.exception))
        self.assertIn(str(errno.EACCES), str(context.exception))
        mkdir.assert_not_called()
        popen.assert_not_called()
        mount.assert_not_called()
        helper.assert_not_called()

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_contract_blocked_warn_is_customer_clean(
            self, mock_nvdev):
        """Live-effect contract-blocked output keeps technical reason only."""
        with self._live_environment():
            self._patch_vera(mock_nvdev)
            blocked = None

            with self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available",
                    return_value=True), \
                    self.mock.patch.object(
                        resctrl_utils, 'managed_mount',
                        side_effect=self._mounted), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair",
                        return_value=blocked):
                with self.assertRaises(GLTCore.GLTTestWarn) as context:
                    mpam_resctrl.test_mpam_l3_max_live_effect(
                        None, self.mock_logger)

            self.assertIn("L3_MAX", str(context.exception))
            self.assertIn("no two distinct legal values", str(context.exception))
            self.assertNotIn("CONTRACT_ADJUDICATION", str(context.exception))
            self.assertNotIn("Owner review", str(context.exception))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_failure_maps_to_fail(self, mock_nvdev):
        """Complete-valid live comparison failure maps to FAIL."""
        with self._live_environment():
            self._patch_vera(mock_nvdev)

            with self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available",
                    return_value=True), \
                    self.mock.patch.object(
                        resctrl_utils, 'managed_mount',
                        side_effect=self._mounted), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair",
                        side_effect=ValueError(
                            "high median too low")):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    mpam_resctrl.test_mpam_l3_max_live_effect(
                        None, self.mock_logger)

            self.assertIn("high median too low", str(context.exception))

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_unexpected_helper_error_maps_to_fail(
            self, mock_nvdev):
        """Unexpected live helper exceptions are decorated fail results."""
        with self._live_environment():
            self._patch_vera(mock_nvdev)

            cleanup_lists = []

            def mounted_with_cleanup(_logger, *, cleanup_errors=None):
                cleanup_lists.append(cleanup_errors)
                return self._mounted(
                    cleanup_errors=cleanup_errors, cleanup_error="umount failed")

            with self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available",
                    return_value=True), \
                    self.mock.patch.object(
                        resctrl_utils, 'managed_mount',
                        side_effect=mounted_with_cleanup), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_probe_percentage_pair",
                        side_effect=ValueError("bad helper")):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "bad helper; cleanup: umount failed"):
                    mpam_resctrl.test_mpam_l3_max_live_effect(
                        None, self.mock_logger)

            self.assertEqual(len(cleanup_lists), 1)
            self.assertIn("umount failed", cleanup_lists[0])

    @mock.patch("src.modules.arm64.glt_test_mpam_resctrl.NvDev")
    def test_l3_max_live_effect_missing_tool_skips_before_mount(self, mock_nvdev):
        """Missing numactl/stress-ng is a prerequisite skip."""
        self._patch_vera(mock_nvdev)

        with self.mock.patch.object(
                mpam_resctrl.glt_utils, "is_cli_available",
                side_effect=lambda cli: cli != "stress-ng"), \
                self.mock.patch.object(resctrl_utils, 'managed_mount'
                                       ) as mount:
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                mpam_resctrl.test_mpam_l3_max_live_effect(
                    None, self.mock_logger)

        self.assertIn("'stress-ng' CLI not available",
                      str(context.exception))
        mount.assert_not_called()


    def test_missing_mandatory_l3_info_blocks_existing_mb_dependency_chains(self):
        """Missing Vera L3_MAX fails info; all eight MB guards retain that dependency."""
        with self.mock.patch.object(mpam_resctrl, "NvDev") as nvdev, \
                self.mock.patch.object(resctrl_utils, "managed_mount", side_effect=self._mounted), \
                self.mock.patch.object(mpam_resctrl.os.path, "isdir", return_value=True), \
                self.mock.patch.object(mpam_resctrl.os.path, "isfile", return_value=True), \
                self.mock.patch.object(mpam_resctrl.os.path, "exists",
                                       side_effect=lambda path: not path.endswith("/L3_MAX")):
            self._patch_vera(nvdev)
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "L3_MAX .* is missing"):
                mpam_resctrl.test_mpam_resctrl_info(None, self.mock_logger)

        dependencies = (
            ("test_mpam_resctrl_mb_max_lim", "test_mpam_resctrl_info"),
            ("test_mpam_mba_hardlim_exposure", "test_mpam_resctrl_mb_max_lim"),
            ("test_mpam_mba_hardlim_valid_values", "test_mpam_mba_hardlim_exposure"),
            ("test_mpam_mba_hardlim_validation_defaults", "test_mpam_mba_hardlim_exposure"),
            ("test_mpam_mbm_assignment_mode", "test_mpam_resctrl_info"),
            ("test_mpam_mbm_counter_info", "test_mpam_mbm_assignment_mode"),
            ("test_mpam_mbm_assignments", "test_mpam_mbm_counter_info"),
            ("test_mpam_mbm_counter_growth", "test_mpam_mbm_assignments"),
        )
        for name, prerequisite in dependencies:
            requires = getattr(mpam_resctrl, name).attr.func.requires
            self.assertEqual(requires.all_true, [f"arm64:{prerequisite}"])
            for status in (GLTCore.GLTTestResult.FAIL, GLTCore.GLTTestResult.SKIP):
                with self.subTest(test=name, prerequisite_status=status):
                    upstream = self.mock.Mock()
                    upstream.__wrapped__ = self.mock.Mock()
                    upstream.__wrapped__.attr.name = prerequisite
                    upstream.__wrapped__.attr.result = (status, "prerequisite unavailable")
                    scheduler = self.mock.Mock()
                    scheduler.get_test_obj_by_name.return_value = upstream
                    self.assertEqual(requires.can_start(scheduler),
                                     (False, f"{prerequisite} in all_true returned False"))
                    scheduler.get_test_obj_by_name.assert_called_once_with(f"arm64:{prerequisite}")


class TestMpamCpuHelpers(GLTTestBase):
    """CPU-list helpers preserve legacy callers unless strict mode opts in."""

    def test_cpus_of_node_default_preserves_legacy_parser_contract(self):
        """Default callers keep order, duplicates, and permissive segments."""
        cases = (
            ("0-3,8\n", [0, 1, 2, 3, 8]),
            ("0,,2\n", [0, 2]),
            ("3-1,5\n", [5]),
            ("5,1,1\n", [5, 1, 1]),
        )
        for cpulist, expected in cases:
            with self.subTest(cpulist=cpulist):
                with self.mock.patch(
                        "builtins.open",
                        self.mock.mock_open(read_data=cpulist)) as open_mock:
                    cpus = mpam_resctrl._cpus_of_node(2)

                self.assertEqual(cpus, expected)
                open_mock.assert_called_once_with(
                    "/sys/devices/system/node/node2/cpulist",
                    "r", encoding="utf-8")

    def test_cpus_of_node_online_only_returns_sorted_online_intersection(self):
        """Strict live attribution ignores offline CPUs and legacy ordering."""
        opened = []
        contents = {
            "/sys/devices/system/node/node2/cpulist": "5,1,3\n",
            "/sys/devices/system/cpu/online": "0-3,8\n",
        }

        def fake_open(path, *_args, **_kwargs):
            opened.append(path)
            return StringIO(contents[path])

        with self.mock.patch("builtins.open", side_effect=fake_open):
            cpus = mpam_resctrl._cpus_of_node(2, online_only=True)

        self.assertEqual(cpus, [1, 3])
        self.assertEqual(opened, [
            "/sys/devices/system/node/node2/cpulist",
            "/sys/devices/system/cpu/online",
        ])

    def test_cpus_of_node_online_only_rejects_empty_online_mask(self):
        """Live attribution needs affirmative online CPU evidence."""
        contents = {
            "/sys/devices/system/node/node2/cpulist": "0-3\n",
            "/sys/devices/system/cpu/online": "\n",
        }

        def fake_open(path, *_args, **_kwargs):
            return StringIO(contents[path])

        with self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(ValueError) as context:
                mpam_resctrl._cpus_of_node(2, online_only=True)

        self.assertIn("/sys/devices/system/cpu/online", str(context.exception))
        self.assertIn("empty", str(context.exception))

    def test_cpus_of_node_online_only_rejects_permissive_legacy_mask(self):
        """Strict live attribution rejects masks the legacy default skips."""
        def fake_open(path, *_args, **_kwargs):
            if path == "/sys/devices/system/node/node2/cpulist":
                return StringIO("0,,2\n")
            raise AssertionError(f"Unexpected open after malformed node mask: {path}")

        with self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(ValueError) as context:
                mpam_resctrl._cpus_of_node(2, online_only=True)

        self.assertIn("/sys/devices/system/node/node2/cpulist",
                      str(context.exception))
        self.assertIn("'0,,2'", str(context.exception))

    def test_cpus_of_node_online_only_reports_node_read_path(self):
        """Strict live attribution reports the node cpulist path on read I/O."""
        node_path = "/sys/devices/system/node/node2/cpulist"
        injected = OSError(errno.EIO, "injected node read error")

        def fake_open(path, *_args, **_kwargs):
            if path == node_path:
                raise injected
            raise AssertionError(f"Unexpected open after node read error: {path}")

        with self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(OSError) as context:
                mpam_resctrl._cpus_of_node(2, online_only=True)

        self.assertIn(node_path, str(context.exception))
        self.assertIn("injected node read error", str(context.exception))


class TestMpamL3MaxDiscovery(GLTTestBase):
    """Unreadable topology must retain the reason, not look absent."""

    _run_decorated_window = _MpamMaxSchemataHelpers._run_decorated_window

    def test_discovery_cache_preserves_prefix_and_level_selection(self):
        """Valid cache indexes still contribute IDs while metadata is ignored."""
        cache_path = "/sys/devices/system/cpu/cpu4/cache"
        real_scandir = mpam_resctrl.os.scandir
        real_open = open
        with TemporaryDirectory() as root:
            for name, level, cache_id in (
                    ("index0", "2", None),
                    ("index1", "3", "9"),
                    ("index2suffix", "3", "10")):
                entry = Path(root, name)
                entry.mkdir()
                (entry / "level").write_text(level, encoding="utf-8")
                if cache_id is not None:
                    (entry / "id").write_text(cache_id, encoding="utf-8")
            Path(root, "metadata").write_text("", encoding="utf-8")

            def cache_open(path, *args, **kwargs):
                return real_open(
                    Path(root) / Path(path).relative_to(cache_path),
                    *args, **kwargs)

            with self.mock.patch.object(
                    mpam_resctrl.os, "scandir",
                    side_effect=lambda _path, root=root: real_scandir(root)), \
                    self.mock.patch("builtins.open", side_effect=cache_open):
                cache_ids = mpam_resctrl._mpam_l3_cache_ids_for_cpu(4)

        self.assertEqual(cache_ids, {9, 10})

    def test_discovery_child_inspection_errors_preserve_path_and_cause(self):
        """Only genuine absence may skip cpulist or occupancy children."""
        cases = (
            (mpam_resctrl._mpam_l3_max_node_cpulists, "node2",
             "/sys/devices/system/node/node2/cpulist"),
            (partial(mpam_resctrl._mpam_l3_max_attribution, "/fake"), "mon_L3_9",
             "/fake/mon_data/mon_L3_9/llc_occupancy"),
        )
        real_scandir = mpam_resctrl.os.scandir
        for discover, name, path in cases:
            for error in (PermissionError("child denied"),
                          OSError("child I/O failure")):
                with self.subTest(path=path, error=type(error).__name__):
                    with TemporaryDirectory() as root:
                        Path(root, name).mkdir()
                        with self.mock.patch.object(
                                mpam_resctrl.os, "scandir",
                                side_effect=lambda _path, root=root: (
                                    real_scandir(root))), \
                                self.mock.patch.object(
                                    mpam_resctrl.os, "lstat",
                                    side_effect=error):
                            with self.assertRaises(
                                    mpam_resctrl._MpamBlocked
                            ) as context:
                                discover()

                    self.assertIn(path, str(context.exception))
                    self.assertIn(str(error), str(context.exception))
                    self.assertIs(context.exception.__cause__, error)

    def test_discovery_empty_directory_remains_empty(self):
        """An accessible empty directory is not an I/O error."""
        with TemporaryDirectory() as root:
            self.assertEqual(
                mpam_resctrl._mpam_l3_max_discovery_names(root), ())

    def test_discovery_error_blocks_live_launch_and_keeps_cleanup(self):
        """Discovery evidence reaches the live failure without starting stress."""
        error = PermissionError("live monitor discovery denied")

        create_group = _fixture_discovery_error_blocks_live_launch_and_keeps_cleanup_create_group

        with self.mock.patch.object(mpam_resctrl, "NvDev", return_value=self.mock.Mock(is_cpu_type=lambda _: True)), \
                self.mock.patch.object(resctrl_utils, "managed_mount", return_value=nullcontext((True, None))), \
                self.mock.patch.object(mpam_resctrl, "_mpam_write_and_require_percentage"), \
                self.mock.patch.object(
                mpam_resctrl.os, 'mkdir',
                side_effect=create_group), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    return_value={'L3_MAX': {0: 50}}), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair",
                    return_value=(10, 100)), \
                self.mock.patch.object(
                    mpam_resctrl.os, "scandir", side_effect=error), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen") as start, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_restore_percentage_schemata"
                ) as restore, \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("high", TestMpamMaxSchemataHelper._window_placement(), "/fake")

        self.assertIn("/fake/mon_data", str(context.exception))
        self.assertIn(str(error), str(context.exception))
        start.assert_not_called()
        self.assertEqual(restore.call_count, 2)
        self.assertEqual(rmdir.call_count, 2)

    def test_discovery_iteration_error_never_returns_partial_cache_ids(self):
        """An observed cache entry cannot hide a later enumeration error."""
        error = OSError("cache iterator failed")
        entry = self.mock.Mock()
        entry.name = "index0"

        def entries():
            yield entry
            raise error

        scan = self.mock.MagicMock()
        scan.__enter__.return_value = entries()
        with self.mock.patch.object(
                mpam_resctrl.os, "scandir", return_value=scan), \
                self.mock.patch("builtins.open", self.mock.mock_open(
                    read_data="3\n")):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_cache_ids_for_cpu(4)

        self.assertIn("/sys/devices/system/cpu/cpu4/cache",
                      str(context.exception))
        self.assertIn(str(error), str(context.exception))
        self.assertIs(context.exception.__cause__, error)
        scan.__exit__.assert_called_once()

    def test_discovery_monitors_preserve_absence_and_sorted_prefix_selection(self):
        """Absent monitors are skipped without changing duplicate-ID order."""
        with TemporaryDirectory() as group:
            monitor_path = Path(group, "mon_data")
            monitor_path.mkdir()
            for name in ("mon_L3_9suffix", "mon_L3_9", "mon_L3_09",
                         "mon_L3_10", "mon_L3_bad", "metadata"):
                (monitor_path / name).mkdir()
            for name in ("mon_L3_9", "mon_L3_09"):
                (monitor_path / name / "llc_occupancy").write_text(
                    "0", encoding="utf-8")
            chosen = monitor_path / "mon_L3_9suffix" / "llc_occupancy"
            chosen.symlink_to("absent-target")

            with self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_node_cpulists",
                    return_value=((2, "/unused/cpulist"),)), \
                    self.mock.patch.object(
                        mpam_resctrl, "_cpus_of_node",
                        return_value=(4,)), \
                    self.mock.patch.object(
                        mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                        return_value={9}):
                attribution = mpam_resctrl._mpam_l3_max_attribution(group)
                occupancy_files = mpam_resctrl._mpam_l3_max_occupancy_files(group, attribution)

        self.assertEqual(attribution["node"], 2)
        self.assertEqual(attribution["cache_ids"], (9,))
        self.assertEqual(occupancy_files, (str(chosen),))

    def test_discovery_nodes_preserve_absence_prefix_and_numeric_order(self):
        """Missing children are skipped; numeric order and links are retained."""
        node_path = "/sys/devices/system/node"
        real_scandir = mpam_resctrl.os.scandir
        real_lstat = mpam_resctrl.os.lstat
        with TemporaryDirectory() as root:
            for name in ("node10", "node2suffix", "node3", "node4", "metadata"):
                Path(root, name).mkdir()
            for name in ("node10", "node2suffix"):
                Path(root, name, "cpulist").write_text("4", encoding="utf-8")
            Path(root, "node4", "cpulist").symlink_to("absent-target")

            def node_lstat(path):
                return real_lstat(
                    Path(root) / Path(path).relative_to(node_path))

            with self.mock.patch.object(
                    mpam_resctrl.os, "scandir",
                    side_effect=lambda _path, root=root: real_scandir(root)), \
                    self.mock.patch.object(
                        mpam_resctrl.os, "lstat", side_effect=node_lstat):
                nodes = mpam_resctrl._mpam_l3_max_node_cpulists()

        self.assertEqual(nodes, (
            (2, node_path + "/node2suffix/cpulist"),
            (4, node_path + "/node4/cpulist"),
            (10, node_path + "/node10/cpulist"),
        ))

    def test_discovery_open_errors_preserve_path_and_cause(self):
        """All attribution discovery roots expose their original I/O error."""
        cases = (
            (partial(mpam_resctrl._mpam_l3_cache_ids_for_cpu, 4),
             "/sys/devices/system/cpu/cpu4/cache"),
            (mpam_resctrl._mpam_l3_max_node_cpulists,
             "/sys/devices/system/node"),
            (partial(mpam_resctrl._mpam_l3_max_attribution, "/fake"),
             "/fake/mon_data"),
        )
        for discover, path in cases:
            for error in (PermissionError("discovery denied"),
                          OSError("discovery I/O failure")):
                with self.subTest(path=path, error=type(error).__name__):
                    with self.mock.patch.object(
                            mpam_resctrl.os, "scandir", side_effect=error):
                        with self.assertRaises(
                                mpam_resctrl._MpamBlocked) as context:
                            discover()

                    self.assertIn(path, str(context.exception))
                    self.assertIn(str(error), str(context.exception))
                    self.assertIs(context.exception.__cause__, error)


class TestMpamMaxSchemataFilesystem(_MpamMaxSchemataHelpers):
    """Validate filesystem discovery and direct helper setup paths."""

    def test_resctrl_control_groups_returns_root_then_sorted_children_with_tasks(
            self):
        """Control-group discovery ignores metadata dirs without tasks."""
        with TemporaryDirectory() as mount:
            root = Path(mount)
            (root / "tasks").write_text("1\n", encoding="utf-8")
            for name in ("zeta", "alpha"):
                group = root / name
                group.mkdir()
                (group / "tasks").write_text("", encoding="utf-8")
            (root / "metadata").mkdir()
            (root / "regular").write_text("", encoding="utf-8")

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount):
                groups = mpam_resctrl._mpam_resctrl_control_groups()

        self.assertEqual(
            groups,
            (
                mount,
                str(root / "alpha"),
                str(root / "zeta"),
            ),
        )

    def test_resctrl_group_for_task_blocks_root_enumeration_errors(self):
        """Membership proof fails closed when root child enumeration fails."""
        for error in (PermissionError("denied"), OSError("io failed")):
            with self.subTest(error=type(error).__name__):
                with TemporaryDirectory() as mount:
                    Path(mount, "tasks").write_text("123\n", encoding="utf-8")

                    with self.mock.patch.object(
                            resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                            self.mock.patch.object(
                                mpam_resctrl.os, "scandir",
                                side_effect=error):
                        with self.assertRaises(
                                mpam_resctrl._MpamBlocked) as context:
                            mpam_resctrl._mpam_resctrl_group_for_task(123)

                self.assertIn(
                    "Could not enumerate resctrl control groups",
                    str(context.exception))

    def test_resctrl_group_for_task_blocks_child_entry_stat_errors(self):
        """A disappearing or unreadable child cannot become a skipped group."""
        class FakeEntry:
            """DirEntry double whose explicit stat fails."""

            name = "child"

            def __init__(self, error):
                self._error = error

            def is_dir(self, follow_symlinks=True):  # pylint: disable=unused-argument
                """Return the value glob would see before explicit stat."""
                return False

            def stat(self, follow_symlinks=True):  # pylint: disable=unused-argument
                """Raise the entry inspection error under test."""
                raise self._error

        class FakeScandir:
            """Context-manager scandir double for one root listing."""

            def __init__(self, entries):
                self._entries = entries

            def __enter__(self):
                return iter(self._entries)

            def __exit__(self, _exc_type, _exc, _traceback):
                return False

        for error in (
                PermissionError("entry denied"),
                FileNotFoundError("entry vanished"),
        ):
            with self.subTest(error=type(error).__name__):
                with TemporaryDirectory() as mount:
                    Path(mount, "tasks").write_text("123\n", encoding="utf-8")

                    with self.mock.patch.object(
                            resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                            self.mock.patch.object(
                                mpam_resctrl.os, "scandir",
                                return_value=FakeScandir((FakeEntry(error),))):
                        with self.assertRaises(
                                mpam_resctrl._MpamBlocked) as context:
                            mpam_resctrl._mpam_resctrl_group_for_task(123)

                self.assertIn(
                    "Could not inspect resctrl control group entry",
                    str(context.exception))

    def test_resctrl_group_for_task_blocks_child_tasks_stat_errors(self):
        """Non-absence errors while checking child tasks files fail closed."""
        class FakeEntry:
            """DirEntry double for an immediate resctrl child directory."""

            name = "child"

            def is_dir(self, follow_symlinks=True):  # pylint: disable=unused-argument
                """Return the value glob would see for a child directory."""
                return True

            def stat(self, follow_symlinks=True):  # pylint: disable=unused-argument
                """Return a directory stat-like object."""
                return self

            st_mode = mpam_resctrl.stat.S_IFDIR | 0o755

        class FakeScandir:
            """Context-manager scandir double for one root listing."""

            def __enter__(self):
                return iter((FakeEntry(),))

            def __exit__(self, _exc_type, _exc, _traceback):
                return False

        real_stat = mpam_resctrl.os.stat

        with TemporaryDirectory() as mount:
            root = Path(mount)
            (root / "tasks").write_text("123\n", encoding="utf-8")
            (root / "child").mkdir()

            def fake_stat(path):
                if path == str(root / "child" / "tasks"):
                    raise PermissionError("tasks denied")
                return real_stat(path)

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                    self.mock.patch.object(
                        mpam_resctrl.os, "scandir",
                        return_value=FakeScandir()), \
                    self.mock.patch.object(
                        mpam_resctrl.os, "stat", side_effect=fake_stat):
                with self.assertRaises(
                        mpam_resctrl._MpamBlocked) as context:
                    mpam_resctrl._mpam_resctrl_group_for_task(123)

        self.assertIn("Could not inspect", str(context.exception))
        self.assertIn("/child/tasks", str(context.exception))

    def test_resctrl_group_for_task_returns_control_without_monitor_group(self):
        """Direct CTRL-only membership records no monitor subgroup."""
        with TemporaryDirectory() as mount:
            Path(mount, "tasks").write_text("123\n", encoding="utf-8")

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount):
                groups = mpam_resctrl._mpam_resctrl_group_for_task(123)

        self.assertEqual(groups, (mount, None))

    def test_resctrl_group_for_task_returns_immediate_monitor_group(self):
        """Monitor subgroup discovery is limited to the owning CTRL group."""
        with TemporaryDirectory() as mount:
            root = Path(mount)
            monitor_group = root / "mon_groups" / "mon_a"
            nested_monitor = monitor_group / "mon_groups" / "nested"
            monitor_group.mkdir(parents=True)
            nested_monitor.mkdir(parents=True)
            (root / "tasks").write_text("123\n", encoding="utf-8")
            (monitor_group / "tasks").write_text("123\n", encoding="utf-8")
            (nested_monitor / "tasks").write_text("123\n", encoding="utf-8")

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount):
                groups = mpam_resctrl._mpam_resctrl_group_for_task(123)

        self.assertEqual(groups, (mount, str(monitor_group)))

    def test_resctrl_group_for_task_blocks_monitor_enumeration_errors(self):
        """Unreadable MON group enumeration cannot pass as no MON subgroup."""
        real_scandir = mpam_resctrl.os.scandir
        with TemporaryDirectory() as mount:
            root = Path(mount)
            monitor_root = root / "mon_groups"
            monitor_root.mkdir()
            (root / "tasks").write_text("123\n", encoding="utf-8")

            def scandir(path):
                if path == str(monitor_root):
                    raise PermissionError("mon scan denied")
                return real_scandir(path)

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                    self.mock.patch.object(
                    mpam_resctrl.os, "scandir", side_effect=scandir):
                with self.assertRaises(
                        mpam_resctrl._MpamBlocked) as context:
                    mpam_resctrl._mpam_resctrl_group_for_task(123)

        self.assertIn("Could not enumerate resctrl monitor groups",
                      str(context.exception))
        self.assertIn("mon scan denied", str(context.exception))

    def test_resctrl_group_for_task_blocks_monitor_tasks_read_errors(self):
        """Unreadable MON tasks cannot pass as no MON subgroup."""
        with TemporaryDirectory() as mount:
            root = Path(mount)
            monitor_group = root / "mon_groups" / "mon_a"
            monitor_group.mkdir(parents=True)
            (root / "tasks").write_text("123\n", encoding="utf-8")
            (monitor_group / "tasks").write_text("123\n", encoding="utf-8")

            def read_group_task_ids(group):
                if group == str(monitor_group):
                    raise PermissionError("monitor tasks denied")
                return {"123"}

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                    self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=read_group_task_ids):
                with self.assertRaises(
                        mpam_resctrl._MpamBlocked) as context:
                    mpam_resctrl._mpam_resctrl_group_for_task(123)

        self.assertIn("unreadable tasks files", str(context.exception))
        self.assertIn("monitor tasks denied", str(context.exception))

    def test_resctrl_group_for_task_blocks_duplicate_monitor_membership(self):
        """Ambiguous MON membership blocks instead of choosing a subgroup."""
        with TemporaryDirectory() as mount:
            root = Path(mount)
            monitor_a = root / "mon_groups" / "mon_a"
            monitor_b = root / "mon_groups" / "mon_b"
            monitor_a.mkdir(parents=True)
            monitor_b.mkdir(parents=True)
            (root / "tasks").write_text("123\n", encoding="utf-8")
            (monitor_a / "tasks").write_text("123\n", encoding="utf-8")
            (monitor_b / "tasks").write_text("123\n", encoding="utf-8")

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount):
                with self.assertRaises(
                        mpam_resctrl._MpamBlocked) as context:
                    mpam_resctrl._mpam_resctrl_group_for_task(123)

        self.assertIn("appears in 2 resctrl monitor groups",
                      str(context.exception))

    def test_l3_max_launch_failure_restores_original_monitor_with_real_discovery(
            self):
        """Launch cleanup restores both CTRL and MON membership before reporting."""
        events = []

        read_tasks = _fixture_resctrl_monitor_read_tasks

        write_tasks = _fixture_resctrl_monitor_write_tasks

        remove_task_from_all = partial(
            _fixture_resctrl_monitor_remove_task_from_all, (read_tasks, write_tasks))

        write_sysfs_value = partial(_fixture_resctrl_monitor_write_sysfs_value,
                                    (events, read_tasks, write_tasks, remove_task_from_all))

        with TemporaryDirectory() as mount:
            root = Path(mount)
            original_monitor = root / "mon_groups" / "original_mon"
            private_group = root / "glt_l3_max_unit"
            original_monitor.mkdir(parents=True)
            private_group.mkdir()
            (root / "tasks").write_text("777\n", encoding="utf-8")
            (original_monitor / "tasks").write_text("777\n", encoding="utf-8")
            (private_group / "tasks").write_text("", encoding="utf-8")

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                    self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                    self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    return_value={'L3_MAX': {0: 50}}), \
                    self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair",
                    return_value=(10, 100)), \
                    self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_occupancy_files",
                    return_value=(str(private_group / "occ"),)), \
                    self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                    self.mock.patch.object(
                    mpam_resctrl.threading, "get_native_id",
                    return_value=777), \
                    self.mock.patch.object(
                    mpam_resctrl.glt_utils, "write_sysfs_value",
                    side_effect=write_sysfs_value), \
                    self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                    self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    side_effect=OSError("no stress-ng")), \
                    self.mock.patch.object(
                    mpam_resctrl, "_mpam_restore_percentage_schemata"), \
                    self.mock.patch.object(mpam_resctrl.os, "rmdir"):
                with self.assertRaises(
                        GLTCore.GLTTestWarn) as context:
                    self._run_decorated_window(
                        "high", TestMpamMaxSchemataHelper._window_placement(), str(private_group))

            self.assertIn("Could not launch stress-ng workload: no stress-ng",
                          str(context.exception))
            self.assertEqual(read_tasks(root / "tasks"), {"777"})
            self.assertEqual(read_tasks(original_monitor / "tasks"), {"777"})
            self.assertEqual(
                events[-2:],
                [
                    ("write", str(root / "tasks"), "777"),
                    ("write", str(original_monitor / "tasks"), "777"),
                ],
            )

    def test_l3_max_helper_accepts_percent_min_zero_and_removes_group(self):
        """percent_min=0 is accepted when legal writes read back exactly."""
        result, group_exists = self._run_temp_helper(
            info=("percentage", "0", "1"),
            initial_line="L3_MAX:1=50;2=50")

        self.assertIn("selected low=0, high=100", result)
        self.assertFalse(group_exists)

    def test_missing_info_parent_fails_without_group(self):
        """Only resource-directory ENOENT is absence; metadata faults fail before mkdir."""
        with self._schemata_environment() as (root, events):
            info = root / "info"
            shutil.rmtree(info)
            with self.assertRaisesRegex(GLTCore.GLTTestFail, 'Could not stat resctrl info directory'):
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])

    def test_info_parent_permission_error_fails_without_group(self):
        """Only resource-directory ENOENT is absence; metadata faults fail before mkdir."""
        with self._schemata_environment() as (root, events):
            info = root / "info"
            real_stat = mpam_resctrl.os.stat

            def inspected(path, *args, **kwargs):
                if str(path) == str(info):
                    raise PermissionError("metadata denied")
                return real_stat(path, *args, **kwargs)

            with self.mock.patch.object(mpam_resctrl.os, "stat", side_effect=inspected):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, 'Could not stat resctrl info directory'):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])

    def test_info_parent_non_directory_fails_without_group(self):
        """Only resource-directory ENOENT is absence; metadata faults fail before mkdir."""
        with self._schemata_environment() as (root, events):
            info = root / "info"
            shutil.rmtree(info)
            info.write_text("regular file", encoding="ascii")
            with self.assertRaisesRegex(GLTCore.GLTTestFail, 'exists but is not a directory'):
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])

    def test_missing_resource_returns_resource_missing_without_group(self):
        """Only resource-directory ENOENT is absence; metadata faults fail before mkdir."""
        with self._schemata_environment() as (root, events):
            info = root / "info"
            resource = info / "L3_MAX"
            shutil.rmtree(resource)
            with self.assertRaisesRegex(GLTCore.GLTTestFail, 'info/L3_MAX is absent'):
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])

    def test_missing_resource_parent_disappears_fails_without_group(self):
        """Single-writer contract uses the first root stat and resource ENOENT only."""
        with self._schemata_environment() as (root, events):
            directory = self.mock.Mock(st_mode=mpam_resctrl.stat.S_IFDIR | 0o755)
            real_stat = mpam_resctrl.os.stat
            resource_stats = []

            def inspected_path(path, *args, **kwargs):
                if str(path) == str(root / "info"):
                    resource_stats.append("root")
                    return directory
                if str(path) == str(root / "info" / "L3_MAX"):
                    resource_stats.append("resource")
                    raise FileNotFoundError(errno.ENOENT, "missing resource")
                return real_stat(path, *args, **kwargs)

            with self.mock.patch.object(mpam_resctrl.os, "stat", side_effect=inspected_path):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "info/L3_MAX is absent"):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(resource_stats, ["root", "resource"])
            self.assertEqual(events, [])

    def test_missing_resource_parent_replaced_fails_without_group(self):
        """Single-writer contract uses the first root stat and resource ENOENT only."""
        with self._schemata_environment() as (root, events):
            directory = self.mock.Mock(st_mode=mpam_resctrl.stat.S_IFDIR | 0o755)
            real_stat = mpam_resctrl.os.stat
            resource_stats = []

            def inspected_path(path, *args, **kwargs):
                if str(path) == str(root / "info"):
                    resource_stats.append("root")
                    return directory
                if str(path) == str(root / "info" / "L3_MAX"):
                    resource_stats.append("resource")
                    raise FileNotFoundError(errno.ENOENT, "missing resource")
                return real_stat(path, *args, **kwargs)

            with self.mock.patch.object(mpam_resctrl.os, "stat", side_effect=inspected_path):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "info/L3_MAX is absent"):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(resource_stats, ["root", "resource"])
            self.assertEqual(events, [])

    def test_info_stat_permission_error_fails_without_group(self):
        """Only resource-directory ENOENT is absence; metadata faults fail before mkdir."""
        with self._schemata_environment() as (root, events):
            info = root / "info"
            resource = info / "L3_MAX"
            real_stat = mpam_resctrl.os.stat

            def inspected(path, *args, **kwargs):
                if str(path) == str(resource):
                    raise PermissionError("metadata denied")
                return real_stat(path, *args, **kwargs)

            with self.mock.patch.object(mpam_resctrl.os, "stat", side_effect=inspected):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, 'Could not stat L3_MAX info directory'):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])

    def test_info_stat_non_directory_fails_without_group(self):
        """Only resource-directory ENOENT is absence; metadata faults fail before mkdir."""
        with self._schemata_environment() as (root, events):
            info = root / "info"
            resource = info / "L3_MAX"
            shutil.rmtree(resource)
            resource.write_text("regular file", encoding="ascii")
            with self.assertRaisesRegex(GLTCore.GLTTestFail, 'exists but is not a directory'):
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, [])


class TestMpamMaxSchemataHelper(_MpamMaxSchemataHelpers):
    """Validate shared DGX-17928 MPAM percentage-schema helper logic."""



    def test_owned_cleanup_preserves_reason_precedence(self):
        """Cleanup removes only its new group and retains primary/cleanup failure order."""
        for phase in ("initial", "initial_remove", "restore", "body_remove", "ok"):
            with self.subTest(phase=phase), TemporaryDirectory() as mount:
                created = []

                with ExitStack() as stack:
                    patches = (
                        self.mock.patch.object(resctrl_utils, "RESCTRL_MOUNT_POINT", new=mount),
                        self.mock.patch.object(mpam_resctrl, "_mpam_read_percentage_info",
                                               return_value=(1, 1)),
                        self.mock.patch.object(mpam_resctrl.os, "mkdir", side_effect=created.append),
                        self.mock.patch.object(resctrl_utils, "read_schemata",
                                               side_effect=ValueError("initial read"))
                        if phase.startswith("initial") else self.mock.patch.object(
                            resctrl_utils, "read_schemata", return_value={"L3_MAX": {1: 50}}),
                        self.mock.patch.object(mpam_resctrl, "_mpam_write_and_require_percentage",
                                               side_effect=ValueError("body blocked"))
                        if phase == "body_remove" else self.mock.patch.object(
                            mpam_resctrl, "_mpam_write_and_require_percentage"),
                        self.mock.patch.object(mpam_resctrl, "_mpam_restore_percentage_schemata",
                                               side_effect=ValueError("restore failed"))
                        if phase == "restore" else self.mock.patch.object(
                            mpam_resctrl, "_mpam_restore_percentage_schemata"),
                        self.mock.patch.object(mpam_resctrl, "_mpam_select_distinct_percentage_pair",
                                               return_value=(10, 100)),
                    )
                    for patch in patches:
                        stack.enter_context(patch)
                    remove = stack.enter_context(self.mock.patch.object(
                        mpam_resctrl.os, "rmdir",
                        side_effect=OSError("remove failed") if "remove" in phase else None))
                    if phase == "ok":
                        mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
                        self.mock_logger.info.assert_called_with("L3_MAX selected low=%s, high=%s.", 10, 100)
                    else:
                        with self.assertRaises(GLTCore.GLTTestFail) as context:
                            mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
                        reason = str(context.exception)
                        self.assertIn(phase.split("_", maxsplit=1)[0], reason)
                        if "remove" in phase:
                            self.assertLess(reason.index(phase.split("_", maxsplit=1)[0]),
                                            reason.index("remove failed"))
                    self.assertEqual(len(created), 1)
                    remove.assert_called_once_with(created[0])


    @staticmethod
    def _window_placement():
        """Return the selected placement used by isolated window fixtures."""
        return {'node': 2, 'cpus': (4, 5), 'cache_ids': (9,)}

    def _live_window(self, label, median, *, samples=None, percentage=None):
        """Build a complete live-window result for comparison tests."""
        if samples is None:
            samples = tuple([median] * mpam_resctrl.MPAM_L3_MAX_LIVE_SAMPLE_COUNT)
        if percentage is None:
            percentage = 10 if label == "low" else 100
        return {'label': label, 'percentage': percentage, 'samples': samples, 'median': mpam_resctrl._median(samples)}

    @staticmethod
    def _stress_summary(skipped=0, passed=None, failed=0,
                        metrics_untrustworthy=0):
        """Build stress-ng status output without bogo-ops metrics."""
        if passed is None:
            passed = mpam_resctrl.MPAM_L3_MAX_LIVE_STRESS_WORKERS
        return (
            f"stress-ng: info: [123] skipped: {skipped}\n"
            f"stress-ng: info: [123] passed: {passed}: vm ({passed})\n"
            f"stress-ng: info: [123] failed: {failed}\n"
            "stress-ng: info: [123] metrics untrustworthy: "
            f"{metrics_untrustworthy}\n"
            "stress-ng: info: [123] successful run completed in 30.01 secs\n"
        )

    def _complete_stress(self, proc, _deadline, _argv, _capture):
        """Model successful wait before returning the expected status summary."""
        proc.wait(timeout=5)
        return self._stress_result()

    @staticmethod
    def _stress_result(returncode=0, output=None):
        """Build the typed stress-ng completion result used by live windows."""
        if output is None:
            output = TestMpamMaxSchemataHelper._stress_summary()
        return subprocess.CompletedProcess([], returncode, stdout=output)

    class _FakeClock:
        """Tiny monotonic clock for live-window deadline tests."""

        def __init__(self, now=0):
            self.now = now
            self.sleeps = []

        def monotonic(self):
            """Return the current fake monotonic time."""
            return self.now

        def sleep(self, seconds):
            """Record fixed sleeps and advance fake time."""
            self.sleeps.append(seconds)
            self.now += seconds

    def _run_l3_max_live_window_restore_case(
            self, group_for_task, *, read_group_task_ids=None,
            monotonic_side_effect=None, sleep_mock=None):
        """Run the real restore/deadline policy using the shared window fixture."""
        with self._window_environment(tasks=read_group_task_ids) as (handles, _, _):
            handles["_mpam_resctrl_group_for_task"].side_effect = group_for_task
            handles["write_sysfs_value"].side_effect = getattr(
                self, "_l3_max_write_task_side_effect", None)
            stress_wait = getattr(self, "_l3_max_stress_wait_side_effect", None)
            if stress_wait is not None:
                handles["_mpam_wait_for_l3_max_stress_exit"].side_effect = stress_wait
            if monotonic_side_effect is not None:
                handles["monotonic"].side_effect = monotonic_side_effect
            else:
                handles["monotonic"].side_effect = None
                handles["monotonic"].return_value = 0
            sleep = sleep_mock or self.mock.Mock()
            handles["sleep"].side_effect = sleep
            result = self._run_decorated_window("high", self._window_placement(), "/fake")
            return result, handles["write_sysfs_value"], sleep

    @contextmanager
    def _window_environment(self, occupancy=(0,), tasks=None):
        """Keep real window lifetime and expose only hardware/process boundaries."""
        clock = self._FakeClock()
        proc = self.mock.Mock(spec=subprocess.Popen)
        proc.pid = 4321
        proc.poll.return_value = None
        def read_tasks(_group):
            last_write = handles["write_sysfs_value"].call_args
            return {"777"} if last_write and last_write.args[0] == "/fake/tasks" else set()
        patches = (
            ("_mpam_l3_max_occupancy_files", self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_occupancy_files", return_value=("/fake/occ",))),
            ("sum_l3_occupancy", self.mock.patch.object(
                resctrl_utils, "sum_l3_occupancy",
                side_effect=chain(occupancy, repeat(occupancy[-1])))),
            ("get_native_id", self.mock.patch.object(
                mpam_resctrl.threading, "get_native_id", return_value=777)),
            ("_mpam_resctrl_group_for_task", self.mock.patch.object(
                mpam_resctrl, "_mpam_resctrl_group_for_task", return_value=("/root", None))),
            ("write_sysfs_value", self.mock.patch.object(mpam_resctrl.glt_utils, "write_sysfs_value")),
            ("_mpam_read_group_task_ids", self.mock.patch.object(
                mpam_resctrl, "_mpam_read_group_task_ids",
                side_effect=tasks if tasks is not None else read_tasks)),
            ("_mpam_write_and_require_percentage", self.mock.patch.object(
                mpam_resctrl, "_mpam_write_and_require_percentage")),
            ("TemporaryFile", self.mock.patch.object(tempfile, "TemporaryFile", side_effect=lambda **_kwargs:
                io.BytesIO(b"stress-ng: debug: assuming killed by OOM killer"))),
            ("Popen", self.mock.patch.object(mpam_resctrl.subprocess, "Popen", return_value=proc)),
            ("_mpam_wait_for_l3_max_membership", self.mock.patch.object(
                mpam_resctrl, "_mpam_wait_for_l3_max_membership")),
            ("_mpam_collect_l3_max_samples", self.mock.patch.object(
                mpam_resctrl, "_mpam_collect_l3_max_samples", return_value=(123,) * 10)),
            ("_mpam_wait_for_l3_max_stress_exit", self.mock.patch.object(
                mpam_resctrl, "_mpam_wait_for_l3_max_stress_exit", return_value=self._stress_result())),
            ("monotonic", self.mock.patch.object(mpam_resctrl.time, "monotonic", side_effect=clock.monotonic)),
            ("sleep", self.mock.patch.object(mpam_resctrl.time, "sleep", side_effect=clock.sleep)),
            ("_mpam_restore_percentage_schemata", self.mock.patch.object(
                mpam_resctrl, "_mpam_restore_percentage_schemata")),
            ("rmdir", self.mock.patch.object(mpam_resctrl.os, "rmdir")),
        )
        with ExitStack() as stack:
            handles = {name: stack.enter_context(patch) for name, patch in patches}
            yield handles, proc, clock

    def test_reopened_stale_baseline_recovers_before_launch(self):
        """Wait for old occupancy to clear without weakening the strict baseline."""
        tasks = chain([set()] * 4, [{"777"}], repeat(set()))
        with self._window_environment((2 * 1024 * 1024, 0), tasks) as (handles, _, clock):
            result = self._run_decorated_window("high", self._window_placement(), "/fake")
            self.assertEqual(result["samples"], (123,) * 10)
            handles["Popen"].assert_called_once()
            self.assertIn(0.1, clock.sleeps)

    def test_reopened_persistent_baseline_stops_at_five_seconds(self):
        """Stale occupancy has its own bounded budget before process launch."""
        with self._window_environment((2 * 1024 * 1024,), repeat(set())) as (handles, _, clock):
            with self.assertRaises(GLTCore.GLTTestWarn) as caught:
                self._run_decorated_window("high", self._window_placement(), "/fake")
            self.assertIn("2097152", str(caught.exception))
            self.assertAlmostEqual(clock.now, 5)
            handles["Popen"].assert_not_called()
            handles["write_sysfs_value"].assert_not_called()

    def test_reopened_resident_margin_boundaries(self):
        """90% is the selected residency precondition, independent of separation."""
        page = mpam_resctrl.mmap.PAGESIZE
        required = (1024 ** 3 * 90 + 99) // 100
        enough_pages = (required + page - 1) // page
        for pages, accepted in ((enough_pages - 1, False),
                                (enough_pages, True), (1024 ** 3 // page, True)):
            with self.subTest(pages=pages), ExitStack() as stack:
                stack.enter_context(self.mock.patch.object(mpam_resctrl.glt_proc, "task_cpu",
                                                          return_value=4))
                stack.enter_context(self.mock.patch.object(mpam_resctrl.glt_proc, "task_allowed_cpus",
                                                          return_value={4, 5}))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    side_effect=[{2: pages - 3}, {2: 1}, {2: 1}, {2: 1}]))
                outcome = nullcontext() if accepted else self.assertRaises(mpam_resctrl._MpamBlocked)
                with outcome:
                    mpam_resctrl._mpam_validate_l3_max_placement(
                        {"1", "2", "3", "4"}, {"1", "2", "3", "4"},
                        self._window_placement(), require_worker_footprint=True)

    def test_reopened_sample_proofs_follow_sleep_before_next_observation(self):
        """Persistent drift after an interval stops before the next observation."""
        state = _SampleProofs("persistent drift")
        proc = self.mock.Mock()
        proc.poll.return_value = None
        with self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot",
                                    side_effect=state.proof), \
                self.mock.patch.object(mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep", side_effect=state.sleep), \
                self.mock.patch.object(resctrl_utils, "sum_l3_occupancy", side_effect=state.read):
            with self.assertRaisesRegex(mpam_resctrl._MpamBlocked, "sample 1 before.*persistent drift"):
                mpam_resctrl._mpam_collect_l3_max_samples(
                    proc, 30, ("/fake", self._window_placement(), ("/fake/occ",)), "high", self.mock_logger)
        self.assertEqual(state.events, [("proof", 0, "before"), "read", "sleep",
                                       ("proof", 1, "before")])
        messages = self._formatted_logs(self.mock_logger)
        self.assertEqual(sum("sample=" in message for message in messages), 1)

    def test_reopened_topology_blockage_keeps_output_after_group_absence(self):
        """A missing leaf retains stress-ng evidence without claiming kernel OOM."""
        with self._window_environment() as (handles, _proc, _):
            events = []
            handles["_mpam_wait_for_l3_max_membership"].side_effect = mpam_resctrl._MpamBlocked(
                "missing workload leaf")
            absent = partial(_fixture_reopened_topology_blockage_keeps_output_after_group_absence_absent, (events,))
            original_output = mpam_resctrl._mpam_l3_max_output
            output = partial(
                _fixture_reopened_topology_blockage_keeps_output_after_group_absence_output, (events, original_output))
            with self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=absent), \
                    self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_output", side_effect=output), \
                    self.assertRaises(GLTCore.GLTTestWarn) as caught:
                self._run_decorated_window("high", self._window_placement(), "/fake")
            self.assertIn("missing workload leaf", str(caught.exception))
            self.assertIn("assuming killed by OOM killer", str(caught.exception))
            self.assertEqual(events, ["gone", "gone", "output", "gone"])
            self.assertIn("--oomable", handles["Popen"].call_args.args[0])

    @staticmethod
    def _formatted_logs(logger):
        """Format emitted log values without inventing an evidence schema."""
        return [call.args[0] % call.args[1:] for call in logger.info.call_args_list
                if call.args and isinstance(call.args[0], str)]

    def _run_l3_max_live_window_mocked(self, *, occupancy_reads=None, task_reads=None):
        """Expose common handles for baseline and supervisor rejection cases."""
        result = error = None
        with self._window_environment(occupancy_reads or (0,), task_reads) as (handles, _, _):
            try:
                result = self._run_decorated_window("high", self._window_placement(), "/fake")
            except (GLTCore.GLTTestWarn, GLTCore.GLTTestFail) as caught:
                error = caught
            aliases = {"occupancy": "sum_l3_occupancy", "write_task": "write_sysfs_value",
                       "restore": "_mpam_restore_percentage_schemata"}
            handles.update({alias: handles[name] for alias, name in aliases.items()})
        return result, error, handles

    def test_l3_max_live_stress_argv_is_exact_contract(self):
        """Native live-effect launches the approved stress-ng command."""
        self.assertEqual(
            mpam_resctrl._mpam_l3_max_stress_argv(2),
            [
                "numactl", "--cpunodebind=2", "--membind=2",
                "stress-ng", "--vm", "4", "--vm-bytes", "1G",
                "--vm-keep", "--oomable", "--timeout", "30s",
            ],
        )

    def test_l3_max_stress_output_capture_is_bounded(self):
        """Failure evidence captures output without unbounded memory growth."""
        proc = self.mock.Mock()
        output_file = tempfile.TemporaryFile(mode="w+b")
        self.addCleanup(output_file.close)
        output_file.write(
            b"a" * (mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES + 3))

        with self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=ProcessLookupError):
            output = mpam_resctrl._mpam_l3_max_output(proc.pid, output_file)

        marker = (
            f"<kept last {mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES} "
            f"of {mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES + 3} bytes>"
        )
        self.assertEqual(output.count(marker), 1)
        self.assertNotIn("truncated after", output)

    def test_l3_max_live_early_stress_exit_reports_status_and_output(self):
        """Early stress-ng exit blocks with rc/output without membership spin."""
        clock = self._FakeClock()
        launched = []
        trailer = b"bind failed\n"
        stress_output = b"x" * (20480 - len(trailer)) + trailer
        marker = (
            f"<kept last {mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES} "
            "of 20480 bytes>"
        )

        class FakePopen:
            """Popen double that exits before membership can be proven."""

            pid = 4321

            def __init__(self, _argv, **kwargs):
                self.wait_timeouts = []
                self.poll_results = []
                self.output_file = kwargs["stdout"]
                self.output_file.write(stress_output)
                kwargs["stdout"].flush()
                launched.append(self)

            def poll(self):
                """Report the early failed status."""
                self.poll_results.append(3)
                return 3

            def wait(self, timeout=None):
                """Let cleanup reap the already-exited leader."""
                self.wait_timeouts.append(timeout)
                return 3

        original_membership = mpam_resctrl._mpam_wait_for_l3_max_membership
        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        original_start = mpam_resctrl.subprocess.Popen
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair",
                                       return_value=(10, 100)), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=("/original", None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen", new=FakePopen), \
                self.mock.patch.object(
                    mpam_resctrl,
                    "_mpam_validate_l3_max_membership_snapshot",
                    side_effect=mpam_resctrl._MpamBlocked(
                        "membership not ready")) as snapshot, \
                self.mock.patch.object(
                    mpam_resctrl.os, "killpg",
                    side_effect=ProcessLookupError), \
                self.mock.patch.object(
                    mpam_resctrl.time, "monotonic",
                    side_effect=clock.monotonic), \
                self.mock.patch.object(
                    mpam_resctrl.time, "sleep", side_effect=clock.sleep):
            environment["_mpam_wait_for_l3_max_membership"].side_effect = original_membership
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            environment["Popen"].side_effect = original_start
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("low", self._window_placement(), "/fake")

        self.assertIn("status 3", str(context.exception))
        self.assertIn("bind failed", str(context.exception))
        self.assertEqual(str(context.exception).count(marker), 1)
        self.assertLessEqual(snapshot.call_count, 1)
        self.assertEqual(len(launched), 1)
        self.assertTrue(launched[0].poll_results)
        self.assertEqual(set(launched[0].poll_results), {3})
        self.assertFalse(launched[0].wait_timeouts)
        self.assertTrue(launched[0].output_file.closed)

    def test_l3_max_live_early_exit_captures_after_straggler_cleanup(self):
        """Exited leader output is read only after surviving group members are gone."""
        clock = self._FakeClock()
        launched = []
        read_states = []
        group_active = {"value": True}
        trailer = b"bind failed\n"
        stress_output = b"x" * (20480 - len(trailer)) + trailer
        marker = (
            f"<kept last {mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES} "
            "of 20480 bytes>"
        )
        original_read_output = mpam_resctrl._mpam_l3_max_output

        class FakePopen:
            """Popen double whose exited leader leaves a live process group."""

            pid = 4321

            def __init__(self, _argv, **kwargs):
                self.wait_timeouts = []
                self.poll_results = []
                self.output_file = kwargs["stdout"]
                self.output_file.write(stress_output)
                self.output_file.flush()
                launched.append(self)

            def poll(self):
                """Report the early failed status."""
                self.poll_results.append(3)
                return 3

            def wait(self, timeout=None):
                """Reap the already-exited leader during cleanup."""
                self.wait_timeouts.append(timeout)
                return 3

        killpg = partial(_fixture_early_exit_captures_after_straggler_cleanup_killpg, (group_active,))

        read_output = partial(_fixture_straggler_output,
                              (read_states, group_active, original_read_output))

        original_membership = mpam_resctrl._mpam_wait_for_l3_max_membership
        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        original_start = mpam_resctrl.subprocess.Popen
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair",
                                       return_value=(10, 100)), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=("/original", None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen", new=FakePopen), \
                self.mock.patch.object(
                    mpam_resctrl,
                    "_mpam_validate_l3_max_membership_snapshot",
                    side_effect=mpam_resctrl._MpamBlocked(
                        "membership not ready")) as snapshot, \
                self.mock.patch.object(
                    mpam_resctrl.os, "killpg", side_effect=killpg) as killpg_mock, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_output",
                    side_effect=read_output, autospec=True), \
                self.mock.patch.object(
                    mpam_resctrl.time, "monotonic",
                    side_effect=clock.monotonic), \
                self.mock.patch.object(
                    mpam_resctrl.time, "sleep", side_effect=clock.sleep):
            environment["_mpam_wait_for_l3_max_membership"].side_effect = original_membership
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            environment["Popen"].side_effect = original_start
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("low", self._window_placement(), "/fake")

        self.assertIn("status 3", str(context.exception))
        self.assertIn("bind failed", str(context.exception))
        self.assertEqual(str(context.exception).count(marker), 1)
        self.assertEqual(read_states, [False])
        self.assertLessEqual(snapshot.call_count, 1)
        self.assertNotIn(
            self.mock.call(4321, mpam_resctrl.signal.SIGTERM),
            killpg_mock.call_args_list)
        self.assertIn(
            self.mock.call(4321, mpam_resctrl.signal.SIGKILL),
            killpg_mock.call_args_list)
        self.assertGreaterEqual(
            killpg_mock.call_args_list.count(self.mock.call(4321, 0)), 3)
        self.assertEqual(len(launched), 1)
        self.assertTrue(launched[0].poll_results)
        self.assertIn(3, launched[0].poll_results)
        self.assertFalse(launched[0].wait_timeouts)
        self.assertTrue(launched[0].output_file.closed)

    def test_l3_max_live_after_first_sample_exit_reports_status_and_output(self):
        """A post-sample stress exit keeps rc/output after cleanup proof."""
        clock = self._FakeClock()
        launched = []
        read_states = []
        occupancy_reads = []
        group_active = {"value": True}
        trailer = b"bind failed\n"
        stress_output = b"x" * (20480 - len(trailer)) + trailer
        marker = (
            f"<kept last {mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES} "
            "of 20480 bytes>"
        )
        original_read_output = mpam_resctrl._mpam_l3_max_output

        class FakePopen:
            """Popen double that exits after the first occupancy sample."""

            pid = 4321

            def __init__(self, _argv, **kwargs):
                self.exited = False
                self.wait_timeouts = []
                self.poll_results = []
                self.output_file = kwargs["stdout"]
                self.output_file.write(stress_output)
                self.output_file.flush()
                launched.append(self)

            def poll(self):
                """Report rc only after the sample reader flips exited."""
                result = 3 if self.exited else None
                self.poll_results.append(result)
                return result

            def wait(self, timeout=None):
                """Reap the already-exited leader during cleanup."""
                self.wait_timeouts.append(timeout)
                self.exited = True
                return 3

        sum_occupancy = partial(
            _fixture_after_first_sample_exit_reports_status_and_output_sum_occupancy, (occupancy_reads, launched))

        killpg = partial(_fixture_early_exit_captures_after_straggler_cleanup_killpg, (group_active,))

        read_output = partial(_fixture_straggler_output,
                              (read_states, group_active, original_read_output))

        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        original_start = mpam_resctrl.subprocess.Popen
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair",
                                       return_value=(10, 100)), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=sum_occupancy), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=("/original", None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen", new=FakePopen), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"), \
                self.mock.patch.object(
                    mpam_resctrl,
                    "_mpam_validate_l3_max_membership_snapshot"
                ) as snapshot, \
                self.mock.patch.object(
                    mpam_resctrl.os, "killpg", side_effect=killpg
                ) as killpg_mock, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_output",
                    side_effect=read_output, autospec=True), \
                self.mock.patch.object(
                    mpam_resctrl.time, "monotonic",
                    side_effect=clock.monotonic), \
                self.mock.patch.object(
                    mpam_resctrl.time, "sleep", side_effect=clock.sleep):
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            environment["Popen"].side_effect = original_start
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("low", self._window_placement(), "/fake")

        self.assertIn("status 3", str(context.exception))
        self.assertIn("after sample 0", str(context.exception))
        self.assertIn("bind failed", str(context.exception))
        self.assertEqual(str(context.exception).count(marker), 1)
        self.assertEqual(read_states, [False])
        self.assertEqual(len(occupancy_reads), 2)
        self.assertTrue(snapshot.called)
        self.assertIn(
            self.mock.call(4321, mpam_resctrl.signal.SIGKILL),
            killpg_mock.call_args_list)
        self.assertNotIn(
            self.mock.call(4321, mpam_resctrl.signal.SIGTERM),
            killpg_mock.call_args_list)
        self.assertFalse(group_active["value"])
        self.assertEqual(len(launched), 1)
        self.assertTrue(launched[0].poll_results)
        self.assertIn(3, launched[0].poll_results)
        self.assertFalse(launched[0].wait_timeouts)
        self.assertTrue(launched[0].output_file.closed)

    def test_l3_max_live_early_exit_skips_output_without_absence_proof(self):
        """Cleanup failure never reads output that may still have active writers."""
        clock = self._FakeClock()
        launched = []
        capture = tempfile.TemporaryFile(mode="w+b")
        self.addCleanup(capture.close)
        trailer = b"bind failed\n"
        stress_output = b"x" * (20480 - len(trailer)) + trailer

        class FakePopen:
            """Popen double whose process group survives forced cleanup."""

            pid = 4321

            def __init__(self, _argv, **kwargs):
                self.wait_timeouts = []
                self.output_file = kwargs["stdout"]
                self.output_file.write(stress_output)
                self.output_file.flush()
                launched.append(self)

            def poll(self):
                """Report the early failed status."""
                return 3

            def wait(self, timeout=None):
                """Reap the already-exited leader during cleanup."""
                self.wait_timeouts.append(timeout)
                return 3

        original_membership = mpam_resctrl._mpam_wait_for_l3_max_membership
        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        original_start = mpam_resctrl.subprocess.Popen
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair",
                                       return_value=(10, 100)), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=("/original", None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen", new=FakePopen), \
                self.mock.patch.object(
                    mpam_resctrl,
                    "_mpam_validate_l3_max_membership_snapshot",
                    side_effect=mpam_resctrl._MpamBlocked(
                        "membership not ready")), \
                self.mock.patch.object(mpam_resctrl.os, "killpg"), \
                self.mock.patch.object(
                    capture, "read", wraps=capture.read
                ) as read_output, \
                self.mock.patch.object(
                    tempfile, "TemporaryFile",
                    return_value=capture), \
                self.mock.patch.object(
                    mpam_resctrl.time, "monotonic",
                    side_effect=clock.monotonic), \
                self.mock.patch.object(
                    mpam_resctrl.time, "sleep", side_effect=clock.sleep):
            environment["_mpam_wait_for_l3_max_membership"].side_effect = original_membership
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            environment["Popen"].side_effect = original_start
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                self._run_decorated_window("low", self._window_placement(), "/fake")

        self.assertIn("output unread: process group not proven gone",
                      str(context.exception))
        self.assertIn("still exists after 5 seconds", str(context.exception))
        read_output.assert_not_called()
        self.assertEqual(len(launched), 1)
        self.assertTrue(launched[0].output_file.closed)


    def test_l3_max_stress_output_capture_retains_late_summary(self):
        """Bounded stress output keeps the footer status after long chatter."""
        proc = self.mock.Mock()
        output_file = tempfile.TemporaryFile(mode="w+b")
        self.addCleanup(output_file.close)
        output_file.write(
            b"a" * (
                mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES + 128
            )
            + b"\n"
            + self._stress_summary().encode("utf-8")
        )

        with self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=ProcessLookupError):
            output = mpam_resctrl._mpam_l3_max_output(proc.pid, output_file)
        summary = mpam_resctrl._mpam_l3_max_parse_stress_summary(output)

        self.assertEqual(summary['passed'], 4)
        self.assertLessEqual(
            len(output.encode("utf-8", errors="replace")),
            mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES + 256,
        )

    def test_l3_max_stress_exit_result_captures_status_and_output(self):
        """Completion status is explicit and output is evidence-only."""
        proc = self.mock.Mock()
        output_file = tempfile.TemporaryFile(mode="w+b")
        self.addCleanup(output_file.close)
        proc._glt_l3_max_output_file = output_file
        output_file.write(b"stress-ng: info: metrics unavailable\n")
        proc.wait.return_value = 7
        proc.output.return_value = "stress-ng: info: metrics unavailable"

        with self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=ProcessLookupError):
            result = mpam_resctrl._mpam_wait_for_l3_max_stress_exit(proc, 30, [], output_file)

        self.assertEqual(result.returncode, 7)
        self.assertIn("metrics unavailable", result.stdout)
        proc.wait.assert_called_once()

    def test_l3_max_stress_summary_accepts_exact_status_counts(self):
        """Successful stress-ng completion requires four passed VM workers."""
        summary = mpam_resctrl._mpam_validate_l3_max_stress_result(
            self._stress_result())

        self.assertEqual(summary['passed'], 4)
        self.assertEqual(summary['skipped'], 0)
        self.assertEqual(summary['failed'], 0)
        self.assertEqual(summary['metrics_untrustworthy'], 0)

    def test_l3_max_stress_summary_rejects_later_conflicting_counts(self):
        """A later contradictory footer count cannot be hidden by re.search."""
        cases = (
            ("skipped", 1),
            ("passed", 3),
            ("failed", 1),
            ("metrics untrustworthy", 1),
        )

        for label, later_count in cases:
            with self.subTest(label=label):
                result = self._stress_result(
                    output=(
                        self._stress_summary()
                        + f"stress-ng: info: [123] {label}: {later_count}\n"
                    ))

                with self.assertRaises(
                        mpam_resctrl._MpamBlocked) as context:
                    mpam_resctrl._mpam_validate_l3_max_stress_result(result)

                self.assertIn(f"multiple '{label}:'", str(context.exception))

    def test_l3_max_stress_summary_rejects_duplicate_same_count(self):
        """Even duplicate equal counts are ambiguous footer evidence."""
        result = self._stress_result(
            output=(
                self._stress_summary()
                + "stress-ng: info: [123] failed: 0 repeated footer\n"
            ))

        with self.assertRaises(mpam_resctrl._MpamBlocked) as context:
            mpam_resctrl._mpam_validate_l3_max_stress_result(result)

        self.assertIn("multiple 'failed:'", str(context.exception))

    def test_l3_max_stress_summary_blocks_when_status_lines_are_absent(self):
        """Return code alone is insufficient without the summary status lines."""
        result = self._stress_result(
            output="stress-ng: info: [123] successful run completed in 30s\n")

        with self.assertRaises(mpam_resctrl._MpamBlocked) as context:
            mpam_resctrl._mpam_validate_l3_max_stress_result(result)

        self.assertIn("required 'skipped:' status summary",
                      str(context.exception))

    def test_l3_max_stress_summary_fails_nonzero_bad_counts(self):
        """Completed stress-ng status lines must report no bad outcomes."""
        result = self._stress_result(
            output=self._stress_summary(failed=1))

        with self.assertRaises(ValueError) as context:
            mpam_resctrl._mpam_validate_l3_max_stress_result(result)

        self.assertIn("non-zero bad counts", str(context.exception))
        self.assertIn("'failed': 1", str(context.exception))

    def test_l3_max_stress_summary_fails_wrong_passed_count(self):
        """The status summary must match the requested four VM workers."""
        result = self._stress_result(
            output=self._stress_summary(passed=3))

        with self.assertRaises(ValueError) as context:
            mpam_resctrl._mpam_validate_l3_max_stress_result(result)

        self.assertIn("passed=3", str(context.exception))
        self.assertIn("expected 4", str(context.exception))

    @contextmanager
    def _calibration_patches(self):
        """Expose the real decorated group sequence with one calibrated placement."""
        with self._live_environment():
            yield

    def test_reported_context_cleanup_fault_stops_before_next_window(self):
        """A context cleanup error without a body exception forbids another launch."""
        with self._live_environment() as (_root, events, _choose, start, state), \
                self.mock.patch.object(tempfile, "TemporaryFile", side_effect=_FaultyCloseCapture), \
                self.mock.patch.object(mpam_resctrl, "_mpam_evaluate_l3_max_live_windows") as evaluate:
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "capture close denied"):
                mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual(start.call_count, 1)
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"] * 2)
            self.assertEqual(len(state.sample_reads), 10)
            evaluate.assert_not_called()
        self.assertNotIn("live-effect PASS", " ".join(self._formatted_logs(self.mock_logger)))

    def test_no_pair_fact_defers_each_outcome_until_owned_cleanup(self):
        """Every MAX case preserves no-pair WARN and cleanup-FAIL precedence."""
        entries = (
            ("L2_MAX", mpam_resctrl.test_mpam_l2_max_schemata),
            ("L3_MAX", mpam_resctrl.test_mpam_l3_max_schemata),
            ("L3_MAX", mpam_resctrl.test_mpam_l3_max_live_effect),
        )
        for resource, entry in entries:
            for fault in (None, "restore", "rmdir", "unmount"):
                with self.subTest(entry=entry.__name__, fault=fault):
                    self._no_pair_cleanup_case(resource, entry, fault)

    def _no_pair_cleanup_case(self, resource, entry, fault):
        """Drive the concrete fact contract through every decorated cleanup boundary."""
        live = entry is mpam_resctrl.test_mpam_l3_max_live_effect
        environment = (self._live_environment() if live else self._schemata_environment(
            resource, info=("percentage", "10", "10")))
        with environment as observed, ExitStack() as stack:
            events = observed[1]
            pair = stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_probe_percentage_pair", return_value=None))
            start = stack.enter_context(self.mock.patch.object(mpam_resctrl.subprocess, "Popen"))
            choose = stack.enter_context(self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_attribution"))
            if fault == "restore":
                stack.enter_context(self.mock.patch.object(mpam_resctrl, "_mpam_restore_percentage_schemata",
                                                          side_effect=ValueError("restore failed")))
            elif fault == "rmdir":
                stack.enter_context(self.mock.patch.object(mpam_resctrl.os, "rmdir",
                                                          side_effect=OSError("rmdir failed")))
            elif fault == "unmount":
                stack.enter_context(self.mock.patch.object(resctrl_utils, "managed_mount",
                    side_effect=partial(self._mounted, cleanup_error="unmount failed")))
            expected = GLTCore.GLTTestFail if fault else GLTCore.GLTTestWarn
            with self.assertRaises(expected) as caught:
                entry(None, self.mock_logger)
            reason = (f"{resource} reported grid {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]!r} "
                      "has no two distinct legal values that both differ from the restored initial "
                      "schemata on every domain.")
            self.assertIn(reason, str(caught.exception))
            if fault:
                self.assertIn(f"{fault} failed", str(caught.exception))
                self.assertNotIn("calibration cleanup failed", str(caught.exception))
            self.assertEqual([kind for kind, _path in events],
                             ["mkdir"] if fault == "rmdir" else ["mkdir", "rmdir"])
            pair.assert_called_once()
            choose.assert_not_called()
            start.assert_not_called()

    def test_policy_reason_preserves_exact_diagnostics_and_valid_bounds(self):
        """Only policy bounds block; valid extremes always contain their first grid value."""
        for minimum, granularity, field in ((-1, 1, "percent_min"), (100, 1, "percent_min"),
                                             (1, 0, "percent_gran"), (1, 101, "percent_gran")):
            value = minimum if field == "percent_min" else granularity
            bounds = "0 <= percent_min < 100" if field == "percent_min" else "1 <= percent_gran <= 100"
            self.assertEqual(mpam_resctrl._mpam_percentage_policy_reason("L3_MAX", minimum, granularity),
                             f"L3_MAX {field} {value} is outside the userspace test-policy range {bounds}.")
        for minimum, granularity in ((0, 1), (0, 100), (99, 1), (99, 100)):
            self.assertIsNone(mpam_resctrl._mpam_percentage_policy_reason("L3_MAX", minimum, granularity))
            self.assertEqual(list(range(minimum, 101, granularity))[0], minimum)

    def test_l3_max_live_effect_runs_six_windows_in_fixed_order(self):
        """Six actual launches preserve calibrated placement and all 60 observations."""
        with self._live_environment() as (_root, _events, _choose, start, state), \
                self.mock.patch.object(mpam_resctrl, "_mpam_evaluate_l3_max_live_windows",
                                       wraps=mpam_resctrl._mpam_evaluate_l3_max_live_windows) as evaluate:
            mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual(start.call_count, 6)
            self.assertEqual(state.percentages, [10, 100] * 3)
            self.assertEqual(state.discoveries, state.groups)
            self.assertEqual(len(state.sample_reads), 60)
            windows = evaluate.call_args.args[0]
            self.assertEqual([window["label"] for window in windows], ["low", "high"] * 3)
            self.assertTrue(all(len(window["samples"]) == 10 for window in windows))
            self.assertTrue(all(call.args[0] == mpam_resctrl._mpam_l3_max_stress_argv(2)
                                for call in start.call_args_list))
        self.assertIn("L3_MAX live-effect PASS", " ".join(self._formatted_logs(self.mock_logger)))

    def test_l3_max_live_effect_calibrates_pair_and_placement_once(self):
        """Real integer-domain calibration finishes cleanup before all six windows."""
        with self._live_environment() as (root, events, choose, window, state):
            mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            choose.assert_called_once()
            self.assertEqual(window.call_count, 6)
            self.assertEqual([event[0] for event in events], ["mkdir", "rmdir"] * 7)
            self.assertEqual(list(root.glob("glt_*")), [])
            self.assertEqual(state.percentages, [10, 100] * 3)
            self.assertIn("L3_MAX placement node=2 cpus=(4, 5) cache IDs=(9,).",
                          self._formatted_logs(self.mock_logger))


    def test_l3_max_window_monitor_paths_belong_to_each_fresh_group(self):
        """A calibrated cache identity must bind to the current group's files."""
        placement = {**self._window_placement(), "cache_ids": (9, 10)}
        with TemporaryDirectory() as mount:
            paths = []
            for name in ("calibration", "low", "high"):
                group = Path(mount) / name
                for cache_id in (9, 10, 99):
                    monitor = group / "mon_data" / f"mon_L3_{cache_id}"
                    monitor.mkdir(parents=True)
                    (monitor / "llc_occupancy").write_text("0\n", encoding="ascii")
                result = mpam_resctrl._mpam_l3_max_occupancy_files(str(group), placement)
                expected = tuple(str(group / "mon_data" / f"mon_L3_{cache_id}"
                                     / "llc_occupancy") for cache_id in (9, 10))
                self.assertEqual(result, expected)
                paths.append(set(result))
            self.assertFalse(paths[0] & paths[1] or paths[1] & paths[2] or paths[0] & paths[2])

    def test_l3_max_window_blocks_missing_selected_cache_monitor(self):
        """A fresh window cannot silently sample only part of the selected cache set."""
        placement = {**self._window_placement(), "cache_ids": (9, 10)}
        with TemporaryDirectory() as group:
            monitor = Path(group) / "mon_data" / "mon_L3_9"
            monitor.mkdir(parents=True)
            (monitor / "llc_occupancy").write_text("0\n", encoding="ascii")
            with self.assertRaises(mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_occupancy_files(group, placement)
        self.assertIn("expected cache IDs [9, 10]", str(context.exception))
        self.assertIn("found selected monitor IDs [9]", str(context.exception))

    def test_l3_max_unknown_body_typeerror_propagates_after_owned_cleanup(self):
        """Unexpected measurement faults remain loud after decorated group removal."""
        with self._live_environment() as (_root, events, _choose, window, _state):
            window.side_effect = TypeError("bad attribution")
            with self.assertRaisesRegex(TypeError, "bad attribution"):
                mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"] * 2)

    def test_l3_max_calibration_cleanup_failure_prevents_windows(self):
        """Calibration restoration failure prevents every measurement window."""
        with self._live_environment() as (_root, _events, _choose, window, _state):
            with self.mock.patch.object(mpam_resctrl, "_mpam_restore_percentage_schemata",
                                        side_effect=ValueError("calibration cleanup failed")):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "calibration cleanup failed"):
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            window.assert_not_called()

    def test_l3_max_live_effect_preserves_second_window_exception(self):
        """A second launch blockage or ordinary error stops later windows and PASS."""
        for error in (mpam_resctrl._MpamBlocked("membership drift"), ValueError("high median collapsed")):
            with self.subTest(error=type(error).__name__), self._live_environment() as (_r, events, _c, start, state):
                start.side_effect = partial(self._launch_with_second_fault, state, error)
                with self.mock.patch.object(mpam_resctrl, "_mpam_evaluate_l3_max_live_windows") as evaluate:
                    expected = (GLTCore.GLTTestWarn if isinstance(error, mpam_resctrl._MpamBlocked)
                                else GLTCore.GLTTestFail)
                    with self.assertRaisesRegex(expected, str(error)):
                        mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
                self.assertEqual(start.call_count, 2)
                self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"] * 3)
                evaluate.assert_not_called()

    @staticmethod
    def _launch_with_second_fault(state, error, argv, **kwargs):
        """Inject one real launch-boundary error after the first completed window."""
        if state.groups:
            raise error
        return state.launch(argv, **kwargs)


    def test_second_window_remove_failure_stops_later_groups(self):
        """Successful measurements cannot advance after window-2 group cleanup fails."""
        with self._live_environment() as (_root, events, _choose, window, _state):
            remove = mpam_resctrl.os.rmdir
            removals = []

            def removed(path):
                removals.append(path)
                if len(removals) == 3:
                    raise OSError("window two rmdir failed")
                remove(path)

            with self.mock.patch.object(mpam_resctrl.os, "rmdir", side_effect=removed):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "window two rmdir failed"):
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual(window.call_count, 2)
            self.assertEqual(len(removals), 3)
            self.assertEqual(sum(kind == "mkdir" for kind, _path in events), 3)

    def _second_window_blocked_case(self, cleanup_failure):
        """A second launch blockage retains its reason after all owned cleanup."""
        with self._live_environment() as (_root, events, _choose, start, state):
            start.side_effect = partial(self._launch_with_second_fault, state,
                                        mpam_resctrl._MpamBlocked("window two blocked"))
            remove = mpam_resctrl.os.rmdir
            removals = []
            with self.mock.patch.object(mpam_resctrl.os, "rmdir",
                    side_effect=partial(self._remove_second_window, remove, removals, cleanup_failure)):
                expected = GLTCore.GLTTestFail if cleanup_failure else GLTCore.GLTTestWarn
                with self.assertRaisesRegex(expected, "window two blocked") as caught:
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual(start.call_count, 2)
            self.assertEqual(len(removals), 3)
            self.assertEqual(sum(kind == "mkdir" for kind, _path in events), 3)
            if cleanup_failure:
                self.assertIn("window two cleanup failed", str(caught.exception))

    @staticmethod
    def _remove_second_window(remove, removals, fail, path):
        """Model a window-two group removal failure after calibration removal."""
        removals.append(path)
        if fail and len(removals) == 3:
            raise OSError("window two cleanup failed")
        remove(path)

    def test_second_window_blocked_stops_later_groups_and_keeps_cleanup_reason(self):
        """Blocked window-2 ends the loop; cleanup failure strengthens WARN to FAIL."""
        for cleanup_failure in (False, True):
            with self.subTest(cleanup_failure=cleanup_failure):
                self._second_window_blocked_case(cleanup_failure)

    def test_l3_max_live_effect_uses_six_unique_private_groups(self):
        """One calibration and six windows each restore and remove their owned group."""
        with self._live_environment() as (_root, events, _choose, _window, state):
            mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            groups = [path for kind, path in events if kind == "mkdir"]
            self.assertEqual(len(groups), 7)
            self.assertEqual(len(set(groups)), 7)
            self.assertEqual([path for kind, path in events if kind == "rmdir"], groups)
            self.assertEqual(state.groups, groups[1:])

    def test_l3_max_live_median_comparison_passes_with_clear_signal(self):
        """U04 comparison uses only per-window medians and dispersion."""
        mib = 1024 * 1024
        windows = (
            self._live_window(
                "low", 2 * mib,
                samples=(mib, 2 * mib, 3 * mib,
                         2 * mib, 2 * mib, 2 * mib,
                         2 * mib, 2 * mib, 2 * mib,
                         2 * mib)),
            self._live_window("high", 6 * mib),
            self._live_window("low", 2 * mib),
            self._live_window("high", 7 * mib),
            self._live_window("low", 3 * mib),
            self._live_window("high", 6 * mib),
        )

        low_center, high_center, separation, dispersion = (
            mpam_resctrl._mpam_evaluate_l3_max_live_windows(windows)
        )

        self.assertEqual(low_center, 2 * mib)
        self.assertEqual(high_center, 6 * mib)
        self.assertEqual(separation, 4 * mib)
        self.assertEqual(dispersion, 1 * mib)

    def test_l3_max_live_dispersion_uses_deviation_from_class_median(self):
        """Asymmetric medians use max distance from median, not full range."""
        mib = 1024 * 1024
        windows = (
            self._live_window("low", 0),
            self._live_window("high", 13 * mib),
            self._live_window("low", 5 * mib),
            self._live_window("high", 13 * mib),
            self._live_window("low", 10 * mib),
            self._live_window("high", 13 * mib),
        )

        low_center, high_center, separation, dispersion = (
            mpam_resctrl._mpam_evaluate_l3_max_live_windows(windows)
        )

        self.assertEqual(low_center, 5 * mib)
        self.assertEqual(high_center, 13 * mib)
        self.assertEqual(separation, 8 * mib)
        self.assertEqual(dispersion, 5 * mib)

    def test_l3_max_live_median_comparison_failure_is_fail(self):
        """Complete streams with no high-at-high signal are live failures."""
        mib = 1024 * 1024
        windows = (
            self._live_window("low", 4 * mib),
            self._live_window("high", 4 * mib),
            self._live_window("low", 4 * mib),
            self._live_window("high", 4 * mib),
            self._live_window("low", 4 * mib),
            self._live_window("high", 4 * mib),
        )

        with self.assertRaises(ValueError) as context:
            mpam_resctrl._mpam_evaluate_l3_max_live_windows(windows)

        self.assertIn("high-window median", str(context.exception))

    def test_l3_max_live_median_overlap_is_fail_even_when_centers_pass(self):
        """U04 requires every low median to stay below every high median."""
        mib = 1024 * 1024
        windows = (
            self._live_window("low", 2 * mib),
            self._live_window("high", 4 * mib),
            self._live_window("low", 5 * mib),
            self._live_window("high", 6 * mib),
            self._live_window("low", 2 * mib),
            self._live_window("high", 6 * mib),
        )

        with self.assertRaises(ValueError) as context:
            mpam_resctrl._mpam_evaluate_l3_max_live_windows(windows)

        self.assertIn("every low-window median", str(context.exception))

    def test_l3_max_live_incomplete_stream_is_blocked(self):
        """Incomplete sample streams are invalid observations, not failures."""
        windows = [
            self._live_window(label, 2 * 1024 * 1024)
            for label in mpam_resctrl.MPAM_L3_MAX_LIVE_PATTERN
        ]
        windows[0] = self._live_window("low", 1, samples=(1, 2, 3))

        with self.assertRaises(mpam_resctrl._MpamBlocked) as context:
            mpam_resctrl._mpam_evaluate_l3_max_live_windows(tuple(windows))

        self.assertIn("incomplete sample stream", str(context.exception))

    def test_l3_max_live_blocks_unexpected_label_sequence_before_u04(self):
        """Label drift is an invalid setup, not a live-effect failure."""
        mib = 1024 * 1024
        windows = (
            self._live_window("low", 2 * mib),
            self._live_window("low", 2 * mib),
            self._live_window("low", 2 * mib),
            self._live_window("high", 6 * mib),
            self._live_window("low", 2 * mib),
            self._live_window("high", 6 * mib),
        )

        with self.assertRaises(mpam_resctrl._MpamBlocked) as context:
            mpam_resctrl._mpam_evaluate_l3_max_live_windows(windows)

        self.assertIn("required low/high sequence", str(context.exception))


    def test_l3_max_live_allows_distinct_private_group_monitor_paths(self):
        """Fresh monitor paths bind to each group while window comparisons stay valid."""
        mib = 1024 * 1024
        windows = []
        paths = []
        with TemporaryDirectory() as mount:
            for index, label in enumerate(mpam_resctrl.MPAM_L3_MAX_LIVE_PATTERN):
                group = Path(mount) / f"glt_l3_max_{index}"
                monitor = group / "mon_data" / "mon_L3_09"
                monitor.mkdir(parents=True)
                (monitor / "llc_occupancy").write_text("0\n", encoding="ascii")
                bound = mpam_resctrl._mpam_l3_max_occupancy_files(
                    str(group), self._window_placement())
                self.assertEqual(bound, (str(monitor / "llc_occupancy"),))
                paths.extend(bound)
                windows.append(self._live_window(
                    label, 2 * mib if label == "low" else 6 * mib))
        self.assertEqual(len(set(paths)), 6)

        low_center, high_center, separation, dispersion = (
            mpam_resctrl._mpam_evaluate_l3_max_live_windows(windows)
        )

        self.assertEqual(low_center, 2 * mib)
        self.assertEqual(high_center, 6 * mib)
        self.assertEqual(separation, 4 * mib)
        self.assertEqual(dispersion, 0)

    def test_l3_max_live_window_launch_order_and_cleanup(self):
        """Task assignment precedes launch; restore/removal happen afterward."""
        state = _LaunchSequence(self)
        events = state.events

        with self.mock.patch.object(
                mpam_resctrl.os, 'mkdir',
                return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    return_value={'L3_MAX': {0: 50}}), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_occupancy_files",
                    return_value=("/fake/occ",)), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=state.sum_occupancy), \
                self.mock.patch.object(
                    mpam_resctrl.threading, "get_native_id",
                    return_value=777), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=("/original", None)), \
                self.mock.patch.object(
                    mpam_resctrl.glt_utils, "write_sysfs_value",
                    side_effect=state.write_task), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=state.read_group_tasks), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()
                ) as write_require, \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    side_effect=state.start), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"
                ) as membership, \
                self.mock.patch.object(
                    mpam_resctrl,
                    "_mpam_validate_l3_max_membership_snapshot"
                ) as sample_validator, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_stress_exit",
                    side_effect=self._complete_stress), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_group_exists",
                    return_value=False), \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       side_effect=state.monotonic), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_restore_percentage_schemata"
                ) as restore, \
                self.mock.patch.object(
                    mpam_resctrl.os, "killpg",
                    side_effect=ProcessLookupError) as killpg, \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            result = self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertEqual(result["label"], "high")
        self.assertEqual(result["percentage"], 100)
        self.assertEqual(result["median"], 5 * 1024 * 1024)
        assign_index = events.index(("write_task", "/fake/tasks", 777))
        start_index = events.index(("start", 2))
        restore_index = events.index(("write_task", "/original/tasks", 777))
        self.assertLess(assign_index, start_index)
        self.assertEqual(events[start_index - 1], ("deadline",))
        self.assertLess(start_index, restore_index)
        occupancy_events = [event for event in events if event[0] == "occupancy"]
        self.assertEqual(
            occupancy_events[:1],
            [("occupancy", False)],
        )
        self.assertNotIn(("occupancy", True), occupancy_events)
        self.assertEqual(
            [event for event in events if event[0] == "read_tasks"][:3],
            [
                ("read_tasks", False),
                ("read_tasks", False),
                ("read_tasks", True),
            ],
        )
        write_require.assert_called_once()

        self.assertEqual(
            membership.call_args_list,
            [
                self.mock.call("/fake", self.mock.ANY, self.mock.ANY, 45),
                self.mock.call(
                    "/fake", self.mock.ANY, self.mock.ANY, 45,
                    require_worker_footprint=True),
            ],
        )
        self.assertTrue(sample_validator.called)
        self.assertEqual(restore.call_count, 2)
        self.assertEqual(rmdir.call_count, 2)
        killpg.assert_not_called()

        messages = self._formatted_logs(self.mock_logger)
        for expected in (
                "L3_MAX window high prelaunch occupancy=0.",
                "L3_MAX window high workload PID=4321.",
                "L3_MAX window high sample=0 occupancy=5242880 deadline slack=45.000s.",
                "L3_MAX window high median=5242880.0 returncode=0.",
                "L3_MAX live window high stress-ng status: passed=4 skipped=0 failed=0 metrics_untrustworthy=0.",
                "L3_MAX live window high deadline slack 45.000s."):
            self.assertIn(expected, messages)

    def test_l3_max_live_empty_baseline_threshold_before_task_assignment(self):
        """The 1 MiB cleanliness bound applies to the empty private group."""
        cases = (
            ("below", mpam_resctrl.MPAM_L3_MAX_LIVE_BASELINE_LIMIT - 1, False),
            ("exact", mpam_resctrl.MPAM_L3_MAX_LIVE_BASELINE_LIMIT, True),
            ("above", mpam_resctrl.MPAM_L3_MAX_LIVE_BASELINE_LIMIT + 1, True),
        )

        for name, prelaunch_occupancy, should_block in cases:
            with self.subTest(name=name):
                _result, error, handles = self._run_l3_max_live_window_mocked(
                    occupancy_reads=[prelaunch_occupancy])
                if should_block:
                    self.assertIsInstance(error, GLTCore.GLTTestWarn)
                    self.assertIn(
                        "prelaunch aggregate llc_occupancy is not clean",
                        str(error))
                    handles["write_task"].assert_not_called()
                    handles["Popen"].assert_not_called()
                else:
                    self.assertIsNone(error)
                    self.assertEqual(
                        [call.args[2] for call in self.mock_logger.info.call_args_list
                         if call.args[0].startswith("L3_MAX window %s prelaunch")][-1],
                        prelaunch_occupancy)
                    handles["Popen"].assert_called_once_with(
                        mpam_resctrl._mpam_l3_max_stress_argv(2),
                        stdout=self.mock.ANY, stderr=subprocess.STDOUT, start_new_session=True)
                self.assertGreaterEqual(handles["occupancy"].call_count, 1)

    def test_l3_max_live_empty_baseline_requires_empty_task_snapshots(self):
        """Both prelaunch task snapshots must prove the group is empty."""
        cases = (
            ("first-nonempty", [{"999"}], "prelaunch task snapshot"),
            ("first-unreadable", [OSError("tasks denied")], "tasks denied"),
            ("second-nonempty", [set(), {"999"}], "prelaunch task snapshot"),
            ("second-unreadable", [set(), OSError("tasks denied")],
             "tasks denied"),
        )

        for name, task_reads, expected in cases:
            with self.subTest(name=name):
                _result, error, handles = self._run_l3_max_live_window_mocked(
                    task_reads=task_reads)
                self.assertIsInstance(error, GLTCore.GLTTestWarn)
                self.assertIn(expected, str(error))
                handles["write_task"].assert_not_called()
                handles["Popen"].assert_not_called()
                self.assertEqual(handles["restore"].call_count, 2)
                self.assertEqual(handles["rmdir"].call_count, 2)

    def test_l3_max_live_empty_baseline_occupancy_read_errors_block_before_task(self):
        """Unreadable empty-group occupancy blocks before task assignment."""
        cases = (
            ("os-error", OSError("occupancy denied")),
            ("value-error", ValueError("bad occupancy")),
        )

        for name, occupancy_error in cases:
            with self.subTest(name=name):
                _result, error, handles = self._run_l3_max_live_window_mocked(
                    occupancy_reads=[occupancy_error],
                    task_reads=[set()])
                self.assertIsInstance(error, GLTCore.GLTTestWarn)
                self.assertIn("Could not read prelaunch baseline",
                              str(error))
                handles["write_task"].assert_not_called()
                handles["Popen"].assert_not_called()
                self.assertEqual(handles["restore"].call_count, 2)
                self.assertEqual(handles["rmdir"].call_count, 2)

    def test_l3_max_live_supervisor_membership_must_be_exact_singleton(self):
        """Supervisor assignment must leave only the supervisor TID in the group."""
        cases = (
            ("missing", set(), "did not exactly match"),
            ("wrong", {"888"}, "did not exactly match"),
            ("extra", {"777", "888"}, "did not exactly match"),
        )

        for name, supervisor_tasks, expected in cases:
            with self.subTest(name=name):
                _result, error, handles = self._run_l3_max_live_window_mocked(
                    task_reads=[set(), set(), supervisor_tasks])
                self.assertIsInstance(error, GLTCore.GLTTestWarn)
                self.assertIn(expected, str(error))
                handles["write_task"].assert_any_call("/fake/tasks", 777)
                handles["Popen"].assert_not_called()
                self.assertEqual(handles["restore"].call_count, 2)
                self.assertEqual(handles["rmdir"].call_count, 2)

    def test_l3_max_live_window_blocks_before_task_mutation_without_group(self):
        """Supervisor membership must be uniquely known before mutation."""
        original_membership = mpam_resctrl._mpam_wait_for_l3_max_membership
        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        original_sleep = mpam_resctrl.time.sleep
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair",
                                       return_value=(10, 100)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy', return_value=0), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    side_effect=mpam_resctrl._MpamBlocked(
                        "not uniquely established")), \
                self.mock.patch.object(
                    mpam_resctrl.glt_utils, "write_sysfs_value"
                ) as write_task, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set()]), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen") as start:
            environment["_mpam_wait_for_l3_max_membership"].side_effect = original_membership
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            environment["sleep"].side_effect = original_sleep
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn("not uniquely established", str(context.exception))
        write_task.assert_not_called()
        start.assert_not_called()

    def test_l3_max_live_window_blocks_enumeration_failure_before_task_mutation(
            self):
        """Incomplete group enumeration prevents task writes and stress launch."""
        with TemporaryDirectory() as mount:
            Path(mount, "tasks").write_text("777\n", encoding="utf-8")

            original_membership = mpam_resctrl._mpam_wait_for_l3_max_membership
            original_samples = mpam_resctrl._mpam_collect_l3_max_samples
            original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
            original_supervisor_group = mpam_resctrl._mpam_resctrl_group_for_task
            original_sleep = mpam_resctrl.time.sleep
            with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair",
                                       return_value=(10, 100)), \
                    self.mock.patch.object(
                        resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                    self.mock.patch.object(
                            mpam_resctrl.os, "scandir",
                            side_effect=PermissionError("scan denied")), \
                    self.mock.patch.object(
                            resctrl_utils, 'sum_l3_occupancy',
                            side_effect=[0, 0]), \
                    self.mock.patch.object(
                            mpam_resctrl.glt_utils, "write_sysfs_value"
                        ) as write_task, \
                    self.mock.patch.object(
                            mpam_resctrl, "_mpam_read_group_task_ids",
                            side_effect=[set(), set()]), \
                    self.mock.patch.object(
                            mpam_resctrl, "_mpam_write_and_require_percentage",
                            return_value=self._ok_schemata_result()), \
                    self.mock.patch.object(
                            mpam_resctrl.subprocess, "Popen",
                            side_effect=OSError("stress launch reached")) as start, \
                    self.mock.patch.object(
                            mpam_resctrl, "_mpam_restore_percentage_schemata"
                        ) as restore, \
                    self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
                environment["_mpam_wait_for_l3_max_membership"].side_effect = original_membership
                environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
                environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
                environment["_mpam_resctrl_group_for_task"].side_effect = original_supervisor_group
                environment["sleep"].side_effect = original_sleep
                with self.assertRaises(
                        GLTCore.GLTTestWarn) as context:
                    self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn(
            "Could not enumerate resctrl control groups",
            str(context.exception))
        write_task.assert_not_called()
        start.assert_not_called()
        self.assertEqual(restore.call_count, 2)
        self.assertEqual(rmdir.call_count, 2)

    def test_l3_max_live_window_blocks_post_warmup_membership_drift(self):
        """Post-warmup sample validation blocks affinity/task drift."""
        class FakeProc:
            """Minimal process double for sample-drift handling."""

            returncode = None
            pid = 4321

            def poll(self):
                """Keep the fake workload alive until validation fails."""
                return self.returncode


            def wait(self, timeout=None):
                """Complete the fixture leader."""
                del timeout
                self.returncode = 0
                return self.returncode

        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=(resctrl_utils.RESCTRL_MOUNT_POINT, None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    return_value=FakeProc()), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"), \
                self.mock.patch.object(
                    mpam_resctrl,
                    "_mpam_validate_l3_max_membership_snapshot",
                    side_effect=mpam_resctrl._MpamBlocked(
                        "post-warmup affinity drift")), \
                self.mock.patch.object(
                    mpam_resctrl,
                    "_mpam_stop_l3_max_workload", return_value=[]) as stop_stress, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"):
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn("post-warmup affinity drift", str(context.exception))
        stop_stress.assert_called_once()

    def test_l3_max_live_restore_success_reproves_original_group(self):
        """Supervisor restore succeeds only after unique group readback."""
        group_for_task = self.mock.Mock(
            side_effect=[("/original", None), ("/original", None)])

        result, write_task, _sleep = self._run_l3_max_live_window_restore_case(
            group_for_task)

        self.assertEqual(result["label"], "high")
        self.assertEqual(
            group_for_task.call_args_list,
            [self.mock.call(777), self.mock.call(777)],
        )
        self.assertIn(
            self.mock.call("/original/tasks", 777),
            write_task.call_args_list,
        )

    def test_l3_max_live_restore_success_reproves_original_monitor_group(self):
        """Successful restore writes CTRL then original non-root MON group."""
        monitor_group = "/original/mon_groups/original_mon"
        group_for_task = self.mock.Mock(side_effect=[
            ("/original", monitor_group),
            ("/original", monitor_group),
        ])

        result, write_task, _sleep = self._run_l3_max_live_window_restore_case(
            group_for_task)

        self.assertEqual(result["label"], "high")
        self.assertEqual(
            write_task.call_args_list,
            [
                self.mock.call("/fake/tasks", 777),
                self.mock.call("/original/tasks", 777),
                self.mock.call(f"{monitor_group}/tasks", 777),
            ],
        )

    def _run_l3_max_live_first_restore_failure_case(
            self, group_for_task, write_task):
        """Run live window until the first post-launch supervisor restore."""
        class FakeProc:
            """Minimal process double for first-restore failure tests."""

            returncode = None
            pid = 4321

            def poll(self):
                """Keep the fake workload alive if membership is reached."""
                return self.returncode

            def wait(self, timeout=None):  # pylint: disable=unused-argument
                """Allow cleanup to reap the fake workload handle."""
                self.returncode = 0
                return 0

        with ExitStack() as stack:
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl.os, 'mkdir',
                return_value=None))
            stack.enter_context(self.mock.patch.object(
                resctrl_utils, 'read_schemata',
                return_value={'L3_MAX': {0: 50}}))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_probe_percentage_pair",
                return_value=(10, 100)))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_occupancy_files",
                return_value=("/fake/occ",)))
            stack.enter_context(self.mock.patch.object(
                resctrl_utils, 'sum_l3_occupancy', side_effect=[0, 0]))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl.threading, "get_native_id", return_value=777))
            group_probe = stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_resctrl_group_for_task",
                side_effect=group_for_task))
            write_task_mock = stack.enter_context(self.mock.patch.object(
                mpam_resctrl.glt_utils, "write_sysfs_value",
                side_effect=write_task))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_read_group_task_ids",
                side_effect=[set(), set(), {"777"}, set()]))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_write_and_require_percentage",
                return_value=self._ok_schemata_result()))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl.subprocess, "Popen",
                return_value=FakeProc()))
            membership = stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_wait_for_l3_max_membership"))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_collect_l3_max_samples",
                return_value=tuple([5 * 1024 * 1024] * 10)))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_wait_for_l3_max_stress_exit",
                side_effect=self._complete_stress))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_group_exists",
                return_value=False))
            stop_stress = stack.enter_context(self.mock.patch.object(
                mpam_resctrl,
                "_mpam_stop_l3_max_workload", return_value=[]))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl.time, "monotonic", return_value=0))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl.time, "sleep"))
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_restore_percentage_schemata"))
            rmdir = stack.enter_context(self.mock.patch.object(
                mpam_resctrl.os, "rmdir"))
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        return context, membership, group_probe, write_task_mock, stop_stress, rmdir

    def test_l3_max_live_first_restore_ctrl_write_failure_fails_immediately(
            self):
        """A first CTRL restore write failure fails before membership proof."""
        original_writes = []

        write_task = partial(_fixture_first_restore_ctrl_write_failure_fails_immediately_write_task, (original_writes,))

        context, membership, group_probe, write_task_mock, stop_stress, rmdir = (
            self._run_l3_max_live_first_restore_failure_case(
                [("/original", None), ("/original", None)], write_task))

        self.assertIn("Could not move supervisor thread 777 back",
                      str(context.exception))
        self.assertIn("root tasks busy", str(context.exception))
        membership.assert_not_called()
        stop_stress.assert_called_once()
        self.assertEqual(rmdir.call_count, 2)
        self.assertEqual(group_probe.call_count, 2)
        self.assertEqual(
            write_task_mock.call_args_list,
            [
                self.mock.call("/fake/tasks", 777),
                self.mock.call("/original/tasks", 777),
                self.mock.call("/original/tasks", 777),
            ],
        )

    def test_l3_max_live_first_restore_monitor_write_failure_fails_immediately(
            self):
        """A first MON restore write failure fails before membership proof."""
        monitor_group = "/original/mon_groups/original_mon"
        monitor_writes = []

        write_task = partial(
            _fixture_first_restore_monitor_write_failure_fails_immediately_write_task, (monitor_group, monitor_writes))

        context, membership, group_probe, write_task_mock, stop_stress, rmdir = (
            self._run_l3_max_live_first_restore_failure_case(
                [("/original", monitor_group),
                 ("/original", monitor_group)],
                write_task))

        self.assertIn(f"{monitor_group}/tasks", str(context.exception))
        self.assertIn("monitor tasks busy", str(context.exception))
        membership.assert_not_called()
        stop_stress.assert_called_once()
        self.assertEqual(rmdir.call_count, 2)
        self.assertEqual(group_probe.call_count, 2)
        self.assertEqual(
            write_task_mock.call_args_list,
            [
                self.mock.call("/fake/tasks", 777),
                self.mock.call("/original/tasks", 777),
                self.mock.call(f"{monitor_group}/tasks", 777),
                self.mock.call("/original/tasks", 777),
                self.mock.call(f"{monitor_group}/tasks", 777),
            ],
        )

    def test_l3_max_live_first_restore_readback_mismatch_fails_immediately(
            self):
        """A first restore readback mismatch fails before membership proof."""
        context, membership, group_probe, write_task_mock, stop_stress, rmdir = (
            self._run_l3_max_live_first_restore_failure_case(
                [("/original", None), ("/fake", None), ("/original", None)],
                None))

        self.assertIn("supervisor thread 777 restored to control /fake",
                      str(context.exception))
        self.assertIn("expected control /original", str(context.exception))
        membership.assert_not_called()
        stop_stress.assert_called_once()
        self.assertEqual(rmdir.call_count, 2)
        self.assertEqual(group_probe.call_count, 3)
        self.assertEqual(
            write_task_mock.call_args_list,
            [
                self.mock.call("/fake/tasks", 777),
                self.mock.call("/original/tasks", 777),
                self.mock.call("/original/tasks", 777),
            ],
        )

    def test_l3_max_live_first_restore_retry_success_remains_failure(self):
        """Cleanup retry success cannot turn the first restore failure into pass."""
        original_writes = []

        write_task = partial(_fixture_first_restore_retry_success_remains_failure_write_task, (original_writes,))

        context, membership, _group_probe, _write_task, stop_stress, rmdir = (
            self._run_l3_max_live_first_restore_failure_case(
                [("/original", None), ("/original", None)], write_task))

        self.assertIn("first restore failed", str(context.exception))
        membership.assert_not_called()
        stop_stress.assert_called_once()
        self.assertEqual(rmdir.call_count, 2)
        self.assertEqual(original_writes, ["/original/tasks", "/original/tasks"])

    def test_l3_max_live_restore_mismatch_fails_cleanup(self):
        """Supervisor restore must return to the exact captured group."""
        group_for_task = self.mock.Mock(
            side_effect=[
                ("/original", None), ("/fake", None), ("/fake", None)])

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_l3_max_live_window_restore_case(group_for_task)

        self.assertIn("supervisor thread 777 restored to control /fake",
                      str(context.exception))
        self.assertIn("expected control /original", str(context.exception))
        self.assertEqual(group_for_task.call_count, 3)

    def test_l3_max_live_restore_monitor_mismatch_fails_cleanup(self):
        """Supervisor restore must reprove the original MON subgroup too."""
        monitor_group = "/original/mon_groups/original_mon"
        cases = (
            ("missing", None),
            ("wrong", "/original/mon_groups/wrong_mon"),
        )

        for name, restored_monitor_group in cases:
            with self.subTest(name=name):
                group_for_task = self.mock.Mock(side_effect=[
                    ("/original", monitor_group),
                    ("/original", restored_monitor_group),
                    ("/original", restored_monitor_group),
                ])

                with self.assertRaises(
                        GLTCore.GLTTestFail) as context:
                    self._run_l3_max_live_window_restore_case(group_for_task)

                self.assertIn(
                    "restored to control /original, monitor "
                    f"{restored_monitor_group}",
                    str(context.exception))
                self.assertIn(
                    f"expected control /original, monitor {monitor_group}",
                    str(context.exception))
                self.assertEqual(group_for_task.call_count, 3)

    def test_l3_max_live_restore_monitor_write_error_fails_cleanup(self):
        """A failed MON restore write prevents cleanup success."""
        monitor_group = "/original/mon_groups/original_mon"
        group_for_task = self.mock.Mock(
            side_effect=[("/original", monitor_group)])

        write_task = partial(_fixture_restore_monitor_write_error_fails_cleanup_write_task, (monitor_group,))

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            with self.mock.patch.object(
                    self, "_l3_max_write_task_side_effect", write_task,
                    create=True):
                self._run_l3_max_live_window_restore_case(group_for_task)

        self.assertIn(
            f"Could not move supervisor thread 777 back to "
            f"{monitor_group}/tasks",
            str(context.exception))
        self.assertIn("monitor group missing", str(context.exception))
        group_for_task.assert_called_once_with(777)

    def test_l3_max_live_restore_read_error_fails_cleanup(self):
        """Supervisor restore readback errors remain cleanup failures."""
        group_for_task = self.mock.Mock(side_effect=[
            ("/original", None),
            mpam_resctrl._MpamBlocked("tasks read denied"),
            mpam_resctrl._MpamBlocked("tasks read denied"),
        ])

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_l3_max_live_window_restore_case(group_for_task)

        self.assertIn(
            "Could not prove supervisor thread 777 returned to control "
            "/original, monitor None",
            str(context.exception))
        self.assertIn("tasks read denied", str(context.exception))
        self.assertEqual(group_for_task.call_count, 3)

    def test_l3_max_live_expired_exit_still_quiesces_then_blocks(self):
        """F1: terminated workload gets fixed quiesce before late BLOCKED."""
        group_for_task = self.mock.Mock(
            side_effect=[("/original", None), ("/original", None)])
        clock = self._FakeClock()

        stress_wait = partial(_fixture_expired_exit_still_quiesces_then_blocks_stress_wait, (clock, self))

        with self.mock.patch.object(
                self, "_l3_max_stress_wait_side_effect", stress_wait,
                create=True):
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_l3_max_live_window_restore_case(
                    group_for_task,
                    monotonic_side_effect=clock.monotonic,
                    sleep_mock=clock.sleep)

        self.assertIn("stress-ng exit", str(context.exception))
        self.assertEqual(
            clock.sleeps,
            [
                mpam_resctrl.MPAM_L3_MAX_LIVE_WARMUP_SEC,
                mpam_resctrl.MPAM_L3_MAX_LIVE_QUIESCE_SEC,
            ],
        )

    def test_l3_max_live_quiesce_overrun_blocks_without_completion(self):
        """F1: a body-valid window that overruns during quiesce is blocked."""
        group_for_task = self.mock.Mock(
            side_effect=[("/original", None), ("/original", None)])
        clock = self._FakeClock()

        stress_wait = partial(_fixture_quiesce_overrun_blocks_without_completion_stress_wait, (clock, self))

        with self.mock.patch.object(
                self, "_l3_max_stress_wait_side_effect", stress_wait,
                create=True):
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_l3_max_live_window_restore_case(
                    group_for_task,
                    monotonic_side_effect=clock.monotonic,
                    sleep_mock=clock.sleep)

        self.assertIn("after cleanup quiesce", str(context.exception))
        self.assertEqual(
            clock.sleeps,
            [
                mpam_resctrl.MPAM_L3_MAX_LIVE_WARMUP_SEC,
                mpam_resctrl.MPAM_L3_MAX_LIVE_QUIESCE_SEC,
            ],
        )

    def test_l3_max_live_residual_tasks_error_still_fails_after_quiesce(self):
        """F1: genuine cleanup errors still fail after fixed quiesce."""
        group_for_task = self.mock.Mock(
            side_effect=[("/original", None), ("/original", None)])
        clock = self._FakeClock()

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_l3_max_live_window_restore_case(
                group_for_task,
                read_group_task_ids=[
                    set(), set(), {"777"}, OSError("tasks unreadable")],
                monotonic_side_effect=clock.monotonic,
                sleep_mock=clock.sleep)

        self.assertIn("Could not verify residual task state",
                      str(context.exception))
        self.assertIn("tasks unreadable", str(context.exception))
        self.assertEqual(
            clock.sleeps,
            [
                mpam_resctrl.MPAM_L3_MAX_LIVE_WARMUP_SEC,
                mpam_resctrl.MPAM_L3_MAX_LIVE_QUIESCE_SEC,
            ],
        )

    def test_l3_max_stop_after_wait_terminates_lingering_group(self):
        """An exited leader leaves stragglers subject to group absence proof."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = 0
        with self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=[None, None, ProcessLookupError]) as killpg:
            errors = []
            mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)
            self.assertEqual(errors, [])
        self.assertEqual(killpg.call_args_list, [self.mock.call(4321, 0), self.mock.call(4321,
            mpam_resctrl.signal.SIGKILL), self.mock.call(4321, 0)])
        proc.poll.assert_called_once_with()
        proc.wait.assert_not_called()


    def test_l3_max_live_nonzero_exit_still_attempts_group_cleanup(self):
        """Nonzero stress-ng exit keeps process-group cleanup evidence alive."""
        class FakeProc:
            """Minimal process double for nonzero-exit cleanup."""

            returncode = None
            pid = 4321

            def poll(self):
                """Keep the fake workload alive for sample collection."""
                return self.returncode


            def wait(self, timeout=None):
                """Complete the fixture leader."""
                del timeout
                self.returncode = 0
                return self.returncode

        proc = FakeProc()

        with self._window_environment() as (_environment, _proc, _clock), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=(resctrl_utils.RESCTRL_MOUNT_POINT, None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    return_value=proc), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_collect_l3_max_samples",
                    return_value=tuple([1] * 10)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_stress_exit",
                    return_value=self._stress_result(7, "stress-ng failed")), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_group_exists",
                    return_value=False), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stop_l3_max_workload"
                ) as stop_after_wait, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"):
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn("stress-ng exited with status 7",
                      str(context.exception))
        self.assertIn("stress-ng failed", str(context.exception))
        stop_after_wait.assert_called_once_with(
            proc, self.mock.ANY, self.mock.ANY, self.mock.ANY)
        self.assertIsInstance(stop_after_wait.call_args.args[1],
                              ValueError)


    def test_l3_max_live_late_nonzero_retains_definitive_failure(self):
        """Returned failure evidence takes precedence over a late observation."""
        clock = self._FakeClock()

        late_wait = partial(_fixture_late_nonzero_retains_definitive_failure_late_wait, (self, clock))

        group_for_task = self.mock.Mock(
            return_value=(resctrl_utils.RESCTRL_MOUNT_POINT, None))
        with self.mock.patch.object(
                self, "_l3_max_stress_wait_side_effect", late_wait, create=True), \
                self.assertRaises((GLTCore.GLTTestFail, GLTCore.GLTTestWarn)) as context:
            self._run_l3_max_live_window_restore_case(
                group_for_task, monotonic_side_effect=clock.monotonic)

        self.assertIsInstance(context.exception, GLTCore.GLTTestFail)
        self.assertIn("stress-ng exited with status 7", str(context.exception))
        self.assertIn("late failure output", str(context.exception))

    def test_l3_max_live_late_failed_summary_retains_definitive_failure(self):
        """Returned failure evidence takes precedence over a late observation."""
        clock = self._FakeClock()

        late_wait = partial(_fixture_late_failed_summary_retains_definitive_failure_late_wait, (self, clock))

        group_for_task = self.mock.Mock(
            return_value=(resctrl_utils.RESCTRL_MOUNT_POINT, None))
        with self.mock.patch.object(
                self, "_l3_max_stress_wait_side_effect", late_wait, create=True), \
                self.assertRaises((GLTCore.GLTTestFail, GLTCore.GLTTestWarn)) as context:
            self._run_l3_max_live_window_restore_case(
                group_for_task, monotonic_side_effect=clock.monotonic)

        self.assertIsInstance(context.exception, GLTCore.GLTTestFail)
        self.assertIn("non-zero bad counts", str(context.exception))

    def test_l3_max_live_late_success_after_wait_is_blocked(self):
        """A clean stress-ng exit after the outer deadline is invalid."""
        class FakeProc:
            """Minimal process double for late-success handling."""

            returncode = None
            pid = 4321


            def poll(self):
                """Return the modelled leader status."""
                return self.returncode


            def wait(self, timeout=None):
                """Complete the fixture leader."""
                del timeout
                self.returncode = 0
                return self.returncode

        proc = FakeProc()

        clock = self._FakeClock()

        late_wait = partial(_fixture_late_success_after_wait_is_blocked_late_wait, (self, clock))

        original_sleep = mpam_resctrl.time.sleep
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=(resctrl_utils.RESCTRL_MOUNT_POINT, None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    return_value=proc), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_sleep_l3_max_with_deadline"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_collect_l3_max_samples",
                    return_value=tuple([1] * 10)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_stress_exit",
                    side_effect=late_wait), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_group_exists",
                    return_value=False), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stop_l3_max_workload"
                ) as stop_after_wait, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       side_effect=clock.monotonic):
            environment["sleep"].side_effect = original_sleep
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn("stress-ng exit", str(context.exception))
        stop_after_wait.assert_called_once_with(
            proc, self.mock.ANY, self.mock.ANY, self.mock.ANY)
        self.assertIsInstance(stop_after_wait.call_args.args[1],
                              mpam_resctrl._MpamBlocked)


    def test_l3_max_live_clean_exit_with_lingering_group_is_failed(self):
        """A clean leader exit with lingering workers is a live failure."""
        class FakeProc:
            """Minimal process double for lingering-group handling."""

            returncode = None
            pid = 4321

            def poll(self):
                """Keep the fake workload alive for sample collection."""
                return self.returncode


            def wait(self, timeout=None):
                """Complete the fixture leader."""
                del timeout
                self.returncode = 0
                return self.returncode

        proc = FakeProc()
        group_alive = {"value": True}

        signal_group = partial(_fixture_clean_exit_with_lingering_group_is_failed_signal_group, (group_alive,))


        with self._window_environment() as (_environment, _proc, _clock), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=(resctrl_utils.RESCTRL_MOUNT_POINT, None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}, set()]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    return_value=proc) as launch, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_collect_l3_max_samples",
                    return_value=tuple([1] * 10)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_stress_exit",
                    return_value=self._stress_result(
                        0, "stress-ng completed")), \
                self.mock.patch.object(
                    mpam_resctrl.os, "killpg", side_effect=signal_group) as killpg, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"):
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn("Process group 4321 still existed",
                      str(context.exception))
        self.assertIn("stress-ng completed", str(context.exception))
        killpg.assert_any_call(4321, mpam_resctrl.signal.SIGKILL)
        self.assertFalse(group_alive["value"])
        self.assertTrue(launch.call_args.kwargs["stdout"].closed)


    def test_l3_max_live_launch_blocked_keeps_restore_cleanup_failure(self):
        """Launch errors are not masked by supervisor restore cleanup errors."""
        write_task = _fixture_launch_blocked_keeps_restore_cleanup_failure_write_task

        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        original_sleep = mpam_resctrl.time.sleep
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=("/original", None)), \
                self.mock.patch.object(
                    mpam_resctrl.glt_utils, "write_sysfs_value",
                    side_effect=write_task), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    side_effect=OSError("no stress-ng")), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"
                ) as membership, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       return_value=0), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            environment["sleep"].side_effect = original_sleep
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn("Could not launch stress-ng workload: no stress-ng",
                      str(context.exception))
        self.assertIn("Could not move supervisor thread 777 back",
                      str(context.exception))
        membership.assert_not_called()
        self.assertEqual(rmdir.call_count, 2)

    def test_l3_max_live_launch_blocked_remains_blocked_with_clean_cleanup(self):
        """Launch CONTRACT_BLOCKED stays blocked when cleanup succeeds."""
        original_samples = mpam_resctrl._mpam_collect_l3_max_samples
        original_wait = mpam_resctrl._mpam_wait_for_l3_max_stress_exit
        original_sleep = mpam_resctrl.time.sleep
        with self._window_environment() as (environment, _proc, _clock), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    side_effect=[0, 0]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_resctrl_group_for_task",
                    return_value=(resctrl_utils.RESCTRL_MOUNT_POINT, None)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[set(), set(), {"777"}]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()), \
                self.mock.patch.object(
                    mpam_resctrl.subprocess, "Popen",
                    side_effect=OSError("no stress-ng")), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_wait_for_l3_max_membership"
                ) as membership, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       return_value=0), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            environment["_mpam_collect_l3_max_samples"].side_effect = original_samples
            environment["_mpam_wait_for_l3_max_stress_exit"].side_effect = original_wait
            environment["sleep"].side_effect = original_sleep
            with self.assertRaises(
                    GLTCore.GLTTestWarn) as context:
                self._run_decorated_window("high", self._window_placement(), "/fake")

        self.assertIn("Could not launch stress-ng workload: no stress-ng",
                      str(context.exception))
        membership.assert_not_called()
        self.assertEqual(rmdir.call_count, 2)

    def test_l3_max_live_window_blocks_real_attribution_missing_per_cpu_cache(self):
        """Incomplete CPU/cache evidence warns after actual calibration restoration."""
        attribution = mpam_resctrl._mpam_l3_max_attribution
        with self._live_environment() as (_root, events, choose, windows, _state):
            choose.side_effect = attribution
            with self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_discovery_names",
                                        side_effect=lambda path: ["node2"] if path == "/sys/devices/system/node"
                                        else ["mon_L3_9"]), \
                    self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch.object(mpam_resctrl, "_cpus_of_node", return_value=(4, 5)), \
                    self.mock.patch.object(mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                                           side_effect=lambda cpu: {9} if cpu == 4 else set()), \
                    self.mock.patch.object(mpam_resctrl.subprocess, "Popen") as start:
                with self.assertRaises(GLTCore.GLTTestWarn) as caught:
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertIn("NUMA node 2", str(caught.exception))
            self.assertIn("CPU 5", str(caught.exception))
            self.assertIn("no L3 cache IDs", str(caught.exception))
            windows.assert_not_called()
            start.assert_not_called()
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"])

    def test_l3_max_live_unexpected_error_keeps_restore_and_remove_failures(self):
        """Unexpected restoration faults propagate after guaranteed owned removal."""
        with self._live_environment() as (_root, events, _choose, window, _state):
            window.side_effect = TypeError("bad attribution")
            with self.mock.patch.object(mpam_resctrl, "_mpam_restore_percentage_schemata",
                                        side_effect=[None, None, TypeError("cleanup restore")]):
                with self.assertRaisesRegex(TypeError, "cleanup restore"):
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"] * 2)

    def test_l3_max_live_selection_probes_upper_and_non_vacuous_pair(self):
        """Real parsed integer domains preserve extremal non-vacuous calibration writes."""
        write_percentage = mpam_resctrl._mpam_write_and_require_percentage
        restore_percentage = mpam_resctrl._mpam_restore_percentage_schemata
        with self._live_environment(initial_line="L3_MAX:1=50;2=50") as (root, events, _choose, _windows, state):
            with self.mock.patch.object(mpam_resctrl, "_mpam_read_percentage_info", return_value=(10, 90)), \
                    self.mock.patch.object(mpam_resctrl, "_mpam_write_and_require_percentage",
                                           wraps=write_percentage) as write, \
                    self.mock.patch.object(mpam_resctrl, "_mpam_restore_percentage_schemata",
                                           wraps=restore_percentage) as restore:
                mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual([call.args[3] for call in write.call_args_list[:3]], [100, 10, 100])
            self.assertEqual(state.percentages, [10, 100] * 3)
            self.assertEqual(write.call_count, 9)
            self.assertEqual(restore.call_count, 8)
            self.assertTrue(all(call.args[2] == (1, 2) for call in write.call_args_list))
            self.assertTrue(all(call.args[3] == {1: 50, 2: 50} for call in restore.call_args_list))
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"] * 7)
            self.assertEqual(list(root.glob("glt_*")), [])

    def test_l3_max_live_selection_uses_gran5_min_max_changing_values(self):
        """Real parsed integer domains preserve extremal non-vacuous calibration writes."""
        write_percentage = mpam_resctrl._mpam_write_and_require_percentage
        restore_percentage = mpam_resctrl._mpam_restore_percentage_schemata
        with self._live_environment(initial_line="L3_MAX:1=12;2=12") as (root, events, _choose, _windows, state):
            with self.mock.patch.object(mpam_resctrl, "_mpam_read_percentage_info", return_value=(5, 5)), \
                    self.mock.patch.object(mpam_resctrl, "_mpam_write_and_require_percentage",
                                           wraps=write_percentage) as write, \
                    self.mock.patch.object(mpam_resctrl, "_mpam_restore_percentage_schemata",
                                           wraps=restore_percentage) as restore:
                mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual([call.args[3] for call in write.call_args_list[:3]], [100, 5, 100])
            self.assertEqual(state.percentages, [5, 100] * 3)
            self.assertEqual(write.call_count, 9)
            self.assertEqual(restore.call_count, 8)
            self.assertTrue(all(call.args[2] == (1, 2) for call in write.call_args_list))
            self.assertTrue(all(call.args[3] == {1: 12, 2: 12} for call in restore.call_args_list))
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"] * 7)
            self.assertEqual(list(root.glob("glt_*")), [])

    def test_l3_max_live_selection_blocks_vacuous_or_too_close_pair(self):
        """A changing pair below the 10-point span warns after calibration cleanup."""
        with self._live_environment(initial_line="L3_MAX:1=40;2=40") as (_root, events, _c, windows, _state):
            with self.mock.patch.object(mpam_resctrl, "_mpam_read_percentage_info", return_value=(95, 5)):
                with self.assertRaisesRegex(GLTCore.GLTTestWarn, "at least 10 percentage points"):
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            windows.assert_not_called()
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"])

    def test_l3_max_live_selection_readback_mismatch_is_strict(self):
        """A kernel-style clamped write fails exact calibration readback before launch."""
        original_write = mpam_resctrl._mpam_write_resource_schemata
        writes = []

        def clamped(path, resource, domains, values):
            line = original_write(path, resource, domains, values)
            writes.append(line)
            if len(writes) == 1:
                Path(path).write_text("L3_MAX:1=99;2=99\n", encoding="ascii")
            return line

        with self._live_environment() as (_root, events, _c, windows, _state):
            with self.mock.patch.object(mpam_resctrl, "_mpam_write_resource_schemata", side_effect=clamped):
                with self.assertRaises(GLTCore.GLTTestFail) as caught:
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertIn("wrote 'L3_MAX:1=100;2=100'", str(caught.exception))
            self.assertIn("read back 'L3_MAX:1=99;2=99'", str(caught.exception))
            self.assertIn("exact per-domain readback is required", str(caught.exception))
            self.assertEqual(writes[-1], "L3_MAX:1=50;2=50")
            windows.assert_not_called()
            self.assertEqual([kind for kind, _path in events], ["mkdir", "rmdir"])

    def test_l3_max_attribution_requires_complete_selected_node_monitors(self):
        """Selected node cache IDs must exactly match selected monitor IDs."""
        discovery = {'/sys/devices/system/node': ['node2'], '/fake/mon_data': ['mon_L3_9', 'mon_L3_10']}

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch.object(
                    mpam_resctrl, "_cpus_of_node",
                    return_value=(4, 5)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                    side_effect=lambda cpu: {9} if cpu == 4 else {10}):
            attribution = mpam_resctrl._mpam_l3_max_attribution("/fake")
            occupancy_files = mpam_resctrl._mpam_l3_max_occupancy_files("/fake", attribution)

        self.assertEqual(attribution["node"], 2)
        self.assertEqual(attribution["cache_ids"], (9, 10))
        self.assertEqual(
            occupancy_files,
            (
                "/fake/mon_data/mon_L3_9/llc_occupancy",
                "/fake/mon_data/mon_L3_10/llc_occupancy",
            ),
        )

    def test_l3_max_attribution_blocks_each_selected_cpu_without_l3_cache_id(self):
        """Every selected online CPU must contribute L3 cache evidence."""
        discovery = {'/sys/devices/system/node': ['node2'], '/fake/mon_data': ['mon_L3_9']}

        cases = (
            ("missing-first", 4, {4: set(), 5: {9}}),
            ("missing-later", 5, {4: {9}, 5: set()}),
        )

        for name, missing_cpu, cache_ids_by_cpu in cases:
            with self.subTest(name=name):
                with self.mock.patch.object(
                        mpam_resctrl, "_mpam_l3_max_discovery_names",
                        side_effect=lambda path: discovery.get(path, [])), \
                        self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                        self.mock.patch.object(
                            mpam_resctrl, "_cpus_of_node",
                            return_value=(4, 5)), \
                        self.mock.patch.object(
                            mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                            side_effect=lambda cpu, cache_ids=cache_ids_by_cpu: (
                                cache_ids[cpu])):
                    with self.assertRaises(
                            mpam_resctrl._MpamBlocked) as context:
                        mpam_resctrl._mpam_l3_max_attribution("/fake")

                self.assertIn("NUMA node 2", str(context.exception))
                self.assertIn(f"CPU {missing_cpu}", str(context.exception))
                self.assertIn("no L3 cache IDs", str(context.exception))

    def test_l3_max_cache_id_discovery_blocks_malformed_cache_level(self):
        """Malformed cache topology cannot silently produce partial L3 IDs."""
        files = {
            "/sys/devices/system/cpu/cpu4/cache/index0/level": "3\n",
            "/sys/devices/system/cpu/cpu4/cache/index0/id": "9\n",
            "/sys/devices/system/cpu/cpu4/cache/index1/level": "not-a-level\n",
        }

        def fake_open(path, *_args, **_kwargs):
            return self.mock.mock_open(read_data=files[path])()

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                return_value=["index0", "index1"]), \
                self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_cache_ids_for_cpu(4)

        self.assertIn("Malformed cache level", str(context.exception))

    def test_l3_max_cache_id_discovery_blocks_unreadable_l3_id(self):
        """Unreadable L3 cache ID is invalid topology, not a skipped index."""
        def fake_open(path, *_args, **_kwargs):
            if path == "/sys/devices/system/cpu/cpu4/cache/index0/id":
                raise OSError("id denied")
            return self.mock.mock_open(read_data="3\n")()

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                return_value=["index0"]), \
                self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_cache_ids_for_cpu(4)

        self.assertIn("Could not read L3 cache ID", str(context.exception))

    def test_l3_max_attribution_blocks_malformed_node_cpulist(self):
        """Malformed NUMA cpulists are invalid live attribution evidence."""
        discovery = {'/sys/devices/system/node': ['node2'], '/fake/mon_data': ['mon_L3_9']}

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch("builtins.open", self.mock.mock_open(
                    read_data="4,bad,5\n")):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_attribution("/fake")

        self.assertIn("Malformed NUMA node 2 CPU list",
                      str(context.exception))
        self.assertIn("'4,bad,5'", str(context.exception))

    def test_l3_max_attribution_filters_offline_cpus(self):
        """Attribution selects the first node with online CPUs only."""
        seen_cpus = []

        discovery = {'/sys/devices/system/node': ['node0', 'node1'], '/fake/mon_data': ['mon_L3_9']}

        def fake_open(path, *_args, **_kwargs):
            data = {
                "/sys/devices/system/node/node0/cpulist": "0-3\n",
                "/sys/devices/system/node/node1/cpulist": "4-5\n",
                "/sys/devices/system/cpu/online": "4-5\n",
            }[path]
            return self.mock.mock_open(read_data=data)()

        def cache_ids_for_cpu(cpu):
            seen_cpus.append(cpu)
            return {9}

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch("builtins.open", side_effect=fake_open), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                    side_effect=cache_ids_for_cpu):
            attribution = mpam_resctrl._mpam_l3_max_attribution("/fake")

        self.assertEqual(attribution["node"], 1)
        self.assertEqual(attribution["cpus"], (4, 5))
        self.assertEqual(seen_cpus, [4, 5])

    def test_l3_max_attribution_sorts_node_ids_numerically(self):
        """NUMA node ordering is numeric, so node2 is selected before node10."""
        discovery = {'/sys/devices/system/node': ['node10', 'node2'], '/fake/mon_data': ['mon_L3_2', 'mon_L3_10']}

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch.object(
                    mpam_resctrl, "_cpus_of_node",
                    side_effect=lambda node, **_kwargs: (node,)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                    side_effect=lambda cpu: {cpu}):
            attribution = mpam_resctrl._mpam_l3_max_attribution("/fake")

        self.assertEqual(attribution["node"], 2)
        self.assertEqual(attribution["cpus"], (2,))
        self.assertEqual(attribution["cache_ids"], (2,))

    def test_l3_max_attribution_blocks_malformed_online_cpulist(self):
        """Malformed online CPU masks fail closed before live attribution."""
        discovery = {'/sys/devices/system/node': ['node2'], '/fake/mon_data': ['mon_L3_9']}

        def fake_open(path, *_args, **_kwargs):
            data = {
                "/sys/devices/system/node/node2/cpulist": "4-5\n",
                "/sys/devices/system/cpu/online": "4,bad,5\n",
            }[path]
            return self.mock.mock_open(read_data=data)()

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_attribution("/fake")

        self.assertIn("Malformed online CPU list", str(context.exception))
        self.assertIn("'4,bad,5'", str(context.exception))

    def test_l3_max_attribution_blocks_missing_selected_node_monitors(self):
        """Missing selected-node monitor coverage is an invalid observation."""
        discovery = {'/sys/devices/system/node': ['node2'], '/fake/mon_data': ['mon_L3_9']}

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch.object(
                    mpam_resctrl, "_cpus_of_node",
                    return_value=(4, 5)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                    side_effect=lambda cpu: {9} if cpu == 4 else {10}):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_attribution("/fake")

        self.assertIn("do not exactly match", str(context.exception))

    def test_l3_max_attribution_blocks_first_online_node_mismatch(self):
        """A later complete node cannot hide the first online node mismatch."""
        discovery = {'/sys/devices/system/node': ['node0', 'node1'], '/fake/mon_data': ['mon_L3_9', 'mon_L3_11']}

        def cache_ids_for_cpu(cpu):
            if cpu == 0:
                return {9}
            if cpu == 1:
                return {10}
            if cpu == 2:
                return {11}
            return set()

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch.object(
                    mpam_resctrl, "_cpus_of_node",
                    side_effect=lambda node, **_kwargs: (0, 1) if node == 0 else (2,)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                    side_effect=cache_ids_for_cpu) as cache_ids:
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_attribution("/fake")

        self.assertIn("selected NUMA node 0", str(context.exception))
        self.assertIn("expected cache IDs [9, 10]", str(context.exception))
        self.assertIn("found selected monitor IDs [9]", str(context.exception))
        self.assertEqual(
            cache_ids.call_args_list,
            [self.mock.call(0), self.mock.call(1)],
        )

    def test_l3_max_attribution_ignores_other_node_monitor_ids(self):
        """Multi-node monitors do not block selected-node occupancy files."""
        discovery = {
            '/sys/devices/system/node': ['node2', 'node3'],
            '/fake/mon_data': ['mon_L3_9', 'mon_L3_10', 'mon_L3_11'],
        }

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_discovery_names",
                side_effect=lambda path: discovery.get(path, [])), \
                self.mock.patch.object(mpam_resctrl.os, "lstat"), \
                self.mock.patch.object(
                    mpam_resctrl, "_cpus_of_node",
                    side_effect=lambda node, **_kwargs: (4, 5) if node == 2 else (6,)), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_cache_ids_for_cpu",
                    side_effect=lambda cpu: {9} if cpu == 4
                    else ({10} if cpu == 5 else {11})):
            attribution = mpam_resctrl._mpam_l3_max_attribution("/fake")
            occupancy_files = mpam_resctrl._mpam_l3_max_occupancy_files("/fake", attribution)

        self.assertEqual(attribution["node"], 2)
        self.assertEqual(attribution["cache_ids"], (9, 10))
        self.assertEqual(
            occupancy_files,
            (
                "/fake/mon_data/mon_L3_9/llc_occupancy",
                "/fake/mon_data/mon_L3_10/llc_occupancy",
            ),
        )

    @staticmethod
    def _l3_max_process_tree(process_ids, task_ids, child_edges):
        """Build the same aggregate membership and explicit direct edges."""
        tids_by_pid = {str(pid): {str(pid)} for pid in process_ids}
        tids_by_pid[min(tids_by_pid)].update(set(task_ids) - set(process_ids))
        children_by_pid = {str(pid): set() for pid in process_ids}
        for parent, child in child_edges:
            children_by_pid[str(parent)].add(str(child))
        return tids_by_pid, children_by_pid


    @staticmethod
    def _l3_max_parent_leaf_edges():
        """Return the observed stress-ng leader -> parent -> leaf topology."""
        return (
            ("100", "200"), ("100", "201"), ("100", "202"), ("100", "203"),
            ("200", "300"), ("201", "301"), ("202", "302"), ("203", "303"),
        )


    def test_l3_max_membership_blocks_unreadable_leaf_children_before_placement(
            self):
        """Unreadable leaf children cannot pass via matching tasks alone."""
        attribution = self._window_placement()
        child_map = {
            "100": ("200", "201", "202", "203"),
            "200": ("204",),
            "201": ("205",),
            "202": ("206",),
            "203": ("207",),
            "204": (),
            "205": (),
            "206": (),
            "207": (),
        }
        workload_ids = set(child_map)
        fake_listdir, fake_open = proc_task_fixture(
            child_map, children_error_tids=("204",))

        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(
                    mpam_resctrl.os, "listdir", side_effect=fake_listdir), \
                self.mock.patch("builtins.open", side_effect=fake_open), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    return_value=workload_ids), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "process_comm",
                    return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_validate_l3_max_placement"
                ) as validate:
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_membership_snapshot(
                    "/fake", 100, attribution, 30)

        self.assertIn("/proc/204/task/204/children", str(context.exception))
        self.assertIn("cannot enumerate children", str(context.exception))
        validate.assert_not_called()

    def test_l3_max_membership_requires_workers_all_ids_and_placement(self):
        """Membership proves all IDs but footprints only exact worker leaves."""
        attribution = self._window_placement()
        parents = {"200", "201", "202", "203"}
        workers = {"300", "301", "302", "303"}
        workload_ids = {"100"} | parents | workers
        process_tree = self._l3_max_process_tree(
            workload_ids, workload_ids, self._l3_max_parent_leaf_edges())

        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    return_value=workload_ids), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stable_workload_process_tree",
                    return_value=process_tree), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "process_comm",
                    return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_validate_l3_max_placement"
                ) as validate:
            mpam_resctrl._mpam_wait_for_l3_max_membership(
                "/fake", 100, attribution, 30)

        validate.assert_called_once_with(
            workload_ids, workers, attribution,
            require_worker_footprint=False)

    def test_l3_max_membership_retries_unstable_tree_until_deadline(self):
        """An unstable process tree keeps retrying but never proves placement."""
        attribution = self._window_placement()
        clock = self._FakeClock()

        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic",
                side_effect=clock.monotonic), \
                self.mock.patch.object(
                    mpam_resctrl.time, "sleep",
                    side_effect=clock.sleep), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    return_value={"100"}), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stable_workload_process_tree",
                    return_value=None) as stable_tree, \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_validate_l3_max_placement"
                ) as validate:
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_wait_for_l3_max_membership(
                    "/fake", 100, attribution, 30)

        self.assertIn("process/thread tree did not stabilize",
                      str(context.exception))
        self.assertIn("Deadline expired within the 5s membership window",
                      str(context.exception))
        self.assertGreaterEqual(stable_tree.call_count, 2)
        self.assertLessEqual(stable_tree.call_count, 60)
        self.assertGreaterEqual(clock.now, 5)
        validate.assert_not_called()

    def test_l3_max_vm_worker_pids_requires_exact_stress_ng_vm_comm(self):
        """Worker identity returns only direct workload leaves, not parents."""
        process_tree = self._l3_max_process_tree(
            {"100", "200", "201", "202", "203", "300", "301", "302", "303"},
            {"100", "200", "201", "202", "203", "300", "301", "302", "303"},
            self._l3_max_parent_leaf_edges())

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "process_comm",
                return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM):
            workers = mpam_resctrl._mpam_l3_max_vm_worker_pids(
                100, process_tree)

        self.assertEqual(workers, {"300", "301", "302", "303"})

    def test_l3_max_vm_worker_pids_blocks_missing_parent_leaf_pair(self):
        """Missing direct parent/leaf pairs cannot satisfy the four workers."""
        process_tree = self._l3_max_process_tree(
            {"100", "200", "201", "202", "300", "301", "302"},
            {"100", "200", "201", "202", "300", "301", "302"},
            (
                ("100", "200"), ("100", "201"), ("100", "202"),
                ("200", "300"), ("201", "301"), ("202", "302"),
            ))

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "process_comm",
                return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_vm_worker_pids(100, process_tree)

        self.assertIn("expected 4 direct", str(context.exception))

    def test_l3_max_vm_worker_pids_blocks_extra_parent_leaf_pair(self):
        """Extra direct parent/leaf pairs are invalid stress-ng topology."""
        process_tree = self._l3_max_process_tree(
            {
                "100", "200", "201", "202", "203", "204",
                "300", "301", "302", "303", "304",
            },
            {
                "100", "200", "201", "202", "203", "204",
                "300", "301", "302", "303", "304",
            },
            self._l3_max_parent_leaf_edges()
            + (("100", "204"), ("204", "304")))

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "process_comm",
                return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_vm_worker_pids(100, process_tree)

        self.assertIn("expected 4 direct", str(context.exception))

    def test_l3_max_vm_worker_pids_blocks_nested_leaf(self):
        """A workload leaf cannot hide another stress-ng VM child."""
        process_tree = self._l3_max_process_tree(
            {
                "100", "200", "201", "202", "203",
                "300", "301", "302", "303", "400",
            },
            {
                "100", "200", "201", "202", "203",
                "300", "301", "302", "303", "400",
            },
            self._l3_max_parent_leaf_edges() + (("300", "400"),))

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "process_comm",
                return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_vm_worker_pids(100, process_tree)

        self.assertIn("Nested stress-ng VM workload topology",
                      str(context.exception))

    def test_l3_max_vm_worker_pids_blocks_ambiguous_parent_children(self):
        """Each instance parent must have exactly one workload leaf child."""
        process_tree = self._l3_max_process_tree(
            {
                "100", "200", "201", "202", "203",
                "300", "301", "302", "303", "304",
            },
            {
                "100", "200", "201", "202", "203",
                "300", "301", "302", "303", "304",
            },
            self._l3_max_parent_leaf_edges() + (("200", "304"),))

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "process_comm",
                return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_vm_worker_pids(100, process_tree)

        self.assertIn("does not have exactly one child",
                      str(context.exception))

    def test_l3_max_vm_worker_pids_blocks_ambiguous_vm_names(self):
        """Loose VM-like descendant names are blocked, not counted."""
        def process_comm(pid):
            if pid == "200":
                return "stress-ng-vm2"
            return mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "process_comm",
                side_effect=process_comm):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_l3_max_vm_worker_pids(
                    100, self._l3_max_process_tree(
                        {
                            "100", "200", "201", "202", "203",
                            "300", "301", "302", "303",
                        },
                        {
                            "100", "200", "201", "202", "203",
                            "300", "301", "302", "303",
                        },
                        self._l3_max_parent_leaf_edges()))

        self.assertIn("Ambiguous stress-ng VM worker identity",
                      str(context.exception))

    def test_l3_max_stable_process_tree_blocks_edge_change_same_pid_set(self):
        """Stable snapshots compare direct edges, not only PID sets."""
        first = self._l3_max_process_tree(
            {"100", "200", "300"}, {"100", "200", "300"},
            (("100", "200"), ("200", "300")))
        second = self._l3_max_process_tree(
            {"100", "200", "300"}, {"100", "200", "300"},
            (("100", "200"), ("100", "300")))

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "process_tree",
                side_effect=[first, second]), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"):
            snapshot = mpam_resctrl._mpam_stable_workload_process_tree(100)

        self.assertIsNone(snapshot)

    def test_l3_max_membership_blocks_wrong_worker_count(self):
        """The live contract requires the exact four stress-ng VM workers."""
        attribution = self._window_placement()
        parents = {"200", "201", "202"}
        workers = {"300", "301", "302"}
        workload_ids = {"100"} | parents | workers
        process_tree = self._l3_max_process_tree(
            workload_ids, workload_ids,
            (
                ("100", "200"), ("100", "201"), ("100", "202"),
                ("200", "300"), ("201", "301"), ("202", "302"),
            ))

        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic",
                side_effect=[0, 0, 0, 6]), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    return_value=workload_ids), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stable_workload_process_tree",
                    return_value=process_tree), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "process_comm",
                    return_value=mpam_resctrl.MPAM_L3_MAX_LIVE_VM_WORKER_COMM), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_validate_l3_max_placement"
                ) as validate:
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_wait_for_l3_max_membership(
                    "/fake", 100, attribution, 30)

        validate.assert_not_called()
        self.assertIn("expected 4 direct", str(context.exception))

    def test_l3_max_membership_blocks_extra_group_task_after_restore(self):
        """Group tasks must exactly match the observed workload IDs."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}
        workload_ids = {"100", "101"} | workers

        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic",
                side_effect=[0, 0, 0, 6]), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    return_value=workload_ids | {"777"}), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stable_workload_process_tree",
                    return_value=self._l3_max_process_tree(
                        {"100"} | workers, workload_ids,
                        (("100", "200"), ("100", "300"),
                         ("100", "400"), ("100", "500")))), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_vm_worker_pids",
                    return_value=workers):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_wait_for_l3_max_membership(
                    "/fake", 100, attribution, 30)

        self.assertIn("must exactly match", str(context.exception))

    def test_l3_max_membership_blocks_second_tasks_read_drift(self):
        """Placement proof is bracketed by stable pre/post tasks reads."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}
        workload_ids = {"100", "101"} | workers

        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    side_effect=[workload_ids, workload_ids | {"777"}]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stable_workload_process_tree",
                    return_value=self._l3_max_process_tree(
                        {"100"} | workers, workload_ids,
                        (("100", "200"), ("100", "300"),
                         ("100", "400"), ("100", "500")))), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_vm_worker_pids",
                    return_value=workers), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_validate_l3_max_placement"):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_membership_snapshot(
                    "/fake", 100, attribution, 30)

        self.assertIn("/fake/tasks changed during membership proof",
                      str(context.exception))
        self.assertIn("extra ['777']", str(context.exception))

    def test_l3_max_membership_rechecks_deadline_after_placement(self):
        """A proof that finishes after the 5s membership window is blocked."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}
        workload_ids = {"100", "101"} | workers

        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic",
                side_effect=[0, 0, 0, 0, 6]), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_group_task_ids",
                    return_value=workload_ids), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_stable_workload_process_tree",
                    return_value=self._l3_max_process_tree(
                        {"100"} | workers, workload_ids,
                        (("100", "200"), ("100", "300"),
                         ("100", "400"), ("100", "500")))), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_vm_worker_pids",
                    return_value=workers), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_validate_l3_max_placement"
                ) as validate:
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_wait_for_l3_max_membership(
                    "/fake", 100, attribution, 30)

        validate.assert_called_once()
        self.assertIn("completed after the membership deadline",
                      str(context.exception))

    def test_l3_max_placement_blocks_tasks_outside_selected_node(self):
        """Placement proof uses observed task CPUs, not only numactl argv."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu",
                side_effect=lambda task_id: 7 if task_id == "200" else 4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    return_value={4, 5}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    return_value={2: 10}):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_placement(
                    {"100", "200"}, workers, attribution)

        self.assertIn("escaped selected node", str(context.exception))

    def test_l3_max_placement_blocks_worker_affinity_outside_node(self):
        """All workload affinity masks must be subsets of selected node CPUs."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    side_effect=lambda pid: {4, 7} if pid == "200" else {4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    return_value={2: 10}):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_placement(
                    {"100"} | workers, workers, attribution)

        self.assertIn("affinity mask", str(context.exception))

    def test_l3_max_placement_blocks_non_worker_affinity_outside_node(self):
        """Non-worker descendants also need node-confined affinity masks."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    side_effect=lambda pid: {4, 7} if pid == "101" else {4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    return_value={2: 10}):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_placement(
                    {"100", "101"} | workers, workers, attribution)

        self.assertIn("101", str(context.exception))
        self.assertIn("affinity mask", str(context.exception))

    def test_l3_max_placement_blocks_parent_affinity_outside_node(self):
        """Instance parents stay in CPU/affinity proof despite leaf footprint."""
        attribution = self._window_placement()
        workers = {"300", "301", "302", "303"}
        parents = {"200", "201", "202", "203"}

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    side_effect=lambda pid: {7} if pid == "200" else {4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    return_value={2: 10}):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_placement(
                    {"100"} | parents | workers, workers, attribution,
                    require_worker_footprint=True)

        self.assertIn("'200': [7]", str(context.exception))

    def test_l3_max_placement_excludes_instance_parents_from_footprint(self):
        """Only workload leaves provide the anonymous NUMA footprint proof."""
        attribution = self._window_placement()
        workers = {"300", "301", "302", "303"}
        parents = {"200", "201", "202", "203"}
        required_pages = (
            mpam_resctrl.MPAM_L3_MAX_LIVE_VM_FOOTPRINT_BYTES
            // mpam_resctrl.mmap.PAGESIZE)

        def numa_pages(pid):
            self.assertIn(pid, workers)
            return {2: required_pages}

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    return_value={4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    side_effect=numa_pages) as read_numa:
            mpam_resctrl._mpam_validate_l3_max_placement(
                {"100"} | parents | workers, workers, attribution,
                require_worker_footprint=True)

        self.assertCountEqual(
            [call.args[0] for call in read_numa.call_args_list],
            workers)

    def test_l3_max_placement_blocks_remote_worker_memory_pages(self):
        """Worker numa_maps must prove selected-node memory locality."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}
        required_pages = (
            mpam_resctrl.MPAM_L3_MAX_LIVE_VM_FOOTPRINT_BYTES
            // mpam_resctrl.mmap.PAGESIZE)

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    return_value={4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    side_effect=lambda pid: {2: required_pages, 3: 1}
                    if pid == "200" else {2: required_pages}):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_placement(
                    {"100"} | workers, workers, attribution,
                    require_worker_footprint=True)

        self.assertIn("memory locality", str(context.exception))

    def test_l3_max_placement_accepts_aggregate_worker_footprint(self):
        """stress-ng --vm-bytes is total, not per-worker, VM memory."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}
        required_pages = (
            mpam_resctrl.MPAM_L3_MAX_LIVE_VM_FOOTPRINT_BYTES
            // mpam_resctrl.mmap.PAGESIZE)

        def numa_pages(pid):
            if pid == "200":
                return {2: required_pages - 3}
            return {2: 1}

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    return_value={4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    side_effect=numa_pages):
            mpam_resctrl._mpam_validate_l3_max_placement(
                {"100"} | workers, workers, attribution,
                require_worker_footprint=True)

    def test_l3_max_placement_blocks_small_aggregate_footprint_after_warmup(
            self):
        """Post-warmup proof requires the selected 90% resident footprint."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}
        short_pages = (
            mpam_resctrl.MPAM_L3_MAX_LIVE_MIN_RESIDENT_BYTES
            // mpam_resctrl.mmap.PAGESIZE
            // len(workers)) - 1

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    return_value={4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    return_value={2: short_pages}):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_placement(
                    {"100"} | workers, workers, attribution,
                    require_worker_footprint=True)

        self.assertIn("aggregate local workload-anonymous footprint",
                      str(context.exception))
        self.assertIn("required at least",
                      str(context.exception))
        self.assertIn(f"{mpam_resctrl.MPAM_L3_MAX_LIVE_MIN_RESIDENT_BYTES} bytes", str(context.exception))

    def test_l3_max_placement_blocks_worker_without_local_footprint(self):
        """Each exact VM worker must have some selected-node anonymous pages."""
        attribution = self._window_placement()
        workers = {"200", "300", "400", "500"}
        required_pages = (
            mpam_resctrl.MPAM_L3_MAX_LIVE_VM_FOOTPRINT_BYTES
            // mpam_resctrl.mmap.PAGESIZE)

        def numa_pages(pid):
            if pid == "200":
                return {2: 0}
            return {2: required_pages}

        with self.mock.patch.object(
                mpam_resctrl.glt_proc, "task_cpu", return_value=4), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_allowed_cpus",
                    return_value={4}), \
                self.mock.patch.object(
                    mpam_resctrl.glt_proc, "task_anonymous_numa_pages_by_node",
                    side_effect=numa_pages):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_validate_l3_max_placement(
                    {"100"} | workers, workers, attribution,
                    require_worker_footprint=True)

        self.assertIn("no local workload-anonymous footprint",
                      str(context.exception))
        self.assertIn("'200': 0", str(context.exception))


    def test_l3_max_sample_stream_checks_deadline_before_tenth_sample(self):
        """The tenth llc_occupancy sample is still live and deadline-gated."""
        proc = self.mock.Mock()
        proc.poll.return_value = None

        with self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot"), \
                self.mock.patch.object(
                mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    return_value=123) as sample:
            samples = mpam_resctrl._mpam_collect_l3_max_samples(
                proc, 30, ("/fake", self._window_placement(), ("/fake/occ",)), "high", self.mock_logger)

        self.assertEqual(samples, tuple([123] * 10))
        self.assertEqual(proc.poll.call_count, 20)
        self.assertEqual(sample.call_count, 10)

    def test_l3_max_sample_stream_blocks_if_tenth_postcheck_exits(self):
        """The tenth sample must still be live after the occupancy read."""
        proc = self.mock.Mock()
        proc.poll.side_effect = [None] * 19 + [0]

        with self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot"), \
                self.mock.patch.object(
                mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    return_value=123):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_collect_l3_max_samples(
                    proc, 30, ("/fake", self._window_placement(), ("/fake/occ",)), "high", self.mock_logger)

        self.assertIn("after sample 9", str(context.exception))

    def test_l3_max_sample_stream_blocks_if_tenth_postcheck_misses_deadline(self):
        """The tenth sample also needs a deadline-valid post-read check."""
        proc = self.mock.Mock()
        proc.poll.return_value = None

        with self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot"), \
                self.mock.patch.object(
                mpam_resctrl.time, "monotonic",
                side_effect=[0] * 47 + [31]), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"), \
                self.mock.patch.object(
                    resctrl_utils, 'sum_l3_occupancy',
                    return_value=123):
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_collect_l3_max_samples(
                    proc, 30, ("/fake", self._window_placement(), ("/fake/occ",)), "high", self.mock_logger)

        self.assertIn("after sample 9", str(context.exception))

    def test_l3_max_sleep_blocks_if_post_sleep_deadline_expires(self):
        """A sleep that overruns its deadline is blocked after waking."""
        with self.mock.patch.object(
                mpam_resctrl.time, "monotonic",
                side_effect=[0, 1.01]), \
                self.mock.patch.object(mpam_resctrl.time, "sleep") as sleep:
            with self.assertRaises(
                    mpam_resctrl._MpamBlocked) as context:
                mpam_resctrl._mpam_sleep_l3_max_with_deadline(
                    1, 1, "warmup")

        sleep.assert_called_once_with(1)
        self.assertIn("deadline expired after warmup", str(context.exception))

    def test_l3_max_sample_stream_validates_membership_around_each_sample(self):
        """Every read has a fresh proof; the tenth also has a post-read proof."""
        state = _SampleProofs()
        proc = self.mock.Mock()
        proc.poll.return_value = None
        with self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot",
                                    side_effect=state.proof), \
                self.mock.patch.object(mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep", side_effect=state.sleep), \
                self.mock.patch.object(resctrl_utils, "sum_l3_occupancy", side_effect=state.read):
            samples = mpam_resctrl._mpam_collect_l3_max_samples(
                proc, 30, ("/fake", self._window_placement(), ("/fake/occ",)), "high", self.mock_logger)
        self.assertEqual(samples, (123,) * 10)
        expected = []
        for index in range(10):
            expected.extend((("proof", index, "before"), "read"))
            expected.append("sleep" if index < 9 else ("proof", 9, "after"))
        self.assertEqual(state.events, expected)

    def test_l3_max_sample_stream_blocks_membership_drift_after_sample(self):
        """Persistent drift after an interval stops before the next observation."""
        state = _SampleProofs("sample affinity drift")
        proc = self.mock.Mock()
        proc.poll.return_value = None
        with self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot",
                                    side_effect=state.proof), \
                self.mock.patch.object(mpam_resctrl.time, "monotonic", return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep", side_effect=state.sleep), \
                self.mock.patch.object(resctrl_utils, "sum_l3_occupancy", side_effect=state.read):
            with self.assertRaisesRegex(mpam_resctrl._MpamBlocked, "sample 1 before.*sample affinity drift"):
                mpam_resctrl._mpam_collect_l3_max_samples(
                    proc, 30, ("/fake", self._window_placement(), ("/fake/occ",)), "high", self.mock_logger)
        self.assertEqual(state.events, [("proof", 0, "before"), "read", "sleep",
                                       ("proof", 1, "before")])
        messages = self._formatted_logs(self.mock_logger)
        self.assertEqual(sum("sample=" in message for message in messages), 1)

    def test_l3_max_process_group_absence_polls_until_absent(self):
        """Post-SIGKILL proof accepts exists/exists/absent within deadline."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = 0
        with self.mock.patch.object(
                mpam_resctrl.os, "killpg",
                side_effect=[None, None, None, None, ProcessLookupError]) as killpg, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       return_value=0), \
                self.mock.patch.object(mpam_resctrl.time, "sleep") as sleep:
            errors = []
            mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)
            self.assertEqual(errors, [])

        self.assertEqual(
            killpg.call_args_list,
            [
                self.mock.call(4321, 0),
                self.mock.call(4321, mpam_resctrl.signal.SIGKILL),
                self.mock.call(4321, 0),
                self.mock.call(4321, 0),
                self.mock.call(4321, 0),
            ],
        )
        self.assertEqual(
            sleep.call_args_list,
            [
                self.mock.call(
                    mpam_resctrl.MPAM_L3_MAX_LIVE_KILL_CLEANUP_POLL_SEC),
                self.mock.call(
                    mpam_resctrl.MPAM_L3_MAX_LIVE_KILL_CLEANUP_POLL_SEC),
            ],
        )


    def test_l3_max_stop_kills_retained_process_group(self):
        """Cleanup still targets the process group after the leader exits."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = 0
        with self.mock.patch.object(
                mpam_resctrl.os, "killpg",
                side_effect=ProcessLookupError) as killpg:
            errors = []
            mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)
            self.assertEqual(errors, [])

        self.assertEqual(killpg.call_args_list, [self.mock.call(4321, 0), self.mock.call(4321, 0)])
        proc.poll.assert_called_once_with()
        proc.wait.assert_not_called()


    def test_l3_max_stop_kills_lingering_group_after_leader_exit(self):
        """A successful leader wait still probes and kills lingering workers."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = 0

        with self.mock.patch.object(
                mpam_resctrl.os, "killpg",
                side_effect=[None, None, ProcessLookupError]
        ) as killpg:
            errors = []
            mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)
            self.assertEqual(errors, [])

        self.assertEqual(
            killpg.call_args_list,
            [
                self.mock.call(4321, 0),
                self.mock.call(4321, mpam_resctrl.signal.SIGKILL),
                self.mock.call(4321, 0),
            ],
        )

        proc.poll.assert_called_once_with()
        proc.wait.assert_not_called()


    def test_l3_max_stop_closes_output_when_sigterm_removes_group(self):
        """The lexical lifetime closes capture after the clean TERM/leader wait."""
        capture = self.mock.Mock(closed=False)
        with self._window_environment() as (_handles, proc, _clock), \
                self.mock.patch.object(tempfile, "TemporaryFile", return_value=capture), \
                self.mock.patch.object(mpam_resctrl.os, "killpg",
                                       side_effect=ProcessLookupError) as killpg:
            self._run_decorated_window("high", self._window_placement(), "/fake")
        capture.close.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=5)
        killpg.assert_any_call(4321, mpam_resctrl.signal.SIGTERM)


    def test_l3_max_stop_reaps_after_sigkill_race(self):
        """A disappeared group after SIGKILL still reaps the Popen handle."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("stress-ng", 5),
            0,
        ]

        with self.mock.patch.object(
                mpam_resctrl.os, "killpg",
                side_effect=[None, ProcessLookupError, ProcessLookupError, ProcessLookupError]) as killpg:
            errors = []
            mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)
            self.assertEqual(errors, [])

        self.assertEqual(
            killpg.call_args_list,
            [
                self.mock.call(4321, mpam_resctrl.signal.SIGTERM),
                self.mock.call(4321, mpam_resctrl.signal.SIGKILL),
                self.mock.call(4321, 0),
                self.mock.call(4321, 0),
            ],
        )
        self.assertEqual(proc.wait.call_count, 2)
        proc.wait.assert_called_with(timeout=5)


    def test_l3_max_stop_surfaces_lingering_group_after_sigkill(self):
        """A signalable group after SIGKILL is a cleanup failure."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("stress-ng", 5),
            0,
        ]

        with self.mock.patch.object(
                mpam_resctrl.os, "killpg",
                side_effect=[None, None, None, None, None]) as killpg, \
                self.mock.patch.object(mpam_resctrl.time, "monotonic",
                                       side_effect=[0, 6]), \
                self.mock.patch.object(mpam_resctrl.time, "sleep"):
            errors = []
            mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)

        self.assertEqual(errors, [
            "Failed to terminate stress-ng process group 4321: "
            "process group 4321 still exists after 5 seconds"])
        self.assertEqual(
            killpg.call_args_list,
            [
                self.mock.call(4321, mpam_resctrl.signal.SIGTERM),
                self.mock.call(4321, mpam_resctrl.signal.SIGKILL),
                self.mock.call(4321, 0),
                self.mock.call(4321, mpam_resctrl.signal.SIGKILL),
                self.mock.call(4321, 0),
            ],
        )


    def test_l3_max_stop_surfaces_sigkill_wait_timeout(self):
        """A process still alive after SIGKILL becomes a cleanup failure."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("stress-ng", 5),
            subprocess.TimeoutExpired("stress-ng", 5),
        ]

        with self.mock.patch.object(mpam_resctrl.os, "killpg",
                                    side_effect=[None, None, ProcessLookupError, ProcessLookupError]):
            errors = []
            mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)
        self.assertEqual(errors, ["Failed to terminate stress-ng process group 4321: "
                                  + str(subprocess.TimeoutExpired("stress-ng", 5))])
        self.assertEqual(proc.wait.call_args_list, [self.mock.call(timeout=5), self.mock.call(timeout=5)])


    def test_percentage_metadata_reader_preserves_raw_policy_values(self):
        """Parsing retains policy facts so decorated cases can decide after cleanup."""
        for resource in ("L2_MAX", "L3_MAX"):
            for minimum, granularity in ((-1, 1), (100, 1), (1, 0), (1, 101)):
                with self.subTest(resource=resource, metadata=(minimum, granularity)):
                    with self._schemata_environment(
                            resource, info=("percentage", str(minimum), str(granularity))):
                        self.assertEqual(
                            mpam_resctrl._mpam_read_percentage_info(resource),
                            (minimum, granularity))

    def test_l3_max_helper_blocks_invalid_policy_bounds(self):
        """Out-of-policy percentage metadata raises a typed block."""
        cases = (
            ("min-negative", "-1", "1", "percent_min"),
            ("min-too-high", "100", "1", "percent_min"),
            ("gran-zero", "1", "0", "percent_gran"),
            ("gran-too-high", "1", "101", "percent_gran"),
        )

        for _name, percent_min, percent_gran, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(GLTCore.GLTTestWarn) as context:
                    self._run_temp_helper(
                        info=("percentage", percent_min, percent_gran))

                self.assertIn(expected, str(context.exception))

    def test_l3_max_helper_fails_malformed_info(self):
        """Malformed percentage metadata is a decorated-case failure reason."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_temp_helper(info=("percentage", "not-an-int", "1"))

        self.assertIn("Could not read L3_MAX percentage info", str(context.exception))

    def test_l3_max_helper_fails_non_percentage_schema(self):
        """Non-percentage MAX schemas fail rather than silently skipping."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_temp_helper(info=("bitmap", "1", "1"))

        self.assertIn("not in percentage mode", str(context.exception))

    def test_l3_max_helper_fails_empty_domain_schemata(self):
        """A present resource with no domains is malformed."""
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_temp_helper(initial_line="L3_MAX:")

        self.assertIn("Malformed L3_MAX entry", str(context.exception))

    def _run_ordered_helper(
            self, *, percent_min=1, percent_gran=1,
            initial_values=None):
        """Run the helper with mocked I/O and return operation order."""
        if initial_values is None:
            initial_values = {"1": 100, "2": 100}
        initial_values = {int(domain): value for domain, value in initial_values.items()}
        domains = tuple(initial_values)
        state = {"values": dict(initial_values)}
        order = []

        def line(values):
            return "L3_MAX:" + ";".join(
                f"{domain}={values[domain]}" for domain in domains)

        def write(_path, resource, write_domains, value_by_domain):
            self.assertEqual(resource, "L3_MAX")
            self.assertEqual(write_domains, domains)
            state["values"] = dict(value_by_domain)
            write_line = line(state["values"])
            order.append(("write", write_line))
            return write_line

        def read(_path, resource):
            self.assertEqual(resource, ("L3_MAX",))
            read_line = line(state["values"])
            order.append(("read", read_line))
            return {"L3_MAX": dict(state["values"])}

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(percent_min, percent_gran)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    side_effect=write), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=read), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            try:
                result = mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            except GLTCore.GLTTestWarn as error:
                result = error

        rmdir.assert_called_once()
        if result is None:
            last = self.mock_logger.info.call_args
            result = last.args[0] % last.args[1:]
        return result, order

    def test_upper_bound_probe_restore_then_high_collision_selects_99(self):
        """Initial 100 forces high-side search below the restored baseline."""
        result, order = self._run_ordered_helper(
            percent_min=1, percent_gran=1,
            initial_values={"1": 100, "2": 100})

        self.assertIn("high=99", result)
        self.assertEqual(
            order,
            [
                ("read", "L3_MAX:1=100;2=100"),
                ("write", "L3_MAX:1=100;2=100"),
                ("read", "L3_MAX:1=100;2=100"),
                ("write", "L3_MAX:1=100;2=100"),
                ("read", "L3_MAX:1=100;2=100"),
                ("write", "L3_MAX:1=1;2=1"),
                ("read", "L3_MAX:1=1;2=1"),
                ("write", "L3_MAX:1=99;2=99"),
                ("read", "L3_MAX:1=99;2=99"),
                ("write", "L3_MAX:1=100;2=100"),
                ("read", "L3_MAX:1=100;2=100"),
            ],
        )

    def test_low_side_collision_selects_one_after_restore(self):
        """Initial 0 forces low-up search to the next legal grid value."""
        result, order = self._run_ordered_helper(
            percent_min=0, percent_gran=1,
            initial_values={"1": 0, "2": 0})

        self.assertIn("low=1", result)
        self.assertEqual(
            order[5:9],
            [
                ("write", "L3_MAX:1=1;2=1"),
                ("read", "L3_MAX:1=1;2=1"),
                ("write", "L3_MAX:1=100;2=100"),
                ("read", "L3_MAX:1=100;2=100"),
            ],
        )

    def test_granularity_grid_uses_largest_reported_grid_value(self):
        """The legality probe uses grid[-1] without an off-grid endpoint."""
        result, order = self._run_ordered_helper(
            percent_min=12, percent_gran=12,
            initial_values={"1": 100, "2": 100})

        self.assertIn("low=12, high=96", result)
        self.assertEqual(order[1], ("write", "L3_MAX:1=96;2=96"))
        self.assertEqual(order[7], ("write", "L3_MAX:1=96;2=96"))

    def test_one_point_grid_still_probes_restores_and_removes_group(self):
        """A one-value grid still exercises write/readback before blocking."""
        result, order = self._run_ordered_helper(
            percent_min=50, percent_gran=100,
            initial_values={"1": 50, "2": 50})

        self.assertIsInstance(result, GLTCore.GLTTestWarn)
        self.assertIn("reported grid [50]", str(result))
        self.assertIn("no two distinct legal values", str(result))
        self.assertEqual(
            order,
            [
                ("read", "L3_MAX:1=50;2=50"),
                ("write", "L3_MAX:1=50;2=50"),
                ("read", "L3_MAX:1=50;2=50"),
                ("write", "L3_MAX:1=50;2=50"),
                ("read", "L3_MAX:1=50;2=50"),
                ("write", "L3_MAX:1=50;2=50"),
                ("read", "L3_MAX:1=50;2=50"),
            ],
        )

    def test_no_legal_pair_returns_contract_blocked(self):
        """Mixed initial domains can make every uniform grid write vacuous."""
        result, order = self._run_ordered_helper(
            percent_min=0, percent_gran=100,
            initial_values={"1": 0, "2": 100})

        self.assertIsInstance(result, GLTCore.GLTTestWarn)
        self.assertIn("no two distinct legal values", str(result))
        self.assertEqual(order[-2:], [
            ("write", "L3_MAX:1=0;2=100"),
            ("read", "L3_MAX:1=0;2=100"),
        ])

    def test_helper_write_error_fails_after_cleanup(self):
        """Write failures remain plain failures, with cleanup still attempted."""
        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    return_value={'L3_MAX': {1: 100}}), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    side_effect=ValueError(
                        "write denied")), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("write denied", str(context.exception))
        rmdir.assert_called_once()

    def test_helper_readback_mismatch_fails_and_removes_group(self):
        """Upper-bound readback mismatch fails, with group removal attempted."""
        readbacks = [
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            (["1"], {"1": 99}, "L3_MAX:1=99"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
        ]

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=[
                            record if isinstance(record, Exception) else
                            {"L3_MAX": {int(domain): value for domain, value in record[1].items()}}
                            for record in readbacks]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    return_value="L3_MAX:1=100"), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("wrote 'L3_MAX:1=100'", str(context.exception))
        self.assertIn("read back 'L3_MAX:1=99'", str(context.exception))
        self.assertIn("exact per-domain readback is required",
                      str(context.exception))
        rmdir.assert_called_once()

    def test_low_candidate_readback_mismatch_fails_and_removes_group(self):
        """Low candidate mismatch fails after deterministic cleanup."""
        readbacks = [
            (["1"], {"1": 50}, "L3_MAX:1=50"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            (["1"], {"1": 50}, "L3_MAX:1=50"),
            (["1"], {"1": 2}, "L3_MAX:1=2"),
            (["1"], {"1": 50}, "L3_MAX:1=50"),
        ]

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=[
                            record if isinstance(record, Exception) else
                            {"L3_MAX": {int(domain): value for domain, value in record[1].items()}}
                            for record in readbacks]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    side_effect=[
                        "L3_MAX:1=100",
                        "L3_MAX:1=50",
                        "L3_MAX:1=1",
                        "L3_MAX:1=50",
                    ]), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("wrote 'L3_MAX:1=1'", str(context.exception))
        self.assertIn("read back 'L3_MAX:1=2'", str(context.exception))
        self.assertIn("exact per-domain readback is required",
                      str(context.exception))
        rmdir.assert_called_once()

    def test_high_candidate_readback_mismatch_fails_and_removes_group(self):
        """High candidate mismatch fails after deterministic cleanup."""
        readbacks = [
            (["1"], {"1": 50}, "L3_MAX:1=50"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            (["1"], {"1": 50}, "L3_MAX:1=50"),
            (["1"], {"1": 1}, "L3_MAX:1=1"),
            (["1"], {"1": 99}, "L3_MAX:1=99"),
            (["1"], {"1": 50}, "L3_MAX:1=50"),
        ]

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=[
                            record if isinstance(record, Exception) else
                            {"L3_MAX": {int(domain): value for domain, value in record[1].items()}}
                            for record in readbacks]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    side_effect=[
                        "L3_MAX:1=100",
                        "L3_MAX:1=50",
                        "L3_MAX:1=1",
                        "L3_MAX:1=100",
                        "L3_MAX:1=50",
                    ]), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("wrote 'L3_MAX:1=100'", str(context.exception))
        self.assertIn("read back 'L3_MAX:1=99'", str(context.exception))
        self.assertIn("exact per-domain readback is required",
                      str(context.exception))
        rmdir.assert_called_once()

    def test_helper_readback_error_fails_and_removes_group(self):
        """Readback failures remain fail reasons, with group removal attempted."""
        readbacks = [
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            ValueError("read denied"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
        ]

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=[
                            record if isinstance(record, Exception) else
                            {"L3_MAX": {int(domain): value for domain, value in record[1].items()}}
                            for record in readbacks]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    return_value="L3_MAX:1=100"), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("read denied", str(context.exception))
        rmdir.assert_called_once()

    def test_unexpected_helper_exception_combines_group_cleanup_failure(self):
        """Unexpected helper exceptions propagate as programming defects."""
        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)),                 self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None),                 self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=TypeError("bad read")),                 self.mock.patch.object(
                    mpam_resctrl.os, "rmdir", side_effect=OSError("busy")) as rmdir:
            with self.assertRaisesRegex(TypeError, "bad read"):
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        rmdir.assert_called_once()

    def test_helper_restore_mismatch_fails_and_removes_group(self):
        """Restore readback mismatches fail, with group removal attempted."""
        readbacks = [
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            (["1"], {"1": 99}, "L3_MAX:1=99"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
        ]

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=[
                            record if isinstance(record, Exception) else
                            {"L3_MAX": {int(domain): value for domain, value in record[1].items()}}
                            for record in readbacks]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    return_value="L3_MAX:1=100"), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("Restored L3_MAX schemata mismatch", str(context.exception))
        rmdir.assert_called_once()

    def test_helper_restore_error_fails_and_removes_group(self):
        """Restore readback failures fail, with group removal attempted."""
        readbacks = [
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
            ValueError("restore read denied"),
            (["1"], {"1": 100}, "L3_MAX:1=100"),
        ]

        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=[
                            record if isinstance(record, Exception) else
                            {"L3_MAX": {int(domain): value for domain, value in record[1].items()}}
                            for record in readbacks]), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    return_value="L3_MAX:1=100"), \
                self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("restore read denied", str(context.exception))
        rmdir.assert_called_once()

    def test_unexpected_restore_exception_combines_with_group_removal(self):
        """Unexpected restore exceptions propagate as programming defects."""
        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)),                 self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None),                 self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    return_value={'L3_MAX': {1: 100}}),                 self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value=self._ok_schemata_result()),                 self.mock.patch.object(
                    mpam_resctrl, "_mpam_restore_percentage_schemata",
                    side_effect=[
                        TypeError("main restore"),
                        TypeError("cleanup restore"),
                    ]),                 self.mock.patch.object(mpam_resctrl.os, "rmdir") as rmdir:
            with self.assertRaisesRegex(TypeError, "cleanup restore"):
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        rmdir.assert_called_once()

    def test_helper_cleanup_success_removes_deterministic_group(self):
        """Successful cleanup removes the private group."""
        with TemporaryDirectory() as mount:
            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                    self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_percentage_info",
                    return_value=(1, 1)), \
                    self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=ValueError(
                        "read failed")):
                with self.assertRaises(
                        GLTCore.GLTTestFail):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

            self.assertEqual(list(Path(mount).glob('glt_l3_max_*')), [])

    def test_helper_cleanup_failure_leaves_created_group_for_manual_inspection(self):
        """Cleanup failure leaves the directory and retains the original failure."""
        uuid_value = self.mock.Mock()
        uuid_value.hex = "abcdef0123456789"

        with TemporaryDirectory() as mount:
            expected = str(
                Path(mount)
                / f"glt_l3_max_{mpam_resctrl.os.getpid()}_abcdef01")

            with self.mock.patch.object(
                    resctrl_utils, 'RESCTRL_MOUNT_POINT', mount), \
                    self.mock.patch.object(
                    mpam_resctrl.uuid, "uuid4", return_value=uuid_value), \
                    self.mock.patch.object(
                    mpam_resctrl, "_mpam_read_percentage_info",
                    return_value=(1, 1)), \
                    self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    side_effect=ValueError(
                        "read failed")), \
                    self.mock.patch.object(
                    mpam_resctrl.os, "rmdir", side_effect=OSError("busy")):
                with self.assertRaises(
                        GLTCore.GLTTestFail):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

            try:
                self.assertTrue(Path(expected).exists())
            finally:
                Path(expected).rmdir()

    def test_helper_cleanup_failure_fails_case(self):
        """Private-group removal failure fails even after successful readback."""
        with self.mock.patch.object(
                mpam_resctrl, "_mpam_read_percentage_info",
                return_value=(1, 1)), \
                self.mock.patch.object(
                    mpam_resctrl.os, 'mkdir',
                    return_value=None), \
                self.mock.patch.object(
                    resctrl_utils, 'read_schemata',
                    return_value={'L3_MAX': {1: 100}}), \
                self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_resource_schemata",
                    return_value="L3_MAX:1=100"), \
                self.mock.patch.object(
                    mpam_resctrl.os, "rmdir",
                    side_effect=OSError("busy")):
            with self.assertRaises(
                    GLTCore.GLTTestFail) as context:
                mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)

        self.assertIn("Failed to remove resctrl group", str(context.exception))


class TestMpamResctrlMaxLim(GLTTestBase):
    """Check write-access rejection through the public max_lim test."""

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        resctrl = Path(temp_dir.name)
        self.max_lim = resctrl / "info" / "MB" / "max_lim"
        self.max_lim.parent.mkdir(parents=True)
        self.max_lim.write_bytes(b"0\n")
        self.max_lim.chmod(0o444)
        nvdev = self.mock.Mock()
        nvdev.is_cpu_type.side_effect = lambda name: name == "vera"
        patches = (
            self.mock.patch.object(mpam_resctrl, "NvDev", return_value=nvdev),
            self.mock.patch.object(resctrl_utils, "RESCTRL_MOUNT_POINT",
                                   str(resctrl)),
            self.mock.patch.object(resctrl_utils, "managed_mount",
                                   return_value=nullcontext((True, None))),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_permission_errors_confirm_read_only(self):
        """Permission and read-only filesystem denials confirm write rejection."""
        for error_number in (errno.EACCES, errno.EPERM, errno.EROFS):
            with self.subTest(errno=error_number), \
                    self.mock.patch.object(mpam_resctrl.os, "open", side_effect=OSError(
                        error_number, "Write access denied")) as mock_open:
                mpam_resctrl.test_mpam_resctrl_mb_max_lim(None, self.mock_logger)

                mock_open.assert_called_once_with(str(self.max_lim), mpam_resctrl.os.O_WRONLY)

    def test_successful_write_open_fails_and_closes_descriptor(self):
        """An unexpectedly writable file fails without leaking its descriptor."""
        with self.mock.patch.object(mpam_resctrl.os, "open", return_value=123) as mock_open, \
                self.mock.patch.object(mpam_resctrl.os, "close") as mock_close:
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "accepted write access"):
                mpam_resctrl.test_mpam_resctrl_mb_max_lim(None, self.mock_logger)

            mock_open.assert_called_once_with(str(self.max_lim), mpam_resctrl.os.O_WRONLY)
            mock_close.assert_called_once_with(123)

    def test_unexpected_open_errors_fail(self):
        """Disappearing files or I/O errors cannot masquerade as write protection."""
        for error_number in (errno.ENOENT, errno.ENODEV, errno.EIO):
            with self.subTest(errno=error_number), \
                    self.mock.patch.object(mpam_resctrl.os, "open", side_effect=OSError(
                        error_number, "Unexpected open failure")):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "Failed to check write protection"):
                    mpam_resctrl.test_mpam_resctrl_mb_max_lim(None, self.mock_logger)




class TestMpamMaxCaptureLifetime(GLTTestBase):
    """Capture failures preserve workload outcomes and ordered cleanup evidence."""

    @contextmanager
    def _capture_environment(self):
        """Run the real lifetime with mocked kernel/process boundaries."""
        proc = self.mock.Mock(pid=4321)
        proc.poll.return_value = 0
        capture = self.mock.Mock()
        capture.tell.return_value = 0
        capture.read.return_value = b""
        patches = (
            self.mock.patch.object(mpam_resctrl.threading, "get_native_id", return_value=777),
            self.mock.patch.object(mpam_resctrl, "_mpam_resctrl_group_for_task",
                                   return_value=("/original", None)),
            self.mock.patch.object(mpam_resctrl.glt_utils, "write_sysfs_value"),
            self.mock.patch.object(mpam_resctrl, "_mpam_read_group_task_ids",
                                   side_effect=[{"777"}, OSError("quiesce unreadable")]),
            self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_restore_supervisor"),
            self.mock.patch.object(tempfile, "TemporaryFile", return_value=capture),
            self.mock.patch.object(mpam_resctrl.subprocess, "Popen", return_value=proc),
            self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=ProcessLookupError),
            self.mock.patch.object(mpam_resctrl.time, "monotonic", return_value=0),
            self.mock.patch.object(mpam_resctrl.time, "sleep"),
        )
        with ExitStack() as stack:
            handles = [stack.enter_context(patch) for patch in patches]
            yield capture, handles[5], handles[6]

    def test_capture_creation_failure_maps_to_launch_blockage(self):
        """An unavailable temporary file blocks launch before Popen is called."""
        cleanup_errors = []
        injected = OSError("capture unavailable")
        with self._capture_environment() as (capture, temporary_file, popen):
            temporary_file.side_effect = injected
            with self.assertRaisesRegex(mpam_resctrl._MpamBlocked, "Could not launch") as caught:
                with mpam_resctrl._mpam_l3_max_workload(
                        "/private", {"node": 2, "cpus": (4, 5), "cache_ids": (9,)}, cleanup_errors):
                    self.fail("workload body must not run after capture setup failure")
        self.assertIs(caught.exception.__cause__, injected)
        popen.assert_not_called()
        capture.close.assert_not_called()
        self.assertEqual(cleanup_errors, [])

    def test_spawn_and_close_failure_preserve_exception_chain_and_close_once(self):
        """Failed spawn closes before blockage conversion, retaining both failures."""
        cleanup_errors = []
        events = []
        spawn_error = OSError("spawn denied")
        close_error = OSError("capture close denied")

        def spawn(*_args, **_kwargs):
            events.append("spawn")
            raise spawn_error

        def close():
            events.append("close")
            raise close_error

        with self._capture_environment() as (capture, _temporary_file, popen):
            popen.side_effect = spawn
            capture.close.side_effect = close
            with self.assertRaisesRegex(mpam_resctrl._MpamBlocked, "Could not launch") as caught:
                with mpam_resctrl._mpam_l3_max_workload(
                        "/private", {"node": 2, "cpus": (4, 5), "cache_ids": (9,)}, cleanup_errors):
                    self.fail("workload body must not run after spawn failure")
        self.assertEqual(events, ["spawn", "close"])
        self.assertIs(caught.exception.__cause__, close_error)
        self.assertIs(close_error.__context__, spawn_error)
        capture.close.assert_called_once_with()
        self.assertEqual(cleanup_errors, [])

    def test_launched_capture_close_preserves_primary_and_precedes_quiesce(self):
        """Close errors remain diagnostics before quiesce errors for every outcome."""
        for primary in (None, ValueError("body failed"),
                        mpam_resctrl._MpamBlocked("body blocked"), ValueError("body error")):
            with self.subTest(primary=primary):
                cleanup_errors = []
                with self._capture_environment() as (capture, _temporary_file, _popen):
                    capture.close.side_effect = OSError("close denied")
                    expected = (self.assertRaisesRegex(type(primary), str(primary))
                                if primary is not None else nullcontext())
                    with expected as caught:
                        with mpam_resctrl._mpam_l3_max_workload(
                                "/private", {"node": 2, "cpus": (4, 5), "cache_ids": (9,)}, cleanup_errors):
                            if primary is not None:
                                raise primary
                capture.close.assert_called_once_with()
                self.assertEqual(cleanup_errors, [
                    "Could not close stress-ng output capture: close denied",
                    "Could not verify residual task state in /private: quiesce unreadable",
                ])
                if isinstance(primary, (ValueError, ValueError)):
                    assert caught is not None
                    self.assertIs(caught.exception, primary)


class TestMpamMaxNativeBoundaries(_MpamMaxSchemataHelpers):
    """Native operation faults must fail after all owned cleanup finishes."""

    @contextmanager
    def _native_entry_environment(self, resource="L3_MAX"):
        """Model kernel-created schemata while running the decorated owner."""
        with TemporaryDirectory() as mount:
            root = Path(mount)
            self._seed_percentage_resource(root, resource)
            mkdir = mpam_resctrl.os.mkdir
            rmdir = mpam_resctrl.os.rmdir
            events = []
            placement = self.mock.MagicMock(node=2, cpus=(4, 5), cache_ids=("1", "2"))
            placement.__getitem__.side_effect = {
                "node": 2, "cpus": (4, 5), "cache_ids": ("1", "2")
            }.__getitem__

            def create(path, *args, **kwargs):
                mkdir(path, *args, **kwargs)
                (Path(path) / "schemata").write_text(
                    f"{resource}:1=50;2=50\n", encoding="ascii")
                events.append("mkdir")

            def remove(path, *args, **kwargs):
                (Path(path) / "schemata").unlink()
                rmdir(path, *args, **kwargs)
                events.append("rmdir")

            @contextmanager
            def mounted(_logger, **_kwargs):
                try:
                    yield True, None
                finally:
                    events.append("unmount")

            with ExitStack() as stack:
                nvdev = stack.enter_context(self.mock.patch.object(mpam_resctrl, "NvDev"))
                self._patch_vera(nvdev)
                stack.enter_context(self.mock.patch.object(
                    resctrl_utils, "RESCTRL_MOUNT_POINT", mount))
                stack.enter_context(self.mock.patch.object(
                    resctrl_utils, "managed_mount", side_effect=mounted))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl.os, "mkdir", side_effect=create))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl.os, "rmdir", side_effect=remove))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl.glt_utils, "is_cli_available", return_value=True))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_require_stress_version"))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_preflight_supervisor_children"))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_attribution",
                    return_value=placement))
                stack.enter_context(self.mock.patch.object(
                    mpam_resctrl, "_mpam_write_and_require_percentage",
                    return_value="exact readback"))
                restore = stack.enter_context(self.mock.patch.object(
                    mpam_resctrl, "_mpam_restore_percentage_schemata"))
                yield events, restore

    def _native_probe_faults_fail_both_schemata_entries_after_cleanup(self, resource, entry, error_type):
        """Both MAX policies classify native probe faults as FAIL, never WARN."""
        with self._native_entry_environment(resource) as (events, restore):
            with self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair",
                    side_effect=error_type("native probe fault")):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "native probe fault"):
                    entry(None, self.mock_logger)
            restore.assert_called_once()
            self.assertEqual(events, ["mkdir", "rmdir", "unmount"])

    def test_native_probe_faults_fail_both_schemata_entries_after_cleanup_l2_max_valueerror(self):
        """Native ValueError becomes FAIL after L2_MAX cleanup."""
        self._native_probe_faults_fail_both_schemata_entries_after_cleanup(
            "L2_MAX", mpam_resctrl.test_mpam_l2_max_schemata, ValueError)

    def test_native_probe_faults_fail_both_schemata_entries_after_cleanup_l2_max_oserror(self):
        """Native OSError becomes FAIL after L2_MAX cleanup."""
        self._native_probe_faults_fail_both_schemata_entries_after_cleanup(
            "L2_MAX", mpam_resctrl.test_mpam_l2_max_schemata, OSError)

    def test_native_probe_faults_fail_both_schemata_entries_after_cleanup_l3_max_valueerror(self):
        """Native ValueError becomes FAIL after L3_MAX cleanup."""
        self._native_probe_faults_fail_both_schemata_entries_after_cleanup(
            "L3_MAX", mpam_resctrl.test_mpam_l3_max_schemata, ValueError)

    def test_native_probe_faults_fail_both_schemata_entries_after_cleanup_l3_max_oserror(self):
        """Native OSError becomes FAIL after L3_MAX cleanup."""
        self._native_probe_faults_fail_both_schemata_entries_after_cleanup(
            "L3_MAX", mpam_resctrl.test_mpam_l3_max_schemata, OSError)

    def _native_group_restore_faults_fail_after_remove_and_unmount(self, error_type):
        """Native restore errors cannot bypass removal or decorated FAIL."""
        with self._native_entry_environment() as (events, restore):
            restore.side_effect = error_type("native restore fault")
            with self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair", return_value=(10, 100)):
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "native restore fault"):
                    mpam_resctrl.test_mpam_l3_max_schemata(None, self.mock_logger)
            self.assertEqual(events, ["mkdir", "rmdir", "unmount"])

    def test_native_group_restore_faults_fail_after_remove_and_unmount_valueerror(self):
        """Native ValueError retains ordered cleanup and FAIL semantics."""
        self._native_group_restore_faults_fail_after_remove_and_unmount(ValueError)

    def test_native_group_restore_faults_fail_after_remove_and_unmount_oserror(self):
        """Native OSError retains ordered cleanup and FAIL semantics."""
        self._native_group_restore_faults_fail_after_remove_and_unmount(OSError)

    def _native_live_window_faults_fail_and_stop_after_second_window(self, error_type):
        """A second-window native discovery fault keeps real workload/cleanup ordering."""
        with self._native_entry_environment() as (events, restore), ExitStack() as stack:
            state = _FreshWorkload(self, {"node": 2, "cpus": (4, 5), "cache_ids": (9,)})
            stack.enter_context(self.mock.patch.object(
                mpam_resctrl, "_mpam_l3_max_attribution", return_value=state.placement))
            patches = (
                self.mock.patch.object(mpam_resctrl, "_mpam_probe_percentage_pair", return_value=(10, 100)),
                self.mock.patch.object(mpam_resctrl, "_mpam_l3_max_occupancy_files",
                    side_effect=partial(_fixture_second_discovery, state, error_type("native window fault"))),
                self.mock.patch.object(resctrl_utils, "sum_l3_occupancy", side_effect=state.occupancy),
                self.mock.patch.object(mpam_resctrl.glt_utils, "write_sysfs_value", side_effect=state.write_task),
                self.mock.patch.object(mpam_resctrl, "_mpam_read_group_task_ids", side_effect=state.read_tasks),
                self.mock.patch.object(mpam_resctrl, "_mpam_resctrl_group_for_task",
                                       return_value=("/original", None)),
                self.mock.patch.object(mpam_resctrl.threading, "get_native_id", return_value=777),
                self.mock.patch.object(mpam_resctrl, "_mpam_wait_for_l3_max_membership"),
                self.mock.patch.object(mpam_resctrl, "_mpam_validate_l3_max_membership_snapshot"),
                self.mock.patch.object(mpam_resctrl.time, "monotonic", return_value=0),
                self.mock.patch.object(mpam_resctrl.time, "sleep"),
            )
            for patch in patches:
                stack.enter_context(patch)
            start = stack.enter_context(self.mock.patch.object(
                mpam_resctrl.subprocess, "Popen", side_effect=state.launch))
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "native window fault"):
                mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            self.assertEqual(start.call_count, 1)
            self.assertEqual(len(state.sample_reads), 10)
            self.assertEqual(restore.call_count, 3)
            self.assertEqual(events, ["mkdir", "rmdir"] * 3 + ["unmount"])

    def test_native_live_window_faults_fail_and_stop_after_second_window_valueerror(self):
        """Native ValueError retains ordered cleanup and FAIL semantics."""
        self._native_live_window_faults_fail_and_stop_after_second_window(ValueError)

    def test_native_live_window_faults_fail_and_stop_after_second_window_oserror(self):
        """Native OSError retains ordered cleanup and FAIL semantics."""
        self._native_live_window_faults_fail_and_stop_after_second_window(OSError)

    def _native_calibration_restore_faults_prevent_all_windows(self, error_type):
        """Failed calibration restoration cannot launch measurement work."""
        with self._native_entry_environment() as (events, restore):
            restore.side_effect = error_type("native calibration restore fault")
            with self.mock.patch.object(
                    mpam_resctrl, "_mpam_probe_percentage_pair", return_value=(10, 100)), \
                    self.mock.patch.object(mpam_resctrl.subprocess, "Popen") as window:
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "native calibration restore fault"):
                    mpam_resctrl.test_mpam_l3_max_live_effect(None, self.mock_logger)
            window.assert_not_called()
            self.assertEqual(events, ["mkdir", "rmdir", "unmount"])

    def test_native_calibration_restore_faults_prevent_all_windows_valueerror(self):
        """Native ValueError retains ordered cleanup and FAIL semantics."""
        self._native_calibration_restore_faults_prevent_all_windows(ValueError)

    def test_native_calibration_restore_faults_prevent_all_windows_oserror(self):
        """Native OSError retains ordered cleanup and FAIL semantics."""
        self._native_calibration_restore_faults_prevent_all_windows(OSError)

    def _native_supervisor_restore_faults_preserve_body_and_capture_cleanup(self, error_type):
        """Restore faults remain ordered diagnostics without losing a body fault."""
        placement = self.mock.MagicMock(node=2)
        placement.__getitem__.side_effect = {"node": 2}.__getitem__
        environment = TestMpamMaxCaptureLifetime()
        environment.setUp()
        self.addCleanup(environment.doCleanups)
        cleanup_errors = []
        primary = error_type("native initial restore fault")
        with environment._capture_environment() as (capture, _temporary_file, _popen):
            with self.mock.patch.object(
                    mpam_resctrl, "_mpam_l3_max_restore_supervisor",
                    side_effect=[primary, error_type("native cleanup restore fault")]):
                with self.assertRaisesRegex(error_type, "native initial restore fault") as caught:
                    with mpam_resctrl._mpam_l3_max_workload(
                            "/private", placement, cleanup_errors):
                        self.fail("launch restoration must finish before yielding")
        self.assertIs(caught.exception, primary)
        capture.close.assert_called_once_with()
        self.assertEqual(cleanup_errors, [
            "native cleanup restore fault",
            "Could not verify residual task state in /private: quiesce unreadable",
        ])

    def test_native_supervisor_restore_faults_preserve_body_and_capture_cleanup_valueerror(self):
        """Native ValueError retains ordered cleanup and FAIL semantics."""
        self._native_supervisor_restore_faults_preserve_body_and_capture_cleanup(ValueError)

    def test_native_supervisor_restore_faults_preserve_body_and_capture_cleanup_oserror(self):
        """Native OSError retains ordered cleanup and FAIL semantics."""
        self._native_supervisor_restore_faults_preserve_body_and_capture_cleanup(OSError)


class TestMpamMaxProcessObservation(GLTTestBase):
    """Only proven group absence permits reading completed-workload evidence."""

    def test_group_absence_requires_process_lookup_error(self):
        """Permission and I/O errors cannot be mistaken for successful absence."""
        with self.mock.patch.object(mpam_resctrl.os, "killpg"):
            self.assertTrue(mpam_resctrl._mpam_l3_max_group_exists(4321))
        for injected in (ProcessLookupError("gone"), PermissionError("denied"),
                         OSError("probe failed")):
            with self.subTest(error=injected), self.mock.patch.object(
                    mpam_resctrl.os, "killpg", side_effect=injected):
                if isinstance(injected, ProcessLookupError):
                    self.assertFalse(mpam_resctrl._mpam_l3_max_group_exists(4321))
                else:
                    with self.assertRaises(type(injected)) as caught:
                        mpam_resctrl._mpam_l3_max_group_exists(4321)
                    self.assertIs(caught.exception, injected)

    def test_output_none_is_distinct_from_empty_completed_capture(self):
        """Live writers withhold output; an empty completed capture remains empty."""
        capture = self.mock.Mock(closed=False)
        with self.mock.patch.object(mpam_resctrl.os, "killpg") as signal_zero:
            self.assertIsNone(mpam_resctrl._mpam_l3_max_output(4321, capture))
        signal_zero.assert_called_once_with(4321, 0)
        capture.seek.assert_not_called()
        capture.read.assert_not_called()
        with self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=ProcessLookupError):
            self.assertEqual(mpam_resctrl._mpam_l3_max_output(4321, io.BytesIO()), "")
            self.assertIsNone(mpam_resctrl._mpam_l3_max_output(4321, None))


    def test_output_tail_preserves_summary_decoding_and_read_error(self):
        """Retain the bounded late evidence, replacing invalid UTF-8 visibly."""
        limit = mpam_resctrl.MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES
        capture = io.BytesIO(b"x" * limit + b"\xff summary\n")
        with self.mock.patch.object(mpam_resctrl.os, "killpg", side_effect=ProcessLookupError):
            output = mpam_resctrl._mpam_l3_max_output(4321, capture)
            self.assertEqual(output, f"<kept last {limit} of {limit + 10} bytes>\n"
                             + "x" * (limit - 10) + "\ufffd summary")
            broken = self.mock.Mock(closed=False)
            broken.seek.side_effect = OSError("read denied")
            self.assertEqual(mpam_resctrl._mpam_l3_max_output(4321, broken),
                             "<could not read process output: read denied>")

    def test_stop_preserves_signal_error_types_as_cleanup_diagnostics(self):
        """Gone groups are harmless; permission and I/O failures stay visible."""
        for signal_error in (ProcessLookupError("gone"), PermissionError("denied"),
                             OSError("signal failed")):
            with self.subTest(error=signal_error):
                proc = self.mock.Mock(pid=4321)
                proc.poll.return_value = None
                proc.wait.return_value = 0
                errors = []
                with self.mock.patch.object(mpam_resctrl.os, "killpg",
                                            side_effect=[signal_error, ProcessLookupError,
                                                         ProcessLookupError]):
                    mpam_resctrl._mpam_stop_l3_max_workload(proc, None, errors, None)
                if isinstance(signal_error, ProcessLookupError):
                    self.assertEqual(errors, [])
                else:
                    self.assertEqual(errors, [
                        f"Failed to terminate stress-ng process group 4321: {signal_error}"])
                if not isinstance(signal_error, (PermissionError, ProcessLookupError)):
                    proc.wait.assert_not_called()
                else:
                    proc.wait.assert_called_once_with(timeout=5)

    def test_output_probe_errors_propagate_without_reading_capture(self):
        """Output cannot establish absence when signal-0 is denied or broken."""
        capture = self.mock.Mock(closed=False)
        for injected in (PermissionError("denied"), OSError("probe failed")):
            with self.subTest(error=injected), self.mock.patch.object(
                    mpam_resctrl.os, "killpg", side_effect=injected):
                with self.assertRaises(type(injected)) as caught:
                    mpam_resctrl._mpam_l3_max_output(4321, capture)
                self.assertIs(caught.exception, injected)
        capture.seek.assert_not_called()
        capture.read.assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
