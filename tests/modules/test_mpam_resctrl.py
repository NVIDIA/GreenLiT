"""Unit tests for ARM64 MPAM resctrl schema and workload policy."""
# pylint: disable=wrong-import-position

from contextlib import contextmanager, ExitStack, nullcontext
import errno
from io import StringIO
from itertools import chain, repeat
import sys
import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase

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


if __name__ == "__main__":
    GLTTestBase.unittest.main()
