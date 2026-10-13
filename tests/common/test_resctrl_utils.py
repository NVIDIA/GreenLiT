"""Unit tests for common/resctrl_utils.py."""

from contextlib import ExitStack
import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase
from src.common import resctrl_utils


class TestResctrlMountState(GLTTestBase):
    """Filesystem support and mount-point discovery from proc files."""

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.filesystems = self.root / "filesystems"
        self.mounts = self.root / "mounts"
        patcher = self.mock.patch.multiple(
            resctrl_utils, PROC_FILESYSTEMS=str(self.filesystems),
            PROC_MOUNTS=str(self.mounts))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fs_supported_matches_last_field_only(self):
        """Only the filesystem name column counts; an unreadable file is an error."""
        with self.assertRaises(OSError):
            resctrl_utils.fs_supported()
        self.filesystems.write_text("nodev\tsysfs\nnodev\tresctrlfs\n", encoding="utf-8")
        self.assertFalse(resctrl_utils.fs_supported())
        self.filesystems.write_text("nodev\tsysfs\nnodev\tresctrl\n\n", encoding="utf-8")
        self.assertTrue(resctrl_utils.fs_supported())

    def test_mount_points_and_is_mounted(self):
        """Every resctrl mount is listed; is_mounted checks the default path."""
        self.mounts.write_text(
            "sysfs /sys sysfs rw 0 0\n"
            "resctrl /mnt/rc resctrl rw 0 0\n"
            "resctrl /sys/fs/resctrl resctrl rw 0 0\n"
            "garbage\n", encoding="utf-8")
        self.assertEqual(resctrl_utils.mount_points(), ["/mnt/rc", "/sys/fs/resctrl"])
        self.assertTrue(resctrl_utils.is_mounted())

        self.mounts.write_text("resctrl /mnt/rc resctrl rw 0 0\n", encoding="utf-8")
        self.assertFalse(resctrl_utils.is_mounted())

    def test_unreadable_proc_mounts_raises(self):
        """An unreadable mount table is an error, not "not mounted"."""
        with self.assertRaises(OSError):
            resctrl_utils.mount_points()
        with self.assertRaises(OSError):
            resctrl_utils.is_mounted()


class TestManagedMount(GLTTestBase):
    """Mount lifecycle owned by the helper."""

    def _run(self, *, mounted_sequence, cli_available=True, run_side_effect=None,
             cleanup_errors=None):
        with ExitStack() as stack:
            stack.enter_context(self.mock.patch.object(
                resctrl_utils, "is_mounted", side_effect=mounted_sequence))
            stack.enter_context(self.mock.patch.object(
                resctrl_utils.glt_utils, "is_cli_available",
                side_effect=cli_available if callable(cli_available) else None,
                return_value=cli_available))
            mock_run = stack.enter_context(self.mock.patch.object(
                resctrl_utils.glt_utils, "run_command", side_effect=run_side_effect))
            with resctrl_utils.managed_mount(
                    self.mock_logger, cleanup_errors=cleanup_errors) as state:
                pass
        return state, mock_run

    def test_existing_mount_is_left_alone(self):
        """A caller-owned mount is neither mounted nor unmounted."""
        state, mock_run = self._run(mounted_sequence=(True,))
        self.assertEqual(state, (True, None))
        mock_run.assert_not_called()

    def test_mounts_then_unmounts_when_absent(self):
        """A helper-owned mount is created and removed around the body."""
        state, mock_run = self._run(mounted_sequence=(False, True))
        self.assertEqual(state, (True, None))
        self.assertEqual(
            [call.args[0] for call in mock_run.call_args_list],
            [["mount", "-t", "resctrl", "resctrl", "/sys/fs/resctrl"],
             ["umount", "/sys/fs/resctrl"]])

    def test_missing_cli_is_reported_without_mounting(self):
        """The first missing CLI is named and nothing is executed."""
        state, mock_run = self._run(
            mounted_sequence=(False,), cli_available=lambda cli: cli != "umount")
        self.assertEqual(state, (False, "umount"))
        mock_run.assert_not_called()

    def test_mount_failure_yields_false_and_skips_unmount(self):
        """A failed mount is reported and never unmounted."""
        state, mock_run = self._run(
            mounted_sequence=(False,), run_side_effect=RuntimeError("mount failed"))
        self.assertEqual(state, (False, None))
        self.assertEqual(mock_run.call_count, 1)

    def test_unreadable_mount_table_is_logged_and_nothing_is_mounted(self):
        """With the mount state unknown, no mount is attempted and the cause is logged."""
        state, mock_run = self._run(mounted_sequence=(OSError("no /proc"),))
        self.assertEqual(state, (False, None))
        mock_run.assert_not_called()
        self.mock_logger.error.assert_called_once()
        self.assertIn("no /proc", str(self.mock_logger.error.call_args))

    def test_unconfirmable_mount_is_reported_and_unmounted(self):
        """A mount that cannot be confirmed yields False and is still cleaned up."""
        state, mock_run = self._run(mounted_sequence=(False, OSError("no /proc")))
        self.assertEqual(state, (False, None))
        self.assertEqual(
            [call.args[0][0] for call in mock_run.call_args_list], ["mount", "umount"])
        self.mock_logger.error.assert_called_once()

    def test_unmount_failure_is_logged_and_collected(self):
        """An unmount failure is logged and appended to cleanup_errors."""
        errors = []
        state, _ = self._run(
            mounted_sequence=(False, True),
            run_side_effect=("", RuntimeError("Failed to unmount resctrl.")),
            cleanup_errors=errors)
        self.assertEqual(state, (True, None))
        self.assertEqual(errors, ["Failed to unmount resctrl."])
        self.mock_logger.warning.assert_called_once_with("Failed to unmount resctrl.")


class TestSchemataParsing(GLTTestBase):
    """Strict parsing of schemata files."""

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.schemata = self.root / "schemata"
        patcher = self.mock.patch.object(
            resctrl_utils, "RESCTRL_MOUNT_POINT", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_read_schemata_returns_requested_resources_only(self):
        """Unrequested resources are ignored and absent ones are not reported."""
        self.schemata.write_text(
            "\n    L3:0=ffff;2=ffff\n    MB:0=100;2=50\n  MB_HLIM:2=0;0=1\n\n",
            encoding="ascii")
        self.assertEqual(
            resctrl_utils.read_schemata(self.schemata, ("MB", "MB_HLIM", "MB_NODE")),
            {"MB": {0: 100, 2: 50}, "MB_HLIM": {2: 0, 0: 1}})

    def test_read_schemata_rejects_malformed_input(self):
        """Duplicate lines, bare resources, bad entries, and duplicate domains fail."""
        cases = {
            "MB:0=100\nMB:2=100\n": "Malformed or duplicate MB line",
            "MB\n": "Malformed or duplicate MB line",
            "MB:0=100;broken\n": "Malformed MB entry",
            "MB:0=100;0=50\n": "Duplicate MB domain 0",
        }
        for content, message in cases.items():
            with self.subTest(content=content):
                self.schemata.write_text(content, encoding="ascii")
                with self.assertRaisesRegex(ValueError, message):
                    resctrl_utils.read_schemata(self.schemata, ("MB",))

    def test_schemata_domains(self):
        """Domains are collected from the root schemata, None when absent."""
        self.schemata.write_text("  L3:0=ffff;2=ffff\nMB:0=100;7=100\n", encoding="utf-8")
        self.assertEqual(resctrl_utils.schemata_domains("L3"), {0, 2})
        self.assertEqual(resctrl_utils.schemata_domains("MB"), {0, 7})
        self.assertIsNone(resctrl_utils.schemata_domains("MB_NODE"))

    def test_schemata_domains_rejects_corrupt_lines(self):
        """Empty entries, bad domains, and repeated domains are errors, not sets."""
        cases = {
            "MB:0=100;;7=100\n": "Malformed MB entry",
            "MB:0=100;x=1\n": "Malformed MB entry",
            "MB:0=100;7\n": "Malformed MB entry",
            "MB:0=100;0=50\n": "Duplicate MB domain 0",
        }
        for content, message in cases.items():
            with self.subTest(content=content):
                self.schemata.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    resctrl_utils.schemata_domains("MB")

    def test_mb_control_resource(self):
        """control_mode selects MB or MB_NODE; its absence means MB."""
        self.assertEqual(resctrl_utils.mb_control_resource(), "MB")
        control_mode = self.root / "info" / "MB" / "control_mode"
        control_mode.parent.mkdir(parents=True)
        for content, expected in (("[legacy] native\n", "MB"), ("legacy [native]\n", "MB_NODE")):
            control_mode.write_text(content, encoding="utf-8")
            self.assertEqual(resctrl_utils.mb_control_resource(), expected)
        control_mode.write_text("legacy native\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no active legacy or native mode"):
            resctrl_utils.mb_control_resource()


class TestMonitoringFiles(GLTTestBase):
    """Parsing of MBM assignment, counter, and occupancy files."""

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)

    def _file(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_read_mbm_assignment_states(self):
        """Only the requested event is parsed and states are keyed by domain."""
        path = self._file("assign", "mbm_local_bytes:0=e\nmbm_total_bytes:0=e;2=_\n\n")
        self.assertEqual(resctrl_utils.read_mbm_assignment_states(path), {0: "e", 2: "_"})
        self.assertEqual(
            resctrl_utils.read_mbm_assignment_states(path, "mbm_local_bytes"), {0: "e"})

    def test_read_mbm_assignment_states_rejects_malformed_input(self):
        """Missing separators, bad entries, duplicates, and absence all fail."""
        cases = {
            "garbage\n": "Malformed MBM assignment line",
            "mbm_total_bytes:0=x\n": "Malformed MBM assignment entry",
            "mbm_total_bytes:0=e;0=_\n": "Duplicate MBM domain 0",
            "mbm_total_bytes:0=e\nmbm_total_bytes:2=e\n": "Duplicate mbm_total_bytes assignment",
            "mbm_local_bytes:0=e\n": "No mbm_total_bytes assignment states",
        }
        for content, message in cases.items():
            with self.subTest(content=content):
                path = self._file("assign", content)
                with self.assertRaisesRegex(ValueError, message):
                    resctrl_utils.read_mbm_assignment_states(path)

    def test_read_mbm_counter_counts(self):
        """Counter counts are keyed by domain and malformed input fails."""
        self.assertEqual(
            resctrl_utils.read_mbm_counter_counts(self._file("cntrs", "0=32;2=0\n")),
            {0: 32, 2: 0})
        cases = {
            "\n": "is empty",
            "0=32;bad\n": "Malformed MBM counter entry",
            "0=32;0=1\n": "Duplicate MBM domain 0",
        }
        for content, message in cases.items():
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, message):
                    resctrl_utils.read_mbm_counter_counts(self._file("cntrs", content))

    def test_sum_l3_occupancy(self):
        """Occupancy is summed across files; non-numeric content fails."""
        files = [self._file("occ0", "1024\n"), self._file("occ1", " 2048 \n")]
        self.assertEqual(resctrl_utils.sum_l3_occupancy(files), 3072)
        self.assertEqual(resctrl_utils.sum_l3_occupancy([]), 0)
        with self.assertRaises(ValueError):
            resctrl_utils.sum_l3_occupancy([self._file("occ2", "Unavailable\n")])


if __name__ == "__main__":
    GLTTestBase.unittest.main()
