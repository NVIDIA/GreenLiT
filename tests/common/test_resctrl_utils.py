"""Unit tests for common/resctrl_utils.py."""

from contextlib import ExitStack, contextmanager
import errno
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


class TestMountOptions(GLTTestBase):
    """Known resctrl flags read from the mount table."""

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        self.mounts = Path(temp_dir.name) / "mounts"
        patcher = self.mock.patch.object(
            resctrl_utils, "PROC_MOUNTS", str(self.mounts))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_known_flags_ignore_generic_options(self):
        """Only resctrl flags are returned, from the default mount point."""
        self.mounts.write_text(
            "resctrl /mnt/rc resctrl rw,cdp 0 0\n"
            "resctrl /sys/fs/resctrl resctrl rw,relatime,cdp,debug,this_is_not_abi 0 0\n",
            encoding="utf-8")
        self.assertEqual(
            resctrl_utils.mount_options(),
            frozenset({"cdp", "debug", "this_is_not_abi"}))

    def test_plain_mount_is_an_empty_set(self):
        """A mount with no resctrl flags is present but empty."""
        self.mounts.write_text(
            "resctrl /sys/fs/resctrl resctrl rw 0 0\n", encoding="utf-8")
        self.assertEqual(resctrl_utils.mount_options(), frozenset())

    def test_absent_or_other_mount_is_none(self):
        """Another resctrl mount, or none at all, is not the default mount."""
        self.mounts.write_text(
            "resctrl /mnt/rc resctrl rw,this_is_not_abi 0 0\n", encoding="utf-8")
        self.assertIsNone(resctrl_utils.mount_options())
        self.mounts.write_text("", encoding="utf-8")
        self.assertIsNone(resctrl_utils.mount_options())

    def test_unreadable_mount_table_raises(self):
        """A missing mount table is an error, not "not mounted"."""
        with self.assertRaises(OSError):
            resctrl_utils.mount_options()

    def test_missing_options_field_raises(self):
        """A mount line without an options field is malformed."""
        self.mounts.write_text("resctrl /sys/fs/resctrl resctrl\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "has no options"):
            resctrl_utils.mount_options()


class TestIommuGroupIds(GLTTestBase):
    """Parsing of iommu_group lines in a resctrl tasks file."""

    def test_pids_and_blank_lines_are_ignored(self):
        """Only iommu_group entries are returned."""
        text = "\n1\n12\niommu_group:0\n  iommu_group:3 \n"
        self.assertEqual(resctrl_utils.parse_iommu_group_ids(text), {0, 3})
        self.assertEqual(resctrl_utils.parse_iommu_group_ids(""), set())

    def test_malformed_and_duplicate_lines_fail(self):
        """A bad id or a repeated id is an error."""
        cases = {
            "iommu_group\n": "Malformed iommu group line",
            "iommu_group:\n": "Malformed iommu group line",
            "iommu_group:foo\n": "Malformed iommu group line",
            "iommu_group:-1\n": "Malformed iommu group line",
            "iommu_group:1 extra\n": "Malformed iommu group line",
            "iommu_group:2\niommu_group:2\n": "Duplicate iommu group 2",
            "iommu_group:1\niommu_group:01\n": "Duplicate iommu group 1",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, message):
                    resctrl_utils.parse_iommu_group_ids(text)


class TestMountWithOptions(GLTTestBase):
    """Mount resctrl with required flags and restore a replaced mount."""

    def _commands(self, mock_run):
        return [call.args[0] for call in mock_run.call_args_list]

    def _mount(self, options=""):
        argv = ["mount", "-t", "resctrl", "resctrl", "/sys/fs/resctrl"]
        if options:
            argv.extend(["-o", options])
        return argv

    def _run(self, option_sequence, *, run_side_effect=None, cleanup_errors=None):
        with ExitStack() as stack:
            mock_options = stack.enter_context(self.mock.patch.object(
                resctrl_utils, "mount_options", side_effect=option_sequence))
            mock_run = stack.enter_context(self.mock.patch.object(
                resctrl_utils.glt_utils, "run_command", side_effect=run_side_effect))
            with resctrl_utils.mount_with_options(
                    self.mock_logger, ("this_is_not_abi",),
                    cleanup_errors=cleanup_errors) as state:
                pass
        return state, mock_run, mock_options

    def test_in_place_mount_is_left_alone(self):
        """An existing mount with exactly the requested flags is not cycled."""
        state, mock_run, mock_options = self._run(
            [frozenset({"this_is_not_abi"})])
        self.assertEqual(state, (True, None))
        mock_run.assert_not_called()
        self.assertEqual(mock_options.call_count, 1)

    def test_mount_containing_requested_flags_is_left_alone(self):
        """Extra flags stay mounted when every requested flag is already present."""
        state, mock_run, mock_options = self._run(
            [frozenset({"debug", "this_is_not_abi"})])
        self.assertEqual(state, (True, None))
        mock_run.assert_not_called()
        self.assertEqual(mock_options.call_count, 1)

    def test_mounts_then_unmounts_when_absent(self):
        """A helper-owned playground mount is removed and not replaced."""
        playground = frozenset({"this_is_not_abi"})
        state, mock_run, mock_options = self._run([None, playground])
        self.assertEqual(state, (True, None))
        self.assertEqual(mock_options.call_count, 2)
        self.assertEqual(
            self._commands(mock_run),
            [self._mount("this_is_not_abi"),
             ["umount", "/sys/fs/resctrl"]])

    def test_plain_mount_is_replaced_and_restored(self):
        """A mount with no resctrl flags is restored without -o."""
        playground = frozenset({"this_is_not_abi"})
        state, mock_run, _ = self._run([frozenset(), playground])
        self.assertEqual(state, (True, None))
        self.assertEqual(
            self._commands(mock_run),
            [["umount", "/sys/fs/resctrl"],
             self._mount("this_is_not_abi"),
             ["umount", "/sys/fs/resctrl"],
             self._mount()])

    def test_saved_flags_are_restored_in_sorted_order(self):
        """Previous resctrl flags are mounted again after the test mount."""
        playground = frozenset({"this_is_not_abi"})
        state, mock_run, _ = self._run(
            [frozenset({"debug", "mba_MBps"}), playground])
        self.assertEqual(state, (True, None))
        self.assertEqual(
            self._commands(mock_run)[-1], self._mount("debug,mba_MBps"))

    def test_failed_unmount_does_not_mount(self):
        """A pre-existing mount that cannot be removed is left unchanged."""
        state, mock_run, _ = self._run(
            [frozenset({"cdp"})],
            run_side_effect=RuntimeError("Failed to unmount resctrl."))
        self.assertEqual(state, (False, "Failed to unmount resctrl."))
        self.assertEqual(self._commands(mock_run), [["umount", "/sys/fs/resctrl"]])
        self.mock_logger.error.assert_called_once()

    def test_failed_mount_restores_the_previous_flags(self):
        """A rejected test mount is rolled back to the saved flags."""
        state, mock_run, _ = self._run(
            [frozenset({"cdp"})],
            run_side_effect=[
                "",
                RuntimeError("Failed to mount resctrl with this_is_not_abi."),
                "",
            ])
        self.assertEqual(state, (False, "Failed to mount resctrl with this_is_not_abi."))
        self.assertEqual(
            self._commands(mock_run),
            [["umount", "/sys/fs/resctrl"],
             self._mount("this_is_not_abi"),
             self._mount("cdp")])

    def test_failed_mount_when_absent_does_not_unmount(self):
        """Nothing is unmounted when the test mount was the first mount."""
        state, mock_run, _ = self._run(
            [None],
            run_side_effect=RuntimeError("Failed to mount resctrl with this_is_not_abi."))
        self.assertEqual(
            state, (False, "Failed to mount resctrl with this_is_not_abi."))
        self.assertEqual(
            self._commands(mock_run),
            [self._mount("this_is_not_abi")])

    def test_failed_restore_is_included_in_the_setup_error(self):
        """A mount failure that also fails to restore names both problems."""
        errors = []
        state, _, _ = self._run(
            [frozenset({"cdp"})],
            run_side_effect=[
                "",
                RuntimeError("Failed to mount resctrl with this_is_not_abi."),
                RuntimeError("Failed to restore resctrl mount with cdp."),
            ],
            cleanup_errors=errors)
        self.assertFalse(state[0])
        self.assertIn("Failed to mount resctrl with this_is_not_abi.", state[1])
        self.assertIn("Also failed to restore", state[1])
        self.assertEqual(errors, ["Failed to restore resctrl mount with cdp."])

    def test_option_mismatch_rolls_back_and_restores(self):
        """A mount that does not show the requested flags is not left active."""
        state, mock_run, _ = self._run([frozenset({"cdp"}), frozenset()])
        self.assertFalse(state[0])
        self.assertIn("do not match", state[1])
        self.assertNotIn("Also failed", state[1])
        self.assertEqual(
            self._commands(mock_run),
            [["umount", "/sys/fs/resctrl"],
             self._mount("this_is_not_abi"),
             ["umount", "/sys/fs/resctrl"],
             self._mount("cdp")])

    def test_option_mismatch_without_a_previous_mount_only_unmounts(self):
        """Rollback of the first mount does not invent a restore mount."""
        state, mock_run, _ = self._run([None, frozenset()])
        self.assertFalse(state[0])
        self.assertEqual(
            self._commands(mock_run),
            [self._mount("this_is_not_abi"),
             ["umount", "/sys/fs/resctrl"]])

    def test_rollback_unmount_failure_does_not_remount(self):
        """A busy test mount is not covered by mounting the saved flags on top."""
        errors = []
        state, mock_run, _ = self._run(
            [frozenset({"cdp"}), frozenset()],
            run_side_effect=["", "", RuntimeError("Failed to unmount resctrl.")],
            cleanup_errors=errors)
        self.assertFalse(state[0])
        self.assertIn("do not match", state[1])
        self.assertIn("Also failed to restore", state[1])
        self.assertEqual(errors, ["Failed to unmount resctrl."])
        self.assertEqual(len(self._commands(mock_run)), 3)

    def test_unreadable_mount_table_is_logged_and_nothing_is_mounted(self):
        """With the mount flags unknown, no mount is attempted."""
        state, mock_run, _ = self._run(OSError("no /proc"))
        self.assertFalse(state[0])
        self.assertIn("no /proc", state[1])
        mock_run.assert_not_called()
        self.mock_logger.error.assert_called_once()

    def test_cleanup_unmount_failure_is_logged_and_collected(self):
        """An exit unmount failure is logged and appended to cleanup_errors."""
        errors = []
        playground = frozenset({"this_is_not_abi"})
        state, _, _ = self._run(
            [None, playground],
            run_side_effect=["", RuntimeError("Failed to unmount resctrl.")],
            cleanup_errors=errors)
        self.assertEqual(state, (True, None))
        self.assertEqual(errors, ["Failed to unmount resctrl."])
        self.mock_logger.warning.assert_called_once_with("%s", "Failed to unmount resctrl.")

    def test_cleanup_restore_failure_is_logged_and_collected(self):
        """A failed restore after a successful body is collected."""
        errors = []
        playground = frozenset({"this_is_not_abi"})
        state, mock_run, _ = self._run(
            [frozenset({"cdp"}), playground],
            run_side_effect=[
                "", "", "",
                RuntimeError("Failed to restore resctrl mount with cdp."),
            ],
            cleanup_errors=errors)
        self.assertEqual(state, (True, None))
        self.assertEqual(errors, ["Failed to restore resctrl mount with cdp."])
        self.assertEqual(self._commands(mock_run)[-1], self._mount("cdp"))
        self.mock_logger.warning.assert_called_once_with(
            "%s", "Failed to restore resctrl mount with cdp.")

    def test_unknown_option_and_bare_string_are_rejected(self):
        """Requested flags are validated before any mount change."""
        with self.assertRaisesRegex(ValueError, "not-a-flag"):
            with resctrl_utils.mount_with_options(self.mock_logger, ["not-a-flag"]):
                pass
        with self.assertRaisesRegex(TypeError, "not a string"):
            with resctrl_utils.mount_with_options(
                    self.mock_logger, "this_is_not_abi"):
                pass


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


class TestMakeMonGroup(GLTTestBase):
    """Monitor-group creation that waits out resctrl RMID limbo."""

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        (self.root / "info").mkdir()
        (self.root / "mon_groups").mkdir()
        self.status = self.root / "info" / "last_cmd_status"
        self.status.write_text("Out of RMIDs\n", encoding="utf-8")
        self.group = str(self.root / "mon_groups" / "glt_probe")
        patcher = self.mock.patch.object(
            resctrl_utils, "RESCTRL_MOUNT_POINT", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)

    @contextmanager
    def _mocks(self, mkdir_effects, clock):
        """Fake mkdir outcomes and the monotonic clock; capture sleeps."""
        with ExitStack() as stack:
            mock_mkdir = stack.enter_context(self.mock.patch.object(
                resctrl_utils.os, "mkdir", side_effect=mkdir_effects))
            mock_sleep = stack.enter_context(self.mock.patch.object(
                resctrl_utils.time, "sleep"))
            stack.enter_context(self.mock.patch.object(
                resctrl_utils.time, "monotonic", side_effect=clock))
            yield mock_mkdir, mock_sleep

    def test_creates_group_without_waiting(self):
        """A free monitor ID creates the group at once with no sleep or log."""
        resctrl_utils.make_mon_group(self.group, logger=self.mock_logger)
        self.assertTrue(Path(self.group).is_dir())
        self.mock_logger.info.assert_not_called()

    def test_retries_enospc_and_ebusy_until_the_id_is_free(self):
        """Transient ENOSPC and EBUSY are retried at the poll interval and logged."""
        effects = [OSError(errno.ENOSPC, "No space left on device"),
                   OSError(errno.EBUSY, "Device or resource busy"), None]
        with self._mocks(effects, clock=[0.0, 0.5, 1.0]) as (mock_mkdir, mock_sleep):
            resctrl_utils.make_mon_group(self.group, logger=self.mock_logger)
        self.assertEqual(mock_mkdir.call_count, 3)
        self.assertEqual([call.args[0] for call in mock_sleep.call_args_list], [0.5, 0.5])
        self.assertEqual(self.mock_logger.info.call_count, 2)
        self.assertIn("Out of RMIDs", str(self.mock_logger.info.call_args_list[0]))

    def test_times_out_with_kernel_status_in_message(self):
        """A persistent ENOSPC fails after the window and names last_cmd_status."""
        effects = [OSError(errno.ENOSPC, "No space left on device")] * 3
        with self._mocks(effects, clock=[0.0, 11.0]) as (mock_mkdir, _):
            with self.assertRaises(OSError) as context:
                resctrl_utils.make_mon_group(self.group, logger=self.mock_logger)
        self.assertEqual(mock_mkdir.call_count, 2)
        self.assertEqual(context.exception.errno, errno.ENOSPC)
        self.assertIn("Out of RMIDs", str(context.exception))
        self.assertIn("10s", str(context.exception))

    def test_unreadable_status_is_reported_as_unavailable(self):
        """A missing last_cmd_status does not hide the timeout failure."""
        self.status.unlink()
        effects = [OSError(errno.ENOSPC, "No space left on device")] * 2
        with self._mocks(effects, clock=[0.0, 11.0]):
            with self.assertRaisesRegex(OSError, "last_cmd_status: unavailable"):
                resctrl_utils.make_mon_group(self.group)

    def test_other_errors_raise_immediately(self):
        """Errors other than ENOSPC and EBUSY are not retried."""
        with self._mocks([OSError(errno.EEXIST, "File exists")], clock=[]) as (_, mock_sleep):
            with self.assertRaises(OSError) as context:
                resctrl_utils.make_mon_group(self.group, logger=self.mock_logger)
        self.assertEqual(context.exception.errno, errno.EEXIST)
        mock_sleep.assert_not_called()
        self.mock_logger.info.assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
