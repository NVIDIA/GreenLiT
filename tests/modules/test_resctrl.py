# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-free contracts for the native x86 resctrl filesystem testcase."""

import builtins
from contextlib import ExitStack, contextmanager
import errno
import importlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import mock

from common import glt_core as GLTCore
from common import resctrl_utils
from greenlit import greenlit
from src.modules.cpu import glt_test_resctrl as resctrl
from src.modules.cpu.glt_test_resctrl import (
    _control_group_capacity, _resctrl_platform_support, _write_schema_once,
)
from tests.test_base import GLTTestBase

SCHEMA = b"   L3:0=ffff;2=00ff\n   MB:0=100;2=100\n SMBA:0=2048\n"
GROUP_NAME = "glt_resctrl_0123456789abcdef"


class ResctrlFixture(GLTTestBase):
    """Model only kernel file creation, writes, rmdir and mount command effects."""

    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(TemporaryDirectory()))
        self.state = {
            "mounted": False, "phase": "precheck", "read_errors": {},
            "mount_error": None, "unmount_error": None, "confirm": True,
            "missing_cli": None, "residual": "", "borrowed": False,
            "mkdir_error": None, "rmdir_error": None,
            "invalid_errno": errno.EINVAL, "accepted_invalid": False,
            "mutate_invalid": False, "mutate_valid": False, "short": False,
            "group_schema": SCHEMA, "config_error": None, "file_errors": {},
        }
        self.events = []
        self.descriptors = {}
        self.real = {
            "mkdir": Path.mkdir, "rmdir": Path.rmdir, "path_open": Path.open, "iterdir": Path.iterdir,
            "open": os.open, "close": os.close, "builtin_open": builtins.open,
            "managed_mount": resctrl_utils.managed_mount,
        }
        self.populate_root()
        patches = (
            mock.patch.object(resctrl_utils, "managed_mount", side_effect=self.managed_mount),
            mock.patch.object(resctrl.platform, "machine", return_value="x86_64"),
            mock.patch.object(resctrl.platform, "release", return_value="unit-kernel"),
            mock.patch.object(resctrl, "CPUINFO_PATH", self.root / "cpuinfo"),
            mock.patch.object(resctrl.uuid, "uuid4", return_value=mock.Mock(hex="0123456789abcdef")),
            mock.patch.multiple(resctrl_utils, RESCTRL_MOUNT_POINT=str(self.root / "resctrl"),
                                PROC_MOUNTS=str(self.root / "mounts"),
                                PROC_FILESYSTEMS=str(self.root / "filesystems")),
            mock.patch.object(resctrl_utils.glt_utils, "is_cli_available",
                              side_effect=self.cli_available),
            mock.patch.object(resctrl_utils.glt_utils, "run_command", side_effect=self.command),
            mock.patch.object(builtins, "open", side_effect=self.open_builtin),
            mock.patch.object(Path, "open", autospec=True, side_effect=self.open_path),
            mock.patch.object(Path, "iterdir", autospec=True, side_effect=self.iterdir),
            mock.patch.object(Path, "mkdir", autospec=True, side_effect=self.mkdir),
            mock.patch.object(Path, "rmdir", autospec=True, side_effect=self.rmdir),
            mock.patch.object(resctrl.os, "open", side_effect=self.open_descriptor),
            mock.patch.object(resctrl.os, "write", side_effect=self.write_descriptor),
            mock.patch.object(resctrl.os, "close", side_effect=self.close_descriptor),
        )
        for patcher in patches:
            stack.enter_context(patcher)

    @contextmanager
    def managed_mount(self, logger, *, cleanup_errors):
        """Inject faults around the real manager, including races after precheck."""
        self.state["phase"] = "manager-entry"
        try:
            with self.real["managed_mount"](logger, cleanup_errors=cleanup_errors) as state:
                yield state
        finally:
            self.state["phase"] = "final"

    def populate_root(self):
        """Prepare allocation files before installing kernel boundary patches."""
        mount = self.root / "resctrl"
        mount.mkdir()
        for name in ("L3", "MB", "SMBA"):
            info = mount / "info" / name
            info.mkdir(parents=True)
            (info / "num_closids").write_text("16\n", encoding="ascii")
        (mount / "schemata").write_bytes(SCHEMA)
        (mount / "tasks").write_bytes(b"")
        (mount / "cpus").write_bytes(b"ff\n")
        (self.root / "config").write_text(
            "CONFIG_X86_CPU_RESCTRL=y\nCONFIG_RESCTRL_FS=y\n", encoding="ascii")
        (self.root / "cpuinfo").write_text("processor : 0\nflags : fpu rdt_a mba\n", encoding="ascii")
        (self.root / "filesystems").write_text("nodev\tresctrl\n", encoding="ascii")
        (self.root / "mounts").write_text("", encoding="ascii")

    def open_builtin(self, path, *args, **kwargs):
        """Redirect the exact release config and guard proc reads."""
        if str(path) == "/boot/config-unit-kernel":
            error = self.state["config_error"]
            if error:
                raise error
            path = self.root / "config"
        self.assertTrue(Path(path).is_relative_to(self.root), path)
        if Path(path) == self.root / "mounts":
            error = self.state["read_errors"].get(self.state["phase"])
            if error:
                raise error
            if self.state["borrowed"] and self.state["phase"] == "manager-entry":
                (self.root / "mounts").write_text(
                    f"resctrl {self.root / 'resctrl'} resctrl rw 0 0\n", encoding="ascii")
        return self.real["builtin_open"](path, *args, **kwargs)

    def open_path(self, path, *args, **kwargs):
        """Keep all pathname I/O under the fake root and inject read errors."""
        self.assertTrue(path.is_relative_to(self.root), path)
        error = self.state["file_errors"].get(path)
        mode = args[0] if args else kwargs.get("mode", "r")
        if error and "r" in mode:
            raise error
        return self.real["path_open"](path, *args, **kwargs)

    def iterdir(self, path):
        """Directory read errors are kernel evidence errors too."""
        self.assertTrue(path.is_relative_to(self.root), path)
        error = self.state["file_errors"].get(path)
        if error:
            raise error
        return self.real["iterdir"](path)

    def cli_available(self, cli):
        """No real PATH lookup is needed."""
        return cli != self.state["missing_cli"]

    def command(self, argv, _failure_message):
        """Simulate exactly the shared manager's mount and unmount argv."""
        mount = self.root / "resctrl"
        self.events.append(("command", argv))
        if argv == ["mount", "-t", "resctrl", "resctrl", str(mount)]:
            error = self.state["mount_error"]
            if error:
                raise error
            self.state["mounted"] = True
            self.state["phase"] = "confirm"
            content = f"resctrl {mount} resctrl rw 0 0\n" if self.state["confirm"] else ""
        else:
            self.assertEqual(argv, ["umount", str(mount)])
            error = self.state["unmount_error"]
            if error:
                raise error
            self.state["mounted"] = False
            content = self.state["residual"]
        (self.root / "mounts").write_text(content, encoding="ascii")
        return ""

    def mkdir(self, path, *args, **kwargs):
        """A successful group mkdir populates virtual kernel files."""
        self.assertEqual(path, self.root / "resctrl" / GROUP_NAME)
        self.events.append(("mkdir", path))
        error = self.state["mkdir_error"]
        if error:
            raise error
        self.real["mkdir"](path, *args, **kwargs)
        (path / "schemata").write_bytes(self.state["group_schema"])
        (path / "tasks").write_bytes(b"")
        (path / "cpus").write_bytes(b"0\n")

    def rmdir(self, path):
        """Virtual group removal is distinct from ordinary populated-dir rmdir."""
        self.assertEqual(path, self.root / "resctrl" / GROUP_NAME)
        self.events.append(("rmdir", path))
        error = self.state["rmdir_error"]
        if error:
            raise error
        for name in ("schemata", "tasks", "cpus"):
            (path / name).unlink(missing_ok=True)
        self.real["rmdir"](path)

    def open_descriptor(self, path, flags):
        """A schema write must never create or truncate a file."""
        self.assertEqual(Path(path), self.root / "resctrl" / GROUP_NAME / "schemata")
        self.assertEqual(flags, os.O_WRONLY)
        descriptor = self.real["open"](path, flags)
        self.descriptors[descriptor] = Path(path)
        return descriptor

    def write_descriptor(self, descriptor, payload):
        """Model whole-buffer kernel acceptance/rejection without ordinary writes."""
        path = self.descriptors[descriptor]
        self.events.append(("write", path, payload))
        if self.state["short"]:
            return len(payload) - 1
        if payload == resctrl.INVALID_SCHEMA:
            if self.state["mutate_invalid"]:
                path.write_bytes(b"MB:0=1\n")
            if not self.state["accepted_invalid"]:
                raise OSError(self.state["invalid_errno"], "kernel rejected schema")
        elif self.state["mutate_valid"]:
            path.write_bytes(b"MB:0=1\n")
        return len(payload)

    def close_descriptor(self, descriptor):
        """Every opened schema descriptor must close even after write failure."""
        self.events.append(("close", self.descriptors.pop(descriptor)))
        self.real["close"](descriptor)

    def assert_clean(self, *, commands=True):
        """Qualification requires owned group and mount cleanup after the body."""
        self.assertFalse((self.root / "resctrl" / GROUP_NAME).exists())
        self.assertFalse(self.descriptors)
        if commands:
            self.assertEqual([item[1][0] for item in self.events if item[0] == "command"],
                             ["mount", "umount"])
            self.assertFalse(self.state["mounted"])


class TestResctrlApplicability(ResctrlFixture):
    """Applicability is factual, with read failures distinct from unsupported."""

    def test_unsupported_architecture_skips_without_effects(self):
        """ARM64 must not borrow the x86 allocation contract."""
        with mock.patch.object(resctrl.platform, "machine", return_value="aarch64"):
            with self.assertRaisesRegex(GLTCore.GLTTestSkip, "aarch64"):
                self.state["phase"] = "precheck"
                resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_config_values_and_missing_keys_are_distinguished(self):
        """The parser cannot distinguish not-set comments from absent keys."""
        for content, reason in (
                ("CONFIG_X86_CPU_RESCTRL=n\nCONFIG_RESCTRL_FS=y\n", "CONFIG_X86_CPU_RESCTRL=n"),
                ("CONFIG_X86_CPU_RESCTRL=y\nCONFIG_RESCTRL_FS=n\n", "CONFIG_RESCTRL_FS=n"),
                ("# CONFIG_X86_CPU_RESCTRL is not set\n", "absent-or-unset")):
            with self.subTest(content=content):
                (self.root / "config").write_text(content, encoding="ascii")
                with self.assertRaisesRegex(GLTCore.GLTTestSkip, reason):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_missing_config_skips_but_unreadable_config_fails(self):
        """Only the exact expected config-file absence is inapplicability."""
        for error, outcome in ((FileNotFoundError("missing config"), GLTCore.GLTTestSkip),
                               (PermissionError("unreadable config"), GLTCore.GLTTestFail)):
            with self.subTest(error=error):
                self.state["config_error"] = error
                with self.assertRaisesRegex(outcome, "config"):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_missing_allocation_flag_skips_but_unparseable_flags_fail(self):
        """Absent rdt_a is unsupported; a malformed evidence file is a failure."""
        for text, outcome, reason in (("flags : fpu mba\n", GLTCore.GLTTestSkip, "rdt_a"),
                                      ("processor : 0\n", GLTCore.GLTTestFail, "flags")):
            with self.subTest(text=text):
                (self.root / "cpuinfo").write_text(text, encoding="ascii")
                with self.assertRaisesRegex(outcome, reason):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_missing_or_unreadable_flags_fail(self):
        """Flags read errors must never be mislabeled missing config."""
        for error in (FileNotFoundError("missing CPU flags"), PermissionError("CPU flags denied")):
            with self.subTest(error=error):
                self.state["file_errors"][self.root / "cpuinfo"] = error
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "CPU flags"):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_platform_support_uses_matching_config_and_exact_flag(self):
        """The existing parser and the literal rdt_a token establish eligibility."""
        self.assertEqual(_resctrl_platform_support(), (True, ""))
        (self.root / "cpuinfo").write_text("flags : rdt_a_fake\n", encoding="ascii")
        supported, reason = _resctrl_platform_support()
        self.assertFalse(supported)
        self.assertIn("rdt_a", reason)

    def test_unregistered_filesystem_fails(self):
        """An eligible kernel must register the required filesystem."""
        (self.root / "filesystems").write_text("nodev\tsysfs\n", encoding="ascii")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "register"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_support_or_precheck_mount_read_errors_fail_without_effects(self):
        """Unavailable proc evidence is not proof of unsupported or unmounted state."""
        (self.root / "filesystems").unlink()
        with self.assertRaises(GLTCore.GLTTestFail):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        (self.root / "filesystems").write_text("nodev\tresctrl\n", encoding="ascii")
        self.state["read_errors"] = {"precheck": PermissionError("mount table denied")}
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "mount table denied"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_existing_mount_skips_without_borrowed_mutation(self):
        """Any existing resctrl mount prevents a dedicated transaction."""
        (self.root / "mounts").write_text("resctrl /borrowed resctrl rw 0 0\n", encoding="ascii")
        with self.assertRaisesRegex(GLTCore.GLTTestSkip, "mount"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])


class TestResctrlLifecycle(ResctrlFixture):
    """Exercise the real manager and group lifecycle with kernel-boundary mocks."""

    def test_eligible_case_exercises_group_and_cleanup(self):
        """A no-op PASS cannot qualify without valid and rejected write effects."""
        self.state["phase"] = "precheck"
        resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        writes = [event[2] for event in self.events if event[0] == "write"]
        self.assertEqual(writes, [SCHEMA, resctrl.INVALID_SCHEMA])
        self.assertEqual([event[0] for event in self.events],
                         ["command", "mkdir", "write", "close", "write", "close", "rmdir", "command"])
        self.assertEqual((self.root / "resctrl" / "schemata").read_bytes(), SCHEMA)
        self.assert_clean()

    def test_missing_cli_skips_without_commands_after_final_check(self):
        """Both CLI prerequisites must exist before any mount effect."""
        for cli in ("mount", "umount"):
            with self.subTest(cli=cli):
                self.state["missing_cli"] = cli
                with self.assertRaisesRegex(GLTCore.GLTTestSkip, cli):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])

    def test_mount_failure_fails_without_group_or_unmount(self):
        """A failed mount command cannot become a successful empty case."""
        self.state["mount_error"] = RuntimeError("mount failed")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "mount"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual([item[0] for item in self.events], ["command"])

    def test_unconfirmed_mount_fails_and_still_unmounts(self):
        """A zero-exit mount without observable mount state cannot qualify."""
        self.state["confirm"] = False
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "mount"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()

    def test_capacity_one_skips_only_after_clean_teardown(self):
        """Root plus one owned control group needs two advertised CLOSIDs."""
        (self.root / "resctrl" / "info" / "SMBA" / "num_closids").write_text("1\n", encoding="ascii")
        with self.assertRaisesRegex(GLTCore.GLTTestSkip, "CLOSID|capacity"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()
        self.assertNotIn("mkdir", [item[0] for item in self.events])

    def test_missing_or_empty_root_schema_fails_after_eligibility(self):
        """No allocation schema is a broken eligible ABI, not inapplicability."""
        path = self.root / "resctrl" / "schemata"
        path.unlink()
        with self.assertRaises(GLTCore.GLTTestFail):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()
        self.events.clear()
        path.write_bytes(b"")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "empty|Empty"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()

    def test_group_schema_must_be_nonempty_and_newline_terminated(self):
        """The exact buffer must be a complete kernel transaction."""
        self.state["group_schema"] = b"L3:0=ffff"
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "newline|schema"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()

    def test_invalid_resource_acceptance_fails(self):
        """The runtime contract requires actual EINVAL from the unknown resource."""
        self.state["accepted_invalid"] = True
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "accepted|EINVAL") as result:
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertNotIn("transaction failed:", str(result.exception))
        self.assert_clean()

    def test_unexpected_errno_fails(self):
        """EACCES is not evidence that the parser rejected an unknown resource."""
        self.state["invalid_errno"] = errno.EACCES
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "13|Permission|EINVAL"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()

    def test_changed_schema_after_rejection_fails(self):
        """A rejected transaction must leave the previously valid schema unchanged."""
        self.state["mutate_invalid"] = True
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "changed|unchanged|mismatch"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()

    def test_changed_valid_roundtrip_fails(self):
        """Exact hexadecimal and padded raw bytes must survive a valid write."""
        self.state["mutate_valid"] = True
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "changed|roundtrip|mismatch"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()

    def test_short_write_fails_without_retry(self):
        """Appending a partial transaction must never stand in for one full write."""
        self.state["short"] = True
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "Short|short"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(len([event for event in self.events if event[0] == "write"]), 1)
        self.assert_clean()

    def test_collision_is_not_removed_or_retried(self):
        """Failed mkdir never grants ownership of a pre-existing group."""
        group = self.root / "resctrl" / GROUP_NAME
        self.real["mkdir"](group)
        marker = group / "untouched"
        marker.write_bytes(b"borrowed group")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "exist"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(marker.read_bytes(), b"borrowed group")
        self.assertNotIn("rmdir", [event[0] for event in self.events])
        self.assertEqual(len([event for event in self.events if event[0] == "mkdir"]), 1)

    def test_body_and_cleanup_errors_are_both_visible(self):
        """Group cleanup must not replace the parser's primary failure."""
        self.state.update(accepted_invalid=True, rmdir_error=PermissionError("rmdir denied"))
        with self.assertRaises(GLTCore.GLTTestFail) as result:
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertRegex(str(result.exception), "accepted|EINVAL")
        self.assertIn("rmdir denied", str(result.exception))
        self.assertEqual([item[1][0] for item in self.events if item[0] == "command"],
                         ["mount", "umount"])

    def test_rmdir_failure_cannot_pass(self):
        """A successful body is insufficient when owned cleanup fails."""
        self.state["rmdir_error"] = PermissionError("rmdir denied")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "rmdir denied"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)


class TestResctrlFinalState(ResctrlFixture):
    """Final state checks run after success, body failure and deferred SKIP."""

    def test_unmount_error_overrides_body_success_and_capacity_skip(self):
        """Neither successful operations nor inapplicability hides failed teardown."""
        self.state["unmount_error"] = RuntimeError("unmount failed")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "unmount failed"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        (self.root / "mounts").write_text("", encoding="ascii")
        (self.root / "resctrl" / "info" / "L3" / "num_closids").write_text("1\n", encoding="ascii")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "unmount failed"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)

    def test_residual_mount_fails_after_each_body_outcome(self):
        """A final borrowed mount must fail qualification without another umount."""
        self.state["residual"] = "resctrl /borrowed resctrl rw 0 0\n"
        for outcome in ("success", "failure", "skip"):
            with self.subTest(outcome=outcome):
                (self.root / "mounts").write_text("", encoding="ascii")
                self.state.update(accepted_invalid=outcome == "failure")
                (self.root / "resctrl" / "info" / "L3" / "num_closids").write_text(
                    "1\n" if outcome == "skip" else "16\n", encoding="ascii")
                self.events.clear()
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "/borrowed"):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
                self.assert_clean()

    def test_final_mount_read_error_fails_after_each_body_outcome(self):
        """Unknown final state overrides success, failure and both deferred skips."""
        self.state["read_errors"] = {"final": PermissionError("final mounts denied")}
        for outcome in ("success", "failure", "skip", "missing-cli", "mount-failure"):
            with self.subTest(outcome=outcome):
                self.state.update(
                    accepted_invalid=outcome == "failure",
                    missing_cli="mount" if outcome == "missing-cli" else None,
                    mount_error=RuntimeError("mount failed") if outcome == "mount-failure" else None)
                (self.root / "resctrl" / "info" / "L3" / "num_closids").write_text(
                    "1\n" if outcome == "skip" else "16\n", encoding="ascii")
                self.events.clear()
                with self.assertRaisesRegex(
                        GLTCore.GLTTestFail,
                        "Cannot verify final resctrl mount state: final mounts denied") as result:
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
                if outcome == "missing-cli":
                    self.assertFalse([event for event in self.events if event[0] == "command"])
                elif outcome == "mount-failure":
                    self.assertIn("Cannot mount", str(result.exception))
                    self.assertEqual([event[0] for event in self.events], ["command"])
                else:
                    self.assert_clean()

    def test_borrowed_race_is_never_unmounted_and_cannot_qualify(self):
        """A precheck is not a lock; final borrowed state prevents PASS."""
        self.state["borrowed"] = True
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "mount"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertFalse([event for event in self.events if event[0] == "command"])
        self.assertFalse((self.root / "resctrl" / GROUP_NAME).exists())

    def test_manager_mount_read_errors_fail_without_false_pass(self):
        """Initial manager read and post-command confirmation are required evidence."""
        self.state["read_errors"] = {"manager-entry": PermissionError("manager mounts denied")}
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "mount"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assertEqual(self.events, [])
        self.state.update(read_errors={"confirm": PermissionError("confirm mounts denied")})
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "mount"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()


class TestResctrlCapacity(ResctrlFixture):
    """Metadata names are stripped; raw hexadecimal schema values are preserved."""

    def test_padded_resource_names_use_corresponding_info_capacity(self):
        """Capacity is the smallest resource limit, without parsing mask values."""
        (self.root / "resctrl" / "info" / "SMBA" / "num_closids").write_text("3\n", encoding="ascii")
        self.assertEqual(_control_group_capacity(self.root / "resctrl"), 3)
        self.assertEqual((self.root / "resctrl" / "schemata").read_bytes(), SCHEMA)

    def test_malformed_schema_and_nonpositive_capacity_are_errors(self):
        """Malformed eligible ABI cannot silently become insufficient capacity."""
        for schema in (b"", b"\n", b"missing-colon\n", b" :0=1\n"):
            with self.subTest(schema=schema):
                (self.root / "resctrl" / "schemata").write_bytes(schema)
                with self.assertRaises(ValueError):
                    _control_group_capacity(self.root / "resctrl")
        (self.root / "resctrl" / "schemata").write_bytes(SCHEMA)
        for value in ("0", "-1", "bad"):
            with self.subTest(value=value):
                (self.root / "resctrl" / "info" / "MB" / "num_closids").write_text(value, encoding="ascii")
                with self.assertRaises(ValueError):
                    _control_group_capacity(self.root / "resctrl")

    def test_required_root_files_and_resource_metadata_errors_fail(self):
        """All required ABI files must be readable on an eligible mounted system."""
        for relative in ("info", "schemata", "tasks", "cpus", "info/L3/num_closids"):
            with self.subTest(relative=relative):
                path = self.root / "resctrl" / relative
                self.state["file_errors"] = {path: PermissionError(f"{relative} denied")}
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "denied"):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
                self.assert_clean()
                self.events.clear()

    def test_group_read_failures_still_remove_owned_group(self):
        """A read failure after ownership does not skip the group finally."""
        path = self.root / "resctrl" / GROUP_NAME / "tasks"
        self.state["file_errors"][path] = PermissionError("group tasks denied")
        with self.assertRaisesRegex(GLTCore.GLTTestFail, "group tasks denied"):
            self.state["phase"] = "precheck"
            resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
        self.assert_clean()

    def test_blank_schema_lines_preserve_capacity_and_raw_bytes(self):
        """Ignored blank lines cannot alter allocation limits or rewrite masks."""
        for schema in (b"\n" + SCHEMA, SCHEMA.replace(b"\n", b"\n\n", 1), SCHEMA + b"\n"):
            with self.subTest(schema=schema):
                (self.root / "resctrl" / "schemata").write_bytes(schema)
                self.assertEqual(_control_group_capacity(self.root / "resctrl"), 16)
                self.assertEqual((self.root / "resctrl" / "schemata").read_bytes(), schema)

    def test_late_schema_or_metadata_error_cannot_qualify(self):
        """A valid prefix must not hide a later definitive allocation ABI error."""
        for suffix in (b"malformed\n", b"UNKNOWN:0=1\n"):
            with self.subTest(suffix=suffix):
                (self.root / "resctrl" / "schemata").write_bytes(SCHEMA + suffix)
                with self.assertRaises(GLTCore.GLTTestFail):
                    self.state["phase"] = "precheck"
                    resctrl.test_resctrl_filesystem_group_lifecycle(None, self.mock_logger)
                self.assert_clean()
                self.events.clear()


class TestResctrlWrite(GLTTestBase):
    """The ABI write is one unbuffered syscall with unconditional close."""

    def test_one_full_write_uses_write_only_existing_file(self):
        """No O_CREAT/O_TRUNC or split buffered writes are permitted."""
        with mock.patch.object(resctrl.os, "open", return_value=123) as opened, \
                mock.patch.object(resctrl.os, "write", return_value=len(SCHEMA)) as written, \
                mock.patch.object(resctrl.os, "close") as closed:
            _write_schema_once(Path("owned/schemata"), SCHEMA)
        opened.assert_called_once_with(Path("owned/schemata"), os.O_WRONLY)
        written.assert_called_once_with(123, SCHEMA)
        closed.assert_called_once_with(123)

    def test_short_write_and_errno_close_without_retry(self):
        """Short writes and kernel errno preserve transaction failure and close."""
        for effect, error in ((len(SCHEMA) - 1, ValueError), (OSError(errno.EINVAL, "invalid"), OSError)):
            with self.subTest(effect=effect):
                with mock.patch.object(resctrl.os, "open", return_value=123), \
                        mock.patch.object(resctrl.os, "write", side_effect=[effect]) as written, \
                        mock.patch.object(resctrl.os, "close") as closed:
                    with self.assertRaises(error):
                        _write_schema_once(Path("owned/schemata"), SCHEMA)
                written.assert_called_once_with(123, SCHEMA)
                closed.assert_called_once_with(123)


class TestResctrlDiscovery(GLTTestBase):
    """Bounded native discovery checks metadata without running a hardware body."""

    def test_cpu_case_is_discoverable_in_standard_and_stress(self):
        """The new runtime coverage must be schedulable in both approved plans."""
        module_name = "glt_test_resctrl"
        missing = object()
        previous = sys.modules.get(module_name, missing)
        try:
            with mock.patch.object(greenlit.glob, "glob", return_value=[resctrl.__file__]):
                for plan in ("standard", "stress"):
                    with self.subTest(plan=plan):
                        cases, error, code = greenlit.discover_target_test_cases(
                            ["cpu"], ["cpu:test_resctrl_filesystem_group_lifecycle"], {plan})
                        self.assertEqual((code, error), (0, ""))
                        self.assertEqual([GLTCore.GLTScheduler.get_test_case_name(case) for case in cases],
                                         ["cpu:test_resctrl_filesystem_group_lifecycle"])
        finally:
            if previous is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

    def test_import_has_no_hardware_or_process_io(self):
        """Discovery may import this module before eligibility is known."""
        with mock.patch.object(builtins, "open", side_effect=AssertionError("import file I/O")), \
                mock.patch.object(Path, "open", side_effect=AssertionError("import path I/O")), \
                mock.patch.object(os, "open", side_effect=AssertionError("import descriptor I/O")), \
                mock.patch.object(resctrl.platform, "machine", side_effect=AssertionError("import platform")), \
                mock.patch.object(resctrl_utils.glt_utils, "run_command",
                                  side_effect=AssertionError("import command")):
            importlib.reload(resctrl)
