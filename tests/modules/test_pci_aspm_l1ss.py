# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the paired L1SS testcase through mocked kernel interfaces."""
# pylint: disable=wrong-import-position

from contextlib import ExitStack, contextmanager, nullcontext
import errno
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from modules.pci import glt_test_pci_aspm_l1ss as l1ss


MODES = ("parent_unsaved", "parent_saved_first")


def _probe_data(mode, value=0x1030f, **overrides):
    """Describe both saved buffers refreshed from readable hardware controls."""
    data = {"mode": mode, "child": "0000:01:00.0", "parent": "0000:00:01.0",
            "capture_error": 0, "restore_error": 0}
    for device in ("child", "parent"):
        for control in (1, 2):
            for phase in ("before", "saved", "after"):
                data[f"{device}_ctl{control}_{phase}"] = value
    data.update(overrides)
    return data


def _probe_text(data):
    """Serialize the observations returned by the helper's proc interface."""
    return "".join(f"{key}={value}\n" for key, value in data.items())


class TestPciAspmL1ssEntrypoint(GLTTestBase):
    """Validate verdicts and the shared build/load/cleanup lifecycle."""

    @contextmanager
    def _environment(self, target_mode=MODES[0], data=None):
        """Run real testcase control flow and inject faults at a selected stage."""
        with ExitStack() as stack:
            patch = stack.enter_context
            directory = Path(patch(tempfile.TemporaryDirectory()))
            for filename in ("Makefile", f"{l1ss.L1SS_PROBE_MODULE}.c"):
                (directory / filename).touch()
            patch(self.mock.patch.object(l1ss, "L1SS_PROBE_DIR", directory))
            proc = patch(self.mock.patch.object(l1ss, "L1SS_PROBE_PROC"))
            sysfs = patch(self.mock.patch.object(l1ss, "L1SS_PROBE_SYSFS"))
            commands = patch(self.mock.patch.object(l1ss.glt_utils, "run_command"))
            load = patch(self.mock.patch.object(l1ss.subprocess, "run"))
            sign = patch(self.mock.patch.object(l1ss.glt_utils, "sign_kernel_module"))
            patch(self.mock.patch.object(l1ss.glt_utils, "is_cli_available", return_value=True))
            config = patch(self.mock.patch.object(l1ss.check_kernel_config, "parse_kconfig",
                          return_value=dict.fromkeys(l1ss.L1SS_REQUIRED_CONFIGS, "y")))
            patch(self.mock.patch.object(l1ss.platform, "release", return_value="fixture-kernel"))
            patch(self.mock.patch.object(Path, "is_dir", return_value=True))
            patch(self.mock.patch.object(l1ss.glt_utils, "chdir_managed", return_value=nullcontext()))
            native = SimpleNamespace(
                proc=proc, sysfs=sysfs, commands=commands, load=load, sign=sign,
                config=config, directory=directory, mode=None, present=False,
                errors={}, load_visible=False, sign_result=True, events=[],
                report=None if data is None else _probe_text(data), reports={},
            )

            def fail_at(stage):
                if (stage in native.errors
                        and (stage in ("rmmod", "clean") or native.mode in (None, target_mode))):
                    raise native.errors[stage]

            def command(argv, *_args, **_kwargs):
                native.events.append(tuple(argv))
                stage = "clean" if argv == ["make", "clean"] else argv[0]
                fail_at(stage)
                if stage == "rmmod":
                    native.present = False

            def insmod(argv, **_kwargs):
                native.events.append(tuple(argv))
                native.present = native.load_visible
                fail_at("load")
                native.present = True

            def signing(*_args):
                fail_at("sign")
                return native.sign_result

            def read_report(**_kwargs):
                native.events.append(("read", native.mode))
                fail_at("read")
                if native.mode in native.reports:
                    return native.reports[native.mode]
                if native.mode == target_mode and native.report is not None:
                    return native.report
                return _probe_text(_probe_data(native.mode))

            def write_mode(value):
                self.assertIn(value, [mode + "\n" for mode in MODES])
                native.mode = value.rstrip("\n")
                native.events.append(("write", native.mode))
                fail_at("write")

            commands.side_effect = command
            load.side_effect = insmod
            sign.side_effect = signing
            proc.exists.return_value = False
            sysfs.exists.side_effect = lambda: native.present
            native.read = self.mock.Mock(side_effect=read_report)
            native.write = self.mock.Mock(side_effect=write_mode)
            original_open = open

            def open_file(path, file_mode="r", **kwargs):
                if path is not proc:
                    return original_open(path, file_mode, **kwargs)
                self.assertEqual(kwargs, {"encoding": "utf-8"})
                handle = self.mock.MagicMock()
                handle.__enter__.return_value = handle
                if file_mode == "r":
                    handle.read = native.read
                else:
                    self.assertEqual(file_mode, "w")
                    handle.write = native.write
                return handle

            patch(self.mock.patch("builtins.open", side_effect=open_file))
            yield native

    def _assert_lifecycle(self, native, modes=MODES, unload=True):
        """Require one shared build/load, immediate per-mode reads, and final cleanup."""
        expected = [("make",), ("insmod", f"{l1ss.L1SS_PROBE_MODULE}.ko")]
        for mode in modes:
            expected.extend((("write", mode), ("read", mode)))
        if unload:
            expected.append(("rmmod", l1ss.L1SS_PROBE_MODULE))
        expected.append(("make", "clean"))
        self.assertEqual(native.events, expected)
        native.load.assert_called_once_with(
            ["insmod", f"{l1ss.L1SS_PROBE_MODULE}.ko"],
            check=True, capture_output=True, text=True, timeout=30)

    def test_capture_errno_mapping_after_validating_restoration(self):
        """Eligibility skips, allocation warns, and unexpected capture errors fail."""
        for mode in MODES:
            for code, exception in ((errno.ENODEV, GLTCore.GLTTestSkip),
                                    (errno.EOPNOTSUPP, GLTCore.GLTTestSkip),
                                    (errno.EBUSY, GLTCore.GLTTestSkip),
                                    (errno.ENOMEM, GLTCore.GLTTestWarn),
                                    (errno.EIO, GLTCore.GLTTestFail)):
                with self.subTest(mode=mode, errno=code), self._environment(mode) as native:
                    native.errors["write"] = OSError(code, "capture unavailable")
                    native.report = _probe_text(_probe_data(mode, capture_error=-code))
                    with self.assertRaisesRegex(exception, mode):
                        l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                    self._assert_lifecycle(native, MODES[:MODES.index(mode) + 1])

    def test_capture_error_without_write_errno_fails(self):
        """A failed capture cannot pass even if the proc write itself succeeded."""
        for mode in MODES:
            with self.subTest(mode=mode), self._environment(mode) as native:
                native.report = _probe_text(_probe_data(mode, capture_error=-errno.EIO))
                with self.assertRaisesRegex(GLTCore.GLTTestFail, mode):
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                self._assert_lifecycle(native, MODES[:MODES.index(mode) + 1])

    def test_cleanup_failure_overrides_primary_outcome(self):
        """Attempt unload and make clean, and preserve the reason cleanup overrides."""
        for code in (None, errno.ENODEV, errno.ENOMEM, errno.EIO):
            with self.subTest(errno=code), self._environment() as native:
                if code is not None:
                    native.errors["write"] = OSError(code, "primary capture result")
                native.errors.update(rmmod=RuntimeError("unload error"),
                                     clean=RuntimeError("build clean error"))
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "Cleanup failed") as caught:
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                self.assertIn("unload error", str(caught.exception))
                self.assertIn("build clean error", str(caught.exception))
                if code is not None:
                    self.assertIn("primary capture result", str(caught.exception))
                self._assert_lifecycle(native, MODES if code is None else MODES[:1])

    def test_definite_load_failure_does_not_remove_unowned_helper(self):
        """A definite failed insmod does not own a helper that appears concurrently."""
        for error in (subprocess.CalledProcessError(1, ["insmod"]), OSError("load failed")):
            with self.subTest(error=type(error)), self._environment() as native:
                native.load_visible = True
                native.errors["load"] = error
                with self.assertRaises(GLTCore.GLTTestWarn):
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                self.assertTrue(native.present)
                self._assert_lifecycle(native, modes=(), unload=False)

    def test_fresh_buffers_pass_in_both_modes_with_one_build_and_load(self):
        """Both modes refresh readable controls, including an initially zero cache."""
        for value in (0, 0x1030f):
            with self.subTest(value=value), self._environment() as native:
                def read(_native=native, _value=value, **_kwargs):
                    _native.events.append(("read", _native.mode))
                    return _probe_text(_probe_data(_native.mode, value=_value))

                native.read.side_effect = read
                l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                native.config.assert_called_once_with("/boot/config-fixture-kernel")
                native.sign.assert_called_once()
                self._assert_lifecycle(native)
                self.assertEqual(native.write.call_args_list,
                                 [self.mock.call(mode + "\n") for mode in MODES])
                self.assertEqual(native.read.call_count, 2)
                self.assertFalse(native.present)

    def test_malformed_or_invalid_report_fails_before_next_mode(self):
        """Bad in-tree evidence is a failure rather than an optional-platform skip."""
        for mode in MODES:
            good = _probe_data(mode)
            invalid = [
                _probe_text(good) + "mode=duplicate\n", _probe_text(good) + "unknown=1\n",
                "missing separator\n", _probe_text({k: v for k, v in good.items() if k != "child"}),
            ]
            for key, value in (("mode", "other_mode"), ("child", "invalid"),
                               ("parent", good["child"]), ("capture_error", 2**31),
                               ("restore_error", -(2**31)-1), ("capture_error", "invalid"),
                               ("parent_ctl1_saved", -1), ("parent_ctl1_saved", 2**32),
                               ("parent_ctl1_after", 0), ("parent_ctl1_before", 0xffffffff)):
                invalid.append(_probe_text({**good, key: value}))
            for number, report in enumerate(invalid):
                with self.subTest(mode=mode, report=number), self._environment(mode) as native:
                    native.report = report
                    with self.assertRaisesRegex(GLTCore.GLTTestFail, mode):
                        l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                    self._assert_lifecycle(native, MODES[:MODES.index(mode) + 1])

    def test_missing_packaged_source_fails_before_build(self):
        """A broken installation is distinct from optional unsupported hardware."""
        for source in ("Makefile", f"{l1ss.L1SS_PROBE_MODULE}.c"):
            with self.subTest(source=source), self._environment() as native:
                (native.directory / source).unlink()
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "source is missing"):
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                native.commands.assert_not_called()
                native.load.assert_not_called()

    def test_preexisting_module_or_proc_is_left_untouched(self):
        """Never reuse or remove another execution's module or proc path."""
        for visible in ("sysfs", "proc"):
            with self.subTest(visible=visible), self._environment() as native:
                if visible == "sysfs":
                    native.present = True
                else:
                    native.proc.exists.return_value = True
                with self.assertRaisesRegex(GLTCore.GLTTestFail, "already present"):
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                native.commands.assert_not_called()
                native.load.assert_not_called()

    def test_read_error_fails_and_cleans_up(self):
        """A helper that cannot report evidence cannot pass or start the next mode."""
        for mode in MODES:
            with self.subTest(mode=mode), self._environment(mode) as native:
                native.errors["read"] = OSError("proc read failed")
                with self.assertRaisesRegex(GLTCore.GLTTestFail, mode):
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                self._assert_lifecycle(native, MODES[:MODES.index(mode) + 1])

    def test_required_config_skips_before_build_without_arch_or_lto_gate(self):
        """Only ASPM and loadable/unloadable modules are kernel-config prerequisites."""
        for missing in l1ss.L1SS_REQUIRED_CONFIGS:
            with self.subTest(missing=missing), self._environment() as native:
                native.config.return_value[missing] = "n"
                with self.assertRaisesRegex(GLTCore.GLTTestSkip, missing):
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                native.commands.assert_not_called()
                native.load.assert_not_called()
        with self._environment() as native:
            native.config.return_value.update(CONFIG_KPROBES="n", CONFIG_KRETPROBES="n",
                                              CONFIG_LTO_CLANG="y", CONFIG_LTO_NONE="n")
            with self.mock.patch.object(l1ss.platform, "machine", return_value="other"):
                l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
            self._assert_lifecycle(native)

    def test_restoration_or_bad_report_overrides_eligibility_errno(self):
        """A SKIP/WARN errno never hides failed restoration or malformed evidence."""
        for mode in MODES:
            for code in (errno.ENODEV, errno.EOPNOTSUPP, errno.EBUSY, errno.ENOMEM):
                for report in (_probe_text(_probe_data(mode, restore_error=-errno.EIO)), "bad report"):
                    with self.subTest(mode=mode, errno=code, report=report):
                        with self._environment(mode) as native:
                            native.report = report
                            native.errors["write"] = OSError(code, "capture unavailable")
                            with self.assertRaisesRegex(GLTCore.GLTTestFail, mode):
                                l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                            self._assert_lifecycle(native, MODES[:MODES.index(mode) + 1])

    def test_setup_warning_cleans_build_products_without_removing_modules(self):
        """Build/sign faults precede load ownership and still clean generated files."""
        for stage in ("build", "sign", "build_decode", "sign_decode"):
            with self.subTest(stage=stage), self._environment() as native:
                if stage == "build":
                    native.errors["make"] = RuntimeError("build failed")
                elif stage == "sign":
                    native.sign_result = False
                elif stage == "build_decode":
                    native.errors["make"] = UnicodeError("build decoding failed")
                else:
                    native.errors["sign"] = UnicodeError("sign decoding failed")
                with self.assertRaises(GLTCore.GLTTestWarn):
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                native.load.assert_not_called()
                self.assertEqual(native.events, [("make",), ("make", "clean")])

    def test_stale_child_or_parent_saved_buffer_fails_either_mode(self):
        """A stale buffer is reported after both safe scenarios have been exercised."""
        for mode in MODES:
            for device in ("child", "parent"):
                for control in (1, 2):
                    for value, saved in ((0, 0xffffffff), (0x1030f, 0xfffefcf0)):
                        with self.subTest(mode=mode, device=device, control=control, value=value):
                            data = _probe_data(mode, value=value)
                            data[f"{device}_ctl{control}_saved"] = saved
                            with self._environment(mode, data) as native:
                                with self.assertRaisesRegex(GLTCore.GLTTestFail, mode) as caught:
                                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                                self.assertIn(l1ss.L1SS_MODES[mode], str(caught.exception))
                                self._assert_lifecycle(native)

    def test_both_stale_scenarios_are_reported_together(self):
        """The first mismatch does not prevent collecting the second patch result."""
        with self._environment() as native:
            native.reports = {
                mode: _probe_text(_probe_data(mode, parent_ctl1_saved=0xffffffff))
                for mode in MODES
            }
            with self.assertRaises(GLTCore.GLTTestFail) as caught:
                l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
            reason = str(caught.exception)
            for mode in MODES:
                self.assertIn(mode, reason)
                self.assertIn(l1ss.L1SS_MODES[mode], reason)
            self._assert_lifecycle(native)

    def test_prior_mismatch_overrides_later_skip_or_warning(self):
        """An unavailable second scenario cannot downgrade the first scenario's FAIL."""
        for code in (errno.ENODEV, errno.EOPNOTSUPP, errno.EBUSY, errno.ENOMEM):
            with self.subTest(errno=code), self._environment(MODES[1]) as native:
                native.reports[MODES[0]] = _probe_text(
                    _probe_data(MODES[0], parent_ctl1_saved=0xffffffff))
                native.report = _probe_text(_probe_data(MODES[1], capture_error=-code))
                native.errors["write"] = OSError(code, "second capture unavailable")
                with self.assertRaises(GLTCore.GLTTestFail) as caught:
                    l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                reason = str(caught.exception)
                self.assertIn(MODES[0], reason)
                self.assertIn(l1ss.L1SS_MODES[MODES[0]], reason)
                self.assertIn(MODES[1], reason)
                self.assertIn("second capture unavailable", reason)
                self._assert_lifecycle(native)

    def test_uncertain_load_only_removes_visible_helper(self):
        """An interrupted or undecodable load requires cleanup when its helper is visible."""
        for error in (subprocess.TimeoutExpired(["insmod"], 30), UnicodeError("decode failed"),
                      subprocess.CalledProcessError(-9, ["insmod"])):
            for visible in (False, True):
                with self.subTest(error=type(error), visible=visible), self._environment() as native:
                    native.errors["load"] = error
                    native.load_visible = visible
                    with self.assertRaises(GLTCore.GLTTestWarn):
                        l1ss.test_pci_aspm_l1ss_save_parent(None, self.mock_logger)
                    self._assert_lifecycle(native, modes=(), unload=visible)
