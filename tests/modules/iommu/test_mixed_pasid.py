# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked contracts for safe mixed-PASID discrimination and ownership."""

import ast
import copy
import errno
import importlib.util
import json
import signal
import struct
import subprocess
import sys
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from common import glt_core as GLTCore
from modules.iommu import glt_test_iommu as native
from modules.iommu.nested_ste import trigger
from modules.iommu.nested_ste.trigger import _vfio_attach_pt as base_attach
from tests.test_base import GLTTestBase


HELPER_PATH = Path(__file__).resolve().parents[3] / "src/bin/glt_iommu_mixed_pasid.py"
OBSERVER_PATH = HELPER_PATH.with_suffix(".bt")
SPEC = importlib.util.spec_from_file_location("mixed_pasid_worker", HELPER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class MixedPasidTestBase(GLTTestBase):
    """Keep unsafe fixtures, unattributed traces and incomplete cleanup red."""

    def setUp(self):
        super().setUp()
        self.fixture = {
            "os_id": "ubuntu", "machine": "aarch64", "smmuv3": True,
            "gpu": "0000:02:00.0", "parent": "0000:00:01.0",
            "direct_parent": "0000:00:01.0",
            "members": ["0000:02:00.0", "0000:00:01.0"],
            "gpu_driver": None, "gpu_override": "", "gpu_used": False,
            "parent_driver": "pcieport", "gpu_pasid_cap": True,
            "parent_pasid_cap": False, "tools_ready": True,
        }
        self.pid = 701
        self.tid = 702
        self.bound = 1 << 20
        self.observer = {
            "pid": 703, "starttime": "123456", "ready": True,
            "live": True, "marker_pid": 701,
            "argv": ["bpftrace", str(OBSERVER_PATH), "701"],
            "program_ids": [11, 12, 13, 14],
        }
        self.context = {"fds": [41, 42], "iommu_fd": 41, "vfio_fd": 42,
                        "device_id": 101, "ioas_id": 202, "hwpt_id": 303}
        # No unit can contact a device or launch an unmocked subprocess.
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        self.guards.enter_context(self.mock.patch.object(worker, "OBSERVATION_TIMEOUT", 0.0))
        for target in ("os.open", "fcntl.ioctl", "subprocess.Popen",
                       "subprocess.run", "os.kill"):
            self.guards.enter_context(self.mock.patch(
                target, side_effect=AssertionError(f"unmocked hardware/process call: {target}")))
        self.guards.enter_context(self.mock.patch.object(worker.os, "getpid", return_value=self.pid))
        self.guards.enter_context(self.mock.patch.object(worker.threading, "get_native_id", return_value=self.tid))
        self.guards.enter_context(self.mock.patch.object(
            worker, "_module_present", create=True,
            side_effect=lambda name: (Path("/sys/module") / name).exists()))


    def _event(self, kind, pasid=1, member=None, **fields):
        result = {"event": kind, "pid": self.pid, "tid": self.tid,
                  "pasid": pasid, "member": member or self.fixture["gpu"]}
        if kind == "entry":
            result.update(parent=self.fixture["parent"], gpu_iommu=1,
                          parent_iommu=1, gpu_limit=self.bound, parent_limit=0)
        result.update(fields)
        return result

    def _records(self, valid_ret=0):
        records = [self._event("entry")]
        if valid_ret == 0:
            records.append(self._event("set"))
        records.append(self._event("return", ret=valid_ret))
        if valid_ret == 0:
            records.append(self._event("block"))
        return records

    @staticmethod
    def _text(records):
        return "\n".join(json.dumps(record) for record in records) + "\n"

    def _worker_mocks(self, **overrides):
        stack = ExitStack()
        defaults = {
            "inspect_fixture": self.fixture, "prepare_observer": dict(
                self.observer, ok=True, process=self.mock.Mock(poll=lambda: None)),
            "allocate_context": self.context,
            "run_requests": {"status": "PASS", "reason": "", "valid": True},
            "cleanup_resources": {"ok": True, "reason": "", "residuals": []},
        }
        defaults.update(overrides)
        stack.enter_context(self.mock.patch.object(worker, "_acquire_gpu"))
        mocks = {}

        def prepare(observer):
            def acknowledge_owned(_pid, acknowledge):
                if observer["ok"]:
                    acknowledge(observer)
                return observer
            return acknowledge_owned

        for name, value in defaults.items():
            mocks[name] = stack.enter_context(self.mock.patch.object(
                worker, name, side_effect=value if isinstance(value, Exception) else
                (prepare(value) if name == "prepare_observer" else None),
                return_value=None if isinstance(value, Exception) else value))
        return stack, mocks

    def _capture_ioctl(self, calls, *args):
        fd, request, payload, mutate = args
        self.assertTrue(mutate)
        calls.append((fd, request, bytes(payload)))
        outputs = {trigger.VFIO_DEVICE_BIND_IOMMUFD: (12, 101),
                   trigger.IOMMU_IOAS_ALLOC: (8, 202), trigger.IOMMU_HWPT_ALLOC: (16, 303)}
        if request in outputs:
            offset, value = outputs[request]
            struct.pack_into("<I", payload, offset, value)
        return 0

    def _ioctl_capture(self):
        calls = []
        return calls, partial(self._capture_ioctl, calls)


class TestMixedPasid(MixedPasidTestBase):
    """Keep unsafe fixtures, unattributed traces and incomplete cleanup red."""

    def _evaluate(self, records=None, detach_result=0):
        return worker.evaluate_trace(
            self._text(self._records() if records is None else records),
            self.fixture, self.pid, self.tid, detach_result)

    def test_native_case_registration(self):
        """The opt-in case must be discovered only under existing IOMMU."""
        attr = getattr(native.test_mixed_pasid_attach, "attr", None)
        self.assertIsNotNone(attr, "mixed-PASID case must have GLTTestAttr")
        self.assertEqual((attr.category, attr.name), ("iommu", "test_mixed_pasid_attach"))
        self.assertEqual(attr.plans, ["disruptive"])
        cases = [fn for fn in vars(native).values() if callable(fn)
                 and getattr(getattr(fn, "attr", None), "name", None)
                 == "test_mixed_pasid_attach"]
        self.assertEqual(len(cases), 1)
        tree = ast.parse(HELPER_PATH.read_text(encoding="utf-8"))
        forbidden = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr in {"fail", "skip"}]
        self.assertEqual(forbidden, [], "only decorated entry records GLT results")

    def test_missing_fixture_skips(self):
        """Only absent ordinary prerequisites skip, before acquisition."""
        bad = {
            "os_id": "other", "machine": "x86_64", "smmuv3": False,
            "members": [self.fixture["gpu"]], "direct_parent": "0000:00:02.0",
            "gpu_driver": "nvidia", "gpu_override": "vfio-pci", "gpu_used": True,
            "parent_driver": None, "gpu_pasid_cap": False,
            "parent_pasid_cap": True, "tools_ready": False,
        }
        for field, value in bad.items():
            with self.subTest(prerequisite=field):
                fixture = dict(self.fixture, **{field: value})
                stack, mocked = self._worker_mocks(inspect_fixture=fixture)
                with stack:
                    result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "SKIP", field)
                self.assertTrue(result["reason"], "SKIP must explain the missing prerequisite")
                mocked["allocate_context"].assert_not_called()
                mocked["prepare_observer"].assert_not_called()
        # Presence alone suffices; invented PCI width cannot become a limit gate.
        fixture = dict(self.fixture, gpu_pasid_width=0, gpu_effective_limit=None)
        stack, mocked = self._worker_mocks(inspect_fixture=fixture)
        with stack:
            result = worker.run_worker(self.mock.Mock(return_value=True))
        with self.subTest(contract="capability_presence_only"):
            self.assertEqual(result["status"], "PASS")
            mocked["allocate_context"].assert_called_once()
        process = self.mock.Mock(pid=self.pid, returncode=0)
        process.stdout.readline.return_value = json.dumps({
            "type": "result", "status": "SKIP", "reason": "missing PASID capability",
            "cleanup_ok": True, "residuals": []}) + "\n"
        process.communicate.return_value = ("", "")
        with self.mock.patch("subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(GLTCore.GLTTestSkip, "PASID"):
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)

    def test_ready_valid_einval_fails(self):
        """Attributable control rejection is FAIL with cleanup and bound retained."""
        report = self._evaluate(self._records(valid_ret=-22))
        with self.subTest(contract="control_result"):
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["valid"])
            self.assertEqual(report["bound"], self.bound)
        stack, mocked = self._worker_mocks(run_requests=report)
        with stack:
            result = worker.run_worker(self.mock.Mock(return_value=True))
        with self.subTest(contract="worker_cleanup"):
            self.assertEqual(result["status"], "FAIL")
            mocked["cleanup_resources"].assert_called_once()
        process = self.mock.Mock(pid=self.pid, returncode=1)
        process.stdout.readline.return_value = json.dumps({
            **report, "type": "result", "cleanup_ok": True, "residuals": []}) + "\n"
        process.communicate.return_value = ("", "")
        with self.mock.patch("subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(
                    GLTCore.GLTTestFail, "valid PASID reached core and returned EINVAL.*b3f6fcd8404f"):
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)

    def test_pasid_uapi_payloads(self):
        """Distinct handles reveal swapped FD/object IDs and incorrect ABI flags."""
        calls, ioctl = self._ioctl_capture()
        with self.mock.patch("os.open", side_effect=[41, 42]), \
                self.mock.patch("fcntl.ioctl", side_effect=ioctl), \
                self.mock.patch.object(worker, "_find_vfio_cdev", return_value="/dev/vfio/devices/vfio9"):
            context = worker.allocate_context(self.fixture)
            worker.run_requests(context, self.fixture, self.mock.Mock(return_value=self._text(self._records())))
        expected = [
            (42, trigger.VFIO_DEVICE_BIND_IOMMUFD, struct.pack("<2IiIQ", 24, 0, 41, 0, 0)),
            (41, trigger.IOMMU_IOAS_ALLOC, struct.pack("<3I", 12, 0, 0)),
            (41, trigger.IOMMU_HWPT_ALLOC, struct.pack("<8IQ2I", 48, 8, 101, 202, 0, 0, 0, 0, 0, 0, 0)),
            (42, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT, struct.pack("<4I", 16, 0, 303, 0)),
            (42, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT, struct.pack("<4I", 16, 1, 303, 1)),
            (42, (ord(';') << 8) | 120, struct.pack("<3I", 12, 1, 1)),
        ]
        self.assertEqual(calls, expected, "the returned bind/IOAS/HWPT IDs must reach the correct next ioctl")
        self.assertEqual(context, self.context)

    def test_empty_ioas_only(self):
        """Base attachment must reuse the empty IOAS, never map DMA or nest."""
        calls, ioctl = self._ioctl_capture()
        with self.mock.patch("os.open", side_effect=[41, 42]), \
                self.mock.patch("fcntl.ioctl", side_effect=ioctl), \
                self.mock.patch.object(worker, "_find_vfio_cdev", return_value="/dev/vfio/devices/vfio9"), \
                self.mock.patch.object(worker, "_vfio_attach_pt", wraps=base_attach) as base:
            context = worker.allocate_context(self.fixture)
        self.assertEqual(context["ioas_id"], 202, "allocate an IOAS before constructing the HWPT")
        base.assert_called_once_with(42, 303)
        self.assertEqual([request for _, request, _ in calls], [
            trigger.VFIO_DEVICE_BIND_IOMMUFD, trigger.IOMMU_IOAS_ALLOC,
            trigger.IOMMU_HWPT_ALLOC, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT])
        self.assertFalse(any(request in {trigger.IOMMU_IOAS_MAP, trigger.IOMMU_VIOMMU_ALLOC,
                                        trigger.IOMMU_VDEVICE_ALLOC} for _, request, _ in calls))

    def test_valid_request_observations(self):
        """Success requires one ordered attributed entry, set, return and block."""
        report = self._evaluate()
        with self.subTest(contract="complete_valid_trace"):
            self.assertTrue(report["valid"], "complete valid-PASID evidence must be accepted")
            self.assertEqual(report["status"], "PASS")
        records = self._records()
        variants = {
            "missing_return": records[:2] + records[3:],
            "duplicate_return": records[:3] + [records[2]] + records[3:],
            "set_after_return": [records[0], records[2], records[1]] + records[3:],
            "companion_set": records[:2] + [self._event("set", member=self.fixture["parent"])] + records[2:],
            "duplicate_set": records[:2] + [records[1]] + records[2:],
            "missing_set": [records[0]] + records[2:],
        }
        for label, trace in variants.items():
            with self.subTest(malformed=label):
                result = self._evaluate(trace)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result["reason"])

    def test_detach_required(self):
        """The detach ioctl and exactly one GPU block are independent proof."""
        records = self._records()
        variants = [
            ("missing_ioctl", records, None), ("failed_ioctl", records, -5),
            ("missing_block", records[:3] + records[4:], 0),
            ("duplicate_block", records[:4] + [records[3]] + records[4:], 0),
            ("companion_block", records[:4] + [self._event("block", member=self.fixture["parent"])] + records[4:], 0),
            ("early_block", [records[0], records[3], records[1], records[2]] + records[4:], 0),
        ]
        for label, trace, detach in variants:
            with self.subTest(malformed=label):
                result = self._evaluate(trace, detach)
                self.assertEqual(result["status"], "FAIL", "counts cannot replace successful explicit detach")
                self.assertTrue(result["reason"])
        self.assertTrue(self._evaluate()["valid"], "no traced core-detach-return is required")

    def test_bound_uses_effective_positive_limits(self):
        """The zero-limit companion must never collapse the capable bound to 0."""
        for limit in (32, self.bound):
            with self.subTest(effective_limit=limit):
                records = self._records()
                for record in records:
                    if record["event"] == "entry":
                        record["gpu_limit"] = limit
                fixture = dict(self.fixture, gpu_pasid_width=3)
                report = worker.evaluate_trace(self._text(records), fixture, self.pid, self.tid, 0)
                self.assertEqual(report["bound"], limit,
                                 "use observed positive limit rather than PCI width or min(0, limit)")
                self.assertTrue(report["valid"])

    def test_early_einval_is_not_core_rejection(self):
        """Only wrapped EINVAL plus attributable bound core entry/return qualifies."""
        failures = [("early", errno.EINVAL), ("wrong_errno", errno.EIO), ("wrong_cause", None)]
        for label, error in failures:
            with self.subTest(error=label):
                failure = trigger.TriggerError("attach error")
                failure.__cause__ = (OSError(error, "attach") if error
                                     else ValueError("not ioctl errno"))
                with self.mock.patch("fcntl.ioctl", side_effect=failure):
                    report = worker.run_requests(self.context, self.fixture, self.mock.Mock(return_value=""))
                self.assertEqual(report["status"], "FAIL", "ioctl errno alone cannot prove core rejection")
                self.assertFalse(report["valid"])
                self.assertTrue(report["reason"])
        failure = trigger.TriggerError("control EINVAL")
        failure.__cause__ = OSError(errno.EINVAL, "control")
        with self.mock.patch.object(worker, "_xioctl", side_effect=failure) as ioctl:
            report = worker.run_requests(self.context, self.fixture,
                                         self.mock.Mock(return_value=self._text(self._records(valid_ret=-22))))
        ioctl.assert_called_once_with(42, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT,
                                      bytearray(struct.pack("<4I", 16, 1, 303, 1)))
        self.assertEqual(report["status"], "FAIL", report)
        self.assertEqual(report["bound"], self.bound)
        self.assertIn("valid PASID reached core and returned EINVAL", report["reason"])

    def test_observer_ready_before_binding(self):
        """Own marker, live PID, four IDs, inventory and handoff precede mutation."""
        invalid = {
            "marker_only": {"program_ids": []}, "dead": {"live": False},
            "foreign_marker": {"marker_pid": self.pid + 1}, "no_marker": {"ready": False},
            "three_ids": {"program_ids": [11, 12, 13]},
            "duplicates": {"program_ids": [11, 11, 12, 12]},
        }
        for label, changes in invalid.items():
            with self.subTest(readiness=label), \
                    self.mock.patch.object(worker, "start_observer",
                                           return_value=dict(self.observer, **changes)), \
                    self.mock.patch.object(worker, "program_inventory",
                                           return_value=[{"id": i} for i in range(11, 15)]):
                handoff = self.mock.Mock(return_value=True)
                ready = worker.prepare_observer(self.pid, handoff)
                self.assertFalse(ready["ok"])
                self.assertTrue(ready["reason"], "each unsafe observer needs a concrete failure reason")
                handoff.assert_not_called()
        for failure in (FileNotFoundError("bpftool"), OSError("inventory failed"), ValueError("invalid JSON")):
            with self.subTest(inventory=str(failure)), \
                    self.mock.patch.object(worker, "start_observer", return_value=self.observer), \
                    self.mock.patch.object(worker, "program_inventory", side_effect=failure):
                ready = worker.prepare_observer(self.pid, self.mock.Mock(return_value=True))
                self.assertFalse(ready["ok"])
                self.assertTrue(ready["reason"])
        for accepted in (False, True):
            with self.subTest(handoff=accepted), \
                    self.mock.patch.object(worker, "start_observer", return_value=self.observer), \
                    self.mock.patch.object(worker, "program_inventory",
                                           return_value=[{"id": i} for i in range(11, 15)]):
                handoff = self.mock.Mock(return_value=accepted)
                ready = worker.prepare_observer(self.pid, handoff)
                self.assertEqual(ready["ok"], accepted)
                handoff.assert_called_once()
        with self.subTest(contract="PATH_bpftool_json"), \
                self.mock.patch("shutil.which", return_value="/usr/bin/bpftool"), \
                self.mock.patch("subprocess.run", return_value=SimpleNamespace(
                    returncode=0, stdout='[{"id": 11}]', stderr="")) as inventory_run:
            self.assertEqual(worker.program_inventory(), [{"id": 11}])
            self.assertEqual(inventory_run.call_args.args[0], ["bpftool", "-j", "prog", "show"])
        order = self.mock.Mock()
        stack, mocked = self._worker_mocks()
        with stack:
            order.attach_mock(mocked["prepare_observer"], "ready")
            order.attach_mock(mocked["allocate_context"], "acquire")
            worker.run_worker(self.mock.Mock(return_value=True))
        with self.subTest(contract="ready_before_acquisition"):
            self.assertEqual([call[0] for call in order.mock_calls], ["ready", "acquire"])
        with self.subTest(contract="separate_observer_invocation"):
            with self.mock.patch("subprocess.Popen") as popen:
                popen.return_value.pid = self.observer["pid"]
                worker.start_observer(self.pid)
            self.assertTrue(popen.called, "start the separate GPL observer")
            argv = popen.call_args.args[0]
            self.assertIn(str(OBSERVER_PATH), argv)
            self.assertIn(str(self.pid), argv)
        with self.subTest(contract="GPL_typed_filtered_observer"):
            source = OBSERVER_PATH.read_text(encoding="utf-8")
            self.assertIn("SPDX-License-Identifier: GPL-2.0-only", source)
            self.assertIn("pid == $1", source)
            self.assertIn("tid", source)
            self.assertEqual(source.count("kfunc:") + source.count("kretfunc:"), 4)

    def test_observation_attribution(self):
        """Independent PID/TID, identity, pointer and limit checks reject mixing."""
        fields = {
            "pid": 999, "tid": 999, "pasid": 2, "member": self.fixture["parent"],
            "parent": self.fixture["gpu"], "gpu_iommu": 0, "parent_iommu": 0,
            "gpu_limit": 0, "parent_limit": 4,
        }
        variants = {}
        for field, value in fields.items():
            records = self._records()
            records[0][field] = value
            variants["wrong_" + field] = records
        records = self._records()
        records[0]["gpu_limit"] = 1
        variants["gpu_limit_one"] = records
        for field in ("pid", "tid", "gpu_iommu", "parent_iommu", "gpu_limit", "parent_limit", "parent"):
            records = self._records()
            del records[0][field]
            variants["missing_" + field] = records
        records = self._records()
        records[-1]["pid"] = 999
        variants["mixed_session"] = records
        records = self._records()
        records.insert(1, records[0].copy())
        variants["duplicate_entry"] = records
        records = self._records()
        records[0], records[1] = records[1], records[0]
        variants["reordered_entry"] = records
        for label, trace in variants.items():
            with self.subTest(malformed=label):
                report = self._evaluate(trace)
                self.assertEqual(report["status"], "FAIL", label)
                self.assertTrue(report["reason"])
        self.assertEqual(self._evaluate()["status"], "PASS")

    def test_cleanup_failure_fails_case(self):
        """No success escapes failed cleanup or an unverifiable BPF inventory."""
        cleanup_failures = ("close failed", "restore failed", "observer reap failed",
                            "inventory failed", "owned program IDs remain", "module unload failed")
        for reason in cleanup_failures:
            with self.subTest(cleanup=reason):
                stack, mocked = self._worker_mocks(cleanup_resources={
                    "ok": False, "reason": reason, "residuals": ["owned state"]})
                with stack:
                    result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "FAIL")
                self.assertIn(reason, result["reason"])
                mocked["cleanup_resources"].assert_called_once()
        for failure in (trigger.TriggerError("acquisition failed"), OSError("open failed")):
            with self.subTest(acquisition=type(failure).__name__):
                stack, mocked = self._worker_mocks(allocate_context=failure)
                with stack:
                    result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "FAIL")
                mocked["cleanup_resources"].assert_called_once()
        resources = dict(self.context, observer=self.observer, modules_loaded=["vfio_pci"])
        for inventory in ([{"id": 11}], OSError("bpftool inventory failed")):
            with self.subTest(inventory=inventory), self.mock.patch("os.close"), \
                    self.mock.patch.object(worker, "restore_gpu", return_value=True), \
                    self.mock.patch.object(worker, "stop_observer", return_value=True), \
                    self.mock.patch("subprocess.run"), \
                    self.mock.patch.object(worker, "program_inventory",
                                           side_effect=inventory if isinstance(inventory, Exception) else None,
                                           return_value=inventory):
                report = worker.cleanup_resources(self.fixture, resources)
                self.assertFalse(report["ok"])
                self.assertTrue(report["reason"])
        process = self.mock.Mock(pid=self.pid, returncode=1)
        process.stdout.readline.return_value = json.dumps({
            "type": "result", "status": "FAIL", "reason": "cleanup failed",
            "cleanup_ok": False, "residuals": ["owned state"]}) + "\n"
        process.communicate.return_value = ("", "")
        with self.mock.patch("subprocess.Popen", return_value=process):
            with self.assertRaises(GLTCore.GLTTestFail) as failed:
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
        self.assertEqual(str(failed.exception), "cleanup failed; worker cleanup attestation failed; residuals: "
                         "['owned state', "
                         "'worker cleanup unverified: GPU binding/override, FDs and temporary modules', "
                         "'observer ownership was not handed off']")
        process.kill.assert_not_called()
        process.terminate.assert_not_called()

    def test_only_owned_resources_cleaned(self):
        """Completed child restores its GPU and leaves preexisting resources alone."""
        resources = dict(self.context, observer=self.observer, modules_loaded=["vfio_pci"],
                         modules_preexisting=["iommufd", "vfio"], gpu_changed=True)
        with self.mock.patch("os.close") as close, \
                self.mock.patch.object(worker, "restore_gpu", return_value=True) as restore, \
                self.mock.patch.object(worker, "stop_observer", return_value=True) as stop, \
                self.mock.patch.object(worker, "program_inventory", return_value=[{"id": 999}]), \
                self.mock.patch.object(Path, "exists", lambda path: path.name in ("iommufd", "vfio")), \
                self.mock.patch("subprocess.run", return_value=SimpleNamespace(
                    returncode=0, stdout="", stderr="")) as run:
            result = worker.cleanup_resources(self.fixture, resources)
        self.assertTrue(result["ok"], "cleanup must attest verified absence of owned IDs")
        self.assertEqual(close.call_args_list, [self.mock.call(42), self.mock.call(41)])
        restore.assert_called_once_with(self.fixture)
        stop.assert_called_once_with(self.observer)
        self.assertEqual([call.args[0] for call in run.call_args_list], [["rmmod", "vfio_pci"]])

    def _supervision(self, alive, handoff=True, stale=False, returncode=None):
        observer = copy.deepcopy(self.observer)
        process = self.mock.Mock(pid=self.pid, returncode=returncode)
        report = {"type": "observer", "observer": observer}
        process.stdout.readline.side_effect = [json.dumps(report) + "\n", ""] if handoff else [""]
        process.poll.return_value = None if alive else returncode
        process.communicate.side_effect = subprocess.TimeoutExpired("worker", 1) if alive else None
        process.communicate.return_value = ("", "crashed")
        identity = observer.copy()
        if stale == "starttime":
            identity["starttime"] = "654321"
        elif stale == "argv":
            identity["argv"] = ["bpftrace", "other-observer.bt", "701"]
        elif stale == "pid":
            identity["pid"] = 999
        stack = ExitStack()
        popen = stack.enter_context(self.mock.patch("subprocess.Popen", return_value=process))
        kill = stack.enter_context(self.mock.patch("os.kill"))
        stack.enter_context(self.mock.patch.object(native, "_mixed_pasid_observer_identity", return_value=identity))
        mutation = stack.enter_context(self.mock.patch.object(native.glt_utils, "write_sysfs_value"))
        run = stack.enter_context(self.mock.patch("subprocess.run"))
        return stack, process, popen, kill, mutation, run







    def test_live_worker_never_unbinds(self):
        """Timeout fails with residuals; only the reverified owned observer stops."""
        for handoff, stale in ((True, False), (False, False), (True, "starttime"),
                               (True, "argv"), (True, "pid")):
            with self.subTest(handoff=handoff, stale=stale):
                stack, process, popen, kill, mutation, run = self._supervision(True, handoff, stale)
                with stack, self.assertRaises(GLTCore.GLTTestFail) as failed:
                    native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
                reason = ("mixed-PASID worker incomplete: observer identity changed before handoff acknowledgment"
                          if stale else "mixed-PASID worker incomplete: Command 'worker' timed out after 1 seconds")
                residuals = ["worker cleanup unverified: GPU binding/override, FDs and temporary modules"]
                if stale:
                    residuals.append("observer identity or stop unverified")
                elif not handoff:
                    residuals.append("observer ownership was not handed off")
                self.assertEqual(str(failed.exception), reason + f"; residuals: {residuals}")
                mutation.assert_not_called()
                run.assert_not_called()
                process.kill.assert_not_called()
                process.terminate.assert_not_called()
                if handoff and not stale:
                    kill.assert_called_once_with(self.observer["pid"], signal.SIGTERM)
                else:
                    kill.assert_not_called()
                self.assertEqual(popen.call_args.args[0][:2], [
                    sys.executable, "/usr/share/greenlit/bin/glt_iommu_mixed_pasid.py"])

    def test_dead_worker_without_cleanup_fails(self):
        """Process exit, including zero, cannot stand in for cleanup attestation."""
        for returncode in (0, 9):
            for stale in (False, "starttime", "argv", "pid"):
                with self.subTest(returncode=returncode, stale=stale):
                    stack, _, _, kill, mutation, run = self._supervision(False, True, stale, returncode)
                    with stack, self.assertRaises(GLTCore.GLTTestFail) as failed:
                        native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
                    reason = ("mixed-PASID worker incomplete: observer identity changed before handoff acknowledgment"
                              if stale else "worker exited without verified cleanup attestation: crashed")
                    residuals = ["worker cleanup unverified: GPU binding/override, FDs and temporary modules"]
                    if stale:
                        residuals.append("observer identity or stop unverified")
                    self.assertEqual(str(failed.exception), reason + f"; residuals: {residuals}")
                    mutation.assert_not_called()
                    run.assert_not_called()
                    if stale:
                        kill.assert_not_called()
                    else:
                        kill.assert_called_once_with(self.observer["pid"], signal.SIGTERM)



class TestMixedPasidSafety(MixedPasidTestBase):
    """Exercise inspection, acquisition and terminal protocol safety boundaries."""

    def _check_acquisition_boundary(self, expired, boundary):
        now = [0]
        alive = [True]
        writes = []
        modules = []
        inspections = []
        observer = dict(self.observer, process=self.mock.Mock(poll=lambda: None if alive[0] else 1))

        def delay():
            if expired:
                now[0] += 61
            else:
                alive[0] = False

        def inspect():
            inspections.append(True)
            if len(inspections) == 2 and boundary == 'inspection':
                delay()
            return self.fixture

        def load(argv, **_):
            modules.append(argv[-1])
            if boundary == 'module' + str(len(modules)):
                delay()

        def write(path, value, **_):
            writes.append((str(path), value))
            if ((path.name == 'driver_override' and boundary == 'override')
                    or (path.name == 'bind' and boundary == 'bind')):
                delay()

        def handoff(_):
            now[0] = 100  # Discovery time before ACK is outside acquisition budget.
            return True

        with self.mock.patch.object(worker, 'inspect_fixture', side_effect=inspect), \
                self.mock.patch.object(worker, 'start_observer', return_value=observer), \
                self.mock.patch.object(worker, 'program_inventory', return_value=[
                    {'id': value} for value in self.observer['program_ids']]), \
                self.mock.patch.object(worker.time, 'monotonic', side_effect=lambda: now[0]), \
                self.mock.patch.object(worker.pci_utils, 'get_pci_driver',
                    side_effect=[None, 'vfio-pci', 'pcieport']), \
                self.mock.patch.object(worker, '_override', side_effect=['', 'vfio-pci']), \
                self.mock.patch.object(Path, 'exists', return_value=False), \
                self.mock.patch.object(Path, 'write_text', autospec=True, side_effect=write), \
                self.mock.patch('subprocess.run', side_effect=load), \
                self.mock.patch.object(worker, 'allocate_context', return_value={}) as acquire, \
                self.mock.patch.object(worker, 'run_requests', return_value={'status': 'PASS'}), \
                self.mock.patch.object(worker, 'cleanup_resources', return_value={
                    'ok': True, 'reason': '', 'residuals': []}) as cleanup:
            result = worker.run_worker(handoff)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['cleanup_ok'])
        self.assertIn('budget' if expired else 'observer', result['reason'])
        self.assertEqual(len(modules), {'inspection': 0, 'module1': 1,
                                       'module2': 2, 'override': 2, 'bind': 2}[boundary])
        self.assertEqual(len(writes), {'override': 1, 'bind': 2}.get(boundary, 0))
        acquire.assert_not_called()
        cleanup.assert_called_once()
        if boundary == 'bind':
            self.assertTrue(cleanup.call_args.args[1]['gpu_changed'])

    def test_acquisition_budget_and_observer_gate_each_next_mutation(self):
        """A successful ACK cannot authorize later writes after budget/liveness loss."""
        self.fixture['module_plan'] = ['iommufd', 'vfio_pci']
        for expired in (True, False):
            for boundary in ('inspection', 'module1', 'module2', 'override', 'bind'):
                with self.subTest(expired=expired, boundary=boundary):
                    self._check_acquisition_boundary(expired, boundary)

    def test_post_ack_module_timeout_cannot_reach_binding(self):
        """A timed-out child module command fails with child-owned cleanup."""
        self.fixture['module_plan'] = []
        observer = dict(self.observer, process=self.mock.Mock(poll=lambda: None))

        def load(argv, **kwargs):
            if kwargs.get('timeout') == 10:
                raise subprocess.TimeoutExpired(argv, 10)

        with self.mock.patch.object(worker, 'inspect_fixture', return_value=self.fixture), \
                self.mock.patch.object(worker, 'start_observer', return_value=observer), \
                self.mock.patch.object(worker, 'program_inventory', return_value=[
                    {'id': value} for value in self.observer['program_ids']]), \
                self.mock.patch.object(worker.pci_utils, 'get_pci_driver',
                    side_effect=[None, 'vfio-pci', 'pcieport']), \
                self.mock.patch.object(worker, '_override', side_effect=['', 'vfio-pci']), \
                self.mock.patch.object(Path, 'write_text') as write, \
                self.mock.patch('subprocess.run', side_effect=load) as command, \
                self.mock.patch.object(worker, 'allocate_context', return_value={}) as acquire, \
                self.mock.patch.object(worker, 'run_requests', return_value={'status': 'PASS'}), \
                self.mock.patch.object(worker, 'cleanup_resources', return_value={
                    'ok': True, 'reason': '', 'residuals': []}):
            result = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['cleanup_ok'])
        self.assertEqual(command.call_args.kwargs.get('timeout'), 10)
        write.assert_not_called()
        acquire.assert_not_called()

    def test_worker_inventory_module_plan_and_unload_commands_are_bounded(self):
        """Stalled queries reject acquisition; stalled unload reports unverified cleanup."""
        def stalled(argv, **kwargs):
            if kwargs.get('timeout') is not None:
                raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
            return SimpleNamespace(stdout='[]' if argv[-1] == 'show' else '')

        with self.subTest(command='bpftool'), \
                self.mock.patch('subprocess.run', side_effect=stalled) as command, \
                self.mock.patch('shutil.which', return_value='/usr/bin/bpftool'):
            with self.assertRaises(subprocess.TimeoutExpired):
                worker.program_inventory()
            self.assertEqual(command.call_args.kwargs.get('timeout'), 10)
        with self.subTest(command='show-depends'), \
                self.mock.patch('subprocess.run', side_effect=stalled) as command, \
                self.mock.patch.object(worker.platform, 'freedesktop_os_release', return_value={'ID': 'ubuntu'}), \
                self.mock.patch.object(worker.platform, 'machine', return_value='aarch64'), \
                self.mock.patch('shutil.which', return_value='/usr/bin/tool'), \
                self.mock.patch.object(Path, 'is_file', return_value=True):
            with self.assertRaises(subprocess.TimeoutExpired):
                worker.inspect_fixture()
            self.assertEqual(command.call_args.kwargs.get('timeout'), 10)
        with self.subTest(command='rmmod'), self.mock.patch('subprocess.run', side_effect=stalled) as command, \
                self.mock.patch.object(Path, 'exists', return_value=False):
            result = worker.cleanup_resources(self.fixture, {'fds': [], 'modules_loaded': ['vfio_pci']})
            self.assertFalse(result['ok'])
            self.assertIn('module unload failed: vfio_pci', result['reason'])
            self.assertTrue(result['residuals'])
            self.assertEqual(command.call_args.kwargs.get('timeout'), 10)

    def test_preinspection_fail_retains_truthful_unacquired_result(self):
        """A clean terminal preinspection failure has no acquired GPU to restore."""
        report = {'type': 'result', 'status': 'FAIL', 'reason': 'candidate inspection denied',
                  'cleanup_ok': True, 'residuals': []}
        process = self.mock.Mock(pid=self.pid, returncode=1)
        process.communicate.return_value = ('', '')
        with self.mock.patch('subprocess.Popen', return_value=process), \
                self.mock.patch.object(native, '_mixed_pasid_first_message', return_value=report), \
                self.mock.patch.object(native, '_mixed_pasid_cleanup_readback') as readback:
            with self.assertRaises(GLTCore.GLTTestFail) as failed:
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
        self.assertEqual(str(failed.exception), report['reason'])
        readback.assert_not_called()

    def test_no_observer_does_not_bypass_verdict_exit_or_cleanup_contract(self):
        """Only the exact clean FAIL attestation qualifies without observer ownership."""
        variants = [({'status': 'PASS', 'valid': True}, 0),
                    ({'cleanup_ok': False}, 1), ({'cleanup_ok': None}, 1),
                    ({'residuals': ['unverified device']}, 1), ({}, 0), ({'status': 'OTHER'}, 1),
                    ({'missing_field': 'cleanup_ok'}, 1), ({'missing_field': 'residuals'}, 1)]
        for changes, exit_code in variants:
            with self.subTest(changes=changes, exit_code=exit_code):
                report = dict({'type': 'result', 'status': 'FAIL', 'reason': 'inspection denied',
                               'cleanup_ok': True, 'residuals': []},
                              **{key: value for key, value in changes.items() if key != 'missing_field'})
                if 'missing_field' in changes:
                    del report[changes['missing_field']]
                process = self.mock.Mock(pid=self.pid, returncode=exit_code)
                process.communicate.return_value = ('', '')
                with self.mock.patch('subprocess.Popen', return_value=process), \
                        self.mock.patch.object(native, '_mixed_pasid_first_message', return_value=report):
                    with self.assertRaises(GLTCore.GLTTestFail) as failed:
                        native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
                if report.get('cleanup_ok') is not True:
                    reason = 'worker cleanup attestation failed'
                elif report['status'] == 'OTHER':
                    reason = 'mixed-PASID worker incomplete: invalid worker verdict'
                elif exit_code == 0 and report['status'] == 'FAIL':
                    reason = 'mixed-PASID worker incomplete: worker exit code disagrees with its verdict'
                else:
                    reason = 'mixed-PASID worker incomplete: worker result has no observer ownership handoff'
                residuals = report.get('residuals', []) + [
                    'worker cleanup unverified: GPU binding/override, FDs and temporary modules',
                    'observer ownership was not handed off']
                self.assertEqual(str(failed.exception), f'inspection denied; {reason}; residuals: {residuals}')

    def test_preinspection_skip_stays_supported_and_malformed_record_fails(self):
        """Preserve clean prerequisite SKIP without accepting malformed protocol output."""
        for malformed in ('clean', 'non-dict', 'duplicate-result'):
            with self.subTest(malformed=malformed):
                report = {'type': 'result', 'status': 'SKIP', 'reason': 'fixture unavailable',
                          'cleanup_ok': True, 'residuals': []}
                process = self.mock.Mock(pid=self.pid, returncode=0)
                output = {'clean': '', 'non-dict': '[]\n',
                          'duplicate-result': json.dumps(report) + '\n'}[malformed]
                process.communicate.return_value = (output, '')
                with self.mock.patch('subprocess.Popen', return_value=process), \
                        self.mock.patch.object(native, '_mixed_pasid_first_message', return_value=report):
                    exception = GLTCore.GLTTestSkip if malformed == 'clean' else GLTCore.GLTTestFail
                    with self.assertRaises(exception) as outcome:
                        native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
                if malformed == 'clean':
                    self.assertEqual(str(outcome.exception), 'fixture unavailable')
                else:
                    reason = ('mixed-PASID worker incomplete: invalid worker protocol record'
                              if malformed == 'non-dict' else
                              'worker exited without verified cleanup attestation: ')
                    residuals = ['worker cleanup unverified: GPU binding/override, FDs and temporary modules',
                                 'observer ownership was not handed off']
                    self.assertEqual(str(outcome.exception), reason + f'; residuals: {residuals}')


class TestMixedPasidProbeOwnership(MixedPasidTestBase):
    """Existing observer and child cleanup paths own inspection failures after ACK."""

    def _check_probe_failure(self, site):
        self.fixture['module_plan'] = []
        diagnostic = 'inspection denied at ' + site
        observer = dict(self.observer, process=self.mock.Mock(poll=lambda: None))
        drivers = [None, 'vfio-pci', 'pcieport']
        if site != 'reinspection':
            drivers[{'prebind': 0, 'post-gpu': 1, 'post-parent': 2}[site]] = OSError(diagnostic)
        with self.mock.patch.object(worker, 'inspect_fixture', side_effect=[
                self.fixture, OSError(diagnostic) if site == 'reinspection' else self.fixture]), \
                self.mock.patch.object(worker, 'start_observer', return_value=observer), \
                self.mock.patch.object(worker, 'program_inventory', side_effect=[
                    [], [{'id': value} for value in self.observer['program_ids']], []]), \
                self.mock.patch.object(worker.pci_utils, 'get_pci_driver', side_effect=drivers), \
                self.mock.patch.object(worker, '_override', side_effect=['', 'vfio-pci']), \
                self.mock.patch.object(Path, 'exists', return_value=False), \
                self.mock.patch.object(Path, 'write_text') as writes, \
                self.mock.patch('subprocess.run'), \
                self.mock.patch.object(worker, 'allocate_context') as acquire, \
                self.mock.patch.object(worker, 'restore_gpu', return_value=True) as restore, \
                self.mock.patch.object(worker, 'stop_observer', return_value=True), \
                self.mock.patch.object(worker, 'cleanup_resources', wraps=worker.cleanup_resources) as cleanup:
            result = worker.run_worker(self.mock.Mock(return_value=True))
        changed = site in ('post-gpu', 'post-parent')
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['reason'], diagnostic)
        self.assertFalse(observer['ok'])
        self.assertEqual(observer['reason'], diagnostic)
        self.assertTrue(result['cleanup_ok'])
        self.assertEqual(result['residuals'], [])
        cleanup.assert_called_once()
        self.assertEqual(cleanup.call_args.args[1]['gpu_changed'], changed)
        self.assertEqual(restore.call_count, int(changed))
        self.assertEqual(writes.call_count, 2 if changed else 0)
        acquire.assert_not_called()

    def test_post_ack_inspection_errors_use_observer_failure_and_owned_cleanup(self):
        """Reinspection and each pre/post-bind probe preserve precise GPU cleanup ownership."""
        for site in ('reinspection', 'prebind', 'post-gpu', 'post-parent'):
            with self.subTest(site=site):
                self._check_probe_failure(site)

    def test_restore_probe_errors_are_cleanup_failures(self):
        """Initial, restored-GPU and restored-parent inspection errors block cleanup attestation."""
        for site in (0, 1, 2):
            with self.subTest(site=site):
                drivers = ['vfio-pci', None, 'pcieport']
                drivers[site] = OSError('restore inspection denied')
                with self.mock.patch.object(worker.pci_utils, 'get_pci_driver', side_effect=drivers), \
                        self.mock.patch.object(worker, '_override', return_value=''), \
                        self.mock.patch.object(Path, 'write_text'):
                    result = worker.cleanup_resources(self.fixture, {
                        'fds': [], 'gpu_changed': True, 'modules_loaded': []})
                self.assertFalse(result['ok'])
                self.assertIn('restore inspection denied', result['reason'])
                self.assertTrue(result['residuals'])


class TestMixedPasidLimitOne(MixedPasidTestBase):
    """Only an attributed limit-one rejection is unsupported after verified cleanup."""

    def test_limit_one_control_rejection_skips(self):
        """No positive PASID exists when the observed exclusive bound is one."""
        text = self._text([self._event('entry', gpu_limit=1), self._event('return', ret=-22)])
        failure = trigger.TriggerError('attributed EINVAL')
        failure.__cause__ = OSError(errno.EINVAL, 'attach')
        with self.mock.patch.object(worker, '_xioctl', side_effect=failure) as ioctl:
            report = worker.run_requests(
                self.context, self.fixture, self.mock.Mock(return_value=text))
        self.assertEqual(report['status'], 'SKIP', report)
        self.assertIn('no usable positive PASID', report['reason'])
        self.assertEqual(report['bound'], 1)
        self.assertFalse(report['valid'])
        self.assertIsNone(report['detach_result'])
        self.assertEqual(report['trace'], text)
        ioctl.assert_called_once_with(42, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT,
                                      bytearray(struct.pack('<4I', 16, 1, 303, 1)))

    def test_limit_one_malformed_evidence_remains_failure(self):
        """Limit one cannot excuse missing attribution, malformed limits or trailing evidence."""
        records = [self._event('entry', gpu_limit=1), self._event('return', ret=-22)]
        variants = {}
        fields = {'pid': 999, 'tid': 999, 'member': self.fixture['parent'],
                  'parent': self.fixture['gpu'], 'gpu_iommu': 0, 'parent_iommu': 0,
                  'parent_limit': 2, 'pasid': 2, 'event': 'set'}
        for field, value in fields.items():
            trace = copy.deepcopy(records)
            trace[0][field] = value
            variants['wrong_' + field] = trace
        for field in records[0]:
            trace = copy.deepcopy(records)
            del trace[0][field]
            variants['missing_' + field] = trace
        for limit in (0, -1, True, False, '1', 1.0, None):
            trace = copy.deepcopy(records)
            trace[0]['gpu_limit'] = limit
            variants['limit_' + repr(limit)] = trace
        variants['wrong_order'] = list(reversed(records))
        variants['trailing_callback'] = records + [self._event('set')]
        variants['second_entry'] = records + records
        failure = trigger.TriggerError('attributed EINVAL')
        failure.__cause__ = OSError(errno.EINVAL, 'attach')
        for label, trace in variants.items():
            with self.subTest(malformed=label), \
                    self.mock.patch.object(worker, '_xioctl', side_effect=failure):
                report = worker.run_requests(self.context, self.fixture,
                                             self.mock.Mock(return_value=self._text(trace)))
                self.assertEqual(report['status'], 'FAIL', report)
                self.assertTrue(report['reason'])

    def test_limit_one_ioctl_disagreement_remains_failure(self):
        """A trace cannot replace the actual wrapped EINVAL ioctl outcome."""
        text = self._text([self._event('entry', gpu_limit=1), self._event('return', ret=-22)])
        for cause in (OSError(errno.EIO, 'attach'), ValueError('not ioctl'), None):
            with self.subTest(cause=repr(cause)):
                failure = trigger.TriggerError('attach failure')
                failure.__cause__ = cause
                with self.mock.patch.object(worker, '_xioctl', side_effect=failure if cause else None):
                    report = worker.run_requests(self.context, self.fixture,
                                                 self.mock.Mock(return_value=text))
                self.assertEqual(report['status'], 'FAIL', report)
                self.assertTrue(report['reason'])

    def test_limit_one_consistent_success_is_not_positive_proof(self):
        """Even a fully consistent success trace contradicts an exclusive PASID bound of one."""
        records = self._records()
        for record in records:
            record['pasid'] = 1
            if record['event'] == 'entry':
                record['gpu_limit'] = 1
        text = self._text(records)
        report = worker.evaluate_trace(text, self.fixture, self.pid, self.tid, 0)
        self.assertEqual(report['status'], 'FAIL', report)
        with self.mock.patch.object(worker, '_xioctl') as ioctl:
            report = worker.run_requests(
                self.context, self.fixture, self.mock.Mock(return_value=text))
        self.assertEqual(report['status'], 'FAIL', report)
        self.assertEqual(report['detach_result'], 0)
        self.assertEqual(ioctl.call_count, 2, 'consistent success still attaches and explicitly detaches only PASID 1')

    def test_limit_one_worker_skip_requires_successful_cleanup(self):
        """The real request classification survives only a successful once-owned cleanup."""
        run_requests = worker.run_requests
        text = self._text([self._event('entry', gpu_limit=1), self._event('return', ret=-22)])
        failure = trigger.TriggerError('attributed EINVAL')
        failure.__cause__ = OSError(errno.EINVAL, 'attach')
        for clean in (True, False):
            with self.subTest(cleanup_ok=clean):
                cleanup = {'ok': clean, 'reason': '' if clean else 'owned cleanup failed',
                           'residuals': [] if clean else ['owned resource remains']}
                observer = dict(self.observer, ok=True, lines=[text],
                                process=self.mock.Mock(poll=lambda: None))
                stack, mocked = self._worker_mocks(
                    prepare_observer=observer, allocate_context=self.context,
                    cleanup_resources=cleanup)
                mocked['run_requests'].side_effect = run_requests
                with stack, self.mock.patch.object(worker, '_xioctl', side_effect=failure):
                    result = worker.run_worker(self.mock.Mock(return_value=True))
                mocked['run_requests'].assert_called_once()
                mocked['cleanup_resources'].assert_called_once()
                self.assertEqual(result['status'], 'SKIP' if clean else 'FAIL', result)
                self.assertEqual(result['cleanup_ok'], clean)
                self.assertEqual(result['residuals'], cleanup['residuals'])
                self.assertIn('no usable positive PASID' if clean else 'owned cleanup failed',
                              result['reason'])
