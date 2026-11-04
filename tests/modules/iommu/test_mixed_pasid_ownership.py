# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict module inspection must preserve proven ownership and incomplete cleanup."""

import errno
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from modules.iommu.glt_test_iommu import _mixed_pasid_cleanup_readback as cleanup_readback
from tests.modules.iommu.test_mixed_pasid import native, worker
from tests.test_base import GLTTestBase


class TestMixedPasidModuleOwnership(GLTTestBase):
    """Exercise real acquisition and cleanup without devices or subprocesses."""

    def setUp(self):
        super().setUp()
        self.fixture = {
            "os_id": "ubuntu", "machine": "aarch64", "smmuv3": True, "tools_ready": True,
            "gpu": "0000:02:00.0", "parent": "0000:00:01.0", "direct_parent": "0000:00:01.0",
            "members": ["0000:02:00.0", "0000:00:01.0"], "gpu_driver": None,
            "gpu_override": "", "gpu_used": False, "parent_driver": "pcieport",
            "gpu_pasid_cap": True, "parent_pasid_cap": False, "module_plan": [],
        }
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        self.gpu_write = self.mock.Mock()
        self.observer = {
            "live": True, "ready": True, "marker_pid": 701, "pid": 703, "starttime": "123456",
            "argv": ["bpftrace", str(worker.OBSERVER_PATH), "701"], "program_ids": [11, 12, 13, 14],
            "process": self.mock.Mock(poll=lambda: None),
        }
        for target in ("os.open", "fcntl.ioctl", "subprocess.Popen", "subprocess.run", "os.kill"):
            self.guards.enter_context(self.mock.patch(
                target, side_effect=AssertionError(f"unmocked hardware/process call: {target}")))
        for name in ("read_text", "write_text"):
            self.guards.enter_context(self.mock.patch.object(
                Path, name, side_effect=AssertionError(f"unmocked sysfs call: {name}")))

    def _worker(self):
        """Mock I/O boundaries while retaining real acquisition and module cleanup."""
        stack = ExitStack()
        stack.enter_context(self.mock.patch.object(worker, "inspect_fixture", return_value=self.fixture))
        stack.enter_context(self.mock.patch.object(worker.os, "getpid", return_value=701))
        stack.enter_context(self.mock.patch.object(worker, "start_observer", return_value=self.observer))
        stack.enter_context(self.mock.patch.object(worker, "program_inventory", side_effect=[
            [], [{"id": value} for value in self.observer["program_ids"]]]))
        stack.enter_context(self.mock.patch.object(worker.time, "monotonic", return_value=0))
        stack.enter_context(self.mock.patch.object(worker.pci_utils, "get_pci_driver",
                                                 side_effect=[None, "vfio-pci", "pcieport"]))
        stack.enter_context(self.mock.patch.object(worker, "_override", side_effect=["", "vfio-pci"]))
        stack.enter_context(self.mock.patch.object(Path, "write_text", self.gpu_write))
        stack.enter_context(self.mock.patch.object(worker, "allocate_context", return_value={}))
        stack.enter_context(self.mock.patch.object(worker, "run_requests", return_value={"status": "PASS"}))
        stack.enter_context(self.mock.patch.object(worker, "restore_gpu", return_value=True))
        stack.enter_context(self.mock.patch.object(worker, "_cleanup_observer", return_value=[]))
        return stack

    def test_cleanup_inventory_owned_and_foreign_ids(self):
        """Cleanup permits foreign programs but never certifies a remaining owned ID."""
        cases = (([], False), ([{"id": 21}], False),
                 ([{"id": 11}], True), ([{"id": 21}, {"id": 11}], True))
        for programs, owned_remains in cases:
            with self.subTest(programs=programs), \
                    self.mock.patch.object(Path, "read_text", return_value=""), \
                    self.mock.patch.object(native.pci_utils, "get_pci_driver", side_effect=[None, "pcieport"]), \
                    self.mock.patch.object(native.glt_utils, "is_cli_available", return_value=True) as available, \
                    self.mock.patch.object(native.glt_utils, "run_command",
                                           return_value=SimpleNamespace(stdout=json.dumps(programs))) as inventory:
                if owned_remains:
                    with self.assertRaisesRegex(ValueError, "^independent cleanup inventory found owned BPF IDs$"):
                        cleanup_readback({"fixture": self.fixture, "modules_loaded": []}, self.observer)
                else:
                    self.assertIsNone(cleanup_readback(
                        {"fixture": self.fixture, "modules_loaded": []}, self.observer))
                available.assert_called_once_with("bpftool")
                inventory.assert_called_once_with(["bpftool", "-j", "prog", "show"],
                                                 "independent bpftool cleanup inventory failed", timeout=10)

    def test_cleanup_rejects_invalid_program_inventory(self):
        """Malformed or non-positive IDs cannot prove the owned BPF programs are gone."""
        cases = ([{"id": 0}], [{"id": -1}], {}, [11], [{}], [{"id": "11"}],
                 [{"id": True}], [{"id": False}], [{"id": None}], [{"id": 1.0}])
        for programs in cases:
            with self.subTest(programs=programs), \
                    self.mock.patch.object(Path, "read_text", return_value=""), \
                    self.mock.patch.object(native.pci_utils, "get_pci_driver", side_effect=[None, "pcieport"]), \
                    self.mock.patch.object(native.glt_utils, "is_cli_available", return_value=True) as available, \
                    self.mock.patch.object(native.glt_utils, "run_command",
                                           return_value=SimpleNamespace(stdout=json.dumps(programs))) as inventory:
                with self.assertRaisesRegex(ValueError, "^invalid independent bpftool JSON inventory$"):
                    cleanup_readback({"fixture": self.fixture, "modules_loaded": []}, self.observer)
                available.assert_called_once_with("bpftool")
                inventory.assert_called_once_with(["bpftool", "-j", "prog", "show"],
                                                 "independent bpftool cleanup inventory failed", timeout=10)

    def test_initial_inspection_fault_cannot_authorize_modprobe(self):
        """Unreadable initial state cannot be treated as permission to acquire a module."""
        self.fixture["module_plan"] = ["vfio_pci"]
        for error in (PermissionError(errno.EACCES, "denied"), OSError(errno.EIO, "inspection failed")):
            with self.subTest(error=type(error).__name__), self._worker(), \
                    self.mock.patch.object(Path, "exists", return_value=False), \
                    self.mock.patch.object(Path, "stat", side_effect=error), \
                    self.mock.patch("subprocess.run") as command:
                result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["reason"], str(error))
                self.assertTrue(result["cleanup_ok"])
                self.assertEqual(result["modules_loaded"], [])
                self.assertEqual(result["modules_preexisting"], [])
                command.assert_not_called()
                self.gpu_write.assert_not_called()

    def test_later_inspection_fault_retains_only_proven_owned_prefix(self):
        """A later fault retains prior ownership and prevents a false clean worker report."""
        self.fixture["module_plan"] = ["prior", "gone", "owned", "unreadable"]
        for error in (PermissionError(errno.EACCES, "denied"), OSError(errno.EIO, "inspection failed")):
            states = [SimpleNamespace(), FileNotFoundError(), FileNotFoundError(), FileNotFoundError(),
                      FileNotFoundError(), SimpleNamespace(), error, FileNotFoundError(), SimpleNamespace()]
            with self.subTest(error=type(error).__name__), self._worker(), \
                    self.mock.patch.object(Path, "exists", lambda path: path.name == "prior"), \
                    self.mock.patch.object(Path, "stat", side_effect=states), \
                    self.mock.patch("subprocess.run") as command:
                result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "FAIL")
                self.assertFalse(result["cleanup_ok"])
                self.assertEqual(result["modules_preexisting"], ["prior"])
                self.assertEqual(result["modules_loaded"], ["owned"])
                self.assertEqual(len(result["residuals"]), 1)
                self.assertIn("unreadable", result["residuals"][0])
                self.assertIn(str(error), result["residuals"][0])
                self.assertIn(str(error), result["reason"])
                self.assertEqual([call.args[0] for call in command.call_args_list],
                                 [["modprobe", "iommufd"], ["modprobe", "vfio_pci"], ["rmmod", "owned"]])

    def test_cleanup_inspection_fault_never_certifies_clean(self):
        """Owned and preexisting inspection faults remain residuals without unloading prior state."""
        for target in ("owned", "prior"):
            for error in (PermissionError(errno.EACCES, "denied"), OSError(errno.EIO, "inspection failed")):
                resources = {"fds": [], "modules_loaded": ["owned"], "modules_preexisting": ["prior"]}
                states = [error, SimpleNamespace()] if target == "owned" else [FileNotFoundError(), error]
                with self.subTest(target=target, error=type(error).__name__), \
                        self.mock.patch.object(Path, "exists", lambda path: path.name == "prior"), \
                        self.mock.patch.object(Path, "stat", side_effect=states), \
                        self.mock.patch("subprocess.run") as command:
                    result = worker.cleanup_resources(self.fixture, resources)
                    self.assertFalse(result["ok"])
                    self.assertEqual(len(result["residuals"]), 1)
                    self.assertIn(target, result["reason"])
                    self.assertIn(str(error), result["reason"])
                    self.assertEqual([call.args[0] for call in command.call_args_list], [["rmmod", "owned"]])

    def test_acquisition_absence_and_new_ownership_preserve_prior_modules(self):
        """Only modules absent initially and present after loading become worker-owned."""
        self.fixture["module_plan"] = ["prior", "owned", "gone"]
        for absence in (FileNotFoundError(), NotADirectoryError()):
            states = [SimpleNamespace(), absence, absence, SimpleNamespace(), absence, absence, SimpleNamespace()]
            with self.subTest(absence=type(absence).__name__), self._worker(), \
                    self.mock.patch.object(Path, "exists", side_effect=[
                        True, False, False, True, False, False, True]), \
                    self.mock.patch.object(Path, "stat", side_effect=states), \
                    self.mock.patch("subprocess.run") as command:
                result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "PASS")
                self.assertTrue(result["cleanup_ok"])
                self.assertEqual(result["modules_preexisting"], ["prior"])
                self.assertEqual(result["modules_loaded"], ["owned"])
                self.assertEqual([call.args[0] for call in command.call_args_list],
                                 [["modprobe", "iommufd"], ["modprobe", "vfio_pci"], ["rmmod", "owned"]])

    def test_partial_modprobe_failure_still_records_loaded_module(self):
        """A later modprobe failure must still clean the successfully loaded prefix."""
        self.fixture["module_plan"] = ["prior", "iommufd", "vfio_pci"]
        error = subprocess.CalledProcessError(1, ["modprobe", "vfio_pci"])
        states = [SimpleNamespace(), FileNotFoundError(), FileNotFoundError(), SimpleNamespace(),
                  FileNotFoundError(), FileNotFoundError(), SimpleNamespace()]
        with self._worker(), \
                self.mock.patch.object(Path, "exists", side_effect=[True, False, False, True, False, False, True]), \
                self.mock.patch.object(Path, "stat", side_effect=states), \
                self.mock.patch("subprocess.run", side_effect=[SimpleNamespace(), error, SimpleNamespace()]) as command:
            result = worker.run_worker(self.mock.Mock(return_value=True))
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["reason"], str(error))
            self.assertTrue(result["cleanup_ok"])
            self.assertEqual(result["modules_preexisting"], ["prior"])
            self.assertEqual(result["modules_loaded"], ["iommufd"])
            self.assertEqual([call.args[0] for call in command.call_args_list],
                             [["modprobe", "iommufd"], ["modprobe", "vfio_pci"], ["rmmod", "iommufd"]])

    def test_cleanup_presence_contract_preserves_preexisting_modules(self):
        """Missing owned modules are clean; remaining ones fail without unloading a prior module."""
        for state in (FileNotFoundError(), NotADirectoryError(), SimpleNamespace()):
            present = not isinstance(state, OSError)
            resources = {"fds": [], "modules_loaded": ["owned"], "modules_preexisting": ["prior"]}
            with self.subTest(state=type(state).__name__), \
                    self.mock.patch.object(Path, "exists", side_effect=[present, True]), \
                    self.mock.patch.object(Path, "stat", side_effect=[state, SimpleNamespace()]), \
                    self.mock.patch("subprocess.run") as command:
                result = worker.cleanup_resources(self.fixture, resources)
                self.assertEqual(result["ok"], not present)
                self.assertEqual(result["residuals"], ["module remains loaded: owned"] if present else [])
                self.assertEqual([call.args[0] for call in command.call_args_list], [["rmmod", "owned"]])
