# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression contracts found while checking the mixed-PASID worker lifecycle."""

import errno
import json
import queue
import signal
import stat
import struct
import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from common.pci_utils import get_pci_driver, is_arm_smmu_v3_backed
from common import glt_core as GLTCore
from modules.iommu.glt_test_iommu import (
    _mixed_pasid_first_message as first_worker_message,
    _mixed_pasid_cleanup_readback as cleanup_readback,
    _mixed_pasid_read_first_line as read_first_worker_line,
)
from tests.modules.iommu.test_mixed_pasid import MixedPasidTestBase, native, worker
from tests.test_base import GLTTestBase


class MixedPasidDiscoveryTestBase(GLTTestBase):
    """Domain-qualified discovery and concrete cleanup evidence must survive."""

    def setUp(self):
        super().setUp()
        self.gpus = ["0000:02:00.0", "0000:03:00.0"]
        self.parents = dict(zip(self.gpus, ["0000:00:01.0", "0000:00:02.0"]))
        self.fixture = {
            "os_id": "ubuntu", "machine": "aarch64", "tools_ready": True,
            "smmuv3": True, "gpu": self.gpus[0], "parent": self.parents[self.gpus[0]],
            "direct_parent": self.parents[self.gpus[0]], "members": [self.gpus[0], self.parents[self.gpus[0]]],
            "gpu_driver": None, "gpu_override": "", "gpu_used": False,
            "parent_driver": "pcieport", "gpu_pasid_cap": True, "parent_pasid_cap": False,
            "module_plan": [],
        }
        self.observer = {"pid": 703, "starttime": "123456", "argv": [
            "bpftrace", str(worker.OBSERVER_PATH), "701"], "program_ids": [11, 12, 13, 14],
            "ready": True, "live": True, "marker_pid": 701}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in ("os.open", "fcntl.ioctl", "os.kill", "subprocess.Popen", "subprocess.run"):
            self.stack.enter_context(self.mock.patch(
                target, side_effect=AssertionError(f"unmocked mutation: {target}")))
        self.stack.enter_context(self.mock.patch.object(
            worker, "_module_present", create=True,
            side_effect=lambda name: (Path("/sys/module") / name).exists()))

    def _discovery(self, raw, bound=(), resolve_modules=False):
        stack = ExitStack()
        values = {
            "platform.freedesktop_os_release": {"ID": "ubuntu"}, "platform.machine": "aarch64",
            "os.geteuid": 0, "shutil.which": "/usr/bin/tool", "_override": "", "_module_plan": [],
            "pci_utils.is_arm_smmu_v3_backed": True,
            "pci_utils.get_bdfs_by_pci_id": list(reversed(self.gpus)),
            "pci_utils.get_bdfs_with_ext_cap": self.gpus,
            "pci_utils.get_pci_class": "0x030200", "pci_utils.unsafe_pci_bdfs": set(),
        }
        if resolve_modules:
            del values['_module_plan']
        for name, value in values.items():
            stack.enter_context(self.mock.patch("tests.modules.iommu.test_mixed_pasid.worker." + name,
                                               return_value=value))
        stack.enter_context(self.mock.patch("common.nv_device.NvDev.get_nvidia_3d_bdf", return_value=raw))
        stack.enter_context(self.mock.patch.object(worker.pci_utils, "get_pci_parent",
                                                 side_effect=lambda gpu: (self.parents[gpu], None)))
        stack.enter_context(self.mock.patch.object(worker.pci_utils, "get_pci_driver", side_effect=lambda bdf:
                                                 "nvidia" if bdf in bound else
                                                 (None if bdf in self.gpus else "pcieport")))
        stack.enter_context(self.mock.patch.object(Path, "is_file", return_value=True))
        stack.enter_context(self.mock.patch.object(Path, "exists", return_value=False))
        stack.enter_context(self.mock.patch.object(Path, "glob", return_value=[]))
        stack.enter_context(self.mock.patch.object(Path, "read_text", return_value="0"))
        stack.enter_context(self.mock.patch.object(Path, "resolve", new=lambda path, **_: path))
        stack.enter_context(self.mock.patch.object(Path, "iterdir", autospec=True, side_effect=lambda path:
                                                 iter([]) if str(path) == '/sys/class/graphics' else
                                                 iter([Path(path.parts[-3]), Path(self.parents[path.parts[-3]])])))
        stack.enter_context(self.mock.patch("subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout="Root Port", stderr="")))
        return stack


class TestMixedPasidDiscovery(MixedPasidDiscoveryTestBase):
    """Domain-qualified discovery and concrete cleanup evidence must survive."""

    def test_domain_qualified_discovery_ignores_bare_legacy_output(self):
        """A domain-zero GPU must not disappear because legacy lspci omits its domain."""
        with self._discovery("02:00.0\n"):
            fixture = worker.inspect_fixture()
        self.assertEqual(fixture.get("gpu"), self.gpus[0])
        self.assertEqual(fixture.get("parent"), self.parents[self.gpus[0]])

    def test_multiple_gpus_select_first_eligible_pair(self):
        """A bound first GPU must not suppress another qualifying shared group."""
        with self._discovery("02:00.0\n03:00.0\n", bound=[self.gpus[0]]):
            fixture = worker.inspect_fixture()
        self.assertEqual(fixture.get("gpu"), self.gpus[1])
        self.assertEqual(set(fixture["members"]), {self.gpus[1], self.parents[self.gpus[1]]})

    def test_first_eligible_selection_is_deterministic(self):
        """Sort qualified PCI candidates instead of depending on enumeration order."""
        with self._discovery("03:00.0\n02:00.0\n"):
            fixture = worker.inspect_fixture()
            worker.pci_utils.get_bdfs_by_pci_id.assert_called_once_with("10de", timeout=10)
        self.assertEqual(fixture.get("gpu"), self.gpus[0])

    def test_failed_cleanup_preserves_child_evidence(self):
        """Concrete cleanup residuals and attributed trace must reach the native result."""
        process = self.mock.Mock(pid=701, returncode=1)
        report = {"type": "result", "status": "FAIL", "reason": "bpftool inventory failed",
                  "cleanup_ok": False, "residuals": ["owned BPF IDs remain: [11]"],
                  "trace": "attributed request evidence"}
        process.communicate.return_value = (json.dumps(report) + "\n", "")
        with self.mock.patch("subprocess.Popen", return_value=process), \
                self.mock.patch.object(native, "_mixed_pasid_first_message", return_value={
                    "type": "observer", "observer": self.observer}), \
                self.mock.patch.object(native, "_mixed_pasid_acknowledge"), \
                self.mock.patch.object(native, "_mixed_pasid_observer_identity", return_value=None):
            with self.assertRaises(GLTCore.GLTTestFail) as failure:
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
        self.assertEqual(str(failure.exception),
                         "bpftool inventory failed; worker cleanup attestation failed; residuals: "
                         "['owned BPF IDs remain: [11]', "
                         "'worker cleanup unverified: GPU binding/override, FDs and temporary modules', "
                         "'observer identity or stop unverified']")
        logged = json.loads(self.mock_logger.info.call_args.args[1])
        self.assertEqual(logged, report)
        process.kill.assert_not_called()
        process.terminate.assert_not_called()

    def test_observer_stays_live_until_first_mutation(self):
        """A handoff snapshot cannot authorize mutation after its observer has died."""
        observer = dict(self.observer, process=self.mock.Mock())
        observer["process"].poll.return_value = 1
        with self.mock.patch.object(worker, "inspect_fixture", return_value=self.fixture), \
                self.mock.patch.object(worker, "start_observer", return_value=observer), \
                self.mock.patch.object(worker, "program_inventory", return_value=[{"id": i} for i in range(11, 15)]), \
                self.mock.patch.object(worker.os, "getpid", return_value=701), \
                self.mock.patch.object(worker.pci_utils, "get_pci_driver",
                    side_effect=[None, "vfio-pci", "pcieport"]), \
                self.mock.patch.object(worker, "_override", side_effect=["", "vfio-pci"]), \
                self.mock.patch.object(Path, "write_text") as write, \
                self.mock.patch("subprocess.run") as mutate, \
                self.mock.patch.object(worker, "allocate_context", return_value={}), \
                self.mock.patch.object(worker, "run_requests", return_value={
                    "status": "PASS", "reason": "", "valid": True}), \
                self.mock.patch.object(worker, "cleanup_resources", return_value={
                    "ok": True, "reason": "", "residuals": []}):
            result = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("observer", result["reason"])
        write.assert_not_called()
        mutate.assert_not_called()

    def test_unavailable_modules_skip_before_ownership(self):
        """Unavailable VFIO/IOMMUFD is an ordinary prerequisite, without acquisition."""
        failure = subprocess.CalledProcessError(1, ["modprobe", "--show-depends", "iommufd"])
        with self._discovery("02:00.0\n"), \
                self.mock.patch.object(worker, "_module_plan", side_effect=failure), \
                self.mock.patch.object(worker, "prepare_observer") as observe, \
                self.mock.patch.object(worker, "allocate_context") as acquire:
            result = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(result["status"], "SKIP")
        self.assertTrue(result["cleanup_ok"])
        self.assertIn("fixture unavailable", result["reason"])
        observe.assert_not_called()
        acquire.assert_not_called()

    def test_unavailable_inventory_skips_without_unowned_cleanup(self):
        """Inventory failure before observer spawn leaves nothing to stop or mutate."""
        for failure in (FileNotFoundError("bpftool"), OSError("inventory failed"), ValueError("invalid JSON")):
            with self.subTest(inventory=str(failure)), \
                    self.mock.patch.object(worker, "inspect_fixture", return_value=self.fixture), \
                    self.mock.patch.object(worker, "program_inventory", side_effect=failure), \
                    self.mock.patch.object(worker, "start_observer") as observe, \
                    self.mock.patch.object(worker, "stop_observer") as stop, \
                    self.mock.patch.object(worker, "allocate_context") as acquire:
                result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "SKIP")
                self.assertTrue(result["cleanup_ok"])
                self.assertTrue(result["reason"])
                self.assertEqual(result["residuals"], [])
                observe.assert_not_called()
                stop.assert_not_called()
                acquire.assert_not_called()

    def test_failed_cleanup_trace_is_logged(self):
        """Failed-cleanup evidence must survive beyond the supervisor's return value."""
        process = self.mock.Mock(pid=701, returncode=1)
        report = {"type": "result", "status": "FAIL", "reason": "inventory failed",
                  "cleanup_ok": False, "residuals": ["owned BPF IDs remain"],
                  "trace": "attributed request evidence"}
        process.communicate.return_value = (json.dumps(report) + "\n", "")
        with self.mock.patch("subprocess.Popen", return_value=process), \
                self.mock.patch.object(native, "_mixed_pasid_first_message", return_value={
                    "type": "observer", "observer": self.observer}), \
                self.mock.patch.object(native, "_mixed_pasid_acknowledge"), \
                self.mock.patch.object(native, "_mixed_pasid_observer_identity", return_value=None):
            with self.assertRaises(GLTCore.GLTTestFail) as failure:
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
        self.assertEqual(str(failure.exception),
                         "inventory failed; worker cleanup attestation failed; residuals: "
                         "['owned BPF IDs remain', "
                         "'worker cleanup unverified: GPU binding/override, FDs and temporary modules', "
                         "'observer identity or stop unverified']")
        process.kill.assert_not_called()
        process.terminate.assert_not_called()
        self.assertTrue(any("attributed request evidence" in str(call.args)
                            for call in self.mock_logger.info.call_args_list),
                        "cleanup failure must retain its attributed trace in the runtime log")


    def test_owned_module_cleanup_preserves_preexisting_dependencies(self):
        """Public worker completion preserves prior modules even after partial acquisition."""
        def verify(partial):
            loaded = {"prior_dependency"}
            self.fixture["module_plan"] = ["prior_dependency", "iommufd", "vfio_pci"]
            observer = dict(self.observer, process=self.mock.Mock(poll=lambda: None))

            def command(args, **_):
                if args == ["modprobe", "vfio_pci"] and partial:
                    raise subprocess.CalledProcessError(1, args)
                if args[0] == "modprobe" and "-r" not in args:
                    loaded.add(args[-1])
                else:
                    loaded.discard(args[-1])
                    if args[0] == "modprobe":
                        loaded.discard("prior_dependency")
                return self.mock.Mock(stdout="")

            with self.mock.patch.object(worker, "inspect_fixture", return_value=self.fixture), \
                    self.mock.patch.object(worker.os, "getpid", return_value=701), \
                    self.mock.patch.object(worker, "start_observer", return_value=observer), \
                    self.mock.patch.object(worker, "program_inventory", side_effect=[
                        [], [{"id": value} for value in observer["program_ids"]], []]), \
                    self.mock.patch.object(worker, "stop_observer", return_value=True), \
                    self.mock.patch.object(worker, "allocate_context", return_value={}), \
                    self.mock.patch.object(worker, "run_requests", return_value={"status": "PASS"}), \
                    self.mock.patch.object(worker.pci_utils, "get_pci_driver",
                        side_effect=[None, "vfio-pci", "pcieport"]), \
                    self.mock.patch.object(worker, "_override", side_effect=["", "vfio-pci"]), \
                    self.mock.patch.object(worker, "restore_gpu", return_value=True), \
                    self.mock.patch.object(Path, "write_text"), \
                    self.mock.patch.object(Path, "exists", lambda path: path.name in loaded), \
                    self.mock.patch("subprocess.run", side_effect=command) as run:
                result = worker.run_worker(self.mock.Mock(return_value=True))
            self.assertIn("prior_dependency", loaded)
            self.assertTrue(result["cleanup_ok"], result)
            self.assertEqual(result["status"], "FAIL" if partial else "PASS")
            self.assertEqual(result["modules_preexisting"], ["prior_dependency"])
            self.assertEqual([call.args[0] for call in run.call_args_list if call.args[0][0] == "rmmod"],
                             [["rmmod", name] for name in reversed(result["modules_loaded"])])

        for partial in (False, True):
            with self.subTest(partial_load_failure=partial):
                verify(partial)

    def test_preexisting_module_loss_fails_cleanup_readback(self):
        """A cleanup attestation cannot hide disappearance of a preexisting module."""
        resources = {"fds": [], "modules_loaded": [], "modules_preexisting": ["prior_dependency"]}
        with self.mock.patch.object(Path, "exists", return_value=False):
            result = worker.cleanup_resources(self.fixture, resources)
        self.assertFalse(result["ok"])
        self.assertIn("prior_dependency", result["reason"])

    def test_parent_final_inventory_timeout_reports_incomplete(self):
        """The 180s child deadline must not lead into an unbounded final inventory."""
        process = self.mock.Mock(pid=701, returncode=0)
        report = {"type": "result", "status": "PASS", "valid": True,
                  "cleanup_ok": True, "residuals": [], "fixture": self.fixture, "modules_loaded": []}
        process.communicate.return_value = (json.dumps(report) + "\n", "")

        def inventory(args, **kwargs):
            if kwargs.get("timeout") is not None:
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            return self.mock.Mock(stdout="[]")

        with self.mock.patch("subprocess.Popen", return_value=process), \
                self.mock.patch.object(native, "_mixed_pasid_first_message", return_value={
                    "type": "observer", "observer": self.observer}), \
                self.mock.patch.object(native, "_mixed_pasid_acknowledge"), \
                self.mock.patch.object(native, "_mixed_pasid_stop_owned", return_value=True), \
                self.mock.patch.object(Path, "read_text", return_value="\n"), \
                self.mock.patch.object(Path, "is_symlink", return_value=False), \
                self.mock.patch.object(Path, "resolve", return_value=SimpleNamespace(name="pcieport")), \
                self.mock.patch.object(native.pci_utils, "get_pci_driver", side_effect=[None, "pcieport"]), \
                self.mock.patch.object(Path, "exists", return_value=False), \
                self.mock.patch("shutil.which", return_value="/usr/bin/bpftool"), \
                self.mock.patch("subprocess.run", side_effect=inventory) as run:
            with self.assertRaises(GLTCore.GLTTestFail) as failure:
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
        self.assertIn("mixed-PASID worker incomplete:", str(failure.exception))
        self.assertTrue(str(failure.exception).endswith("; residuals: "
            "['worker cleanup unverified: GPU binding/override, FDs and temporary modules']"))
        self.assertGreater(run.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(run.call_args.kwargs["timeout"], 60)


    def test_candidate_inspection_errors_fail_without_acquisition(self):
        """An unreadable fixture is a failed inspection, never an unsupported-platform skip."""
        failure = PermissionError("candidate inspection denied")
        with self._discovery("02:00.0\n03:00.0\n"), \
                self.mock.patch.object(worker, "_gpu_fixture", side_effect=failure), \
                self.mock.patch.object(worker, "prepare_observer") as observe:
            with self.subTest(boundary="inspection"):
                with self.assertRaises(worker.TriggerError) as caught:
                    worker.inspect_fixture()
                self.assertIs(caught.exception.__cause__, failure)
            with self.subTest(boundary="worker"):
                result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("candidate inspection denied", result["reason"])
                self.assertTrue(result["cleanup_ok"])
                observe.assert_not_called()

    def test_candidate_inspection_error_does_not_hide_later_eligible_gpu(self):
        """One inaccessible GPU must not suppress another genuinely eligible group."""
        eligible = dict(self.fixture, gpu=self.gpus[1], parent=self.parents[self.gpus[1]],
                        direct_parent=self.parents[self.gpus[1]],
                        members=[self.gpus[1], self.parents[self.gpus[1]]])
        with self._discovery("02:00.0\n03:00.0\n"), \
                self.mock.patch.object(worker, "_gpu_fixture", side_effect=[PermissionError("denied"), eligible]):
            fixture = worker.inspect_fixture()
        self.assertEqual(fixture["gpu"], self.gpus[1])
        self.assertFalse(worker.fixture_reason(fixture))

    def test_clean_unsupported_candidates_keep_ordinary_skip_reason(self):
        """Readable bound GPUs remain an ordinary unavailable-fixture prerequisite."""
        with self._discovery("02:00.0\n03:00.0\n", bound=self.gpus):
            fixture = worker.inspect_fixture()
        self.assertEqual(worker.fixture_reason(fixture), "GPU must be unused and unbound")

    def _console_sysfs(self, entries=(), target=None, failure=None, missing=()):
        """Model both old glob and explicit sysfs inspection without real I/O."""
        stack = ExitStack()
        gpu = Path('/sys/devices/pci0000:00') / self.gpus[0]

        def read(path, **_):
            if path.name == 'boot_vga':
                if 'boot_vga' in missing:
                    raise FileNotFoundError(str(path))
                if failure and failure[0] == 'boot_vga':
                    raise failure[1]
                return '0'
            raise AssertionError(f'unexpected sysfs read: {path}')

        def children(path):
            if str(path) == '/sys/class/graphics':
                if 'graphics' in missing:
                    raise FileNotFoundError(str(path))
                if failure and failure[0] == 'graphics':
                    raise failure[1]
                return iter(path / name for name in entries)
            if path.name == 'devices' and path.parent.name == 'iommu_group':
                return iter([Path(self.gpus[0]), Path(self.parents[self.gpus[0]])])
            raise AssertionError(f'unexpected sysfs enumeration: {path}')

        def resolve(path, **_):
            if str(path).startswith('/sys/class/graphics/'):
                if 'device' in missing:
                    raise FileNotFoundError(str(path))
                if failure and failure[0] == 'device':
                    raise failure[1]
                return target or gpu
            if path.name in self.gpus:
                if failure and failure[0] == 'gpu':
                    raise failure[1]
                return gpu
            raise AssertionError(f'unexpected sysfs resolve: {path}')

        stack.enter_context(self.mock.patch.object(worker.pci_utils, 'get_bdfs_by_pci_id',
                                                   return_value=[self.gpus[0]]))
        stack.enter_context(self.mock.patch.object(Path, 'read_text', new=read))
        stack.enter_context(self.mock.patch.object(Path, 'iterdir', new=children))
        stack.enter_context(self.mock.patch.object(Path, 'resolve', new=resolve))
        stack.enter_context(self.mock.patch.object(Path, 'glob', return_value=[
            Path('/sys/class/graphics') / name / 'device' for name in entries if name.startswith('fb')]))
        return stack

    def test_console_gpu_equal_nested_and_unrelated_targets(self):
        """Equal and descendant framebuffer ownership must refuse acquisition."""
        gpu = Path('/sys/devices/pci0000:00') / self.gpus[0]
        for target, used in ((gpu, True), (gpu / 'graphics', True), (gpu.parent / 'other', False)):
            with self.subTest(target=str(target)), self._discovery(''), \
                    self._console_sysfs(entries=['fb0'], target=target):
                fixture = worker.inspect_fixture()
                self.assertEqual(fixture['gpu_used'], used)
                if used:
                    self.assertEqual(worker.fixture_reason(fixture), 'GPU must be unused and unbound')
                    with self.mock.patch.object(worker, 'prepare_observer') as observe:
                        result = worker.run_worker(self.mock.Mock(return_value=True))
                    self.assertEqual(result['status'], 'SKIP')
                    observe.assert_not_called()

    def test_console_optional_absence_and_non_framebuffer_names(self):
        """fbcon and missing optional nodes cannot invent active ownership."""
        for entries, missing in ((['fbcon'], ()), (['other'], ()), (['fb0'], ('device',)),
                                 ([], ('boot_vga',)), ([], ('graphics',))):
            with self.subTest(entries=entries, missing=missing), self._discovery(''), \
                    self._console_sysfs(entries=entries, missing=missing):
                self.assertFalse(worker.fixture_reason(worker.inspect_fixture()))

    def test_console_inspection_errors_fail_before_observer(self):
        """Permission/I/O failures on optional nodes are refusal, not absence."""
        for node in ('boot_vga', 'graphics', 'device', 'gpu'):
            for error in (PermissionError('denied'), OSError('inspection I/O')):
                with self.subTest(node=node, error=type(error).__name__), self._discovery(''), \
                        self._console_sysfs(entries=['fb0'], failure=(node, error)), \
                        self.mock.patch.object(worker, 'prepare_observer', return_value={
                            'ok': False, 'reason': 'unexpected post-inspection observer'}) as observe:
                    result = worker.run_worker(self.mock.Mock(return_value=True))
                    self.assertEqual(result['status'], 'FAIL')
                    self.assertTrue(result['cleanup_ok'])
                    observe.assert_not_called()

    def test_discovery_uses_suite_root_contract(self):
        """Supported discovery must not duplicate the suite's enforced root gate."""
        with self._discovery(''), self._console_sysfs(), \
                self.mock.patch.object(worker.os, 'geteuid', return_value=1000):
            fixture = worker.inspect_fixture()
        self.assertTrue(fixture['tools_ready'])
        self.assertFalse(worker.fixture_reason(fixture))

    def test_worker_discovery_command_timeouts_preserve_classification(self):
        """Global pre-ACK timeout skips; one uninspectable candidate fails safely."""
        for boundary in ('module-plan', 'capabilities', 'candidates', 'candidate-inspection'):
            with self.subTest(boundary=boundary), \
                    self._discovery('', resolve_modules=boundary == 'module-plan'), \
                    self._console_sysfs(), ExitStack() as stack:
                def command(argv, **kwargs):
                    if kwargs.get('timeout') == 10:
                        raise subprocess.TimeoutExpired(argv, 10)
                    return SimpleNamespace(stdout='Root Port' if argv[0] == 'lspci' else '',
                                           returncode=0, stderr='')

                if boundary == 'module-plan':
                    stack.enter_context(self.mock.patch('subprocess.run', side_effect=command))
                elif boundary == 'candidate-inspection':
                    stack.enter_context(self.mock.patch('subprocess.run', side_effect=command))
                else:
                    name = 'get_bdfs_with_ext_cap' if boundary == 'capabilities' else 'get_bdfs_by_pci_id'
                    def discover(*_, **kwargs):
                        if kwargs.get('timeout') == 10:
                            raise subprocess.TimeoutExpired(['lspci'], 10)
                        return self.gpus
                    stack.enter_context(self.mock.patch.object(worker.pci_utils, name, side_effect=discover))
                observe = stack.enter_context(self.mock.patch.object(worker, 'prepare_observer',
                    return_value={'ok': False, 'reason': 'unexpected post-discovery observer'}))
                result = worker.run_worker(self.mock.Mock(return_value=True))
                self.assertEqual(result['status'], 'FAIL' if boundary == 'candidate-inspection' else 'SKIP')
                self.assertTrue(result['cleanup_ok'])
                if boundary != 'candidate-inspection':
                    self.assertIn('fixture unavailable', result['reason'])
                observe.assert_not_called()


class TestMixedPasidParentLookup(MixedPasidDiscoveryTestBase):
    """Distinguish failed parent inspection from ordinary root-bus absence."""

    def test_empty_gpu_discovery_reports_missing_pair(self):
        """No GPU cannot diagnose an IOMMU backing that was never inspected."""
        with self._discovery(''), \
                self.mock.patch.object(worker.pci_utils, 'get_bdfs_by_pci_id', return_value=[]), \
                self.mock.patch.object(worker, 'prepare_observer') as observe:
            fixture = worker.inspect_fixture()
            result = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(worker.fixture_reason(fixture), "GPU/direct-parent pair missing")
        self.assertEqual(result["reason"], "GPU/direct-parent pair missing")
        self.assertEqual(result["status"], "SKIP")
        self.assertTrue(result["cleanup_ok"])
        observe.assert_not_called()

    def test_parent_lookup_error_fails_inspection_and_worker(self):
        """Actual parent resolution errors cannot become an unsupported-fixture SKIP."""
        for boundary in ('inspection', 'worker'):
            with self.subTest(boundary=boundary), self._discovery(''), \
                    self.mock.patch.object(worker.pci_utils, 'get_pci_parent',
                                           return_value=(None, 'cannot resolve PCI parent')), \
                    self.mock.patch.object(worker, 'prepare_observer') as observe:
                if boundary == 'inspection':
                    with self.assertRaises(worker.TriggerError) as caught:
                        worker.inspect_fixture()
                    self.assertIn('cannot resolve', str(caught.exception))
                else:
                    result = worker.run_worker(self.mock.Mock(return_value=True))
                    self.assertEqual(result['status'], 'FAIL')
                    self.assertIn('cannot resolve', result['reason'])
                    self.assertTrue(result['cleanup_ok'])
                observe.assert_not_called()

    def test_parent_lookup_later_candidate_still_qualifies(self):
        """A failed parent lookup must not hide another genuinely eligible GPU."""
        def parent(gpu):
            if gpu == self.gpus[0]:
                return None, 'cannot resolve PCI parent'
            return self.parents[gpu], None

        with self._discovery(''), \
                self.mock.patch.object(worker.pci_utils, 'get_pci_parent', side_effect=parent):
            fixture = worker.inspect_fixture()
        self.assertEqual(fixture['gpu'], self.gpus[1])
        self.assertFalse(worker.fixture_reason(fixture))

    def test_parent_lookup_root_bus_remains_ordinary_skip(self):
        """An error-free root-bus absence is unsupported, not an inspection failure."""
        with self._discovery(''), \
                self.mock.patch.object(worker.pci_utils, 'get_pci_parent', return_value=(None, None)), \
                self.mock.patch.object(worker, 'prepare_observer') as observe:
            fixture = worker.inspect_fixture()
            result = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(worker.fixture_reason(fixture), "GPU/direct-parent pair missing")
        self.assertEqual(result["reason"], "GPU/direct-parent pair missing")
        self.assertEqual(result['status'], 'SKIP')
        self.assertTrue(result['cleanup_ok'])
        observe.assert_not_called()


class TestMixedPasidProbeIO(MixedPasidDiscoveryTestBase):
    """Real fixture and shared helpers must refuse failed lower-I/O inspection."""

    def _probe_io(self, site, bad_bdfs, code):
        stack = ExitStack()
        current = [None]

        def lstat(path, **_):
            bdf, leaf = str(path).split('/')[-2:]
            current[0] = bdf
            wanted = ((site == 'gpu-driver' and bdf in self.gpus and leaf == 'driver')
                      or (site == 'parent-driver' and bdf in self.parents.values() and leaf == 'driver')
                      or (site == 'smmu-link' and leaf == 'iommu'))
            if wanted and bdf in bad_bdfs:
                raise OSError(code, 'probe inspection denied')
            return SimpleNamespace(st_mode=stat.S_IFDIR if leaf == 'driver' and bdf in self.gpus
                                   else stat.S_IFLNK)

        def scandir(_path):
            if site == 'smmu-scan' and current[0] in bad_bdfs:
                raise OSError(code, 'probe inspection denied')
            context = self.mock.MagicMock()
            context.__enter__.return_value = iter([SimpleNamespace(
                name='arm-smmu-v3.0', path='/sys/devices/platform/arm-smmu-v3.0')])
            return context

        if site.startswith('smmu'):
            stack.enter_context(self.mock.patch.object(worker.pci_utils, 'is_arm_smmu_v3_backed',
                                                       new=is_arm_smmu_v3_backed))
            resolved = '/sys/devices/platform/arm-smmu-v3.0/iommu'
        else:
            stack.enter_context(self.mock.patch.object(worker.pci_utils, 'get_pci_driver', new=get_pci_driver))
            resolved = '/sys/bus/pci/drivers/pcieport'
        stack.enter_context(self.mock.patch('os.lstat', side_effect=lstat))
        stack.enter_context(self.mock.patch('os.scandir', side_effect=scandir))
        stack.enter_context(self.mock.patch('os.path.realpath', return_value=resolved))
        return stack

    def test_real_probe_io_faults_fail_before_observer(self):
        """All-candidate inspection errors FAIL truthfully before acquisition begins."""
        for site in ('gpu-driver', 'parent-driver', 'smmu-link', 'smmu-scan'):
            for code in (errno.EACCES, errno.EIO):
                with self.subTest(site=site, code=code), self._discovery(''), \
                        self._probe_io(site, set(self.gpus) | set(self.parents.values()), code), \
                        self.mock.patch.object(worker, 'prepare_observer', return_value={
                            'ok': False, 'reason': 'unexpected observer'}) as observe, \
                        self.mock.patch.object(worker, 'allocate_context') as acquire:
                    result = worker.run_worker(self.mock.Mock(return_value=True))
                    self.assertEqual(result['status'], 'FAIL')
                    self.assertIn('probe inspection denied', result['reason'])
                    self.assertTrue(result['cleanup_ok'])
                    self.assertEqual(result['residuals'], [])
                    observe.assert_not_called()
                    acquire.assert_not_called()

    def test_real_probe_fault_allows_later_eligible_candidate(self):
        """One uninspectable candidate does not conceal a genuinely eligible pair."""
        for site in ('parent-driver', 'smmu-link', 'smmu-scan'):
            with self.subTest(site=site), self._discovery(''), \
                    self._probe_io(site, {self.gpus[0], self.parents[self.gpus[0]]}, errno.EIO):
                fixture = worker.inspect_fixture()
                self.assertEqual(fixture['gpu'], self.gpus[1])
                self.assertFalse(worker.fixture_reason(fixture))

    def test_unreadable_first_gpu_is_not_selected_as_unbound(self):
        """A suppressed driver error must not make an uninspectable GPU qualify first."""
        with self._discovery(''), self._probe_io('gpu-driver', {self.gpus[0]}, errno.EIO):
            fixture = worker.inspect_fixture()
            self.assertEqual(fixture['gpu'], self.gpus[1])
            self.assertFalse(worker.fixture_reason(fixture))

    def test_real_smmu_initial_absence_remains_ordinary_skip(self):
        """Only genuine initial absence remains an unsupported fixture."""
        with self._discovery(''), self._probe_io('smmu-link', set(self.gpus), errno.ENOENT), \
                self.mock.patch.object(worker, 'prepare_observer') as observe:
            result = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(result['status'], 'SKIP')
        self.assertTrue(result['cleanup_ok'])
        observe.assert_not_called()

    def test_real_smmu_prefix_sibling_skips_before_observer(self):
        """A lexical prefix alone cannot authorize the mixed-PASID observer or acquisition."""
        node = '/sys/devices/platform/arm-smmu-v3.0'
        with self._discovery(''), self._probe_io('smmu-link', set(), errno.EIO), \
                self.mock.patch('os.path.realpath', side_effect=lambda path, **_:
                                node + '-sibling/iommu' if str(path).endswith('/iommu') else node), \
                self.mock.patch.object(worker, 'prepare_observer', return_value={
                    'ok': False, 'reason': 'unexpected observer'}) as observe, \
                self.mock.patch.object(worker, 'allocate_context') as acquire:
            result = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(result['status'], 'SKIP', result)
        self.assertEqual(result['reason'], 'GPU must be backed by ARM SMMUv3')
        self.assertTrue(result['cleanup_ok'])
        self.assertEqual(result['residuals'], [])
        observe.assert_not_called()
        acquire.assert_not_called()


class TestMixedPasidStartup(MixedPasidTestBase):
    """Discovery may finish late without consuming the post-ACK work allowance."""

    def _startup(self, arrival, handoff):
        messages = []

        def get(timeout):
            if arrival > timeout:
                raise queue.Empty("simulated startup arrival exceeds requested timeout")
            return messages.pop(0)

        process = self.mock.Mock(pid=self.pid, returncode=0)
        process.stdin.closed = False
        process.poll.return_value = None
        report = {"type": "result", "status": "PASS" if handoff else "SKIP",
                  "reason": "" if handoff else "ordinary prerequisite missing",
                  "cleanup_ok": True, "residuals": [], "valid": True}
        first = {"type": "observer", "observer": self.observer} if handoff else report
        process.stdout.readline.return_value = json.dumps(first) + "\n"
        process.communicate.return_value = (json.dumps(report) + "\n" if handoff else "", "")
        inbox = SimpleNamespace(put=messages.append, get=get)
        threads = SimpleNamespace(Thread=lambda target, daemon, args=():
                                  SimpleNamespace(start=lambda: target(*args)))
        stack = ExitStack()
        stack.enter_context(self.mock.patch.object(native, "queue", SimpleNamespace(
            Queue=lambda: inbox, Empty=queue.Empty)))
        stack.enter_context(self.mock.patch.object(native, "threading", threads))
        stack.enter_context(self.mock.patch("subprocess.Popen", return_value=process))
        stack.enter_context(self.mock.patch.object(native, "_mixed_pasid_observer_identity",
                                                  return_value=self.observer))
        readback = stack.enter_context(self.mock.patch.object(native, "_mixed_pasid_cleanup_readback"))
        kill = stack.enter_context(self.mock.patch("os.kill"))
        return stack, process, report, readback, kill

    def _assert_accepted(self, arrival, handoff):
        stack, process, report, readback, kill = self._startup(arrival, handoff)
        with stack:
            if handoff:
                self.assertIsNone(native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger))
            else:
                with self.assertRaises(GLTCore.GLTTestSkip) as skipped:
                    native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
                self.assertEqual(str(skipped.exception), report["reason"])
        process.communicate.assert_called_once_with(timeout=180)
        if handoff:
            process.stdin.write.assert_called_once_with("ACK\n")
            process.stdin.flush.assert_called_once_with()
            readback.assert_called_once_with(report, self.observer)
        else:
            process.stdin.write.assert_not_called()
            readback.assert_not_called()
        kill.assert_not_called()
        process.kill.assert_not_called()
        process.terminate.assert_not_called()

    def test_observer_after_35_seconds_reaches_ack_and_clean_result(self):
        """A valid late observer must authorize work and preserve all 180 work seconds."""
        self._assert_accepted(35, True)

    def test_result_after_35_seconds_retains_clean_skip(self):
        """Slow ordinary discovery must not become an incomplete-cleanup failure."""
        self._assert_accepted(35, False)

    def test_startup_boundary_accepts_result_and_observer(self):
        """The deterministic queue accepts availability at its inclusive boundary."""
        for handoff in (False, True):
            with self.subTest(handoff=handoff):
                self._assert_accepted(180, handoff)

    def test_startup_expiry_refuses_ack_without_killing_owner(self):
        """An unreceived handoff never authorizes mutation or a parent-owned stop."""
        for handoff in (False, True):
            with self.subTest(handoff=handoff):
                stack, process, _, readback, kill = self._startup(181, handoff)
                with stack, self.assertRaises(GLTCore.GLTTestFail) as failure:
                    native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
                self.assertEqual(str(failure.exception),
                    "mixed-PASID worker incomplete: simulated startup arrival exceeds requested timeout; residuals: "
                    "['worker cleanup unverified: GPU binding/override, FDs and temporary modules', "
                    "'observer ownership was not handed off']")
                process.stdin.write.assert_not_called()
                process.stdin.close.assert_called_once_with()
                process.communicate.assert_not_called()
                readback.assert_not_called()
                kill.assert_not_called()
                process.kill.assert_not_called()
                process.terminate.assert_not_called()


class TestMixedPasidPreservedContracts(MixedPasidTestBase):
    """Preserve transport bounds, live trace snapshots and owned cleanup failures."""

    def setUp(self):
        super().setUp()
        self.trace_lines = []

    def test_first_message_delivery_and_read_errors(self):
        """The real reader delivers one record and forwards its original I/O failure."""
        process = self.mock.Mock()
        process.stdout.readline.return_value = '{"type": "result"}\n'
        self.assertEqual(first_worker_message(process), {"type": "result"})
        for failure in (OSError("read denied"), ValueError("closed stream")):
            with self.subTest(failure=type(failure).__name__):
                process.stdout.readline.side_effect = failure
                with self.assertRaises(type(failure)) as outcome:
                    first_worker_message(process)
                self.assertIs(outcome.exception, failure)
        process.kill.assert_not_called()
        process.terminate.assert_not_called()

    def test_first_message_timeout_is_bounded_without_stopping_owner(self):
        """The 180-second queue bound must not terminate a possible live ioctl owner."""
        process = self.mock.Mock()
        with self.mock.patch.object(native.queue.Queue, "get", side_effect=queue.Empty) as receive:
            with self.assertRaises(queue.Empty):
                first_worker_message(process)
        receive.assert_called_once_with(timeout=180)
        process.kill.assert_not_called()
        process.terminate.assert_not_called()

    def _read_growing_trace(self, _context, _fixture, trace):
        self.assertEqual(trace(), "first\n")
        self.trace_lines.append("second\n")
        self.assertEqual(trace(), "first\nsecond\n")
        return {"status": "PASS", "reason": "", "valid": True}

    def test_request_trace_reads_see_appended_observations(self):
        """Request validation needs fresh observations rather than a frozen trace."""
        self.trace_lines = ["first\n"]
        observer = dict(self.observer, ok=True, lines=self.trace_lines,
                        process=self.mock.Mock(poll=lambda: None))
        stack, mocked = self._worker_mocks(prepare_observer=observer)
        mocked["run_requests"].side_effect = self._read_growing_trace
        with stack:
            report = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(report["status"], "PASS")
        mocked["run_requests"].assert_called_once()

    def test_inventory_errors_preserve_owned_failure_contract(self):
        """Underlying command errors retain worker evidence and stop only its owned observer."""
        failures = (None, "missing", subprocess.TimeoutExpired("bpftool", 10),
                    subprocess.CalledProcessError(1, "bpftool"), OSError("start denied"))
        for failure in failures:
            with self.subTest(failure=repr(failure)):
                report = {"type": "result", "status": "PASS", "reason": "worker evidence",
                          "valid": True, "cleanup_ok": True,
                          "residuals": ["original evidence"], "fixture": self.fixture, "modules_loaded": []}
                process = self.mock.Mock(pid=self.pid, returncode=0)
                process.stdout.readline.return_value = json.dumps({
                    "type": "observer", "observer": self.observer}) + "\n"
                process.communicate.return_value = (json.dumps(report) + "\n", "")
                with self.mock.patch("subprocess.Popen", return_value=process), \
                        self.mock.patch.object(native, "_mixed_pasid_observer_identity", return_value=self.observer), \
                        self.mock.patch.object(Path, "read_text", return_value=""), \
                        self.mock.patch.object(Path, "is_symlink", return_value=False), \
                        self.mock.patch.object(Path, "resolve", return_value=Path("/drivers/pcieport")), \
                        self.mock.patch.object(native.pci_utils, "get_pci_driver", side_effect=[None, "pcieport"]), \
                        self.mock.patch("shutil.which",
                                        return_value=None if failure == "missing" else "/usr/bin/bpftool"), \
                        self.mock.patch("subprocess.run",
                                        side_effect=failure if isinstance(failure, Exception) else None,
                                        return_value=SimpleNamespace(stdout="[]")) as command, \
                        self.mock.patch("os.kill") as stop, \
                        self.mock.patch.object(native.glt_utils, "write_sysfs_value") as mutation:
                    if failure is None:
                        self.assertIsNone(native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger))
                    else:
                        with self.assertRaises(GLTCore.GLTTestFail) as outcome:
                            native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
                        message = str(outcome.exception)
                        self.assertTrue(message.startswith("worker evidence; mixed-PASID worker incomplete: "))
                        self.assertTrue(message.endswith("; residuals: ['original evidence', "
                            "'worker cleanup unverified: GPU binding/override, FDs and temporary modules']"))
                    if failure not in (None, "missing"):
                        self.assertEqual(command.call_args.kwargs["timeout"], 10)
                    if failure == "missing":
                        command.assert_not_called()
                    if failure is None:
                        stop.assert_not_called()
                    else:
                        stop.assert_called_once_with(self.observer["pid"], signal.SIGTERM)
                    mutation.assert_not_called()
                    process.kill.assert_not_called()
                    process.terminate.assert_not_called()

    def test_worker_missing_inventory_tool_remains_file_not_found(self):
        """Absent prerequisites fail before any inventory command can run."""
        with self.mock.patch("shutil.which", return_value=None), \
                self.mock.patch("subprocess.run") as command:
            with self.assertRaisesRegex(FileNotFoundError, "bpftool is not on PATH"):
                worker.program_inventory()
        command.assert_not_called()


class TestMixedPasidFinalIntegration(MixedPasidTestBase):
    """Final-only contracts for the approved adapter and module-level reader."""

    def test_cleanup_inspection_errors_cannot_certify_absence(self):
        """Lossy Python 3.14 path probes must not turn inspection faults into cleanup."""
        for target in ("gpu", "parent", "module"):
            for error in (PermissionError(errno.EACCES, "denied"), OSError(errno.EIO, "inspection failed")):
                modes = {
                    "gpu": [error],
                    "parent": [FileNotFoundError(), error],
                    "module": [FileNotFoundError(), SimpleNamespace(st_mode=stat.S_IFLNK)],
                }[target]
                with self.subTest(target=target, error=type(error).__name__), \
                        self.mock.patch.object(Path, "read_text", return_value=""), \
                        self.mock.patch.object(Path, "is_symlink", return_value=False), \
                        self.mock.patch.object(Path, "resolve", return_value=Path("/drivers/pcieport")), \
                        self.mock.patch.object(Path, "exists", return_value=False), \
                        self.mock.patch.object(Path, "stat", side_effect=(
                            [FileNotFoundError(), error] if target == "module" else error)), \
                        self.mock.patch.object(worker.pci_utils.os, "lstat", side_effect=modes), \
                        self.mock.patch.object(worker.pci_utils.os.path, "realpath",
                                               return_value="/drivers/pcieport"), \
                        self.mock.patch.object(native.glt_utils, "is_cli_available", return_value=True), \
                        self.mock.patch.object(native.glt_utils, "run_command",
                                               return_value=SimpleNamespace(stdout="[]")) as inventory:
                    with self.assertRaises(OSError) as failed:
                        cleanup_readback({"fixture": self.fixture,
                                          "modules_loaded": ["gone", "vfio_pci"] if target == "module" else []},
                                         self.observer)
                    self.assertIs(failed.exception, error)
                    inventory.assert_not_called()

    def test_cleanup_module_presence_contract(self):
        """Absent components permit inventory; a remaining module still fails first."""
        for absence in (FileNotFoundError(), NotADirectoryError(), None):
            with self.subTest(absence=type(absence).__name__), \
                    self.mock.patch.object(Path, "read_text", return_value=""), \
                    self.mock.patch.object(Path, "is_symlink", return_value=False), \
                    self.mock.patch.object(Path, "resolve", return_value=Path("/drivers/pcieport")), \
                    self.mock.patch.object(Path, "exists", return_value=absence is None), \
                    self.mock.patch.object(Path, "stat", side_effect=absence), \
                    self.mock.patch.object(worker.pci_utils.os, "lstat", side_effect=[
                        FileNotFoundError(), SimpleNamespace(st_mode=stat.S_IFLNK)]), \
                    self.mock.patch.object(worker.pci_utils.os.path, "realpath", return_value="/drivers/pcieport"), \
                    self.mock.patch.object(native.glt_utils, "is_cli_available", return_value=True), \
                    self.mock.patch.object(native.glt_utils, "run_command",
                                           return_value=SimpleNamespace(stdout="[]")) as inventory:
                report = {"fixture": self.fixture, "modules_loaded": ["vfio_pci"]}
                if absence is None:
                    with self.assertRaisesRegex(ValueError, "independent temporary-module cleanup readback failed"):
                        cleanup_readback(report, self.observer)
                    inventory.assert_not_called()
                else:
                    cleanup_readback(report, self.observer)
                    inventory.assert_called_once()

    def test_cleanup_adapter_translates_errors(self):
        """The shared command adapter supplies its documented domain diagnostics."""
        failures = ((subprocess.TimeoutExpired("bpftool", 10), "independent bpftool cleanup inventory failed"),
                    (subprocess.CalledProcessError(1, "bpftool"), "independent bpftool cleanup inventory failed"),
                    (OSError("denied"), "Failed to run bpftool"))
        for error, message in failures:
            with self.subTest(error=type(error).__name__), \
                    self.mock.patch.object(Path, "read_text", return_value=""), \
                    self.mock.patch.object(Path, "is_symlink", return_value=False), \
                    self.mock.patch.object(Path, "resolve", return_value=Path("/drivers/pcieport")), \
                    self.mock.patch.object(worker.pci_utils, "get_pci_driver", side_effect=[None, "pcieport"]), \
                    self.mock.patch("shutil.which", return_value="/usr/bin/bpftool"), \
                    self.mock.patch("subprocess.run", side_effect=error) as command:
                with self.assertRaises(RuntimeError) as failure:
                    cleanup_readback(
                        {"fixture": self.fixture, "modules_loaded": []}, self.observer)
                self.assertEqual(str(failure.exception), message)
                command.assert_called_once_with(["bpftool", "-j", "prog", "show"],
                                                capture_output=True, text=True, check=True, timeout=10)

    def test_runtime_error_seam_stops_only_owned_observer(self):
        """A RuntimeError within supervision cannot escape the incomplete-cleanup outcome."""
        process = self.mock.Mock(pid=self.pid, returncode=0)
        report = {"type": "result", "status": "PASS", "valid": True,
                  "cleanup_ok": True, "reason": "worker reason", "residuals": ["worker residual"]}
        process.stdout.readline.return_value = json.dumps({"type": "observer", "observer": self.observer})
        process.communicate.return_value = (json.dumps(report), "")
        with self.mock.patch("subprocess.Popen", return_value=process), \
                self.mock.patch.object(native, "_mixed_pasid_observer_identity", return_value=self.observer), \
                self.mock.patch.object(native, "_mixed_pasid_cleanup_readback",
                                       side_effect=RuntimeError("adapter failed")), \
                self.mock.patch("os.kill") as stop:
            with self.assertRaises(GLTCore.GLTTestFail) as failure:
                native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger)
        self.assertEqual(str(failure.exception), "worker reason; mixed-PASID worker incomplete: adapter failed; "
                         "residuals: ['worker residual', "
                         "'worker cleanup unverified: GPU binding/override, FDs and temporary modules']")
        stop.assert_called_once_with(self.observer["pid"], signal.SIGTERM)
        process.kill.assert_not_called()
        process.terminate.assert_not_called()

    def test_reader_uses_explicit_module_target(self):
        """The reader callback has explicit ownership arguments and remains a daemon."""
        process = self.mock.Mock()
        with self.mock.patch.object(native.threading, "Thread") as thread, \
                self.mock.patch.object(native.queue.Queue, "get", return_value='{}'):
            self.assertEqual(first_worker_message(process), {})
        self.assertIs(thread.call_args.kwargs["target"], read_first_worker_line)
        self.assertIs(thread.call_args.kwargs["args"][0], process)
        self.assertTrue(thread.call_args.kwargs["daemon"])
        thread.return_value.start.assert_called_once_with()
        self.assertFalse(hasattr(native, "_mixed_pasid_supervise"))
        self.assertFalse(hasattr(worker, "_observer_text"))


class TestMixedPasidComposedStreams(MixedPasidTestBase):
    """Exercise real protocol and polling composition across growing streams."""

    def test_parent_ignores_composed_post_ack_nonresult_records(self):
        """Unrelated JSON dictionaries around one attestation cannot change its outcome."""
        report = {"type": "result", "status": "PASS", "reason": "complete evidence",
                  "cleanup_ok": True, "residuals": [], "valid": True}
        irrelevant = [{"type": "progress", "phase": "prefix"}, {"diagnostic": "infix"},
                      {"type": "progress", "phase": "suffix"}]
        for position in range(len(irrelevant) + 1):
            with self.subTest(result_position=position):
                records = irrelevant[:position] + [report] + irrelevant[position:]
                process = self.mock.Mock(pid=self.pid, returncode=0)
                process.stdout.readline.return_value = json.dumps({
                    "type": "observer", "observer": self.observer}) + "\n"
                process.communicate.return_value = (self._text(records), "")
                with self.mock.patch("subprocess.Popen", return_value=process), \
                        self.mock.patch.object(native, "_mixed_pasid_observer_identity",
                                               return_value=self.observer) as identity, \
                        self.mock.patch.object(native, "_mixed_pasid_cleanup_readback") as readback, \
                        self.mock.patch.object(native.glt_utils, "write_sysfs_value") as mutation, \
                        self.mock.patch("subprocess.run") as command, \
                        self.mock.patch("os.kill") as stop:
                    self.assertIsNone(native.test_mixed_pasid_attach(self.mock_sd, self.mock_logger))
                identity.assert_called_once_with(self.observer["pid"])
                process.stdout.readline.assert_called_once_with()
                process.stdin.write.assert_called_once_with("ACK\n")
                process.stdin.flush.assert_called_once_with()
                process.communicate.assert_called_once_with(timeout=180)
                readback.assert_called_once_with(report, self.observer)
                self.assertEqual(json.loads(self.mock_logger.info.call_args.args[1]), report)
                mutation.assert_not_called()
                command.assert_not_called()
                stop.assert_not_called()
                process.kill.assert_not_called()
                process.terminate.assert_not_called()

    def test_real_requests_poll_from_readiness_to_complete_attributed_trace(self):
        """Collect PASID 1 evidence for the full window after an initial readiness snapshot."""
        ready = self._text([{"event": "ready", "pid": self.pid}])
        complete = ready + self._text(self._records())
        read_trace = self.mock.Mock(side_effect=[ready, complete, complete])
        with self.mock.patch.object(worker, "_xioctl", side_effect=[None, None]) as ioctl, \
                self.mock.patch.object(worker, "OBSERVATION_TIMEOUT", 2.0), \
                self.mock.patch.object(worker.time, "monotonic", side_effect=[0.0, 0.25, 1.0, 2.0]) as clock, \
                self.mock.patch.object(worker.time, "sleep") as sleep:
            report = worker.run_requests(self.context, self.fixture, read_trace)
        self.assertEqual(report["status"], "PASS", report)
        self.assertTrue(report["valid"])
        self.assertEqual(report["bound"], self.bound)
        self.assertEqual(report["detach_result"], 0)
        self.assertEqual(report["trace"], complete)
        self.assertEqual(read_trace.call_count, 3)
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(sleep.call_args_list, [self.mock.call(0.02), self.mock.call(0.02)])
        self.assertEqual(ioctl.call_args_list, [
            self.mock.call(42, worker.VFIO_DEVICE_ATTACH_IOMMUFD_PT,
                           bytearray(struct.pack("<4I", 16, 1, 303, 1))),
            self.mock.call(42, worker.VFIO_DEVICE_DETACH_IOMMUFD_PT,
                           bytearray(struct.pack("<3I", 12, 1, 1)))])
