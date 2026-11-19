# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check device nomination and public PASS/FAIL/SKIP contracts without hardware."""

import logging
import subprocess
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest import TestCase, mock

from common import glt_core as GLTCore
from modules.virt import glt_test_kvm_cacheable_pfnmap as pfnmap


def _driver_module_absent(path):
    """Model sysfs before the optional GPU driver is loaded."""
    return str(path) != '/sys/bus/pci/drivers/' + pfnmap.GPU_DRIVER


def _reset_absent(path, **_):
    """Leave the device present while omitting its reset interface."""
    if Path(path).name == 'reset':
        raise FileNotFoundError('reset absent')


def _framebuffer_entries(_path):
    """Model an active framebuffer without reusing an exhausted iterator."""
    return iter([Path('/sys/class/graphics/fb0')])


class _UnusedGpuSysfs:
    """Own one unused GPU's sysfs state and its fault-injection callbacks."""

    def __init__(self, test_case):
        self.test_case = test_case
        self.values = {'driver': None, 'override': '(null)', 'power': 'auto'}
        self.module_loaded = False

    def driver(self, _bdf):
        """Expose driver ownership through the public PCI helper."""
        return self.values['driver']

    def read(self, path):
        """Read the supported device's identity and mutable sysfs fields."""
        name = Path(path).name
        identity = {'vendor': '0x10de', 'class': '0x030200', 'device': '0x2941',
                    'boot_vga': '0'}
        if name in identity:
            return identity[name]
        if name == 'driver_override':
            return self.values['override']
        if Path(path).parts[-2:] == ('power', 'control'):
            return self.values['power']
        raise AssertionError(f'Unexpected sysfs read: {path}')

    def write(self, path, value):
        """Model VFIO bind/unbind, including changes to runtime power policy."""
        name = Path(path).name
        if name == 'driver_override':
            self.values['override'] = value
        elif name == 'unbind':
            self.values.update(driver=None, power='on')
        elif name == 'bind':
            self.test_case.assertEqual(value, '0008:01:00.0')
            self.values.update(driver=pfnmap.GPU_DRIVER, power='auto')
        elif Path(path).parts[-2:] == ('power', 'control'):
            self.values['power'] = value
        else:
            raise AssertionError(f'Unexpected sysfs write: {path}')

    def entries(self, path, *, group_before=True, group_after='single', stage_override=False):
        """List the GPU, optionally changing its assignment or group at binding."""
        path = Path(path)
        if str(path) == '/sys/class/graphics':
            return iter([])
        if str(path) == '/sys/bus/pci/devices':
            return iter([path / '0008:01:00.0'])
        if str(path).endswith('/iommu_group/devices'):
            if stage_override:
                self.values['override'] = 'new-owner'
            bound = self.values['driver'] == pfnmap.GPU_DRIVER
            if (not bound and not group_before) or (bound and group_after == 'absent'):
                raise FileNotFoundError('group absent')
            members = ['0008:01:00.0']
            if bound and group_after == 'shared':
                members.append('0008:01:00.1')
            return iter(Path(member) for member in members)
        raise AssertionError(f'Unexpected sysfs listing: {path}')

    def read_value(self, path, *, name, value):
        """Replace one identity field while preserving other sysfs reads."""
        return value if Path(path).name == name else self.read(path)

    def write_error(self, path, value, *, name, error):
        """Fail one write without losing other sysfs effects."""
        if Path(path).name == name:
            raise error
        self.write(path, value)

    def write_override_restore_error(self, path, value):
        """Reject only restoration of the original empty driver override."""
        if Path(path).name == 'driver_override' and value == '':
            raise OSError('override restore failed')
        self.write(path, value)

    def read_bad_override(self, path):
        """Reject the temporary override's readback without breaking restoration."""
        value = self.read(path)
        if Path(path).name == 'driver_override' and value == pfnmap.GPU_DRIVER:
            return 'unexpected-override'
        return value

    def write_without_binding(self, path, value):
        """Accept writes while making a bind request have no effect."""
        if Path(path).name != 'bind':
            self.write(path, value)

    def probe(self, *_, returncode):
        """Require ownership before returning the configured native result."""
        self.test_case.assertEqual(self.values['driver'], pfnmap.GPU_DRIVER)
        return subprocess.CompletedProcess([], returncode, '', 'result')

    def autobind(self, *_, **__):
        """Claim a matching staged override during module registration."""
        self.values.update(driver=pfnmap.GPU_DRIVER, power='auto')
        return subprocess.CompletedProcess([], 0, '', '')

    def stage_assignment(self, *_, **__):
        """Simulate another owner staging an assignment during module loading."""
        self.module_loaded = True
        self.values['override'] = 'new-owner'
        return subprocess.CompletedProcess([], 0, '', '')

    def read_unreadable_assignment(self, path):
        """Fail override inspection only after module loading completes."""
        if self.module_loaded and Path(path).name == 'driver_override':
            raise PermissionError('staged override unreadable')
        return self.read(path)


class _OtherPciDevice:
    """Add a staged or disappearing device to the global override inventory."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.path = Path('/sys/bus/pci/devices/0009:01:00.0')
        self.fault = None

    def entries(self, path):
        """Include the second device, or reject an unreadable PCI inventory."""
        if str(path) == '/sys/bus/pci/devices':
            if self.fault == 'inventory':
                raise PermissionError('PCI inventory denied')
            return iter([Path(path) / '0008:01:00.0', self.path])
        return self.gpu.entries(path)

    def read(self, path):
        """Read the second device's override independently of the selected GPU."""
        if Path(path).parent == self.path:
            if self.fault == 'override':
                raise PermissionError('PCI override denied')
            if self.fault == 'removed':
                raise FileNotFoundError('device removed')
            return pfnmap.GPU_DRIVER
        return self.gpu.read(path)


class _RetryGpuInventory:
    """Own two GPUs and the setup, probe and cleanup behavior of each retry case."""

    def __init__(self, test_case, gpu):
        self.test_case = test_case
        self.gpu = gpu
        self.bdfs = ['0008:01:00.0', '0009:01:00.0']
        self.inventory = {self.bdfs[0]: gpu.values, self.bdfs[1]: dict(gpu.values)}
        self.scenario = {}

    def reset(self, first, second, cleanup_fault):
        """Start a case without retaining state from a preceding subtest."""
        self.scenario = {'first': first, 'second': second,
                         'cleanup_fault': cleanup_fault, 'discovered': False}
        for state in self.inventory.values():
            state.update(driver=None, override='(null)', power='auto')

    def driver(self, bdf):
        """Return the owner of the specific selected GPU."""
        return self.inventory[bdf]['driver']

    def read(self, path):
        """Keep each GPU's override and power policy separate."""
        path = Path(path)
        bdf = path.parts[5]
        state = self.inventory[bdf]
        if path.name == 'driver_override':
            if (self.scenario['first'] == 'inspection-error' and bdf == self.bdfs[0]
                    and self.scenario['discovered']):
                raise PermissionError('driver override inspection denied')
            return state['override']
        if path.parts[-2:] == ('power', 'control'):
            return state['power']
        return self.gpu.read(path)

    def write(self, path, value):
        """Track each acquisition and restoration, including unbind failure."""
        path = Path(path)
        bdf = value if path.name in ('bind', 'unbind') else path.parts[5]
        state = self.inventory[bdf]
        if path.name == 'driver_override':
            state['override'] = value
        elif path.name == 'bind':
            state.update(driver=pfnmap.GPU_DRIVER, power='auto')
        elif path.name == 'unbind':
            if self.scenario['cleanup_fault']:
                raise OSError('unbind failed')
            state.update(driver=None, power='on')
        elif path.parts[-2:] == ('power', 'control'):
            state['power'] = value
        else:
            raise AssertionError(path)

    def entries(self, path):
        """Introduce device-specific races when a candidate's group is inspected."""
        path = Path(path)
        if str(path) == '/sys/class/graphics':
            return iter([])
        if str(path) == '/sys/bus/pci/devices':
            return iter(path / bdf for bdf in self.bdfs)
        if str(path).endswith('/iommu_group/devices'):
            bdf = path.parts[5]
            kind = self.scenario['first'] if bdf == self.bdfs[0] else self.scenario['second']
            if bdf == self.bdfs[0]:
                self.scenario['discovered'] = True
            if kind == 'staged-override' and self.inventory[bdf]['driver'] is None:
                self.inventory[bdf]['override'] = 'new-owner'
            members = [bdf]
            if kind == 'shared-group' and self.inventory[bdf]['driver'] == pfnmap.GPU_DRIVER:
                members.append('0000:ff:00.0')
            return iter(Path(member) for member in members)
        raise AssertionError(path)

    def resolve(self, path, **_):
        """Give the two GPUs distinct VFIO groups."""
        return Path('/sys/kernel/iommu_groups') / ('7' if self.bdfs[0] in str(path) else '8')

    def stat(self, path):
        """Make only the configured GPU's VFIO nodes unavailable."""
        if str(path) in ('/dev/vfio/7', '/dev/vfio/noiommu-7'):
            kind = self.scenario['first']
        elif str(path) in ('/dev/vfio/8', '/dev/vfio/noiommu-8'):
            kind = self.scenario['second']
        else:
            kind = None
        if kind == 'missing-node':
            raise FileNotFoundError('VFIO node absent')

    def exists(self, path):
        """Expose a missing host module without marking a GPU unavailable."""
        return not (self.scenario['first'] == 'module-missing' and
                    not _driver_module_absent(path))

    def probe(self, _, bdf, _node, _logger):
        """Verify restoration before probing a second GPU and return its result."""
        self.test_case.assertLessEqual(self.test_case.probe.call_count, len(self.bdfs))
        self.test_case.assertEqual(self.inventory[bdf]['driver'], pfnmap.GPU_DRIVER)
        if bdf == self.bdfs[1]:
            override = 'new-owner' if self.scenario['first'] == 'staged-override' else ''
            self.test_case.assertEqual(self.inventory[self.bdfs[0]],
                                       {'driver': None, 'override': override, 'power': 'auto'})
        code = self.scenario['first'] if bdf == self.bdfs[0] else self.scenario['second']
        if code == 'timeout':
            raise subprocess.TimeoutExpired('probe', 90)
        self.test_case.assertIsInstance(code, int, 'An unavailable setup must not reach the probe')
        diagnostic = {3: 'USEMEM unavailable', 4: 'host capability unavailable'}.get(code, 'result')
        return subprocess.CompletedProcess([], code, '', diagnostic)


class _FilteredGpuInventory:
    """Present discovery with one example of each ineligible device."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.bdfs = [f'000{index}:01:00.0' for index in range(1, 9)]

    def driver(self, bdf):
        """Keep one GPU owned by the host driver."""
        return 'nvidia' if bdf == self.bdfs[3] else self.gpu.values['driver']

    def read(self, path):
        """Supply unsupported identities and an assignment staged by another owner."""
        path = Path(path)
        fields = {(self.bdfs[0], 'vendor'): '0x1234',
                  (self.bdfs[1], 'class'): '0x020000',
                  (self.bdfs[4], 'device'): '0xdead',
                  (self.bdfs[5], 'driver_override'): 'another-owner'}
        key = (path.parts[5], path.name)
        if key in fields:
            return fields[key]
        return self.gpu.read(path)

    def entries(self, path):
        """Give one otherwise eligible GPU a shared group."""
        path = Path(path)
        if str(path) == '/sys/bus/pci/devices':
            return iter(path / bdf for bdf in self.bdfs)
        if str(path) == '/sys/class/graphics':
            return iter([])
        if str(path).endswith('/iommu_group/devices'):
            bdf = path.parts[5]
            members = [bdf, '0007:01:00.1'] if bdf == self.bdfs[6] else [bdf]
            return iter(path / member for member in members)
        raise AssertionError(path)


class TestCacheablePfnmap(TestCase):  # pylint: disable=too-many-public-methods
    """Guest execution is validated separately on a real reserved DUT."""

    def setUp(self):
        self.gpu = _UnusedGpuSysfs(self)
        self.paths = {}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(
            pfnmap.os.environ, {pfnmap.BDF_ENV: "0008:01:00.0"},
        ))
        self.stack.enter_context(mock.patch.object(
            pfnmap.platform, "machine", return_value="aarch64",
        ))
        self.paths['exists'] = self.stack.enter_context(mock.patch.object(
            Path, "exists", autospec=True, return_value=True,
        ))
        self.stack.enter_context(mock.patch.object(pfnmap.os, "access", return_value=True))
        self.driver = self.stack.enter_context(mock.patch.object(
            pfnmap.pci_utils, "get_pci_driver", return_value="nvgrace_gpu_vfio_pci",
        ))
        self.paths['resolve'] = self.stack.enter_context(mock.patch.object(
            Path, "resolve", autospec=True, return_value=Path("/sys/kernel/iommu_groups/7"),
        ))
        self.paths['stat'] = self.stack.enter_context(mock.patch.object(
            Path, "stat", autospec=True, return_value=None,
        ))
        self.probe = self.stack.enter_context(mock.patch.object(
            pfnmap, "_run_probe", return_value=subprocess.CompletedProcess([], 0, "verified", ""),
        ))

    def test_verified_probe_passes(self):
        """A real probe's verified result is accepted as PASS."""
        pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.assertTrue(all(call.args == ("0008:01:00.0",)
                            for call in self.driver.call_args_list))
        self.probe.assert_called_once()

    def test_invalid_override_does_not_fall_back_to_discovery(self):
        """A malformed explicit choice never causes a different GPU to be used."""
        for bdf in ("../0008:01:00.0", "0008:01:00.8"):
            with self.subTest(bdf=bdf), \
                 mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: bdf}), \
                 self.assertRaises(GLTCore.GLTTestSkip):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.driver.assert_not_called()
        self.probe.assert_not_called()

    def test_platform_and_kvm_prerequisites(self):
        """Unsupported platforms skip; denied KVM access fails before acquisition."""
        cases = [("x86_64", True, True, GLTCore.GLTTestSkip),
                 ("aarch64", False, True, GLTCore.GLTTestSkip),
                 ("aarch64", True, False, GLTCore.GLTTestFail)]
        for machine, exists, access, outcome in cases:
            with self.subTest(machine=machine, exists=exists, access=access), \
                 mock.patch.object(pfnmap.platform, "machine", return_value=machine), \
                 mock.patch.object(Path, "exists", return_value=exists), \
                 mock.patch.object(pfnmap.os, "access", return_value=access), \
                 self.assertRaises(outcome):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.driver.assert_not_called()
        self.probe.assert_not_called()

    def test_absence_and_inspection_error_are_distinct(self):
        """An absent nominated group skips; unreadable discovery cannot hide as absence."""
        for driver in ("nvidia", "vfio-pci"):
            self.driver.return_value = driver
            with self.subTest(driver=driver), self.assertRaises(GLTCore.GLTTestSkip):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.driver.return_value = "nvgrace_gpu_vfio_pci"
        cases = [(self.paths['resolve'], FileNotFoundError("group absent"), GLTCore.GLTTestSkip),
                 (self.paths['stat'], FileNotFoundError("node absent"), GLTCore.GLTTestSkip),
                 (self.driver, FileNotFoundError("driver link broken"), GLTCore.GLTTestFail),
                 (self.driver, PermissionError("driver unreadable"), GLTCore.GLTTestFail),
                 (self.paths['resolve'], PermissionError("group unreadable"), GLTCore.GLTTestFail),
                 (self.paths['stat'], PermissionError("node unreadable"), GLTCore.GLTTestFail)]
        for probe, error, outcome in cases:
            probe.side_effect = error
            with self.subTest(error=error), self.assertRaises(outcome):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            probe.side_effect = None
        self.probe.assert_not_called()

    def _unused_gpu(self):
        """Model the public sysfs boundary, including VFIO's power-policy changes."""
        self.driver.side_effect = self.gpu.driver
        reader = self.stack.enter_context(mock.patch.object(
            pfnmap.glt_utils, "read_sysfs_value", side_effect=self.gpu.read,
        ))
        writer = self.stack.enter_context(mock.patch.object(
            pfnmap.glt_utils, "write_sysfs_value", side_effect=self.gpu.write,
        ))
        self.paths['iterdir'] = self.stack.enter_context(mock.patch.object(
            Path, "iterdir", autospec=True, side_effect=self.gpu.entries,
        ))
        self.stack.enter_context(mock.patch.object(pfnmap.pci_utils, "unsafe_pci_bdfs", return_value=set()))
        return self.gpu.values, reader, writer

    def test_unused_gpu_is_prepared_and_restored_for_all_probe_outcomes(self):
        """PASS, native FAIL/SKIP and launch errors all restore the original GPU."""
        values, _, _ = self._unused_gpu()
        for code, outcome in ((0, None), (1, GLTCore.GLTTestFail),
                              (3, GLTCore.GLTTestSkip), (4, GLTCore.GLTTestSkip),
                              (2, GLTCore.GLTTestFail)):
            with self.subTest(code=code):
                values.update(driver=None, override="saved-override", power="on")

                self.probe.side_effect = partial(self.gpu.probe, returncode=code)
                if outcome is None:
                    pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                else:
                    with self.assertRaises(outcome):
                        pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                self.assertEqual(values, {"driver": None, "override": "saved-override", "power": "on"})
        self.probe.side_effect = subprocess.TimeoutExpired("probe", 90)
        with self.assertRaises(GLTCore.GLTTestFail):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.assertIsNone(values["driver"])
        self.assertEqual(values["override"], "saved-override")

    def test_partial_bind_failure_restores_configuration(self):
        """A failed bind does not leave the temporary override or power policy."""
        values, _, writer = self._unused_gpu()
        writer.side_effect = partial(self.gpu.write_error, name='bind',
                                     error=PermissionError('bind denied'))
        with self.assertRaises(GLTCore.GLTTestFail) as raised:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.assertIn("bind denied", str(raised.exception))
        self.assertIsNone(values["driver"])
        self.assertEqual(values["override"], "")
        self.assertEqual(values["power"], "auto")
        self.probe.assert_not_called()

    def test_cleanup_failure_is_fail_and_post_hook_retries_owned_state(self):
        """Cleanup errors override success and retain state for the framework hook."""
        values, _, writer = self._unused_gpu()
        writer.side_effect = partial(self.gpu.write_error, name='unbind',
                                     error=OSError('unbind failed'))
        with self.assertRaises(GLTCore.GLTTestFail) as raised:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.assertIn("cleanup failed", str(raised.exception))
        attr = pfnmap.test_kvm_cacheable_pfnmap.attr
        self.assertTrue(attr.pfnmap_gpu_state)
        writer.side_effect = self.gpu.write
        attr.post_hooks.postfail(mock.Mock(), attr, mock.Mock())
        self.assertFalse(attr.pfnmap_gpu_state)
        self.assertIsNone(values["driver"])
        self.assertEqual(values["override"], "")
        self.assertEqual(values["power"], "auto")

    def test_restore_attempts_power_policy_after_override_error(self):
        """Independent cleanup fields are attempted even when one cannot restore."""
        values, _, writer = self._unused_gpu()
        writer.side_effect = self.gpu.write_override_restore_error
        with self.assertRaises(GLTCore.GLTTestFail):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.assertIsNone(values["driver"])
        self.assertEqual(values["power"], "auto")
        writer.side_effect = self.gpu.write
        attr = pfnmap.test_kvm_cacheable_pfnmap.attr
        attr.post_hooks.postfail(mock.Mock(), attr, mock.Mock())
        self.assertFalse(attr.pfnmap_gpu_state)

    def test_display_and_inspection_gates_prevent_binding(self):
        """Boot displays skip; a permission error is a failure before mutation."""
        _, reader, writer = self._unused_gpu()
        reader.side_effect = partial(self.gpu.read_value, name='boot_vga', value='1')
        with self.assertRaises(GLTCore.GLTTestSkip):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        reader.side_effect = PermissionError("inspection denied")
        with self.assertRaises(GLTCore.GLTTestFail):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_unused_gpu_skip_gates_do_not_mutate(self):
        """Unsupported hardware and missing safe prerequisites leave sysfs untouched."""
        _, reader, writer = self._unused_gpu()
        for gate in ("vendor", "class", "framebuffer", "reset", "absent"):
            with self.subTest(gate=gate), ExitStack() as scope:
                if gate in ("vendor", "class"):
                    value = "0x1234" if gate == "vendor" else "0x020000"
                    reader.side_effect = partial(self.gpu.read_value, name=gate, value=value)
                elif gate == "framebuffer":
                    scope.enter_context(mock.patch.object(
                        self.paths['iterdir'], 'side_effect', new=_framebuffer_entries,
                    ))
                elif gate == "reset":
                    scope.enter_context(mock.patch.object(
                        self.paths['stat'], 'side_effect', new=_reset_absent,
                    ))
                else:
                    scope.enter_context(mock.patch.object(
                        self.paths['stat'], 'side_effect', new=FileNotFoundError('device absent'),
                    ))
                with self.assertRaises(GLTCore.GLTTestSkip):
                    pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            reader.side_effect = self.gpu.read
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_unavailable_module_skips_without_device_writes(self):
        """Missing optional VFIO support does not fabricate a device mutation."""
        _, _, writer = self._unused_gpu()
        with mock.patch.object(self.paths['exists'], 'side_effect', new=_driver_module_absent), \
             mock.patch.object(pfnmap.glt_utils, "run_command", side_effect=RuntimeError('module absent')):
            with self.assertRaises(GLTCore.GLTTestSkip):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_module_autobinding_is_owned_and_restored(self):
        """A matching staged override may bind during modprobe and is still restored."""
        values, _, _ = self._unused_gpu()
        values['override'] = pfnmap.GPU_DRIVER

        with mock.patch.object(self.paths['exists'], 'side_effect', new=_driver_module_absent), \
             mock.patch.object(pfnmap.glt_utils, "run_command", side_effect=self.gpu.autobind):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.assertEqual(values, {'driver': None, 'override': pfnmap.GPU_DRIVER, 'power': 'auto'})
        self.probe.assert_called_once()

    def test_module_load_never_claims_an_unselected_assignment(self):
        """A second staged device prevents module loading in both selection modes."""
        values, reader, writer = self._unused_gpu()
        other = _OtherPciDevice(self.gpu)
        reader.side_effect = other.read
        self.paths['iterdir'].side_effect = other.entries
        for other_bdf in ('0009:01:00.0', '10000:01:00.0'):
            other.path = Path('/sys/bus/pci/devices') / other_bdf
            for nominated, override in (('', '(null)'), ('0008:01:00.0', '(null)'),
                                        ('0008:01:00.0', pfnmap.GPU_DRIVER)):
                values.update(driver=None, override=override, power='auto')
                with self.subTest(other=other_bdf, nominated=nominated, override=override), \
                     mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: nominated}), \
                     mock.patch.object(self.paths['exists'], 'side_effect', new=_driver_module_absent), \
                     mock.patch.object(pfnmap.glt_utils, 'run_command') as load, \
                     self.assertRaises(GLTCore.GLTTestSkip) as raised:
                    pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                self.assertIn(other_bdf, str(raised.exception))
                load.assert_not_called()
                self.assertEqual(values, {'driver': None, 'override': override, 'power': 'auto'})
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_module_safety_inspection_errors_fail_before_mutation(self):
        """An unreadable override inventory must not allow global driver probing."""
        values, reader, writer = self._unused_gpu()
        other = _OtherPciDevice(self.gpu)
        reader.side_effect = other.read
        self.paths['iterdir'].side_effect = other.entries
        for fault in ('inventory', 'override'):
            other.fault = fault
            with self.subTest(fault=fault), \
                 mock.patch.object(self.paths['exists'], 'side_effect', new=_driver_module_absent), \
                 mock.patch.object(pfnmap.glt_utils, 'run_command') as load, \
                 self.assertRaises(GLTCore.GLTTestFail):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            load.assert_not_called()
            self.assertEqual(values, {'driver': None, 'override': '(null)', 'power': 'auto'})
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_loaded_module_does_not_probe_other_staged_assignments(self):
        """An existing driver needs no global probing and leaves other overrides alone."""
        values, reader, writer = self._unused_gpu()
        other = _OtherPciDevice(self.gpu)
        reader.side_effect = other.read
        self.paths['iterdir'].side_effect = other.entries
        with mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
             mock.patch.object(pfnmap.glt_utils, 'run_command') as load:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        load.assert_not_called()
        self.probe.assert_called_once()
        for call in writer.call_args_list:
            self.assertNotIn(other.path.name, str(call.args[0]))
            self.assertNotEqual(other.path.name, call.args[1])
        self.assertEqual(values, {'driver': None, 'override': '', 'power': 'auto'})

    def test_disappearing_pci_override_does_not_block_safe_module_load(self):
        """A hot-removed device has no assignment for module registration to claim."""
        values, reader, _ = self._unused_gpu()
        other = _OtherPciDevice(self.gpu)
        other.fault = 'removed'
        reader.side_effect = other.read
        self.paths['iterdir'].side_effect = other.entries
        with mock.patch.object(self.paths['exists'], 'side_effect', new=_driver_module_absent), \
             mock.patch.object(pfnmap.glt_utils, 'run_command') as load:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        load.assert_called_once_with(['modprobe', pfnmap.GPU_DRIVER],
                                     'Grace GPU VFIO driver unavailable', timeout=30)
        self.probe.assert_called_once()
        self.assertEqual(values, {'driver': None, 'override': '', 'power': 'auto'})

    def test_primary_setup_error_survives_without_unnecessary_restore_writes(self):
        """A denied first write preserves its cause and does not claim mutated state."""
        values, _, writer = self._unused_gpu()
        writer.side_effect = PermissionError('override denied')
        with self.assertLogs('pfnmap-setup', level='ERROR') as captured, \
             self.assertRaises(GLTCore.GLTTestFail) as raised:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), logging.getLogger('pfnmap-setup'))
        self.assertIn('override denied', str(raised.exception))
        self.assertNotIn('cleanup failed', str(raised.exception))
        self.assertIn('override denied', '\n'.join(captured.output))
        self.assertEqual(values, {'driver': None, 'override': '(null)', 'power': 'auto'})
        self.assertFalse(pfnmap.test_kvm_cacheable_pfnmap.attr.pfnmap_gpu_state)

    def test_binding_validation_errors_fail_and_restore(self):
        """A bad readback or a bind that did nothing cannot pass the probe."""
        values, reader, writer = self._unused_gpu()
        for fault in ('override', 'bind'):
            with self.subTest(fault=fault):
                values.update(driver=None, override='(null)', power='auto')

                reader.side_effect = self.gpu.read_bad_override if fault == 'override' else self.gpu.read
                writer.side_effect = self.gpu.write_without_binding if fault == 'bind' else self.gpu.write
                with self.assertRaises(GLTCore.GLTTestFail):
                    pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                self.assertIsNone(values['driver'])
                self.assertEqual(values['override'], '')
                self.assertEqual(values['power'], 'auto')
        self.probe.assert_not_called()

    def test_automatic_discovery_runs_and_restores_without_an_override(self):
        """The normal user flow selects and restores an unused compatible GPU."""
        values, _, _ = self._unused_gpu()
        with mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
             self.assertLogs('pfnmap-discovery', level='INFO') as captured:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), logging.getLogger('pfnmap-discovery'))
        self.assertEqual(self.probe.call_args.args[1], '0008:01:00.0')
        self.assertIn('Using automatically selected VFIO GPU 0008:01:00.0', '\n'.join(captured.output))
        self.assertEqual(values, {'driver': None, 'override': '', 'power': 'auto'})

    def test_automatic_selection_retries_only_restored_unavailable_devices(self):
        """Retry only restored device-specific skips; host-wide skips stop selection."""
        _, reader, writer = self._unused_gpu()
        gpus = _RetryGpuInventory(self, self.gpu)
        bdfs = gpus.bdfs
        self.driver.side_effect = gpus.driver
        reader.side_effect = gpus.read
        writer.side_effect = gpus.write
        self.probe.side_effect = gpus.probe
        self.paths['iterdir'].side_effect = gpus.entries
        self.paths['resolve'].side_effect = gpus.resolve
        self.paths['stat'].side_effect = gpus.stat
        self.paths['exists'].side_effect = gpus.exists
        cases = [(3, 0, None, False, False),
                 (3, 3, GLTCore.GLTTestSkip, False, False),
                 (3, 4, GLTCore.GLTTestSkip, False, False),
                 ('shared-group', 0, None, False, False),
                 ('missing-node', 0, None, False, False),
                 ('staged-override', 0, None, False, False),
                 ('shared-group', 'missing-node', GLTCore.GLTTestSkip, False, False),
                 (3, 'missing-node', GLTCore.GLTTestSkip, False, False),
                 ('missing-node', 3, GLTCore.GLTTestSkip, False, False),
                 ('missing-node', None, GLTCore.GLTTestSkip, True, False),
                 ('shared-group', None, GLTCore.GLTTestFail, False, True),
                 ('module-missing', None, GLTCore.GLTTestSkip, False, False),
                 ('inspection-error', None, GLTCore.GLTTestFail, False, False),
                 (4, None, GLTCore.GLTTestSkip, False, False),
                 (4, None, GLTCore.GLTTestSkip, True, False),
                 (4, None, GLTCore.GLTTestFail, False, True),
                 (0, None, None, False, False),
                 (1, None, GLTCore.GLTTestFail, False, False),
                 (2, None, GLTCore.GLTTestFail, False, False),
                 (-7, None, GLTCore.GLTTestFail, False, False),
                 ('timeout', None, GLTCore.GLTTestFail, False, False),
                 (3, None, GLTCore.GLTTestFail, False, True),
                 (3, None, GLTCore.GLTTestSkip, True, False)]
        for first, second, outcome, explicit, cleanup_fault in cases:
            gpus.reset(first, second, cleanup_fault)
            self.probe.reset_mock()
            writer.reset_mock()
            with self.subTest(first=first, second=second, explicit=explicit, cleanup_fault=cleanup_fault), \
                 mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: bdfs[0] if explicit else ''}), \
                 mock.patch.object(pfnmap.glt_utils, 'run_command',
                                   side_effect=RuntimeError('module unavailable')) as loader:
                if outcome is None:
                    pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                else:
                    with self.assertRaises(outcome) as raised:
                        pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                    if first == second == 3:
                        self.assertIn('USEMEM unavailable', str(raised.exception))
                    if (first == 4 or second == 4) and not cleanup_fault:
                        self.assertIn('host capability unavailable', str(raised.exception))
                    if second == 'missing-node':
                        self.assertIn('has no available VFIO group node', str(raised.exception))
                    if first == 'inspection-error':
                        self.assertIn('inspection denied', str(raised.exception))
                self.assertEqual(loader.call_count, 1 if first == 'module-missing' else 0)
            setup_skips = {'shared-group', 'missing-node', 'staged-override'}
            retry = first == 3 or first in setup_skips
            attempted = bdfs if retry and not explicit and not cleanup_fault else bdfs[:1]
            kinds = {bdfs[0]: first, bdfs[1]: second}
            probed = [bdf for bdf in attempted
                      if kinds[bdf] not in setup_skips | {'module-missing', 'inspection-error'}]
            bound = [bdf for bdf in attempted
                     if kinds[bdf] not in ('staged-override', 'module-missing', 'inspection-error')]
            self.assertEqual([call.args[1] for call in self.probe.call_args_list], probed)
            self.assertEqual([call.args[1] for call in writer.call_args_list
                              if Path(call.args[0]).name == 'bind'], bound)
            if cleanup_fault:
                gpus.scenario['cleanup_fault'] = False
                attr = pfnmap.test_kvm_cacheable_pfnmap.attr
                attr.post_hooks.postfail(mock.Mock(), attr, mock.Mock())
            for bdf, state in gpus.inventory.items():
                override = '' if bdf in bound else '(null)'
                if bdf in attempted and kinds[bdf] == 'staged-override':
                    override = 'new-owner'
                self.assertEqual(state, {'driver': None, 'override': override,
                                         'power': 'auto'})

    def test_automatic_discovery_filters_ineligible_devices(self):
        """Unsafe, owned, unsupported, staged and shared-group GPUs are bypassed."""
        values, reader, writer = self._unused_gpu()
        gpus = _FilteredGpuInventory(self.gpu)
        bdfs = gpus.bdfs
        self.driver.side_effect = gpus.driver
        reader.side_effect = gpus.read
        self.paths['iterdir'].side_effect = gpus.entries
        with mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
             mock.patch.object(pfnmap.pci_utils, 'unsafe_pci_bdfs', return_value={bdfs[2]}), \
             self.assertLogs('pfnmap-filter', level='INFO') as captured:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), logging.getLogger('pfnmap-filter'))
        logged = '\n'.join(captured.output)
        for bdf in bdfs[:2]:
            self.assertNotIn(bdf, logged)
        for bdf in bdfs[2:-1]:
            self.assertIn(bdf, logged)
        self.assertIn('driver override is already staged', logged)
        self.assertIn('IOMMU group is shared', logged)
        self.assertEqual(self.probe.call_args.args[1], bdfs[-1])
        for call in writer.call_args_list:
            self.assertTrue(bdfs[-1] in str(call.args[0]) or call.args[1] == bdfs[-1])
        self.assertIsNone(values['driver'])

    def test_automatic_discovery_absence_skips_but_inspection_errors_fail(self):
        """An empty inventory is distinct from an unreadable safety assessment."""
        _, _, writer = self._unused_gpu()
        with mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
             mock.patch.object(pfnmap.os, 'access', return_value=False) as access:
            self.driver.side_effect = None
            self.driver.return_value = 'nvidia'
            with self.assertRaises(GLTCore.GLTTestSkip):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            with mock.patch.object(pfnmap.pci_utils, 'unsafe_pci_bdfs',
                                   side_effect=PermissionError('safety inspection denied')):
                with self.assertRaises(GLTCore.GLTTestFail) as raised:
                    pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            self.assertIn('safety inspection denied', str(raised.exception))
        access.assert_not_called()
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_empty_gpu_inventory_skips_before_kvm_checks(self):
        """A host without a candidate GPU skips even when KVM access is denied."""
        _, _, writer = self._unused_gpu()
        self.paths['iterdir'].side_effect = None
        self.paths['iterdir'].return_value = []
        with mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
             mock.patch.object(pfnmap.os, 'access', return_value=False) as access, \
             self.assertRaises(GLTCore.GLTTestSkip) as raised:
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.assertIn('No unused, isolated Grace GPU is available', str(raised.exception))
        self.paths['exists'].assert_not_called()
        access.assert_not_called()
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_automatic_candidate_checks_kvm_before_binding(self):
        """A suitable GPU still requires KVM availability and read/write access."""
        _, _, writer = self._unused_gpu()
        for exists, outcome in ((False, GLTCore.GLTTestSkip), (True, GLTCore.GLTTestFail)):
            with self.subTest(kvm_exists=exists), \
                 mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
                 mock.patch.object(self.paths['exists'], 'return_value', new=exists), \
                 mock.patch.object(pfnmap.os, 'access', return_value=False), \
                 self.assertRaises(outcome) as raised:
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            expected = 'Read/write access to /dev/kvm is required' if exists else 'KVM is unavailable'
            self.assertIn(expected, str(raised.exception))
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_explicit_choice_does_not_search_for_a_replacement(self):
        """An unavailable pinned GPU skips rather than touching another device."""
        self.driver.return_value = 'nvidia'
        with mock.patch.object(pfnmap.pci_utils, 'unsafe_pci_bdfs') as discovery, \
             self.assertRaises(GLTCore.GLTTestSkip):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        discovery.assert_not_called()
        self.probe.assert_not_called()

    def test_automatic_selection_rechecks_assignments_before_mutation(self):
        """A staged assignment arriving after discovery is not overwritten."""
        values, _, writer = self._unused_gpu()
        # Change the assignment immediately after its discovery-time read.
        self.paths['iterdir'].side_effect = partial(self.gpu.entries, stage_override=True)
        with mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
             self.assertRaises(GLTCore.GLTTestSkip):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        writer.assert_not_called()
        self.probe.assert_not_called()
        self.assertEqual(values['override'], 'new-owner')

    def test_group_change_after_binding_restores_without_running_guest(self):
        """A newly shared group is rejected before the native probe opens VFIO."""
        values, _, _ = self._unused_gpu()
        self.paths['iterdir'].side_effect = partial(self.gpu.entries, group_after='shared')
        with mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
             self.assertRaises(GLTCore.GLTTestSkip):
            pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
        self.probe.assert_not_called()
        self.assertEqual(values, {'driver': None, 'override': '', 'power': 'auto'})

    def test_group_created_during_binding_must_be_isolated_before_probe(self):
        """Late group creation permits only a singleton before guest execution."""
        values, _, _ = self._unused_gpu()
        for group_after in ('single', 'shared', 'absent'):
            values.update(driver=None, override='(null)', power='auto')
            self.probe.reset_mock()
            self.paths['iterdir'].side_effect = partial(self.gpu.entries, group_before=False,
                                               group_after=group_after)
            with self.subTest(group_after=group_after), \
                 mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}):
                if group_after == 'single':
                    pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                    self.probe.assert_called_once()
                else:
                    with self.assertRaises(GLTCore.GLTTestSkip):
                        pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
                    self.probe.assert_not_called()
                self.assertEqual(values, {'driver': None, 'override': '', 'power': 'auto'})

    def test_assignment_during_module_load_is_not_overwritten(self):
        """A newly staged assignment is preserved, even if its inspection fails."""
        values, reader, writer = self._unused_gpu()
        for fault in ('changed', 'unreadable'):
            values.update(driver=None, override='(null)', power='auto')
            self.gpu.module_loaded = False
            reader.side_effect = self.gpu.read_unreadable_assignment if fault == 'unreadable' else self.gpu.read
            outcome = GLTCore.GLTTestSkip if fault == 'changed' else GLTCore.GLTTestFail
            with self.subTest(fault=fault), \
                 mock.patch.dict(pfnmap.os.environ, {pfnmap.BDF_ENV: ''}), \
                 mock.patch.object(self.paths['exists'], 'side_effect', new=_driver_module_absent), \
                 mock.patch.object(pfnmap.glt_utils, 'run_command', side_effect=self.gpu.stage_assignment), \
                 self.assertRaises(outcome):
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            self.assertEqual(values, {'driver': None, 'override': 'new-owner', 'power': 'auto'})
        writer.assert_not_called()
        self.probe.assert_not_called()

    def test_native_result_classes(self):
        """Only unavailable prerequisites skip; setup cannot become PASS or patch proof."""
        cases = [(1, "BEHAVIOR: device instruction abort verified", GLTCore.GLTTestFail),
                 (1, "ABI: expected system EINVAL and VM capability=1", GLTCore.GLTTestFail),
                 (2, "PROBE: KVM_RUN non-conclusive: ioctl failed", GLTCore.GLTTestFail),
                 (2, "CLEANUP: scratch-byte restoration failed", GLTCore.GLTTestFail),
                 (3, "UNAVAILABLE: nominated VFIO group busy", GLTCore.GLTTestSkip),
                 (4, "UNAVAILABLE: VM capability reports no cacheable PFNMAP support",
                  GLTCore.GLTTestSkip),
                 (2, "PROBE: KVM_RUN non-conclusive: KVM_RUN timed out after 5 seconds",
                  GLTCore.GLTTestFail),
                 (-7, "", GLTCore.GLTTestFail),
                 (9, "unknown probe status", GLTCore.GLTTestFail)]
        for code, diagnostic, outcome in cases:
            self.probe.return_value = subprocess.CompletedProcess([], code, "", diagnostic)
            with self.subTest(code=code, diagnostic=diagnostic), \
                 self.assertRaises(outcome) as raised:
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            self.assertIn(diagnostic, str(raised.exception))
            if code == 2:
                self.assertIn("non-conclusive", str(raised.exception))
                self.assertNotIn("git.kernel.org", str(raised.exception))

    def test_probe_errors_fail_without_claiming_patch_absence(self):
        """Build/launch errors and outer timeouts never pass or imply a missing patch."""
        for error in (subprocess.TimeoutExpired("probe", 90),
                      OSError("execution failed"), RuntimeError("build failed")):
            self.probe.side_effect = error
            with self.subTest(error=error), self.assertRaises(GLTCore.GLTTestFail) as raised:
                pfnmap.test_kvm_cacheable_pfnmap(mock.Mock(), mock.Mock())
            self.assertIn("non-conclusive", str(raised.exception))

    def test_outer_timeout_preserves_partial_output(self):
        """Keep the GPU recovery diagnostic even when PCI cleanup succeeds."""
        values, _, _ = self._unused_gpu()
        cases = [(b"STIMULUS: guest instructions written\n", b"device status: \xff\n",
                  ("STIMULUS: guest instructions written", "device status: \ufffd")),
                 ("STIMULUS: original scratch bytes saved\n", "probe interrupted\n",
                  ("STIMULUS: original scratch bytes saved", "probe interrupted")),
                 (None, None, ())]
        for stdout, stderr, expected in cases:
            values.update(driver=None, override='(null)', power='auto')
            self.probe.side_effect = subprocess.TimeoutExpired(
                "probe", 90, output=stdout, stderr=stderr,
            )
            with self.subTest(stdout=stdout, stderr=stderr), \
                 self.assertLogs("pfnmap-timeout", level="INFO") as captured, \
                 self.assertRaises(GLTCore.GLTTestFail) as raised:
                pfnmap.test_kvm_cacheable_pfnmap(
                    mock.Mock(), logging.getLogger("pfnmap-timeout"),
                )
            logged = "\n".join(captured.output)
            self.assertIn("stdout", logged)
            self.assertIn("stderr", logged)
            for output in expected:
                self.assertIn(output, logged)
            self.assertIn("non-conclusive", str(raised.exception))
            self.assertNotIn("git.kernel.org", str(raised.exception))
            for message in ('GPU 0008:01:00.0', 'BAR4 scratch restoration is unverified',
                            'keep the GPU reserved for recovery even if PCI cleanup succeeds'):
                self.assertIn(message, str(raised.exception))
                self.assertIn(message, logged)
            self.assertEqual(values, {'driver': None, 'override': '', 'power': 'auto'})
            attr = pfnmap.test_kvm_cacheable_pfnmap.attr
            self.assertFalse(attr.pfnmap_gpu_state)
            attr.post_hooks.postfail(mock.Mock(), attr, logging.getLogger('pfnmap-timeout'))
            self.assertEqual(values, {'driver': None, 'override': '', 'power': 'auto'})
