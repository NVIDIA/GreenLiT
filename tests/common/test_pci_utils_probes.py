# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PCI probes against temporary sysfs with lower-I/O fault injection."""

import errno
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from common.pci_utils import get_pci_driver, is_arm_smmu_v3_backed
from tests.test_base import GLTTestBase


class TestPciUtilsProbes(GLTTestBase):
    """Distinguish genuine absence from an inspection that cannot be completed."""

    @contextmanager
    def _sysfs(self):
        """Redirect only sysfs lower-I/O to a real temporary filesystem."""
        lstat, realpath, scandir = os.lstat, os.path.realpath, os.scandir
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def mapped(path):
                return root / str(path).lstrip('/') if str(path).startswith('/sys/') else path

            with patch('os.lstat', side_effect=lambda p, **kw: lstat(mapped(p), **kw)), \
                    patch('os.path.realpath', side_effect=lambda p, **kw: realpath(mapped(p), **kw)), \
                    patch('os.scandir', side_effect=lambda p: scandir(mapped(p))):
                yield root

    def test_driver_compatibility(self):
        """Bound, absent and confirmed nonlink drivers retain their normal values."""
        for kind in ('bound', 'absent', 'nonlink'):
            with self.subTest(kind=kind), self._sysfs() as root:
                device = root / 'sys/bus/pci/devices/0000:01:00.0'
                device.mkdir(parents=True)
                if kind == 'bound':
                    target = root / 'sys/bus/pci/drivers/vfio-pci'
                    target.mkdir(parents=True)
                    (device / 'driver').symlink_to(target, target_is_directory=True)
                elif kind == 'nonlink':
                    (device / 'driver').mkdir()
                with patch('os.path.realpath', wraps=os.path.realpath) as resolve:
                    self.assertEqual(get_pci_driver('0000:01:00.0'), 'vfio-pci' if kind == 'bound' else None)
                    if kind == 'absent':
                        resolve.assert_not_called()

    def test_smmu_compatibility(self):
        """Matching, nonmatching, empty-existing scan and ordinary absence remain Boolean."""
        for kind in ('match', 'nonmatch', 'no-node', 'absent', 'nonlink'):
            with self.subTest(kind=kind), self._sysfs() as root:
                device = root / 'sys/bus/pci/devices/0000:01:00.0'
                device.mkdir(parents=True)
                platform = root / 'sys/devices/platform'
                platform.mkdir(parents=True)
                if kind != 'no-node':
                    (platform / 'arm-smmu-v3.0').mkdir()
                if kind in ('match', 'no-node', 'nonmatch'):
                    target = platform / ('other' if kind == 'nonmatch' else 'arm-smmu-v3.0') / 'iommu'
                    target.mkdir(parents=True, exist_ok=True)
                    (device / 'iommu').symlink_to(target, target_is_directory=True)
                    if kind == 'no-node':
                        # Keep the scan directory existing and empty, with target elsewhere.
                        (device / 'iommu').unlink()
                        outside = root / 'other-iommu'
                        outside.mkdir()
                        (device / 'iommu').symlink_to(outside)
                        (platform / 'arm-smmu-v3.0/iommu').rmdir()
                        (platform / 'arm-smmu-v3.0').rmdir()
                elif kind == 'nonlink':
                    (device / 'iommu').mkdir()
                self.assertEqual(is_arm_smmu_v3_backed('0000:01:00.0'), kind == 'match')

    def test_smmu_resolved_node_requires_component_ancestry(self):
        """A resolved node and its child qualify; a prefix-sharing sibling does not."""
        for relation in ('exact', 'child', 'sibling'):
            with self.subTest(relation=relation), self._sysfs() as root:
                node = root / 'iommu-node'
                node.mkdir()
                target = (node if relation == 'exact' else node / 'child' if relation == 'child'
                          else root / 'iommu-node-sibling/iommu')
                target.mkdir(parents=True, exist_ok=True)
                platform = root / 'sys/devices/platform'
                platform.mkdir(parents=True)
                (platform / 'arm-smmu-v3.0').symlink_to(node, target_is_directory=True)
                device = root / 'sys/bus/pci/devices/0000:01:00.0'
                device.mkdir(parents=True)
                (device / 'iommu').symlink_to(target, target_is_directory=True)
                self.assertEqual(is_arm_smmu_v3_backed('0000:01:00.0'), relation != 'sibling')

    def test_initial_lstat_inspection_errors_propagate(self):
        """EACCES and EIO cannot be reinterpreted as unbound or unsupported."""
        for helper, leaf in ((get_pci_driver, 'driver'), (is_arm_smmu_v3_backed, 'iommu')):
            for code in (errno.EACCES, errno.EIO):
                with self.subTest(helper=helper.__name__, code=code), self._sysfs():
                    target = '/sys/bus/pci/devices/0000:01:00.0/' + leaf
                    lower = os.lstat
                    own_calls = []

                    def inspect(path, target=target, record_call=own_calls.append,
                                code=code, lower=lower, **kwargs):
                        if str(path) == target:
                            record_call(str(path))
                            raise OSError(code, 'probe denied', str(path))
                        return lower(path, **kwargs)

                    with patch('os.lstat', side_effect=inspect):
                        try:
                            with self.assertRaises(OSError):
                                helper('0000:01:00.0')
                        finally:
                            self.assertEqual(own_calls, [target])

    def test_broken_link_resolution_is_not_initial_absence(self):
        """After observing a symlink, failure to resolve its target must propagate."""
        for helper, leaf in ((get_pci_driver, 'driver'), (is_arm_smmu_v3_backed, 'iommu')):
            with self.subTest(helper=helper.__name__), self._sysfs() as root:
                device = root / 'sys/bus/pci/devices/0000:01:00.0'
                device.mkdir(parents=True)
                (root / 'sys/devices/platform').mkdir(parents=True)
                (device / leaf).symlink_to(root / 'missing')
                with self.assertRaises(FileNotFoundError):
                    helper('0000:01:00.0')

    def test_late_resolver_lstat_error_propagates(self):
        """A fault during resolution is distinct from the one successful initial probe."""
        for helper, leaf in ((get_pci_driver, 'driver'), (is_arm_smmu_v3_backed, 'iommu')):
            with self.subTest(helper=helper.__name__), self._sysfs() as root:
                device = root / 'sys/bus/pci/devices/0000:01:00.0'
                device.mkdir(parents=True)
                target = root / 'target'
                target.mkdir()
                (root / 'sys/devices/platform').mkdir(parents=True)
                (device / leaf).symlink_to(target)
                original = '/sys/bus/pci/devices/0000:01:00.0/' + leaf
                lower = os.lstat
                own_calls = []

                def inspect(path, original=original, record_call=own_calls.append,
                            denied_path=device / leaf, lower=lower, **kwargs):
                    if str(path) == original:
                        record_call(str(path))
                    if str(path) == str(denied_path):
                        raise OSError(errno.EIO, 'late resolver fault', str(path))
                    return lower(path, **kwargs)

                with patch('os.lstat', side_effect=inspect):
                    try:
                        with self.assertRaises(OSError):
                            helper('0000:01:00.0')
                    finally:
                        self.assertEqual(own_calls, [original])

    def test_smmu_scandir_error_propagates(self):
        """Enumeration after an observed IOMMU link must not silently become no nodes."""
        for code in (errno.EACCES, errno.EIO, errno.ENOENT):
            with self.subTest(code=code), self._sysfs() as root:
                device = root / 'sys/bus/pci/devices/0000:01:00.0'
                device.mkdir(parents=True)
                platform = root / 'sys/devices/platform'
                platform.mkdir(parents=True)
                (device / 'iommu').symlink_to(platform)
                with patch('os.scandir', side_effect=OSError(code, 'platform scan denied')):
                    with self.assertRaises(OSError):
                        is_arm_smmu_v3_backed('0000:01:00.0')

    def test_smmu_node_resolver_error_precedes_any_match(self):
        """A broken enumerated node cannot attest a nonmatching IOMMU target."""
        with self._sysfs() as root:
            device = root / 'sys/bus/pci/devices/0000:01:00.0'
            device.mkdir(parents=True)
            platform = root / 'sys/devices/platform'
            platform.mkdir(parents=True)
            other = root / 'other'
            other.mkdir()
            (device / 'iommu').symlink_to(other)
            (platform / 'arm-smmu-v3.0').symlink_to(root / 'missing')
            with self.assertRaises(FileNotFoundError):
                is_arm_smmu_v3_backed('0000:01:00.0')
