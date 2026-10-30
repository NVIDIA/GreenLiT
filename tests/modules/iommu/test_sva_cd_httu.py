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

"""Unit tests for the SVA CD hardware Access/Dirty (HTTU) discriminator."""
# pylint: disable=wrong-import-position,protected-access

import contextlib
import subprocess
import sys
from pathlib import Path
from unittest import mock

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from common import glt_core as GLTCore
from src.modules.iommu import glt_test_iommu as iommu

MOD = "src.modules.iommu.glt_test_iommu"

# Feature masks as printed by arm-smmu-v3 at probe.
HTTU_FEATURES = 0x05fe97bf     # HTTU-capable SMMU (HA and HD)
NO_HTTU_FEATURES = 0x001e1fbf  # SMMU without HTTU
# CD word 0 of a live SVA bind captured on an SMMU without HTTU.
CD0_NO_HTTU = 0x5ce205c0003550
CD0_HTTU = CD0_NO_HTTU | iommu.CD0_TCR_HA | iommu.CD0_TCR_HD
PID = 2581
# Captured on Vera when the probe read the wrong argument: V and EPD1 set,
# AA64 clear.
KERNEL_POINTER = 0xffff0000d285a080


def _log(*features):
    return "".join(
        f"arm-smmu-v3 arm-smmu-v3.{i}.auto: ias 48-bit, oas 48-bit "
        f"(features 0x{feat:08x})\n" for i, feat in enumerate(features))


def _trace(cd0, pid=PID, pasid=1):
    return (f"     sva_trigger-{pid}    [266] d.... 358506.702355: gltsvacd: "
            f"(arm_smmu_set_pasid+0x0/0x1b8) pasid={pasid} cd0={cd0:#x}\n")


class TestSvaCdHttuHelpers(GLTTestBase):
    """Pure helper coverage."""

    def test_smmu_features_parses_every_instance(self):
        """Each '(features 0x...)' line yields one mask."""
        self.assertEqual(iommu._smmu_features(_log(HTTU_FEATURES, NO_HTTU_FEATURES)),
                         [HTTU_FEATURES, NO_HTTU_FEATURES])
        self.assertEqual(iommu._smmu_features("no smmu here\n"), [])

    def test_required_bits_follow_common_features(self):
        """Only HA/HD reported by every SMMU are required."""
        self.assertEqual(iommu._required_sva_httu_bits([HTTU_FEATURES] * 4),
                         {"HA": iommu.CD0_TCR_HA, "HD": iommu.CD0_TCR_HD})
        self.assertEqual(iommu._required_sva_httu_bits([NO_HTTU_FEATURES]), {})
        self.assertEqual(
            iommu._required_sva_httu_bits([HTTU_FEATURES, NO_HTTU_FEATURES]), {})
        self.assertEqual(
            iommu._required_sva_httu_bits([iommu.SMMU_FEAT_HA]),
            {"HA": iommu.CD0_TCR_HA})

    def test_cds_filtered_by_pid_and_liveness(self):
        """Foreign PIDs, invalid, released (EPD0) and non-SVA-shaped CDs are ignored."""
        trace = (_trace(CD0_HTTU)
                 + _trace(CD0_NO_HTTU, pid=PID + 1)
                 + _trace(CD0_NO_HTTU & ~iommu.CD0_V)
                 + _trace(CD0_NO_HTTU | iommu.CD0_TCR_EPD0)
                 # A kernel pointer read by a probe on the wrong argument.
                 + _trace(KERNEL_POINTER)
                 + "unrelated line\n")
        self.assertEqual(iommu._sva_cds_for_pid(trace, PID), [(1, CD0_HTTU)])

    @mock.patch(f"{MOD}.platform.machine", return_value="x86_64")
    def test_workload_skip_reason_non_arm64(self, _machine):
        """Non-arm64 hosts are rejected before any device probing."""
        self.assertEqual(iommu._sva_workload_skip_reason(None),
                         "Not ARM64, skip the test")

    def test_cds_missing_bits(self):
        """Missing required bits are reported by name."""
        required = iommu._required_sva_httu_bits([HTTU_FEATURES])
        self.assertEqual(iommu._cds_missing_bits([(1, CD0_HTTU)], required), [])
        self.assertEqual(iommu._cds_missing_bits([(2, CD0_NO_HTTU)], required),
                         [(2, CD0_NO_HTTU, ["HA", "HD"])])

    def test_gpu_addressing_modes(self):
        """nvidia-smi modes are deduplicated; a failing query means unknown."""
        done = subprocess.CompletedProcess([], 0, stdout="ATS\nATS\nHMM\n\n")
        with mock.patch(f"{MOD}.subprocess.run", return_value=done):
            self.assertEqual(iommu._gpu_addressing_modes(), {"ATS", "HMM"})
        error = subprocess.CalledProcessError(2, "nvidia-smi")
        with mock.patch(f"{MOD}.subprocess.run", side_effect=error):
            self.assertEqual(iommu._gpu_addressing_modes(), set())
        with mock.patch(f"{MOD}.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(iommu._gpu_addressing_modes(), set())


@mock.patch(f"{MOD}.subprocess.run")
@mock.patch(f"{MOD}.glt_utils.chdir_managed",
            side_effect=lambda _path: contextlib.nullcontext())
@mock.patch(f"{MOD}._sva_workload_skip_reason", return_value=None)
@mock.patch(f"{MOD}._resolve_tracefs", return_value="/sys/kernel/tracing")
class TestSvaCdHttuGates(GLTTestBase):
    """Skip/fail/pass gates of test_sva_cd_hw_access_dirty."""

    def _run(self, kernel_log, trace, modes=frozenset()):
        with mock.patch(f"{MOD}.glt_utils.get_kernel_log", return_value=kernel_log), \
                mock.patch(f"{MOD}._gpu_addressing_modes", return_value=set(modes)), \
                mock.patch(f"{MOD}._run_with_kprobe", return_value=(trace, PID)) as kp:
            iommu.test_sva_cd_hw_access_dirty(None, mock.Mock())
        return kp

    def test_passes_when_cd_sets_ha_hd(self, *_):
        """HTTU SMMUs and a CD with HA/HD set must PASS."""
        kp = self._run(_log(HTTU_FEATURES, HTTU_FEATURES), _trace(CD0_HTTU))
        self.assertIn("arm_smmu_set_pasid", kp.call_args.kwargs["probe"])
        # Bind only: no munmap invalidation loop.
        self.assertEqual(["./sva_trigger", "0"], kp.call_args.args[1])

    def test_fails_when_cd_lacks_ha_hd(self, *_):
        """HTTU SMMUs and a CD without HA/HD must FAIL naming the commit."""
        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            self._run(_log(HTTU_FEATURES), _trace(CD0_NO_HTTU))
        self.assertIn("74fa4c177ad0", str(ctx.exception))
        self.assertIn("05dc22d5c4fa", str(ctx.exception))

    def test_skips_without_httu(self, *_):
        """SMMUs without HTTU must SKIP before running the workload."""
        with self.assertRaises(GLTCore.GLTTestSkip):
            self._run(_log(NO_HTTU_FEATURES), _trace(CD0_NO_HTTU))

    def test_skips_without_feature_line(self, *_):
        """No SMMU feature line in the log must SKIP."""
        with self.assertRaises(GLTCore.GLTTestSkip):
            self._run("", _trace(CD0_HTTU))

    def test_warns_when_no_sva_cd_captured(self, *_):
        """No CD after a successful bind must WARN as inconclusive."""
        for modes in (set(), {"ATS", "HMM"}, {"[N/A]"}):
            with self.subTest(modes=modes), \
                    self.assertRaises(GLTCore.GLTTestWarn) as ctx:
                self._run(_log(HTTU_FEATURES), _trace(CD0_HTTU, pid=PID + 1),
                          modes=modes)
            self.assertIn("inconclusive", str(ctx.exception))
            self.assertIn("may not use ATS", str(ctx.exception))

    def test_warns_when_ats_gpu_yields_no_sva_cd(self, *_):
        """No CD although every GPU uses ATS blames only the probe."""
        with self.assertRaises(GLTCore.GLTTestWarn) as ctx:
            self._run(_log(HTTU_FEATURES), _trace(CD0_HTTU, pid=PID + 1),
                      modes={"ATS"})
        self.assertIn("probe may not match", str(ctx.exception))
        self.assertNotIn("may not use ATS", str(ctx.exception))

    def test_skips_without_ats_gpu(self, *_):
        """GPUs without ATS addressing must SKIP before the workload runs."""
        with mock.patch(f"{MOD}.glt_utils.get_kernel_log",
                        return_value=_log(HTTU_FEATURES)), \
                mock.patch(f"{MOD}._gpu_addressing_modes",
                           return_value={"HMM", "None"}), \
                mock.patch(f"{MOD}._run_with_kprobe") as kp, \
                self.assertRaises(GLTCore.GLTTestSkip):
            iommu.test_sva_cd_hw_access_dirty(None, mock.Mock())
        kp.assert_not_called()

    def test_skips_when_prerequisites_missing(self, _tracefs, reason, *_):
        """A prerequisite gap must SKIP with its reason."""
        reason.return_value = "No NVIDIA 3D-class GPU found."
        with self.assertRaises(GLTCore.GLTTestSkip):
            self._run(_log(HTTU_FEATURES), _trace(CD0_HTTU))
