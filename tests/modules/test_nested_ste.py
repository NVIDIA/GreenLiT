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

"""Unit tests for nested STE hitless-update helpers."""
# pylint: disable=protected-access,wrong-import-position

import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.glt_core import GLTTestResult, GLTTestSkip
from modules.iommu.nested_ste import trigger as nested_ste_trigger
from modules.iommu import glt_test_nested_ste as nested_ste


class TestNestedSteHelpers(GLTTestBase):
    """Test hardware-independent nested STE helper behavior."""

    def _ready_prechecks(self):
        """Patch hardware prechecks to reach the behavior under test."""
        stack = ExitStack()
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste.platform.machine",
            return_value="aarch64",
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste.os.path.exists",
            return_value=True,
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste.os.path.isdir",
            return_value=True,
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste.glob.glob",
            return_value=["/sys/devices/platform/arm-smmu-v3.0"],
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste._parse_running_kconfig",
            return_value={"CONFIG_ARM_SMMU_V3_IOMMUFD": "y"},
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste.glt_utils.is_cli_available",
            return_value=True,
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste._skip_reason_kprobe_targets",
            return_value=None,
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste._skip_reason_endianness",
            return_value=None,
        ))
        stack.enter_context(self.mock.patch(
            "modules.iommu.glt_test_nested_ste.pci_utils.get_bdfs_with_ext_cap",
            return_value=[],
        ))
        return stack

    def test_parse_trace_counts_disruptive_and_total_writes(self):
        """Trace parser reports disruptive STE writes and total observations.

        A transient V=0 counts as disruptive only on the nested update-safety
        transition (is_target=1, physical STE Config 6<->7)."""
        with TemporaryDirectory() as temp_dir:
            trace_log = Path(temp_dir) / "observe.log"
            trace_log.write_text(
                "PROBE_LIVE\n"
                "STE_WRITE cur_cfg=6 tgt_cfg=7 is_target=1 syncs=3 v_zero=0\n"
                "STE_WRITE cur_cfg=7 tgt_cfg=6 is_target=1 syncs=2 v_zero=1\n"
                "SUMMARY disruptive_writes=1\n",
                encoding="utf-8",
            )

            disruptive, ste_writes, target_writes = nested_ste._parse_trace(
                str(trace_log))

        self.assertEqual(ste_writes, 2)
        self.assertEqual(disruptive, 1)
        self.assertEqual(target_writes, 2)

    def test_parse_trace_ignores_v_zero_on_non_target_transition(self):
        """A V=0 rewrite on a non-6<->7 transition (benign install/teardown)
        must NOT count as disruptive, but must still count toward total writes."""
        with TemporaryDirectory() as temp_dir:
            trace_log = Path(temp_dir) / "observe.log"
            trace_log.write_text(
                "PROBE_LIVE\n"
                "STE_WRITE cur_cfg=0 tgt_cfg=6 is_target=0 syncs=2 v_zero=1\n"
                "STE_WRITE cur_cfg=7 tgt_cfg=0 is_target=0 syncs=1 v_zero=1\n",
                encoding="utf-8",
            )

            disruptive, ste_writes, target_writes = nested_ste._parse_trace(
                str(trace_log))

        self.assertEqual(ste_writes, 2)
        self.assertEqual(disruptive, 0)
        self.assertEqual(target_writes, 0)

    def test_parse_trace_ignores_non_ste_write_lines(self):
        """Only STE_WRITE lines contribute to parser counters."""
        with TemporaryDirectory() as temp_dir:
            trace_log = Path(temp_dir) / "observe.log"
            trace_log.write_text(
                "Attaching 5 probes...\n"
                "STE_WRITE cur_cfg=6 tgt_cfg=7 is_target=1 syncs=1 v_zero=0\n",
                encoding="utf-8",
            )

            disruptive, ste_writes, target_writes = nested_ste._parse_trace(
                str(trace_log))

        self.assertEqual(ste_writes, 1)
        self.assertEqual(disruptive, 0)
        self.assertEqual(target_writes, 1)

    def test_parse_trace_counts_target_transitions_for_liveness(self):
        """The parser must report how many is_target=1 (Config 6<->7) writes
        occurred so the verdict can require the transition under test was
        actually exercised before certifying it hitless."""
        with TemporaryDirectory() as temp_dir:
            trace_log = Path(temp_dir) / "observe.log"
            trace_log.write_text(
                "PROBE_LIVE\n"
                "STE_WRITE cur_cfg=6 tgt_cfg=7 is_target=1 syncs=3 v_zero=0\n"
                "STE_WRITE cur_cfg=7 tgt_cfg=6 is_target=1 syncs=3 v_zero=0\n"
                "STE_WRITE cur_cfg=0 tgt_cfg=6 is_target=0 syncs=2 v_zero=1\n"
                "SUMMARY disruptive_writes=0\n",
                encoding="utf-8",
            )

            disruptive, ste_writes, target_writes = nested_ste._parse_trace(
                str(trace_log))

        self.assertEqual(disruptive, 0)
        self.assertEqual(ste_writes, 3)
        self.assertEqual(target_writes, 2)

    def test_run_trigger_reports_trigger_error_as_fail_reason(self):
        """_run_trigger surfaces TriggerError text (not stuck) instead of raising."""
        with self.mock.patch(
            "modules.iommu.glt_test_nested_ste.trigger.run_trigger",
            side_effect=nested_ste_trigger.TriggerError("boom"),
        ):
            reason, worker_stuck = nested_ste._run_trigger(
                "0004:01:00.0", 1, logger=self.mock_logger)

        self.assertEqual(reason, "boom")
        self.assertFalse(worker_stuck)

    def test_run_trigger_flags_worker_stuck_on_timeout(self):
        """A TriggerTimeout (worker still stuck in a kernel ioctl holding fds)
        is surfaced with worker_stuck=True so the caller skips device restore."""
        with self.mock.patch(
            "modules.iommu.glt_test_nested_ste.trigger.run_trigger",
            side_effect=nested_ste_trigger.TriggerTimeout("timed out after 120s"),
        ):
            reason, worker_stuck = nested_ste._run_trigger(
                "0004:01:00.0", 32, logger=self.mock_logger)

        self.assertIn("timed out", reason)
        self.assertTrue(worker_stuck)

    def test_run_trigger_returns_none_on_success(self):
        """_run_trigger returns (None, False) when the ioctl sequence completes."""
        with self.mock.patch(
            "modules.iommu.glt_test_nested_ste.trigger.run_trigger",
            return_value="TRANSITIONS_OK iters=32",
        ):
            reason, worker_stuck = nested_ste._run_trigger(
                "0004:01:00.0", 32, logger=self.mock_logger)

        self.assertIsNone(reason)
        self.assertFalse(worker_stuck)

    def test_nested_ste_test_skips_on_missing_kprobe_targets(self):
        """Missing kprobe target symbols record SKIP (not FAIL)."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._skip_reason_kprobe_targets",
            return_value="kprobe target arm_smmu_write_entry missing from /proc/kallsyms",
        ):
            with self.assertRaises(GLTTestSkip):
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd, self.mock_logger)

        self.assertEqual(
            nested_ste.test_nested_ste_hitless_update.attr.result[0],
            GLTTestResult.SKIP,
        )

    def test_nested_ste_test_skips_without_arm_smmu_driver(self):
        """Missing arm-smmu-v3 driver records the specific platform SKIP."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.os.path.isdir",
            return_value=False,
        ):
            with self.assertRaises(GLTTestSkip) as ctx:
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd, self.mock_logger)

        self.assertIn("no arm-smmu-v3 platform driver", str(ctx.exception))

    def test_nested_ste_test_skips_without_arm_smmu_instance(self):
        """A present driver without probed instances records a distinct SKIP."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.glob.glob",
            return_value=[],
        ):
            with self.assertRaises(GLTTestSkip) as ctx:
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd, self.mock_logger)

        self.assertIn("no SMMU instances probed", str(ctx.exception))

    def test_nested_ste_test_skips_when_no_candidate_binds(self):
        """When candidates exist but none bind to vfio-pci, the test SKIPs."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._discover_candidate_bdfs",
            return_value=["0004:01:00.0", "0004:02:00.0"],
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._bind_usable_candidate",
            return_value=(None, ["0004:01:00.0: nope", "0004:02:00.0: nope"]),
        ):
            with self.assertRaises(GLTTestSkip) as ctx:
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd, self.mock_logger)

        self.assertIn("no usable vfio-pci endpoint", str(ctx.exception))
        self.assertEqual(
            nested_ste.test_nested_ste_hitless_update.attr.result[0],
            GLTTestResult.SKIP,
        )

    def test_nested_ste_test_skips_when_discovery_finds_nothing(self):
        """Decorated test records SKIP when discovery finds no safe PCI device."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._discover_candidate_bdfs",
            return_value=[],
        ):
            with self.assertRaises(GLTTestSkip) as ctx:
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd,
                    self.mock_logger,
                )

        self.assertIn("no suitable PCI BDF", str(ctx.exception))
        self.assertEqual(
            nested_ste.test_nested_ste_hitless_update.attr.result[0],
            GLTTestResult.SKIP,
        )

    def test_discover_candidate_bdfs_fail_closed_on_unsafe_oserror(self):
        """Unreadable route/mount tables yield None (fail-closed probe failure)."""
        with self.mock.patch(
            "modules.iommu.glt_test_nested_ste.pci_utils.unsafe_pci_bdfs",
            side_effect=OSError("neither route table readable"),
        ):
            result = nested_ste._discover_candidate_bdfs(
                set(), logger=self.mock_logger)

        self.assertIsNone(result)
        self.mock_logger.warning.assert_called()
        self.assertIn("fail-closed", self.mock_logger.warning.call_args[0][0])

    def test_display_device_safety_boundary(self):
        """Only an idle, isolated, resettable, non-console GPU is safe."""
        bdf = "0008:01:00.0"
        path = f"/sys/bus/pci/devices/{bdf}"
        cases = (
            ("safe", None, True, False, [bdf], [], True),
            ("bound", "nvidia", True, False, [bdf], [], False),
            ("boot-vga", None, True, True, [bdf], [], False),
            ("shared", None, True, False, [bdf, "0008:01:00.1"], [], False),
            ("framebuffer", None, True, False, [bdf],
             [f"{path}/graphics/fb0"], False),
            ("no-reset", None, False, False, [bdf], [], False),
        )
        for name, driver, reset, boot_vga, group, framebuffers, expected in cases:
            with self.subTest(name=name), self.mock.patch(
                "modules.iommu.glt_test_nested_ste._current_driver",
                return_value=driver,
            ), self.mock.patch(
                "modules.iommu.glt_test_nested_ste.os.path.isfile",
                side_effect=lambda candidate, reset=reset, boot_vga=boot_vga: (
                    reset if candidate.endswith("/reset") else boot_vga
                ),
            ), self.mock.patch(
                "modules.iommu.glt_test_nested_ste.glt_utils.read_sysfs_value",
                return_value="1" if boot_vga else "0",
            ), self.mock.patch(
                "modules.iommu.glt_test_nested_ste.glob.glob",
                side_effect=lambda pattern, group=group, framebuffers=framebuffers: (
                    [f"{path}/iommu_group/devices/{member}" for member in group]
                    if pattern.endswith("/iommu_group/devices/*") else framebuffers
                ),
            ):
                self.assertEqual(
                    nested_ste._display_device_is_safe(bdf), expected)

    def test_discover_candidate_bdfs_orders_pciutils_detected_ats_first(self):
        """ATS membership from pciutils takes priority over lexical BDF order."""
        paths = [
            "/sys/bus/pci/devices/0004:01:00.0",
            "/sys/bus/pci/devices/0004:02:00.0",
        ]
        with self.mock.patch(
            "modules.iommu.glt_test_nested_ste.pci_utils.unsafe_pci_bdfs",
            return_value=set(),
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.glob.glob",
            return_value=paths,
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.os.path.islink",
            return_value=True,
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._is_arm_smmu_v3_backed",
            return_value=True,
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.pci_utils.get_pci_class",
            return_value="0x020000",
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._current_driver",
            return_value="mlx5_core",
        ):
            result = nested_ste._discover_candidate_bdfs(
                {"0004:02:00.0"}, logger=self.mock_logger)

        self.assertEqual(result, ["0004:02:00.0", "0004:01:00.0"])

    def test_discover_candidate_bdfs_without_ats_uses_bdf_order(self):
        """An empty ATS set keeps the documented MEV-only BDF fallback."""
        paths = [
            "/sys/bus/pci/devices/0004:02:00.0",
            "/sys/bus/pci/devices/0004:01:00.0",
        ]
        with self.mock.patch(
            "modules.iommu.glt_test_nested_ste.pci_utils.unsafe_pci_bdfs",
            return_value=set(),
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.glob.glob",
            return_value=paths,
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.os.path.islink",
            return_value=True,
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._is_arm_smmu_v3_backed",
            return_value=True,
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.pci_utils.get_pci_class",
            return_value="0x020000",
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._current_driver",
            return_value="mlx5_core",
        ):
            result = nested_ste._discover_candidate_bdfs(
                set(), logger=self.mock_logger)

        self.assertEqual(result, ["0004:01:00.0", "0004:02:00.0"])

    def test_nested_ste_test_skips_on_missing_bpftrace(self):
        """Decorated test records SKIP when bpftrace tooling is missing."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.glt_utils.is_cli_available",
            side_effect=lambda tool: tool != "bpftrace",
        ):
            with self.assertRaises(GLTTestSkip):
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd,
                    self.mock_logger,
                )

        self.assertEqual(
            nested_ste.test_nested_ste_hitless_update.attr.result[0],
            GLTTestResult.SKIP,
        )

    def test_nested_ste_test_skips_on_missing_lspci(self):
        """Decorated test records SKIP when pciutils tooling is missing."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste.glt_utils.is_cli_available",
            side_effect=lambda tool: tool != "lspci",
        ):
            with self.assertRaises(GLTTestSkip) as ctx:
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd,
                    self.mock_logger,
                )

        self.assertIn("required tool lspci missing", str(ctx.exception))
        self.assertEqual(
            nested_ste.test_nested_ste_hitless_update.attr.result[0],
            GLTTestResult.SKIP,
        )

    def test_nested_ste_test_skips_on_missing_iommufd_config(self):
        """Decorated test records SKIP when nested iommufd support is absent."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._parse_running_kconfig",
            return_value={},
        ):
            with self.assertRaises(GLTTestSkip):
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd,
                    self.mock_logger,
                )

        self.assertEqual(
            nested_ste.test_nested_ste_hitless_update.attr.result[0],
            GLTTestResult.SKIP,
        )

    def test_nested_ste_test_accepts_modular_iommufd_config(self):
        """CONFIG_ARM_SMMU_V3_IOMMUFD=m proceeds past the config precheck."""
        with self._ready_prechecks(), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._parse_running_kconfig",
            return_value={"CONFIG_ARM_SMMU_V3_IOMMUFD": "m"},
        ), self.mock.patch(
            "modules.iommu.glt_test_nested_ste._discover_candidate_bdfs",
            return_value=[],
        ):
            with self.assertRaises(GLTTestSkip) as ctx:
                nested_ste.test_nested_ste_hitless_update(
                    self.mock_sd, self.mock_logger)

        self.assertIn("no suitable PCI BDF", str(ctx.exception))


class TestNestedSteTrigger(GLTTestBase):
    """Test the pure-Python ioctl trigger module in isolation from GLT."""

    def test_struct_sizes_match_documented_uapi_abi(self):
        """Flat Struct layouts and remaining ctypes mirrors match UAPI sizes."""
        expected_struct_sizes = {
            nested_ste_trigger._IOAS_ALLOC: 12,
            nested_ste_trigger._IOAS_MAP: 40,
            nested_ste_trigger._HWPT_ALLOC: 48,
            nested_ste_trigger._HW_INFO: 40,
            nested_ste_trigger._VIOMMU_ALLOC: 40,
            nested_ste_trigger._VDEVICE_ALLOC: 24,
            nested_ste_trigger._VFIO_BIND: 24,
            nested_ste_trigger._VFIO_ATTACH: 16,
        }
        for layout, expected in expected_struct_sizes.items():
            with self.subTest(fmt=layout.format):
                self.assertEqual(layout.size, expected)

        expected_ctypes_sizes = {
            nested_ste_trigger.IommuHwptArmSmmuv3: 16,
            nested_ste_trigger.IommuHwInfoArmSmmuv3: 40,
        }
        for struct_type, expected in expected_ctypes_sizes.items():
            with self.subTest(struct=struct_type.__name__):
                self.assertEqual(
                    nested_ste_trigger.ctypes.sizeof(struct_type), expected)

        for layout, (_name, expected) in nested_ste_trigger._ABI.items():
            with self.subTest(abi=layout.format):
                self.assertEqual(layout.size, expected)
        for cls, expected in nested_ste_trigger._CTYPES_ABI.items():
            with self.subTest(ctypes_abi=cls.__name__):
                self.assertEqual(nested_ste_trigger.ctypes.sizeof(cls), expected)

    def test_ioctl_request_numbers_use_plain_io_encoding(self):
        """Request numbers match _IO(type, nr) == (type << 8) | nr."""
        self.assertEqual(nested_ste_trigger.IOMMU_IOAS_ALLOC, 0x3B81)
        self.assertEqual(nested_ste_trigger.IOMMU_IOAS_MAP, 0x3B85)
        self.assertEqual(nested_ste_trigger.IOMMU_HWPT_ALLOC, 0x3B89)
        self.assertEqual(nested_ste_trigger.IOMMU_GET_HW_INFO, 0x3B8A)
        self.assertEqual(nested_ste_trigger.IOMMU_VIOMMU_ALLOC, 0x3B90)
        self.assertEqual(nested_ste_trigger.IOMMU_VDEVICE_ALLOC, 0x3B91)
        self.assertEqual(nested_ste_trigger.VFIO_DEVICE_BIND_IOMMUFD, 0x3B76)
        self.assertEqual(nested_ste_trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT, 0x3B77)

    def test_uapi_constants_preserve_wire_values(self):
        """UAPI constants remain identical to their kernel definitions."""
        self.assertEqual(
            nested_ste_trigger._IOAS_MAP_FIXED_IOVA
            | nested_ste_trigger._IOAS_MAP_WRITEABLE
            | nested_ste_trigger._IOAS_MAP_READABLE,
            0x7,
        )
        self.assertEqual(nested_ste_trigger._HWPT_ALLOC_NEST_PARENT, 0x1)
        self.assertEqual(nested_ste_trigger._HWPT_DATA_ARM_SMMUV3, 0x2)
        self.assertEqual(nested_ste_trigger._HW_INFO_INPUT_TYPE, 0x1)
        self.assertEqual(nested_ste_trigger._HW_INFO_TYPE_ARM_SMMUV3, 0x2)
        self.assertEqual(nested_ste_trigger._VIOMMU_TYPE_ARM_SMMUV3, 0x1)
        self.assertEqual(nested_ste_trigger._VIOMMU_TYPE_TEGRA241_CMDQV, 0x2)
        self.assertEqual(nested_ste_trigger._STE_CFG_BYPASS, 0x4)
        self.assertEqual(nested_ste_trigger._STE_CFG_S1_TRANS, 0x5)
        self.assertEqual(nested_ste_trigger._STE_S1FMT_64K_L2, 0x2)
        self.assertEqual(nested_ste_trigger._STE_S1DSS_BYPASS, 0x1)

    def test_cd_map_len_derives_from_runtime_page_size(self):
        """CD_MAP_LEN tracks the actual runtime page size."""
        self.assertEqual(nested_ste_trigger.CD_MAP_LEN, nested_ste_trigger.mmap.PAGESIZE)

    def test_ste_make_s1dss_bypass_matches_expected_bit_layout(self):
        """S1DSS-bypass STE qwords match the documented bitfield layout."""
        w0, w1 = nested_ste_trigger._ste_make_s1dss_bypass(0x10000)

        self.assertEqual(w0, 0x1002B)
        self.assertEqual(w1, 0x1)

    def test_ste_make_nested_s2_bypass_matches_expected_bit_layout(self):
        """S2-bypass STE qwords match the documented bitfield layout."""
        w0, w1 = nested_ste_trigger._ste_make_nested_s2_bypass()

        self.assertEqual(w0, 0x9)
        self.assertEqual(w1, 0x0)

    def test_ste_make_s1dss_bypass_defaults_to_eats_abt(self):
        """Without an explicit eats=, S1DSS-bypass keeps EATS=ABT (0)."""
        w0, w1 = nested_ste_trigger._ste_make_s1dss_bypass(0x10000)

        self.assertEqual(w0, 0x1002B)
        self.assertEqual(w1, 0x1)

    def test_ste_make_s1dss_bypass_eats_abt_is_a_no_op_bitwise(self):
        """Explicit eats=ABT and the implicit default produce identical qwords."""
        default_w0, default_w1 = nested_ste_trigger._ste_make_s1dss_bypass(0x10000)
        explicit_w0, explicit_w1 = nested_ste_trigger._ste_make_s1dss_bypass(
            0x10000, eats=nested_ste_trigger._STE_EATS_ABT)

        self.assertEqual((default_w0, default_w1), (explicit_w0, explicit_w1))

    def test_ste_make_nested_s2_bypass_eats_trans_sets_eats_field(self):
        """eats=TRANS on the S2-bypass builder sets STRTAB_STE_1_EATS to TRANS."""
        w0, w1 = nested_ste_trigger._ste_make_nested_s2_bypass(
            eats=nested_ste_trigger._STE_EATS_TRANS)

        self.assertEqual(w0, 0x9)
        self.assertEqual(w1, 0x10000000)
        self.assertEqual(
            (w1 >> nested_ste_trigger.STE_1_EATS_SHIFT) & 0x3,
            nested_ste_trigger._STE_EATS_TRANS)

    def test_eats_field_position_matches_kernel_strtab_ste_1_eats(self):
        """STE_1_EATS lands the field at bits [29:28] via GENMASK-style mask."""
        self.assertEqual(nested_ste_trigger.STE_1_EATS, 0x30000000)
        self.assertEqual(nested_ste_trigger.STE_1_EATS_SHIFT, 28)
        self.assertEqual(nested_ste_trigger._STE_EATS_ABT, 0)
        self.assertEqual(nested_ste_trigger._STE_EATS_TRANS, 1)

    def test_field_prep_rejects_overflow(self):
        """_field_prep raises when the value does not fit the mask."""
        with self.assertRaises(ValueError):
            nested_ste_trigger._field_prep(nested_ste_trigger.STE_1_EATS, 4)

    def test_s1dss_and_s2bypass_eats_transition_toggles_eats_and_cfg_together(self):
        """Guest STEs used in the transition loop differ in both CFG and EATS."""
        s1dss_w0, s1dss_w1 = nested_ste_trigger._ste_make_s1dss_bypass(0x10000)
        s2bypass_eats_w0, s2bypass_eats_w1 = nested_ste_trigger._ste_make_nested_s2_bypass(
            eats=nested_ste_trigger._STE_EATS_TRANS)

        self.assertNotEqual(s1dss_w0, s2bypass_eats_w0)
        self.assertNotEqual(s1dss_w1, s2bypass_eats_w1)

    def test_ste_struct_serializes_to_little_endian_bytes(self):
        """IommuHwptArmSmmuv3(ste=(w0, w1)) serializes to little-endian bytes."""
        w0, w1 = nested_ste_trigger._ste_make_s1dss_bypass(0x10000)
        ste = nested_ste_trigger.IommuHwptArmSmmuv3(ste=(w0, w1))

        expected = w0.to_bytes(8, "little") + w1.to_bytes(8, "little")
        self.assertEqual(bytes(ste), expected)

    def test_run_trigger_returns_worker_status_and_sets_thread_comm(self):
        """run_trigger preserves the observer thread name and success result."""
        with self.mock.patch(
            "modules.iommu.nested_ste.trigger._set_thread_name",
        ) as set_name, self.mock.patch(
            "modules.iommu.nested_ste.trigger._run_sequence",
            return_value="TRANSITIONS_OK iters=32",
        ):
            result = nested_ste_trigger.run_trigger(
                "0004:01:00.0", 32, timeout=5)

        self.assertEqual(result, "TRANSITIONS_OK iters=32")
        set_name.assert_called_once_with(nested_ste_trigger.TRIGGER_COMM)

    def test_trigger_worker_is_daemon_with_observer_name(self):
        """A stuck ioctl worker cannot keep the GreenLiT process alive."""
        worker = nested_ste_trigger._TriggerWorker("0004:01:00.0", 1)

        self.assertTrue(worker.daemon)
        self.assertEqual(worker.name, nested_ste_trigger.TRIGGER_COMM)

    def test_run_trigger_raises_trigger_error_on_sequence_failure(self):
        """run_trigger surfaces failures from the worker thread as TriggerError."""
        with self.mock.patch(
            "modules.iommu.nested_ste.trigger._set_thread_name",
        ), self.mock.patch(
            "modules.iommu.nested_ste.trigger._run_sequence",
            side_effect=nested_ste_trigger.TriggerError("no vfio cdev"),
        ):
            with self.assertRaises(nested_ste_trigger.TriggerError) as ctx:
                nested_ste_trigger.run_trigger("0004:01:00.0", 0, timeout=5)

        self.assertIn("no vfio cdev", str(ctx.exception))

    def test_run_trigger_raises_trigger_timeout_on_timeout(self):
        """A worker thread that never finishes is reported as TriggerTimeout."""
        import time  # pylint: disable=import-outside-toplevel

        def _never_returns(*_a, **_kw):
            time.sleep(0.3)
            return "unreachable"

        with self.mock.patch(
            "modules.iommu.nested_ste.trigger._set_thread_name",
        ), self.mock.patch(
            "modules.iommu.nested_ste.trigger._run_sequence",
            side_effect=_never_returns,
        ):
            with self.assertRaises(nested_ste_trigger.TriggerTimeout) as ctx:
                nested_ste_trigger.run_trigger("0004:01:00.0", 0, timeout=0.05)

        self.assertIn("timed out", str(ctx.exception))


if __name__ == "__main__":
    GLTTestBase.unittest.main()
