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

"""Unit tests for NOHZ idle load-balancer SMT core selection."""

# pylint: disable=consider-using-with,protected-access,too-many-public-methods

import gzip
from pathlib import Path
import tempfile
from unittest import mock

from common import glt_core as GLTCore
from common.cpu.activity import CpuActivity, ObserverUnavailable, TickState
from src.modules.cpu import glt_test_nohz_idle_core as nohz
from tests.test_base import GLTTestBase


class TestNohzIdleCore(GLTTestBase):
    """Unit tests for glt_test_nohz_idle_core."""

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.cmdline_path = self.root / "cmdline"
        self.cmdline_path.write_text("BOOT_IMAGE=/vmlinuz ro\n", encoding="utf-8")
        self.schedstat_path = self.root / "schedstat"
        self._write_schedstat_domains(range(200))

    @staticmethod
    def _activity(values):
        return {
            cpu: CpuActivity(*counters) for cpu, counters in values.items()
        }

    @staticmethod
    def _make_layout(partial_core=(0, 176), idle_cores=((2, 178),)):
        """Return the two-core SMT layout used by trial tests."""
        return nohz._NohzSmtLayout(
            first_idle_cpu=partial_core[0],
            trigger_cpu=partial_core[1],
            partial_core=partial_core,
            idle_cores=idle_cores,
        )

    @staticmethod
    def _tick_states(cpus, last_tick_ns=1000, nohz_enabled=True, tick_stopped=True):
        return {
            cpu: TickState(
                last_tick_ns=last_tick_ns,
                nohz_enabled=nohz_enabled,
                tick_stopped=tick_stopped,
            )
            for cpu in cpus
        }

    def _write_schedstat_domains(self, domain_cpus, all_cpus=(0, 1, 2, 3, 176, 177, 178, 179)):
        domain_set = set(domain_cpus)
        lines = ["version 17", "timestamp 100"]
        for cpu in sorted(set(all_cpus) | domain_set):
            lines.append(f"cpu{cpu} 0 0 0 0 0 0 0 0 0")
            if cpu in domain_set:
                lines.append("domain0 SMT 01 0 0 0")
        self.schedstat_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    def _write_smt_sysfs(self, online_mask, pairs, smt_active="1"):
        cpu_root = self.root / "cpu"
        (cpu_root / "smt").mkdir(parents=True, exist_ok=True)
        (cpu_root / "smt" / "active").write_text(
            f"{smt_active}\n", encoding="utf-8"
        )
        (cpu_root / "online").write_text(f"{online_mask}\n", encoding="utf-8")
        for lo, hi in pairs:
            for cpu in (lo, hi):
                topo = cpu_root / f"cpu{cpu}" / "topology"
                topo.mkdir(parents=True, exist_ok=True)
                (topo / "thread_siblings_list").write_text(
                    f"{lo},{hi}\n", encoding="utf-8"
                )
        return cpu_root

    def test_check_observer_kconfig_fails_when_both_sources_missing(self):
        """Missing /boot/config and /proc/config.gz fails fast instead of passing."""
        boot_root = self.root / "boot"
        boot_root.mkdir()
        verdict, reason = nohz._check_observer_kconfig(
            boot_root, self.root / "missing-config.gz", "test"
        )
        self.assertEqual(GLTCore.GLTTestResult.FAIL, verdict)
        self.assertIn("neither file exists", reason)

    def test_check_observer_kconfig_falls_back_to_proc_config_gz(self):
        """Compressed /proc/config.gz satisfies the Kconfig check when /boot is absent."""
        boot_root = self.root / "boot"
        boot_root.mkdir()
        gz_path = self.root / "config.gz"
        with gzip.open(gz_path, "wt", encoding="latin-1") as stream:
            stream.write("CONFIG_NO_HZ_COMMON=y\nCONFIG_KALLSYMS=y\n")
        verdict, reason = nohz._check_observer_kconfig(
            boot_root, gz_path, "test"
        )
        self.assertEqual(GLTCore.GLTTestResult.PASS, verdict, reason)

    def test_check_observer_kconfig_skips_when_required_option_disabled(self):
        """Missing CONFIG_NO_HZ_COMMON or CONFIG_KALLSYMS returns SKIP."""
        boot_root = self.root / "boot"
        boot_root.mkdir()
        (boot_root / "config-test").write_text(
            "CONFIG_NO_HZ_COMMON=y\n# CONFIG_KALLSYMS is not set\n",
            encoding="utf-8",
        )
        verdict, reason = nohz._check_observer_kconfig(
            boot_root, self.root / "missing-config.gz", "test"
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIn("CONFIG_KALLSYMS=y", reason)

    def test_classify_kick_targets_discards_contended_partial_core_kick(self):
        """A partial-core kick is discarded if the idle core woke a task."""
        layout = self._make_layout()
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (11, 114, 100, 5),
            176: (25, 5, 500, 2),
            2: (1, 214, 1200, 1),
            178: (0, 215, 0, 0),
        })
        ticks = self._tick_states((0, 2, 178))
        result = nohz._classify_kick_targets(
            layout, ((2, 178),), after, post, ticks, ticks, targets=[0]
        )
        self.assertIsNone(result.conclusive_target)
        self.assertIsNone(result.failed_partial_cpu)
        self.assertIn("witness state was not sustained", result.reason)

    def test_classify_kick_targets_discards_partial_core_kick_when_witness_tick_restarts(
        self,
    ):
        """A partial-core first kick is discarded if the witness left NOHZ idle."""
        layout = self._make_layout()
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (11, 114, 100, 5),
            176: (25, 5, 500, 2),
            2: (0, 215, 0, 0),
            178: (0, 215, 0, 0),
        })
        after_tick = self._tick_states((0, 2, 178), last_tick_ns=1000)
        post_tick = dict(after_tick)
        post_tick[2] = TickState(
            last_tick_ns=2000, nohz_enabled=True, tick_stopped=False
        )
        result = nohz._classify_kick_targets(
            layout,
            ((2, 178),),
            after,
            post,
            after_tick,
            post_tick,
            targets=[0],
        )
        self.assertIsNone(result.conclusive_target)
        self.assertIsNone(result.failed_partial_cpu)
        self.assertIn("NOHZ tick-stopped and task-idle", result.reason)

    def test_classify_kick_targets_discards_pass_even_if_cpu0_kicked_later(
        self,
    ):
        """Task runtime/timeslices on CPU 0 void PASS even when targets=[2, 0]."""
        layout = self._make_layout()
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (11, 114, 900, 6),
            176: (25, 5, 500, 2),
            2: (1, 214, 0, 0),
            178: (0, 215, 0, 0),
        })
        ticks = self._tick_states((0,))
        result = nohz._classify_kick_targets(
            layout, ((2, 178),), after, post, ticks, ticks, targets=[2, 0]
        )
        self.assertIsNone(result.conclusive_target)
        self.assertIsNone(result.failed_partial_cpu)
        self.assertIn("competing activity", result.reason)

    def test_classify_kick_targets_discards_pass_when_first_idle_exits_nohz_tick(
        self,
    ):
        """Restarted NOHZ tick on CPU 0 during the probe voids PASS."""
        layout = self._make_layout()
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (10, 115, 100, 5),
            176: (25, 5, 500, 2),
            2: (1, 214, 0, 0),
            178: (0, 215, 0, 0),
        })
        after_tick = self._tick_states((0,), last_tick_ns=1000, tick_stopped=True)
        post_tick = self._tick_states((0,), last_tick_ns=2000, tick_stopped=False)
        result = nohz._classify_kick_targets(
            layout, ((2, 178),), after, post, after_tick, post_tick, targets=[2]
        )
        self.assertIsNone(result.conclusive_target)
        self.assertIsNone(result.failed_partial_cpu)
        self.assertIn("exited NOHZ tick-stopped idle", result.reason)

    def test_classify_kick_targets_discards_pass_when_sibling_kicked_with_task(
        self,
    ):
        """Task activity on target sibling voids PASS even if kicked later."""
        layout = self._make_layout()
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (10, 115, 100, 5),
            176: (25, 5, 500, 2),
            2: (1, 214, 0, 0),
            178: (1, 214, 800, 1),
        })
        ticks = self._tick_states((0,))
        result = nohz._classify_kick_targets(
            layout, ((2, 178),), after, post, ticks, ticks, targets=[2, 178]
        )
        self.assertIsNone(result.conclusive_target)
        self.assertIsNone(result.failed_partial_cpu)
        self.assertIn("sibling of target CPU 2 woke a task", result.reason)

    def test_classify_kick_targets_flags_partial_core_first_kick(self):
        """A partial-core first kick fails when the idle core stayed task-idle."""
        layout = self._make_layout()
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (11, 114, 100, 5),
            176: (25, 5, 500, 2),
            2: (1, 214, 0, 0),
            178: (0, 215, 0, 0),
        })
        ticks = self._tick_states((0, 2, 178))
        result = nohz._classify_kick_targets(
            layout, ((2, 178),), after, post, ticks, ticks, targets=[0, 2]
        )
        self.assertIsNone(result.conclusive_target)
        self.assertEqual(0, result.failed_partial_cpu)
        self.assertEqual((0, 176), result.failed_partial_core)
        self.assertEqual("", result.reason)

    def test_classify_kick_targets_flags_smt4_partial_core_extra_sibling_kick(
        self,
    ):
        """Selecting an extra idle sibling on an SMT4 partial core fails."""
        layout = self._make_layout(
            partial_core=(0, 176, 352, 528),
            idle_cores=((2, 178, 354, 530),),
        )
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
            354: (0, 210, 0, 0),
            530: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (10, 115, 100, 5),
            176: (25, 5, 500, 2),
            2: (0, 215, 0, 0),
            178: (0, 215, 0, 0),
            354: (0, 215, 0, 0),
            530: (0, 215, 0, 0),
        })
        ticks = self._tick_states((0, 2, 178, 354, 530))
        result = nohz._classify_kick_targets(
            layout,
            ((2, 178, 354, 530),),
            after,
            post,
            ticks,
            ticks,
            targets=[352],
        )
        self.assertIsNone(result.conclusive_target)
        self.assertEqual(352, result.failed_partial_cpu)
        self.assertEqual((0, 176, 352, 528), result.failed_partial_core)
        self.assertEqual("", result.reason)

    def test_classify_kick_targets_rejects_unproven_first_target(self):
        """A first kick outside pre-proven idle cores is rejected."""
        layout = self._make_layout(idle_cores=((2, 178), (3, 179)))
        after = self._activity({176: (20, 5, 500, 2)})
        post = self._activity({176: (25, 5, 500, 2)})
        ticks = self._tick_states((0,))
        result = nohz._classify_kick_targets(
            layout, ((2, 178),), after, post, ticks, ticks, targets=[3]
        )
        self.assertIsNone(result.conclusive_target)
        self.assertIsNone(result.failed_partial_cpu)
        self.assertIn("not on a core proven fully idle", result.reason)

    def test_counter_read_failure_fails_immediately_before_trials(self):
        """Unexpected observer errors fail upfront before running any trial."""
        layout = self._make_layout()
        logger = mock.MagicMock()
        with mock.patch.object(
            nohz,
            "_check_observer_kconfig",
            return_value=(GLTCore.GLTTestResult.PASS, ""),
        ), mock.patch.object(
            nohz,
            "_select_nohz_smt_layout",
            return_value=(GLTCore.GLTTestResult.PASS, layout, ""),
        ), mock.patch.object(
            nohz, "_tracing_root", return_value=Path("/sys/kernel/tracing")
        ), mock.patch.object(
            nohz,
            "read_cpu_activity",
            side_effect=RuntimeError("bad schedstat header"),
        ) as reader, mock.patch.object(
            nohz, "_run_nohz_ilb_trial"
        ) as run_trial:
            with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "observer check failed"
            ):
                nohz.test_nohz_ilb_prefers_fully_idle_smt_core(None, logger)
            self.assertEqual(1, reader.call_count)
            run_trial.assert_not_called()

    def test_evaluate_precondition_accepts_tick_stopped_idle_and_busy_sibling(
        self,
    ):
        """Continuously NOHZ tick-stopped idle CPUs and a busy sibling qualify."""
        layout = self._make_layout()
        before = self._activity({
            0: (10, 100, 100, 5),
            176: (10, 5, 500, 2),
            2: (0, 200, 0, 0),
            178: (0, 200, 0, 0),
        })
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        ticks = self._tick_states((0, 2, 178))
        ok, idle_cores, reason = nohz._evaluate_precondition(
            layout, before, after, ticks, ticks
        )
        self.assertTrue(ok)
        self.assertEqual(((2, 178),), idle_cores)
        self.assertEqual("", reason)

    def test_evaluate_precondition_rejects_idle_but_ticking_partial_thread(
        self,
    ):
        """An idle CPU 0 whose NOHZ tick is not stopped is ineligible."""
        layout = self._make_layout()
        before = self._activity({
            0: (10, 100, 100, 5),
            176: (10, 5, 500, 2),
            2: (0, 200, 0, 0),
            178: (0, 200, 0, 0),
        })
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        before_ticks = self._tick_states((0, 2, 178))
        after_ticks = dict(before_ticks)
        after_ticks[0] = TickState(
            last_tick_ns=2000, nohz_enabled=True, tick_stopped=False
        )
        ok, _idle_cores, reason = nohz._evaluate_precondition(
            layout, before, after, before_ticks, after_ticks
        )
        self.assertFalse(ok)
        self.assertIn("not continuously NOHZ tick-stopped", reason)

    def test_evaluate_precondition_rejects_idle_but_ticking_witness_core(self):
        """An idle witness core whose tick advanced is excluded from pre_idle_cores."""
        layout = self._make_layout()
        before = self._activity({
            0: (10, 100, 100, 5),
            176: (10, 5, 500, 2),
            2: (0, 200, 0, 0),
            178: (0, 200, 0, 0),
        })
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        before_ticks = self._tick_states((0, 2, 178), last_tick_ns=1000)
        after_ticks = dict(before_ticks)
        after_ticks[2] = TickState(
            last_tick_ns=2000, nohz_enabled=True, tick_stopped=True
        )
        ok, idle_cores, reason = nohz._evaluate_precondition(
            layout, before, after, before_ticks, after_ticks
        )
        self.assertFalse(ok)
        self.assertEqual((), idle_cores)
        self.assertIn("no candidate SMT core stayed fully NOHZ tick-stopped", reason)

    def test_evaluate_precondition_rejects_yielding_or_idle_sibling(self):
        """Idle ticks on the busy sibling reject the pre-window."""
        layout = self._make_layout()
        before = self._activity({
            0: (10, 100, 100, 5),
            176: (10, 5, 500, 2),
            2: (0, 200, 0, 0),
            178: (0, 200, 0, 0),
        })
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (19, 6, 900, 4),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        ticks = self._tick_states((0, 2, 178))
        ok, _idle_cores, reason = nohz._evaluate_precondition(
            layout, before, after, ticks, ticks
        )
        self.assertFalse(ok)
        self.assertIn("continuously busy", reason)

    def test_format_tracing_cpumask_formats_low_and_high_cpus(self):
        """Linux bitmap hex chunks split cleanly on 32-bit boundaries."""
        self.assertEqual("2", nohz._format_tracing_cpumask({1}))
        self.assertEqual("1,00000000", nohz._format_tracing_cpumask({32}))
        self.assertEqual(
            "10000,00000000,00000000,00000000,00000000,00000000",
            nohz._format_tracing_cpumask({176}),
        )

    def test_is_nohz_disabled_on_cmdline_honors_last_kstrtobool_token(self):
        """Last nohz= parameter wins and recognizes kstrtobool false strings."""
        self.assertTrue(nohz._is_nohz_disabled_on_cmdline("ro nohz=0"))
        self.assertTrue(nohz._is_nohz_disabled_on_cmdline("nohz=on nohz=false"))
        self.assertFalse(nohz._is_nohz_disabled_on_cmdline("nohz=off nohz=on"))

    def test_observer_unavailable_skips_before_running_trials(self):
        """Missing /proc/schedstat skips cleanly before mutating affinity."""
        layout = self._make_layout()
        logger = mock.MagicMock()
        with mock.patch.object(
            nohz,
            "_check_observer_kconfig",
            return_value=(GLTCore.GLTTestResult.PASS, ""),
        ), mock.patch.object(
            nohz,
            "_select_nohz_smt_layout",
            return_value=(GLTCore.GLTTestResult.PASS, layout, ""),
        ), mock.patch.object(
            nohz, "_tracing_root", return_value=Path("/sys/kernel/tracing")
        ), mock.patch.object(
            nohz,
            "read_cpu_activity",
            side_effect=ObserverUnavailable("missing /proc/schedstat"),
        ), mock.patch.object(nohz, "_run_nohz_ilb_trial") as run_trial:
            with self.assertRaisesRegex(
                GLTCore.GLTTestSkip, "observer is unavailable"
            ):
                nohz.test_nohz_ilb_prefers_fully_idle_smt_core(None, logger)
            run_trial.assert_not_called()

    def test_parse_nohz_kick_targets_filters_by_source_cpu_and_func(self):
        """Only nohz_csd_func events queued from trigger_cpu are returned."""
        trace = (
            "       python3-14223   [001] d.h1.  9927.995125: csd_queue_cpu: "
            "cpu=2 callsite=generic_exec_single+0x5c/0x388 "
            "func=nohz_csd_func csd=0000000039ed53b0\n"
            "       python3-14223   [003] d.h1.  9927.996124: csd_queue_cpu: "
            "cpu=0 callsite=generic_exec_single+0x5c/0x388 "
            "func=nohz_csd_func csd=0000000039ed53b0\n"
            "       python3-14223   [001] d.h1.  9927.997124: csd_queue_cpu: "
            "cpu=0 callsite=ttwu_queue_wakelist+0x170/0x330 "
            "func=sched_ttwu_pending csd=0000000039ed53b0\n"
            "       python3-14222   [001] dlh1.  9927.998124: csd_queue_cpu: "
            "cpu=0 callsite=generic_exec_single+0x5c/0x388 "
            "func=nohz_csd_func csd=0000000039ed53b0\n"
        )
        self.assertEqual([2, 0], nohz._parse_nohz_kick_targets(trace, 1))

    def test_run_trial_judges_only_first_kick_and_configures_sender_tracefs(self):
        """A later fallback ILB kick to CPU 0 without task runtime still passes."""
        layout = self._make_layout()
        tracing_root = self.root / "tracing"
        instance_dir = tracing_root / "instances" / f"glt_nohz_ilb_{nohz.os.getpid()}"
        before = self._activity({
            0: (10, 100, 100, 5),
            176: (10, 5, 500, 2),
            2: (0, 200, 0, 0),
            178: (0, 200, 0, 0),
        })
        after = self._activity({
            0: (10, 110, 100, 5),
            176: (20, 5, 500, 2),
            2: (0, 210, 0, 0),
            178: (0, 210, 0, 0),
        })
        post = self._activity({
            0: (11, 114, 100, 5),
            176: (25, 5, 500, 2),
            2: (1, 214, 0, 0),
            178: (0, 215, 0, 0),
        })
        ticks = self._tick_states((0, 2, 178))
        recorded_cpumask = {}
        poll_sleeps = []

        def fake_mkdir(path_self, exist_ok=False):
            del exist_ok
            nohz.os.makedirs(path_self / nohz.CSD_QUEUE_REL, exist_ok=True)
            nohz.os.makedirs(path_self / "per_cpu" / "cpu176", exist_ok=True)
            (path_self / "tracing_on").write_text("0\n", encoding="utf-8")
            (path_self / "tracing_cpumask").write_text("ff\n", encoding="utf-8")
            (path_self / "buffer_size_kb").write_text("1408\n", encoding="utf-8")
            (path_self / nohz.CSD_QUEUE_REL / "enable").write_text(
                "0\n", encoding="utf-8"
            )
            (path_self / "trace").write_text("", encoding="utf-8")
            (path_self / "per_cpu" / "cpu176" / "stats").write_text(
                "entries: 2\noverrun: 0\ncommit overrun: 0\ndropped events: 0\n",
                encoding="utf-8",
            )

        def fake_sleep(seconds):
            if seconds == nohz.PROBE_POLL_SECONDS:
                poll_sleeps.append(seconds)
                recorded_cpumask["value"] = (
                    instance_dir / "tracing_cpumask"
                ).read_text(encoding="utf-8").strip()
                (instance_dir / "trace").write_text(
                    "  python3-99 [176] d.h1. 1.001: csd_queue_cpu: "
                    "cpu=2 callsite=generic_exec_single+0x1/0x2 "
                    "func=nohz_csd_func csd=0x1\n"
                    "  python3-99 [176] d.h1. 1.005: csd_queue_cpu: "
                    "cpu=0 callsite=generic_exec_single+0x1/0x2 "
                    "func=nohz_csd_func csd=0x1\n",
                    encoding="utf-8",
                )

        spinner_calls = []

        def fake_spawn(cpu, gate_read_fd=None):
            spinner_calls.append((cpu, gate_read_fd is not None))
            proc = mock.MagicMock()
            proc.poll.return_value = None
            proc.stdout = None
            proc.stderr = None
            return proc

        (tracing_root / "instances").mkdir(parents=True)
        with mock.patch.object(
            Path, "mkdir", autospec=True, side_effect=fake_mkdir
        ), mock.patch.object(Path, "rmdir"), mock.patch.object(
            nohz.os,
            "sched_getaffinity",
            return_value={0, 2, 176, 178},
            create=True,
        ), mock.patch.object(
            nohz.os, "sched_setaffinity", create=True
        ) as set_affinity, mock.patch.object(
            nohz, "_spawn_spinner", side_effect=fake_spawn
        ), mock.patch.object(
            nohz, "read_cpu_activity", side_effect=[before, after, post]
        ), mock.patch.object(
            nohz, "read_tick_states", side_effect=[ticks, ticks, ticks]
        ) as read_ticks, mock.patch.object(
            nohz.time, "sleep", side_effect=fake_sleep
        ):
            result = nohz._run_nohz_ilb_trial(layout, tracing_root)

        self.assertEqual(2, result.conclusive_target)
        self.assertIsNone(result.failed_partial_cpu)
        self.assertEqual("", result.reason)
        self.assertEqual(1, len(poll_sleeps))
        self.assertEqual(
            [
                mock.call({0, 2, 178}),
                mock.call({0, 2, 178}),
                mock.call({0, 2, 178}),
            ],
            read_ticks.call_args_list,
        )
        self.assertEqual(
            "10000,00000000,00000000,00000000,00000000,00000000",
            recorded_cpumask["value"],
        )
        self.assertEqual(
            [(176, False), (176, True)],
            spinner_calls,
        )
        self.assertEqual(
            [
                mock.call(0, {176}),
                mock.call(0, {0, 2, 176, 178}),
            ],
            set_affinity.call_args_list,
        )

    def test_run_trial_propagates_affinity_restore_error(self):
        """Failure to restore GreenLiT process CPU affinity raises RuntimeError."""
        layout = self._make_layout()
        tracing_root = self.root / "tracing"
        (tracing_root / "instances").mkdir(parents=True)

        def fake_mkdir(path_self, exist_ok=False):
            del exist_ok
            nohz.os.makedirs(path_self / nohz.CSD_QUEUE_REL, exist_ok=True)
            (path_self / "tracing_on").write_text("0\n", encoding="utf-8")
            (path_self / "tracing_cpumask").write_text("2\n", encoding="utf-8")
            (path_self / "buffer_size_kb").write_text("1408\n", encoding="utf-8")
            (path_self / nohz.CSD_QUEUE_REL / "enable").write_text(
                "0\n", encoding="utf-8"
            )
            (path_self / "trace").write_text("", encoding="utf-8")

        with mock.patch.object(
            Path, "mkdir", autospec=True, side_effect=fake_mkdir
        ), mock.patch.object(Path, "rmdir"), mock.patch.object(
            nohz.os,
            "sched_getaffinity",
            return_value={0, 1, 2, 176, 177, 178},
            create=True,
        ), mock.patch.object(
            nohz.os,
            "sched_setaffinity",
            side_effect=[None, OSError("restore denied")],
            create=True,
        ), mock.patch.object(
            nohz, "_spawn_spinner", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(
                RuntimeError, "restore GreenLiT process affinity"
            ):
                nohz._run_nohz_ilb_trial(layout, tracing_root)

    def test_select_nohz_smt_layout_accepts_two_smt_cores_and_null_nohz_full(
        self,
    ):
        """Two complete SMT cores and '(null)' in nohz_full succeed."""
        cpu_root = self._write_smt_sysfs(
            "0-1,176-177", ((0, 176), (1, 177))
        )
        (cpu_root / "nohz_full").write_text("(null)\n", encoding="utf-8")

        verdict, layout, reason = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={0, 1, 176, 177},
        )
        self.assertEqual(GLTCore.GLTTestResult.PASS, verdict, reason)
        self.assertEqual(0, layout.first_idle_cpu)
        self.assertEqual(176, layout.trigger_cpu)
        self.assertEqual((0, 176), layout.partial_core)
        self.assertEqual(((1, 177),), layout.idle_cores)

    def test_select_nohz_smt_layout_excludes_cpus_without_sched_domain(self):
        """CPUs on null sched_domain (cpuset load_balance=0) are excluded."""
        cpu_root = self._write_smt_sysfs(
            "0-2,176-178", ((0, 176), (1, 177), (2, 178))
        )
        self._write_schedstat_domains({1, 177})
        verdict, layout, _ = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={0, 1, 2, 176, 177, 178},
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIsNone(layout)

    def test_select_nohz_smt_layout_fails_on_invalid_smt_active(self):
        """Invalid smt/active contents return FAIL instead of SKIP."""
        cpu_root = self._write_smt_sysfs(
            "0-2,176-178", ((0, 176), (1, 177), (2, 178)), smt_active="2"
        )
        verdict, layout, reason = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={0, 1, 2, 176, 177, 178},
        )
        self.assertEqual(GLTCore.GLTTestResult.FAIL, verdict)
        self.assertIsNone(layout)
        self.assertIn("Cannot interpret SMT active value", reason)

    def test_select_nohz_smt_layout_fails_on_permission_error(self):
        """PermissionError on smt/active returns FAIL rather than SKIP."""
        cpu_root = self.root / "cpu"
        cpu_root.mkdir()
        with mock.patch.object(
            Path, "read_text", side_effect=PermissionError("denied")
        ):
            verdict, layout, reason = nohz._select_nohz_smt_layout(
                cpu_root, self.cmdline_path, self.schedstat_path
            )
        self.assertEqual(GLTCore.GLTTestResult.FAIL, verdict)
        self.assertIsNone(layout)
        self.assertIn("denied", reason)

    def test_select_nohz_smt_layout_skips_when_cpuset_excludes_lowest_hk_cpu(
        self,
    ):
        """Restricted affinity excluding the lowest housekeeping CPU skips."""
        cpu_root = self._write_smt_sysfs(
            "0-3,176-179", ((0, 176), (1, 177), (2, 178), (3, 179))
        )
        verdict, layout, _ = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={1, 2, 3, 177, 178, 179},
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIsNone(layout)

    def test_select_nohz_smt_layout_skips_when_idle_cores_are_isolated(self):
        """Idle candidate cores in the isolated mask skip the test."""
        cpu_root = self._write_smt_sysfs(
            "0-2,176-178", ((0, 176), (1, 177), (2, 178))
        )
        (cpu_root / "isolated").write_text("1,2,177,178\n", encoding="utf-8")

        verdict, layout, _ = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={0, 1, 2, 176, 177, 178},
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIsNone(layout)

    def test_select_nohz_smt_layout_skips_when_nohz_disabled_on_cmdline(self):
        """Kernel command line nohz=off or nohz=0 skips cleanly."""
        cpu_root = self._write_smt_sysfs(
            "0-2,176-178", ((0, 176), (1, 177), (2, 178))
        )
        self.cmdline_path.write_text(
            "BOOT_IMAGE=/vmlinuz nohz=0 ro\n", encoding="utf-8"
        )
        verdict, layout, reason = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={0, 1, 2, 176, 177, 178},
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIsNone(layout)
        self.assertIn("NOHZ idle tick stopping is disabled", reason)

    def test_select_nohz_smt_layout_skips_when_nohz_full_is_active(self):
        """Any non-empty nohz_full mask skips cleanly."""
        cpu_root = self._write_smt_sysfs(
            "0-2,176-178", ((0, 176), (1, 177), (2, 178))
        )
        (cpu_root / "nohz_full").write_text("2,178\n", encoding="utf-8")

        verdict, layout, reason = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={0, 1, 2, 176, 177, 178},
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIsNone(layout)
        self.assertIn("nohz_full is active", reason)

    def test_select_nohz_smt_layout_skips_when_smt_inactive(self):
        """Inactive SMT returns SKIP."""
        cpu_root = self._write_smt_sysfs(
            "0-2,176-178", ((0, 176), (1, 177), (2, 178)), smt_active="0"
        )
        verdict, layout, _ = nohz._select_nohz_smt_layout(
            cpu_root, self.cmdline_path, self.schedstat_path
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIsNone(layout)

    def test_select_nohz_smt_layout_skips_with_fewer_than_two_cores(self):
        """A single SMT core cannot separate partial and fully idle cores."""
        cpu_root = self.root / "cpu"
        (cpu_root / "smt").mkdir(parents=True, exist_ok=True)
        (cpu_root / "smt" / "active").write_text("1\n", encoding="utf-8")
        (cpu_root / "online").write_text("0-3\n", encoding="utf-8")
        for cpu in range(4):
            topo = cpu_root / f"cpu{cpu}" / "topology"
            topo.mkdir(parents=True, exist_ok=True)
            (topo / "thread_siblings_list").write_text(
                "0-3\n", encoding="utf-8"
            )
        verdict, layout, _ = nohz._select_nohz_smt_layout(
            cpu_root,
            self.cmdline_path,
            self.schedstat_path,
            affinity={0, 1, 2, 3},
        )
        self.assertEqual(GLTCore.GLTTestResult.SKIP, verdict)
        self.assertIsNone(layout)

    def test_test_fails_when_first_kick_selects_partial_core(self):
        """Selecting CPU 0 on the first kick with a sustained witness fails."""
        layout = self._make_layout()
        logger = mock.MagicMock()
        with mock.patch.object(
            nohz,
            "_check_observer_kconfig",
            return_value=(GLTCore.GLTTestResult.PASS, ""),
        ), mock.patch.object(
            nohz,
            "_select_nohz_smt_layout",
            return_value=(GLTCore.GLTTestResult.PASS, layout, ""),
        ), mock.patch.object(
            nohz, "_tracing_root", return_value=Path("/sys/kernel/tracing")
        ), mock.patch.object(
            nohz, "read_cpu_activity", return_value={}
        ), mock.patch.object(
            nohz, "read_tick_states", return_value={}
        ), mock.patch.object(
            nohz,
            "_run_nohz_ilb_trial",
            return_value=nohz._TrialResult(
                failed_partial_cpu=0,
                failed_partial_core=(0, 176),
            ),
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as raised:
                nohz.test_nohz_ilb_prefers_fully_idle_smt_core(None, logger)
            self.assertIn("293f9611ae73", str(raised.exception))
            self.assertIn(
                "does not establish source-commit presence",
                str(raised.exception),
            )

    def test_test_passes_after_three_conclusive_first_kick_trials(self):
        """Contended trials are discarded while three clean first kicks pass."""
        layout = self._make_layout()
        logger = mock.MagicMock()
        clean = nohz._TrialResult(conclusive_target=2)
        contended = nohz._TrialResult(
            reason=(
                "first kick targeted partially busy core CPU 0, but witness "
                "state was not sustained: no pre-window-proven SMT core "
                "remained task-idle across the probe."
            ),
        )
        with mock.patch.object(
            nohz,
            "_check_observer_kconfig",
            return_value=(GLTCore.GLTTestResult.PASS, ""),
        ), mock.patch.object(
            nohz,
            "_select_nohz_smt_layout",
            return_value=(GLTCore.GLTTestResult.PASS, layout, ""),
        ), mock.patch.object(
            nohz, "_tracing_root", return_value=Path("/sys/kernel/tracing")
        ), mock.patch.object(
            nohz, "read_cpu_activity", return_value={}
        ), mock.patch.object(
            nohz, "read_tick_states", return_value={}
        ), mock.patch.object(
            nohz,
            "_run_nohz_ilb_trial",
            side_effect=[contended, clean, clean, clean],
        ) as run_trial:
            nohz.test_nohz_ilb_prefers_fully_idle_smt_core(None, logger)
            self.assertEqual(4, run_trial.call_count)

    def test_test_skips_when_contention_exhausts_budget(self):
        """Pure contention exhaustion skips with the last discard reason."""
        layout = self._make_layout()
        logger = mock.MagicMock()
        with mock.patch.object(
            nohz,
            "_check_observer_kconfig",
            return_value=(GLTCore.GLTTestResult.PASS, ""),
        ), mock.patch.object(
            nohz,
            "_select_nohz_smt_layout",
            return_value=(GLTCore.GLTTestResult.PASS, layout, ""),
        ), mock.patch.object(
            nohz, "_tracing_root", return_value=Path("/sys/kernel/tracing")
        ), mock.patch.object(
            nohz, "read_cpu_activity", return_value={}
        ), mock.patch.object(
            nohz, "read_tick_states", return_value={}
        ), mock.patch.object(
            nohz,
            "_run_nohz_ilb_trial",
            return_value=nohz._TrialResult(
                reason="first kick target CPU 3 was not on a core proven fully idle in the pre-window.",
            ),
        ):
            with self.assertRaisesRegex(
                GLTCore.GLTTestSkip, "not on a core proven fully idle"
            ):
                nohz.test_nohz_ilb_prefers_fully_idle_smt_core(None, logger)

    def test_test_warns_when_observer_gap_exhausts_budget(self):
        """Trace overrun or zero-kick observer gaps warn when exhausting trials."""
        layout = self._make_layout()
        logger = mock.MagicMock()
        with mock.patch.object(
            nohz,
            "_check_observer_kconfig",
            return_value=(GLTCore.GLTTestResult.PASS, ""),
        ), mock.patch.object(
            nohz,
            "_select_nohz_smt_layout",
            return_value=(GLTCore.GLTTestResult.PASS, layout, ""),
        ), mock.patch.object(
            nohz, "_tracing_root", return_value=Path("/sys/kernel/tracing")
        ), mock.patch.object(
            nohz, "read_cpu_activity", return_value={}
        ), mock.patch.object(
            nohz, "read_tick_states", return_value={}
        ), mock.patch.object(
            nohz,
            "_run_nohz_ilb_trial",
            return_value=nohz._TrialResult(
                observer_gap=True,
                reason="captured no nohz_csd_func kicks.",
            ),
        ):
            with self.assertRaisesRegex(
                GLTCore.GLTTestWarn, "captured no nohz_csd_func kicks"
            ):
                nohz.test_nohz_ilb_prefers_fully_idle_smt_core(None, logger)

    def test_trace_lost_events_detects_overruns_and_lost_markers(self):
        """Overrun counters or [LOST markers reject the trace capture."""
        clean_stats = (
            "entries: 4\noverrun: 0\ncommit overrun: 0\ndropped events: 0\n"
        )
        self.assertFalse(nohz._trace_lost_events(clean_stats, "# tracer: nop\n"))
        self.assertTrue(
            nohz._trace_lost_events(
                "entries: 4\noverrun: 2\ncommit overrun: 0\n", ""
            )
        )
        self.assertTrue(
            nohz._trace_lost_events(
                "entries: 4\noverrun: 0\ncommit overrun: 0\ndropped events: 1\n",
                "",
            )
        )
        self.assertTrue(
            nohz._trace_lost_events(clean_stats, "CPU:1 [LOST 5 EVENTS]\n")
        )
