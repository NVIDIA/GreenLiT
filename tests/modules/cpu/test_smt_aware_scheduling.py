"""Unit tests for DGX-17050 SMT-aware scheduler coverage."""
# pylint: disable=consider-using-with,protected-access,too-many-lines,too-many-public-methods

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import glt_core as GLTCore
from src.modules.cpu import glt_test_smt_aware_scheduling as smt_sched
from tests.test_base import GLTTestBase


DEPLOYED_WAKEE_PATH = smt_sched.WAKEE_SCRIPT_PATH
WAKEE_SOURCE_PATH = (
    Path(__file__).parents[3] / "src" / "bin" / "smt_aware_wakee.py"
)


class TestSmtAwareSchedulingHelpers(GLTTestBase):
    """Test topology selection and placement-result helpers."""

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.cpu_root = self.root / "cpu"
        self.node_root = self.root / "node"
        (self.cpu_root / "smt").mkdir(parents=True)
        (self.cpu_root / "smt" / "active").write_text("1\n", encoding="ascii")
        wakee_path_patch = self.mock.patch.object(
            smt_sched, "WAKEE_SCRIPT_PATH", WAKEE_SOURCE_PATH
        )
        wakee_path_patch.start()
        self.addCleanup(wakee_path_patch.stop)

    def _add_core(self, siblings, capacity):
        sibling_text = ",".join(str(cpu) for cpu in siblings)
        for cpu in siblings:
            topology = self.cpu_root / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            (topology / "thread_siblings_list").write_text(
                sibling_text, encoding="ascii"
            )
            (self.cpu_root / f"cpu{cpu}" / "cpu_capacity").write_text(
                f"{capacity}\n", encoding="ascii"
            )

    def _add_node(self, node, cpus):
        node_path = self.node_root / f"node{node}"
        node_path.mkdir(parents=True)
        (node_path / "cpulist").write_text(cpus, encoding="ascii")

    def _layout(self):
        source = smt_sched._SmtCore(capacity=1024, node=0, siblings=(2, 3))
        candidates = (
            smt_sched._SmtCore(capacity=1018, node=0, siblings=(4, 5)),
            smt_sched._SmtCore(capacity=1013, node=0, siblings=(6, 7)),
        )
        return smt_sched._ProbeLayout(candidates=candidates, source=source)

    def _trial_layout(self):
        layout = self._layout()
        return smt_sched._TrialLayout(
            candidates=(layout.candidates[0],),
            source=layout.source,
            hint_seed_core=layout.candidates[1],
            hint_seed_cpu=6,
        )

    def _set_online(self, cpus):
        self.cpu_root.mkdir(parents=True, exist_ok=True)
        (self.cpu_root / "online").write_text(cpus, encoding="ascii")

    @staticmethod
    def _activity(values):
        activity = {}
        for cpu, counters in values.items():
            busy_ticks, idle_ticks = counters[:2]
            runtime_ns = counters[2] if len(counters) > 2 else busy_ticks
            timeslices = counters[3] if len(counters) > 3 else busy_ticks
            activity[cpu] = smt_sched._CpuActivity(
                busy_ticks=busy_ticks,
                idle_ticks=idle_ticks,
                runtime_ns=runtime_ns,
                timeslices=timeslices,
            )
        return activity

    def test_layout_prefers_nonzero_max_capacity_source(self):
        """CPU0 is avoided when another maximum-capacity SMT core exists."""
        self._set_online("0-7")
        self._add_node(0, "0-7")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1024)
        self._add_core((4, 5), 1018)
        self._add_core((6, 7), 1013)

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity=set(range(8))
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.PASS)
        self.assertEqual(reason, "")
        self.assertEqual(layout.source.siblings, (2, 3))
        self.assertEqual(layout.source.capacity, 1024)
        self.assertEqual(layout.candidate_cpus, (4, 6))

    def test_layout_selects_asymmetric_numa_node(self):
        """A node-local capacity range is valid below another node's maximum."""
        self._set_online("0-9")
        self._add_node(0, "0-3")
        self._add_node(1, "4-9")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1024)
        self._add_core((4, 5), 1018)
        self._add_core((6, 7), 1013)
        self._add_core((8, 9), 1012)

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity=set(range(10))
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.PASS)
        self.assertEqual(reason, "")
        self.assertEqual(layout.source.node, 1)
        self.assertEqual(layout.source.siblings, (4, 5))
        self.assertEqual(layout.source.capacity, 1018)
        self.assertEqual(layout.candidate_cpus, (6, 8))

    def test_layout_ignores_empty_numa_nodes(self):
        """Memory-only or disabled NUMA nodes may expose an empty CPU list."""
        self._set_online("0-5")
        self._add_node(0, "0-5")
        self._add_node(1, "")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1018)
        self._add_core((4, 5), 1013)

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity=set(range(6))
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.PASS)
        self.assertEqual(reason, "")
        self.assertEqual(layout.source.siblings, (0, 1))
        self.assertEqual(layout.candidate_cpus, (2, 4))

    def test_layout_skips_when_smt_is_inactive(self):
        """A valid non-SMT state is unsupported rather than a failure."""
        (self.cpu_root / "smt" / "active").write_text("0\n", encoding="ascii")

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity={0, 1}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.SKIP)
        self.assertIsNone(layout)
        self.assertIn("SMT is not active", reason)

    def test_layout_rejects_invalid_smt_active_value(self):
        """An invalid SMT state is an inspection failure, not unsupported."""
        (self.cpu_root / "smt" / "active").write_text("2\n", encoding="ascii")

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity={0, 1}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.FAIL)
        self.assertIsNone(layout)
        self.assertIn("Cannot interpret SMT active value", reason)

    def test_layout_returns_fail_when_affinity_is_unreadable(self):
        """An affinity inspection error is returned as a failure verdict."""
        self._set_online("0-1")

        with self.mock.patch.object(
                smt_sched.os, "sched_getaffinity", side_effect=OSError("denied")):
            verdict, layout, reason = smt_sched._select_probe_layout(
                self.cpu_root, self.node_root
            )

        self.assertEqual(verdict, GLTCore.GLTTestResult.FAIL)
        self.assertIsNone(layout)
        self.assertIn("Cannot read the GreenLiT process CPU affinity", reason)

    def test_layout_skips_uniform_cpu_capacities(self):
        """Uniform capacities cannot exercise the asymmetric scheduler path."""
        self._set_online("0-3")
        self._add_node(0, "0-3")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1024)

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity={0, 1, 2, 3}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.SKIP)
        self.assertIsNone(layout)
        self.assertIn("distinct CPU capacity", reason)

    def test_layout_skips_without_a_separate_hint_seed_core(self):
        """One lower-capacity core cannot both prime and test the idle hint."""
        self._set_online("0-3")
        self._add_node(0, "0-3")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1018)

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity=set(range(4))
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.SKIP)
        self.assertIsNone(layout)
        self.assertIn("at least two lower-capacity", reason)

    def test_layout_skips_incomplete_affinity_cores(self):
        """Both SMT siblings must be usable to control source and candidate cores."""
        self._set_online("0-3")
        self._add_node(0, "0-3")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1018)

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity={0, 2}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.SKIP)
        self.assertIsNone(layout)
        self.assertIn("Fewer than two complete", reason)

    def test_layout_rejects_malformed_sibling_mask(self):
        """A sibling mask that omits its own CPU cannot support a verdict."""
        self._set_online("0-3")
        self._add_node(0, "0-3")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1018)
        (self.cpu_root / "cpu0" / "topology" / "thread_siblings_list").write_text(
            "1\n", encoding="ascii"
        )

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity={0, 1, 2, 3}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.FAIL)
        self.assertIsNone(layout)
        self.assertIn("does not contain CPU 0", reason)

    def test_layout_rejects_nonreciprocal_sibling_masks(self):
        """Every sibling must report the same complete SMT-core mask."""
        self._set_online("0-4")
        self._add_node(0, "0-4")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1018)
        (self.cpu_root / "cpu3" / "topology" / "thread_siblings_list").write_text(
            "3-4\n", encoding="ascii"
        )

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity={0, 1, 2, 3, 4}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.FAIL)
        self.assertIsNone(layout)
        self.assertIn("not reciprocal", reason)

    def test_layout_rejects_different_sibling_capacities(self):
        """The oracle requires equal nominal capacities within an SMT core."""
        self._set_online("0-3")
        self._add_node(0, "0-3")
        self._add_core((0, 1), 1024)
        self._add_core((2, 3), 1018)
        (self.cpu_root / "cpu3" / "cpu_capacity").write_text(
            "1016\n", encoding="ascii"
        )

        verdict, layout, reason = smt_sched._select_probe_layout(
            self.cpu_root, self.node_root, affinity={0, 1, 2, 3}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.FAIL)
        self.assertIsNone(layout)
        self.assertIn("different CPU capacities", reason)

    def test_process_state_handles_spaces_in_command(self):
        """The state parser finds the field after the closing command parenthesis."""
        proc_dir = self.root / "proc" / "123"
        proc_dir.mkdir(parents=True)
        (proc_dir / "stat").write_text(
            "123 (wakee worker) S 1 2 3\n", encoding="ascii"
        )

        self.assertEqual(smt_sched._process_state(123, self.root / "proc"), "S")

    def test_read_cpu_activity_parses_busy_and_idle_ticks(self):
        """The first eight /proc/stat counters produce non-overlapping totals."""
        stat_path = self.root / "stat"
        schedstat_path = self.root / "schedstat"
        stat_path.write_text(
            "cpu 1 2 3 4 5 6 7 8 9 10\n"
            "cpu2 1 2 3 10 4 5 6 7 8 9\n"
            "cpu3 0 0 0 20 0 0 0 0 0 0\n",
            encoding="ascii",
        )
        schedstat_path.write_text(
            "version 17\n"
            "timestamp 123\n"
            "cpu2 0 0 0 0 0 0 12345 0 9\n"
            "cpu3 0 0 0 0 0 0 23456 0 10\n",
            encoding="ascii",
        )

        activity = smt_sched._read_cpu_activity(
            {2, 3}, stat_path, schedstat_path
        )

        self.assertEqual(
            activity[2],
            smt_sched._CpuActivity(
                busy_ticks=24, idle_ticks=10,
                runtime_ns=12345, timeslices=9,
            ),
        )
        self.assertEqual(
            activity[3],
            smt_sched._CpuActivity(
                busy_ticks=0, idle_ticks=20,
                runtime_ns=23456, timeslices=10,
            ),
        )

    def test_read_cpu_activity_rejects_missing_cpu(self):
        """An incomplete accounting snapshot cannot qualify a trial."""
        stat_path = self.root / "stat"
        schedstat_path = self.root / "schedstat"
        stat_path.write_text(
            "cpu2 1 2 3 10 4 5 6 7\n",
            encoding="ascii",
        )
        schedstat_path.write_text(
            "version 17\n"
            "timestamp 123\n"
            "cpu2 0 0 0 0 0 0 12345 0 9\n"
            "cpu3 0 0 0 0 0 0 23456 0 10\n",
            encoding="ascii",
        )

        with self.assertRaisesRegex(RuntimeError, "Missing CPU activity.*3"):
            smt_sched._read_cpu_activity({2, 3}, stat_path, schedstat_path)

    def test_read_cpu_ticks_ignores_malformed_cpu_label(self):
        """A numeric row without the procfs CPU prefix is not accepted."""
        stat_path = self.root / "stat"
        stat_path.write_text(
            "2 1 2 3 10 4 5 6 7\n",
            encoding="ascii",
        )

        with self.assertRaisesRegex(RuntimeError, "Missing CPU activity.*2"):
            smt_sched._read_cpu_ticks({2}, stat_path)

    def test_read_cpu_activity_reads_schedstat_before_stat(self):
        """A baseline cannot hide a task starting between the two sources."""
        calls = []
        with self.mock.patch.object(
                smt_sched, "_read_cpu_schedstat",
                side_effect=lambda *_: calls.append("schedstat") or {2: (3, 4)}
        ), self.mock.patch.object(
                smt_sched, "_read_cpu_ticks",
                side_effect=lambda *_: calls.append("stat") or {2: (1, 2)}
        ):
            smt_sched._read_cpu_activity({2})

        self.assertEqual(calls, ["schedstat", "stat"])

    def test_read_cpu_activity_rejects_unknown_schedstat_version(self):
        """An unknown field layout cannot silently qualify scheduler state."""
        schedstat_path = self.root / "schedstat"
        schedstat_path.write_text(
            "version 18\n"
            "timestamp 123\n"
            "cpu2 0 0 0 0 0 0 12345 0 9\n",
            encoding="ascii",
        )

        with self.assertRaisesRegex(
                smt_sched._ObserverUnavailable,
                "Unsupported schedstat version 18"):
            smt_sched._read_cpu_schedstat({2}, schedstat_path)

    def test_read_cpu_activity_skips_missing_schedstat(self):
        """A kernel without scheduler counters cannot support the oracle."""
        with self.assertRaisesRegex(
                smt_sched._ObserverUnavailable,
                "Cannot read per-CPU scheduler activity"):
            smt_sched._read_cpu_schedstat(
                {2}, self.root / "missing-schedstat"
            )

    def test_read_cpu_activity_fails_malformed_schedstat_header(self):
        """A present but malformed known interface is a hard probe error."""
        schedstat_path = self.root / "schedstat"
        schedstat_path.write_text("not-a-version\n", encoding="ascii")

        with self.assertRaisesRegex(
                RuntimeError, "Missing schedstat version") as context:
            smt_sched._read_cpu_schedstat({2}, schedstat_path)

        self.assertNotIsInstance(
            context.exception, smt_sched._ObserverUnavailable
        )

    def test_read_cpu_activity_fails_schedstat_io_error(self):
        """A real observer I/O failure is not classified as unsupported."""
        with self.mock.patch.object(
                Path, "read_text", side_effect=OSError("I/O failure")):
            with self.assertRaisesRegex(RuntimeError, "I/O failure") as context:
                smt_sched._read_cpu_schedstat({2}, self.root / "schedstat")

        self.assertNotIsInstance(
            context.exception, smt_sched._ObserverUnavailable
        )

    def test_cpu_stayed_quiet_detects_subtick_scheduler_activity(self):
        """High-resolution counters reject work hidden between tick updates."""
        before = self._activity({2: (100, 100, 1000, 10)})
        after = self._activity({2: (100, 110, 1500, 11)})

        self.assertFalse(smt_sched._cpu_stayed_quiet(before, after, 2))

    def test_cpu_stayed_quiet_allows_interrupt_accounting(self):
        """Interrupt ticks alone do not prove that a competing task ran."""
        before = self._activity({2: (100, 100, 1000, 10)})
        after = self._activity({2: (101, 109, 1000, 10)})

        self.assertTrue(smt_sched._cpu_stayed_quiet(before, after, 2))

    def test_qualify_trial_layout_filters_active_candidate(self):
        """An active core is filtered before one quiet core is reserved."""
        layout = self._layout()
        extra_candidate = smt_sched._SmtCore(
            capacity=1010, node=0, siblings=(8, 9)
        )
        layout = smt_sched._ProbeLayout(
            candidates=(*layout.candidates, extra_candidate),
            source=layout.source,
        )
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
            8: (100, 100), 9: (100, 100),
        })
        after = self._activity({
            2: (110, 100), 3: (100, 110),
            4: (100, 110), 5: (100, 110),
            6: (101, 109), 7: (100, 110),
            8: (100, 110), 9: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            trial_layout, activity_before = smt_sched._qualify_trial_layout(
                layout, busy_cpu=2, previous_cpu=3
            )

        self.assertEqual(trial_layout.candidate_cpus, (4,))
        self.assertEqual(trial_layout.hint_seed_core.siblings, (8, 9))
        self.assertEqual(trial_layout.hint_seed_cpu, 8)
        self.assertIs(activity_before, before)

    def test_qualify_trial_layout_rejects_active_previous_cpu(self):
        """Source contention invalidates the old-kernel placement oracle."""
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (110, 100), 3: (101, 109),
            4: (100, 110), 5: (100, 110),
            6: (100, 110), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            with self.assertRaisesRegex(
                    smt_sched._TrialContended, "previous CPU 3"):
                smt_sched._qualify_trial_layout(
                    self._layout(), busy_cpu=2, previous_cpu=3
                )

    def test_qualify_trial_layout_rejects_too_few_quiet_candidates(self):
        """One quiet core cannot both prime the hint and witness placement."""
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (110, 100), 3: (100, 110),
            4: (100, 110), 5: (100, 110),
            6: (101, 109), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            with self.assertRaisesRegex(
                    smt_sched._TrialContended, "Fewer than two candidate"):
                smt_sched._qualify_trial_layout(
                    self._layout(), busy_cpu=2, previous_cpu=3
                )

    def test_qualify_trial_layout_requires_observed_waker_activity(self):
        """Inactive accounting cannot assert that the source sibling was busy."""
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (100, 110), 3: (100, 110),
            4: (100, 110), 5: (100, 110),
            6: (100, 110), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            with self.assertRaisesRegex(RuntimeError, "did not observe the waker"):
                smt_sched._qualify_trial_layout(
                    self._layout(), busy_cpu=2, previous_cpu=3
                )

    def test_qualify_trial_layout_accepts_subtick_waker_activity(self):
        """Scheduler counters can establish waker activity without a tick."""
        before = self._activity({
            2: (100, 100, 1000, 10), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (100, 110, 1500, 11), 3: (100, 110),
            4: (100, 110), 5: (100, 110),
            6: (100, 110), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            trial_layout, _ = smt_sched._qualify_trial_layout(
                self._layout(), busy_cpu=2, previous_cpu=3
            )

        self.assertEqual(trial_layout.candidate_cpus, (4,))
        self.assertEqual(trial_layout.hint_seed_cpu, 6)

    def test_validate_trial_activity_accepts_quiet_destination_sibling(self):
        """Wakee runtime is allowed only on the selected candidate thread."""
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (110, 100), 3: (100, 110),
            4: (101, 109), 5: (100, 110),
            6: (102, 108), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", return_value=after):
            smt_sched._validate_trial_activity(
                self._trial_layout(), previous_cpu=3, destination_cpu=4,
                activity_before=before,
            )

    def test_validate_trial_activity_rejects_destination_sibling_load(self):
        """Candidate-sibling activity invalidates a patched-kernel witness."""
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (110, 100), 3: (100, 110),
            4: (101, 109), 5: (101, 109),
            6: (100, 110), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", return_value=after):
            with self.assertRaisesRegex(
                    smt_sched._TrialContended, "activity on sibling 5"):
                smt_sched._validate_trial_activity(
                    self._trial_layout(), previous_cpu=3, destination_cpu=4,
                    activity_before=before,
                )

    def test_validate_trial_activity_rejects_prev_without_idle_candidate(self):
        """A previous-CPU result needs a fully idle candidate witness."""
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (110, 100), 3: (101, 109),
            4: (101, 109), 5: (100, 110),
            6: (101, 109), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", return_value=after):
            with self.assertRaisesRegex(
                    smt_sched._TrialContended, "No candidate SMT core"):
                smt_sched._validate_trial_activity(
                    self._trial_layout(), previous_cpu=3, destination_cpu=3,
                    activity_before=before,
                )

    def test_validate_trial_activity_rejects_hint_seed_sibling_load(self):
        """Hint priming is invalid when the reserved core was not fully idle."""
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (110, 100), 3: (100, 110),
            4: (101, 109), 5: (100, 110),
            6: (101, 109), 7: (101, 109),
        })
        with self.mock.patch.object(
                smt_sched, "_read_cpu_activity", return_value=after):
            with self.assertRaisesRegex(
                    smt_sched._TrialContended, "seed CPU 6.*sibling 7"):
                smt_sched._validate_trial_activity(
                    self._trial_layout(), previous_cpu=3, destination_cpu=4,
                    activity_before=before,
                )

    def test_readline_timeout_bounds_incomplete_line(self):
        """A partial child message cannot block beyond the requested timeout."""
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, write_fd)
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            os.write(write_fd, b"READY")
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                smt_sched._readline_with_timeout(stream, 0.01, "readiness")

    def test_wait_until_sleeping_rejects_uninterruptible_state(self):
        """Only the interruptible pipe-sleep state proves wakee readiness."""
        with self.mock.patch.object(
                smt_sched, "_process_state", side_effect=["D", "S"]
        ) as process_state, self.mock.patch.object(
                smt_sched.time, "monotonic", side_effect=[0.0, 0.1, 0.2]
        ), self.mock.patch.object(smt_sched.time, "sleep") as sleep:
            smt_sched._wait_until_sleeping(123, timeout=1.0)

        self.assertEqual(process_state.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_placement_result_passes_with_each_direction_proven(self):
        """Every clean trial in both directions must select a candidate."""
        verdict, reason = smt_sched._placement_result(
            self._layout(), {
                2: (4, 4, 6, 4, 6),
                3: (6, 4, 6, 4, 6),
            }
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.PASS)
        self.assertEqual(reason, "")

    def test_placement_result_identifies_missing_target_behavior(self):
        """Repeated wakeups on the partial source core fail targeted behavior."""
        verdict, reason = smt_sched._placement_result(
            self._layout(), {
                2: (2, 2, 2, 2, 2),
                3: (3, 3, 3, 3, 3),
            }
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.FAIL)
        self.assertIn("previous CPU 2", reason)
        self.assertIn("previous CPU 3", reason)
        self.assertIn(smt_sched.FIX_COMMIT, reason)
        self.assertIn("targeted runtime behavior", reason)
        self.assertIn("does not exercise the complete patch series", reason)
        self.assertNotIn("complete patch series are present", reason)
        self.assertIn("does not by itself prove", reason)

    def test_placement_result_rejects_missing_sibling_result(self):
        """An incomplete probe result cannot accidentally satisfy the oracle."""
        verdict, reason = smt_sched._placement_result(
            self._layout(), {2: (4,)}
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.FAIL)
        self.assertIn("incomplete uncontaminated results", reason)
        self.assertIn("previous CPU 2 -> [4]", reason)
        self.assertIn("previous CPU 3 -> []", reason)
        self.assertNotIn(smt_sched.FIX_COMMIT, reason)

    def test_placement_result_warns_mixed_results_without_patch_hint(self):
        """A split clean outcome is inconclusive rather than a patch failure."""
        verdict, reason = smt_sched._placement_result(
            self._layout(), {
                2: (2, 4, 4, 4, 4),
                3: (6, 6, 6, 6, 6),
            }
        )

        self.assertEqual(verdict, GLTCore.GLTTestResult.WARN)
        self.assertIn("mixed uncontaminated placements", reason)
        self.assertNotIn(smt_sched.FIX_COMMIT, reason)

    def test_probe_discards_contention_and_collects_each_direction(self):
        """Contended trials do not count toward the required clean evidence."""
        layout = self._layout()
        trial_4 = smt_sched._WakeupTrial(
            candidate_cpus=(4,), destination_cpu=4, hint_seed_cpu=6
        )
        trial_6 = smt_sched._WakeupTrial(
            candidate_cpus=(6,), destination_cpu=6, hint_seed_cpu=4
        )
        with self.mock.patch.object(
                smt_sched, "PROBE_ATTEMPTS_PER_SIBLING", 2
        ), self.mock.patch.object(
                smt_sched, "PROBE_MAX_TRIALS_PER_SIBLING", 3
        ), self.mock.patch.object(
                smt_sched, "_run_wakeup_trial", side_effect=[
                    smt_sched._TrialContended("noise"),
                    trial_4,
                    trial_6,
                    trial_4,
                    trial_6,
                ]
        ) as trial:
            results = smt_sched._run_wakeup_probe(layout, self.mock_logger)

        self.assertEqual(results, {3: (4, 6), 2: (4, 6)})
        self.assertEqual(trial.call_count, 5)

    def test_probe_is_inconclusive_when_clean_trial_budget_is_exhausted(self):
        """Contention exhaustion is an unsupported measurement environment."""
        with self.mock.patch.object(
                smt_sched, "PROBE_ATTEMPTS_PER_SIBLING", 1
        ), self.mock.patch.object(
                smt_sched, "PROBE_MAX_TRIALS_PER_SIBLING", 2
        ), self.mock.patch.object(
                smt_sched, "_run_wakeup_trial",
                side_effect=smt_sched._TrialContended("noise")
        ):
            with self.assertRaisesRegex(
                    smt_sched._ProbeInconclusive,
                    "only 0/1 uncontaminated"):
                smt_sched._run_wakeup_probe(self._layout(), self.mock_logger)

    def test_prime_idle_core_hint_migrates_away_and_back(self):
        """The waker proves the source went idle before returning to wake."""
        seed_core = self._layout().candidates[1]
        before = self._activity({
            2: (100, 100, 1000, 10), 3: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (100, 110, 1500, 12), 3: (100, 110),
            6: (110, 100), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched.os, "sched_setaffinity"
        ) as set_affinity, self.mock.patch.object(
                smt_sched, "_get_current_cpu", side_effect=[6, 2]
        ), self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            smt_sched._prime_idle_core_hint(
                busy_cpu=2, previous_cpu=3, seed_core=seed_core
            )

        self.assertEqual(
            set_affinity.call_args_list,
            [self.mock.call(0, {6}), self.mock.call(0, {2})],
        )

    def test_prime_idle_core_hint_returns_after_contended_window(self):
        """A rejected priming window cannot leave the waker on its seed CPU."""
        seed_core = self._layout().candidates[1]
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (100, 110), 3: (101, 109),
            6: (110, 100), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched.os, "sched_setaffinity"
        ) as set_affinity, self.mock.patch.object(
                smt_sched, "_get_current_cpu", side_effect=[6, 2]
        ), self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            with self.assertRaisesRegex(
                    smt_sched._TrialContended, "previous CPU 3 was active"):
                smt_sched._prime_idle_core_hint(
                    busy_cpu=2, previous_cpu=3, seed_core=seed_core
                )

        self.assertEqual(
            set_affinity.call_args_list,
            [self.mock.call(0, {6}), self.mock.call(0, {2})],
        )

    def test_prime_idle_core_hint_requires_source_idle_progress(self):
        """Migration without observed source idle cannot prove hint refresh."""
        seed_core = self._layout().candidates[1]
        before = self._activity({
            2: (100, 100), 3: (100, 100),
            6: (100, 100), 7: (100, 100),
        })
        after = self._activity({
            2: (101, 100), 3: (100, 110),
            6: (110, 100), 7: (100, 110),
        })
        with self.mock.patch.object(
                smt_sched.os, "sched_setaffinity"
        ) as set_affinity, self.mock.patch.object(
                smt_sched, "_get_current_cpu", side_effect=[6, 2]
        ), self.mock.patch.object(
                smt_sched, "_read_cpu_activity", side_effect=[before, after]
        ), self.mock.patch.object(smt_sched, "_busy_wait"):
            with self.assertRaisesRegex(
                    smt_sched._TrialContended, "source CPU 2 did not enter"):
                smt_sched._prime_idle_core_hint(
                    busy_cpu=2, previous_cpu=3, seed_core=seed_core
                )

        self.assertEqual(
            set_affinity.call_args_list,
            [self.mock.call(0, {6}), self.mock.call(0, {2})],
        )

    def test_wakeup_trial_applies_and_restores_affinity(self):
        """The trial controls both tasks and always restores the caller mask."""
        layout = self._layout()
        process = self.mock.MagicMock(pid=123)
        process.stdout = self.mock.MagicMock()
        process.stdin = self.mock.MagicMock()
        process.stderr = self.mock.MagicMock()
        original_affinity = {0, 1, 2, 3, 4, 5, 6, 7}
        wakee_affinity = {3, 4, 6}
        seed_core = smt_sched._SmtCore(
            capacity=1010, node=0, siblings=(8, 9)
        )
        trial_layout = smt_sched._TrialLayout(
            candidates=layout.candidates,
            source=layout.source,
            hint_seed_core=seed_core,
            hint_seed_cpu=8,
        )
        activity_before = self._activity({
            2: (100, 100), 3: (100, 100),
            4: (100, 100), 5: (100, 100),
            6: (100, 100), 7: (100, 100),
            8: (100, 100), 9: (100, 100),
        })

        with self.mock.patch.object(
                smt_sched.subprocess, "Popen", return_value=process
        ) as popen, self.mock.patch.object(
                smt_sched, "_readline_with_timeout", side_effect=["READY", "4"]
        ), self.mock.patch.object(
                smt_sched, "_wait_until_sleeping"
        ), self.mock.patch.object(
                smt_sched, "_get_current_cpu", return_value=2
        ), self.mock.patch.object(
                smt_sched, "_qualify_trial_layout",
                return_value=(trial_layout, activity_before)
        ), self.mock.patch.object(
                smt_sched, "_prime_idle_core_hint"
        ) as prime_hint, self.mock.patch.object(
                smt_sched, "_validate_trial_activity"
        ) as validate_activity, self.mock.patch.object(
                smt_sched, "_stop_wakee"
        ) as stop_wakee, self.mock.patch.object(
                smt_sched.os, "sched_getaffinity",
                side_effect=[original_affinity, wakee_affinity]
        ), self.mock.patch.object(
                smt_sched.os, "sched_setaffinity"
        ) as set_affinity:
            observation = smt_sched._run_wakeup_trial(layout, 2, 3)

        self.assertEqual(
            observation,
            smt_sched._WakeupTrial(
                candidate_cpus=(4, 6), destination_cpu=4, hint_seed_cpu=8
            ),
        )
        popen.assert_called_once_with(
            [sys.executable, str(WAKEE_SOURCE_PATH), "3"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(
            set_affinity.call_args_list,
            [
                self.mock.call(0, {2}),
                self.mock.call(123, wakee_affinity),
                self.mock.call(0, original_affinity),
            ],
        )
        process.stdin.write.assert_called_once_with(b"W")
        process.stdin.flush.assert_called_once_with()
        stop_wakee.assert_called_once_with(process)
        prime_hint.assert_called_once_with(2, 3, seed_core)
        validate_activity.assert_called_once_with(
            trial_layout, 3, 4, activity_before
        )

    def test_wakeup_trial_restores_affinity_after_protocol_error(self):
        """A child protocol failure cannot leave GreenLiT CPU-pinned."""
        layout = self._layout()
        process = self.mock.MagicMock(pid=123)
        process.stdout = self.mock.MagicMock()
        process.stdin = self.mock.MagicMock()
        process.stderr = self.mock.MagicMock()
        original_affinity = {0, 1, 2, 3, 4, 5, 6, 7}

        with self.mock.patch.object(
                smt_sched.subprocess, "Popen", return_value=process
        ), self.mock.patch.object(
                smt_sched, "_readline_with_timeout",
                side_effect=RuntimeError("bad marker")
        ), self.mock.patch.object(
                smt_sched, "_stop_wakee"
        ) as stop_wakee, self.mock.patch.object(
                smt_sched.os, "sched_getaffinity", return_value=original_affinity
        ), self.mock.patch.object(
                smt_sched.os, "sched_setaffinity"
        ) as set_affinity:
            with self.assertRaisesRegex(RuntimeError, "bad marker"):
                smt_sched._run_wakeup_trial(layout, 2, 3)

        set_affinity.assert_called_once_with(0, original_affinity)
        stop_wakee.assert_called_once_with(process)

    def test_wakeup_trial_normalizes_initial_affinity_error(self):
        """An unreadable caller affinity becomes a per-test probe failure."""
        with self.mock.patch.object(
                smt_sched.os, "sched_getaffinity", side_effect=OSError("denied")
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot read.*affinity"):
                smt_sched._run_wakeup_trial(self._layout(), 2, 3)

    def test_wakeup_trial_fails_when_installed_wakee_is_missing(self):
        """A broken package fails before launching an uncontrolled child."""
        missing_path = self.root / "missing-wakee"
        with self.mock.patch.object(
                smt_sched, "WAKEE_SCRIPT_PATH", missing_path
        ), self.mock.patch.object(
                smt_sched.os, "sched_getaffinity", return_value={0, 1}
        ), self.mock.patch.object(smt_sched.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(
                    RuntimeError, "Installed SMT-aware wakee script is missing"):
                smt_sched._run_wakeup_trial(self._layout(), 2, 3)

        popen.assert_not_called()

    def test_wakeup_trial_normalizes_cleanup_error(self):
        """A wakee cleanup failure cannot escape as an unclassified exception."""
        process = self.mock.MagicMock(pid=123)
        process.stdout = self.mock.MagicMock()
        process.stdin = self.mock.MagicMock()
        process.stderr = self.mock.MagicMock()
        original_affinity = {0, 1, 2, 3, 4, 5, 6, 7}

        with self.mock.patch.object(
                smt_sched.subprocess, "Popen", return_value=process
        ), self.mock.patch.object(
                smt_sched, "_readline_with_timeout",
                side_effect=RuntimeError("bad marker")
        ), self.mock.patch.object(
                smt_sched, "_stop_wakee", side_effect=OSError("no process")
        ), self.mock.patch.object(
                smt_sched.os, "sched_getaffinity", return_value=original_affinity
        ), self.mock.patch.object(smt_sched.os, "sched_setaffinity"):
            with self.assertRaisesRegex(RuntimeError, "stop wakee process"):
                smt_sched._run_wakeup_trial(self._layout(), 2, 3)

    def test_wakee_source_script_exists(self):
        """The private wakee source is a Python file covered by lint."""
        self.assertEqual(
            DEPLOYED_WAKEE_PATH,
            Path("/usr/share/greenlit/bin/smt_aware_wakee.py"),
        )
        self.assertTrue(WAKEE_SOURCE_PATH.is_file())

    @GLTTestBase.unittest.skipUnless(
        hasattr(os, "sched_setaffinity"), "requires Linux CPU affinity"
    )
    def test_wakee_script_protocol(self):
        """The real wakee preserves its two-line stdout protocol."""
        previous_cpu = min(os.sched_getaffinity(0))
        result = subprocess.run(
            [
                sys.executable,
                str(WAKEE_SOURCE_PATH),
                str(previous_cpu),
            ],
            input=b"W",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=smt_sched.PROBE_TIMEOUT_SECONDS,
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertEqual(result.stderr, b"")
        self.assertEqual(
            result.stdout,
            f"READY\n{previous_cpu}\n".encode("ascii"),
        )


class TestSmtAwareSchedulingRuntime(GLTTestBase):
    """Test GreenLiT result classification for the decorated runtime test."""

    def setUp(self):
        super().setUp()
        source = smt_sched._SmtCore(capacity=1024, node=0, siblings=(2, 3))
        candidate = smt_sched._SmtCore(capacity=1018, node=0, siblings=(4, 5))
        self.layout = smt_sched._ProbeLayout(
            candidates=(candidate,), source=source
        )
        self.nvdev = self.mock.patch.object(smt_sched, "NvDev").start()
        self.read_activity = self.mock.patch.object(
            smt_sched, "_read_cpu_activity", return_value={}
        ).start()
        self.addCleanup(self.mock.patch.stopall)
        self.nvdev.return_value.is_cpu_type.return_value = True

    def test_runtime_skips_non_vera(self):
        """Non-Vera systems skip before inspecting topology."""
        self.nvdev.return_value.is_cpu_type.return_value = False
        with self.mock.patch.object(smt_sched, "_select_probe_layout") as select_layout:
            with self.assertRaisesRegex(GLTCore.GLTTestSkip, "not a Vera CPU"):
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )
        select_layout.assert_not_called()

    def test_runtime_fails_topology_inspection_error(self):
        """Unreadable or malformed topology is a hard inspection failure."""
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.FAIL, None, "bad mask"
                )
        ):
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "bad mask"):
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )

    def test_runtime_skips_unsupported_topology(self):
        """Valid topology that cannot exercise the patch is skipped explicitly."""
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.SKIP, None, "SMT inactive"
                )
        ):
            with self.assertRaisesRegex(GLTCore.GLTTestSkip, "SMT inactive"):
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )

    def test_runtime_fails_probe_execution_error(self):
        """Affinity, child, and timeout failures cannot produce a kernel verdict."""
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.PASS, self.layout, ""
                )
        ), self.mock.patch.object(
                smt_sched, "_run_wakeup_probe", side_effect=RuntimeError("timeout")
        ):
            with self.assertRaisesRegex(GLTCore.GLTTestFail, "timeout"):
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )

    def test_runtime_skips_unavailable_activity_observer(self):
        """An unsupported schedstat interface cannot yield a patch verdict."""
        self.read_activity.side_effect = smt_sched._ObserverUnavailable(
            "missing schedstat"
        )
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.PASS, self.layout, ""
                )
        ), self.mock.patch.object(smt_sched, "_run_wakeup_probe") as probe:
            with self.assertRaisesRegex(
                    GLTCore.GLTTestSkip, "observer is unavailable"):
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )

        probe.assert_not_called()

    def test_runtime_skips_contention_exhaustion(self):
        """A noisy environment is inconclusive rather than a kernel failure."""
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.PASS, self.layout, ""
                )
        ), self.mock.patch.object(
                smt_sched, "_run_wakeup_probe",
                side_effect=smt_sched._ProbeInconclusive("too much activity")
        ):
            with self.assertRaisesRegex(
                    GLTCore.GLTTestSkip, "probe was inconclusive"):
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )

    def test_runtime_fails_repeated_partial_core_placements(self):
        """Missing targeted behavior fails without claiming source absence."""
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.PASS, self.layout, ""
                )
        ), self.mock.patch.object(
                smt_sched, "_run_wakeup_probe", return_value={
                    2: (2, 2, 2, 2, 2),
                    3: (3, 3, 3, 3, 3),
                }
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )

        self.assertIn(smt_sched.FIX_COMMIT, str(context.exception))

    def test_runtime_warns_mixed_clean_placements_without_patch_hint(self):
        """Contradictory controlled behavior warns without source attribution."""
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.PASS, self.layout, ""
                )
        ), self.mock.patch.object(
                smt_sched, "_run_wakeup_probe", return_value={
                    2: (2, 4, 4, 4, 4),
                    3: (4, 4, 4, 4, 4),
                }
        ):
            with self.assertRaisesRegex(
                    GLTCore.GLTTestWarn,
                    "mixed uncontaminated placements") as context:
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )

        self.assertNotIn(smt_sched.FIX_COMMIT, str(context.exception))

    def test_runtime_passes_fully_idle_core_placements(self):
        """Consistent candidate-core witnesses in both directions pass."""
        results = {
            2: (4, 4, 4, 4, 4),
            3: (4, 4, 4, 4, 4),
        }
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.PASS, self.layout, ""
                )
        ), self.mock.patch.object(
                smt_sched, "_run_wakeup_probe", return_value=results
        ):
            smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                None, self.mock_logger
            )

        self.mock_logger.info.assert_any_call(
            "PASS: Vera wakeups selected fully idle SMT cores in both sibling "
            "directions: %s",
            results,
        )

    def test_runtime_fails_unexpected_topology_verdict(self):
        """A successful topology verdict must include a usable layout."""
        with self.mock.patch.object(
                smt_sched, "_select_probe_layout", return_value=(
                    GLTCore.GLTTestResult.PASS, None, ""
                )
        ):
            with self.assertRaisesRegex(
                    GLTCore.GLTTestFail, "Unexpected topology-selection verdict"):
                smt_sched.test_vera_smt_aware_asym_capacity_wakeup(
                    None, self.mock_logger
                )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
