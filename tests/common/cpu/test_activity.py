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

"""Unit tests for common/cpu/activity.py."""

# pylint: disable=consider-using-with,protected-access

from pathlib import Path
import tempfile

from src.common.cpu import activity as cpu_activity
from tests.test_base import GLTTestBase


class TestCpuActivity(GLTTestBase):
    """Test Linux /proc/stat and /proc/schedstat CPU activity helpers."""

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    @staticmethod
    def _snapshot(entries):
        return {
            cpu: cpu_activity.CpuActivity(*counters)
            for cpu, counters in entries.items()
        }

    def test_activity_delta_rejects_backward_counters(self):
        """Non-monotonic counters fail loudly rather than wrapping."""
        before = self._snapshot({2: (10, 20, 1000, 5)})
        after = self._snapshot({2: (9, 25, 1000, 5)})

        with self.assertRaisesRegex(RuntimeError, "counters moved backwards"):
            cpu_activity.activity_delta(before, after, 2)

    def test_activity_delta_rejects_missing_cpu(self):
        """Missing CPU keys in either snapshot raise RuntimeError."""
        before = self._snapshot({2: (10, 20, 1000, 5)})
        after = self._snapshot({})

        with self.assertRaisesRegex(RuntimeError, "Missing CPU 2"):
            cpu_activity.activity_delta(before, after, 2)

    def test_cpu_stayed_quiet_allows_interrupt_accounting(self):
        """Interrupt ticks alone do not prove that a competing task ran."""
        before = self._snapshot({2: (100, 100, 1000, 10)})
        after = self._snapshot({2: (101, 109, 1000, 10)})

        self.assertTrue(cpu_activity.cpu_stayed_quiet(before, after, 2))

    def test_cpu_stayed_quiet_detects_subtick_scheduler_activity(self):
        """High-resolution counters reject work hidden between tick updates."""
        before = self._snapshot({2: (100, 100, 1000, 10)})
        after = self._snapshot({2: (100, 110, 1500, 11)})

        self.assertFalse(cpu_activity.cpu_stayed_quiet(before, after, 2))

    def test_read_cpu_activity_fails_malformed_schedstat_header(self):
        """A present but malformed known interface is a hard probe error."""
        schedstat_path = self.root / "schedstat"
        schedstat_path.write_text("not-a-version\n", encoding="ascii")

        with self.assertRaisesRegex(
            RuntimeError, "Missing schedstat version"
        ) as context:
            cpu_activity._read_cpu_schedstat({2}, schedstat_path)

        self.assertNotIsInstance(
            context.exception, cpu_activity.ObserverUnavailable
        )

    def test_read_cpu_activity_fails_schedstat_io_error(self):
        """A real observer I/O failure is not classified as unsupported."""
        with self.mock.patch.object(
            Path, "read_text", side_effect=OSError("I/O failure")
        ):
            with self.assertRaisesRegex(
                RuntimeError, "I/O failure"
            ) as context:
                cpu_activity._read_cpu_schedstat({2}, self.root / "schedstat")

        self.assertNotIsInstance(
            context.exception, cpu_activity.ObserverUnavailable
        )

    def test_read_cpu_activity_fails_schedstat_permission_error(self):
        """PermissionError on /proc/schedstat propagates as RuntimeError."""
        with self.mock.patch.object(
            Path, "read_text", side_effect=PermissionError("denied")
        ):
            with self.assertRaisesRegex(
                RuntimeError, "denied"
            ) as context:
                cpu_activity._read_cpu_schedstat({2}, self.root / "schedstat")

        self.assertNotIsInstance(
            context.exception, cpu_activity.ObserverUnavailable
        )

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

        activity = cpu_activity.read_cpu_activity(
            {2, 3}, stat_path, schedstat_path
        )

        self.assertEqual(
            activity[2],
            cpu_activity.CpuActivity(
                busy_ticks=24,
                idle_ticks=10,
                runtime_ns=12345,
                timeslices=9,
            ),
        )
        self.assertEqual(
            activity[3],
            cpu_activity.CpuActivity(
                busy_ticks=0,
                idle_ticks=20,
                runtime_ns=23456,
                timeslices=10,
            ),
        )

    def test_read_cpu_activity_reads_schedstat_before_stat(self):
        """A baseline cannot hide a task starting between the two sources."""
        calls = []
        with self.mock.patch.object(
            cpu_activity,
            "_read_cpu_schedstat",
            side_effect=lambda *_: calls.append("schedstat") or {2: (3, 4)},
        ), self.mock.patch.object(
            cpu_activity,
            "_read_cpu_ticks",
            side_effect=lambda *_: calls.append("stat") or {2: (1, 2)},
        ):
            cpu_activity.read_cpu_activity({2})

        self.assertEqual(calls, ["schedstat", "stat"])

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
            cpu_activity.read_cpu_activity({2, 3}, stat_path, schedstat_path)

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
            cpu_activity.ObserverUnavailable,
            "Unsupported schedstat version 18",
        ):
            cpu_activity._read_cpu_schedstat({2}, schedstat_path)

    def test_read_cpu_activity_skips_missing_schedstat(self):
        """A kernel without scheduler counters cannot support the oracle."""
        with self.assertRaisesRegex(
            cpu_activity.ObserverUnavailable,
            "Cannot read per-CPU scheduler activity",
        ):
            cpu_activity._read_cpu_schedstat(
                {2}, self.root / "missing-schedstat"
            )

    def test_read_cpu_ticks_ignores_malformed_cpu_label(self):
        """A numeric row without the procfs CPU prefix is not accepted."""
        stat_path = self.root / "stat"
        stat_path.write_text(
            "2 1 2 3 10 4 5 6 7\n",
            encoding="ascii",
        )

        with self.assertRaisesRegex(RuntimeError, "Missing CPU activity.*2"):
            cpu_activity._read_cpu_ticks({2}, stat_path)

    def test_read_sched_domain_cpus_returns_only_cpus_with_domains(self):
        """CPUs without domain<N> rows are excluded as null-domain CPUs."""
        schedstat_path = self.root / "schedstat"
        schedstat_path.write_text(
            "version 17\n"
            "timestamp 100\n"
            "cpu0 0 0 0 0 0 0 0 0 0\n"
            "cpu1 0 0 0 0 0 0 0 0 0\n"
            "domain0 SMT 02 0 0 0\n"
            "cpu2 0 0 0 0 0 0 0 0 0\n"
            "domain0 SMT 04 0 0 0\n",
            encoding="ascii",
        )
        self.assertEqual(
            {1, 2}, cpu_activity.read_sched_domain_cpus(schedstat_path)
        )

    def test_read_tick_states_parses_timer_list_sections(self):
        """Both modern .nohz flags and legacy .nohz_mode values are parsed."""
        timer_list = self.root / "timer_list"
        timer_list.write_text(
            "Timer List Version: v0.9\n"
            "cpu: 0\n"
            "  .nohz           : 1\n"
            "  .last_tick      : 123000 nsecs\n"
            "  .tick_stopped   : 1\n"
            "cpu: 2\n"
            "  .nohz_mode      : 2\n"
            "  .last_tick      : 456000 nsecs\n"
            "  .tick_stopped   : 0\n",
            encoding="utf-8",
        )
        states = cpu_activity.read_tick_states({0, 2}, timer_list)
        self.assertEqual(
            cpu_activity.TickState(
                last_tick_ns=123000, nohz_enabled=True, tick_stopped=True
            ),
            states[0],
        )
        self.assertEqual(
            cpu_activity.TickState(
                last_tick_ns=456000, nohz_enabled=True, tick_stopped=False
            ),
            states[2],
        )
