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

"""Unit tests for read-only common/glt_proc.py observations."""

import io

from tests.test_base import GLTTestBase
from src.common import glt_proc


def proc_task_fixture(
        child_map, *, task_error_pids=(), empty_task_pids=(),
        children_error_tids=()):
    """Return proc task/children mocks for workload-tree tests."""
    children_by_pid = {
        str(pid): tuple(str(child) for child in children)
        for pid, children in child_map.items()
    }
    task_error_pids = {str(pid) for pid in task_error_pids}
    empty_task_pids = {str(pid) for pid in empty_task_pids}
    children_error_tids = {str(tid) for tid in children_error_tids}

    def task_names(pid):
        if pid in task_error_pids:
            raise PermissionError(f"cannot enumerate tasks for {pid}")
        if pid in empty_task_pids:
            return []
        return [pid]

    def fake_listdir(path):
        prefix = "/proc/"
        suffix = "/task"
        if path.startswith(prefix) and path.endswith(suffix):
            pid = path[len(prefix):-len(suffix)]
            return task_names(pid)
        return []

    def fake_open(path, *_args, **_kwargs):
        parts = path.split("/")
        if (len(parts) == 6 and parts[1] == "proc"
                and parts[3] == "task" and parts[5] == "children"):
            pid = parts[2]
            tid = parts[4]
            if tid in children_error_tids:
                raise PermissionError(
                    f"cannot enumerate children for {tid}")
            return io.StringIO(" ".join(children_by_pid.get(pid, ())))
        raise FileNotFoundError(path)

    return fake_listdir, fake_open


class TestProcObservation(GLTTestBase):
    """Neutral proc text observations stay independent of MPAM verdict policy."""

    def test_l3_max_process_tree_complete_proc_snapshot_succeeds(self):
        """Readable proc task/children state produces the exact live tree."""
        fake_listdir, fake_open = proc_task_fixture({
            "100": ("200", "201"),
            "200": ("300",),
            "201": (),
            "300": (),
        })

        with self.mock.patch.object(
                glt_proc.os, "listdir", side_effect=fake_listdir), \
                self.mock.patch("builtins.open", side_effect=fake_open):
            process_tree = glt_proc.process_tree(100)

        self.assertEqual(set(process_tree[0]), {"100", "200", "201", "300"})
        self.assertEqual(set().union(*process_tree[0].values()), {"100", "200", "201", "300"})
        self.assertEqual(
            tuple(sorted((parent, child) for parent, children in process_tree[1].items()
                         for child in children)),
            (("100", "200"), ("100", "201"), ("200", "300")),
        )


    def test_l3_max_process_tree_blocks_task_directory_enumeration_error(self):
        """Unreadable /proc/<pid>/task is an incomplete observation."""
        fake_listdir, fake_open = proc_task_fixture(
            {"100": ("200",), "200": ()}, task_error_pids=("100",))

        with self.mock.patch.object(
                glt_proc.os, "listdir", side_effect=fake_listdir), \
                self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(
                    OSError) as context:
                glt_proc.process_tree(100)

        self.assertIn("/proc/100/task", str(context.exception))
        self.assertIn("cannot enumerate tasks", str(context.exception))


    def test_l3_max_process_tree_blocks_empty_task_enumeration(self):
        """An empty live task directory cannot prove complete membership."""
        fake_listdir, fake_open = proc_task_fixture(
            {"100": ("200",), "200": ()}, empty_task_pids=("100",))

        with self.mock.patch.object(
                glt_proc.os, "listdir", side_effect=fake_listdir), \
                self.mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(
                    OSError) as context:
                glt_proc.process_tree(100)

        self.assertIn("/proc/100/task", str(context.exception))
        self.assertIn("no task entries", str(context.exception))


    def test_l3_max_anonymous_numa_pages_ignores_generic_and_file_pages(self):
        """Only workload anonymous VM mappings prove stress-ng locality."""
        read_data = (
            "7f000000 default file=/usr/lib/libc.so N2=1 N3=99\n"
            "80000000 default anon=10 dirty=10 N2=10\n"
            "81000000 default heap anon=3 dirty=3 N2=3\n"
            "82000000 default stack anon=2 dirty=2 N2=2\n"
        )

        with self.mock.patch("builtins.open", self.mock.mock_open(
                read_data=read_data)):
            pages = glt_proc.task_anonymous_numa_pages_by_node("200")

        self.assertEqual(pages, {2: 10})


    def test_l3_max_anonymous_numa_pages_blocks_remote_anonymous_pages(self):
        """Remote anonymous pages remain invalid for selected-node proof."""
        read_data = "80000000 default anon=10 dirty=10 N2=9 N3=1\n"

        with self.mock.patch("builtins.open", self.mock.mock_open(
                read_data=read_data)):
            pages = glt_proc.task_anonymous_numa_pages_by_node("200")

        self.assertEqual(pages, {2: 9, 3: 1})

    def test_proc_identity_cpu_and_affinity_from_text(self):
        """Mechanical readers preserve proc comm framing and strict CPU-list parsing."""
        contents = {
            "/proc/42/comm": "stress-ng-vm\n",
            "/proc/42/stat": "42 (worker name) R " + "0 " * 35 + "4\n",
            "/proc/42/status": "Name:\tworker\nCpus_allowed_list:\t4-5,8\n",
        }
        with self.mock.patch("builtins.open", side_effect=lambda path, *_a, **_k:
                             io.StringIO(contents[path])):
            self.assertEqual(glt_proc.process_comm(42), "stress-ng-vm")
            self.assertEqual(glt_proc.task_cpu(42), 4)
            self.assertEqual(glt_proc.task_allowed_cpus(42), {4, 5, 8})
        for reader, text in ((glt_proc.task_cpu, "malformed stat"),
                             (glt_proc.task_allowed_cpus, "Name:\tworker\n")):
            with self.subTest(reader=reader.__name__), self.mock.patch(
                    "builtins.open", self.mock.mock_open(read_data=text)):
                with self.assertRaises(ValueError):
                    reader(42)
