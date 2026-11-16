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

"""Read-only Linux process and task observations."""

import os
import re

from common.cpu_topology import parse_cpu_mask


def process_tree(pid: int | str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return recursive process IDs, thread IDs, and direct edges."""
    tids_by_pid = {}
    children_by_pid = {}
    pending = [str(pid)]
    while pending:
        current = pending.pop(0)
        if current in tids_by_pid:
            continue
        tids_by_pid[current] = {current}
        children_by_pid[current] = set()
        task_root = f"/proc/{current}/task"
        try:
            task_names = sorted(os.listdir(task_root))
        except OSError as error:
            raise OSError(
                f"Could not enumerate live tasks under {task_root}: "
                f"{error}") from error
        if not task_names:
            raise OSError(
                f"Could not enumerate live tasks under {task_root}: "
                "no task entries found.")
        for tid in task_names:
            if tid:
                tids_by_pid[current].add(tid)
            task_dir = os.path.join(task_root, tid)
            children_path = os.path.join(task_dir, "children")
            try:
                with open(children_path, "r", encoding="utf-8") as f:
                    for child in f.read().split():
                        if not child:
                            continue
                        children_by_pid[current].add(child)
                        if child not in tids_by_pid:
                            pending.append(child)
            except OSError as error:
                raise OSError(
                    f"Could not enumerate child processes from "
                    f"{children_path}: {error}") from error
    return tids_by_pid, children_by_pid


def process_comm(pid: int | str) -> str:
    """Return /proc/<pid>/comm for one process."""
    with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as f:
        return f.read().strip()


def task_cpu(task_id: int | str) -> int:
    """Return the last observed CPU for one task from /proc/<tid>/stat."""
    with open(f"/proc/{task_id}/stat", "r", encoding="utf-8") as f:
        line = f.read().strip()
    comm_end = line.rfind(")")
    if comm_end < 0:
        raise ValueError(f"malformed /proc/{task_id}/stat")
    fields = line[comm_end + 2:].split()
    return int(fields[36])


def task_allowed_cpus(task_id: int | str) -> set[int]:
    """Return the Linux CPU affinity mask for one task/process."""
    with open(f"/proc/{task_id}/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Cpus_allowed_list:"):
                return parse_cpu_mask(line.split(":", maxsplit=1)[1])
    raise ValueError(f"missing Cpus_allowed_list for task {task_id}")


def _numa_maps_line_is_anonymous(line: str) -> bool:
    """Return True for workload anonymous mappings relevant to stress-ng --vm."""
    fields = line.split()
    if len(fields) < 2:
        return False
    if any(field in ("heap", "stack") for field in fields[1:]):
        return False
    if any(field.startswith("file=") for field in fields[1:]):
        return False
    return any(
        field.startswith("anon=")
        for field in fields[1:])


def task_anonymous_numa_pages_by_node(task_id: int | str) -> dict[int, int]:
    """Return workload anonymous page counts from /proc/<pid>/numa_maps."""
    pages_by_node = {}
    with open(f"/proc/{task_id}/numa_maps", "r", encoding="utf-8") as f:
        for line in f:
            if not _numa_maps_line_is_anonymous(line):
                continue
            for match in re.finditer(r"\bN(\d+)=(\d+)\b", line):
                node = int(match.group(1))
                pages_by_node[node] = (
                    pages_by_node.get(node, 0) + int(match.group(2)))
    if not pages_by_node:
        raise ValueError(
            f"no workload anonymous NUMA page attribution for task {task_id}")
    return pages_by_node
