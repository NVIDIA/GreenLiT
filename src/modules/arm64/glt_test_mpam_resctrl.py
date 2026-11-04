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

"""ARM64 MPAM and resctrl validation tests."""
# pylint: disable=too-many-lines

from contextlib import contextmanager
import errno
import fnmatch
import glob
import mmap
import os
import platform
import re
import signal
import stat
import struct
import subprocess
from statistics import median as _median
import tempfile
import threading
import time
import uuid

from common.cpu_topology import parse_cpu_mask
from common import glt_core as GLTCore
from common import glt_proc
from common import glt_utils
from common import resctrl_utils
from common.nv_device import NvDev
from scripts import check_kernel_config


PROC_IOMEM = "/proc/iomem"
MPAM_DEBUGFS_PATH = "/sys/kernel/debug/mpam"
MPAMF_IDR_HAS_MBW_PART = 1 << 26
MPAM_PERCENT_MAX = 100
MPAM_PERCENTAGE_NO_PAIR_REASON = (
    "{resource} reported grid {grid!r} has no two distinct legal values that "
    "both differ from the restored initial schemata on every domain."
)
MPAM_L3_MAX_LIVE_PATTERN = ("low", "high", "low", "high", "low", "high")
MPAM_L3_MAX_LIVE_MIN_SPAN = 10
MPAM_L3_MAX_LIVE_BASELINE_LIMIT = 1024 * 1024
MPAM_L3_MAX_LIVE_BASELINE_TIMEOUT_SEC = 5
MPAM_L3_MAX_LIVE_BASELINE_POLL_SEC = 0.1
MPAM_L3_MAX_LIVE_MIN_SEPARATION_BYTES = 1024 * 1024
MPAM_L3_MAX_LIVE_WARMUP_SEC = 5
MPAM_L3_MAX_LIVE_SAMPLE_COUNT = 10
MPAM_L3_MAX_LIVE_SAMPLE_INTERVAL_SEC = 1
MPAM_L3_MAX_LIVE_OUTER_TIMEOUT_SEC = 45
MPAM_L3_MAX_LIVE_STRESS_TIMEOUT_SEC = 30
MPAM_L3_MAX_LIVE_MEMBERSHIP_TIMEOUT_SEC = 5
MPAM_L3_MAX_LIVE_QUIESCE_SEC = 2
MPAM_L3_MAX_LIVE_KILL_CLEANUP_TIMEOUT_SEC = 5
MPAM_L3_MAX_LIVE_KILL_CLEANUP_POLL_SEC = 0.1
MPAM_L3_MAX_LIVE_STRESS_WORKERS = 4
MPAM_L3_MAX_LIVE_VM_BYTES = "1G"
MPAM_L3_MAX_LIVE_VM_FOOTPRINT_BYTES = 1024 * 1024 * 1024
MPAM_L3_MAX_LIVE_MIN_RESIDENT_BYTES = (MPAM_L3_MAX_LIVE_VM_FOOTPRINT_BYTES * 90 + 99) // 100
MPAM_L3_MAX_LIVE_VM_WORKER_COMM = "stress-ng-vm"
MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES = 16 * 1024
MPAM_PLATFORM_CAPABILITIES = {
    "grace": {
        "expected_dirs": ("L3", "L3_MON", "MB"),
        "optional_dirs": ("MB_HLIM",),
        "expected_files": ("last_cmd_status",),
        "expected_mb_mon_features": (),
    },
    "vera": {
        # MB_HLIM is checked against max_lim by DGX-17719; L2_MAX is optional.
        "expected_dirs": ("L3", "L3_MAX", "L3_MON", "MB", "MB_MON"),
        "optional_dirs": (),
        "expected_files": ("last_cmd_status",),
        "expected_mb_mon_features": (resctrl_utils.MBM_TOTAL_BYTES,),
    },
}


class _MpamBlocked(RuntimeError):
    """Helper blockage for decorated callers to convert to GLT warn."""

    early_exit = None


def _read_mb_hlim_schemata(path):
    """Read MB and MB_HLIM domain values, rejecting malformed schemata."""
    schemas = resctrl_utils.read_schemata(path, ("MB", "MB_HLIM"))
    if set(schemas) != {"MB", "MB_HLIM"}:
        raise ValueError(f"Missing MB or MB_HLIM resource in {path}")
    if schemas["MB"].keys() != schemas["MB_HLIM"].keys():
        raise ValueError(f"MB and MB_HLIM domains do not match in {path}")
    return schemas


def _mpam_combined_failure(failure_reason, cleanup_errors):
    """Return one failure message preserving body and cleanup failures."""
    if failure_reason and cleanup_errors:
        return failure_reason + "; cleanup: " + "; ".join(cleanup_errors)
    if cleanup_errors:
        return "; ".join(cleanup_errors)
    return failure_reason


def _mpam_read_percentage_info(resource):
    """Read percentage schema metadata for an MPAM MAX resource."""
    info_dir = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info")
    try:
        info_stat = os.stat(info_dir)
    except OSError as error:
        raise ValueError(
            f"Could not stat resctrl info directory {info_dir}: {error}"
        ) from error
    if not stat.S_ISDIR(info_stat.st_mode):
        raise ValueError(
            f"resctrl info path {info_dir} exists but is not a directory."
        )

    resource_dir = os.path.join(info_dir, resource)
    try:
        resource_stat = os.stat(resource_dir)
    except OSError as error:
        if error.errno == errno.ENOENT:
            return None
        raise ValueError(
            f"Could not stat {resource} info directory {resource_dir}: {error}"
        ) from error
    if not stat.S_ISDIR(resource_stat.st_mode):
        raise ValueError(
            f"{resource} info path {resource_dir} exists but is not a directory."
        )

    try:
        schema_format = glt_utils.read_sysfs_value(
            os.path.join(resource_dir, "schema_format"))
        percent_min = int(glt_utils.read_sysfs_value(
            os.path.join(resource_dir, "percent_min")))
        percent_gran = int(glt_utils.read_sysfs_value(
            os.path.join(resource_dir, "percent_gran")))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not read {resource} percentage info under {resource_dir}: "
            f"{error}"
        ) from error

    if schema_format != "percentage":
        raise ValueError(
            f"{resource} is not in percentage mode "
            f"(schema_format={schema_format})."
        )

    return percent_min, percent_gran


def _mpam_percentage_policy_reason(resource, percent_min, percent_gran):
    """Return the unsupported userspace percentage policy reason, if any."""
    if not 0 <= percent_min < MPAM_PERCENT_MAX:
        return (
            f"{resource} percent_min {percent_min} is outside the "
            f"userspace test-policy range 0 <= percent_min < "
            f"{MPAM_PERCENT_MAX}."
        )
    if not 1 <= percent_gran <= MPAM_PERCENT_MAX:
        return (
            f"{resource} percent_gran {percent_gran} is outside the "
            f"userspace test-policy range 1 <= percent_gran <= "
            f"{MPAM_PERCENT_MAX}."
        )
    return None


def _mpam_write_resource_schemata(schemata_path, resource, domains,
                                  value_by_domain):
    """Write one resource's schemata line."""
    write_line = f"{resource}:" + ";".join(
        f"{domain}={value_by_domain[domain]}" for domain in domains)
    try:
        glt_utils.write_sysfs_value(schemata_path, write_line)
    except OSError as error:
        raise ValueError(
            f"Could not write {resource} schemata {write_line!r} to "
            f"{schemata_path}: {error}"
        ) from error
    return write_line


def _mpam_write_and_require_percentage(schemata_path, resource, domains, value):
    """Write one percentage value and fail if exact readback is not proven."""
    expected = {domain: value for domain in domains}
    write_line = _mpam_write_resource_schemata(
        schemata_path, resource, domains, expected)
    schemas = resctrl_utils.read_schemata(schemata_path, (resource,))
    if resource not in schemas:
        raise ValueError(f"{resource} resource line is missing from {schemata_path}.")
    read_values = schemas[resource]
    readback = f"{resource}:" + ";".join(
        f"{domain}={value}" for domain, value in read_values.items())
    if read_values != expected:
        raise ValueError(
            f"{resource} wrote {write_line!r}, read back "
            f"{readback!r}; exact per-domain readback is required."
        )
    return f"{resource} wrote {write_line!r}, read back {readback!r}."


def _mpam_restore_percentage_schemata(schemata_path, resource, domains,
                                      initial_values):
    """Restore and exact-read the captured initial schemata values."""
    write_line = _mpam_write_resource_schemata(
        schemata_path, resource, domains, initial_values)
    schemas = resctrl_utils.read_schemata(schemata_path, (resource,))
    if resource not in schemas:
        raise ValueError(f"{resource} resource line is missing from {schemata_path}.")
    read_values = schemas[resource]
    readback = f"{resource}:" + ";".join(
        f"{domain}={value}" for domain, value in read_values.items())
    if read_values != initial_values:
        raise ValueError(
            f"Restored {resource} schemata mismatch: wrote {write_line!r}, "
            f"read back {readback!r}."
        )


def _mpam_value_changes_all_domains(value, initial_values):
    """Return True when a uniform write differs from every initial domain."""
    return all(initial != value for initial in initial_values.values())


def _mpam_select_distinct_percentage_pair(schemata_path, resource, domains,
                                          initial_values, grid):
    """Select deterministic extremal low/high values with exact readback."""
    selected_low = None
    for candidate in grid:
        if not _mpam_value_changes_all_domains(candidate, initial_values):
            continue
        _mpam_write_and_require_percentage(
            schemata_path, resource, domains, candidate)
        selected_low = candidate
        break

    selected_high = None
    for candidate in reversed(grid):
        if candidate == selected_low:
            continue
        if not _mpam_value_changes_all_domains(candidate, initial_values):
            continue
        _mpam_write_and_require_percentage(
            schemata_path, resource, domains, candidate)
        selected_high = candidate
        break

    if selected_low is None or selected_high is None:
        return None
    return selected_low, selected_high


def _mpam_probe_percentage_pair(schemata_path, resource, domains,
                                initial_values, grid):
    """Probe the upper bound, restore, then select a distinct low/high pair."""
    _mpam_write_and_require_percentage(
        schemata_path, resource, domains, grid[-1])
    _mpam_restore_percentage_schemata(
        schemata_path, resource, domains, initial_values)
    return _mpam_select_distinct_percentage_pair(
        schemata_path, resource, domains, initial_values, grid)


def _cpus_of_node(node, online_only=False) -> list:
    """Return CPU ids owned by a NUMA node.

    Default callers keep the original permissive cpulist parser.  Live
    attribution opts into strict parsing and filters by the kernel online set.
    """
    cpus = []
    cpulist_path = f"/sys/devices/system/node/node{node}/cpulist"
    try:
        with open(cpulist_path, "r", encoding="utf-8") as f:
            spec = f.read().strip()
    except OSError as error:
        if not online_only:
            raise
        raise OSError(
            f"Could not read NUMA node {node} CPU list at {cpulist_path}: "
            f"{error}") from error
    if not online_only:
        for part in spec.split(","):
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-")
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(part))
        return cpus

    try:
        node_cpus = parse_cpu_mask(spec)
    except ValueError as error:
        raise ValueError(
            f"Malformed NUMA node {node} CPU list at {cpulist_path}: "
            f"{spec!r}") from error

    online_path = "/sys/devices/system/cpu/online"
    try:
        with open(online_path, "r", encoding="utf-8") as f:
            online_spec = f.read().strip()
    except OSError as error:
        raise OSError(
            f"Could not read online CPU list at {online_path}: {error}"
        ) from error
    try:
        online_cpus = parse_cpu_mask(online_spec)
    except ValueError as error:
        raise ValueError(
            f"Malformed online CPU list at {online_path}: {online_spec!r}"
        ) from error
    if not online_cpus:
        raise ValueError(f"Online CPU list at {online_path} is empty.")
    return sorted(node_cpus & online_cpus)


def _mpam_debugfs_subdirs(path, pattern) -> list:
    """Return the sorted subdirectories of path whose names match pattern.

    Unlike glob.glob(), directory enumeration errors propagate as OSError,
    so a caller can distinguish a failed inspection of advertised MPAM
    debugfs topology (any other OSError) from an absent optional directory
    (FileNotFoundError).
    """
    with os.scandir(path) as entries:
        return sorted(
            entry.path for entry in entries
            if fnmatch.fnmatchcase(entry.name, pattern) and entry.is_dir()
        )


def _mpam_class_mbw_part_components(class_level):
    """Return the component IDs of a class whose RIS advertise MBW partitioning.

    Reads MPAMF_IDR for every RIS the driver discovered under
    class.<class_level> in debugfs and tests the architectural HAS_MBW_PART
    bit. This is a proxy for the driver's MB gate, MPAMF_MBW_IDR.HAS_MAX
    with BWA_WD > 0, which debugfs does not expose. Memory components are
    keyed by NUMA node ID and cache components by cache ID. Returns None
    when MPAM debugfs or the class is absent. Enumeration and read errors
    propagate as OSError; a malformed entry raises ValueError.
    """
    class_path = os.path.join(MPAM_DEBUGFS_PATH, f"class.{class_level}")
    try:
        component_paths = _mpam_debugfs_subdirs(class_path, "comp.*")
    except FileNotFoundError:
        return None

    components = set()
    for component_path in component_paths:
        component = os.path.basename(component_path).removeprefix("comp.")
        if not component.isdigit():
            raise ValueError(
                f"Invalid MPAM component path: {component_path}")
        for vmsc_path in _mpam_debugfs_subdirs(component_path, "vmsc.*"):
            for ris_path in _mpam_debugfs_subdirs(vmsc_path, "msc.*_ris.*"):
                idr_path = os.path.join(ris_path, "mpamf_idr")
                idr = int(glt_utils.read_sysfs_value(idr_path), 16)
                if idr & MPAMF_IDR_HAS_MBW_PART:
                    components.add(int(component))
    return components


def _mpam_l3_max_discovery_names(path):
    """List one attribution directory without suppressing I/O failures."""
    try:
        with os.scandir(path) as entries:
            return tuple(entry.name for entry in entries)
    except OSError as error:
        raise _MpamBlocked(
            f"Could not enumerate L3_MAX attribution directory {path}: "
            f"{error}") from error


def _mpam_l3_cache_ids_for_cpu(cpu):
    """Return L3 cache IDs visible for one CPU."""
    cache_ids = set()
    cache_path = f"/sys/devices/system/cpu/cpu{cpu}/cache"
    for name in _mpam_l3_max_discovery_names(cache_path):
        if not name.startswith("index"):
            continue
        index = os.path.join(cache_path, name)
        try:
            with open(os.path.join(index, "level"), encoding="utf-8") as f:
                level_text = f.read().strip()
        except OSError as error:
            raise _MpamBlocked(
                f"Could not read cache level for CPU {cpu} at {index}: "
                f"{error}") from error
        try:
            level = int(level_text)
        except ValueError as error:
            raise _MpamBlocked(
                f"Malformed cache level for CPU {cpu} at {index}: "
                f"{level_text!r}") from error
        if level != 3:
            continue
        try:
            with open(os.path.join(index, "id"), encoding="utf-8") as f:
                cache_id_text = f.read().strip()
        except OSError as error:
            raise _MpamBlocked(
                f"Could not read L3 cache ID for CPU {cpu} at {index}: "
                f"{error}") from error
        try:
            cache_ids.add(int(cache_id_text))
        except ValueError as error:
            raise _MpamBlocked(
                f"Malformed L3 cache ID for CPU {cpu} at {index}: "
                f"{cache_id_text!r}") from error
    return cache_ids


def _mpam_l3_max_node_cpulists():
    """Return NUMA node cpulist paths sorted by numeric node ID."""
    node_cpulists = []
    node_path = "/sys/devices/system/node"
    for name in _mpam_l3_max_discovery_names(node_path):
        match = re.match(r"node(\d+)", name)
        if match is None:
            continue
        cpulist = os.path.join(node_path, name, "cpulist")
        try:
            os.lstat(cpulist)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise _MpamBlocked(
                f"Could not inspect NUMA cpulist {cpulist}: {error}") from error
        node_cpulists.append((int(match.group(1)), cpulist))
    return tuple(sorted(node_cpulists))


def _mpam_l3_max_monitor_files(group):
    """Return advertised L3 monitor files from this private group."""
    monitor_files = {}
    monitor_path = os.path.join(group, "mon_data")
    for name in sorted(_mpam_l3_max_discovery_names(monitor_path)):
        mon_match = re.match(r"mon_L3_(\d+)", name)
        if mon_match is None:
            continue
        path = os.path.join(monitor_path, name, "llc_occupancy")
        try:
            os.lstat(path)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise _MpamBlocked(
                f"Could not inspect L3 monitor {path}: {error}") from error
        monitor_files[int(mon_match.group(1))] = path

    return monitor_files


def _mpam_l3_max_occupancy_files(group, placement):
    """Resolve every selected cache monitor in the new window's own group."""
    monitor_files = _mpam_l3_max_monitor_files(group)
    selected_monitor_ids = set(monitor_files) & set(placement['cache_ids'])
    if selected_monitor_ids != set(placement['cache_ids']):
        raise _MpamBlocked(
            "L3 monitor IDs do not exactly match selected NUMA node "
            f"{placement['node']}: expected cache IDs {list(placement['cache_ids'])}, "
            f"found selected monitor IDs {sorted(selected_monitor_ids)}.")
    return tuple(monitor_files[cache_id] for cache_id in placement['cache_ids'])


def _mpam_l3_max_attribution(group):
    """Select NUMA node, CPUs and cache IDs using calibration monitors."""
    monitor_files = _mpam_l3_max_monitor_files(group)
    for node, _cpulist in _mpam_l3_max_node_cpulists():
        try:
            cpus = tuple(_cpus_of_node(node, online_only=True))
        except (OSError, ValueError) as error:
            raise _MpamBlocked(str(error)) from error
        if not cpus:
            continue
        cache_ids = set()
        for cpu in cpus:
            cpu_cache_ids = _mpam_l3_cache_ids_for_cpu(cpu)
            if not cpu_cache_ids:
                raise _MpamBlocked(
                    f"Selected NUMA node {node} online CPU {cpu} has no "
                    "L3 cache IDs for live attribution.")
            cache_ids.update(cpu_cache_ids)
        selected_monitor_ids = set(monitor_files) & cache_ids
        if selected_monitor_ids != cache_ids:
            raise _MpamBlocked(
                "L3 monitor IDs do not exactly match selected NUMA node "
                f"{node}: expected cache IDs {sorted(cache_ids)}, found "
                f"selected monitor IDs {sorted(selected_monitor_ids)}."
            )
        return {"node": node, "cpus": cpus, "cache_ids": tuple(sorted(cache_ids))}
    raise _MpamBlocked(
        "Could not build complete NUMA/CPU/L3-cache attribution for "
        f"{group}; no matching llc_occupancy files found.")


def _mpam_read_group_task_ids(group):
    """Return task IDs currently listed in one resctrl group tasks file."""
    with open(os.path.join(group, "tasks"), "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _mpam_resctrl_control_groups():
    """Return resctrl control groups that expose a tasks file."""
    groups = [resctrl_utils.RESCTRL_MOUNT_POINT]
    child_groups = []
    try:
        with os.scandir(resctrl_utils.RESCTRL_MOUNT_POINT) as entries:
            for entry in entries:
                group = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, entry.name)
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise _MpamBlocked(
                        "Could not inspect resctrl control group entry "
                        f"{group}: {error}") from error
                if not stat.S_ISDIR(entry_stat.st_mode):
                    continue
                tasks_path = os.path.join(group, "tasks")
                try:
                    os.stat(tasks_path)
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise _MpamBlocked(
                        "Could not inspect resctrl control group tasks file "
                        f"{tasks_path}: {error}") from error
                child_groups.append(group)
    except OSError as error:
        raise _MpamBlocked(
            "Could not enumerate resctrl control groups under "
            f"{resctrl_utils.RESCTRL_MOUNT_POINT}: {error}") from error
    groups.extend(sorted(child_groups))
    return tuple(dict.fromkeys(groups))


def _mpam_resctrl_monitor_groups(control_group):
    """Return immediate monitor groups under one resctrl control group."""
    monitor_root = os.path.join(control_group, "mon_groups")
    try:
        monitor_root_stat = os.stat(monitor_root)
    except FileNotFoundError:
        return ()
    except OSError as error:
        raise _MpamBlocked(
            "Could not inspect resctrl monitor group directory "
            f"{monitor_root}: {error}") from error
    if not stat.S_ISDIR(monitor_root_stat.st_mode):
        raise _MpamBlocked(
            f"resctrl monitor group path {monitor_root} exists but is not "
            "a directory.")

    monitor_groups = []
    try:
        with os.scandir(monitor_root) as entries:
            for entry in entries:
                monitor_group = os.path.join(monitor_root, entry.name)
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise _MpamBlocked(
                        "Could not inspect resctrl monitor group entry "
                        f"{monitor_group}: {error}") from error
                if stat.S_ISDIR(entry_stat.st_mode):
                    monitor_groups.append(monitor_group)
    except OSError as error:
        raise _MpamBlocked(
            "Could not enumerate resctrl monitor groups under "
            f"{monitor_root}: {error}") from error
    return tuple(sorted(monitor_groups))


def _mpam_resctrl_monitor_group_for_task(control_group, task_text):
    """Find the optional immediate monitor group for one control-group task."""
    monitor_groups = _mpam_resctrl_monitor_groups(control_group)
    monitor_matches = []
    monitor_read_errors = []
    for monitor_group in monitor_groups:
        try:
            task_ids = _mpam_read_group_task_ids(monitor_group)
        except OSError as error:
            monitor_read_errors.append(f"{monitor_group}/tasks: {error}")
            continue
        if task_text in task_ids:
            monitor_matches.append(monitor_group)
    if monitor_read_errors:
        raise _MpamBlocked(
            "Could not prove supervisor resctrl monitor group before "
            "mutation; unreadable tasks files: "
            f"{', '.join(monitor_read_errors)}.")
    if len(monitor_matches) > 1:
        raise _MpamBlocked(
            f"Supervisor task {task_text} appears in "
            f"{len(monitor_matches)} resctrl monitor groups "
            f"({monitor_matches}); expected at most one before mutating "
            "task membership.")
    return monitor_matches[0] if monitor_matches else None


def _mpam_resctrl_group_for_task(task_id):
    """Find the unique resctrl CTRL and optional MON group for one task."""
    task_text = str(task_id)
    matches = []
    read_errors = []
    for group in _mpam_resctrl_control_groups():
        try:
            task_ids = _mpam_read_group_task_ids(group)
        except OSError as error:
            read_errors.append(f"{group}/tasks: {error}")
            continue
        if task_text in task_ids:
            matches.append(group)
    if read_errors:
        raise _MpamBlocked(
            "Could not prove supervisor resctrl group before mutation; "
            f"unreadable tasks files: {', '.join(read_errors)}.")
    if len(matches) != 1:
        raise _MpamBlocked(
            f"Supervisor task {task_text} appears in {len(matches)} resctrl "
            f"control groups ({matches}); expected exactly one before "
            "mutating task membership.")
    control_group = matches[0]
    return (
        control_group,
        _mpam_resctrl_monitor_group_for_task(control_group, task_text),
    )


def _mpam_l3_max_preflight_supervisor_children():
    """Prove the current thread exposes the proc children interface."""
    thread_id = threading.get_native_id()
    children_path = f"/proc/self/task/{thread_id}/children"
    try:
        with open(children_path, "r", encoding="utf-8") as f:
            f.read()
    except OSError as error:
        raise _MpamBlocked(
            "Could not read supervisor child-process interface "
            f"{children_path}: errno {error.errno}: {error}") from error


def _mpam_stable_workload_process_tree(pid):
    """Return the workload tree only after two identical snapshots."""
    try:
        first = glt_proc.process_tree(pid)
        time.sleep(0.05)
        second = glt_proc.process_tree(pid)
    except OSError as error:
        raise _MpamBlocked(str(error)) from error
    if first == second:
        return second
    return None


def _mpam_l3_max_vm_descendant_comms(supervisor_pid, process_tree):
    """Return exact descendant comm values, blocking ambiguous VM-like names."""
    comm_by_pid = {}
    ambiguous_descendants = []
    for process_id in set(process_tree[0]) - {supervisor_pid}:
        try:
            comm = glt_proc.process_comm(process_id)
        except OSError as error:
            raise _MpamBlocked(
                f"Could not read stress-ng descendant {process_id} comm: "
                f"{error}") from error
        comm_by_pid[process_id] = comm
        if comm == MPAM_L3_MAX_LIVE_VM_WORKER_COMM:
            continue
        if comm.startswith("stress-ng") or "vm" in comm.lower():
            ambiguous_descendants.append(f"{process_id}:{comm}")
    if ambiguous_descendants:
        raise _MpamBlocked(
            "Ambiguous stress-ng VM worker identity; expected exact "
            f"comm {MPAM_L3_MAX_LIVE_VM_WORKER_COMM!r}, observed "
            f"{sorted(ambiguous_descendants)}.")
    return comm_by_pid


def _mpam_require_l3_max_vm_comm(comm_by_pid, pid, role):
    """Require one process to have the exact stress-ng VM comm."""
    comm = comm_by_pid.get(pid)
    if comm != MPAM_L3_MAX_LIVE_VM_WORKER_COMM:
        raise _MpamBlocked(
            f"stress-ng {role} {pid} has comm {comm!r}; expected "
            f"{MPAM_L3_MAX_LIVE_VM_WORKER_COMM!r}.")


def _mpam_l3_max_worker_leaf_for_parent(process_tree, comm_by_pid, parent_pid):
    """Return the single workload leaf for one stress-ng VM instance parent."""
    _mpam_require_l3_max_vm_comm(comm_by_pid, parent_pid, "direct child")
    child_pids = process_tree[1].get(str(parent_pid), set())
    if len(child_pids) != 1:
        raise _MpamBlocked(
            f"stress-ng instance parent {parent_pid} does not have "
            f"exactly one child: {sorted(child_pids)}.")
    worker_pid = next(iter(child_pids))
    _mpam_require_l3_max_vm_comm(comm_by_pid, worker_pid, "workload leaf")
    nested_children = process_tree[1].get(str(worker_pid), set())
    if nested_children:
        raise _MpamBlocked(
            "Nested stress-ng VM workload topology below leaf "
            f"{worker_pid}: {sorted(nested_children)}.")
    return worker_pid


def _mpam_l3_max_vm_worker_pids(supervisor_pid, process_tree):
    """Return exact stress-ng VM workload-leaf process IDs."""
    supervisor_pid = str(supervisor_pid)
    comm_by_pid = _mpam_l3_max_vm_descendant_comms(
        supervisor_pid, process_tree)
    parent_pids = process_tree[1].get(supervisor_pid, set())
    if len(parent_pids) != MPAM_L3_MAX_LIVE_STRESS_WORKERS:
        raise _MpamBlocked(
            "stress-ng workload topology has "
            f"{len(parent_pids)} direct instance parents; expected "
            f"{MPAM_L3_MAX_LIVE_STRESS_WORKERS} direct parent/leaf pairs.")

    worker_pids = {
        _mpam_l3_max_worker_leaf_for_parent(
            process_tree, comm_by_pid, parent_pid)
        for parent_pid in sorted(parent_pids)
    }

    expected_descendants = parent_pids | worker_pids
    unexpected_descendants = (
        set(process_tree[0]) - {supervisor_pid} - expected_descendants)
    if unexpected_descendants:
        raise _MpamBlocked(
            "stress-ng workload topology has unexpected descendants outside "
            f"the direct parent/leaf pairs: {sorted(unexpected_descendants)}.")
    if len(worker_pids) != MPAM_L3_MAX_LIVE_STRESS_WORKERS:
        raise _MpamBlocked(
            "stress-ng workload topology produced "
            f"{len(worker_pids)} direct workload leaves; expected "
            f"{MPAM_L3_MAX_LIVE_STRESS_WORKERS}.")
    return worker_pids


def _mpam_validate_l3_max_placement(
        task_ids, worker_pids, attribution, require_worker_footprint=False):
    """Require workload CPU affinity, actual CPU, and worker memory locality."""
    allowed_cpus = set(attribution['cpus'])
    observed = {}
    try:
        for task_id in task_ids:
            observed[task_id] = glt_proc.task_cpu(task_id)
    except (OSError, ValueError, IndexError) as error:
        raise _MpamBlocked(
            f"Could not prove workload CPU placement: {error}") from error
    outside = {
        task_id: cpu for task_id, cpu in observed.items()
        if cpu not in allowed_cpus
    }
    if outside:
        raise _MpamBlocked(
            "Workload task CPU placement escaped selected node "
            f"{attribution['node']}: {outside}; allowed CPUs "
            f"{sorted(allowed_cpus)}.")

    try:
        task_affinity = {
            task_id: glt_proc.task_allowed_cpus(task_id)
            for task_id in task_ids
        }
    except (OSError, ValueError) as error:
        raise _MpamBlocked(
            f"Could not prove workload CPU affinity: {error}") from error
    affinity_outside = {
        task_id: sorted(cpus)
        for task_id, cpus in task_affinity.items()
        if not cpus or not cpus.issubset(allowed_cpus)
    }
    if affinity_outside:
        raise _MpamBlocked(
            "Workload CPU affinity mask is not confined to selected node "
            f"{attribution['node']}: {affinity_outside}; allowed CPUs "
            f"{sorted(allowed_cpus)}.")

    if not require_worker_footprint:
        return

    try:
        worker_numa_pages = {
            worker_pid: glt_proc.task_anonymous_numa_pages_by_node(worker_pid)
            for worker_pid in worker_pids
        }
    except (OSError, ValueError) as error:
        raise _MpamBlocked(
            f"Could not prove worker memory locality: {error}") from error
    worker_local_bytes = {
        worker_pid: (
            pages_by_node.get(attribution['node'], 0) * mmap.PAGESIZE)
        for worker_pid, pages_by_node in worker_numa_pages.items()
    }
    remote_pages = {
        worker_pid: {
            node: pages for node, pages in pages_by_node.items()
            if node != attribution['node'] and pages > 0
        }
        for worker_pid, pages_by_node in worker_numa_pages.items()
    }
    remote_pages = {
        worker_pid: pages_by_node
        for worker_pid, pages_by_node in remote_pages.items()
        if pages_by_node
    }
    empty_local_footprints = {
        worker_pid: local_bytes
        for worker_pid, local_bytes in worker_local_bytes.items()
        if local_bytes <= 0
    }
    aggregate_local_bytes = sum(worker_local_bytes.values())
    if (remote_pages or empty_local_footprints
            or aggregate_local_bytes < MPAM_L3_MAX_LIVE_MIN_RESIDENT_BYTES):
        raise _MpamBlocked(
            "Worker memory locality is not confined to selected NUMA node "
            f"{attribution['node']}: remote pages {remote_pages}, workers with "
            "no local workload-anonymous footprint "
            f"{empty_local_footprints}; aggregate local workload-anonymous "
            f"footprint {aggregate_local_bytes} bytes, required at least "
            f"{MPAM_L3_MAX_LIVE_MIN_RESIDENT_BYTES} bytes across all "
            "stress-ng VM workers.")


def _mpam_validate_l3_max_membership_snapshot(
        group, pid, attribution, deadline, require_worker_footprint=False):
    """Require one exact workload membership and placement snapshot."""
    _mpam_require_l3_max_deadline(deadline, "membership validation")
    try:
        pre_placement_group_tasks = _mpam_read_group_task_ids(group)
    except OSError as error:
        raise _MpamBlocked(
            f"Could not read {group}/tasks for membership proof: "
            f"{error}") from error
    snapshot = _mpam_stable_workload_process_tree(pid)
    if snapshot is None:
        raise _MpamBlocked(
            "stress-ng process/thread tree did not stabilize.")
    task_ids = set().union(*snapshot[0].values())
    worker_pids = _mpam_l3_max_vm_worker_pids(pid, snapshot)
    if pre_placement_group_tasks != task_ids:
        missing = task_ids - pre_placement_group_tasks
        extra = pre_placement_group_tasks - task_ids
        raise _MpamBlocked(
            f"stress-ng observed IDs {sorted(task_ids)} must exactly match "
            f"{group}/tasks; missing {sorted(missing)}, "
            f"extra {sorted(extra)}.")
    _mpam_validate_l3_max_placement(
        task_ids, worker_pids, attribution,
        require_worker_footprint=require_worker_footprint)
    try:
        post_placement_group_tasks = _mpam_read_group_task_ids(group)
    except OSError as error:
        raise _MpamBlocked(
            f"Could not re-read {group}/tasks after placement proof: "
            f"{error}") from error
    if post_placement_group_tasks != pre_placement_group_tasks:
        missing = pre_placement_group_tasks - post_placement_group_tasks
        extra = post_placement_group_tasks - pre_placement_group_tasks
        raise _MpamBlocked(
            f"{group}/tasks changed during membership proof; missing "
            f"{sorted(missing)}, extra {sorted(extra)}.")
    _mpam_require_l3_max_deadline(
        deadline, "after membership validation")


def _mpam_wait_for_l3_max_membership(
        group, workload, attribution, deadline, require_worker_footprint=False):
    """Wait for full recursive workload membership and placement proof."""
    pid = getattr(workload, "pid", workload)
    membership_deadline = min(
        deadline,
        time.monotonic() + MPAM_L3_MAX_LIVE_MEMBERSHIP_TIMEOUT_SEC)
    last_reason = (
        f"stress-ng PID {pid} did not produce a complete membership proof.")
    while time.monotonic() <= membership_deadline:
        if hasattr(workload, "poll"):
            _mpam_raise_if_l3_max_stress_exited(workload, "membership proof")
        try:
            _mpam_validate_l3_max_membership_snapshot(
                group, pid, attribution, deadline,
                require_worker_footprint=require_worker_footprint)
        except _MpamBlocked as error:
            last_reason = str(error)
            time.sleep(0.1)
            continue
        now = time.monotonic()
        if now > membership_deadline:
            last_reason = (
                "stress-ng membership proof completed after the membership "
                "deadline.")
            break
        return
    raise _MpamBlocked(
        f"{last_reason} Deadline expired within the "
        f"{MPAM_L3_MAX_LIVE_MEMBERSHIP_TIMEOUT_SEC}s membership window.")


def _mpam_l3_max_stress_argv(node):
    """Return the exact stress-ng command used for L3_MAX live-effect."""
    return [
        "numactl", f"--cpunodebind={node}", f"--membind={node}",
        "stress-ng", "--vm", str(MPAM_L3_MAX_LIVE_STRESS_WORKERS),
        "--vm-bytes", MPAM_L3_MAX_LIVE_VM_BYTES,
        "--vm-keep", "--oomable", "--timeout", f"{MPAM_L3_MAX_LIVE_STRESS_TIMEOUT_SEC}s",
    ]


def _mpam_l3_max_stress_result_message(result):
    """Return human-readable stress-ng completion evidence."""
    if result.stdout:
        return (
            f"stress-ng exited with status {result.returncode}. "
            f"Captured output: {result.stdout}"
        )
    return f"stress-ng exited with status {result.returncode}."


def _mpam_l3_max_blocked_message(error, output):
    """Keep the primary blockage and append evidence only after cleanup."""
    message = str(error)
    if output is None and error.early_exit is not None:
        return f"{message} output unread: process group not proven gone."
    if output:
        return f"{message} Captured output: {output}"
    return message


def _mpam_raise_if_l3_max_stress_exited(proc, phase):
    """Raise a typed blocked result if the stress-ng leader already exited."""
    returncode = proc.poll()
    if returncode is not None:
        error = _MpamBlocked(
            f"stress-ng exited with status {returncode} during {phase}.")
        error.early_exit = returncode, phase
        raise error


def _mpam_l3_max_stress_summary_count(output, label):
    """Parse one stress-ng summary count from bounded process output."""
    matches = re.findall(
        rf"^\s*stress-ng:\s+\w+:\s+\[\d+\]\s+{label}:\s+(\d+)\b",
        output,
        re.MULTILINE,
    )
    if not matches:
        raise _MpamBlocked(
            f"stress-ng output did not include required '{label}:' "
            "status summary.")
    if len(matches) != 1:
        raise _MpamBlocked(
            f"stress-ng output included multiple '{label}:' status "
            f"summaries {matches}; expected exactly one.")
    return int(matches[0])


def _mpam_l3_max_parse_stress_summary(output):
    """Parse stress-ng status lines without using performance metrics."""
    if not re.search(
            r"^\s*stress-ng:\s+\w+:\s+\[\d+\]\s+successful run completed\b",
            output,
            re.MULTILINE):
        raise _MpamBlocked(
            "stress-ng output did not include a successful completion line.")
    return {
        'skipped': _mpam_l3_max_stress_summary_count(output, "skipped"),
        'passed': _mpam_l3_max_stress_summary_count(output, "passed"),
        'failed': _mpam_l3_max_stress_summary_count(output, "failed"),
        'metrics_untrustworthy': _mpam_l3_max_stress_summary_count(
            output, "metrics untrustworthy"),
    }


def _mpam_validate_l3_max_stress_result(result):
    """Validate stress-ng status output and return parsed summary counts."""
    if result.returncode != 0:
        raise ValueError(
            _mpam_l3_max_stress_result_message(result))
    summary = _mpam_l3_max_parse_stress_summary(result.stdout)
    if summary['passed'] != MPAM_L3_MAX_LIVE_STRESS_WORKERS:
        raise ValueError(
            "stress-ng status summary reported an unexpected passed count: "
            f"passed={summary['passed']}, expected "
            f"{MPAM_L3_MAX_LIVE_STRESS_WORKERS}.")
    bad_counts = {
        "skipped": summary['skipped'],
        "failed": summary['failed'],
        "metrics untrustworthy": summary['metrics_untrustworthy'],
    }
    bad_counts = {
        name: count for name, count in bad_counts.items() if count != 0
    }
    if bad_counts:
        raise ValueError(
            "stress-ng status summary reported non-zero bad counts: "
            f"{bad_counts}.")
    return summary


def _mpam_wait_remaining(deadline):
    """Return remaining seconds before the monotonic deadline."""
    return max(0.0, deadline - time.monotonic())


def _mpam_require_l3_max_deadline(deadline, phase):
    """Block when the shared launch-time outer deadline has already expired."""
    if time.monotonic() > deadline:
        raise _MpamBlocked(
            f"L3_MAX live-effect deadline expired before {phase}.")


def _mpam_sleep_l3_max_with_deadline(seconds, deadline, phase):
    """Sleep only when the shared deadline has enough remaining budget."""
    if _mpam_wait_remaining(deadline) < seconds:
        raise _MpamBlocked(
            f"L3_MAX live-effect deadline would expire during {phase}.")
    time.sleep(seconds)
    if time.monotonic() > deadline:
        raise _MpamBlocked(
            f"L3_MAX live-effect deadline expired after {phase}.")


def _mpam_collect_l3_max_samples(proc, deadline, resources, label, logger):
    """Collect ten live llc_occupancy samples while workload is running."""
    group, placement, occupancy_files = resources
    samples = []
    for index in range(MPAM_L3_MAX_LIVE_SAMPLE_COUNT):
        _mpam_require_l3_max_deadline(deadline, f"sample {index}")
        _mpam_raise_if_l3_max_stress_exited(
            proc, f"sample {index} ({len(samples)}/"
            f"{MPAM_L3_MAX_LIVE_SAMPLE_COUNT} samples collected)")
        try:
            _mpam_validate_l3_max_membership_snapshot(
                group, proc.pid, placement, deadline,
                require_worker_footprint=True)
        except _MpamBlocked as error:
            raise _MpamBlocked(
                f"sample {index} before membership validation failed: {error}") from error
        try:
            sample_value = resctrl_utils.sum_l3_occupancy(occupancy_files)
        except (OSError, ValueError) as error:
            raise _MpamBlocked(
                f"Could not read numeric llc_occupancy sample {index}: {error}"
            ) from error
        samples.append(sample_value)
        logger.info(
            "L3_MAX window %s sample=%s occupancy=%s deadline slack=%.3fs.",
            label, index, sample_value, _mpam_wait_remaining(deadline))
        if index == MPAM_L3_MAX_LIVE_SAMPLE_COUNT - 1:
            try:
                _mpam_validate_l3_max_membership_snapshot(
                    group, proc.pid, placement, deadline,
                    require_worker_footprint=True)
            except _MpamBlocked as error:
                raise _MpamBlocked(
                    f"sample {index} after membership validation failed: {error}") from error
        _mpam_require_l3_max_deadline(deadline, f"after sample {index}")
        _mpam_raise_if_l3_max_stress_exited(
            proc, f"after sample {index}")
        if index != MPAM_L3_MAX_LIVE_SAMPLE_COUNT - 1:
            _mpam_sleep_l3_max_with_deadline(
                MPAM_L3_MAX_LIVE_SAMPLE_INTERVAL_SEC,
                deadline,
                f"sample interval {index}",
            )
    return tuple(samples)


def _mpam_l3_max_group_exists(pgid):
    """Return False only when signal-0 proves the launched group absent."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _mpam_l3_max_output(pgid, capture):
    """Return bounded captured output only after the group is absent."""
    if capture is None or capture.closed:
        return None
    if _mpam_l3_max_group_exists(pgid):
        return None
    try:
        capture.seek(0, os.SEEK_END)
        output_size = capture.tell()
        truncated = output_size > MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES
        if truncated:
            capture.seek(-MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES, os.SEEK_END)
            data = capture.read(MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES)
        else:
            capture.seek(0)
            data = capture.read()
    except OSError as error:
        return f"<could not read process output: {error}>"
    text = data.decode("utf-8", errors="replace").strip()
    if truncated:
        return (
            f"<kept last {MPAM_L3_MAX_LIVE_OUTPUT_LIMIT_BYTES} of {output_size} bytes>\n"
            f"{text}"
        )
    return text


def _mpam_wait_for_l3_max_stress_exit(proc, deadline, argv, capture):
    """Wait for stress-ng to exit within the outer deadline."""
    try:
        returncode = proc.wait(timeout=_mpam_wait_remaining(deadline))
    except subprocess.TimeoutExpired as error:
        output = _mpam_l3_max_output(proc.pid, capture)
        output_note = f" Captured output: {output}" if output else ""
        raise _MpamBlocked(
            "stress-ng did not exit within the live-effect deadline."
            f"{output_note}"
        ) from error
    return subprocess.CompletedProcess(argv, returncode, stdout=_mpam_l3_max_output(proc.pid, capture))


def _mpam_l3_max_restore_supervisor(
        thread_id, supervisor_group, supervisor_monitor_group):
    """Restore CTRL before MON and prove the exact original membership."""
    try:
        glt_utils.write_sysfs_value(
            os.path.join(supervisor_group, "tasks"), thread_id)
    except OSError as error:
        raise ValueError(
            f"Could not move supervisor thread {thread_id} back to "
            f"{supervisor_group}/tasks: {error}") from error
    if supervisor_monitor_group is not None:
        try:
            glt_utils.write_sysfs_value(
                os.path.join(supervisor_monitor_group, "tasks"), thread_id)
        except OSError as error:
            raise ValueError(
                f"Could not move supervisor thread {thread_id} back to "
                f"{supervisor_monitor_group}/tasks: {error}") from error
    try:
        restored_group, restored_monitor_group = (
            _mpam_resctrl_group_for_task(thread_id))
    except _MpamBlocked as error:
        raise ValueError(
            f"Could not prove supervisor thread {thread_id} returned to "
            f"control {supervisor_group}, monitor "
            f"{supervisor_monitor_group}: {error}") from error
    if (restored_group, restored_monitor_group) != (
            supervisor_group, supervisor_monitor_group):
        raise ValueError(
            f"supervisor thread {thread_id} restored to control "
            f"{restored_group}, monitor {restored_monitor_group}; "
            f"expected control {supervisor_group}, monitor "
            f"{supervisor_monitor_group}.")


def _mpam_l3_max_require_stress_version():
    """Require the supported status-summary format before resctrl mutation."""
    minimum = "0.16.00"
    try:
        result = subprocess.run(
            ["stress-ng", "--version"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _MpamBlocked(
            f"stress-ng >= {minimum} is required; version check failed: {error}") from error
    match = re.search(r"^stress-ng, version (\d+\.\d+\.\d+)\b", result.stdout)
    if (result.returncode != 0 or match is None
            or glt_utils.parse_linux_version(match[1]) < glt_utils.parse_linux_version(minimum)):
        raise _MpamBlocked(
            f"stress-ng >= {minimum} is required; observed version output "
            f"{result.stdout.strip()!r} (exit {result.returncode}).")


def _mpam_wait_l3_max_clean_baseline(group, occupancy_files):
    """Poll stale cache occupancy with empty task proofs around each read."""
    deadline = time.monotonic() + MPAM_L3_MAX_LIVE_BASELINE_TIMEOUT_SEC
    while True:
        try:
            if _mpam_read_group_task_ids(group):
                raise _MpamBlocked(f"prelaunch task snapshot for {group} was not empty.")
            occupancy = resctrl_utils.sum_l3_occupancy(occupancy_files)
            if _mpam_read_group_task_ids(group):
                raise _MpamBlocked(
                    f"prelaunch task snapshot for {group} was not empty after occupancy read.")
        except (OSError, ValueError) as error:
            raise _MpamBlocked(f"Could not read prelaunch baseline for {group}: {error}") from error
        if occupancy < MPAM_L3_MAX_LIVE_BASELINE_LIMIT:
            return occupancy
        remaining = _mpam_wait_remaining(deadline)
        if remaining <= 0:
            raise _MpamBlocked(
                "prelaunch aggregate llc_occupancy is not clean after "
                f"{MPAM_L3_MAX_LIVE_BASELINE_TIMEOUT_SEC}s: {occupancy} bytes.")
        time.sleep(min(MPAM_L3_MAX_LIVE_BASELINE_POLL_SEC, remaining))


def _mpam_stop_l3_max_workload(proc, primary_error, cleanup_errors, capture):
    """Stop this workload with independent leader and group-absence budgets."""
    errors = []
    output = None
    try:
        if proc.poll() is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    errors.append(str(error))
                try:
                    proc.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired as error:
                    if sig == signal.SIGKILL:
                        errors.append(str(error))
        deadline = time.monotonic() + MPAM_L3_MAX_LIVE_KILL_CLEANUP_TIMEOUT_SEC
        if _mpam_l3_max_group_exists(proc.pid):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                errors.append(str(error))
        while _mpam_l3_max_group_exists(proc.pid):
            if time.monotonic() >= deadline:
                errors.append(
                    f"process group {proc.pid} still exists after "
                    f"{MPAM_L3_MAX_LIVE_KILL_CLEANUP_TIMEOUT_SEC} seconds")
                break
            time.sleep(MPAM_L3_MAX_LIVE_KILL_CLEANUP_POLL_SEC)
        cleanup_errors.extend(
            f"Failed to terminate stress-ng process group {proc.pid}: {error}"
            for error in errors)
        output = (_mpam_l3_max_output(proc.pid, capture)
                  if isinstance(primary_error, _MpamBlocked) else None)
    except (OSError, subprocess.TimeoutExpired) as error:
        cleanup_errors.append(
            f"Failed to terminate stress-ng process group {proc.pid}: {error}")
    return output


def _mpam_quiesce_l3_max_workload(group, deadline, cleanup_errors):
    """Verify the post-workload task state even when quiescence misses its deadline."""
    time.sleep(MPAM_L3_MAX_LIVE_QUIESCE_SEC)
    try:
        _mpam_require_l3_max_deadline(deadline, "after cleanup quiesce")
    finally:
        try:
            residual_tasks = _mpam_read_group_task_ids(group)
        except OSError as error:
            cleanup_errors.append(f"Could not verify residual task state in {group}: {error}")
        else:
            if residual_tasks:
                cleanup_errors.append(
                    f"residual tasks remained in {group}: {sorted(residual_tasks)}")


# Keep lexical capture close, supervisor restoration and primary-error ordering together.
# TODO: Define a shared ownership/lifecycle boundary before generalizing these resources.
@contextmanager
def _mpam_l3_max_workload(group, placement, cleanup_errors):
    # pylint: disable=too-many-branches
    """Own supervisor restoration, workload teardown, capture and quiescence."""
    proc = None
    supervisor_assigned = False
    thread_id = None
    primary_error = None
    try:
        thread_id = threading.get_native_id()
        supervisor_group, supervisor_monitor_group = (
            _mpam_resctrl_group_for_task(thread_id))
        try:
            glt_utils.write_sysfs_value(
                os.path.join(group, "tasks"), thread_id)
            supervisor_assigned = True
            supervisor_tasks = _mpam_read_group_task_ids(group)
        except OSError as error:
            raise _MpamBlocked(
                f"Could not prove supervisor membership in {group}: {error}"
            ) from error
        expected_supervisor_tasks = {str(thread_id)}
        if supervisor_tasks != expected_supervisor_tasks:
            raise _MpamBlocked(
                f"supervisor thread {thread_id} membership in {group}/tasks "
                f"did not exactly match {sorted(expected_supervisor_tasks)} "
                f"before launch: {sorted(supervisor_tasks)}.")

        deadline = time.monotonic() + MPAM_L3_MAX_LIVE_OUTER_TIMEOUT_SEC
        try:
            capture = tempfile.TemporaryFile(mode="w+b")
            try:
                proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    _mpam_l3_max_stress_argv(placement['node']),
                    stdout=capture, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError:
                capture.close()
                raise
        except OSError as error:
            raise _MpamBlocked(
                f"Could not launch stress-ng workload: {error}"
            ) from error
        _mpam_l3_max_restore_supervisor(
            thread_id, supervisor_group, supervisor_monitor_group)
        supervisor_assigned = False

        yield proc, deadline, capture
    except (_MpamBlocked, ValueError, OSError) as error:
        primary_error = error
    finally:
        if supervisor_assigned:
            try:
                _mpam_l3_max_restore_supervisor(
                    thread_id, supervisor_group, supervisor_monitor_group)
            except (ValueError, OSError) as error:
                cleanup_errors.append(str(error))
        if proc is not None:
            try:
                output = _mpam_stop_l3_max_workload(proc, primary_error, cleanup_errors, capture)
            finally:
                try:
                    capture.close()
                except OSError as error:
                    cleanup_errors.append(f"Could not close stress-ng output capture: {error}")
            primary_error = (_MpamBlocked(_mpam_l3_max_blocked_message(primary_error, output))
                             if isinstance(primary_error, _MpamBlocked) else primary_error)
            try:
                _mpam_quiesce_l3_max_workload(group, deadline, cleanup_errors)
            except _MpamBlocked as error:
                primary_error = primary_error or error
    if primary_error is not None:
        raise primary_error


def _mpam_evaluate_l3_max_live_windows(windows):
    """Evaluate complete low/high per-window medians."""
    if len(windows) != len(MPAM_L3_MAX_LIVE_PATTERN):
        raise _MpamBlocked(
            f"incomplete L3_MAX live window set: {len(windows)}/"
            f"{len(MPAM_L3_MAX_LIVE_PATTERN)}.")
    for window in windows:
        if len(window['samples']) != MPAM_L3_MAX_LIVE_SAMPLE_COUNT:
            raise _MpamBlocked(
                f"incomplete sample stream for {window['label']}="
                f"{window['percentage']}: {len(window['samples'])}/"
                f"{MPAM_L3_MAX_LIVE_SAMPLE_COUNT}.")
    observed_pattern = tuple(window['label'] for window in windows)
    if observed_pattern != MPAM_L3_MAX_LIVE_PATTERN:
        raise _MpamBlocked(
            "L3_MAX live windows did not preserve the required low/high "
            f"sequence: expected {MPAM_L3_MAX_LIVE_PATTERN}, observed "
            f"{observed_pattern}.")
    low_medians = [window['median'] for window in windows
                   if window['label'] == "low"]
    high_medians = [window['median'] for window in windows
                    if window['label'] == "high"]
    if max(low_medians) >= min(high_medians):
        raise ValueError(
            "L3_MAX live-effect comparison failed: every low-window median must "
            f"be below every high-window median; low medians {low_medians}, "
            f"high medians {high_medians}.")
    low_center = _median(low_medians)
    high_center = _median(high_medians)
    separation = high_center - low_center
    dispersion = max(
        max(abs(median - low_center) for median in low_medians),
        max(abs(median - high_center) for median in high_medians),
    )
    if separation <= dispersion:
        raise ValueError(
            "L3_MAX live-effect comparison failed: signal separation "
            f"{separation} is not above dispersion {dispersion}.")
    if separation < MPAM_L3_MAX_LIVE_MIN_SEPARATION_BYTES:
        raise ValueError(
            "L3_MAX live-effect comparison failed: signal separation "
            f"{separation} bytes is below 1 MiB.")
    return low_center, high_center, separation, dispersion


def _mpam_msc_regions():
    """Return MPAM MSC MMIO regions from /proc/iomem."""
    msc_iomem_re = re.compile(
        r"^\s*(?P<start>[0-9a-fA-F]+)-(?P<end>[0-9a-fA-F]+)\s+:\s+"
        r"(?P<name>mpam_msc\.\S+)\s+MPAM:MSC\s*$"
    )
    regions = []
    with open(PROC_IOMEM, "r", encoding="utf-8") as iomem:
        for line in iomem:
            match = msc_iomem_re.match(line)
            if match is None:
                continue
            start = int(match.group("start"), 16)
            end = int(match.group("end"), 16)
            if start and end >= start:
                regions.append(
                    (match.group("name"), start, end - start + 1))
    return regions


def _mpam_read32(registers, offset):
    """Read a little-endian 32-bit value from an MSC register view."""
    return struct.unpack_from("<I", registers, offset)[0]


def _mpam_read_idr(registers):
    """Read the complete MPAM feature identification register."""
    mpamf_idr = 0x0000
    mpamf_idr_hi = 0x0004
    idr_ext = 1 << 28

    low = _mpam_read32(registers, mpamf_idr)
    high = _mpam_read32(registers, mpamf_idr_hi) if low & idr_ext else 0
    return low | (high << 32)


def _mpam_write32(registers, offset, value):
    """Write a little-endian 32-bit value to an MSC register view."""
    struct.pack_into("<I", registers, offset, value & 0xFFFFFFFF)


def _validate_mpam_init_registers(  # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
        registers, label):
    """Validate init-programmed allocation registers for each RIS/PARTID."""
    mpamf_iidr = 0x0018
    mpamf_cpor_idr = 0x0030
    mpamf_ccap_idr = 0x0038
    mpamf_mbw_idr = 0x0040
    mpamf_partid_nrw_idr = 0x0050
    mpamcfg_part_sel = 0x0100
    mpamcfg_cmax = 0x0108
    mpamcfg_cmin = 0x0110
    mpamcfg_mbw_min = 0x0200
    mpamcfg_mbw_max = 0x0208
    mpamcfg_cpbm = 0x1000

    idr_has_ccap_part = 1 << 24
    idr_has_cpor_part = 1 << 25
    idr_has_mbw_part = 1 << 26
    idr_has_partid_nrw = 1 << 31
    idr_has_ris = 1 << 32
    ccap_idr_has_cmin = 1 << 29
    ccap_idr_no_cmax = 1 << 30
    mbw_idr_has_min = 1 << 10
    mbw_idr_has_max = 1 << 11
    mbw_max_hardlim = 1 << 31
    part_sel_internal = 1 << 16
    t241_msc_iidr = 0x2410036B

    failures = []
    checks = 0

    original_selector = _mpam_read32(registers, mpamcfg_part_sel)
    ris_count = 0
    try:
        _mpam_write32(registers, mpamcfg_part_sel, 0)
        initial_idr = _mpam_read_idr(registers)
        iidr = _mpam_read32(registers, mpamf_iidr)
        ris_count = (
            ((initial_idr >> 56) & 0xF) + 1
            if initial_idr & idr_has_ris else 1
        )
        ris_features = []

        for ris in range(ris_count):
            _mpam_write32(registers, mpamcfg_part_sel, ris << 24)
            idr = _mpam_read_idr(registers)
            partid_max = idr & 0xFFFF
            if idr & idr_has_partid_nrw:
                partid_max = min(
                    partid_max,
                    _mpam_read32(
                        registers, mpamf_partid_nrw_idr) & 0xFFFF
                )
            ccap_idr = (
                _mpam_read32(registers, mpamf_ccap_idr)
                if idr & idr_has_ccap_part else 0
            )
            cpor_idr = (
                _mpam_read32(registers, mpamf_cpor_idr)
                if idr & idr_has_cpor_part else 0
            )
            mbw_idr = (
                _mpam_read32(registers, mpamf_mbw_idr)
                if idr & idr_has_mbw_part else 0
            )
            ris_features.append(
                (idr, ccap_idr, cpor_idr, mbw_idr, partid_max))

        for ris, features in enumerate(ris_features):
            idr, ccap_idr, cpor_idr, mbw_idr, partid_max = features
            for partid in range(min(2, partid_max + 1)):
                selector = (ris << 24) | partid
                if idr & idr_has_partid_nrw:
                    selector |= part_sel_internal
                _mpam_write32(registers, mpamcfg_part_sel, selector)

                if idr & idr_has_ccap_part:
                    cmax_width = ccap_idr & 0x3F
                    if cmax_width and not (
                            ccap_idr & ccap_idr_no_cmax):
                        actual = (
                            _mpam_read32(registers, mpamcfg_cmax) & 0xFFFF)
                        expected = (
                            (1 << cmax_width) - 1) << (16 - cmax_width)
                        checks += 1
                        if (actual & expected) != expected:
                            failures.append(
                                f"{label} RIS {ris} PARTID {partid} CMAX: "
                                f"expected allowed bits 0x{expected:x}, read "
                                f"0x{actual:04x}"
                            )

                    if ccap_idr & ccap_idr_has_cmin:
                        actual = (
                            _mpam_read32(registers, mpamcfg_cmin) & 0xFFFF)
                        checks += 1
                        if actual != 0:
                            failures.append(
                                f"{label} RIS {ris} PARTID {partid} CMIN: "
                                f"expected 0x0000, read 0x{actual:04x}"
                            )

                cpbm_width = cpor_idr & 0xFFFF
                for word in range((cpbm_width + 31) // 32):
                    implemented = min(32, cpbm_width - word * 32)
                    expected = (
                        0xFFFFFFFF
                        if implemented == 32 else (1 << implemented) - 1
                    )
                    actual = _mpam_read32(registers, mpamcfg_cpbm + word * 4)
                    checks += 1
                    if actual != expected:
                        failures.append(
                            f"{label} RIS {ris} PARTID {partid} CPBM word "
                            f"{word}: expected "
                            f"0x{expected:08x}, read 0x{actual:08x}"
                        )

                bwa_width = min(mbw_idr & 0x3F, 16)
                significant = ((1 << bwa_width) - 1) << (16 - bwa_width)
                if bwa_width and mbw_idr & mbw_idr_has_min:
                    actual = _mpam_read32(registers, mpamcfg_mbw_min) & 0xFFFF
                    expected = 0
                    if iidr == t241_msc_iidr:
                        granule = 1 << (16 - bwa_width)
                        delta = (5 * 0xFFFF) // 100 - 1
                        reduced = (
                            significant - delta
                            if significant > delta else 0
                        )
                        expected = max(reduced, granule)
                    checks += 1
                    if (actual & significant) != (expected & significant):
                        failures.append(
                            f"{label} RIS {ris} PARTID {partid} MBW_MIN: "
                            f"expected implemented "
                            f"bits 0x{expected & significant:04x}, read "
                            f"0x{actual:04x} (mask 0x{significant:04x})"
                        )

                if bwa_width and mbw_idr & mbw_idr_has_max:
                    actual = _mpam_read32(registers, mpamcfg_mbw_max)
                    checks += 1
                    if (actual & significant) != significant:
                        failures.append(
                            f"{label} RIS {ris} PARTID {partid} MBW_MAX: "
                            f"expected implemented "
                            f"bits 0x{significant:04x}, read "
                            f"0x{actual & 0xFFFF:04x}"
                        )

                    max_limit = (mbw_idr >> 8) & 0x3
                    if max_limit != 3:
                        expected_hardlim = int(max_limit == 2)
                        actual_hardlim = int(bool(actual & mbw_max_hardlim))
                        if actual_hardlim != expected_hardlim:
                            failures.append(
                                f"{label} RIS {ris} PARTID {partid} MBW_MAX "
                                f"HARDLIM: expected "
                                f"{expected_hardlim} for MAX_LIM={max_limit}, "
                                f"read {actual_hardlim}"
                            )
    finally:
        _mpam_write32(registers, mpamcfg_part_sel, original_selector)

    return checks, ris_count, failures


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_driver_init_registers(  # pylint: disable=too-many-branches,too-many-locals
        _, __):
    """Validate allocation defaults programmed during MPAM driver init.

    Skips when CONFIG_ARM64_MPAM is not enabled or none of the target
    allocation registers is supported. Fails if resctrl is already
    mounted so driver-init defaults can be observed without changing a
    caller-owned mount. Checks capability-gated CMAX, CMIN, CPBM, MBW_MIN,
    and MBW_MAX values for up to the first two supported PARTIDs on every
    RIS.

    Raw /dev/mem access cannot synchronize PART_SEL with the bound driver or
    enforce the MSC CPU-accessibility mask. Replace it with a kernel-mediated
    snapshot interface when one becomes available.
    """
    dev_mem = "/dev/mem"
    platform_device_path = "/sys/bus/platform/devices"

    nvdev = NvDev()
    if not nvdev.is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not an NVIDIA Vera CPU; test is Vera-specific.")

    mpam_config = "CONFIG_ARM64_MPAM"
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"Kernel config not found at {kconfig_path}; "
            f"cannot verify {mpam_config}."
        )
    mpam = kconfig.get(mpam_config)
    if mpam != "y":
        GLTCore.GLTTestResult.skip(
            f"{mpam_config} is not enabled (found {mpam!r}); MPAM unavailable."
        )

    try:
        resctrl_mounts = resctrl_utils.mount_points()
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot determine resctrl mount state from {resctrl_utils.PROC_MOUNTS}: "
            f"{error}"
        )
    if resctrl_mounts:
        GLTCore.GLTTestResult.fail(
            "resctrl is mounted at "
            + ", ".join(resctrl_mounts)
            + "; unmount it before running this test so MPAM driver-init "
            "defaults can be observed."
        )

    regions = []
    try:
        regions = _mpam_msc_regions()
        if not regions:
            GLTCore.GLTTestResult.fail(
                "No device-specific MPAM MSC MMIO regions were found.")
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Cannot read MPAM MSC resources from {PROC_IOMEM}: {error}")

    unbound = [
        name for name, _, _ in regions
        if not os.path.islink(
            os.path.join(platform_device_path, name, "driver"))
    ]
    if unbound:
        GLTCore.GLTTestResult.fail(
            "MPAM MSC driver initialization is not complete for: "
            + ", ".join(unbound)
        )

    checked_registers = 0
    failures = []
    access_errors = []
    for name, base, size in regions:
        page_size = mmap.PAGESIZE
        map_base = base & ~(page_size - 1)
        register_base = base - map_base
        map_length = (
            register_base + size + page_size - 1) & ~(page_size - 1)
        descriptor = None
        try:
            descriptor = os.open(dev_mem, os.O_RDWR | os.O_SYNC)
            with mmap.mmap(
                    descriptor, map_length, flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=map_base) as mapping:
                with memoryview(mapping)[
                        register_base:register_base + size] as registers:
                    checks, _ris_count, register_failures = (
                        _validate_mpam_init_registers(
                            registers, f"{name} @ 0x{base:x}")
                    )
        except (OSError, ValueError, struct.error) as error:
            access_errors.append(
                f"{name} @ 0x{base:x} through {dev_mem}: {error}")
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)
        checked_registers += checks
        failures.extend(register_failures)

    if failures:
        GLTCore.GLTTestResult.fail(
            "MPAM MSC driver-init register validation failed:\n"
            + "\n".join(f"- {failure}" for failure in failures)
            + "\nKernel commit 09b89d2 should be backported."
        )
    if access_errors:
        GLTCore.GLTTestResult.fail(
            "Raw access was unavailable for one or more MPAM MSCs:\n"
            + "\n".join(f"- {error}" for error in access_errors)
        )
    if checked_registers == 0:
        GLTCore.GLTTestResult.skip(
            "No discovered MPAM MSC advertises CMAX, CMIN, CPBM, "
            "MBW_MIN, or MBW_MAX. driver-init register validation is not "
            "applicable."
        )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mb_max_lim")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mba_hardlim_exposure(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        _, logger):
    """
    Verify Vera exposes MB_HLIM exactly when MBA max_lim is zero.

    Compare the reported capability with the info directory and root
    schemata. Exposed MB_HLIM and MB resources must have the same domains.
    This checks exposure, not the hardware probe or HARDLIM programming.

    JIRA: DGX-17719
    PATCH: NVIDIA SAUCE f72a73016915fd1d429d244aeef45e68c92a0aba
    PATCH: NVIDIA SAUCE 1046a99778640817a76d4a1b3f19258ec2141434
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("MB_HLIM exposure validation is Vera-specific.")

    max_lim_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB", "max_lim")
    hlim_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB_HLIM")
    schemata_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "schemata")
    cleanup_errors = []

    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        try:
            with open(max_lim_path, "r", encoding="ascii") as f:
                max_lim = f.read()
            schemas = resctrl_utils.read_schemata(schemata_path, ("MB", "MB_HLIM"))
            try:
                hlim_stat = os.lstat(hlim_path)
            except FileNotFoundError:
                hlim_stat = None
        except (OSError, UnicodeError) as e:
            GLTCore.GLTTestResult.fail(f"Failed to inspect MB_HLIM exposure: {e}")
        except ValueError as e:
            GLTCore.GLTTestResult.fail(f"Malformed resctrl schemata: {e}")

        if max_lim not in ("0\n", "1\n", "2\n", "3\n"):
            GLTCore.GLTTestResult.fail(
                f"Invalid max_lim content at {max_lim_path}: {max_lim!r}.")
        max_lim = int(max_lim)

        if "MB" not in schemas:
            GLTCore.GLTTestResult.fail(
                f"MB resource is missing from {schemata_path}.")

        expected = max_lim == 0
        info_present = hlim_stat is not None
        schema_present = "MB_HLIM" in schemas
        logger.info(
            "MAX_LIM=%d, info/MB_HLIM present=%s, MB_HLIM schemata present=%s",
            max_lim, info_present, schema_present)
        if info_present != expected or schema_present != expected:
            GLTCore.GLTTestResult.fail(
                f"MB_HLIM exposure mismatch for max_lim={max_lim}: "
                f"expected info directory and schemata entry "
                f"{'present' if expected else 'absent'}, "
                f"found info entry present={info_present}, "
                f"schemata entry present={schema_present}.")
        if expected:
            if not stat.S_ISDIR(hlim_stat.st_mode):
                GLTCore.GLTTestResult.fail(
                    f"{hlim_path} exists but is not a directory.")
            if schemas["MB_HLIM"].keys() != schemas["MB"].keys():
                GLTCore.GLTTestResult.fail(
                    f"MB_HLIM domains {sorted(schemas['MB_HLIM'])} do not match "
                    f"MB domains {sorted(schemas['MB'])}.")
            logger.info("MB_HLIM and MB share domains %s.", sorted(schemas["MB"]))

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_mba_hardlim_exposure")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mba_hardlim_valid_values(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        _, logger):
    """
    Verify per-domain MB_HLIM 0/1 writes preserve all other MBA controls.

    Establish a zero baseline in a private, empty group, enable domains
    individually, then disable them individually. Repeat each per-domain
    write to exercise valid no-op updates. Read back all MB and MB_HLIM
    values after every write. This validates cached resctrl configuration,
    not physical HARDLIM programming.

    JIRA: DGX-17722
    PATCH: NVIDIA SAUCE 407fd9260213b8c16771755c9867dc4c42352ba3
    PATCH: NVIDIA SAUCE 442299b435c3a0bb459d62df990968b1db9869b6
    PATCH: NVIDIA SAUCE a3f61fa60cfc25ae781c282fac15a2ba0e7cf0a7
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("MB_HLIM valid-value validation is Vera-specific.")

    cleanup_errors = []
    group = None
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        try:
            with open(os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB", "max_lim"),
                      "r", encoding="ascii") as info_file:
                max_lim = info_file.read()
            if max_lim not in ("0\n", "1\n", "2\n", "3\n"):
                raise ValueError(f"Invalid max_lim content: {max_lim!r}")
            if max_lim != "0\n":
                GLTCore.GLTTestResult.skip(
                    f"MB_HLIM is not writable with max_lim={max_lim.strip()}.")

            root_schema = _read_mb_hlim_schemata(
                os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "schemata"))
            group_path = os.path.join(
                resctrl_utils.RESCTRL_MOUNT_POINT, f"glt_mb_hlim_valid_{os.getpid()}_{time.time_ns()}")
            os.mkdir(group_path)
            group = group_path  # Never adopt or remove a pre-existing group.
            schemata_path = os.path.join(group, "schemata")
            initial = _read_mb_hlim_schemata(schemata_path)
            if initial["MB_HLIM"].keys() != root_schema["MB_HLIM"].keys():
                raise ValueError("Private-group MB_HLIM domains differ from the root group.")
            domains = sorted(initial["MB_HLIM"])
            expected = {"MB": initial["MB"], "MB_HLIM": dict(initial["MB_HLIM"])}

            # Defaults are covered separately by DGX-17720; set our own baseline.
            write_steps = [(dict.fromkeys(domains, 0), 1)]
            for value in (1, 0):
                write_steps.extend(({domain: value}, 2) for domain in domains)
            if len(domains) == 1:
                logger.info("One MB_HLIM domain; cross-domain isolation cannot be checked.")

            for updates, repetitions in write_steps:
                expected["MB_HLIM"].update(updates)
                payload = ("MB_HLIM:" + ";".join(
                    f"{domain}={value}" for domain, value in updates.items()) + "\n").encode("ascii")
                for attempt in range(repetitions):
                    try:
                        with open(schemata_path, "wb", buffering=0) as schemata_file:
                            written = schemata_file.write(payload)
                    except OSError as error:
                        raise OSError(
                            error.errno, f"Valid MB_HLIM write {payload!r} failed: {error}"
                        ) from error
                    if written != len(payload):
                        GLTCore.GLTTestResult.fail(
                            f"Short MB_HLIM write for {payload!r}: "
                            f"expected {len(payload)} bytes, wrote {written}.")
                    actual = _read_mb_hlim_schemata(schemata_path)
                    if actual != expected:
                        GLTCore.GLTTestResult.fail(
                            f"After valid write {payload!r}, expected {expected}, read {actual}.")
                    logger.info(
                        "Verified %r (write %d/%d); other MB_HLIM domains and MB unchanged.",
                        payload, attempt + 1, repetitions)
        except (OSError, ValueError) as error:
            GLTCore.GLTTestResult.fail(f"MB_HLIM valid-value validation failed: {error}")
        finally:
            if group is not None:
                try:
                    os.rmdir(group)
                except OSError as error:
                    message = f"Failed to remove test-owned resctrl group {group}: {error}"
                    logger.warning("%s", message)
                    cleanup_errors.append(message)

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_mba_hardlim_exposure")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mba_hardlim_validation_defaults(  # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks,too-many-statements
        _, logger):
    """
    Verify MB_HLIM rejects invalid values and initializes new groups to zero.

    Use only a private, empty control group. Rejected writes must return
    EINVAL and preserve MB and MB_HLIM, including a staged multi-domain
    write when possible. Recreate the group after enabling hard limiting
    and require zero defaults again. This checks the resctrl interface,
    not physical HARDLIM programming.

    JIRA: DGX-17720
    PATCH: NVIDIA SAUCE 407fd9260213b8c16771755c9867dc4c42352ba3
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("MB_HLIM validation is Vera-specific.")

    cleanup_errors = []
    group = None
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        try:
            with open(os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB", "max_lim"),
                      "r", encoding="ascii") as info_file:
                max_lim = info_file.read()
            if max_lim not in ("0\n", "1\n", "2\n", "3\n"):
                raise ValueError(f"Invalid max_lim content: {max_lim!r}")
            if max_lim != "0\n":
                GLTCore.GLTTestResult.skip(
                    f"MB_HLIM is not writable with max_lim={max_lim.strip()}.")
            with open(os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB_HLIM",
                                   "schema_format"), "r", encoding="ascii") as info_file:
                schema_format = info_file.read()
            if schema_format != "0/1\n":
                raise ValueError(f"Unexpected MB_HLIM schema_format: {schema_format!r}")

            root_schema = _read_mb_hlim_schemata(
                os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "schemata"))
            domains = sorted(root_schema["MB_HLIM"])
            group_path = os.path.join(
                resctrl_utils.RESCTRL_MOUNT_POINT, f"glt_mb_hlim_{os.getpid()}_{time.time_ns()}")
            schemata_path = os.path.join(group_path, "schemata")
            # kstrtou32(..., 10, ...) accepts +1 and 01; they are not invalid.
            invalid_values = ("2", "-1", "0x1", "abc", "1.0", "4294967296", "")

            for generation in range(2):
                os.mkdir(group_path)
                group = group_path  # Own the group only after mkdir succeeds.
                initial = _read_mb_hlim_schemata(schemata_path)
                if initial["MB_HLIM"] != dict.fromkeys(domains, 0):
                    GLTCore.GLTTestResult.fail(
                        f"New group {group} must default all MB_HLIM domains to 0; "
                        f"read {initial['MB_HLIM']}.")
                logger.info("Group creation %d: MB_HLIM defaults to 0 on domains %s.",
                            generation + 1, domains)
                if generation:
                    break

                for baseline in (0, 1):
                    payload = ("MB_HLIM:" + ";".join(
                        f"{domain}={baseline}" for domain in domains) + "\n").encode("ascii")
                    with open(schemata_path, "wb", buffering=0) as schemata_file:
                        if schemata_file.write(payload) != len(payload):
                            GLTCore.GLTTestResult.fail("Short MB_HLIM baseline write.")
                    expected = {
                        "MB": initial["MB"],
                        "MB_HLIM": dict.fromkeys(domains, baseline),
                    }
                    if _read_mb_hlim_schemata(schemata_path) != expected:
                        GLTCore.GLTTestResult.fail(
                            f"Could not establish MB_HLIM={baseline} without changing MB.")

                    invalid_assignments = [
                        f"{domain}={value}" for domain in domains for value in invalid_values
                    ]
                    if len(domains) > 1:
                        invalid_assignments.append(
                            f"{domains[0]}={1 - baseline};{domains[1]}=2")
                    else:
                        logger.info("One MB_HLIM domain; multi-domain staging check not applicable.")

                    for assignment in invalid_assignments:
                        payload = f"MB_HLIM:{assignment}\n".encode("ascii")
                        with open(schemata_path, "wb", buffering=0) as schemata_file:
                            try:
                                schemata_file.write(payload)
                            except OSError as error:
                                if error.errno != errno.EINVAL:
                                    raise OSError(
                                        error.errno, f"Expected EINVAL for {payload!r}: {error}"
                                    ) from error
                            else:
                                GLTCore.GLTTestResult.fail(
                                    f"Invalid MB_HLIM write was accepted: {payload!r}")
                        actual = _read_mb_hlim_schemata(schemata_path)
                        if actual != expected:
                            GLTCore.GLTTestResult.fail(
                                f"Rejected write {payload!r} changed configuration: "
                                f"expected {expected}, read {actual}.")
                        logger.debug("Rejected %r with EINVAL; MB and MB_HLIM unchanged.", payload)
                    logger.info("MB_HLIM=%d: verified %d invalid writes preserve configuration.",
                                baseline, len(invalid_assignments))

                # Leave the old group's values at 1 to exercise reinitialization.
                os.rmdir(group)
                group = None
        except (OSError, ValueError) as error:
            GLTCore.GLTTestResult.fail(f"MB_HLIM validation failed: {error}")
        finally:
            if group is not None:
                try:
                    os.rmdir(group)
                except OSError as error:
                    message = f"Failed to remove test-owned resctrl group {group}: {error}"
                    logger.warning("%s", message)
                    cleanup_errors.append(message)

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_info")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_mb_max_lim(_, logger):  # pylint: disable=too-many-branches
    """
    Verify info/MB/max_lim is read-only and reports a valid capability.

    PATCH: NVIDIA SAUCE 3d3b9ca0419275e0c6992e25052fb7d719edf573
    PATCH: NVIDIA SAUCE e2a11af3dfd0c2c12481c1b63ad3289f78b5fbe1
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This CPU does not expose the MB/max_lim capability interface."
        )

    max_lim_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB", "max_lim")
    backport_hint = (
        "Consider backporting NVIDIA SAUCE commits "
        "3d3b9ca0419275e0c6992e25052fb7d719edf573 and "
        "e2a11af3dfd0c2c12481c1b63ad3289f78b5fbe1."
    )

    cleanup_errors = []
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        try:
            st = os.stat(max_lim_path)
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to stat {max_lim_path}: {e}. {backport_hint}"
            )

        if not stat.S_ISREG(st.st_mode):
            GLTCore.GLTTestResult.fail(
                f"{max_lim_path} exists but is not a regular file. {backport_hint}"
            )

        if st.st_mode & 0o222:
            GLTCore.GLTTestResult.fail(
                f"{max_lim_path} is writable (mode {oct(st.st_mode)}). "
                f"It must be strictly read-only. {backport_hint}"
            )

        try:
            with open(max_lim_path, "rb") as f:
                content = f.read()
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to read {max_lim_path}: {e}. {backport_hint}"
            )

        try:
            decoded = content.decode("ascii")
        except UnicodeDecodeError:
            GLTCore.GLTTestResult.fail(
                f"{max_lim_path} contains non-ASCII characters: {content!r}. "
                f"{backport_hint}"
            )

        if decoded not in ("0\n", "1\n", "2\n", "3\n"):
            GLTCore.GLTTestResult.fail(
                f"{max_lim_path} content {content!r} is not exactly one ASCII "
                "decimal integer from 0 through 3 followed by a newline. "
                f"{backport_hint}"
            )

        try:
            descriptor = os.open(max_lim_path, os.O_WRONLY)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                GLTCore.GLTTestResult.fail(
                    f"Failed to check write protection for {max_lim_path}: "
                    f"{error}. {backport_hint}")
        else:
            os.close(descriptor)
            GLTCore.GLTTestResult.fail(
                f"{max_lim_path} accepted write access. "
                f"It must be strictly read-only. {backport_hint}")

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_mount(_, logger):
    """
    Verify the ARM64 MPAM resctrl filesystem can be mounted.

    Skips when MPAM/resctrl support is not enabled in the kernel; fails when
    the feature is enabled but the resctrl filesystem cannot be mounted.

    JIRA: DGX-16759
    """
    mpam_config = "CONFIG_ARM64_MPAM"
    resctrl_fs_config = "CONFIG_RESCTRL_FS"

    # Skip gate: kernel configuration must enable MPAM and the resctrl fs.
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"Kernel config not found at {kconfig_path}; "
            f"cannot verify {mpam_config} and {resctrl_fs_config}."
        )

    mpam = kconfig.get(mpam_config)
    if mpam != "y":
        GLTCore.GLTTestResult.skip(
            f"{mpam_config} is not enabled (found {mpam!r}); MPAM unavailable."
        )

    resctrl_fs = kconfig.get(resctrl_fs_config)
    if resctrl_fs != "y":
        GLTCore.GLTTestResult.skip(
            f"{resctrl_fs_config} is not enabled (found {resctrl_fs!r}); "
            "resctrl filesystem unavailable."
        )

    try:
        fs_registered = resctrl_utils.fs_supported()
    except OSError as e:
        GLTCore.GLTTestResult.fail(
            f"Cannot read {resctrl_utils.PROC_FILESYSTEMS} to check for the "
            f"'{resctrl_utils.RESCTRL_FS_TYPE}' filesystem: {e}")
    if not fs_registered:
        GLTCore.GLTTestResult.fail(
            f"'{resctrl_utils.RESCTRL_FS_TYPE}' filesystem not registered in {resctrl_utils.PROC_FILESYSTEMS}."
        )

    with resctrl_utils.managed_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")


# Grace does not run the Vera-specific driver-init register test.
if not NvDev().is_cpu_type("grace"):
    test_mpam_resctrl_mount = GLTCore.GLTRequires.must_pass(
        "arm64", "test_mpam_driver_init_registers"
    )(test_mpam_resctrl_mount)


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_info(_, logger):  # pylint: disable=too-many-branches
    """
    Verify the ARM64 MPAM resctrl info directory is present and accessible.

    JIRA: DGX-16760
    """
    info_dir = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info")

    with resctrl_utils.managed_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        if not os.path.isdir(info_dir):
            GLTCore.GLTTestResult.fail(
                f"resctrl info directory not found at {info_dir}; "
                "MPAM resctrl info interface unavailable."
            )

        platform_profile = ""
        nvdev = NvDev()
        for profile in MPAM_PLATFORM_CAPABILITIES:
            if nvdev.is_cpu_type(profile):
                platform_profile = profile
                break
        if not platform_profile:
            GLTCore.GLTTestResult.skip(
                "This is not an NVIDIA Grace or Vera CPU."
            )
        platform_capabilities = MPAM_PLATFORM_CAPABILITIES[platform_profile]

        errors = []
        for name in platform_capabilities["expected_dirs"]:
            path = os.path.join(info_dir, name)
            if not os.path.exists(path):
                errors.append(f"{name} (expected directory) is missing")
            elif not os.path.isdir(path):
                errors.append(f"{name} exists but is not a directory")

        for name in platform_capabilities["expected_files"]:
            path = os.path.join(info_dir, name)
            if not os.path.exists(path):
                errors.append(f"{name} (expected file) is missing")
            elif not os.path.isfile(path):
                errors.append(f"{name} exists but is not a file")

        for name in platform_capabilities["optional_dirs"]:
            path = os.path.join(info_dir, name)
            if os.path.exists(path) and not os.path.isdir(path):
                errors.append(f"{name} exists but is not a directory")

        if errors:
            GLTCore.GLTTestResult.fail(
                f"resctrl info directory {info_dir} is missing expected "
                f"entries: {'; '.join(errors)}."
            )

        expected_mb_mon_features = platform_capabilities["expected_mb_mon_features"]
        if expected_mb_mon_features:
            mon_features = os.path.join(info_dir, "MB_MON", resctrl_utils.MON_FEATURES)
            try:
                with open(mon_features, "r", encoding="utf-8") as f:
                    features = f.read().split()
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Cannot read MB_MON {resctrl_utils.MON_FEATURES} at {mon_features}: {e}."
                )
            missing_features = tuple(
                feature for feature in expected_mb_mon_features
                if feature not in features
            )
            if missing_features:
                GLTCore.GLTTestResult.fail(
                    f"MB_MON {resctrl_utils.MON_FEATURES} is missing expected "
                    f"features: {', '.join(missing_features)}."
                )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_info")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mbm_assignment_mode(_, logger):
    """
    Verify the active MPAM MBM counter assignment mode.

    PATCH: NVIDIA SAUCE db33ffbdf1ce
    """
    mode_path = os.path.join(
        resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB_MON", "mbm_assign_mode")

    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This CPU does not expose the supported MBM assignment mode "
            "interface."
        )

    cleanup_errors = []
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        if not os.path.isfile(mode_path):
            GLTCore.GLTTestResult.fail(
                f"MBM assignment mode file {mode_path} is missing. "
                "Consider backporting commit db33ffbdf1ce "
                '("Fix monitor capability and MBM assign for MB events").'
            )

        try:
            modes = glt_utils.read_sysfs_value(mode_path).split()
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to read MBM assignment mode file {mode_path}: {e}")

        if "[mbm_event]" not in modes:
            GLTCore.GLTTestResult.fail(
                f"MBM event assignment mode is not enabled in {mode_path} "
                f"(read {modes!r}). "
                "Consider backporting commit db33ffbdf1ce "
                '("Fix monitor capability and MBM assign for MB events").'
            )

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_mbm_assignment_mode")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mbm_counter_info(_, logger):  # pylint: disable=too-many-branches
    """
    Verify MPAM MBM total and available counter information.

    PATCH: NVIDIA SAUCE db33ffbdf1ce
    """
    monitor_info = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB_MON")
    counter_paths = {
        "num_mbm_cntrs": os.path.join(monitor_info, "num_mbm_cntrs"),
        "available_mbm_cntrs": os.path.join(
            monitor_info, "available_mbm_cntrs"),
    }

    cleanup_errors = []
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        counts_by_file = {}
        for name, path in counter_paths.items():
            if not os.path.isfile(path):
                GLTCore.GLTTestResult.fail(
                    f"MBM counter information file {path} is missing. "
                    "Consider backporting commit db33ffbdf1ce "
                    '("Fix monitor capability and MBM assign for MB events").'
                )

            try:
                counts = resctrl_utils.read_mbm_counter_counts(path)
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to read MBM counter information file {path}: {e}")
            except ValueError as e:
                GLTCore.GLTTestResult.fail(
                    f"{e}. Consider backporting commit db33ffbdf1ce "
                    '("Fix monitor capability and MBM assign for MB events").'
                )

            if name == "num_mbm_cntrs":
                for domain, count in counts.items():
                    if count == 0:
                        GLTCore.GLTTestResult.fail(
                            f"MBM domain {domain} reports 0 total counters "
                            "in num_mbm_cntrs. Consider backporting commit "
                            "db33ffbdf1ce "
                            '("Fix monitor capability and MBM assign for '
                            'MB events").'
                        )
            counts_by_file[name] = counts

        total_counts = counts_by_file["num_mbm_cntrs"]
        available_counts = counts_by_file["available_mbm_cntrs"]
        if total_counts.keys() != available_counts.keys():
            GLTCore.GLTTestResult.fail(
                f"MBM counter domain mismatch: num_mbm_cntrs has "
                f"{list(total_counts)}, available_mbm_cntrs has "
                f"{list(available_counts)}. "
                "Consider backporting commit db33ffbdf1ce "
                '("Fix monitor capability and MBM assign for MB events").'
            )

        for domain, total in total_counts.items():
            available = available_counts[domain]
            if available > total:
                GLTCore.GLTTestResult.fail(
                    f"MBM domain {domain} reports {available} available "
                    f"counters but only {total} total counters. "
                    "Consider backporting commit db33ffbdf1ce "
                    '("Fix monitor capability and MBM assign for MB events").'
                )

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_mbm_counter_info")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mbm_assignments(_, logger):  # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks,too-many-statements
    """
    Verify MPAM MBM assignments in a temporary monitor group.

    JIRA: DGX-17560
    PATCH: NVIDIA SAUCE db33ffbdf1ce
    """
    mon_groups_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "mon_groups")
    group_path = os.path.join(
        mon_groups_path,
        f"glt_mbm_assignments_{os.getpid()}_{time.time_ns()}",
    )
    assignment_path = os.path.join(group_path, "mbm_MB_assignments")
    available_path = os.path.join(
        resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB_MON", "available_mbm_cntrs")

    cleanup_errors = []
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        if not os.path.isdir(mon_groups_path):
            GLTCore.GLTTestResult.fail(
                f"resctrl monitor group directory {mon_groups_path} is "
                "missing."
            )

        try:
            resctrl_utils.make_mon_group(group_path, logger=logger)
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to create temporary resctrl monitor group "
                f"{group_path}: {e}"
            )

        try:
            if not os.path.isfile(assignment_path):
                GLTCore.GLTTestResult.fail(
                    f"MBM assignment file {assignment_path} is missing. "
                    "Consider backporting commit db33ffbdf1ce "
                    '("Fix monitor capability and MBM assign for MB events").'
                )

            try:
                original_states = resctrl_utils.read_mbm_assignment_states(assignment_path)
                available_counts = resctrl_utils.read_mbm_counter_counts(available_path)
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to read valid MBM assignment information: {e}"
                )

            safe_domains = [
                domain for domain, state in original_states.items()
                if state == "e" or (
                    state == "_" and available_counts.get(domain, 0) > 0)
            ]
            if not safe_domains:
                GLTCore.GLTTestResult.skip(
                    "No MBM counter is available for a safe assignment "
                    "test."
                )

            for selected_domain in safe_domains:
                counter_path = os.path.join(
                    group_path, "mon_data",
                    f"mon_MB_{selected_domain:02d}",
                    resctrl_utils.MBM_TOTAL_BYTES,
                )
                if not os.path.isfile(counter_path):
                    GLTCore.GLTTestResult.fail(
                        f"MBM monitor file {counter_path} is missing. "
                        "Consider backporting commit db33ffbdf1ce "
                        '("Fix monitor capability and MBM assign for '
                        'MB events").'
                    )

                original_state = original_states[selected_domain]
                transitions = (
                    ("_", "e") if original_state == "e" else ("e", "_")
                )

                for expected_state in transitions:
                    expected_states = original_states.copy()
                    expected_states[selected_domain] = expected_state

                    try:
                        with open(assignment_path, "w", encoding="utf-8") as f:
                            f.write(
                                f"{resctrl_utils.MBM_TOTAL_BYTES}:"
                                f"{selected_domain}={expected_state}\n"
                            )
                    except OSError as e:
                        GLTCore.GLTTestResult.fail(
                            f"Failed to set {resctrl_utils.MBM_TOTAL_BYTES} domain "
                            f"{selected_domain} to {expected_state!r}: {e}"
                        )

                    try:
                        states = resctrl_utils.read_mbm_assignment_states(assignment_path)
                    except (OSError, ValueError) as e:
                        GLTCore.GLTTestResult.fail(
                            f"Failed to read back MBM assignments from "
                            f"{assignment_path}: {e}"
                        )

                    if states != expected_states:
                        GLTCore.GLTTestResult.fail(
                            f"MBM assignment readback mismatch after "
                            f"setting {resctrl_utils.MBM_TOTAL_BYTES} domain "
                            f"{selected_domain} to {expected_state!r}: "
                            f"expected {expected_states}, read {states}."
                        )

                    try:
                        with open(counter_path, "r", encoding="utf-8") as f:
                            counter_value = f.read().strip()
                    except OSError as e:
                        GLTCore.GLTTestResult.fail(
                            f"Failed to read MBM monitor file "
                            f"{counter_path}: {e}"
                        )

                    if expected_state == "e" and not counter_value.isdigit():
                        GLTCore.GLTTestResult.fail(
                            f"Assigned MBM monitor file {counter_path} "
                            f"did not contain a non-negative integer "
                            f"(read {counter_value!r})."
                        )
                    if (expected_state == "_"
                            and counter_value != "Unassigned"):
                        GLTCore.GLTTestResult.fail(
                            f"Unassigned MBM monitor file {counter_path} "
                            f"did not report 'Unassigned' "
                            f"(read {counter_value!r})."
                        )
        finally:
            try:
                os.rmdir(group_path)
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to remove temporary resctrl monitor group "
                    f"{group_path}: {e}"
                )

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_mbm_assignments")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mbm_counter_growth(  # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks,too-many-statements
        _, logger):
    """
    Verify that MPAM MBM bytes grow under NUMA-bound memory traffic.

    Require nondecreasing numeric samples and growth above the baseline.
    This checks counter liveness, not byte-accurate scaling, NRDY recovery,
    overflow handling, or completeness of a kernel patch series.
    The target NVIDIA kernel's MB_MON assignment domains use NUMA node
    IDs, so no separate MB control domain-type metadata is required.

    JIRA: DGX-17570
    """
    workload_duration_sec = 10
    sample_interval_sec = 1
    baseline_timeout_sec = 5
    stress_workers = 4
    stress_l3_size_mib = 160
    stress_l3_size = f"{stress_l3_size_mib}M"
    # Released stress-ng versions allocate three arrays, each four times
    # the configured L3 size divided among workers. Allow 128 MiB overhead.
    min_free_mem_kib = (12 * stress_l3_size_mib + 128) * 1024

    mon_groups_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "mon_groups")
    group_path = os.path.join(
        mon_groups_path,
        f"glt_mbm_growth_{os.getpid()}_{time.time_ns()}",
    )
    assignment_path = os.path.join(group_path, "mbm_MB_assignments")
    available_path = os.path.join(
        resctrl_utils.RESCTRL_MOUNT_POINT, "info", "MB_MON", "available_mbm_cntrs")
    root_tasks_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "tasks")
    group_tasks_path = os.path.join(group_path, "tasks")

    cleanup_errors = []
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        for cli in ("stress-ng", "numactl"):
            if not glt_utils.is_cli_available(cli):
                GLTCore.GLTTestResult.skip(
                    f"'{cli}' CLI not available. Cannot drive MBM traffic.")

        if not os.path.isdir(mon_groups_path):
            GLTCore.GLTTestResult.fail(
                f"resctrl monitor group directory {mon_groups_path} is "
                "missing."
            )
        try:
            resctrl_utils.make_mon_group(group_path, logger=logger)
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to create temporary resctrl monitor group "
                f"{group_path}: {e}"
            )

        proc = None
        pgid = None
        try:
            if not os.path.isfile(assignment_path):
                GLTCore.GLTTestResult.fail(
                    f"MBM assignment file {assignment_path} is missing.")
            if not os.path.isfile(available_path):
                GLTCore.GLTTestResult.fail(
                    f"MBM counter information file {available_path} is "
                    "missing."
                )

            try:
                original_states = resctrl_utils.read_mbm_assignment_states(assignment_path)
                available_counts = resctrl_utils.read_mbm_counter_counts(available_path)
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to read valid MBM assignment information: {e}"
                )

            assignment_domains = original_states.keys()
            available_counter_domains = available_counts.keys()
            if assignment_domains != available_counter_domains:
                GLTCore.GLTTestResult.fail(
                    f"MBM assignment domain mismatch: {assignment_path} has "
                    f"{list(assignment_domains)}, {available_path} has "
                    f"{list(available_counter_domains)}."
                )

            candidate_domains = [
                domain for domain, state in original_states.items()
                if state == "e"
            ]
            candidate_domains.extend(
                domain for domain, state in original_states.items()
                if state == "_" and available_counts[domain] > 0
            )

            # On the target NVIDIA kernel, enabled MB assignments require
            # the MBA resource and memory MBM monitor to share a class.
            # That class's component IDs are NUMA node IDs, independently
            # of the optional MB control domain_id metadata.
            selected_domain = None
            for domain in candidate_domains:
                meminfo_path = (
                    f"/sys/devices/system/node/node{domain}/meminfo")
                try:
                    with open(meminfo_path, "r", encoding="utf-8") as meminfo:
                        mem_total_kib = None
                        mem_free_kib = None
                        for line in meminfo:
                            fields = line.split()
                            if len(fields) >= 4 and fields[2] == "MemTotal:":
                                mem_total_kib = int(fields[3])
                            elif len(fields) >= 4 and fields[2] == "MemFree:":
                                mem_free_kib = int(fields[3])
                except (OSError, ValueError) as e:
                    GLTCore.GLTTestResult.fail(
                        f"Failed to read NUMA node memory size from "
                        f"{meminfo_path}: {e}"
                    )

                if mem_total_kib is None or mem_free_kib is None:
                    GLTCore.GLTTestResult.fail(
                        f"NUMA node meminfo {meminfo_path} does not "
                        "report MemTotal and MemFree."
                    )
                cpulist_path = (
                    f"/sys/devices/system/node/node{domain}/cpulist")
                try:
                    domain_cpus = _cpus_of_node(domain)
                except (OSError, ValueError) as e:
                    GLTCore.GLTTestResult.fail(
                        f"Failed to read NUMA node CPU list from "
                        f"{cpulist_path}: {e}"
                    )
                if (mem_total_kib > 0 and domain_cpus
                        and mem_free_kib >= min_free_mem_kib):
                    selected_domain = domain
                    break
                if mem_free_kib < min_free_mem_kib:
                    logger.info(
                        "Skipping MBM domain %d: %d KiB free, need %d KiB.",
                        domain, mem_free_kib, min_free_mem_kib)

            if selected_domain is None:
                GLTCore.GLTTestResult.skip(
                    "No MBM domain with online CPUs and at least "
                    f"{min_free_mem_kib // 1024} MiB of free memory is "
                    "available for a bound traffic test."
                )

            counter_path = os.path.join(
                group_path, "mon_data",
                f"mon_MB_{selected_domain:02d}",
                resctrl_utils.MBM_TOTAL_BYTES,
            )
            if not os.path.isfile(counter_path):
                GLTCore.GLTTestResult.fail(
                    f"MBM monitor file {counter_path} is missing.")

            if original_states[selected_domain] == "_":
                try:
                    with open(assignment_path, "w", encoding="utf-8") as f:
                        f.write(
                            f"{resctrl_utils.MBM_TOTAL_BYTES}:"
                            f"{selected_domain}=e\n"
                        )
                except OSError as e:
                    GLTCore.GLTTestResult.fail(
                        f"Failed to assign {resctrl_utils.MBM_TOTAL_BYTES} domain "
                        f"{selected_domain}: {e}"
                    )

                try:
                    assigned_states = resctrl_utils.read_mbm_assignment_states(
                        assignment_path)
                except (OSError, ValueError) as e:
                    GLTCore.GLTTestResult.fail(
                        f"Failed to read back MBM assignments from "
                        f"{assignment_path}: {e}"
                    )
                expected_states = original_states.copy()
                expected_states[selected_domain] = "e"
                if assigned_states != expected_states:
                    GLTCore.GLTTestResult.fail(
                        f"MBM assignment readback mismatch: expected "
                        f"{expected_states}, read {assigned_states}."
                    )

            baseline = None
            counter_value = ""
            baseline_deadline = time.monotonic() + baseline_timeout_sec
            while time.monotonic() < baseline_deadline:
                try:
                    counter_value = glt_utils.read_sysfs_value(counter_path)
                except OSError as e:
                    GLTCore.GLTTestResult.fail(
                        f"Failed to read MBM monitor file "
                        f"{counter_path}: {e}"
                    )

                if counter_value.isdigit():
                    baseline = int(counter_value)
                    break
                if counter_value != "Unavailable":
                    GLTCore.GLTTestResult.fail(
                        f"Assigned MBM monitor file {counter_path} did not "
                        f"contain a non-negative integer "
                        f"(read {counter_value!r})."
                    )
                time.sleep(0.5)

            if baseline is None:
                GLTCore.GLTTestResult.fail(
                    f"MBM monitor file {counter_path} did not become numeric "
                    f"within {baseline_timeout_sec} seconds "
                    f"(last read {counter_value!r})."
                )

            # A root MON group only accepts tasks from the root CTRL group.
            # Move the child before exec so every stress-ng worker inherits
            # the selected PARTID/PMG without moving the GLT runner itself.
            workload_wrapper = (
                'printf "%s\\n" "$$" > "$1" && '
                'printf "%s\\n" "$$" > "$2" && '
                'exec numactl "--cpunodebind=$3" "--membind=$4" '
                'stress-ng --stream "$5" --stream-l3-size "$6" '
                '--stream-madvise hugepage --timeout "$7"'
            )
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    "/bin/sh", "-c", workload_wrapper, "glt-mbm-growth",
                    root_tasks_path, group_tasks_path, str(selected_domain),
                    str(selected_domain), str(stress_workers),
                    stress_l3_size, f"{workload_duration_sec}s",
                ],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pgid = proc.pid

            samples = [baseline]
            for _sample in range(workload_duration_sec + 1):
                time.sleep(sample_interval_sec)
                try:
                    counter_value = glt_utils.read_sysfs_value(counter_path)
                except OSError as e:
                    GLTCore.GLTTestResult.fail(
                        f"Failed to read MBM monitor file "
                        f"{counter_path}: {e}"
                    )

                if counter_value.isdigit():
                    current = int(counter_value)
                    if current < samples[-1]:
                        GLTCore.GLTTestResult.fail(
                            f"{resctrl_utils.MBM_TOTAL_BYTES} decreased under traffic: "
                            f"{samples[-1]} -> {current}."
                        )
                    samples.append(current)
                elif counter_value != "Unavailable":
                    GLTCore.GLTTestResult.fail(
                        f"Assigned MBM monitor file {counter_path} became "
                        f"unreadable under traffic (read {counter_value!r})."
                    )

                if proc.poll() is not None:
                    break

            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                GLTCore.GLTTestResult.fail(
                    "stress-ng workload did not exit within the expected "
                    "duration."
                )
            # stress-ng distinguishes resource/implementation limitations
            # from workload failures (status 2).
            if returncode in (3, 4):
                GLTCore.GLTTestResult.skip(
                    "stress-ng could not run the memory traffic workload "
                    f"due to unavailable resources or support (status {returncode})."
                )
            if returncode != 0:
                GLTCore.GLTTestResult.fail(
                    f"stress-ng workload exited with status {returncode}.")

            if len(samples) < 2:
                GLTCore.GLTTestResult.fail(
                    f"{resctrl_utils.MBM_TOTAL_BYTES} produced no numeric sample "
                    "during memory traffic."
                )
            if samples[-1] <= baseline:
                GLTCore.GLTTestResult.fail(
                    f"{resctrl_utils.MBM_TOTAL_BYTES} did not grow under memory "
                    f"traffic ({baseline} -> {samples[-1]} across "
                    f"{len(samples)} samples)."
                )

            logger.info(
                "%s grew under memory traffic on MB domain %d: %d -> %d "
                "across %d samples.",
                resctrl_utils.MBM_TOTAL_BYTES, selected_domain, baseline,
                samples[-1], len(samples),
            )
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        message = (
                            "Failed to stop MBM memory traffic process group "
                            f"{pgid}."
                        )
                        logger.warning(message)
                        cleanup_errors.append(message)

            if os.path.isfile(assignment_path):
                try:
                    with open(assignment_path, "w", encoding="utf-8") as f:
                        f.write(f"{resctrl_utils.MBM_TOTAL_BYTES}:*=_\n")
                except OSError as e:
                    logger.warning(
                        "Failed to release MBM counters from %s: %s",
                        assignment_path, e)

            try:
                os.rmdir(group_path)
            except OSError as e:
                message = (
                    f"Failed to remove temporary resctrl monitor group "
                    f"{group_path}: {e}"
                )
                logger.warning(message)
                cleanup_errors.append(message)

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_schemata(_, logger):
    """
    Verify the ARM64 MPAM resctrl schemata interface is readable.

    JIRA: DGX-16761
    """
    schemata_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "schemata")
    schemata_line_re = re.compile(r"^\s*\S+:\d+=\S+")

    with resctrl_utils.managed_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        if not os.path.isfile(schemata_path):
            GLTCore.GLTTestResult.fail(
                f"resctrl schemata {schemata_path} does not exist or is not "
                "a file after mount."
            )

        try:
            with open(schemata_path, "r", encoding="utf-8") as f:
                content = f.read()

                if not content.strip():
                    GLTCore.GLTTestResult.fail(
                        f"resctrl schemata {schemata_path} is empty."
                    )

                if not any(schemata_line_re.match(line) for line in content.splitlines()):
                    GLTCore.GLTTestResult.fail(
                        f"resctrl schemata {schemata_path} contains no "
                        "well-formed resource control line."
                    )

        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to read resctrl schemata {schemata_path}: {e}"
            )


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
def test_mpam_cpuless_numa_domains(_, logger):  # pylint: disable=too-many-branches
    """
    Verify CPU-less memory MSCs receive a usable CPU affinity.

    Requires mounted debugfs with MPAM component information. ACPI memory
    MSCs use class.255 and Linux NUMA node IDs as component IDs. Check the
    RIS affinity directly, independently of resctrl control names or
    schemata domain IDs. Skip when no memory MSC belongs to a CPU-less node.
    Fail when the advertised debugfs topology cannot be enumerated.

    PATCH: 24ecc38401ef NVIDIA: VR: SAUCE: arm_mpam: Handle CPU-less numa nodes
    JIRA: DGX-17718
    """
    memory_class_path = os.path.join(MPAM_DEBUGFS_PATH, "class.255")
    try:
        component_paths = _mpam_debugfs_subdirs(memory_class_path, "comp.*")
    except FileNotFoundError:
        # A missing path is confirmed absence: either MPAM debugfs is
        # unavailable or the kernel advertises no memory MSC class.
        if not os.path.isdir(MPAM_DEBUGFS_PATH):
            GLTCore.GLTTestResult.skip(
                f"MPAM debug information is unavailable at {MPAM_DEBUGFS_PATH}; "
                "requires CONFIG_DEBUG_FS and mounted debugfs."
            )
        component_paths = []
    except OSError as error:
        GLTCore.GLTTestResult.fail(
            f"Failed to inspect MPAM memory components at "
            f"{memory_class_path}: {error}"
        )
    checked_components = 0
    for component_path in component_paths:
        node = os.path.basename(component_path).removeprefix("comp.")
        if not node.isdigit():
            GLTCore.GLTTestResult.fail(
                f"Invalid MPAM memory component path: {component_path}")
        cpulist_path = f"/sys/devices/system/node/node{node}/cpulist"
        try:
            cpulist = glt_utils.read_sysfs_value(cpulist_path)
        except OSError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to read CPU list for MPAM memory node {node} "
                f"from {cpulist_path}: {error}"
            )
        if cpulist:
            continue

        try:
            ris_paths = [
                ris_path
                for vmsc_path in _mpam_debugfs_subdirs(component_path, "vmsc.*")
                for ris_path in _mpam_debugfs_subdirs(vmsc_path, "msc.*_ris.*")
            ]
        except OSError as error:
            GLTCore.GLTTestResult.fail(
                f"Failed to inspect MPAM RIS information for CPU-less NUMA "
                f"node {node} at {component_path}: {error}"
            )
        if not ris_paths:
            GLTCore.GLTTestResult.fail(
                f"No MPAM RIS affinity information for CPU-less NUMA "
                f"node {node} at {component_path}."
            )

        for ris_path in ris_paths:
            affinity_path = os.path.join(ris_path, "affinity")
            try:
                affinity = glt_utils.read_sysfs_value(affinity_path)
            except OSError as error:
                GLTCore.GLTTestResult.fail(
                    f"Failed to read MPAM RIS affinity for CPU-less NUMA "
                    f"node {node} from {affinity_path}: {error}"
                )
            # debugfs_create_cpumask() exposes a hexadecimal bitmap (%*pb),
            # unlike the decimal CPU lists in NUMA sysfs.
            if not re.fullmatch(r"[0-9a-fA-F]+(?:,[0-9a-fA-F]+)*", affinity):
                GLTCore.GLTTestResult.fail(
                    f"Malformed MPAM RIS affinity {affinity!r} "
                    f"in {affinity_path}."
                )
            if not int(affinity.replace(",", ""), 16):
                GLTCore.GLTTestResult.fail(
                    f"MPAM RIS {affinity_path} has empty CPU affinity for "
                    f"CPU-less NUMA node {node}; check the CPU-less NUMA "
                    "affinity fix 24ecc38401ef."
                )
            logger.info("CPU-less NUMA node %s: MPAM RIS affinity %s = %s",
                        node, affinity_path, affinity)
        checked_components += 1

    if not checked_components:
        GLTCore.GLTTestResult.skip(
            "No MPAM memory MSC belongs to a CPU-less NUMA node."
        )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_l2_max_schemata(_, logger):  # pylint: disable=too-many-branches
    """
    Verify the Vera MPAM L2_MAX percentage schemata write/readback path.

    JIRA: DGX-17928
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not an NVIDIA Vera CPU; test is Vera-specific.")

    cleanup_errors = []
    status, reason = "ok", ""
    grid = None
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            status, reason = "fail", "Failed to mount resctrl."
        else:
            try:
                info = _mpam_read_percentage_info("L2_MAX")
                if info is None:
                    status = "not_exposed"
                    reason = "info/L2_MAX is absent; L2_MAX resource directory is not exposed."
                else:
                    reason = _mpam_percentage_policy_reason("L2_MAX", *info)
                    if reason:
                        status = "blocked"
                    else:
                        grid = list(range(info[0], MPAM_PERCENT_MAX + 1, info[1]))
            except (ValueError, OSError) as error:
                status, reason = "fail", str(error)
        if status == "ok":
            try:
                suffix = uuid.uuid4().hex[:8]
                group = os.path.join(
                    resctrl_utils.RESCTRL_MOUNT_POINT,
                    f"glt_l2_max_{os.getpid()}_{suffix}")
                try:
                    os.mkdir(group)
                except FileExistsError:
                    raise ValueError(
                        f"resctrl group {group} already exists; refusing to "
                        "remove a group not created by this test invocation.") from None
                schemata_path = os.path.join(group, "schemata")
                initial_values = None
                try:
                    schemas = resctrl_utils.read_schemata(schemata_path, ("L2_MAX",))
                    if "L2_MAX" not in schemas:
                        raise ValueError(f"L2_MAX resource line is missing from {schemata_path}.")
                    initial_values = schemas["L2_MAX"]
                    domains = tuple(initial_values)
                    pair = _mpam_probe_percentage_pair(
                        schemata_path, "L2_MAX", domains, initial_values, grid)
                    if pair is None:
                        status = "blocked"
                        reason = MPAM_PERCENTAGE_NO_PAIR_REASON.format(
                            resource="L2_MAX", grid=grid)
                    else:
                        low, high = pair
                        logger.info("L2_MAX selected low=%s, high=%s.", low, high)
                finally:
                    try:
                        if initial_values is not None:
                            _mpam_restore_percentage_schemata(
                                schemata_path, "L2_MAX", domains, initial_values)
                    except (ValueError, OSError) as error:
                        cleanup_errors.append(str(error))
                    finally:
                        try:
                            os.rmdir(group)
                        except OSError as error:
                            cleanup_errors.append(f"Failed to remove resctrl group {group}: {error}")
            except (ValueError, OSError) as error:
                status, reason = "fail", str(error)
    if cleanup_errors:
        GLTCore.GLTTestResult.fail(_mpam_combined_failure(reason, cleanup_errors))
    if status == "not_exposed":
        GLTCore.GLTTestResult.skip(
            "L2_MAX is not exposed on this Vera boot after resctrl "
            f"prerequisites passed because {reason} L2_MAX is "
            "capability-conditional, so no L2_MAX assertion was run."
        )
    if status == "blocked":
        GLTCore.GLTTestResult.warn(
            "L2_MAX percentage schemata could not be validated: "
            f"{reason}.")
    if status == "fail":
        GLTCore.GLTTestResult.fail(reason)


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_l3_max_schemata(_, logger):  # pylint: disable=too-many-branches
    """
    Verify the Vera MPAM L3_MAX percentage schemata write/readback path.

    JIRA: DGX-17928
    """
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not an NVIDIA Vera CPU; test is Vera-specific.")

    cleanup_errors = []
    status, reason = "ok", ""
    grid = None
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            status, reason = "fail", "Failed to mount resctrl."
        else:
            try:
                info = _mpam_read_percentage_info("L3_MAX")
                if info is None:
                    status = "not_exposed"
                    reason = "info/L3_MAX is absent; L3_MAX resource directory is not exposed."
                else:
                    reason = _mpam_percentage_policy_reason("L3_MAX", *info)
                    if reason:
                        status = "blocked"
                    else:
                        grid = list(range(info[0], MPAM_PERCENT_MAX + 1, info[1]))
            except (ValueError, OSError) as error:
                status, reason = "fail", str(error)
        if status == "ok":
            try:
                suffix = uuid.uuid4().hex[:8]
                group = os.path.join(
                    resctrl_utils.RESCTRL_MOUNT_POINT,
                    f"glt_l3_max_{os.getpid()}_{suffix}")
                try:
                    os.mkdir(group)
                except FileExistsError:
                    raise ValueError(
                        f"resctrl group {group} already exists; refusing to "
                        "remove a group not created by this test invocation.") from None
                schemata_path = os.path.join(group, "schemata")
                initial_values = None
                try:
                    schemas = resctrl_utils.read_schemata(schemata_path, ("L3_MAX",))
                    if "L3_MAX" not in schemas:
                        raise ValueError(f"L3_MAX resource line is missing from {schemata_path}.")
                    initial_values = schemas["L3_MAX"]
                    domains = tuple(initial_values)
                    pair = _mpam_probe_percentage_pair(
                        schemata_path, "L3_MAX", domains, initial_values, grid)
                    if pair is None:
                        status = "blocked"
                        reason = MPAM_PERCENTAGE_NO_PAIR_REASON.format(
                            resource="L3_MAX", grid=grid)
                    else:
                        low, high = pair
                        logger.info("L3_MAX selected low=%s, high=%s.", low, high)
                finally:
                    try:
                        if initial_values is not None:
                            _mpam_restore_percentage_schemata(
                                schemata_path, "L3_MAX", domains, initial_values)
                    except (ValueError, OSError) as error:
                        cleanup_errors.append(str(error))
                    finally:
                        try:
                            os.rmdir(group)
                        except OSError as error:
                            cleanup_errors.append(f"Failed to remove resctrl group {group}: {error}")
            except (ValueError, OSError) as error:
                status, reason = "fail", str(error)
    if cleanup_errors:
        GLTCore.GLTTestResult.fail(_mpam_combined_failure(reason, cleanup_errors))
    if status == "not_exposed":
        GLTCore.GLTTestResult.fail(
            "L3_MAX is not exposed on this Vera boot after resctrl "
            f"prerequisites passed because {reason}; no L3_MAX assertion "
            "was run."
        )
    if status == "blocked":
        GLTCore.GLTTestResult.warn(
            "L3_MAX percentage schemata could not be validated: "
            f"{reason}.")
    if status == "fail":
        GLTCore.GLTTestResult.fail(reason)


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("stress")
def test_mpam_l3_max_live_effect(_, logger):
    # pylint: disable=too-many-branches,too-many-nested-blocks
    """Validate DGX-17928 L3_MAX live effect with stress-ng llc_occupancy."""
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip(
            "This is not an NVIDIA Vera CPU; test is Vera-specific.")
    for cli in ("numactl", "stress-ng"):
        if not glt_utils.is_cli_available(cli):
            GLTCore.GLTTestResult.skip(
                f"'{cli}' CLI not available; cannot run L3_MAX live-effect.")
    try:
        _mpam_l3_max_require_stress_version()
    except _MpamBlocked as error:
        GLTCore.GLTTestResult.skip(str(error))
    try:
        _mpam_l3_max_preflight_supervisor_children()
    except _MpamBlocked as error:
        GLTCore.GLTTestResult.warn(str(error))

    cleanup_errors = []
    failure_reason = None
    blocked_reason = None
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            failure_reason = "Failed to mount resctrl."
        else:
            try:
                info = _mpam_read_percentage_info("L3_MAX")
                if info is None:
                    raise ValueError("info/L3_MAX is absent; L3_MAX resource directory is not exposed.")
                blocked_reason = _mpam_percentage_policy_reason("L3_MAX", *info)
                if not blocked_reason:
                    grid = list(range(info[0], MPAM_PERCENT_MAX + 1, info[1]))
                    suffix = uuid.uuid4().hex[:8]
                    group = os.path.join(
                        resctrl_utils.RESCTRL_MOUNT_POINT,
                        f"glt_l3_max_{os.getpid()}_{suffix}")
                    try:
                        os.mkdir(group)
                    except FileExistsError:
                        raise ValueError(
                            f"resctrl group {group} already exists; refusing to "
                            "remove a group not created by this test invocation.") from None
                    schemata_path = os.path.join(group, "schemata")
                    initial_values = None
                    try:
                        schemas = resctrl_utils.read_schemata(schemata_path, ("L3_MAX",))
                        if "L3_MAX" not in schemas:
                            raise ValueError(f"L3_MAX resource line is missing from {schemata_path}.")
                        initial_values = schemas["L3_MAX"]
                        domains = tuple(initial_values)
                        pair = _mpam_probe_percentage_pair(
                            schemata_path, "L3_MAX", domains, initial_values, grid)
                        if pair is None:
                            blocked_reason = MPAM_PERCENTAGE_NO_PAIR_REASON.format(
                                resource="L3_MAX", grid=grid)
                        else:
                            low, high = pair
                            if high - low < MPAM_L3_MAX_LIVE_MIN_SPAN:
                                raise _MpamBlocked(
                                    f"L3_MAX selected percentage span {high - low} requires "
                                    f"at least {MPAM_L3_MAX_LIVE_MIN_SPAN} percentage points.")
                            placement = _mpam_l3_max_attribution(group)
                            logger.info("L3_MAX placement node=%s cpus=%s cache IDs=%s.",
                                        placement['node'], placement['cpus'], placement['cache_ids'])
                    finally:
                        try:
                            if initial_values is not None:
                                _mpam_restore_percentage_schemata(
                                    schemata_path, "L3_MAX", domains, initial_values)
                        except (ValueError, OSError) as error:
                            cleanup_errors.append(str(error))
                        finally:
                            try:
                                os.rmdir(group)
                            except OSError as error:
                                cleanup_errors.append(f"Failed to remove resctrl group {group}: {error}")
                    if cleanup_errors and not blocked_reason:
                        raise ValueError("L3_MAX calibration cleanup failed.")
                    if not blocked_reason:
                        windows = []
                        for label in MPAM_L3_MAX_LIVE_PATTERN:
                            percentage = low if label == "low" else high
                            suffix = uuid.uuid4().hex[:8]
                            group = os.path.join(
                                resctrl_utils.RESCTRL_MOUNT_POINT,
                                f"glt_l3_max_{os.getpid()}_{suffix}")
                            try:
                                os.mkdir(group)
                            except FileExistsError:
                                raise ValueError(
                                    f"resctrl group {group} already exists; refusing to "
                                    "remove a group not created by this test invocation.") from None
                            schemata_path = os.path.join(group, "schemata")
                            initial_values = None
                            try:
                                schemas = resctrl_utils.read_schemata(schemata_path, ("L3_MAX",))
                                if "L3_MAX" not in schemas:
                                    raise ValueError(f"L3_MAX resource line is missing from {schemata_path}.")
                                initial_values = schemas["L3_MAX"]
                                domains = tuple(initial_values)
                                write_result = _mpam_write_and_require_percentage(
                                    schemata_path, "L3_MAX", domains, percentage)
                                logger.info("L3_MAX window %s percentage readback: %s", label, write_result)
                                occupancy_files = _mpam_l3_max_occupancy_files(group, placement)
                                prelaunch_occupancy = _mpam_wait_l3_max_clean_baseline(
                                    group, occupancy_files)
                                logger.info(
                                    "L3_MAX window %s prelaunch occupancy=%s.",
                                    label, prelaunch_occupancy)

                                with _mpam_l3_max_workload(
                                        group, placement, cleanup_errors) as (proc, deadline, capture):
                                    logger.info("L3_MAX window %s workload PID=%s.", label, proc.pid)
                                    _mpam_wait_for_l3_max_membership(
                                        group, proc, placement, deadline)
                                    _mpam_sleep_l3_max_with_deadline(
                                        MPAM_L3_MAX_LIVE_WARMUP_SEC, deadline, "warmup")
                                    _mpam_wait_for_l3_max_membership(
                                        group, proc, placement, deadline,
                                        require_worker_footprint=True)
                                    samples = _mpam_collect_l3_max_samples(
                                        proc, deadline, (group, placement, occupancy_files), label, logger)
                                    stress_result = _mpam_wait_for_l3_max_stress_exit(
                                        proc, deadline, _mpam_l3_max_stress_argv(placement['node']), capture)
                                    if _mpam_l3_max_group_exists(proc.pid):
                                        raise ValueError(
                                            f"{_mpam_l3_max_stress_result_message(stress_result)} "
                                            f"Process group {proc.pid} still existed after wait.")
                                    stress_summary = _mpam_validate_l3_max_stress_result(stress_result)
                                    _mpam_require_l3_max_deadline(deadline, "stress-ng exit")
                                    window_median = _median(samples)
                                    deadline_slack = _mpam_wait_remaining(deadline)
                                    if stress_result.stdout:
                                        logger.info(
                                            "L3_MAX live window %s stress-ng output: %s",
                                            label, stress_result.stdout)
                                    logger.info(
                                        "L3_MAX live window %s stress-ng status: passed=%d skipped=%d "
                                        "failed=%d metrics_untrustworthy=%d.",
                                        label, stress_summary['passed'], stress_summary['skipped'],
                                        stress_summary['failed'], stress_summary['metrics_untrustworthy'])
                                    logger.info(
                                        "L3_MAX live window %s deadline slack %.3fs.",
                                        label, deadline_slack)
                                    logger.info(
                                        "L3_MAX window %s median=%s returncode=%s.",
                                        label, window_median, stress_result.returncode)
                                if cleanup_errors:
                                    break
                                windows.append({"label": label, "percentage": percentage,
                                                "samples": tuple(samples), "median": window_median})
                            finally:
                                try:
                                    if initial_values is not None:
                                        _mpam_restore_percentage_schemata(
                                            schemata_path, "L3_MAX", domains, initial_values)
                                except (ValueError, OSError) as error:
                                    cleanup_errors.append(str(error))
                                finally:
                                    try:
                                        os.rmdir(group)
                                    except OSError as error:
                                        cleanup_errors.append(f"Failed to remove resctrl group {group}: {error}")
                            if cleanup_errors:
                                break
                        if not cleanup_errors:
                            low_center, high_center, separation, dispersion = (
                                _mpam_evaluate_l3_max_live_windows(windows))
                            logger.info(
                                "L3_MAX live-effect PASS: low median %.1f, "
                                "high median %.1f, separation %.1f, dispersion %.1f.",
                                low_center, high_center, separation, dispersion)
            except _MpamBlocked as error:
                blocked_reason = str(error)
            except (ValueError, OSError) as error:
                failure_reason = str(error)

    failure_message = _mpam_combined_failure(
        (failure_reason or blocked_reason)
        if cleanup_errors else failure_reason, cleanup_errors)
    if failure_message:
        GLTCore.GLTTestResult.fail(failure_message)
    if blocked_reason:
        GLTCore.GLTTestResult.warn(
            "L3_MAX live-effect validation could not run: "
            f"{blocked_reason}")


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_schemata")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_msc_domain_completeness(_, logger):  # pylint: disable=too-many-branches
    """
    Verify every discovered memory MPAM component has an MB domain.

    Domain setup once kept only the first MSC component matching a CPU,
    leaving the remaining components without a resctrl domain. Compare
    the NUMA-keyed component IDs under MPAM debugfs class.255 with the
    domains the MB control publishes. This inventory is populated before
    resctrl domain creation, so it includes components domain setup lost.
    Skip when fewer than two components exist or their inventory is
    unavailable. A missing MB line is inconclusive: HAS_MBW_PART does
    not prove the class supports the HAS_MAX and BWA_WD requirements for
    MB allocation. Legacy MB domains matching L3 skip when cache backing
    cannot be excluded, even if those IDs also match the memory nodes.
    Native MB_NODE control is node-scoped and has no such ambiguity.

    PATCH: 2eacf36cf734 NVIDIA: VR: SAUCE: arm_mpam: Include all MSC components during domain setup
    JIRA: DGX-17724
    """
    class_path = os.path.join(MPAM_DEBUGFS_PATH, "class.255")
    try:
        component_paths = _mpam_debugfs_subdirs(class_path, "comp.*")
        expected_nodes = set()
        for component_path in component_paths:
            component = os.path.basename(component_path).removeprefix("comp.")
            if not component.isdigit():
                raise ValueError(
                    f"Invalid MPAM component path: {component_path}")
            expected_nodes.add(int(component))
    except FileNotFoundError:
        GLTCore.GLTTestResult.skip(
            f"MPAM memory component inventory at {class_path} is unavailable."
        )
    except (OSError, ValueError) as e:
        GLTCore.GLTTestResult.fail(
            f"Failed to read the MPAM memory component inventory: {e}")

    if len(expected_nodes) < 2:
        GLTCore.GLTTestResult.skip(
            "MPAM debugfs exposes fewer than two memory components "
            f"{sorted(expected_nodes)}; this topology cannot distinguish "
            "domain setup that keeps only the first CPU-matching component."
        )

    cleanup_errors = []
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors
    ) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")

        try:
            mb_resource = resctrl_utils.mb_control_resource()
            mb_domains = resctrl_utils.schemata_domains(mb_resource)
            l3_domains = resctrl_utils.schemata_domains("L3")
        except (OSError, ValueError) as e:
            GLTCore.GLTTestResult.fail(
                f"Failed to read the resctrl schemata domains: {e}")

        if mb_domains is None:
            GLTCore.GLTTestResult.skip(
                f"resctrl publishes no {mb_resource} control. "
                "MPAMF_IDR.HAS_MBW_PART alone does not establish the "
                "MPAMF_MBW_IDR.HAS_MAX and BWA_WD requirements for MB "
                "allocation, so unsupported control and incomplete domain "
                "setup cannot be distinguished."
            )
        if (mb_resource == "MB" and l3_domains is not None
                and mb_domains == l3_domains):
            # A cache class backs MB only if the L3 RIS implement MBW
            # partitioning. Without that, treat equal domain sets as
            # memory-backed and compare them with the expected nodes.
            try:
                l3_mbw_part = _mpam_class_mbw_part_components(3)
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to inspect the L3 MSC capabilities in "
                    f"{MPAM_DEBUGFS_PATH}: {e}")
            l3_can_back_mb = bool(l3_mbw_part) or (
                l3_mbw_part is None and not os.path.isdir(MPAM_DEBUGFS_PATH))
            if l3_can_back_mb:
                GLTCore.GLTTestResult.skip(
                    f"{mb_resource} domains {sorted(mb_domains)} match the L3 "
                    "cache domains and cache backing cannot be excluded, "
                    "so memory MSC domain completeness cannot be determined "
                    f"through {mb_resource}."
                )

        # The driver selects one class using capabilities intersected
        # across all its components; it does not expose only the subset
        # whose individual RIS advertise bandwidth partitioning.
        if mb_domains != expected_nodes:
            GLTCore.GLTTestResult.fail(
                f"{mb_resource} exposes domain(s) {sorted(mb_domains)} "
                f"but MPAM debugfs discovered memory component(s) "
                f"{sorted(expected_nodes)}. "
                "Check kernel commit 2eacf36cf734."
            )

    if cleanup_errors:
        GLTCore.GLTTestResult.fail(cleanup_errors[0])


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_schemata")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_l3_occupancy(_, logger):  # pylint: disable=too-many-branches
    """
    Verify the ARM64 MPAM L3 cache occupancy monitor is exposed and readable.

    JIRA: DGX-16762
    """
    schemata_path = resctrl_utils.RESCTRL_MOUNT_POINT + "/schemata"
    l3_line_re = re.compile(r"^\s*L3:")
    l3_mon_features = resctrl_utils.RESCTRL_MOUNT_POINT + "/info/L3_MON/mon_features"
    l3_monitor_glob = "mon_data/mon_L3_*"

    with resctrl_utils.managed_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        # the L3 resource must be present in the schemata.
        try:
            with open(schemata_path, "r", encoding="utf-8") as f:
                l3_present = any(l3_line_re.match(line) for line in f)
        except OSError:
            l3_present = False
        if not l3_present:
            GLTCore.GLTTestResult.fail(
                "L3 resource not present in resctrl schemata.")

        # L3_MON must advertise the llc_occupancy feature.
        try:
            with open(l3_mon_features, "r", encoding="utf-8") as f:
                occupancy_advertised = any(
                    field == "llc_occupancy"
                    for line in f for field in line.split())
        except OSError:
            occupancy_advertised = False
        if not occupancy_advertised:
            GLTCore.GLTTestResult.fail(
                "L3 occupancy monitoring not advertised.")

        # Every L3 monitoring domain must expose a readable occupancy counter.
        monitor_dirs = [
            path for path in glob.glob(
                os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, l3_monitor_glob))
            if os.path.isdir(path)
        ]
        if not monitor_dirs:
            GLTCore.GLTTestResult.fail(
                "L3 occupancy advertised but no mon_L3 monitor directory found."
            )

        for monitor_dir in monitor_dirs:
            occupancy_file = os.path.join(monitor_dir, "llc_occupancy")
            if not os.path.isfile(occupancy_file):
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy file {occupancy_file} is missing."
                )

            try:
                with open(occupancy_file, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to read L3 occupancy file {occupancy_file}: {e}"
                )

            if not raw.isdigit():
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy file {occupancy_file} did not contain a "
                    f"non-negative integer (read {raw!r})."
                )


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_l3_occupancy")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_l3_occupancy_workload(_, logger):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """
    Verify the ARM64 MPAM L3 cache allocation bounds occupancy under load.

    Restricts a resctrl control group to a partial L3 allocation (CBM) and
    drives a cache-filling stress-ng workload, verifying that the group's
    peak llc_occupancy stays bounded by the allocated capacity and does not
    leak past the CBM.

    JIRA: DGX-16956
    """
    workload_duration_sec = 10
    sample_interval_sec = 1
    tolerance = 0.25
    alloc_fraction = 4
    stress_workers = 4
    stress_vm_bytes = "256M"

    with resctrl_utils.managed_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        for cli in ("stress-ng", "numactl"):
            if not glt_utils.is_cli_available(cli):
                GLTCore.GLTTestResult.skip(
                    f"'{cli}' CLI not available, cannot drive L3 occupancy "
                    "workload.")

        # Pick the first NUMA node that actually owns CPUs (default node 0).
        node = 0
        for cpulist in sorted(glob.glob(
                "/sys/devices/system/node/node*/cpulist")):
            try:
                with open(cpulist, "r", encoding="utf-8") as f:
                    if f.read().strip():
                        match = re.search(r"node(\d+)", cpulist)
                        if match is None:
                            continue
                        node = int(match.group(1))
                        break
            except OSError:
                continue

        group = resctrl_utils.RESCTRL_MOUNT_POINT + "/glt_l3_occ_" + str(os.getpid())
        try:
            os.mkdir(group)
        except FileExistsError:
            # Stale own-named group from a prior killed run / PID reuse.
            # Removing it auto-reparents any stray tasks; retry once.
            try:
                os.rmdir(group)
                os.mkdir(group)
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not create a clean resctrl group {group}: {e}"
                )
        except OSError as e:
            GLTCore.GLTTestResult.fail(
                f"Could not create resctrl group {group}: {e}"
            )

        time.sleep(1)
        proc = None
        pgid = None
        try:
            node_l3_ids = set()
            node_l3_size = None
            try:
                node_cpus = _cpus_of_node(node)
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Failed to read NUMA node CPU list from "
                    f"/sys/devices/system/node/node{node}/cpulist: {e}"
                )
            for cpu in node_cpus:
                for idx in glob.glob(
                        f"/sys/devices/system/cpu/cpu{cpu}/cache/index*"):
                    try:
                        with open(idx + "/level", encoding="utf-8") as f:
                            if f.read().strip() != "3":
                                continue
                        with open(idx + "/id", encoding="utf-8") as f:
                            node_l3_ids.add(int(f.read().strip()))
                        if node_l3_size is None:
                            with open(idx + "/size", encoding="utf-8") as f:
                                node_l3_size = f.read().strip()
                    except (OSError, ValueError):
                        continue

            occupancy_files = []
            for path in sorted(glob.glob(
                    os.path.join(group, "mon_data/mon_L3_*/llc_occupancy"))):
                m = re.search(r"mon_L3_(\d+)", path)
                if m and (not node_l3_ids or int(m.group(1)) in node_l3_ids):
                    occupancy_files.append(path)
            if not occupancy_files:
                GLTCore.GLTTestResult.fail(
                    f"No L3 monitoring domain matched node {node} "
                    f"(L3 cache id(s) {sorted(node_l3_ids)}) under "
                    f"{group}/mon_data; cannot correlate occupancy with the "
                    "bound workload."
                )

            try:
                baseline = resctrl_utils.sum_l3_occupancy(occupancy_files)
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read baseline L3 occupancy from "
                    f"{occupancy_files}: {e}"
                )

            # Derive L3 geometry so the allocation cap can be expressed in
            # bytes.
            try:
                with open(resctrl_utils.RESCTRL_MOUNT_POINT + "/info/L3/cbm_mask",
                          "r", encoding="utf-8") as f:
                    total_ways = bin(int(f.read().strip(), 16)).count("1")
                with open(resctrl_utils.RESCTRL_MOUNT_POINT + "/info/L3/min_cbm_bits",
                          "r", encoding="utf-8") as f:
                    min_cbm_bits = int(f.read().strip())
                if node_l3_size is None:
                    raise ValueError(
                        f"no level-3 cache found for CPUs of node {node}")
                size_str = node_l3_size
                unit = size_str[-1]
                multiplier = {"K": 1024, "M": 1024 ** 2,
                              "G": 1024 ** 3}.get(unit.upper())
                if multiplier is not None:
                    l3_bytes = int(size_str[:-1]) * multiplier
                else:
                    l3_bytes = int(size_str)
                if total_ways <= 0:
                    raise ValueError("non-positive way count")
                bytes_per_way = l3_bytes // total_ways
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not determine L3 geometry for allocation test: {e}"
                )


            allocated_ways = max(min_cbm_bits, total_ways // alloc_fraction)
            partial_mask_val = (1 << allocated_ways) - 1
            partial_mask = format(partial_mask_val, "x")
            with open(group + "/schemata", "r", encoding="utf-8") as f:
                l3_line = next(
                    (ln for ln in f.read().splitlines()
                     if ln.strip().startswith("L3:")), None)
            domains = [entry.split("=", 1)[0].strip()
                       for entry in l3_line.strip()[len("L3:"):].split(";")
                       if "=" in entry]
            write_line = "L3:" + ";".join(
                f"{dom}={partial_mask}" for dom in domains)
            try:
                with open(group + "/schemata", "w", encoding="utf-8") as f:
                    f.write(write_line + "\n")
                with open(group + "/schemata", "r", encoding="utf-8") as f:
                    readback = next(
                        (ln for ln in f.read().splitlines()
                         if ln.strip().startswith("L3:")), "")
                applied = [entry.split("=", 1)[1].strip()
                           for entry in readback.strip()[len("L3:"):].split(";")
                           if "=" in entry]
                # Compare numeric masks: kernels may zero-pad hex on readback
                # (e.g. write "ff", read "00ff") without changing the value.
                if (not applied
                        or any(int(m, 16) != partial_mask_val for m in applied)):
                    raise OSError(f"readback {readback.strip()!r}")
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not apply partial L3 allocation {partial_mask}: {e}"
                )

            capacity_bound = int(
                allocated_ways * bytes_per_way * (1 + tolerance))


            try:
                with open(group + "/tasks", "w", encoding="utf-8") as f:
                    f.write(f"{threading.get_native_id()}\n")
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not move the test task into {group}: {e}"
                )
            try:
                proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    [
                        "numactl", f"--cpunodebind={node}",
                        f"--membind={node}",
                        "stress-ng", "--vm", str(stress_workers),
                        "--vm-bytes", stress_vm_bytes,
                        "--vm-keep", "--timeout", f"{workload_duration_sec}s",
                    ],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                pgid = proc.pid
            finally:
                with open(resctrl_utils.RESCTRL_MOUNT_POINT + "/tasks", "w",
                          encoding="utf-8") as f:
                    f.write(f"{threading.get_native_id()}\n")

            # Sample peak occupancy live, it decays slowly after the workload
            # exits, so the meaningful value is captured during the run.
            peak = baseline
            for _i in range(workload_duration_sec + 1):
                exited = proc.poll() is not None
                try:
                    peak = max(peak, resctrl_utils.sum_l3_occupancy(occupancy_files))
                except (OSError, ValueError):
                    pass
                if exited:
                    break
                time.sleep(sample_interval_sec)

            logger.info(
                "L3 occupancy under workload: baseline=%.1f MiB, "
                "peak=%.1f MiB, cap=%.1f MiB (%d way(s) allocated).",
                baseline / (1024 * 1024), peak / (1024 * 1024),
                capacity_bound / (1024 * 1024), allocated_ways)

            try:
                returncode = proc.wait(timeout=workload_duration_sec + 5)
            except subprocess.TimeoutExpired:
                GLTCore.GLTTestResult.fail(
                    "stress-ng workload did not exit within the expected "
                    "duration, cannot verify L3 occupancy under load."
                )
            if returncode != 0:
                GLTCore.GLTTestResult.fail(
                    f"stress-ng workload exited with status {returncode}. "
                    "Cannot verify L3 occupancy under load."
                )

            if peak <= baseline:
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy did not rise above the "
                    f"{baseline / (1024 * 1024):.1f} MiB baseline. The "
                    "workload may not have been bound to the resctrl group."
                )

            if peak > capacity_bound:
                GLTCore.GLTTestResult.fail(
                    f"L3 occupancy {peak / (1024 * 1024):.1f} MiB exceeded the "
                    f"{capacity_bound / (1024 * 1024):.1f} MiB cap for a "
                    f"{allocated_ways}-way allocation (leaked past CBM)."
                )
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            try:
                os.rmdir(group)
            except OSError as e:
                logger.warning(
                    "Failed to remove resctrl group %s during cleanup: %s",
                    group, e)


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_mba_schemata(_, logger):  # pylint: disable=too-many-branches,too-many-locals
    """
    Verify the ARM64 MPAM MBA percentage control path writes and reads back.

    JIRA: DGX-16763
    """
    group = None

    with resctrl_utils.managed_mount(logger) as (mount_succeeded, missing_cli):
        if missing_cli:
            GLTCore.GLTTestResult.skip(
                f"'{missing_cli}' CLI not available. Cannot mount resctrl.")
        if not mount_succeeded:
            GLTCore.GLTTestResult.fail("Failed to mount resctrl.")
        try:
            info_mb = resctrl_utils.RESCTRL_MOUNT_POINT + "/info/MB"
            try:
                with open(info_mb + "/schema_format", "r", encoding="utf-8") as f:
                    schema_format = f.read().strip()
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read MBA schema_format under {info_mb}: {e}")
            if schema_format != "percentage":
                GLTCore.GLTTestResult.skip(
                    f"MBA not in percentage mode (schema_format={schema_format}).")

            try:
                with open(info_mb + "/min_bandwidth", "r", encoding="utf-8") as f:
                    min_bw = int(f.read().strip())
                with open(info_mb + "/bandwidth_gran", "r", encoding="utf-8") as f:
                    gran = int(f.read().strip())
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read MBA percentage info files under {info_mb}: {e}")
            if gran <= 0:
                GLTCore.GLTTestResult.fail(
                    f"Invalid MBA bandwidth_gran {gran} under {info_mb}; "
                    "granularity must be positive.")
            if not 0 <= min_bw <= 100:
                GLTCore.GLTTestResult.fail(
                    f"Invalid MBA min_bandwidth {min_bw} under {info_mb}. "
                    "Expected a percentage between 0 and 100.")

            # Create a private control group
            group = resctrl_utils.RESCTRL_MOUNT_POINT + "/glt_mba_" + str(os.getpid())
            try:
                os.mkdir(group)
            except FileExistsError:
                try:
                    os.rmdir(group)
                    os.mkdir(group)
                except OSError as e:
                    GLTCore.GLTTestResult.fail(
                        f"Could not create a clean resctrl group {group}: {e}"
                    )
            except OSError as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not create resctrl group {group}: {e}"
                )

            # Parse the MB control domains from the new group's schemata.
            # Native control mode publishes the line as MB_NODE, not MB.
            group_schemata = group + "/schemata"
            try:
                mb_resource = resctrl_utils.mb_control_resource()
                domains = sorted(resctrl_utils.read_schemata(
                    group_schemata, (mb_resource,)).get(mb_resource, {}))
            except (OSError, ValueError) as e:
                GLTCore.GLTTestResult.fail(
                    f"Could not read MB schemata from {group}: {e}"
                )
            if not domains:
                GLTCore.GLTTestResult.fail(
                    f"no {mb_resource} domains parsed from group schemata.")

            # Valid steps are min_bw + N*gran; pick first, middle, and last so
            # coarse granularity still exercises distinct levels (e.g. 0/50/100).
            n_steps = (100 - min_bw) // gran
            valid = [min_bw + gran * i for i in range(n_steps + 1)]
            pick = {0, len(valid) // 2, len(valid) - 1}
            levels = sorted(valid[i] for i in pick)

            for level in levels:
                write_line = f"{mb_resource}:" + ";".join(
                    f"{dom}={level}" for dom in domains)
                try:
                    with open(group_schemata, "w", encoding="utf-8") as f:
                        f.write(write_line + "\n")
                    applied = resctrl_utils.read_schemata(
                        group_schemata, (mb_resource,)).get(mb_resource, {})
                    for dom in domains:
                        if applied.get(dom) != level:
                            GLTCore.GLTTestResult.fail(
                                f"MBA domain {dom}: wrote {level}, read back "
                                f"{applied.get(dom)!r} ({mb_resource} line "
                                f"read {applied!r})"
                            )
                except (OSError, ValueError) as e:
                    GLTCore.GLTTestResult.fail(
                        f"Could not apply MBA level {level} to {group}: {e}"
                    )

        finally:
            if group is not None:
                try:
                    os.rmdir(group)
                except OSError as e:
                    logger.warning(
                        "Failed to remove resctrl group %s during cleanup: %s",
                        group, e)
