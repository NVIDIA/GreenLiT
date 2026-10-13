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

"""
Helpers for the Linux resctrl filesystem.

These helpers read and parse the resctrl interface and manage a
test-owned mount. They return facts or raise OSError/ValueError; the
calling test decides whether a condition is a skip or a failure.
"""

from contextlib import contextmanager
import os
import re

from common import glt_utils

RESCTRL_FS_TYPE = "resctrl"
RESCTRL_MOUNT_POINT = "/sys/fs/resctrl"
PROC_FILESYSTEMS = "/proc/filesystems"
PROC_MOUNTS = "/proc/mounts"
MON_FEATURES = "mon_features"
MBM_TOTAL_BYTES = "mbm_total_bytes"


def fs_supported() -> bool:
    """Return True if the kernel registers the resctrl filesystem type.

    Raises OSError when PROC_FILESYSTEMS cannot be read. That is not the
    same as "not registered", so the caller decides how to report it.
    """
    with open(PROC_FILESYSTEMS, "r", encoding="utf-8") as f:
        return any(line.split()[-1] == RESCTRL_FS_TYPE
                   for line in f if line.strip())


def mount_points():
    """Return every mount point currently using the resctrl filesystem."""
    with open(PROC_MOUNTS, "r", encoding="utf-8") as f:
        return [
            fields[1] for line in f
            if len(fields := line.split()) >= 3
            and fields[2] == RESCTRL_FS_TYPE
        ]


def is_mounted() -> bool:
    """Return True if resctrl is mounted at RESCTRL_MOUNT_POINT.

    Raises OSError when PROC_MOUNTS cannot be read.
    """
    return RESCTRL_MOUNT_POINT in mount_points()


@contextmanager
def managed_mount(logger, *, cleanup_errors=None):
    """Yield resctrl mount state and clean up only a test-owned mount.

    Yields (mount_succeeded, missing_cli). missing_cli is the first of
    mount/umount that is not on PATH when a mount is required, else None.
    Callers skip when missing_cli is set and fail when mount_succeeded is
    False.

    mount_succeeded is also False when PROC_MOUNTS cannot be read. The
    cause is logged as an error and, because the current state is unknown,
    no mount is attempted.

    Unmount failures are logged. When cleanup_errors is a list, the failure
    message is also appended so a decorated caller can fail after a
    successful body without replacing an earlier skip or fail.
    """
    mounted_by_test = False
    missing_cli = None
    mount_succeeded = False
    try:
        try:
            mount_succeeded = is_mounted()
        except OSError as error:
            logger.error(
                "Cannot determine resctrl mount state from %s: %s",
                PROC_MOUNTS, error)
            needs_mount = False
        else:
            needs_mount = not mount_succeeded

        if needs_mount:
            missing_cli = next(
                (cli for cli in ("mount", "umount")
                 if not glt_utils.is_cli_available(cli)),
                None,
            )
            if missing_cli is None:
                try:
                    glt_utils.run_command(
                        ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE,
                         RESCTRL_MOUNT_POINT],
                        "Failed to mount resctrl.",
                    )
                except RuntimeError:
                    pass
                else:
                    mounted_by_test = True
                    try:
                        mount_succeeded = is_mounted()
                    except OSError as error:
                        logger.error(
                            "Cannot confirm the resctrl mount from %s: %s",
                            PROC_MOUNTS, error)

        yield mount_succeeded, missing_cli
    finally:
        if mounted_by_test:
            try:
                glt_utils.run_command(
                    ["umount", RESCTRL_MOUNT_POINT],
                    "Failed to unmount resctrl.",
                )
            except RuntimeError as error:
                logger.warning("Failed to unmount resctrl.")
                if cleanup_errors is not None:
                    cleanup_errors.append(str(error))


def _parse_schemata_entries(resource, values, path, value_pattern):
    """Return {domain: value_text} for one schemata line's entries.

    Every entry must match "<decimal domain>=<value_pattern>" and no domain
    may repeat. Raises ValueError otherwise, naming the file and entry.
    """
    entry_re = re.compile(rf"([0-9]+)=({value_pattern})")
    domains = {}
    for entry in values.split(";"):
        match = entry_re.fullmatch(entry)
        if match is None:
            raise ValueError(f"Malformed {resource} entry in {path}: {entry!r}")
        domain = int(match.group(1))
        if domain in domains:
            raise ValueError(f"Duplicate {resource} domain {domain} in {path}")
        domains[domain] = match.group(2)
    return domains


def read_schemata(path, resources):
    """Return {resource: {domain: value}} for the named schemata resources.

    Values must be decimal integers, as for the MB resources. Only lines
    for the requested resources are parsed; other resources are ignored
    and a requested resource that is absent is simply missing from the
    result. Raises ValueError on a malformed or duplicate line, a
    malformed entry, or a duplicate domain.
    """
    with open(path, "r", encoding="ascii") as schemata_file:
        content = schemata_file.read()

    schemas = {}
    for line in content.splitlines():
        resource, separator, values = line.strip().partition(":")
        resource = resource.strip()
        if resource not in resources:
            continue
        if not separator or resource in schemas:
            raise ValueError(f"Malformed or duplicate {resource} line in {path}: {line!r}")
        schemas[resource] = {
            domain: int(value)
            for domain, value in _parse_schemata_entries(
                resource, values, path, "[0-9]+").items()
        }
    return schemas


def mb_control_resource() -> str:
    """Return the MB control name that appears in a group's schemata.

    info/MB/control_mode selects between the legacy "MB" line and the
    native node-scoped "MB_NODE" line. Platforms without the emulated
    control publish no control_mode and always use "MB". Raises
    ValueError when the file marks neither mode active.
    """
    control_mode_path = os.path.join(
        RESCTRL_MOUNT_POINT, "info", "MB", "control_mode")
    if not os.path.isfile(control_mode_path):
        return "MB"

    modes = glt_utils.read_sysfs_value(control_mode_path).split()
    if "[native]" in modes:
        return "MB_NODE"
    if "[legacy]" in modes:
        return "MB"
    raise ValueError(
        f"MB control mode file {control_mode_path} has no active legacy "
        f"or native mode (read {modes!r})."
    )


def schemata_domains(resource):
    """Return the set of domain IDs on a resource's root schemata line.

    Values are not interpreted, so this works for bitmask resources such
    as L3 as well as for MB. resctrl right-aligns resource names, so each
    line is stripped before it is matched. Returns None when the root
    schemata file carries no line for the resource. Raises ValueError on
    a malformed entry or a duplicate domain.
    """
    schemata_path = os.path.join(RESCTRL_MOUNT_POINT, "schemata")
    prefix = f"{resource}:"

    with open(schemata_path, "r", encoding="utf-8") as schemata:
        line = next(
            (raw.strip() for raw in schemata
             if raw.strip().startswith(prefix)),
            None,
        )
    if line is None:
        return None

    return set(_parse_schemata_entries(
        resource, line[len(prefix):], schemata_path, r"[^;=\s]+"))


def read_mbm_assignment_states(path, event=MBM_TOTAL_BYTES):
    """Return {domain: state} for an event in a resctrl MBM assignment file.

    A state is "e" (assigned) or "_" (unassigned). Raises ValueError on a
    malformed line or entry, a duplicate event line or domain, or when the
    event has no assignment states.
    """
    content = glt_utils.read_sysfs_value(path)
    states = None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, rest = line.partition(":")
        if not sep:
            raise ValueError(
                f"Malformed MBM assignment line {line!r} in {path}")
        if name != event:
            continue
        if states is not None:
            raise ValueError(
                f"Duplicate {event} assignment line in {path}")
        states = {}
        for entry in rest.split(";"):
            match = re.fullmatch(r"([0-9]+)=([e_])", entry)
            if match is None:
                raise ValueError(
                    f"Malformed MBM assignment entry {entry!r} in {path}")
            domain = int(match.group(1))
            if domain in states:
                raise ValueError(
                    f"Duplicate MBM domain {domain} in {path}")
            states[domain] = match.group(2)
    if not states:
        raise ValueError(
            f"No {event} assignment states in {path}")
    return states


def read_mbm_counter_counts(path):
    """Return {domain: count} from an MBM counter information file."""
    content = glt_utils.read_sysfs_value(path)
    if not content:
        raise ValueError(f"MBM counter information file {path} is empty")
    counts = {}
    for entry in content.split(";"):
        match = re.fullmatch(r"([0-9]+)=([0-9]+)", entry)
        if match is None:
            raise ValueError(
                f"Malformed MBM counter entry {entry!r} in {path}")
        domain = int(match.group(1))
        if domain in counts:
            raise ValueError(f"Duplicate MBM domain {domain} in {path}")
        counts[domain] = int(match.group(2))
    return counts


def sum_l3_occupancy(occupancy_files) -> int:
    """Return total llc_occupancy summed across the given mon_L3 domains."""
    total = 0
    for occ in occupancy_files:
        with open(occ, "r", encoding="utf-8") as f:
            total += int(f.read().strip())
    return total
