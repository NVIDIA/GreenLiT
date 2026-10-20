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
import errno
import os
import re
import time

from common import glt_utils

RESCTRL_FS_TYPE = "resctrl"
RESCTRL_MOUNT_POINT = "/sys/fs/resctrl"
PROC_FILESYSTEMS = "/proc/filesystems"
PROC_MOUNTS = "/proc/mounts"
MON_FEATURES = "mon_features"
MBM_TOTAL_BYTES = "mbm_total_bytes"
MON_GROUP_RETRY_INTERVAL_SEC = 0.5
RESCTRL_MOUNT_OPTIONS = frozenset({
    "cdp",
    "cdpl2",
    "debug",
    "mba_MBps",
    "this_is_not_abi",
})


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


def mount_options():
    """Return the known resctrl flags at RESCTRL_MOUNT_POINT.

    Returns None when resctrl is not mounted there. Returns a frozenset of
    the known flags, possibly empty, when it is. Generic flags such as rw
    are ignored. Raises OSError when PROC_MOUNTS cannot be read, and
    ValueError when the matching mount line has no options field.
    """
    with open(PROC_MOUNTS, "r", encoding="utf-8") as mounts:
        for line in mounts:
            fields = line.split()
            if (len(fields) < 3 or fields[1] != RESCTRL_MOUNT_POINT
                    or fields[2] != RESCTRL_FS_TYPE):
                continue
            if len(fields) < 4:
                raise ValueError(
                    f"resctrl mount line in {PROC_MOUNTS} has no options: {line!r}")
            return frozenset(
                flag for flag in fields[3].split(",")
                if flag in RESCTRL_MOUNT_OPTIONS)
    return None


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


def _requested_mount_options(options):
    """Return options as a frozenset, rejecting a bare string or unknown flags."""
    if isinstance(options, str):
        raise TypeError(
            "resctrl mount options must be a sequence of flag names, not a string")
    requested = frozenset(options)
    unknown = requested - RESCTRL_MOUNT_OPTIONS
    if unknown:
        raise ValueError(
            "Unknown resctrl mount option(s): " + ", ".join(sorted(unknown)))
    return requested


def _restore_previous_mount(logger, saved_options, cleanup_errors):
    """Remount saved flags. None means resctrl was not previously mounted."""
    if saved_options is None:
        return True
    option_text = ",".join(sorted(saved_options)) or "no resctrl options"
    failure_message = f"Failed to restore resctrl mount with {option_text}."
    command = ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE, RESCTRL_MOUNT_POINT]
    if saved_options:
        command.extend(["-o", option_text])
    try:
        glt_utils.run_command(command, failure_message)
    except RuntimeError as error:
        logger.warning("%s", failure_message)
        if cleanup_errors is not None:
            cleanup_errors.append(str(error))
        return False
    return True


def _unmount_for_cleanup(logger, cleanup_errors):
    """Unmount resctrl during rollback or exit. Return False when it fails."""
    failure_message = "Failed to unmount resctrl."
    try:
        glt_utils.run_command(["umount", RESCTRL_MOUNT_POINT], failure_message)
    except RuntimeError as error:
        logger.warning("%s", failure_message)
        if cleanup_errors is not None:
            cleanup_errors.append(str(error))
        return False
    return True


def _replace_mount(logger, saved_options, cleanup_errors):
    """Remove the helper mount and restore the mount found on entry.

    Return False when the unmount or the restore mount fails.
    """
    if not _unmount_for_cleanup(logger, cleanup_errors):
        return False
    return _restore_previous_mount(logger, saved_options, cleanup_errors)


def _switch_mount(logger, requested, saved_options, cleanup_errors):
    """Replace the current mount with requested.

    Return (ready, error, owned). owned is True only when the requested
    mount is in place and the caller must restore saved_options on exit.
    A failed switch rolls back before returning.
    """
    if saved_options is not None:
        try:
            glt_utils.run_command(
                ["umount", RESCTRL_MOUNT_POINT], "Failed to unmount resctrl.")
        except RuntimeError as error:
            logger.error("%s", error)
            return False, str(error), False

    option_text = ",".join(sorted(requested)) or "no resctrl options"
    failure_message = f"Failed to mount resctrl with {option_text}."
    command = ["mount", "-t", RESCTRL_FS_TYPE, RESCTRL_FS_TYPE, RESCTRL_MOUNT_POINT]
    if requested:
        command.extend(["-o", option_text])
    try:
        glt_utils.run_command(command, failure_message)
    except RuntimeError as error:
        logger.error("%s", error)
        if _restore_previous_mount(logger, saved_options, cleanup_errors):
            return False, str(error), False
        return False, f"{error} Also failed to restore the previous resctrl mount.", False

    try:
        confirmed = mount_options()
    except (OSError, ValueError) as error:
        message = f"Cannot confirm resctrl mount options from {PROC_MOUNTS}: {error}"
    else:
        if confirmed == requested:
            return True, None, True
        message = (
            f"resctrl mount options {confirmed!r} do not match "
            f"requested {sorted(requested)!r}.")

    logger.error("%s", message)
    if not _replace_mount(logger, saved_options, cleanup_errors):
        message += " Also failed to restore the previous resctrl mount."
    return False, message, False


@contextmanager
def mount_with_options(logger, options, *, cleanup_errors=None):
    """Yield resctrl mounted with the requested resctrl flags.

    options is a sequence of known resctrl mount flags. When the mount at
    RESCTRL_MOUNT_POINT already includes every requested flag, it is left
    in place, including any extra resctrl flags. Otherwise a pre-existing
    mount is unmounted, resctrl is mounted with those flags, and on exit
    that mount is removed. Saved flags are then mounted again. A
    filesystem that was not mounted on entry stays unmounted.

    Unmounting resctrl resets every group and schemata. Restoring the
    mount brings the previous flags back, not the previous groups.

    Yields (mount_succeeded, detail). detail is None on success and the
    setup error otherwise. Callers fail with detail when mount_succeeded
    is False.

    Setup failures yield mount_succeeded False and do not leave the
    requested mount in place. Cleanup failures are logged. When
    cleanup_errors is a list, the failure message is appended so a
    decorated caller can fail after a successful body.
    """
    requested = _requested_mount_options(options)
    mount_succeeded = False
    setup_error = None
    mounted_by_helper = False
    saved_options = None
    try:
        try:
            current = mount_options()
        except (OSError, ValueError) as error:
            setup_error = (
                f"Cannot determine resctrl mount options from {PROC_MOUNTS}: {error}")
            logger.error("%s", setup_error)
        else:
            if current is not None and requested <= current:
                mount_succeeded = True
            else:
                saved_options = current
                mount_succeeded, setup_error, mounted_by_helper = _switch_mount(
                    logger, requested, saved_options, cleanup_errors)

        yield mount_succeeded, setup_error
    finally:
        if mounted_by_helper and mount_succeeded:
            _replace_mount(logger, saved_options, cleanup_errors)


def parse_iommu_group_ids(text):
    """Return the iommu group ids listed in resctrl tasks-file text.

    Blank lines and pid lines are ignored. A line that starts with
    iommu_group but not iommu_group: is malformed. Raises ValueError
    on a malformed iommu_group line or a duplicate id.
    """
    prefix = "iommu_group:"
    group_ids = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(prefix):
            if line.startswith("iommu_group"):
                raise ValueError(f"Malformed iommu group line: {line!r}")
            continue
        id_text = line[len(prefix):]
        if not id_text.isdecimal():
            raise ValueError(f"Malformed iommu group line: {line!r}")
        group_id = int(id_text)
        if group_id in group_ids:
            raise ValueError(f"Duplicate iommu group {group_id}")
        group_ids.add(group_id)
    return group_ids


def make_mon_group(path, timeout_sec=10.0, *, logger=None):
    """Create a resctrl monitor group, waiting for a monitor ID if needed.

    resctrl parks a deleted group's monitor ID (RMID) in a limbo list and
    only returns it to the free pool on a periodic check, about once per
    second. On MPAM with few PMGs a control group may own a single spare
    RMID, so mkdir right after a sibling group was removed fails with
    ENOSPC ("Out of RMIDs") or EBUSY. Retry those two errors every
    MON_GROUP_RETRY_INTERVAL_SEC until timeout_sec has elapsed since the
    first failure. Any other OSError propagates unchanged.

    On timeout raise OSError carrying the last errno and the kernel's
    info/last_cmd_status text. When logger is given, log the wait.
    """
    status_path = os.path.join(RESCTRL_MOUNT_POINT, "info", "last_cmd_status")
    deadline = None
    while True:
        try:
            os.mkdir(path)
        except OSError as error:
            if error.errno not in (errno.ENOSPC, errno.EBUSY):
                raise
            try:
                status = glt_utils.read_sysfs_value(status_path)
            except OSError:
                status = "unavailable"
            now = time.monotonic()
            if deadline is None:
                deadline = now + timeout_sec
                if logger is not None:
                    logger.info(
                        "Waiting up to %gs for a free resctrl monitor ID to "
                        "create %s (last_cmd_status: %s).",
                        timeout_sec, path, status)
            if now >= deadline:
                raise OSError(
                    error.errno,
                    f"{error.strerror} after waiting {timeout_sec:g}s for a "
                    f"free resctrl monitor ID (last_cmd_status: {status})",
                ) from error
            time.sleep(min(MON_GROUP_RETRY_INTERVAL_SEC, deadline - now))
        else:
            if deadline is not None and logger is not None:
                waited = time.monotonic() - (deadline - timeout_sec)
                logger.info(
                    "Created resctrl monitor group %s after waiting %.1fs.",
                    path, waited)
            return


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
