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

"""ARM64 resctrl IOMMU group listing and assignment tests."""

import errno
import os
import platform
import time

from common import glt_core as GLTCore
from common import glt_utils
from common import resctrl_utils
from scripts import check_kernel_config


def _last_cmd_status():
    """Return info/last_cmd_status, or a note that it could not be read."""
    path = os.path.join(
        resctrl_utils.RESCTRL_MOUNT_POINT, "info", "last_cmd_status")
    try:
        return glt_utils.read_sysfs_value(path)
    except OSError as error:
        return f"unavailable ({error})"


def _read_group_ids(tasks_path):
    """Return iommu group ids from a tasks file, failing on I/O or parse errors."""
    try:
        content = glt_utils.read_sysfs_value(tasks_path)
    except OSError as error:
        raise RuntimeError(
            f"Cannot read resctrl tasks file {tasks_path}: {error}") from error
    try:
        return resctrl_utils.parse_iommu_group_ids(content)
    except ValueError as error:
        raise RuntimeError(
            f"Malformed iommu_group entries in {tasks_path}: {error}") from error


def _write_group(tasks_path, group_id):
    """Write one iommu_group token to a tasks file."""
    token = f"iommu_group:{group_id}\n"
    with open(tasks_path, "w", encoding="utf-8") as tasks_file:
        tasks_file.write(token)


def _require_kernel_configs():
    """Return a skip reason unless MPAM, resctrl, and IOMMU support are built in."""
    required_configs = (
        "CONFIG_ARM64_MPAM",
        "CONFIG_RESCTRL_FS",
        "CONFIG_RESCTRL_IOMMU",
    )
    kconfig_path = "/boot/config-" + platform.uname().release
    try:
        kconfig = check_kernel_config.parse_kconfig(kconfig_path)
    except FileNotFoundError:
        return (
            f"Kernel config not found at {kconfig_path}; "
            f"cannot verify {', '.join(required_configs)}.")

    for name in required_configs:
        value = kconfig.get(name)
        if value != "y":
            return f"{name} is not enabled (found {value!r})."
    return None


def _require_hidden_syntax():
    """Require iommu_group:0 to be rejected as a pid while the option is off."""
    tasks_path = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "tasks")
    try:
        _write_group(tasks_path, 0)
    except OSError as error:
        status = _last_cmd_status()
        if error.errno == errno.EINVAL and "Task list parsing error" in status:
            return
        raise RuntimeError(
            f"Writing iommu_group:0 without this_is_not_abi failed with "
            f"errno {error.errno} ({error.strerror}), last_cmd_status {status!r}. "
            "Expected EINVAL and 'Task list parsing error'.") from error
    raise RuntimeError(
        "iommu_group syntax was accepted without the this_is_not_abi mount "
        "option.")


def _reject_hidden_syntax(logger):
    """Check the hidden syntax unless this_is_not_abi is already mounted."""
    try:
        current = resctrl_utils.mount_options()
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot determine resctrl mount options from "
            f"{resctrl_utils.PROC_MOUNTS}: {error}") from error
    if current is not None and "this_is_not_abi" in current:
        return

    if current is not None:
        _require_hidden_syntax()
        return

    cleanup_errors = []
    with resctrl_utils.managed_mount(
            logger, cleanup_errors=cleanup_errors) as (mounted, _):
        if not mounted:
            raise RuntimeError("Failed to mount resctrl.")
        _require_hidden_syntax()
    if cleanup_errors:
        raise RuntimeError(cleanup_errors[0])


def _require_move_error(tasks_path, group_id, expected_errno, status_text):
    """Require a tasks write to fail with errno and last_cmd_status text."""
    token = f"iommu_group:{group_id}"
    try:
        _write_group(tasks_path, group_id)
    except OSError as error:
        status = _last_cmd_status()
        if error.errno == errno.EINVAL and "Task list parsing error" in status:
            raise RuntimeError(
                "iommu_group syntax is disabled "
                f"(last_cmd_status {status!r}). "
                "Mount resctrl with this_is_not_abi.") from error
        if error.errno == expected_errno and status_text in status:
            return
        raise RuntimeError(
            f"Writing {token} to {tasks_path} failed with errno {error.errno} "
            f"({error.strerror}), last_cmd_status {status!r}. "
            f"Expected errno {expected_errno} and {status_text!r}.") from error
    raise RuntimeError(
        f"Writing {token} to {tasks_path} succeeded. "
        f"Expected errno {expected_errno} and {status_text!r}.")


def _move_group(tasks_path, group_id):
    """Assign an iommu group, failing with last_cmd_status when the write fails."""
    token = f"iommu_group:{group_id}"
    try:
        _write_group(tasks_path, group_id)
    except OSError as error:
        raise RuntimeError(
            f"Failed to write {token} to {tasks_path}: {error}; "
            f"last_cmd_status {_last_cmd_status()!r}.") from error


def _require_membership(group_id, present, absent):
    """Require group_id to be listed exactly in the present tasks files."""
    token = f"iommu_group:{group_id}"
    for tasks_path in present:
        if group_id not in _read_group_ids(tasks_path):
            raise RuntimeError(f"{token} missing from {tasks_path}.")
    for tasks_path in absent:
        if group_id in _read_group_ids(tasks_path):
            raise RuntimeError(f"{token} still listed in {tasks_path}.")


def _create_ctrl_group(path):
    """Create a control group or fail."""
    try:
        os.mkdir(path)
    except OSError as error:
        raise RuntimeError(
            f"Failed to create resctrl control group {path}: {error}") from error


def _create_mon_group(path, logger):
    """Create a monitor group or fail."""
    try:
        resctrl_utils.make_mon_group(path, logger=logger)
    except OSError as error:
        raise RuntimeError(
            f"Failed to create resctrl monitor group {path}: {error}") from error


def _remove_groups(paths, logger, cleanup_errors):
    """Remove created groups, children first, recording rmdir failures."""
    for path in reversed(paths):
        try:
            os.rmdir(path)
        except OSError as error:
            message = f"Failed to remove resctrl group {path}: {error}"
            logger.warning("%s", message)
            cleanup_errors.append(message)


def _return_group_to_root(root_tasks, group_id, logger, cleanup_errors):
    """Assign group_id to the root tasks file, recording a failed write."""
    try:
        _move_group(root_tasks, group_id)
    except RuntimeError as error:
        logger.warning("%s", error)
        cleanup_errors.append(str(error))


def _listed_group_id():
    """Return the lowest iommu group listed on the root tasks file."""
    iommu_groups = "/sys/kernel/iommu_groups"
    root_tasks = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "tasks")
    group_ids = _read_group_ids(root_tasks)
    if not group_ids:
        raise RuntimeError(
            "resctrl root tasks file lists no iommu_group entries.")
    missing = [
        group_id for group_id in sorted(group_ids)
        if not os.path.isdir(os.path.join(iommu_groups, str(group_id)))
    ]
    if missing:
        raise RuntimeError(
            f"resctrl lists iommu groups that are absent from {iommu_groups}: "
            f"{missing}.")
    return min(group_ids)


def _assign_one_group(logger, cleanup_errors):
    """Move one listed group through control and monitor groups, then back.

    A failure after the group leaves the root tasks file writes it back
    before the created directories are removed. Directory removal leaves
    the assignment where it is, and rmdir can fail while the group is
    still assigned.
    """
    root_tasks = os.path.join(resctrl_utils.RESCTRL_MOUNT_POINT, "tasks")
    unknown_iommu_group = 2147483647
    _require_move_error(
        root_tasks, unknown_iommu_group, errno.ESRCH, "No matching iommu_group")
    group_id = _listed_group_id()
    logger.info("Assigning iommu_group:%s through resctrl groups.", group_id)

    token = f"{os.getpid()}_{time.time_ns()}"
    ctrl_path = os.path.join(
        resctrl_utils.RESCTRL_MOUNT_POINT, f"glt_iommu_{token}")
    ctrl_tasks = os.path.join(ctrl_path, "tasks")
    ctrl_mon = os.path.join(ctrl_path, "mon_groups", f"glt_iommu_mon_{token}")
    ctrl_mon_tasks = os.path.join(ctrl_mon, "tasks")
    root_mon = os.path.join(
        resctrl_utils.RESCTRL_MOUNT_POINT, "mon_groups", f"glt_iommu_mon_{token}")
    root_mon_tasks = os.path.join(root_mon, "tasks")
    created = []
    restore_group = False
    body_error = None
    try:
        try:
            _create_ctrl_group(ctrl_path)
            created.append(ctrl_path)
            _move_group(ctrl_tasks, group_id)
            restore_group = True
            _require_membership(group_id, (ctrl_tasks,), (root_tasks,))

            _create_mon_group(ctrl_mon, logger)
            created.append(ctrl_mon)
            _move_group(ctrl_mon_tasks, group_id)
            _require_membership(
                group_id, (ctrl_mon_tasks, ctrl_tasks), (root_tasks,))

            _create_mon_group(root_mon, logger)
            created.append(root_mon)
            _require_move_error(
                root_mon_tasks, group_id, errno.EINVAL,
                "Can't move iommu_group to different control group")
            _require_membership(
                group_id, (ctrl_mon_tasks, ctrl_tasks),
                (root_tasks, root_mon_tasks))

            _move_group(root_tasks, group_id)
            restore_group = False
            _require_membership(
                group_id, (root_tasks,),
                (ctrl_tasks, ctrl_mon_tasks, root_mon_tasks))
        except RuntimeError as caught:
            body_error = caught
    finally:
        if restore_group:
            _return_group_to_root(root_tasks, group_id, logger, cleanup_errors)
        _remove_groups(created, logger, cleanup_errors)

    if body_error is None:
        return
    extra = next(
        (item for item in cleanup_errors if item not in str(body_error)), None)
    if extra is None:
        raise body_error
    raise RuntimeError(f"{body_error} {extra}") from body_error


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_iommu_group_query(_, logger):
    """
    Require the default resctrl tasks file to name a real IOMMU group.

    The assignment test moves one listed group, so this check passes only
    when the root tasks file names a directory under
    /sys/kernel/iommu_groups. A listed line means the tasks-file walk
    accepted that group as a member of the default resource group. That
    listing comes from the linux-nvidia assignment series: resctrl
    assignment 2e9f350f0a33, helpers d2ec08a0ea33, and SMMU get/set
    668c0fb85146.
    A group that does not match the default closid is omitted, and an
    empty listing fails this check.

    The iommu_group syntax is enabled by mounting resctrl with
    this_is_not_abi. A mount that already includes that option is left in
    place. A mount without it is replaced for the test and restored
    afterward.

    PATCH: NVIDIA SAUCE 2e9f350f0a3338f7a172b11837a95547a05a1275
    PATCH: d2ec08a0ea333093daa6c8ba6a03cd5803586b7a
    PATCH: 668c0fb851463efd2444a553685a31fcb1c0d91e
    """
    skip_reason = _require_kernel_configs()
    if skip_reason:
        GLTCore.GLTTestResult.skip(skip_reason)

    try:
        cleanup_errors = []
        with resctrl_utils.mount_with_options(
                logger, ("this_is_not_abi",),
                cleanup_errors=cleanup_errors,
        ) as (mount_succeeded, detail):
            if not mount_succeeded:
                raise RuntimeError(
                    detail or "Failed to mount resctrl with this_is_not_abi.")
            group_id = _listed_group_id()
            logger.info(
                "Default resctrl group lists iommu_group:%s.", group_id)

        if cleanup_errors:
            raise RuntimeError(cleanup_errors[0])
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(str(error))


@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_iommu_group_query")
@GLTCore.GLTRequires.must_pass("arm64", "test_mpam_resctrl_mount")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_mpam_resctrl_iommu_group_assign(_, logger):
    """
    Verify resctrl can list and move IOMMU groups through the tasks file.

    The iommu_group syntax is enabled by mounting resctrl with
    this_is_not_abi. A mount that already includes that option is left in
    place. A mount without it is replaced for the test and restored
    afterward. Unmount resets resctrl groups and schemata; only the
    previous mount flags come back.

    One listed group is moved into a new control group and a child monitor
    group, then returned to the default group. The move updates that
    group's SMMU MPAM labels and does not change schemata. A later failure
    writes the group back to the root tasks file before removing the test
    directories.

    PATCH: NVIDIA SAUCE bf70ae35bce2
    """
    skip_reason = _require_kernel_configs()
    if skip_reason:
        GLTCore.GLTTestResult.skip(skip_reason)

    try:
        _reject_hidden_syntax(logger)

        cleanup_errors = []
        with resctrl_utils.mount_with_options(
                logger, ("this_is_not_abi",),
                cleanup_errors=cleanup_errors,
        ) as (mount_succeeded, detail):
            if not mount_succeeded:
                raise RuntimeError(
                    detail or "Failed to mount resctrl with this_is_not_abi.")
            _assign_one_group(logger, cleanup_errors)

        if cleanup_errors:
            raise RuntimeError(cleanup_errors[0])
    except RuntimeError as error:
        GLTCore.GLTTestResult.fail(str(error))
