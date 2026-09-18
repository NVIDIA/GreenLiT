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

"""Workqueue CACHE_SHARD affinity-scope validation."""

import errno
import os

from common import glt_core as GLTCore
from common import glt_utils


CACHE_SHARD_SCOPE = "cache_shard"
CACHE_SHARD_SIZE_PARAM = "/sys/module/workqueue/parameters/cache_shard_size"
DEFAULT_AFFINITY_SCOPE_PARAM = "/sys/module/workqueue/parameters/default_affinity_scope"
EXPECTED_DEFAULT_CACHE_SHARD_SIZE = 8
INVALID_CACHE_SHARD_PREFIX = "cache_shard_invalid"
PROC_CMDLINE = "/proc/cmdline"
SYS_MODULE_ROOT = "/sys/module"
WORKQUEUE_MODULE_NAME = "workqueue"
CURRENT_DEFAULT_SCOPE_LIMITATION = (
    "This validates the current effective workqueue default only; "
    "default_affinity_scope is writable at runtime and this observation cannot "
    "prove the compiled-in default if privileged userspace changed it after boot."
)
WORKQUEUE_AFFINITY_SCOPES = (
    "cpu",
    "smt",
    "cache",
    "numa",
    "system",
    CACHE_SHARD_SCOPE,
)
WQ_AFFINITY_SCOPE_FILE = "affinity_scope"
WQ_ROOTS = ("/sys/devices/virtual/workqueue", "/sys/bus/workqueue/devices")
WQ_PREFERRED_NAMES = (
    "writeback",
    "events",
    "events_unbound",
    "ib-comp-unb-wq",
    "ib-comp-wq",
    "nvme-wq",
)

def _canonical_scope(scope_display):
    """Return the scope name from a raw affinity_scope sysfs display string."""
    scope = scope_display.strip()
    prefix = "default ("
    if scope.startswith(prefix):
        end = scope.find(")", len(prefix))
        if end != -1:
            return scope[len(prefix):end].strip()
    return scope


def _has_workqueue_cmdline_override(parameter_name):
    """Return whether /proc/cmdline contains a workqueue.<parameter_name> override."""
    try:
        cmdline = glt_utils.read_sysfs_value(PROC_CMDLINE)
    except (IOError, OSError):
        return False

    prefix = f"workqueue.{parameter_name}="
    return any(token.startswith(prefix) for token in cmdline.split())


def _find_affinity_scope_path(workqueue_roots=WQ_ROOTS):
    """Return a readable workqueue affinity_scope sysfs path, if one exists."""
    visited_dirs = set()

    for root in workqueue_roots:
        candidates = [os.path.join(root, name) for name in WQ_PREFERRED_NAMES]
        try:
            discovered = sorted(os.listdir(root))
        except (IOError, OSError):
            discovered = []

        candidates.extend(
            os.path.join(root, name)
            for name in discovered
            if name != "power"
        )

        for workqueue_dir in candidates:
            if workqueue_dir in visited_dirs:
                continue
            visited_dirs.add(workqueue_dir)
            if not os.path.isdir(workqueue_dir):
                continue

            scope_path = os.path.join(workqueue_dir, WQ_AFFINITY_SCOPE_FILE)
            if not os.path.isfile(scope_path):
                continue

            try:
                glt_utils.read_sysfs_value(scope_path)
            except (IOError, OSError):
                continue
            return scope_path

    return None


def _module_loaded(module_name, module_root=SYS_MODULE_ROOT, path_is_dir=os.path.isdir):
    """Return whether a kernel module is currently loaded."""
    sysfs_name = module_name.replace("-", "_")
    return path_is_dir(os.path.join(module_root, sysfs_name))


def _read_cache_shard_size():
    """Read and validate the workqueue cache_shard_size module parameter."""
    try:
        raw_value = glt_utils.read_sysfs_value(CACHE_SHARD_SIZE_PARAM)
    except (IOError, OSError) as error:
        return None, f"{CACHE_SHARD_SIZE_PARAM} is not readable: {error}"

    try:
        shard_size = int(raw_value)
    except ValueError:
        return None, f"{CACHE_SHARD_SIZE_PARAM} is not an integer: {raw_value!r}"

    if shard_size <= 0:
        return None, f"{CACHE_SHARD_SIZE_PARAM} is not positive: {shard_size}"

    return shard_size, ""


def _scope_restore_token(scope_display):
    """Return the token that restores a raw affinity_scope display value."""
    if scope_display.startswith("default ("):
        return "default"
    return _canonical_scope(scope_display)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_workqueue_cache_shard_default_scope(_, logger):  # pylint: disable=too-many-branches
    """
    Verify workqueue exposes CACHE_SHARD and currently uses it by default.
    """
    shard_size, reason = _read_cache_shard_size()
    if reason:
        if _module_loaded(WORKQUEUE_MODULE_NAME):
            GLTCore.GLTTestResult.fail(
                f"{reason}. Kernel is missing valid WQ_AFFN_CACHE_SHARD support; "
                f"{CACHE_SHARD_SIZE_PARAM} must be readable on target kernels."
            )
        GLTCore.GLTTestResult.skip(
            f"{reason}. Cannot validate WQ_AFFN_CACHE_SHARD because "
            f"/sys/module/{WORKQUEUE_MODULE_NAME} is unavailable; this may be a "
            "non-target kernel or a system without workqueue module sysfs."
        )

    logger.info("workqueue cache_shard_size=%d", shard_size)
    if shard_size != EXPECTED_DEFAULT_CACHE_SHARD_SIZE:
        if _has_workqueue_cmdline_override("cache_shard_size"):
            logger.info(
                "workqueue cache_shard_size default check skipped because "
                "workqueue.cache_shard_size is set on /proc/cmdline."
            )
        else:
            GLTCore.GLTTestResult.fail(
                f"workqueue cache_shard_size is {shard_size}, expected "
                f"{EXPECTED_DEFAULT_CACHE_SHARD_SIZE}. Consider checking whether the kernel "
                "has WQ_AFFN_CACHE_SHARD support or was booted with a custom "
                "workqueue.cache_shard_size value."
            )

    scope = None
    source = ""
    raw_scope = ""
    try:
        raw_scope = glt_utils.read_sysfs_value(DEFAULT_AFFINITY_SCOPE_PARAM)
    except (IOError, OSError):
        raw_scope = ""

    if raw_scope:
        source = DEFAULT_AFFINITY_SCOPE_PARAM
        scope = _canonical_scope(raw_scope)
    else:
        scope_path = _find_affinity_scope_path()
        if scope_path:
            source = scope_path
            raw_scope = glt_utils.read_sysfs_value(scope_path)
            if raw_scope.startswith("default (") and ")" in raw_scope:
                scope = _canonical_scope(raw_scope)

    if scope is None:
        detail = (
            f"{DEFAULT_AFFINITY_SCOPE_PARAM} and WQ_SYSFS default affinity_scope "
            "display are unavailable."
        )
        if source:
            detail = (
                f"{DEFAULT_AFFINITY_SCOPE_PARAM} is unavailable and {source} "
                f"reports {raw_scope!r}; only a 'default (...)' queue display "
                "proves the current effective global default scope."
            )
        GLTCore.GLTTestResult.skip(
            f"Cannot determine current effective workqueue default affinity scope; "
            f"{detail} {CURRENT_DEFAULT_SCOPE_LIMITATION}"
        )

    logger.info(
        "workqueue current effective default affinity scope from %s: %s. %s",
        source,
        raw_scope,
        CURRENT_DEFAULT_SCOPE_LIMITATION,
    )
    if scope != CACHE_SHARD_SCOPE:
        if _has_workqueue_cmdline_override("default_affinity_scope"):
            GLTCore.GLTTestResult.skip(
                "workqueue.default_affinity_scope is set on /proc/cmdline; "
                "cannot validate the configured default from observed current "
                f"effective scope {raw_scope!r}. {CURRENT_DEFAULT_SCOPE_LIMITATION}"
            )
        GLTCore.GLTTestResult.fail(
            f"workqueue current effective default affinity scope is {raw_scope!r}, expected "
            f"{CACHE_SHARD_SCOPE!r}. Consider checking whether the kernel is missing "
            "the cache-shard default affinity configuration. "
            f"{CURRENT_DEFAULT_SCOPE_LIMITATION}"
        )


@GLTCore.GLTRequires.must_pass("cpu", "test_workqueue_cache_shard_default_scope")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_workqueue_cache_shard_scope_sysfs_round_trip(_, logger):  # pylint: disable=too-many-branches
    """
    Verify affinity_scope accepts cache_shard and rejects prefix-only matches.
    """
    scope_path = _find_affinity_scope_path()
    if not scope_path:
        GLTCore.GLTTestResult.skip(
            "No readable workqueue affinity_scope sysfs file was found; "
            "CONFIG_WQ_SYSFS may be disabled."
        )

    logger.info("Testing workqueue affinity_scope sysfs path: %s", scope_path)
    try:
        original_scope = glt_utils.read_sysfs_value(scope_path)
    except (IOError, OSError) as error:
        GLTCore.GLTTestResult.fail(f"Cannot read {scope_path}: {error}")

    logger.info(
        "Original workqueue affinity_scope on %s was %r",
        scope_path,
        original_scope,
    )

    changed_scope = False
    failure_reason = ""
    try:
        for scope in WORKQUEUE_AFFINITY_SCOPES:
            logger.info("Writing workqueue affinity_scope %r to %s", scope, scope_path)
            try:
                glt_utils.write_sysfs_value(scope_path, scope)
                changed_scope = True
                try:
                    read_back = glt_utils.read_sysfs_value(scope_path)
                except (IOError, OSError) as error:
                    failure_reason = (
                        f"Cannot read {scope_path} after writing {scope!r}: {error}."
                    )
                else:
                    logger.info("workqueue affinity_scope read back %r", read_back)
                    if _canonical_scope(read_back) != scope:
                        failure_reason = (
                            f"Writing {scope!r} to {scope_path} read back {read_back!r}."
                        )
            except (IOError, OSError) as error:
                failure_reason = f"Writing {scope!r} to {scope_path} failed: {error}."

            if failure_reason:
                if scope == CACHE_SHARD_SCOPE:
                    failure_reason = (
                        f"{failure_reason} Consider checking whether the kernel is missing "
                        "WQ_AFFN_CACHE_SHARD support or exact affinity-scope parsing."
                    )
                break

        if not failure_reason:
            try:
                logger.info(
                    "Verifying invalid workqueue affinity_scope %r is rejected",
                    INVALID_CACHE_SHARD_PREFIX,
                )
                glt_utils.write_sysfs_value(scope_path, INVALID_CACHE_SHARD_PREFIX)
                invalid_read_back = glt_utils.read_sysfs_value(scope_path)
                failure_reason = (
                    f"Invalid affinity scope {INVALID_CACHE_SHARD_PREFIX!r} was "
                    f"accepted by {scope_path}; read back {invalid_read_back!r}. "
                    "Consider checking whether the kernel has exact affinity-scope parsing."
                )
            except (IOError, OSError) as error:
                error_number = getattr(error, "errno", None)
                if error_number == errno.EINVAL:
                    logger.info(
                        "Invalid workqueue affinity_scope %r was rejected with EINVAL",
                        INVALID_CACHE_SHARD_PREFIX,
                    )
                else:
                    failure_reason = (
                        f"Writing invalid affinity scope {INVALID_CACHE_SHARD_PREFIX!r} "
                        f"to {scope_path} failed with unexpected error {error} "
                        f"(errno={error_number}); expected errno.EINVAL to prove "
                        "exact affinity-scope parsing."
                    )
    finally:
        if changed_scope:
            try:
                logger.info(
                    "Restoring workqueue affinity_scope on %s to %r",
                    scope_path,
                    original_scope,
                )
                restore_token = _scope_restore_token(original_scope)
                glt_utils.write_sysfs_value(scope_path, restore_token)
                restored_scope = glt_utils.read_sysfs_value(scope_path)
                logger.info("Restored workqueue affinity_scope read back %r", restored_scope)
                restore_error = ""
                if restored_scope != original_scope and (
                        original_scope.startswith("default (") or
                        _canonical_scope(restored_scope) != _canonical_scope(original_scope)):
                    restore_error = (
                        f"Failed to restore {scope_path} to {original_scope!r}; "
                        f"wrote {restore_token!r}, read back {restored_scope!r}"
                    )
            except (IOError, OSError) as error:
                restore_error = (
                    f"Failed to restore {scope_path} to {original_scope!r}: {error}"
                )
            if restore_error:
                if failure_reason:
                    failure_reason = f"{failure_reason}; additionally, {restore_error}"
                else:
                    failure_reason = restore_error

    if failure_reason:
        GLTCore.GLTTestResult.fail(failure_reason)
