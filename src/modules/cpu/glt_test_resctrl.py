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

"""Check the x86 resctrl filesystem ABI and an exclusively owned control group.

Requires root, allocation support and operationally exclusive resctrl use.
This observes filesystem integration, not CAT/MBA efficacy or source relocation.
Parser source: https://github.com/torvalds/linux/blob/7168ae330e8105296210b2b2d31d791d5c073345/fs/resctrl/ctrlmondata.c
"""

import errno
import os
from pathlib import Path
import platform
import uuid

from common import glt_core as GLTCore
from common import resctrl_utils
from scripts.check_kernel_config import parse_kconfig

CPUINFO_PATH = Path("/proc/cpuinfo")
INVALID_SCHEMA = b"GLT_INVALID_RESOURCE:0=1\n"


def _resctrl_platform_support() -> tuple[bool, str]:
    """Return applicability facts; propagate unexpected evidence-read errors."""
    machine = platform.machine()
    if machine != "x86_64":
        return False, f"Requires x86_64 allocation support; architecture is {machine}"
    config_path = f"/boot/config-{platform.release()}"
    try:
        config = parse_kconfig(config_path)
    except FileNotFoundError:
        return False, f"Missing running-kernel config {config_path}"
    for name in ("CONFIG_X86_CPU_RESCTRL", "CONFIG_RESCTRL_FS"):
        value = config.get(name, "absent-or-unset")
        if value != "y":
            return False, f"{name}={value} in parsed configuration {config_path}; requires y"
    for line in CPUINFO_PATH.read_text(encoding="ascii").splitlines():
        name, separator, value = line.partition(":")
        if name.strip() == "flags" and separator and value.split():
            if "rdt_a" not in value.split():
                return False, f"CPU flags in {CPUINFO_PATH} do not advertise rdt_a"
            return True, ""
    raise ValueError(f"No parseable CPU flags line in {CPUINFO_PATH}")


def _control_group_capacity(root: Path) -> int:
    """Read the allocation ABI and return its minimum positive CLOSID capacity."""
    # Readable root ABI nodes prevent a schema-only PASS on an incomplete mount.
    list((root / "info").iterdir())
    schema = (root / "schemata").read_bytes()
    if not schema.strip():
        raise ValueError(f"Empty allocation schema at {root / 'schemata'}")
    (root / "tasks").read_bytes()
    (root / "cpus").read_bytes()
    # Root plus one group needs two CLOSIDs; other allocation failures remain unexpected.
    capacities = []
    for line in schema.decode("ascii").splitlines():
        if not line.strip():
            continue
        name, separator, _ = line.partition(":")
        name = name.strip()
        if not separator or not name:
            raise ValueError(f"Malformed allocation schema line: {line!r}")
        path = root / "info" / name / "num_closids"
        capacity = int(path.read_text(encoding="ascii").strip())
        if capacity < 1:
            raise ValueError(f"Nonpositive control-group capacity at {path}: {capacity}")
        capacities.append(capacity)
    return min(capacities)


def _write_schema_once(path: Path, payload: bytes) -> None:
    """Write one complete transaction to an existing kernel schema file."""
    descriptor = os.open(path, os.O_WRONLY)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise ValueError(f"Short schema write to {path}: {written}/{len(payload)} bytes")
    finally:
        os.close(descriptor)


# Keep ownership, assertions and final cleanup together so verdict precedence stays visible.
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_resctrl_filesystem_group_lifecycle(_, logger):  # pylint: disable=too-many-branches
    """Check raw schema roundtrip, invalid-resource rejection and owned cleanup.

    Requires root, x86_64 allocation support and exclusive, unmounted resctrl.
    Changes only one newly created group and a test-owned mount. The mount
    precheck is not a lock. This does not prove CAT/MBA efficacy or source location.
    Nonempty group schemata must round-trip byte-for-byte. Writing an unknown
    resource must return EINVAL and leave the group schemata unchanged.
    """
    try:
        supported, reason = _resctrl_platform_support()
    except (OSError, ValueError) as error:
        GLTCore.GLTTestResult.fail(f"Cannot determine resctrl platform support: {error}")
    if not supported:
        GLTCore.GLTTestResult.skip(reason)
    try:
        registered = resctrl_utils.fs_supported()
        existing_mounts = resctrl_utils.mount_points()
    except OSError as error:
        GLTCore.GLTTestResult.fail(f"Cannot read resctrl filesystem/mount evidence: {error}")
    if not registered:
        GLTCore.GLTTestResult.fail("Eligible kernel did not register the resctrl filesystem")
    if existing_mounts:
        GLTCore.GLTTestResult.skip(f"Requires exclusive unmounted resctrl; existing mounts: {existing_mounts}")
    root = Path(resctrl_utils.RESCTRL_MOUNT_POINT)
    cleanup_errors = []
    # This text only preserves primary diagnostics if cleanup also fails.
    primary_failure = ""
    deferred_skip = None
    try:
        with resctrl_utils.managed_mount(logger, cleanup_errors=cleanup_errors) as state:
            mounted, missing_cli = state
            if missing_cli:
                deferred_skip = f"Required resctrl CLI unavailable: {missing_cli}"
                return  # The finally block emits SKIP only after all cleanup checks.
            if not mounted:
                primary_failure = f"Cannot mount and confirm resctrl at {root}"
                GLTCore.GLTTestResult.fail(primary_failure)
            capacity = _control_group_capacity(root)
            if capacity < 2:
                deferred_skip = f"Control-group capacity is {capacity} CLOSID; root plus one group requires 2"
                return  # The finally block emits SKIP only after all cleanup checks.
            group_path = root / f"glt_resctrl_{uuid.uuid4().hex}"
            group_path.mkdir()
            try:
                schema_path = group_path / "schemata"
                schema = schema_path.read_bytes()
                if not schema.strip() or not schema.endswith(b"\n"):
                    primary_failure = f"Expected nonempty newline-terminated schema at {schema_path}"
                    GLTCore.GLTTestResult.fail(primary_failure)
                # Require group ABI readability beyond schema; empty task lists and masks are valid.
                (group_path / "tasks").read_bytes()
                (group_path / "cpus").read_bytes()
                _write_schema_once(schema_path, schema)
                if schema_path.read_bytes() != schema:
                    primary_failure = f"Schema changed after valid roundtrip at {schema_path}"
                    GLTCore.GLTTestResult.fail(primary_failure)
                try:
                    _write_schema_once(schema_path, INVALID_SCHEMA)
                except OSError as error:
                    if error.errno != errno.EINVAL:
                        primary_failure = (
                            f"Writing invalid resource to {schema_path} failed with {error}; "
                            "expected EINVAL")
                        GLTCore.GLTTestResult.fail(primary_failure)
                else:
                    primary_failure = f"Invalid resource accepted at {schema_path}; expected EINVAL"
                    GLTCore.GLTTestResult.fail(primary_failure)
                if schema_path.read_bytes() != schema:
                    primary_failure = f"Schema changed after invalid-resource rejection at {schema_path}"
                    GLTCore.GLTTestResult.fail(primary_failure)
            finally:
                try:
                    group_path.rmdir()
                except OSError as error:
                    cleanup_errors.append(f"Cannot remove owned resctrl group {group_path}: {error}")
    except (OSError, ValueError, RuntimeError) as error:
        # A manager-exit operation error must not be overridden by a pending SKIP.
        deferred_skip = None
        primary_failure = f"resctrl filesystem/group transaction failed: {error}"
        GLTCore.GLTTestResult.fail(primary_failure)
    finally:
        try:
            remaining = resctrl_utils.mount_points()
        except OSError as error:
            cleanup_errors.append(f"Cannot verify final resctrl mount state: {error}")
        else:
            if remaining:
                cleanup_errors.append(f"Residual resctrl mounts after manager exit: {remaining}")
        if cleanup_errors:
            GLTCore.GLTTestResult.fail("; ".join(filter(None, [primary_failure, *cleanup_errors])))
        if deferred_skip:
            GLTCore.GLTTestResult.skip(deferred_skip)
