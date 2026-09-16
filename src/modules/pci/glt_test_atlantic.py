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

"""Runtime diagnostics for the Aquantia Atlantic network driver."""

import json
import re
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils


ATLANTIC_DRIVER = "atlantic"
PCI_DEVICES_PATH = Path("/sys/bus/pci/devices")
MIN_RETAINED_BOOT_COUNT_FOR_HISTORY = 20
PCI_BDF_RE = re.compile(
    r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$"
)

# The upstream report places the faulting RIP 22 lines before ``Call Trace:``
# because a register dump separates them. Keep the pre-trace context bounded,
# but large enough to retain that production failure shape.
_FAILURE_CONTEXT_BEFORE_LINES = 64
_FAILURE_TRACE_AFTER_LINES = 80
_SKB_RX_FRAG_FRAME_RE = re.compile(
    r"\bskb_add_rx_frag(?:_netmem)?\b"
)
_ATLANTIC_RX_CLEAN_FRAME_RE = re.compile(
    r"^\s*(?:kernel:\s*)?(?:\?\s+)?"
    r"(?:__)?aq_ring_rx_clean\+0x[0-9a-fA-F]+/0x[0-9a-fA-F]+"
    r"(?:\s+\[\s*atlantic\s*\])?\s*$",
    re.MULTILINE,
)
_CALL_TRACE_RE = re.compile(r"\bCall [Tt]race:")
_TRACE_END_RE = re.compile(r"---\[ end trace|</(?:IRQ|TASK)>")


def _get_kernel_release_for_boot(boot_id):
    """Best-effort kernel release for one retained boot."""
    try:
        result = glt_utils.run_command(
            [
                "journalctl",
                "--no-pager",
                f"--boot={boot_id}",
                "-k",
                "-n1",
                "-o",
                "json",
            ],
            f"Failed to retrieve kernel metadata for retained boot {boot_id}",
        )
        record = json.loads(result.stdout)
    except (RuntimeError, json.JSONDecodeError):
        return "unknown"

    kernel_release = record.get("_KERNEL_RELEASE")
    if isinstance(kernel_release, str) and kernel_release:
        return kernel_release
    return "unknown"


def _find_bound_atlantic_devices(pci_devices_path=PCI_DEVICES_PATH):
    """Return PCI BDFs bound to Atlantic, surfacing discovery errors."""
    try:
        device_paths = sorted(pci_devices_path.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise RuntimeError(
            f"Cannot enumerate PCI devices at {pci_devices_path}: {error}"
        ) from error

    devices = []
    for device_path in device_paths:
        if not PCI_BDF_RE.fullmatch(device_path.name):
            continue

        driver_path = device_path / "driver"
        try:
            driver = driver_path.resolve(strict=True).name
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"Cannot resolve PCI driver for {device_path.name}: {error}"
            ) from error

        if driver == ATLANTIC_DRIVER:
            devices.append(device_path.name)

    return devices


def _iter_call_trace_blocks(lines):
    """Yield bounded call-trace blocks with nearby fault context."""
    for index, line in enumerate(lines):
        if not _CALL_TRACE_RE.search(line):
            continue

        start = max(0, index - _FAILURE_CONTEXT_BEFORE_LINES)
        for before_index in range(index - 1, start - 1, -1):
            if (
                _CALL_TRACE_RE.search(lines[before_index])
                or _TRACE_END_RE.search(lines[before_index])
            ):
                start = before_index + 1
                break

        end = min(len(lines), index + _FAILURE_TRACE_AFTER_LINES)
        for after_index in range(index + 1, end):
            if (
                _CALL_TRACE_RE.search(lines[after_index])
                or _TRACE_END_RE.search(lines[after_index])
            ):
                end = after_index + 1
                break

        yield "\n".join(lines[start:end])


def _has_atlantic_rx_fragment_overflow_failure(kernel_log):
    """Return whether the log contains the known Atlantic RX failure."""
    lines = kernel_log.splitlines()
    for trace_block in _iter_call_trace_blocks(lines):
        if (
            _SKB_RX_FRAG_FRAME_RE.search(trace_block)
            and _ATLANTIC_RX_CLEAN_FRAME_RE.search(trace_block)
        ):
            return True

    return False


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_atlantic_rx_fragment_overflow_failure_retained_journals(_, logger):
    """
    Check retained kernel journals for the known Atlantic RX fragment overflow crash.

    This is a passive runtime diagnostic. A clean result means only that the
    failure signature was not observed in readable retained boot logs; it is
    not proof that the RX fragment guard patch is present or that the
    exceptional RX descriptor condition was exercised.

    JIRA: DGX-17293
    PATCH: upstream 5ffcb7b890f6, Ubuntu backport 506fa5a0d30a
    """
    try:
        journal_boots = glt_utils.list_journal_boots()
    except RuntimeError as error:
        GLTCore.GLTTestResult.skip(
            "Cannot enumerate retained journal boots for Atlantic RX failure "
            f"diagnostic: {error}"
        )

    readable_boots = []
    unreadable_boots = []
    for boot in journal_boots:
        try:
            kernel_log = glt_utils.get_kernel_log(boot_id=boot.boot_id)
        except RuntimeError as error:
            unreadable_boots.append((boot, str(error)))
            continue

        readable_boots.append(boot)
        if _has_atlantic_rx_fragment_overflow_failure(kernel_log):
            kernel_release = _get_kernel_release_for_boot(boot.boot_id)
            GLTCore.GLTTestResult.fail(
                "Known Atlantic RX fragment-overflow failure signature "
                "observed in retained boot "
                f"{boot.boot_id} (index {boot.index}, kernel "
                f"{kernel_release}). Check whether upstream 5ffcb7b890f6 or "
                "Ubuntu backport 506fa5a0d30a is missing from that boot's "
                "kernel. This is a failure observation only; it does not "
                "prove the current running kernel's patch level."
            )

    if not readable_boots:
        GLTCore.GLTTestResult.skip(
            "No retained boot kernel logs are readable; Atlantic RX failure "
            "diagnostic cannot determine whether the failure was observed."
        )

    try:
        devices = _find_bound_atlantic_devices()
    except RuntimeError as error:
        GLTCore.GLTTestResult.skip(
            "Cannot determine whether any PCI device is bound to the "
            "atlantic driver after no retained failure signature was "
            f"observed: {error}"
        )

    if not devices:
        GLTCore.GLTTestResult.skip(
            "No PCI device is currently bound to the atlantic driver, and no "
            "retained Atlantic RX fragment-overflow failure signature was "
            "observed."
        )

    if unreadable_boots:
        unreadable_ids = ", ".join(
            boot.boot_id for boot, _ in unreadable_boots[:5]
        )
        logger.warning(
            "Unable to read kernel logs for %d of %d retained journal "
            "boot(s); those boot(s) were not counted as clean. First "
            "unreadable boot ID(s): %s.",
            len(unreadable_boots),
            len(journal_boots),
            unreadable_ids,
        )

    if len(readable_boots) < MIN_RETAINED_BOOT_COUNT_FOR_HISTORY:
        logger.warning(
            "Only %d readable retained boot journal(s) were checked; this is "
            "a limited observation window, not proof of patch presence.",
            len(readable_boots),
        )

    logger.info(
        "Known Atlantic RX fragment-overflow failure signature was not "
        "observed in %d readable retained boot journal(s) out of %d listed "
        "retained boot(s) for atlantic-bound device(s): %s. This does not "
        "prove upstream 5ffcb7b890f6 or Ubuntu backport 506fa5a0d30a is "
        "present, nor that the exceptional RX condition was exercised.",
        len(readable_boots),
        len(journal_boots),
        ", ".join(devices),
    )
