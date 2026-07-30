# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test memory zone properties and the longterm GUP pin path.
"""
import ctypes
import errno
import fcntl
import mmap
import os
import re
import struct
from common import glt_core as GLTCore


def _parse_zoneinfo(logger):
    """Parse /proc/zoneinfo and return a list of zone dicts.

    Each dict has keys: node (int), zone (str), min (int|None), managed (int|None).
    """
    zones = []
    current = {}

    try:
        with open("/proc/zoneinfo", "r", encoding="utf-8") as f:
            for line in f:
                # Zone header line: "Node 0, zone      DMA"
                # Starts a new zone block; save the previous one and skip
                # to the next line (header itself has no watermark data).
                m = re.match(r"Node\s+(\d+),\s+zone\s+(\S+)", line)
                if m:
                    if current:
                        zones.append(current)
                    current = {
                        "node": int(m.group(1)),
                        "zone": m.group(2),
                        "min": None,
                        "managed": None,
                    }
                    continue

                # Non-header lines: extract watermark fields from the
                # indented key-value pairs inside the current zone block.
                stripped = line.strip()
                parts = stripped.split()
                if len(parts) == 2:
                    if parts[0] in ("min", "managed"):
                        try:
                            current[parts[0]] = int(parts[1])
                        except ValueError:
                            logger.error(
                                "Non-integer '%s' value in /proc/zoneinfo: %r",
                                parts[0], parts[1],
                            )
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"Failed to parse /proc/zoneinfo: {e}")
        return None

    # the last zone in the file.
    if current:
        zones.append(current)

    return zones


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_zone_movable_min_watermark(_, logger):
    """
    Test if ZONE_MOVABLE min watermark is properly capped.

    With the patch, ZONE_MOVABLE is treated like ZONE_HIGHMEM in
    __setup_per_zone_wmarks(): its min watermark is capped via
    clamp(managed/1024, 32, 128) instead of receiving a proportional
    share of min_free_kbytes.

    JIRA: DGX-9464
    PATCH: https://lore.kernel.org/all/20230625031656.23941-1-liuq131@chinatelecom.cn/T/#u
    """
    # min_pages = clamp(zone_managed_pages(zone) / 1024, SWAP_CLUSTER_MAX, 128)
    swap_cluster_max = 32
    wmark_min_cap = 128
    managed_threshold = wmark_min_cap * 1024

    zones = _parse_zoneinfo(logger)
    if zones is None:
        GLTCore.GLTTestResult.skip(
            "Could not read /proc/zoneinfo. Cannot validate ZONE_MOVABLE "
            "watermarks.")

    movable = [z for z in zones if z["zone"] == "Movable"]
    if not movable:
        GLTCore.GLTTestResult.skip("No ZONE_MOVABLE zones present.")

    qualifying = [z for z in movable
                  if z["min"] is not None
                  and z["managed"] is not None
                  and z["managed"] > managed_threshold]
    if not qualifying:
        GLTCore.GLTTestResult.skip(
            f"No ZONE_MOVABLE zones with managed > {managed_threshold} pages.")

    failures = []
    for z in qualifying:
        node = z["node"]
        wmark_min = z["min"]

        if wmark_min < swap_cluster_max or wmark_min > wmark_min_cap:
            failures.append(
                f"NUMA node {node}: min={wmark_min} "
                f"(expected [{swap_cluster_max}..{wmark_min_cap}])")

    if failures:
        GLTCore.GLTTestResult.fail(
            "ZONE_MOVABLE min watermark not capped. "
            "Consider backporting commit 416ef04 to fix this issue. "
            "If min_free_kbytes was manually adjusted, "
            "this failure can be ignored. "
            "Details: " + "; ".join(failures))


GUP_TEST_PATH = "/sys/kernel/debug/gup_test"


def _max_page_order(logger):
    """Return MAX_PAGE_ORDER, or None when it cannot be determined.

    /proc/pagetypeinfo prints one free-count column per order,
    so the last order named in that header line is it.
    """
    pagetypeinfo_path = "/proc/pagetypeinfo"
    try:
        with open(pagetypeinfo_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Free pages count per migrate type at order" not in line:
                    continue
                orders = re.findall(r"\d+", line.split("order", 1)[1])
                if orders:
                    return int(orders[-1])
    except OSError as e:
        logger.error("Failed to read %s: %s", pagetypeinfo_path, e)
    return None


def _pin_longterm_ioctl_codes():
    """Return (START, STOP, plt_size) for gup_test longterm pin ioctls."""
    # struct pin_longterm_test: two u64 + u32, 8-byte aligned → 24 bytes.
    plt_size = struct.calcsize("=QQI4x")
    # _IOW('g', 7, struct pin_longterm_test) / PIN_LONGTERM_TEST_START.
    start_req = (1 << 30) | (plt_size << 16) | (ord("g") << 8) | 7
    # _IO('g', 8) / PIN_LONGTERM_TEST_STOP.
    stop_req = (ord("g") << 8) | 8
    return start_req, stop_req, plt_size


def _try_longterm_pin(vaddr, length, logger):
    """PIN_LONGTERM_TEST_START over [vaddr, vaddr+length), then STOP.

    Returns a (status, detail) tuple:
      ("ok", "") on full start/stop success
      ("gup_lock_failed", message) when the shared lock is unavailable
      ("gup_open_failed", message) when opening gup_test fails
      ("start_errno", errno) when PIN_LONGTERM_TEST_START fails
      ("stop_failed", message) when STOP fails after a successful START

    gup_test keeps long-term pin state in globals, so any client can block
    another's START with EINVAL, and release() on any gup_test fd stops
    whichever pin is active. The lock below mitigates that and must be honored
    by every gup_test consumer. It is acquired before opening gup_test because
    merely closing that fd can stop the pin. The caller must already have
    mapped and faulted in the range.
    """
    start_req, stop_req, plt_size = _pin_longterm_ioctl_codes()
    args = bytearray(plt_size)
    struct.pack_into(
        "=QQI4x", args, 0,
        vaddr, length,
        0x1,  # PIN_LONGTERM_TEST_FLAG_USE_WRITE
    )

    # Removable once DUT kernels carry the upstream patch being submitted,
    # "mm/gup_test: Track long-term pins per open file", which moves pin state
    # into file->private_data so a pin is owned by the fd that created it.
    lock_path = "/run/lock/gup_test.lock"
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if lock_fd is not None:
            os.close(lock_fd)
        return ("gup_lock_failed",
                f"Could not acquire shared gup_test lock {lock_path}: {e}")

    try:
        try:
            gfd = os.open(GUP_TEST_PATH, os.O_RDWR)
        except OSError as e:
            return ("gup_open_failed",
                    f"Could not open {GUP_TEST_PATH}: {e}")

        pinned = False
        try:
            try:
                fcntl.ioctl(gfd, start_req, args)
            except OSError as e:
                logger.error(
                    "PIN_LONGTERM_TEST_START failed for %d bytes: %s",
                    length, e)
                return ("start_errno", e.errno)
            pinned = True
            try:
                fcntl.ioctl(gfd, stop_req)
            except OSError as e:
                return ("stop_failed",
                        "PIN_LONGTERM_TEST_STOP failed after pin: "
                        + os.strerror(e.errno))
            pinned = False
            return ("ok", "")
        finally:
            if pinned:
                try:
                    fcntl.ioctl(gfd, stop_req)
                except OSError as e:
                    logger.error(
                        "PIN_LONGTERM_TEST_STOP during cleanup: %s", e)
            os.close(gfd)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _gup_longterm_preflight(page_size, logger):
    """Check the longterm-pin ioctl ABI with one faulted page."""
    try:
        with mmap.mmap(
                -1, page_size,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as mapping:
            mapping[0] = 1
            vaddr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            status, detail = _try_longterm_pin(vaddr, page_size, logger)
    except (OSError, OverflowError, ValueError) as e:
        return ("skip",
                f"Could not run gup_test longterm-pin ABI preflight: {e}")

    if status == "ok":
        return ("ok", "")
    if status in ("gup_lock_failed", "gup_open_failed"):
        return ("skip", detail)
    if status == "start_errno" and detail in (errno.ENOTTY, errno.EINVAL):
        return ("skip",
                "gup_test lacks the required longterm-pin ioctl ABI, or "
                "another global longterm test is already active.")
    return ("warn",
            f"gup_test longterm-pin ABI preflight failed: "
            f"{status!r}, {detail!r}")


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("stress")
def test_gup_longterm_pin_past_kmalloc_limit(_, logger):  # pylint: disable=too-many-branches
    """Detect missing FOLL_LONGTERM folio-array allocation avoidance.

    Pin the same mapping as a just-below-limit prefix and a two-page tail
    before pinning it whole. If both controls succeed, every page is pinnable
    and the caller's large page array can be allocated. A whole-range ENOMEM
    is then reported fail-closed: it matches the internal folio* kmalloc
    ceiling, but the ABI cannot prove which allocation failed.

    Uses ordinary anonymous memory so folios stay longterm-pinnable.

    Each ioctl lifecycle holds /run/lock/gup_test.lock, which every gup_test
    consumer on the DUT must honor. Kernel-side state is global, and release()
    on any gup_test fd stops the global pin.

    JIRA: DGX-12728
    PATCH: https://github.com/torvalds/linux/commit/94efde1d
    """
    if not os.path.exists(GUP_TEST_PATH):
        GLTCore.GLTTestResult.skip(
            f"{GUP_TEST_PATH} is missing. CONFIG_GUP_TEST=y and debugfs are "
            "required to request a FOLL_LONGTERM pin from userspace.")

    max_page_order = _max_page_order(logger)
    if max_page_order is None:
        GLTCore.GLTTestResult.skip(
            "Could not determine MAX_PAGE_ORDER.")

    page_size = os.sysconf("SC_PAGE_SIZE")
    if page_size != 4096:
        GLTCore.GLTTestResult.skip(
            f"PAGE_SIZE is {page_size}, not 4096; the ~2GiB folio-kmalloc "
            "pin ceiling this test targets is specific to 4KB pages.")

    preflight, detail = _gup_longterm_preflight(page_size, logger)
    if preflight == "skip":
        GLTCore.GLTTestResult.skip(detail)
    if preflight == "warn":
        GLTCore.GLTTestResult.warn(detail)

    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    # folio* kmalloc_array ceiling ≈ PAGE_SIZE << MAX_PAGE_ORDER (94efde1d1539).
    limit_pages = (page_size << max_page_order) // ptr_size
    if limit_pages < 2:
        GLTCore.GLTTestResult.skip(
            f"Computed folio-kmalloc limit is only {limit_pages} pages.")

    # One page past the kmalloc folio* array ceiling.
    pin_bytes = (limit_pages + 1) * page_size

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) * 1024
                    break
            else:
                raise ValueError("MemAvailable not found")
            # Headroom so the fault-in is not starved by unrelated reclaim.
            need = pin_bytes + 512 * 1024 * 1024
            if mem_available < need:
                raise ValueError(
                    f"MemAvailable {mem_available} bytes is below the "
                    f"{need} bytes required for the pin "
                    f"({pin_bytes} bytes + margin)")

        # MemAvailable is host-wide. Honor finite cgroup-v2 limits before
        # faulting the mapping so a memory-capped container is not OOM-killed.
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            cgroup_path = next(
                (line.strip().split("::", 1)[1]
                 for line in f if line.startswith("0::")),
                None)
        if cgroup_path is not None:
            cgroup_root = "/sys/fs/cgroup"
            cgroup_dir = os.path.normpath(
                os.path.join(cgroup_root, cgroup_path.lstrip("/")))
            while cgroup_dir.startswith(cgroup_root):
                max_path = os.path.join(cgroup_dir, "memory.max")
                current_path = os.path.join(cgroup_dir, "memory.current")
                if os.path.isfile(max_path):
                    with open(max_path, "r", encoding="utf-8") as f:
                        memory_max = f.read().strip()
                    if memory_max != "max":
                        with open(current_path, "r", encoding="utf-8") as f:
                            memory_current = int(f.read().strip())
                        cgroup_available = int(memory_max) - memory_current
                        if cgroup_available < need:
                            raise ValueError(
                                f"cgroup has {cgroup_available} bytes "
                                f"available, below the {need} bytes required")
                if cgroup_dir == cgroup_root:
                    break
                cgroup_dir = os.path.dirname(cgroup_dir)
    except (OSError, ValueError, IndexError) as e:
        GLTCore.GLTTestResult.skip(
            f"Cannot size the FOLL_LONGTERM pin: {e}")

    try:
        with mmap.mmap(
                -1, pin_bytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as mapping:
            # One store per page faults it in without building a pin_bytes-sized
            # temporary buffer.
            for offset in range(0, pin_bytes, page_size):
                mapping[offset] = 1
            vaddr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))

            # Prove caller allocation and migration for every page without
            # crossing the internal folio* kmalloc_array ceiling.
            control_bytes = (limit_pages - 1) * page_size
            controls = (
                (vaddr, control_bytes),
                (vaddr + control_bytes, pin_bytes - control_bytes),
            )
            for control_addr, control_length in controls:
                status, detail = _try_longterm_pin(
                    control_addr, control_length, logger)
                if status in ("gup_lock_failed", "gup_open_failed"):
                    GLTCore.GLTTestResult.skip(detail)
                if status != "ok":
                    GLTCore.GLTTestResult.warn(
                        f"FOLL_LONGTERM control pin failed: "
                        f"{status!r}, {detail!r}. Cannot attribute a "
                        "whole-range ENOMEM to the folio* allocation.")

            status, detail = _try_longterm_pin(vaddr, pin_bytes, logger)

            if status == "ok":
                # Pin succeeded: redundant folio kmalloc is not blocking.
                return
            if status in ("gup_lock_failed", "gup_open_failed"):
                GLTCore.GLTTestResult.skip(detail)
            elif status == "stop_failed":
                # START succeeded, so the folio-kmalloc bug is not present;
                # STOP failing is an unrelated cleanup problem.
                GLTCore.GLTTestResult.warn(detail)
            elif status == "start_errno":
                if detail in (errno.ENOTTY, errno.EINVAL):
                    GLTCore.GLTTestResult.skip(
                        "gup_test ioctl ABI mismatch on FOLL_LONGTERM pin.")
                elif detail == errno.ENOMEM:
                    GLTCore.GLTTestResult.fail(
                        f"FOLL_LONGTERM pin of {pin_bytes} bytes returned "
                        "ENOMEM after prefix and tail controls succeeded. "
                        "This is an inconclusive ENOMEM result: userspace "
                        "cannot distinguish the folio-array allocation from "
                        "gup_test page-vector allocation, longterm migration, "
                        "or memory pressure. If it reproduces, consider "
                        "backporting commit 94efde1d1539 "
                        '("mm/gup: avoid an unnecessary allocation call '
                        'for FOLL_LONGTERM cases").')
                else:
                    GLTCore.GLTTestResult.warn(
                        f"FOLL_LONGTERM pin of {pin_bytes} bytes failed "
                        f"with errno {detail} ({os.strerror(detail)}), "
                        "not the ENOMEM signature of a missing "
                        "folio-kmalloc avoidance fix.")
            else:
                GLTCore.GLTTestResult.warn(
                    f"Unexpected longterm-pin status {status!r}: {detail!r}")
    except (OSError, OverflowError, ValueError) as e:
        GLTCore.GLTTestResult.skip(
            f"Could not mmap/fault-in {pin_bytes} bytes for the "
            f"FOLL_LONGTERM pin: {e}")
