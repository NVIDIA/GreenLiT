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

"""Validate Vera anonymous mTHP contpte consumer behavior."""

import ctypes
import errno
import fcntl
import os
import platform
import zlib
from pathlib import Path

from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev

MTHP_MODULE = "mthp_contpte_probe"
MTHP_MODULE_PATH = Path(__file__).resolve().parent / "mthp_contpte"
MTHP_MODULE_STATE_PATH = Path("/sys/module") / MTHP_MODULE
MTHP_DEVICE = "/dev/glt_mthp_contpte"
THP_SYSFS_ROOT = Path("/sys/kernel/mm/transparent_hugepage")
PROT_READ = 1
PROT_WRITE = 2
MAP_PRIVATE = 2
MAP_FIXED = 0x10
MAP_ANONYMOUS = 0x20
MADV_HUGEPAGE = 14


class MthpContpteResult(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("mapped_pages", ctypes.c_uint32),
        ("cont_pages", ctypes.c_uint32),
        ("pfn_contiguous", ctypes.c_uint32),
        ("page_size", ctypes.c_uint32),
        ("cont_pte_size", ctypes.c_uint32),
        ("cont_ptes", ctypes.c_uint32),
        ("large_folio", ctypes.c_uint32),
        ("folio_order", ctypes.c_uint32),
    ]


MTHP_CONTPTE_IOCTL = (
    (3 << 30)
    | (ctypes.sizeof(MthpContpteResult) << 16)
    | (ord("G") << 8)
    | 0x60
)


class MthpSkip(Exception):
    pass


class MthpMapping:
    def __init__(self, size, page_size):
        self.size = size
        self.page_size = page_size
        self.address = None
        self.reserve_address = None
        self.reserve_size = size * 2
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_long,
        ]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

        reserve = self.libc.mmap(
            None, self.reserve_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0,
        )
        if reserve == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mTHP reservation mmap failed")
        self.reserve_address = reserve
        self.address = (reserve + size - 1) & ~(size - 1)
        mapped = self.libc.mmap(
            self.address, size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0,
        )
        if mapped != self.address:
            self.close()
            raise OSError(ctypes.get_errno(), "aligned mTHP mmap failed")
        if self.libc.madvise(self.address, size, MADV_HUGEPAGE):
            self.close()
            raise OSError(ctypes.get_errno(), "MADV_HUGEPAGE failed")

    def fill(self, pattern):
        for offset in range(0, self.size, self.page_size):
            ctypes.memset(
                self.address + offset,
                (pattern + offset // self.page_size) % 251 + 1,
                self.page_size,
            )

    def verify(self, pattern):
        for offset in range(0, self.size, self.page_size):
            expected = bytes([
                (pattern + offset // self.page_size) % 251 + 1
            ]) * self.page_size
            actual = ctypes.string_at(self.address + offset, self.page_size)
            if actual != expected:
                raise ValueError(f"Data mismatch at mapping offset {offset}")

    def set_first_page_access(self, access):
        if self.libc.mprotect(self.address, self.page_size, access):
            raise OSError(ctypes.get_errno(), "mprotect on mTHP subpage failed")

    def close(self):
        if self.reserve_address is not None:
            address = self.reserve_address
            self.reserve_address = None
            self.address = None
            if self.libc.munmap(address, self.reserve_size):
                raise OSError(ctypes.get_errno(), "mTHP reservation munmap failed")


def _read_kernel_config_text():
    try:
        with open("/proc/config.gz", "rb") as config_file:
            return zlib.decompress(config_file.read(), 16 + zlib.MAX_WBITS).decode()
    except OSError:
        with open(f"/boot/config-{platform.release()}", encoding="utf-8") as config_file:
            return config_file.read()


def _mthp_candidates(root=THP_SYSFS_ROOT):
    root = Path(root)
    candidates = []
    for enabled_path in root.glob("hugepages-*kB/enabled"):
        enabled = enabled_path.read_text(encoding="utf-8").strip()
        if "[always]" not in enabled and "[madvise]" not in enabled:
            continue
        size = int(enabled_path.parent.name.removeprefix("hugepages-").removesuffix("kB")) * 1024
        stats_path = enabled_path.parent / "stats" / "anon_fault_alloc"
        if stats_path.is_file():
            candidates.append((size, stats_path))
    return sorted(candidates)


def _read_alloc_count(stats_path):
    return int(Path(stats_path).read_text(encoding="utf-8").strip())


def _evaluate_snapshot(result, initial):
    if result.cont_ptes < 2 or result.size != result.cont_pte_size:
        return "skip", "Mapping size does not match an architecture CONT block"
    if result.mapped_pages != result.cont_ptes:
        return (
            "skip",
            f"Only {result.mapped_pages}/{result.cont_ptes} pages were mapped; "
            "the page-table walk is incomplete",
        )
    if not result.large_folio or (1 << result.folio_order) != result.cont_ptes:
        return "fail", "Probe did not attribute the CONT block to one anonymous large folio"
    if not result.pfn_contiguous:
        return "fail", "mTHP CONT block does not have contiguous PFNs"
    if initial and result.cont_pages != result.cont_ptes:
        return (
            "fail",
            f"mTHP allocation succeeded but only {result.cont_pages}/{result.cont_ptes} "
            "PTEs are CONT; the contpte consumer was not used",
        )
    if not initial and result.cont_pages != 0:
        return "fail", f"{result.cont_pages} PTEs remain CONT after subpage mprotect; full unfold did not occur"
    return "pass", ""


def _probe_snapshot(device_fd, address, size):
    result = MthpContpteResult(addr=address, size=size)
    try:
        fcntl.ioctl(device_fd, MTHP_CONTPTE_IOCTL, result, True)
    except OSError as err:
        if err.errno in (errno.EOPNOTSUPP, errno.EAGAIN):
            raise MthpSkip(f"Probe could not obtain a stable supported snapshot: {err}") from err
        raise
    return result


def _exercise_candidate(device_fd, candidate, page_size):
    size, stats_path = candidate
    before = _read_alloc_count(stats_path)
    mapping = None
    try:
        mapping = MthpMapping(size, page_size)
        mapping.fill(1)
        if _read_alloc_count(stats_path) <= before:
            raise MthpSkip(f"No mTHP allocation was observed for {size} bytes")
        initial = _probe_snapshot(device_fd, mapping.address, size)
        outcome, reason = _evaluate_snapshot(initial, initial=True)
        if outcome == "skip":
            raise MthpSkip(reason)
        if outcome == "fail":
            raise ValueError(reason)

        mapping.set_first_page_access(PROT_READ)
        mapping.verify(1)
        unfolded = _probe_snapshot(device_fd, mapping.address, size)
        outcome, reason = _evaluate_snapshot(unfolded, initial=False)
        if outcome == "skip":
            raise MthpSkip(reason)
        if outcome == "fail":
            raise ValueError(reason or "mTHP unfold validation failed")

        mapping.set_first_page_access(PROT_READ | PROT_WRITE)
        mapping.fill(2)
        mapping.verify(2)
        return size
    finally:
        if mapping is not None:
            mapping.close()


def _run_mthp_probe(logger, module_dir=MTHP_MODULE_PATH,
                    module_state_path=MTHP_MODULE_STATE_PATH,
                    headers_root=Path("/lib/modules"), kernel_release=None,
                    device_path=MTHP_DEVICE):
    if kernel_release is None:
        kernel_release = platform.release()
    module_dir = Path(module_dir)
    module_state_path = Path(module_state_path)

    try:
        config_text = _read_kernel_config_text()
    except OSError as err:
        return "skip", f"Cannot inspect mTHP/module configuration: {err}"
    required = ("CONFIG_ARM64_CONTPTE=y", "CONFIG_MODULES=y", "CONFIG_MODULE_UNLOAD=y")
    missing = [name for name in required if name not in config_text]
    if missing:
        return "skip", "mTHP contpte requires " + ", ".join(missing)

    try:
        candidates = _mthp_candidates()
    except (OSError, ValueError) as err:
        return "skip", f"Cannot enumerate mTHP sizes: {err}"
    if not candidates:
        return "skip", "No mTHP size with always/madvise and allocation statistics is enabled"

    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return "skip", f"{command} is required for the mTHP contpte probe"
    headers = Path(headers_root) / kernel_release / "build"
    if not headers.is_dir():
        return "skip", f"Matching kernel headers are unavailable: {headers}"
    if not module_dir.is_dir():
        return "skip", f"mTHP probe sources are missing: {module_dir}"
    if module_state_path.exists():
        return "skip", f"{MTHP_MODULE} is already loaded; refusing an unknown instance"

    loaded_by_test = False
    outcome = "pass"
    reason = ""
    cleanup_errors = []
    device_fd = None

    with glt_utils.chdir_managed(module_dir):
        try:
            try:
                glt_utils.run_command(["make"], "Cannot build mTHP contpte probe", timeout=120)
            except RuntimeError as err:
                outcome = "skip"
                reason = str(err)
            if outcome == "pass" and not glt_utils.sign_kernel_module(f"{MTHP_MODULE}.ko", logger):
                outcome = "fail"
                reason = f"Cannot sign {MTHP_MODULE}.ko for Secure Boot"
            if outcome == "pass":
                try:
                    glt_utils.run_command(
                        ["insmod", f"{MTHP_MODULE}.ko"],
                        "Cannot load mTHP contpte probe", timeout=30,
                    )
                    loaded_by_test = True
                except RuntimeError as err:
                    if module_state_path.exists():
                        outcome = "fail"
                        reason = f"{err}; module ownership is unknown; manual cleanup required"
                    else:
                        outcome = "skip"
                        reason = str(err)
            if outcome == "pass":
                try:
                    device_fd = os.open(device_path, os.O_RDWR)
                except OSError as err:
                    outcome = "skip"
                    reason = f"Cannot open {device_path}: {err}"
            if outcome == "pass":
                skip_reasons = []
                try:
                    for candidate in candidates:
                        try:
                            size = _exercise_candidate(
                                device_fd, candidate, os.sysconf("SC_PAGE_SIZE")
                            )
                            outcome = "pass"
                            reason = f"mTHP contpte allocation/unfold and data integrity passed for {size} bytes"
                            break
                        except MthpSkip as err:
                            skip_reasons.append(str(err))
                    else:
                        outcome = "skip"
                        reason = "No usable mTHP candidate: " + "; ".join(skip_reasons)
                except (OSError, RuntimeError, ValueError) as err:
                    outcome = "fail"
                    reason = f"mTHP contpte validation failed: {err}"
        finally:
            if device_fd is not None:
                try:
                    os.close(device_fd)
                except OSError as err:
                    cleanup_errors.append(str(err))
            if loaded_by_test:
                try:
                    glt_utils.run_command(
                        ["rmmod", MTHP_MODULE], "Failed to unload mTHP contpte probe", timeout=30,
                    )
                except RuntimeError as err:
                    cleanup_errors.append(str(err))
            try:
                glt_utils.run_command(
                    ["make", "clean"], "Failed to clean mTHP contpte probe", timeout=120,
                )
            except RuntimeError as err:
                cleanup_errors.append(str(err))

    if cleanup_errors:
        cleanup_reason = "; ".join(cleanup_errors)
        if outcome == "fail":
            reason = f"{reason}; cleanup also failed: {cleanup_reason}"
        else:
            outcome = "fail"
            reason = f"mTHP probe cleanup failed: {cleanup_reason}"
    return outcome, reason


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_vera_mthp_contpte_consumer(_, logger):
    """Verify real Vera mTHP CONT allocation and subpage unfold behavior."""
    if platform.machine() != "aarch64":
        GLTCore.GLTTestResult.skip("mTHP contpte validation is arm64-specific")
    if not NvDev().is_cpu_type("vera"):
        GLTCore.GLTTestResult.skip("mTHP contpte validation is Vera-specific")
    outcome, reason = _run_mthp_probe(logger)
    if outcome == "fail":
        GLTCore.GLTTestResult.fail(reason)
    if outcome == "skip":
        GLTCore.GLTTestResult.skip(reason)
    logger.info("%s", reason)
