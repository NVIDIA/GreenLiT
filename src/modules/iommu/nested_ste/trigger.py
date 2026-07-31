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
Pure-Python port of the former nest_ste_trig C helper.

Deterministically exercises nested SMMUv3 STE transitions between
S1DSS-bypass (cfg=S1_TRANS) and S2-bypass (cfg=BYPASS) via iommufd/vfio
ioctls, so ``observe_ste.bt`` can watch arm_smmu_write_entry() for
disruptive (STE.V=0) rewrites. Every S1DSS<->S2bypass attach/replace also
carries an EATS_TRANS toggle simultaneous with the CFG change (see
_ste_make_s1dss_bypass / _ste_make_nested_s2_bypass), so the sequence
exercises both the MEV and EATS halves of the update-safety fix.

Ioctl argument layouts mirror ``include/uapi/linux/iommufd.h`` and
``include/uapi/linux/vfio.h`` as of the update_safe series tip
https://git.kernel.org/torvalds/c/7cad80048595. Flat structs are packed
with ``struct.Struct``; the ARM SMMUv3 nested payloads that need a stable
pointer for ``data_uptr`` remain ctypes. ABI sizes are checked in unit
tests (not at import time) so a mismatch cannot abort GreenLiT suite
discovery. Host endianness is gated via a test skip helper.

The ioctl request numbers all use the plain ``_IO(type, nr)`` form: under
``include/uapi/asm-generic/ioctl.h`` that macro expands to
``_IOC(_IOC_NONE, type, nr, 0)``, and with direction=NONE and size=0 the
encoding collapses to ``(type << 8) | nr`` -- there is no direction/size
payload to get wrong here, unlike ``_IOW``/``_IOR``/``_IOWR``.
"""

import ctypes
import fcntl
import mmap
import os
import re
import struct
import threading

# --------------------------------------------------------------------------
# comm/thread naming (must stay in sync with observe_ste.bt's `comm == ...`)
# --------------------------------------------------------------------------

TRIGGER_COMM = "nest_ste_trig"
_PR_SET_NAME = 15

_libc = None  # pylint: disable=invalid-name  # lazily-initialized singleton, not a constant


def _get_libc():
    """Lazily bind libc's prctl(). Deferred (rather than resolved at module
    import time) so importing this module -- e.g. to unit-test the pure
    Python helpers below -- does not require a Linux libc to be present."""
    global _libc  # pylint: disable=global-statement
    if _libc is None:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl.restype = ctypes.c_int
        libc.prctl.argtypes = [
            ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ]
        _libc = libc
    return _libc


def _set_thread_name(name):
    """Set the calling thread's comm via prctl(PR_SET_NAME, ...).

    Each Linux thread (task_struct) has its own independent comm, so this
    only affects the calling thread -- not the whole GLT process -- which
    is what lets observe_ste.bt's `comm == "nest_ste_trig"` kprobe filter
    keep working now that the ioctls run inside the GLT process instead of
    a standalone binary.
    """
    encoded = name.encode("ascii")
    if len(encoded) > 15:
        raise ValueError(
            f"thread name {name!r} exceeds 15 bytes (PR_SET_NAME/TASK_COMM_LEN limit)")
    buf = ctypes.create_string_buffer(encoded, 16)
    libc = _get_libc()
    if libc.prctl(_PR_SET_NAME, ctypes.addressof(buf), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), "prctl(PR_SET_NAME)")


# --------------------------------------------------------------------------
# ioctl request numbers -- see module docstring for the _IO() encoding note.
# --------------------------------------------------------------------------

IOMMUFD_TYPE = ord(';')
VFIO_TYPE = ord(';')
VFIO_BASE = 100


def _io(ioc_type, nr):
    return (ioc_type << 8) | nr


IOMMU_IOAS_ALLOC = _io(IOMMUFD_TYPE, 0x81)
IOMMU_IOAS_MAP = _io(IOMMUFD_TYPE, 0x85)
IOMMU_HWPT_ALLOC = _io(IOMMUFD_TYPE, 0x89)
IOMMU_GET_HW_INFO = _io(IOMMUFD_TYPE, 0x8A)
IOMMU_VIOMMU_ALLOC = _io(IOMMUFD_TYPE, 0x90)
IOMMU_VDEVICE_ALLOC = _io(IOMMUFD_TYPE, 0x91)

VFIO_DEVICE_BIND_IOMMUFD = _io(VFIO_TYPE, VFIO_BASE + 18)
VFIO_DEVICE_ATTACH_IOMMUFD_PT = _io(VFIO_TYPE, VFIO_BASE + 19)

# --------------------------------------------------------------------------
# UAPI flags/enums (include/uapi/linux/iommufd.h @ 7cad80048595)
# --------------------------------------------------------------------------

_IOAS_MAP_FIXED_IOVA = 1 << 0
_IOAS_MAP_WRITEABLE = 1 << 1
_IOAS_MAP_READABLE = 1 << 2

_HWPT_ALLOC_NEST_PARENT = 1 << 0

_HWPT_DATA_NONE = 0
_HWPT_DATA_ARM_SMMUV3 = 2

_HW_INFO_INPUT_TYPE = 1 << 0
_HW_INFO_TYPE_ARM_SMMUV3 = 2

_VIOMMU_TYPE_ARM_SMMUV3 = 1
_VIOMMU_TYPE_TEGRA241_CMDQV = 2

# --------------------------------------------------------------------------
# Flat UAPI layouts as struct.Struct (include/uapi/linux/{iommufd,vfio}.h
# @ 7cad80048595). ARM nested payloads stay as ctypes below.
# --------------------------------------------------------------------------

# struct iommu_ioas_alloc: size, flags, out_ioas_id
_IOAS_ALLOC = struct.Struct("<3I")
# struct iommu_ioas_map: size, flags, ioas_id, reserved, user_va, length, iova
_IOAS_MAP = struct.Struct("<4I3Q")
# struct iommu_hwpt_alloc: size, flags, dev_id, pt_id, out_hwpt_id, reserved,
#   data_type, data_len, data_uptr, fault_id, reserved2
_HWPT_ALLOC = struct.Struct("<8IQ2I")
# struct iommu_hw_info: size, flags, dev_id, data_len, data_uptr, data_type,
#   out_max_pasid_log2 + 3 reserved bytes (as one u32), out_capabilities
_HW_INFO = struct.Struct("<4IQIIQ")
# struct iommu_viommu_alloc: size, flags, viommu_type, dev_id, hwpt_id,
#   out_viommu_id, data_len, reserved, data_uptr
_VIOMMU_ALLOC = struct.Struct("<8IQ")
# struct iommu_vdevice_alloc: size, viommu_id, dev_id, out_vdevice_id, virt_id
_VDEVICE_ALLOC = struct.Struct("<4IQ")
# struct vfio_device_bind_iommufd: argsz, flags, iommufd, out_devid, token_uuid_ptr
_VFIO_BIND = struct.Struct("<2IiIQ")
# struct vfio_device_attach_iommufd_pt: argsz, flags, pt_id, pasid
_VFIO_ATTACH = struct.Struct("<4I")


class IommuHwptArmSmmuv3(ctypes.Structure):
    """struct iommu_hwpt_arm_smmuv3 (iommufd.h @ 7cad80048595); ste[2] is __aligned_le64[2].

    Field values are constructed explicitly as little-endian qwords (see
    _ste_make_*) and stored via plain c_uint64, which only reproduces the
    C code's htole64() byte layout on a little-endian host. All real-world
    aarch64 Linux userspace GreenLiT targets are little-endian, matching
    the assumption the rest of this codebase already makes (e.g. STE/CD
    bitfield math elsewhere), so this is checked at import time below.
    """
    _fields_ = [
        ("ste", ctypes.c_uint64 * 2),
    ]


class IommuHwInfoArmSmmuv3(ctypes.Structure):
    """struct iommu_hw_info_arm_smmuv3 (include/uapi/linux/iommufd.h @ 7cad80048595)"""
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("idr", ctypes.c_uint32 * 6),
        ("iidr", ctypes.c_uint32),
        ("aidr", ctypes.c_uint32),
    ]


_ABI = {
    _IOAS_ALLOC: ("iommu_ioas_alloc", 12),
    _IOAS_MAP: ("iommu_ioas_map", 40),
    _HWPT_ALLOC: ("iommu_hwpt_alloc", 48),
    _HW_INFO: ("iommu_hw_info", 40),
    _VIOMMU_ALLOC: ("iommu_viommu_alloc", 40),
    _VDEVICE_ALLOC: ("iommu_vdevice_alloc", 24),
    _VFIO_BIND: ("vfio_device_bind_iommufd", 24),
    _VFIO_ATTACH: ("vfio_device_attach_iommufd_pt", 16),
}

_CTYPES_ABI = {
    IommuHwptArmSmmuv3: 16,
    IommuHwInfoArmSmmuv3: 40,
}

# ABI sizes are verified in unit tests (not at import time): an import-time
# raise would abort GreenLiT's whole-suite module discovery. Byteorder is
# gated via _skip_reason_endianness() in glt_test_nested_ste.py.

# --------------------------------------------------------------------------
# STE bitfield math (mirrors drivers/iommu/arm/arm-smmu-v3/arm-smmu-v3.h;
# the standalone nested_ste.h this originally mirrored was retired when the
# trigger was ported to pure Python -- see module docstring)
# --------------------------------------------------------------------------


def _genmask(hi, lo):
    """GENMASK_ULL(hi, lo): contiguous 1-bits from lo through hi inclusive."""
    return ((1 << (hi - lo + 1)) - 1) << lo


def _field_prep(mask, val):
    """FIELD_PREP(mask, val): place val into the field described by mask."""
    shift = (mask & -mask).bit_length() - 1
    if (val << shift) & ~mask:
        raise ValueError(f"value {val:#x} overflows field {mask:#x}")
    return (val << shift) & mask


STE_0_V = _genmask(0, 0)
STE_0_CFG = _genmask(3, 1)
STE_0_S1FMT = _genmask(5, 4)
STE_0_S1CTXPTR = _genmask(51, 6)


_STE_CFG_BYPASS = 4
_STE_CFG_S1_TRANS = 5

_STE_S1FMT_64K_L2 = 2

_STE_S1DSS_BYPASS = 0x1

# STRTAB_STE_1_EATS: GENMASK_ULL(29, 28), a 2-bit field. Only ABT (0) and
# TRANS (1) are accepted by the kernel's arm_smmu_validate_vste() -- S1CHK
# (2) and the reserved value (3) are rejected with -EIO for a nested guest
# STE, so those are intentionally not modeled here.
STE_1_EATS = _genmask(29, 28)
STE_1_EATS_SHIFT = (STE_1_EATS & -STE_1_EATS).bit_length() - 1

_STE_EATS_ABT = 0x0
_STE_EATS_TRANS = 0x1

CD_IOVA = 0x10000

# Anonymous mmap length for the (unused-but-mapped) CD buffer. mmap() is
# inherently page-aligned, so deriving the length from the actual runtime
# page size (rather than a hardcoded 4096) fixes nested HWPT setup on
# systems with a non-4K page size (e.g. 64K pages).
CD_MAP_LEN = mmap.PAGESIZE


def _ste_make_s1dss_bypass(cd_iova, eats=_STE_EATS_ABT):
    """cfg=S1_TRANS (CD-table) guest STE, S1DSS=bypass.

    EATS defaults to ABT: for a CFG=S1_TRANS nested domain, the kernel's
    arm_smmu_attach_dev_nested() takes this guest-requested EATS value as
    the *sole* source of state.disable_ats (see arm-smmu-v3-iommufd.c),
    unlike the CFG=BYPASS case below where the guest's EATS request is
    validated but otherwise ignored. Requesting ABT here therefore
    deterministically forces the physical STE's EATS field to 0 regardless
    of whether the underlying PCI function actually supports ATS -- the
    fixed half of the EATS<->CFG co-transition (see
    _ste_make_nested_s2_bypass for the other half).
    """
    w0 = STE_0_V
    w0 |= _field_prep(STE_0_CFG, _STE_CFG_S1_TRANS)
    w0 |= _field_prep(STE_0_S1FMT, _STE_S1FMT_64K_L2)
    w0 |= _field_prep(STE_0_S1CTXPTR, cd_iova >> 6)
    w1 = _STE_S1DSS_BYPASS
    w1 |= _field_prep(STE_1_EATS, eats)
    return w0, w1


def _ste_make_nested_s2_bypass(eats=_STE_EATS_ABT):
    """cfg=BYPASS guest STE (S2-only nested passthrough).

    Unlike CFG=S1_TRANS, the kernel does not honor this field for
    enable_ats bookkeeping when cfg=BYPASS (arm_smmu_validate_vste() only
    range-checks it); the physical STE's EATS ends up driven solely by
    arm_smmu_ats_supported(), i.e. real PCI ATS capability on the attached
    function. Passing eats=_STE_EATS_TRANS here is still the "VM has ATS
    on" request a well-behaved guest would make in this state -- it makes
    the guest-STE-level transition read TRANS->ABT end to end, matching
    the VM behavior 7cad80048595 describes, even though the field's actual
    causal effect on the physical write is gated by hardware capability
    (see trigger module docstring / caller's risk notes).
    """
    w0 = STE_0_V | _field_prep(STE_0_CFG, _STE_CFG_BYPASS)
    w1 = _field_prep(STE_1_EATS, eats)
    return w0, w1


# --------------------------------------------------------------------------
# ioctl plumbing
# --------------------------------------------------------------------------


class TriggerError(Exception):
    """Raised when the nested-STE ioctl trigger sequence fails."""


class TriggerTimeout(TriggerError):
    """Raised when the worker thread is still alive after the timeout, i.e.
    stuck inside a kernel ioctl still holding /dev/iommu + vfio fds.

    Distinct from a plain TriggerError so the caller can tell that the
    device must NOT be unbound/rebound/restored -- doing so would race the
    in-flight ioctl the stuck worker still owns -- and can recommend a
    reboot instead of touching the device."""


_BDF_RE = re.compile(
    r"^(?P<dom>[0-9a-fA-F]+):(?P<bus>[0-9a-fA-F]+):(?P<dev>[0-9a-fA-F]+)\.(?P<fn>[0-9a-fA-F]+)$")


def _valid_bdf(bdf):
    """Sanity-check a caller-supplied BDF (defense in depth, mirrors the
    old C trigger's valid_bdf() -- BDFs end up embedded in sysfs/devfs
    paths, so reject anything that looks like path traversal or is out of
    the legal PCI bus/device/function range."""
    if not bdf or "/" in bdf or ".." in bdf:
        return False
    match = _BDF_RE.match(bdf)
    if not match:
        return False
    bus = int(match.group("bus"), 16)
    dev = int(match.group("dev"), 16)
    fn = int(match.group("fn"), 16)
    return bus <= 0xFF and dev <= 0x1F and fn <= 0x7


def _find_vfio_cdev(bdf):
    sysfs_dir = f"/sys/bus/pci/devices/{bdf}/vfio-dev"
    try:
        entries = os.listdir(sysfs_dir)
    except OSError:
        return None
    for name in sorted(entries):
        if name.startswith("vfio"):
            return f"/dev/vfio/devices/{name}"
    return None


def _xioctl(fd, request, buf):
    """Issue one ioctl into ``buf``, retrying on EINTR.

    ``buf`` is a mutable ``bytearray`` (or other writable buffer). A
    bytearray is used rather than a ctypes structure because it
    unconditionally supports the writable buffer protocol that
    fcntl.ioctl's mutate_flag relies on, regardless of how the request's
    encoded direction bits happen to be set -- and for these commands the
    kernel does not consult those bits anyway (see module docstring):
    iommufd/vfio determine the real read/write shape from the embedded
    ``size``/``data_len`` fields, not the ioctl number.
    """
    while True:
        try:
            fcntl.ioctl(fd, request, buf, True)
            break
        except InterruptedError:
            continue
        except OSError as exc:
            raise TriggerError(f"ioctl 0x{request:x} failed: {exc}") from exc


def _ioas_alloc(iommufd):
    buf = bytearray(_IOAS_ALLOC.pack(_IOAS_ALLOC.size, 0, 0))
    _xioctl(iommufd, IOMMU_IOAS_ALLOC, buf)
    return _IOAS_ALLOC.unpack_from(buf)[2]


def _ioas_map(iommufd, ioas_id, addr, length, iova):
    buf = bytearray(_IOAS_MAP.pack(
        _IOAS_MAP.size,
        (_IOAS_MAP_FIXED_IOVA | _IOAS_MAP_READABLE
         | _IOAS_MAP_WRITEABLE),
        ioas_id,
        0,
        addr,
        length,
        iova,
    ))
    _xioctl(iommufd, IOMMU_IOAS_MAP, buf)


def _vfio_bind_iommufd(vfio_fd, iommufd):
    buf = bytearray(_VFIO_BIND.pack(_VFIO_BIND.size, 0, iommufd, 0, 0))
    _xioctl(vfio_fd, VFIO_DEVICE_BIND_IOMMUFD, buf)
    return _VFIO_BIND.unpack_from(buf)[3]


def _vfio_attach_pt(vfio_fd, pt_id):
    buf = bytearray(_VFIO_ATTACH.pack(_VFIO_ATTACH.size, 0, pt_id, 0))
    _xioctl(vfio_fd, VFIO_DEVICE_ATTACH_IOMMUFD_PT, buf)


def _hwpt_alloc_parent(iommufd, dev_id, ioas_id):
    buf = bytearray(_HWPT_ALLOC.pack(
        _HWPT_ALLOC.size,
        _HWPT_ALLOC_NEST_PARENT,
        dev_id,
        ioas_id,
        0,
        0,
        _HWPT_DATA_NONE,
        0,
        0,
        0,
        0,
    ))
    _xioctl(iommufd, IOMMU_HWPT_ALLOC, buf)
    return _HWPT_ALLOC.unpack_from(buf)[4]


def _viommu_alloc(iommufd, dev_id, parent_hwpt_id, viommu_type):
    buf = bytearray(_VIOMMU_ALLOC.pack(
        _VIOMMU_ALLOC.size,
        0,
        viommu_type,
        dev_id,
        parent_hwpt_id,
        0,
        0,
        0,
        0,
    ))
    _xioctl(iommufd, IOMMU_VIOMMU_ALLOC, buf)
    return _VIOMMU_ALLOC.unpack_from(buf)[5]


def _vdevice_alloc(iommufd, viommu_id, dev_id):
    buf = bytearray(_VDEVICE_ALLOC.pack(
        _VDEVICE_ALLOC.size, viommu_id, dev_id, 0, 0))
    _xioctl(iommufd, IOMMU_VDEVICE_ALLOC, buf)


def _hwpt_alloc_nested(iommufd, dev_id, viommu_id, ste_data):
    # ste_data must be a named local kept alive by the caller for the
    # duration of this call -- data_uptr is a raw pointer the kernel
    # dereferences, so the buffer it points at must not be a temporary.
    buf = bytearray(_HWPT_ALLOC.pack(
        _HWPT_ALLOC.size,
        0,
        dev_id,
        viommu_id,
        0,
        0,
        _HWPT_DATA_ARM_SMMUV3,
        ctypes.sizeof(ste_data),
        ctypes.addressof(ste_data),
        0,
        0,
    ))
    _xioctl(iommufd, IOMMU_HWPT_ALLOC, buf)
    return _HWPT_ALLOC.unpack_from(buf)[4]


def _get_hw_info_smmuv3(iommufd, dev_id, info):
    # info must be a named local kept alive by the caller (see note above).
    buf = bytearray(_HW_INFO.pack(
        _HW_INFO.size,
        _HW_INFO_INPUT_TYPE,
        dev_id,
        ctypes.sizeof(info),
        ctypes.addressof(info),
        _HW_INFO_TYPE_ARM_SMMUV3,
        0,
        0,
    ))
    _xioctl(iommufd, IOMMU_GET_HW_INFO, buf)
    data_type = _HW_INFO.unpack_from(buf)[5]
    if data_type != _HW_INFO_TYPE_ARM_SMMUV3:
        raise TriggerError(
            "IOMMU_GET_HW_INFO: device did not report ARM_SMMUV3 hw info")


# --------------------------------------------------------------------------
# top-level sequence (mirrors setup_nesting()/run_transitions()/main())
# --------------------------------------------------------------------------


def _setup_nesting(iommufd, vfio_fd, cd_addr):
    """Allocate the IOAS/HWPT/vIOMMU/vDEVICE graph and both nested HWPTs.

    cd_addr is the address of an already-mapped (and caller-owned) CD
    buffer -- ownership stays with the caller (_run_sequence) so its
    `finally` can release it regardless of which step here fails.

    Returns (hwpt_s1dss_id, hwpt_s2bypass_eats_id).
    """
    ioas_id = _ioas_alloc(iommufd)

    _ioas_map(iommufd, ioas_id, cd_addr, CD_MAP_LEN, CD_IOVA)

    dev_id = _vfio_bind_iommufd(vfio_fd, iommufd)

    hwinfo = IommuHwInfoArmSmmuv3()
    _get_hw_info_smmuv3(iommufd, dev_id, hwinfo)

    parent_hwpt_id = _hwpt_alloc_parent(iommufd, dev_id, ioas_id)

    viommu_id = None
    last_exc = None
    for viommu_type in (_VIOMMU_TYPE_TEGRA241_CMDQV, _VIOMMU_TYPE_ARM_SMMUV3):
        try:
            viommu_id = _viommu_alloc(iommufd, dev_id, parent_hwpt_id, viommu_type)
            break
        except TriggerError as exc:
            last_exc = exc
    if viommu_id is None:
        raise TriggerError(
            f"IOMMU_VIOMMU_ALLOC failed for all candidate vIOMMU types: {last_exc}")

    _vdevice_alloc(iommufd, viommu_id, dev_id)

    # S1DSS-bypass guest STE, EATS=ABT (deterministically forces the
    # physical STE's EATS field to 0 -- see _ste_make_s1dss_bypass).
    w0, w1 = _ste_make_s1dss_bypass(CD_IOVA, eats=_STE_EATS_ABT)
    ste_s1dss = IommuHwptArmSmmuv3(ste=(w0, w1))
    hwpt_s1dss_id = _hwpt_alloc_nested(iommufd, dev_id, viommu_id, ste_s1dss)

    # S2-bypass guest STE requesting EATS=TRANS. Alternating with S1DSS
    # carries an EATS_TRANS toggle simultaneous with the CFG change
    # (physical Config 7<->6), per 7cad80048595.
    w0, w1 = _ste_make_nested_s2_bypass(eats=_STE_EATS_TRANS)
    ste_s2bypass_eats = IommuHwptArmSmmuv3(ste=(w0, w1))
    hwpt_s2bypass_eats_id = _hwpt_alloc_nested(
        iommufd, dev_id, viommu_id, ste_s2bypass_eats)

    return hwpt_s1dss_id, hwpt_s2bypass_eats_id


def _run_transitions(vfio_fd, hwpt_s1dss_id, hwpt_s2bypass_eats_id, iters):
    # Alternate S1DSS-bypass (EATS=ABT) <-> S2-bypass (EATS=TRANS). Each
    # attach/replace flips STE CFG (physical Config 7<->6 / MEV) and
    # EATS_TRANS simultaneously.
    for _ in range(iters):
        _vfio_attach_pt(vfio_fd, hwpt_s1dss_id)
        _vfio_attach_pt(vfio_fd, hwpt_s2bypass_eats_id)
    return f"TRANSITIONS_OK iters={iters}"


def _run_sequence(bdf, iters):
    if not _valid_bdf(bdf):
        raise TriggerError(f"invalid PCI BDF: {bdf}")

    cdev = _find_vfio_cdev(bdf)
    if not cdev:
        raise TriggerError(f"no vfio cdev for {bdf}")

    iommufd = os.open("/dev/iommu", os.O_RDWR | os.O_CLOEXEC)
    vfio_fd = None
    cd_mem = mmap.mmap(-1, CD_MAP_LEN, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)
    cd_view = (ctypes.c_char * CD_MAP_LEN).from_buffer(cd_mem)
    try:
        vfio_fd = os.open(cdev, os.O_RDWR | os.O_CLOEXEC)
        cd_addr = ctypes.addressof(cd_view)
        hwpt_s1dss_id, hwpt_s2bypass_eats_id = _setup_nesting(
            iommufd, vfio_fd, cd_addr)
        return _run_transitions(
            vfio_fd, hwpt_s1dss_id, hwpt_s2bypass_eats_id, iters)
    finally:
        if vfio_fd is not None:
            os.close(vfio_fd)
        os.close(iommufd)
        del cd_view
        cd_mem.close()


class _TriggerWorker(threading.Thread):
    """Run the ioctl sequence while retaining its result for the caller."""

    def __init__(self, bdf, iters):
        super().__init__(name=TRIGGER_COMM, daemon=True)
        self.bdf = bdf
        self.iters = iters
        self.status = None
        self.error = None

    def run(self):
        try:
            _set_thread_name(TRIGGER_COMM)
            self.status = _run_sequence(self.bdf, self.iters)
        except BaseException as exc:  # pylint: disable=broad-except
            self.error = exc


def run_trigger(bdf, iters, timeout=None):
    """Run the nested-STE attach/replace ioctl sequence against bdf.

    Ioctls run on a dedicated worker thread renamed to TRIGGER_COMM so
    observe_ste.bt's ``comm == "nest_ste_trig"`` filter matches. Raises
    TriggerError / TriggerTimeout; returns a short status string on success.
    """
    worker = _TriggerWorker(bdf, iters)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TriggerTimeout(
            f"trigger sequence timed out after {timeout}s (possible kernel ioctl hang)")
    if worker.error is not None:
        error = worker.error
        if isinstance(error, TriggerError):
            raise error
        raise TriggerError(str(error)) from error
    return worker.status
