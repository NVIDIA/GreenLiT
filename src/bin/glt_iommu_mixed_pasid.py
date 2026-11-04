# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, child-owned mixed-PASID transaction; never map DMA or unbind the port."""

import errno
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import glt_utils, pci_utils
from modules.iommu.nested_ste.trigger import (
    IOMMU_HWPT_ALLOC,
    VFIO_DEVICE_ATTACH_IOMMUFD_PT,
    TriggerError,
    _HWPT_ALLOC,
    _VFIO_ATTACH,
    _find_vfio_cdev,
    _ioas_alloc,
    _vfio_attach_pt,
    _vfio_bind_iommufd,
    _xioctl,
)

OBSERVER_PATH = Path(__file__).with_suffix(".bt").resolve()
VFIO_DEVICE_DETACH_IOMMUFD_PT = (ord(';') << 8) | 120
VFIO_DETACH = struct.Struct("<3I")
OBSERVATION_TIMEOUT = 2.0


def _override(bdf):
    value = (Path("/sys/bus/pci/devices") / bdf / "driver_override").read_text(encoding="utf-8").strip()
    return "" if value == "(null)" else value


def _module_present(name):
    """Inspect module presence without suppressing failures to read its state."""
    try:
        (Path("/sys/module") / name).stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _module_plan():
    """Resolve only the required modules' declared dependencies, without loading."""
    modules = []
    for name in ("iommufd", "vfio_pci"):
        result = subprocess.run(["modprobe", "--show-depends", name],
                                capture_output=True, text=True, check=True, shell=False, timeout=10)
        for line in result.stdout.splitlines():
            fields = line.split()
            if fields and fields[0] == "insmod":
                module = Path(fields[1]).name.split(".ko")[0].replace("-", "_")
                if module not in modules:
                    modules.append(module)
            elif fields and fields[0] != "builtin":
                raise ValueError("unsupported module installation command")
    return modules


def _gpu_fixture(gpu, capabilities):
    """Read one candidate's topology, ownership and capability existence."""
    fixture = {"gpu": gpu}
    parent, error = pci_utils.get_pci_parent(gpu)
    if error:
        raise OSError(error)
    if parent is None:
        return fixture
    device = Path("/sys/bus/pci/devices") / gpu
    parent_info = subprocess.run(["lspci", "-D", "-s", parent, "-vv"],
                                 capture_output=True, text=True, check=True, shell=False, timeout=10)
    used = gpu in pci_utils.unsafe_pci_bdfs()
    try:
        boot_vga = (device / "boot_vga").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        boot_vga = "0"
    used = used or boot_vga != "0"
    gpu_path = device.resolve(strict=True)
    try:
        framebuffers = list(Path("/sys/class/graphics").iterdir())
    except FileNotFoundError:
        framebuffers = []
    for framebuffer in framebuffers:
        if not re.fullmatch(r"fb[0-9]+", framebuffer.name):
            continue
        try:
            target = (framebuffer / "device").resolve(strict=True)
        except FileNotFoundError:
            continue
        used = used or target == gpu_path or gpu_path in target.parents
    fixture.update(gpu=gpu, parent=parent, direct_parent=parent,
                   parent_root_port="Root Port" in parent_info.stdout,
                   members=sorted(path.name for path in (device / "iommu_group/devices").iterdir()),
                   smmuv3=pci_utils.is_arm_smmu_v3_backed(gpu),
                   gpu_driver=pci_utils.get_pci_driver(gpu), gpu_override=_override(gpu), gpu_used=used,
                   parent_driver=pci_utils.get_pci_driver(parent), gpu_pasid_cap=gpu in capabilities,
                   parent_pasid_cap=parent in capabilities)
    return fixture


def inspect_fixture():
    """Select the first eligible domain-qualified GPU, without interpreting PASID width."""
    tools = ("bpftrace", "bpftool", "modprobe", "lspci")
    fixture = {"os_id": platform.freedesktop_os_release().get("ID"),
               "machine": platform.machine(),
               "tools_ready": (sys.byteorder == "little"
                               and all(shutil.which(tool) for tool in tools)
                               and Path("/sys/kernel/btf/vmlinux").is_file()
                               and OBSERVER_PATH.is_file())}
    if fixture["os_id"] != "ubuntu" or fixture["machine"] != "aarch64" or not fixture["tools_ready"]:
        return fixture
    fixture["module_plan"] = _module_plan()
    capabilities = set(pci_utils.get_bdfs_with_ext_cap("Process Address Space ID", timeout=10))
    candidate = fixture
    inspection_error = None
    for gpu in sorted(pci_utils.get_bdfs_by_pci_id("10de", timeout=10)):
        if (not pci_utils.BDF_STRICT_RE.fullmatch(gpu)
                or not (pci_utils.get_pci_class(gpu) or "").lower().startswith("0x0302")):
            continue
        try:
            candidate = dict(fixture, **_gpu_fixture(gpu, capabilities))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            inspection_error = exc
            continue  # An unreadable candidate cannot qualify; another group may be eligible.
        if not fixture_reason(candidate):
            return candidate
    if inspection_error is not None:
        raise TriggerError(f"candidate inspection failed: {inspection_error}") from inspection_error
    return candidate


def fixture_reason(fixture):
    """Ordinary missing prerequisites may skip only before acquisition."""
    requirements = (
        (fixture.get("os_id") == "ubuntu", "Ubuntu is required"),
        (fixture.get("machine") == "aarch64", "ARM64 is required"),
        (fixture.get("tools_ready"), "little endian, BTF, bpftrace, bpftool and VFIO/IOMMUFD modules required"),
        (bool(fixture.get("gpu")) and bool(fixture.get("parent")), "GPU/direct-parent pair missing"),
        (fixture.get("smmuv3"), "GPU must be backed by ARM SMMUv3"),
        (fixture.get("direct_parent") == fixture.get("parent"), "direct-parent identity mismatch"),
        (set(fixture.get("members", [])) == {fixture.get("gpu"), fixture.get("parent")}
         and len(fixture.get("members", [])) == 2, "IOMMU group must contain exactly the GPU and its parent"),
        (fixture.get("parent_root_port", True), "companion must be a root port"),
        (fixture.get("gpu_driver") is None and not fixture.get("gpu_used"), "GPU must be unused and unbound"),
        (fixture.get("gpu_override") in ("", None), "GPU driver override must be null"),
        (fixture.get("parent_driver") == "pcieport", "parent must remain on pcieport"),
        (fixture.get("gpu_pasid_cap") is True, "GPU PASID capability missing"),
        (fixture.get("parent_pasid_cap") is False, "parent must have no PASID capability"),
    )
    return next((reason for ok, reason in requirements if not ok), "")


def _acquisition_checkpoint(resources):
    """Authorize only the next mutation while observer and ACK budget remain live."""
    if resources["observer"]["process"].poll() is not None:
        raise TriggerError("observer died before acquisition")
    if time.monotonic() >= resources["acquisition_deadline"]:
        raise TriggerError("acquisition budget expired")


def _acquire_gpu(fixture, resources):
    """After handoff, recheck topology and acquire only the previously idle GPU."""
    current = inspect_fixture()
    reason = fixture_reason(current)
    keys = ("gpu", "parent", "members", "gpu_driver", "gpu_override", "parent_driver")
    if reason or any(current.get(key) != fixture.get(key) for key in keys):
        raise TriggerError(reason or "fixture changed before acquisition")
    _acquisition_checkpoint(resources)
    absent = [name for name in fixture["module_plan"] if not _module_present(name)]
    resources["modules_preexisting"] = [name for name in fixture["module_plan"] if name not in absent]
    try:
        for name in ("iommufd", "vfio_pci"):
            _acquisition_checkpoint(resources)
            subprocess.run(["modprobe", name], capture_output=True, text=True, check=True, shell=False, timeout=10)
    finally:
        resources["modules_loaded"] = []
        for name in absent:
            try:
                if _module_present(name):
                    resources["modules_loaded"].append(name)
            except OSError as exc:
                resources["module_inspection_errors"] = [f"module ownership inspection failed: {name}: {exc}"]
                raise
    if pci_utils.get_pci_driver(fixture["gpu"]) is not None or _override(fixture["gpu"]):
        raise TriggerError("GPU ownership changed before binding")
    _acquisition_checkpoint(resources)
    resources["gpu_changed"] = True
    device = Path("/sys/bus/pci/devices") / fixture["gpu"]
    (device / "driver_override").write_text("vfio-pci", encoding="utf-8")
    if _override(fixture["gpu"]) != "vfio-pci":
        raise TriggerError("GPU override readback failed")
    _acquisition_checkpoint(resources)
    Path("/sys/bus/pci/drivers/vfio-pci/bind").write_text(fixture["gpu"], encoding="utf-8")
    if (pci_utils.get_pci_driver(fixture["gpu"]) != "vfio-pci"
            or pci_utils.get_pci_driver(fixture["parent"]) != "pcieport"):
        raise TriggerError("GPU binding or parent preservation failed")


def allocate_context(fixture):
    """Bind owned FDs and allocate an empty compatible HWPT."""
    context = fixture.get("resources", {"fds": []})
    context["iommu_fd"] = os.open("/dev/iommu", os.O_RDWR)
    context["fds"].append(context["iommu_fd"])
    cdev = _find_vfio_cdev(fixture["gpu"])
    if cdev is None:
        raise TriggerError("GPU VFIO cdev is missing")
    context["vfio_fd"] = os.open(cdev, os.O_RDWR)
    context["fds"].append(context["vfio_fd"])
    context["device_id"] = _vfio_bind_iommufd(context["vfio_fd"], context["iommu_fd"])
    context["ioas_id"] = _ioas_alloc(context["iommu_fd"])
    payload = bytearray(_HWPT_ALLOC.pack(48, 8, context["device_id"], context["ioas_id"], 0, 0, 0, 0, 0, 0, 0))
    _xioctl(context["iommu_fd"], IOMMU_HWPT_ALLOC, payload)
    context["hwpt_id"] = _HWPT_ALLOC.unpack(payload)[4]
    _vfio_attach_pt(context["vfio_fd"], context["hwpt_id"])
    return context


def _records(text):
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("observer record is not an object")
        if record.get("event") != "ready":
            records.append(record)
    return records


def _entry_limit(record, fixture, worker_pid, worker_tid):
    required = {"event": "entry", "pid": worker_pid, "tid": worker_tid,
                "member": fixture["gpu"], "parent": fixture["parent"],
                "gpu_iommu": 1, "parent_iommu": 1, "parent_limit": 0}
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("core entry identity, pointers or companion limit mismatch")
    limit = record.get("gpu_limit")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("GPU effective max_pasids must be positive")
    return min(value for value in (limit, record["parent_limit"]) if value > 0)


def _sequence(records, fixture, worker_pid, worker_tid):
    for record in records:
        if record.get("pid") != worker_pid or record.get("tid") != worker_tid:
            raise ValueError("observer PID/TID attribution mismatch")
        if record.get("member") != fixture["gpu"]:
            raise ValueError("unexpected member callback or core identity")
    return [(record.get("event"), record.get("pasid"), record.get("ret")) for record in records]


def _valid_evidence(records, fixture, identity):
    if not records:
        raise ValueError("incomplete evidence")
    pid, tid = identity
    bound = _entry_limit(records[0], fixture, pid, tid)
    sequence = _sequence(records, fixture, pid, tid)
    good = [("entry", 1, None), ("set", 1, None), ("return", 1, 0), ("block", 1, None)]
    control = [("entry", 1, None), ("return", 1, -errno.EINVAL)]
    if sequence == control:
        return bound, False
    if sequence == good:
        return bound, True
    if sequence == good[:len(sequence)] and len(sequence) < len(good):
        raise ValueError("incomplete evidence")
    raise ValueError("valid PASID needs one ordered entry/set/return0, explicit detach0 and GPU block")


def evaluate_trace(text, fixture, worker_pid, worker_tid, detach_result):
    """Require exact ordered per-request evidence, independently of kernel filters."""
    result = {"status": "FAIL", "reason": "", "bound": None, "valid": False}
    try:
        records = _records(text)
        bound, valid = _valid_evidence(records, fixture, (worker_pid, worker_tid))
        if valid and detach_result != 0:
            raise ValueError("valid PASID needs one ordered entry/set/return0, explicit detach0 and GPU block")
        result.update(bound=bound, valid=valid)
        if bound == 1:
            raise ValueError("effective limit 1 cannot prove a valid positive PASID")
        result["status"] = "PASS" if valid else "FAIL"
        result["reason"] = "" if valid else (
            "valid PASID reached core and returned EINVAL (expected to succeed with b3f6fcd8404f)")
    except (ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc)
    return result


def _trace_until(read_trace):
    deadline = time.monotonic() + OBSERVATION_TIMEOUT
    while True:
        text = read_trace()
        if time.monotonic() >= deadline:
            return text
        time.sleep(0.02)


def _expected_einval(error):
    return isinstance(error.__cause__, OSError) and error.__cause__.errno == errno.EINVAL


def run_requests(context, fixture, read_trace):
    """Exercise PASID 1 and check the final bounded trace snapshot."""
    detach_result = None
    valid_error = None
    failure = {"status": "FAIL", "reason": "", "bound": None,
               "valid": False, "detach_result": None}
    try:
        payload = bytearray(_VFIO_ATTACH.pack(16, 1, context["hwpt_id"], 1))
        try:
            _xioctl(context["vfio_fd"], VFIO_DEVICE_ATTACH_IOMMUFD_PT, payload)
        except TriggerError as exc:
            if not _expected_einval(exc):
                raise
            valid_error = exc
        if valid_error is None:
            _xioctl(context["vfio_fd"], VFIO_DEVICE_DETACH_IOMMUFD_PT,
                    bytearray(VFIO_DETACH.pack(12, 1, 1)))
            detach_result = 0
        text = _trace_until(read_trace)
        records = _records(text)
        observed_bound, valid = _valid_evidence(records, fixture, (os.getpid(), threading.get_native_id()))
        if valid != (valid_error is None):
            raise ValueError("valid ioctl result disagrees with attributed core evidence")
        if observed_bound == 1 and not valid:
            return dict(failure, status="SKIP",
                        reason="observed effective limit 1 has no usable positive PASID",
                        bound=observed_bound, trace=text)
        result = evaluate_trace(text, fixture, os.getpid(), threading.get_native_id(), detach_result)
        result.update(detach_result=detach_result, trace=text)
        return result
    except (TriggerError, OSError, ValueError, KeyError, TypeError) as exc:
        failure.update(reason=str(exc), detach_result=detach_result)
        return failure


def program_inventory():
    """Use PATH bpftool and require a successful, well-shaped JSON inventory."""
    if not glt_utils.is_cli_available("bpftool"):
        raise FileNotFoundError("bpftool is not on PATH")
    result = subprocess.run(["bpftool", "-j", "prog", "show"],
                            capture_output=True, text=True, check=True, shell=False, timeout=10)
    programs = json.loads(result.stdout)
    if not isinstance(programs, list) or any(
            not isinstance(program, dict) or not isinstance(program.get("id"), int)
            or isinstance(program["id"], bool)
            or program["id"] <= 0 for program in programs):
        raise ValueError("invalid bpftool JSON program inventory")
    return programs


def _observer_identity(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text(encoding="utf-8")
    return {"pid": pid, "starttime": stat[stat.rfind(")") + 2:].split()[19],
            "argv": (root / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")}


def _owned_ids(pid):
    return sorted({int(line.split(":", 1)[1].strip())
                   for info in (Path("/proc") / str(pid) / "fdinfo").iterdir()
                   for line in info.read_text(encoding="utf-8").splitlines() if line.startswith("prog_id:")})


def start_observer(worker_pid):
    """Start four typed probes; observe own BEGIN, process identity and fdinfo IDs."""
    argv = [shutil.which("bpftrace") or "bpftrace", "-q", "-B", "line", str(OBSERVER_PATH), str(worker_pid)]
    # D4: ownership transfers to child cleanup; a context exit would wait here.
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # pylint: disable=consider-using-with
                               text=True, bufsize=1)
    observer = {"pid": process.pid, "starttime": None, "argv": argv, "ready": False,
                "live": False, "marker_pid": None, "program_ids": [],
                "process": process, "lines": []}

    def collect():
        for line in process.stdout:
            observer["lines"].append(line)

    observer["reader"] = threading.Thread(target=collect, daemon=True)
    observer["reader"].start()
    deadline = time.monotonic() + 15
    while process.poll() is None and time.monotonic() < deadline:
        try:
            observer.update(_observer_identity(process.pid))
            observer["program_ids"] = _owned_ids(process.pid)
            marker = {"event": "ready", "pid": worker_pid}
            observer["ready"] = any(json.loads(line) == marker for line in "".join(observer["lines"]).splitlines())
            observer["marker_pid"] = worker_pid if observer["ready"] else None
            if observer["ready"] and len(observer["program_ids"]) >= 4:
                break
        except (OSError, ValueError, IndexError):
            pass  # Probe loading and fdinfo population are asynchronous until the deadline.
        time.sleep(0.05)
    observer["live"] = process.poll() is None
    return observer


def prepare_observer(worker_pid, handoff):
    """No binding is possible until readiness, inventory and handoff all pass."""
    observer = {"pid": None, "program_ids": []}
    try:
        program_inventory()
        observer = start_observer(worker_pid)
        owned = set(observer["program_ids"])
        if not (observer["live"] and observer["ready"] and observer["marker_pid"] == worker_pid
                and len(owned) >= 4 and observer.get("starttime") and owned.issubset({
                    program["id"] for program in program_inventory()})):
            raise ValueError("observer needs own BEGIN, live identity and four distinct fdinfo-owned program IDs")
        if not handoff(observer):
            raise ValueError("observer ownership handoff was not acknowledged")
        observer.update(ok=True, reason="")
    except (OSError, ValueError, subprocess.SubprocessError, TriggerError) as exc:
        observer.update(ok=False, reason=str(exc))
    return observer


def restore_gpu(fixture):
    """Restore only the acquired GPU, then read back both members and override."""
    gpu = fixture["gpu"]
    driver = pci_utils.get_pci_driver(gpu)
    if driver not in (None, "vfio-pci"):
        return False
    if driver == "vfio-pci":
        Path("/sys/bus/pci/drivers/vfio-pci/unbind").write_text(gpu, encoding="utf-8")
    (Path("/sys/bus/pci/devices") / gpu / "driver_override").write_text(
        fixture["gpu_override"] or "\n", encoding="utf-8")
    return (pci_utils.get_pci_driver(gpu) == fixture["gpu_driver"]
            and _override(gpu) == (fixture["gpu_override"] or "")
            and pci_utils.get_pci_driver(fixture["parent"]) == "pcieport")


def stop_observer(observer):
    """Reap this child's own Popen; never signal a PID without its live identity."""
    process = observer.get("process")
    if process is None:
        return False
    try:
        if process.poll() is None:
            identity = _observer_identity(observer["pid"])
            if any(identity[key] != observer[key] for key in ("pid", "starttime", "argv")):
                return False
            process.terminate()
        process.wait(timeout=5)
        observer["reader"].join(timeout=5)
        if observer["reader"].is_alive():
            return False
        return process.poll() is not None
    finally:
        if process.poll() is not None:
            process.stdout.close()


def _cleanup_observer(observer):
    """Reap the acquired observer and independently inventory its program IDs."""
    errors = []
    if observer and observer.get("pid") is not None:
        try:
            if not stop_observer(observer):
                errors.append("observer reap failed")
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"observer reap failed: {exc}")
        try:
            remaining = {program["id"] for program in program_inventory()} & set(observer["program_ids"])
            if remaining:
                errors.append(f"owned program IDs remain: {sorted(remaining)}")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"inventory failed: {exc}")
    return errors


def cleanup_resources(fixture, resources):
    """Close, restore, reap, inventory and unload only acquired owned resources."""
    errors = list(resources.get("module_inspection_errors", []))
    for fd in reversed(resources.get("fds", [])):
        try:
            os.close(fd)
        except OSError as exc:
            errors.append(f"close failed: {exc}")
    if resources.get("gpu_changed", False):
        try:
            if not restore_gpu(fixture):
                errors.append("GPU/parent/override restore readback failed")
        except OSError as exc:
            errors.append(f"restore failed: {exc}")
    errors.extend(_cleanup_observer(resources.get("observer")))
    for module in reversed(resources.get("modules_loaded", [])):
        try:
            subprocess.run(["rmmod", module], capture_output=True, text=True, check=True, shell=False, timeout=10)
            if _module_present(module):
                errors.append(f"module remains loaded: {module}")
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"module unload failed: {module}: {exc}")
    for module in resources.get("modules_preexisting", []):
        try:
            if not _module_present(module):
                errors.append(f"preexisting module missing: {module}")
        except OSError as exc:
            errors.append(f"preexisting module inspection failed: {module}: {exc}")
    return {"ok": not errors, "reason": "; ".join(errors), "residuals": errors}


def _public_observer(observer):
    return {key: observer[key] for key in ("pid", "starttime", "argv", "program_ids")}


def run_worker(handoff):
    """One child owns every acquired resource until verified cleanup is complete."""
    resources = {"fds": [], "modules_loaded": [], "gpu_changed": False, "handoff_complete": False}
    result = {"status": "FAIL", "reason": "", "cleanup_ok": False, "residuals": []}
    try:
        fixture = inspect_fixture()
        reason = fixture_reason(fixture)
    except TriggerError as exc:
        return dict(result, reason=str(exc), cleanup_ok=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return dict(result, status="SKIP", reason=f"fixture unavailable: {exc}", cleanup_ok=True)
    if reason:
        return dict(result, status="SKIP", reason=reason, cleanup_ok=True)
    try:
        def acquire_after_handoff(observer):
            resources["observer"] = observer
            if not handoff(_public_observer(observer)):
                return False
            resources["handoff_complete"] = True
            # Checkpoint bound only: it cannot cancel an entered kernel/sysfs call.
            # 60s acquisition + one final 10s command is below the parent's 180s wait.
            resources["acquisition_deadline"] = time.monotonic() + 60
            _acquire_gpu(fixture, resources)
            return True

        observer = prepare_observer(os.getpid(), acquire_after_handoff)
        resources["observer"] = observer
        if not observer["ok"]:
            result.update(status="FAIL" if resources["handoff_complete"] else "SKIP",
                          reason=observer["reason"])
        else:
            fixture["resources"] = resources
            _acquisition_checkpoint(resources)
            context = allocate_context(fixture)
            result.update(run_requests(context, fixture, lambda: "".join(observer["lines"])))
    except (OSError, ValueError, TriggerError, subprocess.SubprocessError) as exc:
        result.update(status="FAIL", reason=str(exc))
    finally:
        cleanup = cleanup_resources(fixture, resources)
        result.update(cleanup_ok=cleanup["ok"], residuals=cleanup["residuals"])
        if not cleanup["ok"]:
            result.update(status="FAIL", reason="; ".join(filter(None, (result["reason"], cleanup["reason"]))))
    fixture.pop("resources", None)
    result.update(fixture=fixture, modules_loaded=resources["modules_loaded"],
                  modules_preexisting=resources.get("modules_preexisting", []))
    return result


def main():
    """Line protocol: acknowledged observer ownership, then final cleanup report."""
    def handoff(observer):
        print(json.dumps({"type": "observer", "observer": observer}), flush=True)
        return sys.stdin.readline().strip() == "ACK"

    result = run_worker(handoff)
    print(json.dumps(dict(result, type="result")), flush=True)
    return 0 if result["status"] in ("PASS", "SKIP") else 1


if __name__ == "__main__":
    raise SystemExit(main())
