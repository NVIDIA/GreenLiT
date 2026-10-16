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

"""PCI bridge window type check for pbus_select_window_for_type()."""

import os
import platform
import re
import subprocess

from common import glt_core as GLTCore
from common import glt_utils

# include/linux/ioport.h
IORESOURCE_TYPE_BITS = 0x00001F00
IORESOURCE_DISABLED = 0x10000000
IORESOURCE_STARTALIGN = 0x00080000
IORESOURCE_UNSET = 0x20000000

_CLAIMED_BITS = IORESOURCE_DISABLED | IORESOURCE_STARTALIGN | IORESOURCE_UNSET
_UPSTREAM_SHA = "e5f72cb9cea599dc9f5a9b80a33560a1d06f01cc"
_NVIDIA_SHA = "96f42087b05801263b1f4c1129f525c0bbcfa3eb"
_MAKE_TIMEOUT_SECS = 120
_MODULE_TIMEOUT_SECS = 30
_PROBE_MODULE = "bridge_window_probe"
_PROBE_PROC = "/proc/greenlit_pci_bridge_window"
_PROBE_LINE = re.compile(
    r"^(?P<bdf>\S+) (?P<name>io|mem) (?P<flags>0x[0-9a-fA-F]+)$"
)


def classify_untyped_window(flags):
    """Classify one bridge-window flags word.

    A window that already has IORESOURCE_IO or IORESOURCE_MEM is typed.
    A typeless window that sizing claimed has IORESOURCE_DISABLED,
    IORESOURCE_STARTALIGN, or IORESOURCE_UNSET. Anything else was left
    alone.
    """
    if flags & IORESOURCE_TYPE_BITS:
        return "typed"
    if flags & _CLAIMED_BITS:
        return "claimed"
    return "clear"


def parse_probe_report(text):
    """Return (bdf, slot, flags) rows from the probe proc file."""
    windows = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = _PROBE_LINE.match(line)
        if match is None:
            raise ValueError(
                f"Unreadable bridge-window probe line {line_no}: {raw}"
            )
        windows.append((
            match.group("bdf"),
            match.group("name"),
            int(match.group("flags"), 16),
        ))
    return windows


def _unload_bridge_window_probe(logger):
    """Unload the probe if it is still loaded."""
    if not os.path.exists(f"/sys/module/{_PROBE_MODULE}"):
        return
    try:
        subprocess.run(
            ["rmmod", _PROBE_MODULE],
            check=True,
            capture_output=True,
            text=True,
            timeout=_MODULE_TIMEOUT_SECS,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip() or "rmmod failed"
        logger.error("Failed to unload %s: %s", _PROBE_MODULE, detail)
    except (subprocess.TimeoutExpired, OSError) as error:
        logger.error("Failed to unload %s: %s", _PROBE_MODULE, error)


def _clean_bridge_window_probe(logger):
    """Remove build products. A clean failure does not hide the verdict."""
    try:
        glt_utils.run_command(
            ["make", "clean"],
            "Failed to clean the bridge-window probe",
            timeout=_MAKE_TIMEOUT_SECS,
        )
    except RuntimeError as error:
        logger.error("%s", error)


def _bridge_window_tooling_skip_reason():
    """Return why this host cannot build the probe, or None."""
    for command in ("make", "insmod", "rmmod"):
        if not glt_utils.is_cli_available(command):
            return (
                f"{command} is unavailable; cannot build or load the "
                "bridge-window probe."
            )
    kernel_build_dir = os.path.join(
        "/lib/modules", platform.uname().release, "build"
    )
    if not os.path.isdir(kernel_build_dir):
        return (
            "Matching running-kernel headers are unavailable: "
            f"{kernel_build_dir}"
        )
    return None


def _build_and_load_bridge_window_probe(logger):
    """Build, sign, and load the probe.

    Return a warning reason when the toolchain or loader cannot run the probe.
    """
    try:
        glt_utils.run_command(
            ["make"],
            "Failed to build the bridge-window probe",
            timeout=_MAKE_TIMEOUT_SECS,
        )
    except RuntimeError as error:
        return f"Cannot build the bridge-window probe: {error}"

    module_ko = f"{_PROBE_MODULE}.ko"
    if not glt_utils.sign_kernel_module(module_ko, logger):
        return (
            f"Cannot sign {module_ko} for Secure Boot. "
            "See docs/user-guides/secure-boot.md for setup."
        )

    try:
        subprocess.run(
            ["insmod", module_ko],
            check=True,
            capture_output=True,
            text=True,
            timeout=_MODULE_TIMEOUT_SECS,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        return (
            "Cannot load the bridge-window probe "
            f"(insmod exit {error.returncode}){suffix}"
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        return f"Cannot load the bridge-window probe: {error}"
    return None


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_pci_bridge_window_type(_, logger):
    """
    Untyped I/O bridge windows must not be sized.

    pbus_select_window_for_type() returned the I/O slot even when the
    resource had no IORESOURCE_IO. Bridge sizing then set
    IORESOURCE_DISABLED or IORESOURCE_STARTALIGN, and a failed
    assignment left IORESOURCE_UNSET. Non-prefetch and prefetch windows
    are typed before sizing, so only the I/O slot is observable. The
    sysfs resource file hides DISABLED and UNSET, so the flags come
    from a probe module.

    PATCH: https://git.kernel.org/torvalds/c/e5f72cb9cea5
           linux-nvidia-6.18-next: 96f42087b058
           PCI: Validate window resource type in pbus_select_window_for_type()
    """
    tooling_skip = _bridge_window_tooling_skip_reason()
    if tooling_skip:
        GLTCore.GLTTestResult.skip(tooling_skip)

    module_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        _PROBE_MODULE,
    )
    if not os.path.isdir(module_dir):
        GLTCore.GLTTestResult.fail(
            f"Bridge window probe sources are missing at {module_dir}."
        )

    module_loaded = False
    windows = []
    with glt_utils.chdir_managed(module_dir):
        try:
            _unload_bridge_window_probe(logger)
            load_skip = _build_and_load_bridge_window_probe(logger)
            if load_skip:
                GLTCore.GLTTestResult.warn(load_skip)
            module_loaded = True
            try:
                with open(_PROBE_PROC, encoding="utf-8") as handle:
                    windows = parse_probe_report(handle.read())
            except (OSError, ValueError) as error:
                GLTCore.GLTTestResult.fail(
                    "Cannot read bridge window flags from the probe: "
                    f"{error}"
                )
        except OSError as error:
            GLTCore.GLTTestResult.fail(f"Bridge window probe failed: {error}")
        finally:
            if module_loaded:
                _unload_bridge_window_probe(logger)
            _clean_bridge_window_probe(logger)

    if not windows:
        GLTCore.GLTTestResult.skip("No PCI-to-PCI bridge found.")

    grouped = {"claimed": [], "clear": []}
    for bdf, name, flags in windows:
        kind = classify_untyped_window(flags)
        if kind in grouped:
            grouped[kind].append(f"{bdf} {name} {flags:#x}")
    claimed = grouped["claimed"]
    clear = grouped["clear"]

    if claimed:
        GLTCore.GLTTestResult.fail(
            "Untyped I/O bridge window was sized "
            f"(missing upstream {_UPSTREAM_SHA} / "
            f"linux-nvidia-6.18-next {_NVIDIA_SHA}): "
            + ", ".join(claimed)
        )
    if not clear:
        GLTCore.GLTTestResult.skip(
            "No PCI bridge has an untyped I/O window; "
            "the window type check is not observable."
        )
    logger.info(
        "Left %s untyped I/O bridge window(s) unsized: %s",
        len(clear),
        ", ".join(clear),
    )
