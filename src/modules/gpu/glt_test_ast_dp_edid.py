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

"""ASPEED DisplayPort EDID wait must sleep instead of busy-waiting."""

import os
import re
import subprocess
import tempfile

from common import glt_core as GLTCore
from common import glt_utils

AST_DP_EDID_SLEEP_COMMIT = "c7c31f8dc54aa3c9b2c994b5f1ff7e740a654e97"
AST_DP_EDID_FUNCTION = "ast_astdp_read_edid_block"

_SYM_HEADER_RE = re.compile(r"^[0-9a-f]+\s+<([^>]+)>:")
_MSLEEP_RE = re.compile(r"(?:<msleep(?:@plt)?>)|(?:\bR_[A-Z0-9_]+\s+msleep\b)")
_BUSY_WAIT_RE = re.compile(
    r"(?:<(?:mdelay|__const_udelay|__udelay)(?:@plt)?>)|"
    r"(?:\bR_[A-Z0-9_]+\s+(?:mdelay|__const_udelay|__udelay)\b)"
)
_COMPRESSED_MODULE = {
    ".ko.xz": ("xz", "-dc"),
    ".ko.gz": ("gzip", "-dc"),
    ".ko.zst": ("zstd", "-dc"),
    ".ko.bz2": ("bzip2", "-dc"),
}


def extract_function_disassembly(objdump_text, symbol):
    """Return objdump lines for *symbol*, stopping at the next symbol."""
    start = None
    body = []
    for line in (objdump_text or "").splitlines():
        match = _SYM_HEADER_RE.match(line.strip())
        if match:
            if match.group(1) == symbol:
                start = True
                continue
            if start:
                break
        elif start:
            body.append(line)
    if start is None:
        return None
    return "\n".join(body)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_ast_dp_edid_uses_msleep(_, logger):  # pylint: disable=too-many-branches
    """
    AST DisplayPort EDID reads must wait with msleep rather than mdelay.

    Busy-waiting in mdelay() stalls the CPU and can trip kernel timeouts
    during boot. The patch replaces that wait in ast_astdp_read_edid_block.

    PATCH: https://git.kernel.org/torvalds/c/c7c31f8dc54aa3c9b2c994b5f1ff7e740a654e97
    """
    if not glt_utils.is_cli_available("objdump"):
        GLTCore.GLTTestResult.skip("objdump is not available.")
    if not glt_utils.is_cli_available("modinfo"):
        GLTCore.GLTTestResult.skip("modinfo is not available.")

    try:
        modinfo = glt_utils.run_command(
            ["modinfo", "-n", "ast"], "Cannot locate the ast kernel module"
        )
    except RuntimeError as error:
        GLTCore.GLTTestResult.skip(str(error))

    packed = (modinfo.stdout or "").strip()
    if not packed or packed == "(builtin)":
        GLTCore.GLTTestResult.skip(
            "ast is not a loadable kernel module on this kernel."
        )

    decompress = None
    for suffix, argv in _COMPRESSED_MODULE.items():
        if packed.endswith(suffix):
            decompress = argv
            break
    if decompress is None and not packed.endswith(".ko"):
        GLTCore.GLTTestResult.skip(
            "ast is not a loadable kernel module on this kernel."
        )
    if not os.path.isfile(packed):
        GLTCore.GLTTestResult.skip(
            "ast is not a loadable kernel module on this kernel."
        )
    if decompress is not None and not glt_utils.is_cli_available(decompress[0]):
        GLTCore.GLTTestResult.skip(
            f"{decompress[0]} is not available to decompress ast."
        )

    unpacked = None
    try:
        module_path = packed
        if decompress is not None:
            handle, unpacked = tempfile.mkstemp(prefix="glt-ast-", suffix=".ko")
            os.close(handle)
            try:
                with open(unpacked, "wb") as out:
                    subprocess.run(
                        [*decompress, packed],
                        stdout=out,
                        stderr=subprocess.DEVNULL,
                        check=True,
                        timeout=60,
                    )
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                GLTCore.GLTTestResult.skip(
                    "Cannot decompress the ast kernel module."
                )
            module_path = unpacked

        try:
            dumped = glt_utils.run_command(
                ["objdump", "-dr", module_path],
                "Cannot disassemble the ast kernel module",
                timeout=120,
            )
        except RuntimeError as error:
            GLTCore.GLTTestResult.skip(str(error))

        body = extract_function_disassembly(dumped.stdout, AST_DP_EDID_FUNCTION)
        if body is None:
            GLTCore.GLTTestResult.skip(
                "ASPEED DisplayPort EDID reader is not present in the ast module."
            )
        if _MSLEEP_RE.search(body) is None or _BUSY_WAIT_RE.search(body):
            GLTCore.GLTTestResult.fail(
                "AST DisplayPort EDID wait still busy-waits. Include kernel commit "
                f"{AST_DP_EDID_SLEEP_COMMIT} to sleep instead of mdelay."
            )
        logger.info(
            "ast_astdp_read_edid_block in %s calls msleep", packed
        )
    finally:
        if unpacked is not None:
            try:
                os.unlink(unpacked)
            except OSError:
                pass
