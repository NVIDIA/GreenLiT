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

"""Blocked wakee process for the SMT-aware scheduler placement probe."""

import ctypes
import os
import sys


def main(argv=None):
    """Block on stdin, then report the logical CPU selected at wakeup."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise ValueError("expected one previous-CPU argument")
    previous_cpu = int(args[0])

    sched_getcpu = ctypes.CDLL(None, use_errno=True).sched_getcpu
    sched_getcpu.argtypes = []
    sched_getcpu.restype = ctypes.c_int

    os.sched_setaffinity(0, {previous_cpu})
    if sched_getcpu() != previous_cpu:
        raise RuntimeError("wakee did not reach its requested previous CPU")

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    if sys.stdin.buffer.read(1) != b"W":
        raise RuntimeError("wakee did not receive its wake token")

    destination_cpu = sched_getcpu()
    if destination_cpu < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))

    sys.stdout.write(f"{destination_cpu}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
