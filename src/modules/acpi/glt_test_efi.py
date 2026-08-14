# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Python script to test EFI runtime services support.
"""
import os
import fcntl
import struct
import subprocess
from common import glt_core as GLTCore
from common import glt_utils

# Define the IOCTL command for EFI_RUNTIME_GET_SUPPORTED_MASK
# From linux/drivers/firmware/efi/test/efi_test.h
# #define EFI_RUNTIME_GET_SUPPORTED_MASK _IOR('p', 0x0C, unsigned int)
#
# _IOR macro creates an ioctl request code for reading data from the device
# 'p' is the magic number, 0x0C is the sequence number, and unsigned int is the data type
EFI_RUNTIME_GET_SUPPORTED_MASK = 0x8004700C  # Correct IOCTL value

def perform_efi_mask_ioctl(logger):
    """
    Perform the IOCTL call to get the EFI runtime services mask

    Args:
        logger: Logger object for logging

    Returns:
        int: The EFI runtime services mask

    Raises:
        IOError: If the IOCTL call fails
        OSError: If there's an OS-level error
        struct.error: If there's an error unpacking the result
    """
    # Load the EFI test module first
    logger.debug("Loading the EFI test module...")
    result = subprocess.run(["modprobe", "efi_test"], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    logger.debug(f"modprobe output: {result.stdout}")

    if not os.path.exists("/dev/efi_test"):
        raise FileNotFoundError(
            "EFI test device file (/dev/efi_test) does not exist after loading the module")

    with open("/dev/efi_test", "rb") as fd:
        buf = bytearray(4)  # unsigned int is 4 bytes

        fcntl.ioctl(fd.fileno(), EFI_RUNTIME_GET_SUPPORTED_MASK, buf)

        # Convert the buffer to an integer - 'I' for unsigned int (32-bit)
        mask = struct.unpack("I", buf)[0]

    logger.debug(f"EFI runtime services supported mask: 0x{mask:x}")
    return mask

_kexec_conditional_reboot_hook = GLTCore.PostHooks.for_conditional_reboot(
    "TriggerKexec",
    glt_utils.is_kexec_boot,
)


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("disruptive")
@GLTCore.GLTPlan("stress")
def test_efi_runtime_services_mask(sd, logger):
    """
    Test EFI runtime services supported mask by performing
    IOCTL EFI_RUNTIME_GET_SUPPORTED_MASK on /dev/efi_test.

    This test requires the system to have booted via kexec to properly
    verify EFI runtime services functionality after a kexec boot.

    This tests whether the kernel's EFI runtime services interface is working correctly
    and reports the services that are supported.

    JIRA: DGX-8217
    PATCH: N/A
    """
    if not glt_utils.is_efi_system():
        GLTCore.GLTTestResult.skip("Skipping test: System is not using EFI")

    ok, reason = glt_utils.handle_kexec_flow(
        test_efi_runtime_services_mask.__name__, sd, logger,
    )
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    try:
        mask = perform_efi_mask_ioctl(logger)

        if mask == 0:
            GLTCore.GLTTestResult.fail("EFI runtime service is not working. " +
                                       "Please update BIOS to the latest version.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to load EFI test module: {e.stderr}")
        GLTCore.GLTTestResult.fail(f"Failed to load EFI test module: {e.stderr}")
    except FileNotFoundError as e:
        logger.error(str(e))
        GLTCore.GLTTestResult.fail(str(e))
    except (IOError, OSError, struct.error) as e:
        logger.error(f"Error accessing or processing EFI data: {e}")
        GLTCore.GLTTestResult.fail(f"Error accessing or processing EFI data: {e}")


test_efi_runtime_services_mask.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_kexec_conditional_reboot_hook,
    postfail=_kexec_conditional_reboot_hook,
    postafail=_kexec_conditional_reboot_hook,
)
