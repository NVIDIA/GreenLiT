# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
The utilities for GreenLit
"""

# pylint: disable=consider-using-with
from contextlib import contextmanager
from datetime import datetime
import glob
import os
import platform
import re
import shutil
import signal
import subprocess

from common.nv_device import NvDev


# Default location for MOK (Machine Owner Key) files used to sign
# kernel modules when Secure Boot is enabled.
# This is the Ubuntu default path used by dkms and shim-signed.
# See docs/user-guides/secure-boot.md for setup instructions.
MOK_KEY_DIR = "/var/lib/shim-signed/mok"
MOK_PRIVATE_KEY = os.path.join(MOK_KEY_DIR, "MOK.priv")
MOK_CERTIFICATE = os.path.join(MOK_KEY_DIR, "MOK.der")


def is_secure_boot_enabled():
    """Check if UEFI Secure Boot is enabled.

    Tries mokutil first (most reliable), falls back to reading the
    EFI variable directly from sysfs.

    Returns:
        bool: True if Secure Boot is enabled, False otherwise.
    """
    # Prefer mokutil -- works regardless of sysfs permissions
    if shutil.which("mokutil"):
        try:
            result = subprocess.run(
                ["mokutil", "--sb-state"], check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "SecureBoot enabled" in result.stdout:
                return True
            if "SecureBoot disabled" in result.stdout:
                return False
        except OSError:
            pass  # mokutil failed, try sysfs fallback

    # Fallback: read EFI variable directly from efivarfs.
    # Format: 4-byte uint32 attribute header + variable data.
    # Per UEFI spec (Section 32.6.1), SecureBoot is a single UINT8:
    #   data[0:4] = attributes, data[4] = 0x01 (enabled) or 0x00.
    sb_path = "/sys/firmware/efi/efivars/" \
              "SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c"
    try:
        with open(sb_path, "rb") as f:
            data = f.read()
            return len(data) >= 5 and data[4] == 1
    except (IOError, PermissionError, FileNotFoundError):
        return False


def has_mok_keys():
    """Check if MOK signing keys are available.

    Returns:
        bool: True if both MOK.priv and MOK.der exist.
    """
    return os.path.isfile(MOK_PRIVATE_KEY) and \
        os.path.isfile(MOK_CERTIFICATE)


# Well-known path where users can place sign-file manually (e.g. from
# Buildroot or a custom kernel build).
SIGN_FILE_OVERRIDE = "/usr/local/bin/sign-file"


def _find_sign_file():
    """Find the sign-file utility from the kernel build system.

    Search order:
      1. User-provided override at /usr/local/bin/sign-file
      2. PATH lookup
      3. Distro kernel header paths (Ubuntu, RHEL, Debian kbuild)
      4. Glob fallback across all installed header versions

    Returns:
        str or None: Path to sign-file if found, None otherwise.
    """
    # 1. User override (for Buildroot / custom kernels)
    if os.path.isfile(SIGN_FILE_OVERRIDE) and \
            os.access(SIGN_FILE_OVERRIDE, os.X_OK):
        return SIGN_FILE_OVERRIDE

    # 2. PATH lookup
    sign_file_path = shutil.which("sign-file")
    if sign_file_path:
        return sign_file_path

    # 3. Distro-specific paths
    kernel_version = platform.release()
    candidates = [
        f"/usr/src/linux-headers-{kernel_version}/scripts/sign-file",
        f"/usr/src/kernels/{kernel_version}/scripts/sign-file",
        "/usr/lib/linux-kbuild/scripts/sign-file",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # 4. Glob fallback across all installed header versions
    matches = glob.glob("/usr/src/linux-headers-*/scripts/sign-file")
    matches += glob.glob("/usr/src/kernels/*/scripts/sign-file")
    if matches:
        return sorted(matches)[-1]  # Use the latest version

    return None


def sign_kernel_module(module_path, logger=None):
    """Sign a kernel module with MOK key for Secure Boot.

    If Secure Boot is not enabled, this is a no-op and returns True.
    If Secure Boot is enabled but MOK keys are not found, returns False
    with guidance.

    Args:
        module_path: Path to the .ko file to sign.
        logger: Optional logger for debug output.

    Returns:
        bool: True if signing succeeded or was not needed.
    """
    if not is_secure_boot_enabled():
        if logger:
            logger.debug("Secure Boot not enabled, skipping module signing.")
        return True

    if not has_mok_keys():
        if logger:
            logger.error(
                "Secure Boot is enabled but MOK keys not found at %s. "
                "See docs/user-guides/secure-boot.md for setup instructions.",
                MOK_KEY_DIR)
        return False

    sign_file = _find_sign_file()
    if not sign_file:
        if logger:
            logger.error(
                "Cannot find 'sign-file' utility. Either install kernel "
                "headers (apt install linux-headers-$(uname -r) / "
                "dnf install kernel-devel-$(uname -r)), or copy sign-file "
                "from your kernel build tree to %s",
                SIGN_FILE_OVERRIDE)
        return False

    cmd = [sign_file, "sha256", MOK_PRIVATE_KEY, MOK_CERTIFICATE, module_path]
    if logger:
        logger.debug("Signing module: %s", cmd)

    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error("Module signing failed: %s", e.stderr)
        return False

    return True


def arm_for_reset(testcase, sd):
    """Arm the system for potential hardware reset"""
    sd.progress_record_time(testcase, "MightReset")
    start_systemd_timer()
    systemd_enable()
    # Send SIGINT to GLT to save the progress for future use.
    # If hardware reset occurs, the service will resume the test
    # If it doesn't happen, the timer starts GLT to continue the test
    # Use shell=False to prevent command injection (CWE-78)
    subprocess.run(["pkill", "-INT", "glt-test-suite"], shell=False, check=False)
    return False


@contextmanager
def chdir_managed(path):
    """
    Resource managed chdir
    Also updates PWD environment variable for subprocess compatibility
    """
    old_cwd = os.getcwd()
    old_pwd = os.environ.get('PWD')
    os.chdir(path)
    os.environ['PWD'] = os.getcwd()  # Update PWD for subprocess.run() with shell=False
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if old_pwd is not None:
            os.environ['PWD'] = old_pwd
        elif 'PWD' in os.environ:
            del os.environ['PWD']


def check_kernel_log(msg) -> bool:
    """
    DGX-10357
    Check kernel log
    Return True if msg is in kernel log
    """
    if is_cli_available("journalctl"):
        # Use shell=False to prevent command injection (CWE-78)
        result = subprocess.run(["journalctl", "-b0", "-k"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and msg in result.stdout:
            return True

    # FIXME: DGX-10224 pylint: disable=fixme
    elif is_cli_available("dmesg"):
        # Use shell=False to prevent command injection (CWE-78)
        result = subprocess.run(["dmesg"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and msg in result.stdout:
            return True

    return False


def collect_logs(directory) -> tuple[bool, str]:
    """
    Collect logs under specified directory.
    Return False if failed to collect logs.
    """
    if directory is None:
        return (False, "directory is empty")
    if not is_cli_available("nvidia-bug-report.sh"):
        return (False, "nvidia-bug-report.sh is unavailable")

    try:
        # Use shell=False to prevent command injection (CWE-78)
        output_file = f"{directory}/glt-nvidia-bug-report"
        subprocess.run(["nvidia-bug-report.sh", "--safe-mode", "--output-file", output_file],
                       shell=False, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, PermissionError, FileNotFoundError, OSError) as e:
        return (False, f"nvidia-bug-report.sh excuting failed: {e}")

    return (True, "")


def compare_version(version1, version2):
    """
    Compares two version strings using either rpmdev-vercmp,
    dpkg --compare-versions or basic version comparison.

    Args:
        version1 (str): The first version string.
        version2 (str): The second version string.

    Returns:
        0 if versions are equal.
        1 if version1 > version2.
        -1 if version1 < version2.
        None if an error occurred.
    """
    result = None

    # Try dpkg --compare-versions first
    if is_cli_available("dpkg"):
        return compare_version_dpkg(version1, version2)

    #  Try rpmdev-vercmp if available and dpkg comparison failed
    if is_cli_available("rpmdev-vercmp") and result is None:
        return compare_version_rpm(version1, version2)

    # Fallback to basic linux kernel version comparison if other methods failed
    if result is None:
        v1 = parse_linux_version(version1)
        v2 = parse_linux_version(version2)

        # Compare components one by one
        for i in range(max(len(v1), len(v2))):
            # Treat missing components as lower than any existing component
            if i >= len(v1):
                result = -1
                break
            if i >= len(v2):
                result = 1
                break

            # Compare components
            if v1[i] != v2[i]:
                # For numeric components, compare as numbers
                if isinstance(v1[i], int) and isinstance(v2[i], int):
                    result = 1 if v1[i] > v2[i] else -1
                # For string components or mixed, compare as strings
                else:
                    str1 = str(v1[i])
                    str2 = str(v2[i])
                    result = 1 if str1 > str2 else -1
                break

        # If we get here and result is still None, versions are equal
        if result is None:
            result = 0

    return result


def compare_version_dpkg(version1, version2):
    """
    Compares two version strings using dpkg --compare-versions.

    Args:
        version1 (str): The first version string.
        version2 (str): The second version string.

    Returns:
        0 if versions are equal.
        1 if version1 > version2.
        -1 if version1 < version2.
        None if an error occurred.
    """
    result = None

    # Use shell=False to prevent command injection (CWE-78)
    gt_result = subprocess.run(["dpkg", "--compare-versions", version1, "gt", version2],
                               shell=False, check=False)
    lt_result = subprocess.run(["dpkg", "--compare-versions", version1, "lt", version2],
                               shell=False, check=False)
    eq_result = subprocess.run(["dpkg", "--compare-versions", version1, "eq", version2],
                               shell=False, check=False)

    if gt_result.returncode == 0:
        result = 1
    elif lt_result.returncode == 0:
        result = -1
    elif eq_result.returncode == 0:
        result = 0
    return result


def compare_version_rpm(version1, version2):
    """
    Compares two version strings using rpmdev-vercmp.

    Args:
        version1 (str): The first version string.
        version2 (str): The second version string.

    Returns:
        0 if versions are equal.
        1 if version1 > version2.
        -1 if version1 < version2.
    """
    result = None
    # Use shell=False to prevent command injection (CWE-78)
    cmd_result = subprocess.run(["rpmdev-vercmp", version1, version2], shell=False, check=False)
    if cmd_result.returncode == 0:
        result = 0
    elif cmd_result.returncode == 11:
        result = 1
    elif cmd_result.returncode == 12:
        result = -1
    return result


def create_cache_session() -> str:
    """Create cache directory for test session"""
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    cache_dir = f"/var/cache/greenlit/{current_time}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_session_list() -> list:
    """Returns a list of cached session path"""
    cache_dir = "/var/cache/greenlit"
    entries = os.listdir(cache_dir)
    files = [os.path.join(cache_dir, f) for f in entries if os.path.isdir(os.path.join(cache_dir, f))]
    return sorted(files)


def get_cpu_model_name(logger) -> str | None:
    """
    Get the CPU model name from lscpu output.

    Args:
        logger: Logger instance to log warnings.

    Returns:
        str: The CPU model name, or None if not found or lscpu unavailable.
    """
    if not is_cli_available("lscpu"):
        logger.warning("lscpu not found. Install util-linux package.")
        return None

    result = subprocess.run("lscpu", text=True, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.warning("Failed to run lscpu: %s", result.stderr)
        return None

    match = re.search(r"Model name:\s+(.+)", result.stdout)
    if match:
        return match.group(1).strip()
    return None


def is_cli_available(cli) -> bool:
    """Check a cli available"""
    path = shutil.which(cli)
    return path is not None


def is_efi_system():
    """
    Check if the system is using EFI.

    Returns:
        bool: True if the system uses EFI, False otherwise
    """
    return os.path.exists("/sys/firmware/efi")


def is_kexec_boot():
    """
    Check if system has booted via our kexec path by looking for the
    GLT-specific marker in the kernel command line.
    """
    try:
        with open("/proc/cmdline", "r", encoding="utf-8") as f:
            cmdline = f.read().strip()
    except FileNotFoundError:
        return False

    return "glt_kexec_boot=1" in cmdline


def log_system_info(logger):
    """
    Log system information such as distro, architecture, kernel version and
    BaseOS details
    """
    release_info = NvDev.get_release_info()

    distro_name = release_info.get("NAME", "Unknown")
    distro_version = release_info.get("VERSION_ID", "Unknown")
    architecture = platform.machine()
    kernel_version = platform.release()

    logger.info("===== System Information =====")
    logger.info("Distribution: %s", distro_name)
    logger.info("Release: %s", distro_version)
    logger.info("Architecture: %s", architecture)
    logger.info("Kernel: %s", kernel_version)

    logger.info("DMI:")
    cmd = "grep . /sys/class/dmi/id/* 2>/dev/null || true"
    result = subprocess.run(cmd, shell=True, check=False,
                            stdout=subprocess.PIPE,
                            text=True)

    # Process output even if the command returns non-zero
    dmi_info = result.stdout.strip()
    if dmi_info:
        # Log each line of the DMI information
        for line in dmi_info.split('\n'):
            logger.info(line)
    else:
        logger.warning("No DMI information available")

    try:
        with open('/etc/baseos-release', 'r', encoding='utf-8') as f:
            logger.info("BaseOS Information:")
            for line in f:
                logger.info(line.strip())

        with open('/etc/l4t-release', 'r', encoding='utf-8') as f:
            logger.info("L4T Information:")
            for line in f:
                logger.info(line.strip())

        with open('/etc/platform-release', 'r', encoding='utf-8') as f:
            logger.info("Platform Information:")
            for line in f:
                logger.info(line.strip())
    except (FileNotFoundError, PermissionError):
        pass

    logger.info("============================")


def parse_linux_version(version):
    """Parse Linux kernel version string into comparable components.

    Examples:
    - 5.15.0-91-generic -> [5, 15, 0, 91, 'generic']
    - 6.1.0-18-amd64 -> [6, 1, 0, 18, 'amd64']
    - 3.10.0-1160.108.1.el7 -> [3, 10, 0, 1160, 108, 1, 'el7']
    """
    # First split on main delimiters
    parts = []
    # Split version string into main version and distribution parts
    version_parts = version.split('-', 1)

    # Handle the main version numbers (x.y.z)
    parts.extend(int(x) for x in version_parts[0].split('.'))

    # Handle distribution-specific parts if they exist
    if len(version_parts) > 1:
        for part in version_parts[1].split('.'):
            for subpart in part.split('-'):
                try:
                    parts.append(int(subpart))
                except ValueError:
                    # Keep non-numeric parts (like 'generic', 'amd64', 'el7')
                    parts.append(subpart)

    return parts


def reboot(testcase, sd):
    """Reboot system with saving status"""
    sd.progress_record_time(testcase, "TriggerReboot")
    if systemd_enable():
        os.system("reboot")
        os._exit(0)
    else:
        # If systemd unavailable, guess in development mode,
        # Send SIGTERM to self to emulate system off.
        os.kill(os.getpid(), signal.SIGTERM)
    return False


# pylint: disable=too-many-return-statements,too-many-branches
def reboot_with_kexec(testcase, sd, custom_cmdline=None) -> tuple[bool, str | None]:
    """
    Reboot using kexec, recording progress so the test can resume after boot.
    On success, triggers the kexec reboot and does not return (process exits
    via systemctl kexec, or SIGTERM in dev mode).

    Args:
        custom_cmdline: Optional string of kernel options to append to the
            command line (whitespace stripped). Must not contain newline.

    Returns:
        (False, reason): On failure. reason describes what went wrong; caller
                         should treat as test failure or skip.
        (True, None): On success immediately exits before triggering reboot (caller
                      normally does not see this).
    """
    if not is_cli_available("kexec"):
        return (False, "kexec CLI not found, please install kexec-tools")

    if custom_cmdline is not None and any(c in custom_cmdline for c in "\n\r\0"):
        return False

    sd.progress_record_time(testcase, "TriggerKexec")

    try:
        kernel_version = platform.release()
        initrd_path = None
        vmlinuz_path = None

        for file in glob.glob(f"/boot/*{kernel_version}*"):
            if "kdump" in file or "rescue" in file:
                continue
            if initrd_path is None and ("initramfs" in file or "initrd" in file):
                initrd_path = file
            elif vmlinuz_path is None and "vmlinuz" in file:
                vmlinuz_path = file

        if not initrd_path:
            return (False, "No matching initrd/initramfs in /boot for current kernel")
        if not vmlinuz_path:
            return (False, "No matching vmlinuz in /boot for current kernel")

        try:
            with open("/proc/cmdline", "r", encoding="utf-8") as f:
                current_cmdline = f.read().strip()
        except (FileNotFoundError, PermissionError):
            return (False, "Cannot read /proc/cmdline")

        kexec_cmdline = " ".join(
            filter(None, [current_cmdline, (custom_cmdline or "").strip(), "glt_kexec_boot=1"])
        )

        # Use shell=False to prevent command injection (CWE-78)
        subprocess.run(
            [
                "kexec",
                "-l",
                vmlinuz_path,
                f"--initrd={initrd_path}",
                f"--command-line={kexec_cmdline}"
            ],
            shell=False,
            check=True,
            capture_output=True,
            text=True
        )

        kexec_loaded_path = "/sys/kernel/kexec_loaded"
        try:
            with open(kexec_loaded_path, "r", encoding="utf-8") as f:
                if f.read().strip() != "1":
                    return (False, "Kernel not loaded for kexec after load attempt")
        except FileNotFoundError:
            return (False, "kexec_loaded sysfs not available")

        if systemd_enable():
            os.system("systemctl kexec")
            # systemctl kexec will ask systemd to perform a kexec reboot,
            # The request returns as soon as it's sent. The actual reboot
            # happens later. Use os._exit(0) to prevent the function from
            # returning so the caller test won't be marked as Completed
            os._exit(0)
        else:
            # If systemd unavailable, guess in development mode,
            # Send SIGTERM to self to emulate system off.
            # TODO: Do not silently kill the process.
            # Consider passing logger to record the error.
            os.kill(os.getpid(), signal.SIGTERM)

    except subprocess.CalledProcessError as e:
        return (False, f"{e.stderr}")

    return (True, None)


def handle_kexec_flow(test_name, sd, logger, custom_cmdline=None):
    """
    Generic kexec boot flow handler.

    Manages the three-phase kexec lifecycle:
      1. If a previous kexec was triggered but the system didn't kexec-boot, fail.
      2. If not yet kexec-booted, initiate kexec (process exits, does not return).
      3. If kexec boot confirmed, disable the systemd service and signal proceed.

    Args:
        test_name:      The test function __name__ (for progress tracking).
        sd:             Scheduler object.
        logger:         Logger object.
        custom_cmdline: Optional extra kernel cmdline args for kexec.

    Returns:
        (True, None)    - kexec boot verified, proceed with test.
        (False, reason) - failure, reason describes what went wrong.
    """
    if sd.progress_check(test_name, "TriggerKexec") and not is_kexec_boot():
        return (False, "System did not boot via kexec as expected")

    if not is_kexec_boot():
        logger.info("System has not booted via kexec. Initiating kexec boot...")
        ok, reason = reboot_with_kexec(test_name, sd, custom_cmdline=custom_cmdline)
        if not ok:
            return (False, reason)
        return (False, "kexec reboot was triggered but process did not exit")

    logger.info("System has booted via kexec, proceeding with test...")
    systemd_enable(enable=False)
    return (True, None)


def reboot_with_reisub(testcase, sd):
    """
    Magic reboot system with saving status
    Useful when there's kernel process lockup
    """
    sd.progress_record_time(testcase, "REISUB")
    if systemd_enable():
        # Not using 'e' (SIGTERM) or 'i' (SIGKILL) so the REISUB process stays
        # alive. 5 seconds delay to let glt-test-suite dump the stack
        # shell=True required for sh -c with multiple commands; command is hardcoded with no user input
        subprocess.Popen(r"""sh -c 'sleep 5; \
                         echo s > /proc/sysrq-trigger; \
                         echo u > /proc/sysrq-trigger; \
                         echo b > /proc/sysrq-trigger'""",
                         start_new_session=True, restore_signals=False, shell=True)

        # Use shell=False to prevent command injection (CWE-78)
        subprocess.run(["pkill", "-INT", "glt-test-suite"], shell=False, check=False)
    else:
        # If systemd unavailable, guess in development mode,
        # Send SIGTERM to self to emulate system off.
        os.kill(os.getpid(), signal.SIGTERM)

    return False


def start_systemd_timer(timer="greenlit", start=True):
    """Start or stop systemd timer."""
    if start:
        operation = "start"
    else:
        operation = "stop"
    timer_unit = f"{timer}.timer"
    # Use shell=False to prevent command injection (CWE-78)
    p = subprocess.run(["systemctl", operation, timer_unit], shell=False, check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(p.returncode == 0)


def systemd_enable(service="greenlit", enable=True):
    """Enable or Disable systemd service."""
    if enable:
        operation = "enable"
    else:
        operation = "disable"
    # Use shell=False to prevent command injection (CWE-78)
    p = subprocess.run(["systemctl", operation, service], shell=False, check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(p.returncode == 0)


def is_systemd_running():
    """Check if the system is running with systemd."""
    try:
        with open('/proc/1/comm', 'r', encoding='utf-8') as f:
            init_process = f.read().strip()
            return init_process == 'systemd'
    except (FileNotFoundError, IOError):
        return False
