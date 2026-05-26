#!/bin/bash
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

set -e

UNINSTALL=false
LOCAL_WHL_DIR=""
PYTHON_CMD="${PYTHON_CMD:-python3}"
INSTALL_CONFIG="/usr/share/greenlit/install.env"
SAVED_PYTHON=""
SAVED_SCRIPTS=""

# Default configuration for system-level installation
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"

# Function to check networking available or not
check_network() {
  # -c 1: Send 1 packet
  # -W 1: Wait max 1 second
  # >/dev/null: Silence output
  ping -c 1 -W 1 8.8.8.8 >/dev/null 2>&1
}

# Function to check command availability
check_command() {
  if ! command -v "$1" &> /dev/null; then
    return 1
  fi
  return 0
}

ensure_python_cmd() {
  if ! check_command "${PYTHON_CMD}"; then
    echo "ERROR: Python command '${PYTHON_CMD}' is not available."
    echo "Set PYTHON_CMD to a Python >= 3.9 interpreter if needed."
    exit 1
  fi

  PYTHON_CMD="$(command -v "${PYTHON_CMD}")"

  if ! "${PYTHON_CMD}" - <<'PY' >/dev/null 2>&1
import sys
if sys.version_info < (3, 9):
    raise SystemExit(1)
PY
  then
    echo "ERROR: ${PYTHON_CMD} must be Python >= 3.9 for GreenLiT installation."
    echo "Set PYTHON_CMD to a supported interpreter, e.g. PYTHON_CMD=python3.11."
    exit 1
  fi

  if ! "${PYTHON_CMD}" -m pip --version >/dev/null 2>&1; then
    echo "ERROR: pip is required for ${PYTHON_CMD}."
    echo "Please install pip for ${PYTHON_CMD} and rerun install.sh."
    exit 1
  fi

  SAVED_PYTHON="${PYTHON_CMD}"
  SAVED_SCRIPTS="$("${PYTHON_CMD}" - <<'PY'
import sysconfig
print(sysconfig.get_path("scripts"))
PY
)"
  echo "Using ${PYTHON_CMD} for GreenLiT Python package installation"
}

load_install_config() {
  if [ -f "${INSTALL_CONFIG}" ]; then
    # shellcheck disable=SC1090
    . "${INSTALL_CONFIG}"
  fi
}

write_install_config() {
  mkdir -p /usr/share/greenlit
  {
    printf 'SAVED_PYTHON=%q\n' "${SAVED_PYTHON:-${PYTHON_CMD}}"
    printf 'SAVED_SCRIPTS=%q\n' "${SAVED_SCRIPTS:-/usr/local/bin}"
  } > "${INSTALL_CONFIG}"
}

enable_libvirt_services() {
  if ! check_command virsh || ! check_command systemctl; then
    return
  fi

  if systemctl enable --now libvirtd virtlogd virtlockd 2>/dev/null; then
    return
  fi
  if systemctl enable --now virtqemud virtlogd virtlockd 2>/dev/null; then
    return
  fi

  echo "WARNING: Unable to enable libvirt services automatically."
  echo "WARNING: Please start libvirtd or virtqemud before running virtualization tests."
}

# Function to print usage information
print_usage() {
  echo "GreenLiT Installation Script"
  echo "Usage: $0 [--help] [--uninstall] [--local-whl DIR_PATH]"
  echo
  echo "Options:"
  echo "  --help                Display this help message and exit"
  echo "  --uninstall           Uninstall GreenLiT from the system"
  echo "  --local-whl DIR_PATH  Install from local wheel files in specified directory"
  echo
  echo "GreenLiT installs system-wide by default for kernel testing"
}

# Function to check if running with root privileges
check_root_privileges() {
  if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script requires root privileges to install system components"
    echo "Please run with sudo or as root user"
    exit 1
  fi
}

# Function to parse command line arguments
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --help)
        print_usage
        exit 0
        ;;
      --uninstall)
        UNINSTALL=true
        shift
        ;;
      --local-whl)
        if [ -z "$2" ]; then
          echo "ERROR: --local-whl requires a directory path"
          print_usage
          exit 1
        fi
        LOCAL_WHL_DIR="$2"
        shift 2
        ;;
      *)
        echo "ERROR: Unknown option: $1"
        print_usage
        exit 1
        ;;
    esac
  done
}

# Function to check and update sudo secure_path
check_sudo_secure_path() {
  echo "Checking sudo secure_path configuration..."
  local scripts_dir="${SAVED_SCRIPTS:-/usr/local/bin}"
  if [ -f /etc/sudoers ] && grep -q "secure_path" /etc/sudoers; then
    if ! grep -q "secure_path.*${scripts_dir}" /etc/sudoers; then
      echo "WARNING: ${scripts_dir} not found in sudo secure_path"
      echo "This may cause issues when running 'sudo glt-test-suite'"
      echo
      echo "Options to fix this issue:"
      echo "1. Edit /etc/sudoers with 'sudo visudo' and add ${scripts_dir} to secure_path"
      echo "2. Create a symlink: sudo ln -s ${scripts_dir}/glt-test-suite /usr/bin/glt-test-suite"
      echo "3. Use the full path when running: sudo ${scripts_dir}/glt-test-suite"
      echo
      if [[ -t 0 ]]; then
        read -p "Would you like to create a symlink to /usr/bin/glt-test-suite? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
          echo "Creating symlink from ${scripts_dir}/glt-test-suite to /usr/bin/glt-test-suite..."
          ln -sf "${scripts_dir}/glt-test-suite" /usr/bin/glt-test-suite
          echo "Symlink created successfully"
        fi
      else
        echo "Skipping symlink prompt because stdin is not interactive."
      fi
    fi
  fi
}

# Function to install OS-specific prerequisites
install_prerequisites() {
  echo "Detecting system package manager..."
  if check_network && check_command apt; then
    echo "Debian/Ubuntu-based system detected"
    echo "Installing required system packages..."
    apt update
    apt install -y python3-pip tar build-essential # common tools
    apt install -y gpiod # gpiodetect
    apt install -y freeipmi # bmc-info
    apt install -y ipmitool linux-tools-common
    apt install -y stress-ng # TPM QSPI interrupt timeout test and other stress tests
    apt install -y kexec-tools
    apt install -y libvirt-clients libvirt-daemon-system qemu-system-arm qemu-utils # virsh/libvirt tests

    # Try to install kernel headers with error handling
    if ! apt install -y linux-headers-"$(uname -r)"; then
      echo "WARNING: Failed to install linux-headers-$(uname -r)"
      echo "This could be due to a custom kernel or missing repository."
      echo
      echo "Please try one of the following options:"
      echo "1. Install the generic headers: sudo apt install linux-headers-generic"
      echo "2. For HWE kernel: sudo apt install linux-headers-generic-hwe-$(lsb_release -rs)"
      echo "3. Search available headers: apt search linux-headers-$(uname -r | cut -d'-' -f1)"
    fi
  elif check_network && check_command dnf; then
    echo "Red Hat/Fedora-based system detected"
    echo "Installing required system packages..."
    dnf install -y python3-pip tar ipmitool perf freeipmi kernel-devel kernel-headers libgpiod-utils stress-ng kexec-tools
    dnf install -y libvirt-client libvirt-daemon libvirt-daemon-driver-qemu qemu-kvm qemu-img
  elif check_network && check_command zypper; then
    echo "openSUSE/SLES system detected"
    echo "Installing required system packages..."
    zypper --non-interactive refresh
    zypper --non-interactive install python3-pip tar ipmitool perf freeipmi kernel-default-devel linux-glibc-devel libgpiod stress-ng kexec-tools
    zypper --non-interactive install libvirt-client libvirt-daemon libvirt-daemon-qemu qemu-arm qemu-tools
  elif check_network && check_command yum; then
    echo "Older Red Hat-based system detected"
    echo "Installing required system packages..."
    yum install -y python3-pip tar ipmitool perf freeipmi libgpiod-utils stress-ng kexec-tools
    yum install -y libvirt-client libvirt-daemon libvirt-daemon-driver-qemu qemu-kvm qemu-img
  else
    echo "WARNING: Unable to detect package manager or network unavailable."
    echo "WARNING: Manual package installation required."
    echo "Please ensure the following packages are installed on your system:"
    echo "  - python3-pip"
    echo "  - tar"
    echo "  - ipmitool"
    echo "  - perf/linux-tools"
    echo "  - freeipmi"
    echo "  - stress-ng"
    echo "  - kexec-tools"
    echo "  - virsh/libvirt and qemu-system-aarch64 or qemu-kvm support"
    echo "  - Linux kernel headers (matching your running kernel)"
  fi

  enable_libvirt_services

  ensure_python_cmd
}

uninstall_existing_greenlit() {
  if "${PYTHON_CMD}" -m pip show greenlit >/dev/null 2>&1; then
    echo "Uninstalling existing GreenLiT package via pip..."
    if ! "${PYTHON_CMD}" -m pip uninstall -y greenlit; then
      echo "ERROR: Existing GreenLiT package could not be uninstalled."
      exit 1
    fi
  fi
}

install_greenlit() {
  echo "Installing GreenLiT package and dependencies via pip..."

  uninstall_existing_greenlit

  # Install from pyproject.toml (pip ≥ 21.3)
  if ! "${PYTHON_CMD}" -m pip install --ignore-installed .; then
    echo "ERROR: pip installation failed. Please check your Python environment."
    exit 1
  fi

  echo "GreenLiT package installation via pip complete"
}

# Install from local wheel files
install_from_local_wheels() {
  if [ ! -d "$LOCAL_WHL_DIR" ]; then
    echo "ERROR: Wheel directory '$LOCAL_WHL_DIR' does not exist"
    exit 1
  fi

  echo "Installing from local wheels in: $LOCAL_WHL_DIR"

  # Get unique package names from wheel files into array
  mapfile -t WHL_PKGS < <(find "$LOCAL_WHL_DIR" -name "*.whl" -exec basename {} \; | cut -d'-' -f1 | sort -u)

  if [ ${#WHL_PKGS[@]} -eq 0 ]; then
    echo "ERROR: No wheel files found in '$LOCAL_WHL_DIR'"
    exit 1
  fi

  echo "Installing packages: ${WHL_PKGS[*]}"

  if [[ " ${WHL_PKGS[*]} " == *" greenlit "* ]]; then
    uninstall_existing_greenlit
  fi

  # Install wheels without internet access
  if ! "${PYTHON_CMD}" -m pip install --ignore-installed --no-index --find-links "$LOCAL_WHL_DIR" "${WHL_PKGS[@]}"; then
    echo "ERROR: Local wheel installation failed"
    exit 1
  fi

  echo "Local wheel installation complete"
}

setup_configuration() {
  echo "Creating GreenLiT configuration directories..."
  mkdir -p /usr/share/greenlit/confs.d
  mkdir -p /usr/share/greenlit/data
  mkdir -p /usr/share/greenlit/executing_dir

  echo "Copying configuration files..."
  cp confs/hardware-manifest.json /usr/share/greenlit/
  cp confs/glt-confs.d/* /usr/share/greenlit/confs.d/
  cp -r data/* /usr/share/greenlit/data/

  echo "Copying external test scripts..."
  if [ -d "external_tests" ]; then
    cp -r external_tests/* /usr/share/greenlit/executing_dir/
    # Ensure all scripts are executable
    find /usr/share/greenlit/executing_dir -type f -name "*.sh" -exec chmod +x {} \;
  fi
  write_install_config
}

setup_systemd() {
  echo "Configuring systemd service for GreenLiT..."
  mkdir -p /usr/local/lib/systemd/system

  cp confs/greenlit.service /usr/local/lib/systemd/system/
  sed -i "s|/usr/bin/glt-test-suite|${SAVED_SCRIPTS:-/usr/local/bin}/glt-test-suite|" \
    /usr/local/lib/systemd/system/greenlit.service

  cp confs/greenlit.timer /usr/local/lib/systemd/system/

  systemctl daemon-reload
  echo "Systemd service configuration complete"
}

uninstall_greenlit() {
  echo "======================================================="
  echo "           Starting GreenLiT Uninstallation            "
  echo "======================================================="
  echo
  load_install_config

  echo "Stopping and disabling GreenLiT systemd service..."
  systemctl disable --now greenlit.service 2>/dev/null || true

  echo "Stopping and disabling GreenLiT systemd timer..."
  systemctl disable --now greenlit.timer 2>/dev/null || true

  echo "Removing systemd service files..."
  rm -f /usr/local/lib/systemd/system/greenlit.service
  rm -f /usr/local/lib/systemd/system/greenlit.timer
  systemctl daemon-reload

  echo "Removing GreenLiT configuration files..."
  rm -rf /usr/share/greenlit

  echo "Uninstalling GreenLiT package..."
  local uninstall_python="${SAVED_PYTHON:-${PYTHON_CMD}}"
  if check_command "${uninstall_python}" && "${uninstall_python}" -m pip --version >/dev/null 2>&1; then
    "${uninstall_python}" -m pip uninstall -y greenlit 2>/dev/null || true
  else
    echo "WARNING: Skipping Python package uninstall because ${uninstall_python} with pip is unavailable."
  fi

  echo "Removing GreenLiT executables..."
  rm -f "${SAVED_SCRIPTS:-/usr/local/bin}/glt-test-suite"
  rm -f /usr/bin/glt-test-suite

  echo
  echo "======================================================="
  echo "           GreenLiT Uninstallation Complete            "
  echo "======================================================="
}

# Function to check Secure Boot status and warn about MOK keys
check_secure_boot() {
  local sb_enabled=false

  # Check via mokutil first
  if check_command mokutil; then
    if mokutil --sb-state 2>/dev/null | grep -q "SecureBoot enabled"; then
      sb_enabled=true
    fi
  else
    # Fallback: read EFI variable directly
    local sb_var="/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c"
    if [ -f "${sb_var}" ]; then
      local val
      val=$(od -An -tx1 -j4 -N1 "${sb_var}" 2>/dev/null | tr -d ' ')
      if [ "${val}" = "01" ]; then
        sb_enabled=true
      fi
    fi
  fi

  if [ "${sb_enabled}" = false ]; then
    return
  fi

  echo
  echo "NOTE: Secure Boot is enabled on this system."
  echo "Some GreenLiT tests build and load custom kernel modules."
  echo "These modules must be signed with a MOK key or they will be rejected."

  local mok_dir="/var/lib/shim-signed/mok"
  if [ -f "${mok_dir}/MOK.priv" ] && [ -f "${mok_dir}/MOK.der" ]; then
    echo "MOK keys found at ${mok_dir} -- module signing is ready."
  else
    echo
    echo "WARNING: No MOK keys found at ${mok_dir}."
    echo "Options:"
    echo "  1. Disable Secure Boot in UEFI/BIOS firmware settings"
    echo "  2. Set up MOK keys: see docs/user-guides/secure-boot.md"
    echo "     Quick setup on Ubuntu:"
    echo "       sudo update-secureboot-policy --new-key"
    echo "       sudo update-secureboot-policy --enroll-key"
  fi
  echo
}

print_completion() {
  echo "======================================================="
  echo "           GreenLiT Installation Complete              "
  echo "======================================================="
  echo
  echo "GreenLiT installed via pip (system-wide)"
  echo
  echo "You can now run the GreenLiT test suite with the command: glt-test-suite"
  echo
  echo "Common usage examples:"
  echo "  List available tests:   sudo glt-test-suite --list --sanity"
  echo "  Run sanity tests:       sudo glt-test-suite --sanity --arm64"
  echo "  Run with verbose mode:  sudo glt-test-suite --verbose --standard --tpm"
  echo
  echo "Kernel patch checker (run from Linux kernel git repo):"
  echo "  # For Grace:"
  echo "  check-kernel-patches -b <branch> /usr/share/greenlit/data/kernel-patches/grace.txt"
  echo "  # For Vera:"
  echo "  check-kernel-patches -b <branch> /usr/share/greenlit/data/kernel-patches/grace.txt /usr/share/greenlit/data/kernel-patches/vera.txt"
  echo
  echo "Kernel config checker:"
  echo "  # For Grace:"
  echo "  check-kernel-config -c /boot/config-\$(uname -r) /usr/share/greenlit/data/kernel-config/grace.txt"
  echo "  # For Vera:"
  echo "  check-kernel-config -c /boot/config-\$(uname -r) /usr/share/greenlit/data/kernel-config/grace.txt /usr/share/greenlit/data/kernel-config/vera.txt"
  echo
  echo "Please run both checkers before running the tests."
  echo
  echo "For more information, refer to the GreenLiT documentation"
}

main() {
  parse_args "$@"
  check_root_privileges

  if [ "$UNINSTALL" = true ]; then
    uninstall_greenlit
    return
  fi

  install_prerequisites

  if [ -n "$LOCAL_WHL_DIR" ]; then
    echo "======================================================="
    echo "      Starting GreenLiT Local Wheels Installation     "
    echo "======================================================="
    echo

    install_from_local_wheels

  else
    echo "======================================================="
    echo "           Starting GreenLiT Installation              "
    echo "======================================================="
    echo

    install_greenlit
  fi

  setup_configuration
  setup_systemd
  check_sudo_secure_path
  check_secure_boot
  print_completion
}

main "$@"
