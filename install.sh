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
  if [ -f /etc/sudoers ] && grep -q "secure_path" /etc/sudoers; then
    if ! grep -q "secure_path.*\/usr\/local\/bin" /etc/sudoers; then
      echo "WARNING: /usr/local/bin not found in sudo secure_path"
      echo "This may cause issues when running 'sudo glt-test-suite'"
      echo
      echo "Options to fix this issue:"
      echo "1. Edit /etc/sudoers with 'sudo visudo' and add /usr/local/bin to secure_path"
      echo "2. Create a symlink: sudo ln -s /usr/local/bin/glt-test-suite /usr/bin/glt-test-suite"
      echo "3. Use the full path when running: sudo /usr/local/bin/glt-test-suite"
      echo
      read -p "Would you like to create a symlink to /usr/bin/glt-test-suite? (y/n): " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating symlink from /usr/local/bin/glt-test-suite to /usr/bin/glt-test-suite..."
        ln -sf /usr/local/bin/glt-test-suite /usr/bin/glt-test-suite
        echo "Symlink created successfully"
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
  elif check_network && check_command zypper; then
    echo "openSUSE/SLES system detected"
    echo "Installing required system packages..."
    zypper --non-interactive refresh
    zypper --non-interactive install python3-pip tar ipmitool perf freeipmi kernel-default-devel linux-glibc-devel libgpiod stress-ng kexec-tools
  elif check_network && check_command yum; then
    echo "Older Red Hat-based system detected"
    echo "Installing required system packages..."
    yum install -y python3-pip tar ipmitool perf freeipmi libgpiod-utils stress-ng kexec-tools
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
    echo "  - Linux kernel headers (matching your running kernel)"
  fi

  if ! check_command pip3; then
    echo "ERROR: pip3 is required but not installed. Please install Python pip manually."
    exit 1
  fi
}

install_greenlit() {
  echo "Installing GreenLiT package and dependencies via pip..."

  # Install from pyproject.toml (pip ≥ 21.3)
  if ! pip3 install --ignore-installed .; then
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

  # Install wheels without internet access
  if ! pip3 install --no-index --find-links "$LOCAL_WHL_DIR" "${WHL_PKGS[@]}"; then
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
}

setup_systemd() {
  echo "Configuring systemd service for GreenLiT..."
  mkdir -p /usr/local/lib/systemd/system

  cp confs/greenlit.service /usr/local/lib/systemd/system/
  sed -i 's|/usr/bin/glt-test-suite|/usr/local/bin/glt-test-suite|' /usr/local/lib/systemd/system/greenlit.service

  cp confs/greenlit.timer /usr/local/lib/systemd/system/

  systemctl daemon-reload
  echo "Systemd service configuration complete"
}

uninstall_greenlit() {
  echo "======================================================="
  echo "           Starting GreenLiT Uninstallation            "
  echo "======================================================="
  echo

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
  if check_command pip3; then
    pip3 uninstall -y greenlit 2>/dev/null || true
  fi

  echo "Removing GreenLiT executables..."
  rm -f /usr/local/bin/glt-test-suite
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
