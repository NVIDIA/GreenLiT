# GreenLiT - Test Framework for Linux-based Distributions

GreenLiT provides automated tests to verify system functionality and stability on Linux-based distributions.

## Index

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Linux Source Tree (Optional)](#linux-source-tree-optional)
  - [Install](#install)
  - [Secure Boot](#secure-boot)
- [Basic Usage](#basic-usage)
  - [Kernel Patch Checker](#kernel-patch-checker-pre-requisite)
  - [Kernel Configuration Checker](#kernel-configuration-checker-pre-requisite)
  - [Running Tests](#running-tests)
  - [Test Plans](#test-plans)
  - [Skipping Tests](#skipping-tests)
- [Test Categories](#test-categories)
- [Test Reports](#test-reports)
- [Recommended workflow](#recommended-workflow)
- [Advanced Usage](#advanced-usage)
  - [Adding External Tests](#adding-external-tests-to-greenlit)
  - [Enabling Kexec Tests](#enabling-kexec-tests-optional)
  - [Building lmbench](#building-lmbench-optional)
  - [Building stress-ng](#building-stress-ng-optional)

## Installation

### Prerequisites

The following packages are required and will be automatically installed by the installation script:

- build-essential/development tools
- Python 3 with pip
- Linux kernel headers
- ipmitool and other hardware utilities
- libvirt and QEMU tools for virtualization tests

If you prefer to install prerequisites manually:

**Debian/Ubuntu**:

```bash
sudo apt update
sudo apt install build-essential python3-pip python3-setuptools tar ipmitool linux-tools-common freeipmi-tools
sudo apt install libvirt-clients libvirt-daemon-system qemu-system-arm qemu-utils
# Install kernel headers (may require specific handling for custom kernels)
sudo apt install linux-headers-$(uname -r)
# If the above fails, try one of these alternatives:
# sudo apt install linux-headers-generic
# sudo apt install linux-headers-generic-hwe-$(lsb_release -rs)
```

Different Ubuntu releases may use different QEMU package names or dependency
sets. Install the package that provides `qemu-system-aarch64` on arm64 systems.

**RHEL/CentOS**:

```bash
sudo dnf install python3-pip python3-setuptools tar ipmitool perf freeipmi kernel-devel kernel-headers
sudo dnf install libvirt-client libvirt-daemon libvirt-daemon-driver-qemu qemu-kvm qemu-img
```

**SUSE**:

```bash
sudo zypper --non-interactive refresh
sudo zypper --non-interactive install python3-pip python3-setuptools tar ipmitool perf freeipmi kernel-default-devel linux-glibc-devel
sudo zypper --non-interactive install libvirt-client libvirt-daemon libvirt-daemon-qemu qemu-arm qemu-tools
```

If libvirt is not socket-activated by the package installation, start it before running virtualization tests:

```bash
sudo systemctl enable --now libvirtd virtlogd virtlockd
# On systems using modular libvirt daemons:
sudo systemctl enable --now virtqemud virtlogd virtlockd
```

You can confirm libvirt access with:

```bash
virsh --connect qemu:///system uri
```

### Linux Source Tree (Optional)

Required for kernel selftests and certain hardware tests.

To set up kernel sources:

1. Obtain the kernel source tarball that matches your running kernel version
2. Extract it to `/usr/local/share/greenlit/linux`:
   ```bash
   sudo mkdir -p /usr/local/share/greenlit
   sudo tar -xf linux-source-*.tar.gz -C /usr/local/share/greenlit/
   sudo mv /usr/local/share/greenlit/linux-* /usr/local/share/greenlit/linux
   ```

### Install

```bash
sudo ./install.sh
```

### Secure Boot

Some GreenLiT tests build and load custom kernel modules at runtime. If your system has **UEFI Secure Boot** enabled, these modules must be signed with a Machine Owner Key (MOK) or the kernel will reject them.

You have two options:

1. **Disable Secure Boot** in the UEFI/BIOS firmware settings, or
2. **Enroll a MOK key** so GreenLiT can sign modules automatically -- see [Secure Boot and Module Signing](secure-boot.md) for step-by-step instructions.

> **Note**: If your system already has MOK keys (e.g., from a DKMS driver installation such as NVIDIA GPU drivers), GreenLiT will reuse them automatically from `/var/lib/shim-signed/mok/`.

### Uninstall

```bash
sudo ./install.sh --uninstall
```
**Note**: Kernel sources are not automatically removed and must be cleaned up manually:

```bash
# Remove kernel sources (if previously extracted)
sudo rm -rf /usr/local/share/greenlit/linux
```

## Basic Usage

### Kernel Patch Checker (Pre-requisite)

Before running the test suite, verify that required upstream kernel patches are present in your kernel branch. Run this from within a Linux kernel git repository.

**After installation:**
```bash
cd /path/to/linux-source
check-kernel-patches -b <branch> /usr/share/greenlit/data/kernel-patches/<sha-list.txt>
```

**Without installation (from repo):**
```bash
cd /path/to/linux-source
python3 <greenlit-repo>/src/scripts/check_kernel_patches.py -b <branch> <sha-list.txt>
```

**Example:**
```bash
# After installation
check-kernel-patches -b my-kernel-branch /usr/share/greenlit/data/kernel-patches/grace.txt

# Without installation
python3 ~/greenlit/src/scripts/check_kernel_patches.py -b my-kernel-branch \
  ~/greenlit/data/kernel-patches/grace.txt ~/greenlit/data/kernel-patches/vera.txt

# Use grace.txt for Grace, grace.txt and vera.txt for Vera
```

**Options:**
- `-b, --branch`: Target branch (default: current)
- `-u, --upstream`: Upstream ref (default: upstream/master)

**Status meanings:**
- `EXISTS`: Commit found in branch with matching diff
- `REVIEW`: Commit found but diff differs - needs manual review
- `ABSENT`: Commit not found in branch (needs backporting)
- `SERIES`: Shown after EXISTS/REVIEW if patch series has missing members

**Notes in parentheses:**
| Note | Meaning |
|------|---------|
| `(sha)` | Cherry-pick marker references the input SHA |
| `(sha, no SHA marker)` | Found by title match, no cherry-pick marker |
| `(sha, different marker SHA)` | Marker references a different upstream SHA |
| `(sha, SHA not in upstream)` | Marker SHA not in configured upstream |

**Exit code:** Returns count of REVIEW + ABSENT commits (0 = all commits exist)

### Kernel Configuration Checker (Pre-requisite)

After running the patch checker, verify your kernel configuration meets NVIDIA's requirements.

**After installation:**
```bash
check-kernel-config -c /boot/config-$(uname -r) /usr/share/greenlit/data/kernel-config/grace.txt
```

**Without installation (from repo):**
```bash
python3 ~/greenlit/src/scripts/check_kernel_config.py -c /boot/config-$(uname -r) <reference.txt>
```

**Example:**
```bash
# After installation - Grace platform
check-kernel-config -c /boot/config-$(uname -r) /usr/share/greenlit/data/kernel-config/grace.txt

# After installation - Vera platform (extends Grace)
check-kernel-config -c /boot/config-$(uname -r) \
  /usr/share/greenlit/data/kernel-config/grace.txt \
  /usr/share/greenlit/data/kernel-config/vera.txt

# Without installation
python3 ~/greenlit/src/scripts/check_kernel_config.py -c .config \
  ~/greenlit/data/kernel-config/grace.txt

# Use grace.txt for Grace, grace.txt and vera.txt for Vera
```

**Options:**
- `-c, --config`: Path to kernel config file (e.g., `/boot/config-$(uname -r)` or `.config`)
- `-v, --verbose`: Show passed configs in addition to failures
- `<reference-files>`: One or more reference specification files (positional arguments)

**Status meanings:**
- `[PASS]`: Config meets requirements (shown with `-v`)
- `[FAIL]`: Config does not meet requirements

**Exit code:** Returns count of failed configs (0 = all configs pass)

### Running Tests

- Default sanity tests: `sudo glt-test-suite`
- Specific categories: `sudo glt-test-suite --tpm --arm64 --ipmi`
- Running external test scripts: `sudo glt-test-suite --ext_executable`
- Verbose output: `sudo glt-test-suite --verbose`
- List available tests: `sudo glt-test-suite --list`
- Display help information: `sudo glt-test-suite --help`

### Test Plans

- **Sanity** (default): `sudo glt-test-suite --sanity`
- **Standard**: `sudo glt-test-suite --standard`
- **Stress**: `sudo glt-test-suite --stress`

### Skipping Tests

GreenLiT offers several ways to skip specific tests:

#### Using Configuration Files

You can configure specific tests to be skipped by adding them to the configuration file at `/usr/share/greenlit/confs.d/`:

```yaml
# Example: /usr/share/greenlit/confs.d/custom_skips.yaml
tests:
  glt_test_tpm_access:  # Test name to configure
    conditional-skip: "true"  # Always skip this test
    skip-reason: "TPM hardware not available on this system"

  glt_test_pmu:  # Another test to configure
    conditional-skip: "uname -r | grep -q '5.15'"  # Skip only on kernel 5.15
    skip-reason: "Not compatible with kernel 5.15"
```

## Test Categories

- Memory Management: `--mm`
- I2C Bus: `--i2c`
- PCI Subsystem: `--pci`
- Hardware Tracing: `--hwtracing`
- Virtualization: `--virt`
- Real-time Clock: `--rtc`
- GPU Driver: `--gpu`
- ARM64 Architecture: `--arm64`
- Trusted Platform Module: `--tpm`
- Power Management: `--power`
- IPMI Interface: `--ipmi`
- IO Memory Management: `--iommu`
- General Purpose IO: `--gpio`
- Interrupt Requests: `--irq`

## Test Reports

After executing tests, GreenLiT generates a summary report showing test results and saves it to `/var/cache/greenlit/`. Here's how to interpret the output:

### Summary Section

The summary section provides an overview of test results:

```bash
Summary:
    PASSED            : Number of tests that passed
    FAILED            : Number of tests that failed
    Allowed FAILURE   : Tests that failed but are marked as "allowed to fail"
    SKIPPED           : Tests that were skipped due to configuration or prerequisites
```

### Details Section

The details section shows the status of each test:

```bash
Details:
    [N/A]     : Test was not applicable for the selected test plan
    [PASSED]  : Test completed successfully
    [FAILED]  : Test failed
    [SKIPPED] : Test was skipped, with reason provided below the test
    [ALLOWED] : Test failed but is marked as an allowed failure
```

### Test Sessions

Test sessions and logs are stored in the `/var/cache/greenlit/` directory with a timestamp:

- View previous test sessions: `ls /var/cache/greenlit/`
- Resume a specific test session: `sudo glt-test-suite --resume <session-number>`
  - Example: `sudo glt-test-suite --resume -1` (resume the most recent session)

## Recommended workflow

On a **Linux** target system, a typical end-to-end run follows this order. Use one or more `<reference-files>` per platform (Grace: `grace.txt`; Vera: `grace.txt` and `vera.txt`); see the kernel checker sections above.

1. **Check the host** — GreenLiT requires Linux (install, kernel checks, and tests are not supported on macOS).

   ```bash
   uname -s
   cat /etc/os-release
   uname -m
   uname -r
   ```

2. **Install** — See [Installation](#installation).

   ```bash
   sudo ./install.sh
   ```

3. **Verify kernel patches** (optional, when validating a kernel branch) — See [Kernel Patch Checker](#kernel-patch-checker-pre-requisite).

   ```bash
   cd /path/to/linux-source
   check-kernel-patches -b <branch> \
     /usr/share/greenlit/data/kernel-patches/<reference-files>
   ```

   `<reference-files>` are one or more patch list paths (for example `grace.txt`, or `grace.txt` and `vera.txt` for Vera).

4. **Verify kernel configuration** — See [Kernel Configuration Checker](#kernel-configuration-checker-pre-requisite).

   ```bash
   check-kernel-config -c /boot/config-$(uname -r) \
     /usr/share/greenlit/data/kernel-config/<reference-files>
   ```

   `<reference-files>` are one or more config reference paths (same Grace/Vera pattern as above).

5. **Run tests** — See [Running Tests](#running-tests) and [Test Plans](#test-plans).

   ```bash
   sudo glt-test-suite --sanity
   ```

6. **Review results** — See [Test Reports](#test-reports). Session logs are under `/var/cache/greenlit/`.

## Advanced Usage

### Adding External Tests to GreenLiT

Follow these steps to add your custom tests to GreenLiT:

- Step 1: Place your test executable

  - Copy your test executable to the directory:

   ```bash
   /usr/share/greenlit/executing_dir
   ```

  - Ensure the file has execute permissions:

   ```bash
   sudo chmod +x /usr/share/greenlit/executing_dir/your_test
   ```

- Step 2: Run the external tests

  Execute the following command to run all external tests:

  ```bash
  sudo glt-test-suite --ext_executable
  ...
  2025-09-26 14:49:36 - greenlit(main) - INFO: == Session /var/cache/greenlit/2025-09-26-14:49:36 started ==
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Running /usr/share/greenlit/executing_dir/a/failed.sh
  2025-09-26 14:49:36 - greenlit(perform_tests) - INFO: [FAILED] Finished /usr/share/greenlit/executing_dir/a/failed.sh. Exit code: 1. Test failed with unexpected exit code. Stdout: I wanna failed
  ,Stderr:
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Running /usr/share/greenlit/executing_dir/b/failed.sh
  2025-09-26 14:49:36 - greenlit(perform_tests) - INFO: [FAILED] Finished /usr/share/greenlit/executing_dir/b/failed.sh. Exit code: 1. Test failed with unexpected exit code. Stdout: I wanna failed
  ,Stderr:
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Running /usr/share/greenlit/executing_dir/zz/afailed.sh
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Finished /usr/share/greenlit/executing_dir/zz/afailed.sh: Allowed FAILURE: Finished /usr/share/greenlit/executing_dir/zz/afailed.sh. Exit code: 2. Test is expected to fail and did fail. Stdout: I wanna failed but allowed
  ,Stderr:
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Running /usr/share/greenlit/executing_dir/zz/skip.sh
  2025-09-26 14:49:36 - greenlit(perform_tests) - INFO: [SKIPPED] Finished /usr/share/greenlit/executing_dir/zz/skip.sh. Exit code: 4. Test is expected to skip. Stdout: I wanna be skipped.
  ,Stderr:
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Running /usr/share/greenlit/executing_dir/afailed.sh
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Finished /usr/share/greenlit/executing_dir/afailed.sh: Allowed FAILURE: Finished /usr/share/greenlit/executing_dir/afailed.sh. Exit code: 2. Test is expected to fail and did fail. Stdout: I wanna failed but allowed
  ,Stderr:
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Running /usr/share/greenlit/executing_dir/failed.sh
  2025-09-26 14:49:36 - greenlit(perform_tests) - INFO: [FAILED] Finished /usr/share/greenlit/executing_dir/failed.sh. Exit code: 1. Test failed with unexpected exit code. Stdout: I wanna failed
  ,Stderr:
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Running /usr/share/greenlit/executing_dir/pass.sh
  2025-09-26 14:49:36 - glt_ext_launcher(test_func) - INFO: Finished /usr/share/greenlit/executing_dir/pass.sh: PASSED: Finished /usr/share/greenlit/executing_dir/pass.sh. Exit code: 0. PASS. Stdout: I wanna pass
  ,Stderr:
  2025-09-26 14:49:47 - greenlit(show_report) - INFO:
  ===========================================================

  Summary:
    PASSED            : 3
    FAILED            : 3
    Allowed FAILURE   : 0
    SKIPPED           : 1

  Details:
    [N/A] mm:test_module_allocation
    [N/A] i2c:test_i2c_and_ipmi_device
    [N/A] i2c:test_ipmi_i2c_pec
    [N/A] tpm:test_nvidia_tpm
    [N/A] acpi:test_efi_runtime_services_mask
    [N/A] acpi:test_acpi_hmat_msg
    [N/A] pci:test_nvidia_pci_device_reset_method
    [N/A] gpio:test_tegra241_gpio_device
    [N/A] virt:test_virt_kselftest
    [N/A] irq:test_its_msi_irq
    [N/A] irq:test_gic_errata
    [N/A] arm64:test_all_pmu_events
    [N/A] arm64:test_kaslr_offset
    [N/A] ipmi:test_bmc_manufacturer_name
    [N/A] gpu:test_ast_gpu_driver
    [N/A] gpu:test_fb_efi_addresses
    [N/A] hwtracing:test_perf_metric_id
    [N/A] hwtracing:test_perf_coresight
    [N/A] iommu:test_iommu_id_mapping
    [N/A] rtc:test_rtc_alarm_property
    [N/A] power:test_reboot
    [FAILED] external:a_failed_sh
            - Finished /usr/share/greenlit/executing_dir/a/failed.sh. Exit code: 1. Test failed with unexpected exit code. Stdout: I wanna failed
    Stderr:
    [FAILED] external:b_failed_sh
            - Finished /usr/share/greenlit/executing_dir/b/failed.sh. Exit code: 1. Test failed with unexpected exit code. Stdout: I wanna failed
    Stderr:
    [PASSED] external:zz_afailed_sh
    [SKIPPED] external:zz_skip_sh
            - Finished /usr/share/greenlit/executing_dir/zz/skip.sh. Exit code: 4. Test is expected to skip. Stdout: I wanna be skipped.
    Stderr:
    [PASSED] external:afailed_sh
    [FAILED] external:failed_sh
            - Finished /usr/share/greenlit/executing_dir/failed.sh. Exit code: 1. Test failed with unexpected exit code. Stdout: I wanna failed
    Stderr:
    [PASSED] external:pass_sh
    [N/A] arm64:test_nvidia_perf_events

  ===========================================================
  2025-09-26 14:49:47 - greenlit(main) - INFO: == Session /var/cache/greenlit/2025-09-26-14:49:36 completed ==
    ```
  

- Additional Notes:
  - You can organize executables in subdirectories for better organization
  - GreenLiT automatically discovers and runs all executable files in this directory tree
  - Tests are executed in alphabetical order, with subdirectories processed first
  - No additional configuration is required - just add your executable and run!

### Enabling Kexec Tests (Optional)

#### Prerequisites

To enable kexec tests, you need:

1. **kexec-tools package installed**
2. **Kernel files present in /boot**
3. **Proper system permissions**

#### Installing kexec-tools

**Debian/Ubuntu**:
```bash
sudo apt update
sudo apt install kexec-tools
```

**RHEL/CentOS/Fedora**:
```bash
sudo dnf install kexec-tools
```

**SUSE**:
```bash
sudo zypper install kexec-tools
```

#### Verifying Kexec

To verify kexec is working before enabling the test:

```bash
# Check if kexec is installed
which kexec

# Check if kernel files exist
ls -la /boot/vmlinuz-$(uname -r)
ls -la /boot/initrd.img-$(uname -r)  # or initramfs-$(uname -r).img

# Test kexec load (doesn't actually reboot)
sudo kexec -l /boot/vmlinuz-$(uname -r) --initrd=/boot/initrd.img-$(uname -r) --reuse-cmdline
```

**Note**: If kexec is not available or working in your environment, the test will be automatically skipped with a helpful error message.

### Building lmbench (Optional)

Some GreenLiT tests (such as the CPU clock rate verification test) require the `mhz` utility from the lmbench benchmark suite. While many distributions provide an lmbench package, you can also build it from source if needed.

#### Installing from Package Manager

**Debian/Ubuntu**:
```bash
sudo apt update
sudo apt install lmbench
```

#### Building lmbench from Source

**Install Build Dependencies**

Make sure all necessary build tools are installed before proceeding.

**Debian/Ubuntu**:
```bash
sudo apt update
sudo apt install build-essential libtirpc-dev
```

**RHEL/CentOS/Fedora**:
```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install libtirpc-devel
# Or for older versions:
# sudo yum groupinstall "Development Tools"
# sudo yum install libtirpc-devel
```

**Note**: On RHEL systems, you may need to enable the CodeReady Builder (CRB) repository to install development packages:
```bash
# RHEL 9+
sudo dnf config-manager --set-enabled crb

# RHEL 8
sudo dnf config-manager --set-enabled powertools
```

**SUSE**:
```bash
sudo zypper install -t pattern devel_basis
sudo zypper install libtirpc-devel
```


1. Download and extract lmbench from GitHub:
   ```bash
   git clone https://github.com/intel/lmbench.git
   cd lmbench
   ```

2. Build lmbench:
   ```bash
   cd src
   make
   ```

3. Install the binaries (optional):
   ```bash
   sudo mkdir -p /usr/local/lib/lmbench/bin
   sudo cp bin/*/mhz /usr/local/bin/mhz
   ```

4. Verify installation:
   ```bash
   which mhz
   mhz
   ```

**Note**: Some systems may require additional build flags for libtirpc:
```bash
make CPPFLAGS="-I /usr/include/tirpc" LDFLAGS="-l tirpc"
```

### Building stress-ng (Optional)

Some GreenLiT tests (such as the TPM QSPI interrupt timeout test) require the `stress-ng` utility to generate CPU and timer stress. While many distributions provide a stress-ng package, you can also build it from source if needed.

#### Installing from Package Manager

**Debian/Ubuntu**:
```bash
sudo apt update
sudo apt install stress-ng
```

**RHEL/CentOS/Fedora**:
```bash
sudo dnf install stress-ng
```
**Note**: If an error such as "Error: Failed to download metadata for repo 'fluent-bit': Cannot download repomd.xml: Cannot download repodata/repomd.xml: All mirrors were tried" occurred, try adding `--disablerepo=fluent-bit` in the command.

**SUSE**:
```bash
sudo zypper install stress-ng
```

#### Building stress-ng from Source

**Install Build Dependencies**

Make sure all necessary build tools are installed before proceeding.

**Debian/Ubuntu**:
```bash
sudo apt update
sudo apt install build-essential libbsd-dev libkeyutils-dev libapparmor-dev libattr1-dev libcap-dev zlib1g-dev
```

**RHEL/CentOS/Fedora**:
```bash
sudo dnf makecache -y
sudo dnf groupinstall "Development Tools" -y
sudo dnf install libbsd-devel keyutils-libs-devel libattr-devel libcap-devel zlib-devel -y
```

**Note**: On RHEL systems, you may need to enable the CodeReady Builder (CRB) repository to install development packages:
```bash
# RHEL 9+
sudo subscription-manager repos --enable codeready-builder-for-rhel-9-$(arch)-rpms
```

**SUSE**:
```bash
sudo zypper refresh
sudo zypper update
sudo zypper install -t pattern devel_basis
sudo zypper install libbsd-devel libapparmor-devel libattr-devel libcap-devel zlib-devel
```

For more questions, please refer to [Building stress-ng](https://github.com/ColinIanKing/stress-ng?tab=readme-ov-file#building-stress-ng)
