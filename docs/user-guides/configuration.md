# GreenLiT Configuration Guide

This guide explains how to configure GreenLiT for your specific environment and testing needs.

## Index

- [Configuration Files: Location, Format, and Ordering](#configuration-files-location-format-and-ordering)
- [YAML Configuration Structure](#yaml-configuration-structure)
- [Field Definitions](#field-definitions)
  - [Category](#category)
  - [Test Case Name](#test-case-name)
  - [Operation](#operation)
  - [Condition Types](#condition-types)
  - [Extra Conditions](#extra-conditions)
- [Configuration Examples](#configuration-examples)
  - [Example 1: Allow Failure for a Test on Specific Distributions](#example-1-allow-failure-for-a-test-on-specific-distributions)
  - [Example 2: Skip a Test Based on a Script Condition](#example-2-skip-a-test-based-on-a-script-condition)
  - [Example 3: Skip a Test Based on Distribution and Kernel Version](#example-3-skip-a-test-based-on-distribution-and-kernel-version)
- [Creating Custom Configurations](#creating-custom-configurations)

## Configuration Files: Location, Format, and Ordering

GreenLiT reads YAML files from `/usr/share/greenlit/confs.d/` to determine which test-scope customizations apply to a test cycle. A default configuration, `00_default.yaml`, is installed at this location once GreenLiT is installed; you can add or modify files there as needed.

Files are read in lexicographic filename order. Use zero-padded numeric prefixes to control precedence; for example, `00-base.yaml` is read before `01-custom.yaml`.

If the same test case is defined in multiple YAML files, the later file's definition **overwrites** the earlier one.

## YAML Configuration Structure

A GreenLiT configuration file follows this structure:

```yaml
[category]:[test_case_name]:  
  [operation]:  
    - condition:  
        name: [condition name as a string]
        [condition_type]:
        {extra_conditions:}  
```

## Field Definitions

### Category

The test suite category in GreenLiT.

Available categories can be found using:
```bash
sudo glt-test-suite --help
```

Examples include:
- `mm` - Memory management tests
- `i2c` - I2C bus tests
- `pci` - PCI subsystem tests
- `arm64` - ARM64 architecture tests
- `tpm` - TPM module tests
- And others as listed in the help output

### Test Case Name

The name of the specific test case within a category.

You can retrieve the list of test cases with:
```bash
sudo glt-test-suite --list
```

Test case identifiers use the `category:test_case_name` format described in the
user manual.

### Operation

Specifies how the test case should be handled.

Supported operations:
- `allow-failure`: Executes the test case but marks failures as "allowed-failure" instead of "failure".
- `conditional-skip`: Skips the test case if certain conditions are met.

### Condition Types

Defines the type of condition used to modify test behavior.

Supported types:
- `distro`: The `NAME` field in `platform.freedesktop_os_release()` or `/etc/os-release`.
- `script`: Runs the given string as a **bash** script — regardless of any shebang line it contains — and checks its exit code.

### Extra Conditions

Optional additional conditions based on the condition type.

If condition type is `distro`:
- `release`: Compares the `VERSION_ID` field from `platform.freedesktop_os_release()` or `/etc/os-release`.
- `kernel-uname`: Compares against the kernel release (`uname -r`, i.e. `platform.release()`), using distro version comparison rules.
  - Supported formats:
    - `== 6.12` (exact match)
    - `> 6.12` (version higher than 6.12)
    - `>= 6.12` (version 6.12 or higher)
    - `< 6.12` (version lower than 6.12)
    - `<= 6.12` (version 6.12 or lower)
  - A real `uname -r` string usually has more components than a bare release
    number (e.g. `6.8.0-1024.27-generic`), and version comparison treats a
    longer string as greater than a shorter prefix of itself. Compare against
    a fully-qualified release string for a reliable match — see Example 3.

## Configuration Examples

### Example 1: Allow Failure for a Test on Specific Distributions

```yaml
arm64:test_all_pmu_events:
  allow-failure:
    - condition:
        name: "Perf is not ready on Ubuntu 20.04"
        distro: "Ubuntu"
        release: "20.04"
    - condition:
        name: "Coresight is not ready on Redhat 9.5"
        distro: "Red Hat"
        release: "9.5"
```

### Example 2: Skip a Test Based on a Script Condition

```yaml
arm64:test_all_pmu_events:
  conditional-skip:
    - condition:
        name: "Skip if NFS is present in the kernel command line"
        script: |
          #!/bin/bash
          grep -q 'nfs' /proc/cmdline
```

### Example 3: Skip a Test Based on Distribution and Kernel Version

```yaml
arm64:test_all_pmu_events:
  conditional-skip:
    - condition:
        name: "Skip on Ubuntu 22.04 with kernels older than 6.8.0-1024.27"
        distro: "Ubuntu"
        release: "22.04"
        kernel-uname: "< 6.8.0-1024.27"
```

## Creating Custom Configurations

To create a custom configuration:

1. Create a new YAML file in the `/usr/share/greenlit/confs.d/` directory with a name that ensures it's loaded after the default configurations (e.g., `99-custom.yaml`).

2. Add your custom configuration using the format described above.

3. Save the file and run GreenLiT to apply your custom configuration.
