# GreenLiT Configuration Guide

This guide explains how to configure GreenLiT for your specific environment and testing needs.

## Index

- [Customizing Test Scope](#customizing-test-scope)
  - [Configuration Files Location, Format and Ordering](#configuration-files-location-format-and-ordering)
  - [YAML Configuration Structure](#yaml-configuration-structure)
  - [Field Definitions](#field-definitions)
    - [Category](#category)
    - [Test Case Name](#test-case-name)
## Customizing Test Scope

### Configuration Files Location, Format and Ordering

GreenLiT reads YAML files from the location `/usr/share/greenlit/confs.d/` to determine what test scope-related customizations need to be done for test cycles.
A default set of customizations `00_default.yaml` will exist inside this location once you've installed GreenLiT.  You can add or modify configurations based on actual needs.

Files are read in ascending alphabetical order by filename, for example:
- `00-base.yaml` would be read before 
- `01-custom.yaml`

If a test case is defined in multiple YAML files, the later ones will **overwrite** the previous definitions.

### YAML Configuration Structure

A GreenLiT configuration file follows this structure:

```yaml
[category]:[test_case_name]:  
  [operation]:  
    - condition:  
        name: [condition name as a string]
        [condition_type]:
        {extra_conditions:}  
```

#### Field Definitions

##### Category
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

#### Test Case Name
The name of the specific test case within a category.

You can retrieve the list of test cases with:
```bash
sudo glt-test-suite --list
```

##### Operation
Specifies how the test case should be handled.

Supported operations:
- `allow-failure`: Executes the test case but marks failures as "allowed-failure" instead of "failure".
- `conditional-skip`: Skips the test case if certain conditions are met.

##### Condition Types
Defines the type of condition used to modify test behavior.

Supported types:
- `distro`: The `NAME` field in `platform.freedesktop_os_release()` or `/etc/os-release`.
- `script`: Executes a shell script and checks its exit code.

##### Extra Conditions
Optional additional conditions based on the condition type.

If condition type is `distro`:
- `release`: Compares the `VERSION_ID` field from `platform.freedesktop_os_release()` or `/etc/os-release`.
- `kernel-uname`: Compares the first two values of the `uname` output.
  - Supported formats:
    - `== 6.12` (exact match)
    - `> 6.12` (version higher than 6.12)
    - `>= 6.12` (version 6.12 or higher)
    - `< 6.12` (version lower than 6.12)
    - `<= 6.12` (version 6.12 or lower)

### Configuration Examples

#### Example 1: Allow Failure for a Test on Specific Distributions

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

#### Example 2: Skip a Test Based on a Script Condition

```yaml
arm64:test_all_pmu_events:
  conditional-skip:
    - condition:
        name: "Assume if NFS presenting in kernel command line"
        script: |
          #!/usr/bin/python3
          return 'nfs' in open('/proc/cmdline').read()
```

#### Example 3: Skip a Test Based on Distribution and Kernel Version

```yaml
arm64:test_all_pmu_events:
  conditional-skip:
    - condition:
        name: "Skip because is not ready on Ubuntu 20.04 or kernel version is lower than 5.15"
        distro: "Ubuntu"
        release: "20.04"
        kernel-uname: "<= 5.15"
```

### Creating Custom Configurations

To create a custom configuration:

1. Create a new YAML file in the `/usr/share/greenlit/confs.d/` directory with a name that ensures it's loaded after the default configurations (e.g., `99-custom.yaml`).

2. Add your custom configuration using the format described above.

3. Save the file and run GreenLiT to apply your custom configuration.
