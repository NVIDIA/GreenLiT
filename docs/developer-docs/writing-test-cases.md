# Writing Test Cases for GreenLiT

This guide explains how to write effective test cases for the GreenLiT test framework.

## Index

- [Introduction & Objectives](#introduction--objectives)
- [Prerequisites](#prerequisites)
- [Do We Need a New Test Case?](#do-we-need-a-new-test-case)
- [Create a Test Case that Does Not Require Reboots](#create-a-test-case-that-does-not-require-reboots)
  - [Example Test Case Structure](#example-test-case-structure)
  - [Categorizing Your Test Case](#categorizing-your-test-case)
  - [Choosing Test Plans](#choosing-test-plans)
  - [Naming and Formatting Rules](#naming-and-formatting-rules)
  - [Using GLTTestAttr](#using-glttestattr)
  - [Leveraging GreenLiT Utilities](#leveraging-greenlit-utilities)
  - [Using GLTLibvirt for Virtualization Tests](#using-gltlibvirt-for-virtualization-tests)
  - [Test Results: FAILED, SKIPPED, or WARNING](#test-results-failed-skipped-or-warning)
  - [Managing Test Case Dependencies](#managing-test-case-dependencies)
  - [Post-Test Actions with Post Hooks](#post-test-actions-with-post-hooks)
- [Create a Test Case that Requires Reboots](#create-a-test-case-that-requires-reboots)
  - [Create a Test Case that Requires kexec Reboots](#create-a-test-case-that-requires-kexec-reboots)
  - [Conditional reboot after kexec](#conditional-reboot-after-kexec)
- [Create an External Test Case](#create-an-external-test-case)
  - [When to Use External Tests](#when-to-use-external-tests)
  - [Directory Structure](#directory-structure)
  - [Exit Codes](#exit-codes)
  - [Creating an External Test](#creating-an-external-test)
  - [Example External Test Scripts](#example-external-test-scripts)
  - [Test Discovery](#test-discovery)
  - [Environment Variables](#environment-variables)
  - [Best Practices](#best-practices)
  - [Organizing External Tests](#organizing-external-tests)
  - [Limitations](#limitations)
- [Inclusion of Additional Packages](#inclusion-of-additional-packages)
- [Licensing Considerations](#licensing-considerations)
- [Follow Ups](#follow-ups)
- [Appendix: Allowing Test Result Exceptions](#appendix-allowing-test-result-exceptions)

## Introduction & Objectives

A quick step-by-step guide for those who need to create GreenLiT test cases and get them running in test cycles, without deep-diving into the logic behind how GreenLiT manages the test case execution.

**Note**: This doc will not focus on any development process related topics. For those please refer to the prerequisites section.

## Prerequisites

Before creating test cases, you should have:

- An understanding of **GreenLiT Development Process** for how to:
  - Create working branches
  - Create an issue/ticket ID to ensure the change is used to address that issue.
  - Go through merge request process
- Background knowledge on **Python**

### Important: Test Scenarios

Keep in mind that your test case needs to work in **all** of the following test scenarios:

## Do We Need a New Test Case?

Before creating a new test case, check if existing test cases cover what you'd like to test.

"Existing" meaning:

- The GreenLiT specific tests located in GreenLiT repository, **or**
- Upstream kselftests that are included in test cycles

There will be no need to create a duplicated test case if either of the above two already covers what you'd like to check.

## Create a Test Case that Does Not Require Reboots

Please refer to any test case (i.e. a function that starts with `test_` under `<category>/glt_test_<feature>.py`) for more examples.

### Example Test Case Structure

The basic contents of a test case would look like the following code block. While you could simply make copies of existing tests and start your test case creation journey from there, scroll down further and refer to the breakdown of this piece of code to understand what each section means, and how you can use them to design your test case:

```python
import subprocess
from common import glt_core as GLTCore
from common import glt_utils
from common.nv_device import NvDev

# ...(other test cases omitted for brevity)

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_pci_device_reset_method(sd, logger):
    """
    Test primary reset method of Hopper/NVIDIA GFX device.

    PATCH: https://git.kernel.org/torvalds/c/6937b7dd4349 v5.15+
    """

    for tests in sd.progress:
        if test_nvidia_pci_device_reset_method.__name__ not in tests:
            continue
        if "MightReset" in tests[test_nvidia_pci_device_reset_method.__name__]:

            # ...(section omitted for brevity)

            if p.returncode == 0 and p.stdout.strip() != '0':
                logger.info("System reset didn't happen")
                glt_utils.systemd_enable(enable=False)
                glt_utils.start_systemd_timer(start=False)
                return

            GLTCore.GLTTestResult.fail("System was reset")

    lspci_output = NvDev.get_nvidia_3d_bdf()
    pci_dev_addrs = list(lspci_output.splitlines())

    if not pci_dev_addrs:
        GLTCore.GLTTestResult.skip("SKIP: NVIDIA hopper pcie device ID not found.")

    # ...(section omitted for brevity)

    glt_utils.arm_for_reset(test_nvidia_pci_device_reset_method.__name__, sd)
```

### Categorizing Your Test Case

#### Step 1: Decide which category your test case should fit in

- Existing categories can be found [here](../../src/modules/)
- Under your preferred category, locate which `glt_test_<feature>.py` you'd like to host your test case in and add your new test case there
- Create a new `glt_test_<feature>.py` if needed

#### Step 2: Creating a new category (if needed)

In rare scenarios where none of the existing categories fits your needs, create new ones by duplicating and modifying existing files and directories.

When any new "category" (i.e. new directory under `/src/modules`) is introduced, **Update GreenLiT architecture.md** to contain information about your new category is required.

### Choosing Test Plans

Decide which test plan(s) your test case should be included into.

Choose from the 3 valid test plans:

- `"sanity"` - Quick smoke tests for basic functionality
- `"standard"` - Comprehensive testing for standard validation
- `"stress"` - Extended stress and stability testing

Use the decorator `@GLTCore.GLTPlan("<test plan name>")` with your test case to include the test into a certain plan. Use multiple decorators if your test should be in multiple test plans.

**Example:**

```python
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
def test_my_feature(_, logger):
    """Test description"""
    # test implementation
```

### Naming and Formatting Rules

- **One test case = one function**
  - Refrain from testing multiple features within one function/test case unless there's a reason to do so

- **Test case names must start with `test_`** and follow the pattern:
  ```python
  test_<feature>(sd, logger):
  ```
  
  Where:
  - `sd` - "scheduler" (likely needed if your test steps contain a reboot)
  - `logger` - for when you need to record additional logs throughout the test

- **If either one (or none) is not needed for your test**, replace arguments with placeholders for consistency in test case function signatures:
  ```python
  test_<feature>(sd, _)    # Only need scheduler
  test_<feature>(_, logger) # Only need logger
  test_<feature>(_, __)     # Need neither
  ```

#### Docstring Requirements

**Required elements:**

- Short description of what is tested
- Include link to related backport patch(es) if available

**Example:**

```python
def test_nvidia_pci_device_reset_method(sd, logger):
    """
    Test primary reset method of Hopper/NVIDIA GFX device.

    PATCH: https://git.kernel.org/torvalds/c/6937b7dd4349 v5.15+
    """
```

#### Handling Known Bugs and TODOs

Use a comment above the line that requires modifying:

```python
# FIXME DGX-5678 <issue description>
```

### Using GLTTestAttr

`@GLTCore.GLTTestAttr()` collects and stores test case attributes. **Please wrap your test case function with it.**

```python
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
def test_my_feature(_, logger):
    """Test description"""
    pass
```

### Leveraging GreenLiT Utilities

Various functions and methods are available under:

- **`glt_utils`**: General Linux related utilities
  - Example: Check if command/tool is available in the DUT
- **`nv_device`**: NVIDIA specific utilities
  - Example: Check if CPU is Grace

**Important:** Chances are a command you'd like to use in your test case has a ready-to-be-called function available in the above modules. Use those instead of creating your own to prevent reinventing the wheel.

If existing functions under these modules require an update to fit your needs, or if you feel that the commands you are about to use would likely be reused in other test cases in the future, instead of creating the steps inside your new test case, please update these modules and have your test case call them.

### Using GLTLibvirt for Virtualization Tests

Use `common.virt.glt_libvirt.GLTLibvirt` when a test case needs to create or
manage local libvirt/QEMU domains. The helper centralizes `virsh` invocation,
domain XML generation, debug logging, and common cleanup behavior.

Before using it, fail or skip early if virtualization tools are unavailable:

```python
if not glt_utils.is_cli_available("virsh"):
    GLTCore.GLTTestResult.fail("Please install 'virsh' to perform the test.")
```

Create the helper with the test logger. The constructor validates that `virsh`
can connect to `qemu:///system`, so catch `ConnectionError` and convert it to a
GreenLiT test result:

```python
try:
    libvirt_conn = GLTLibvirt(logger=logger)
except ConnectionError as e:
    GLTCore.GLTTestResult.fail(str(e))
```

For direct-kernel boot domains, build XML through `generate_domain_xml()` and
use `define_and_start()` for the common define/start flow:

```python
from common.virt.glt_libvirt import GLTLibvirt


@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("stress")
def test_my_virtualization_case(_, logger):
    domain_name = "glt-my-test-vm"
    libvirt_conn = GLTLibvirt(logger=logger)
    xml_config = libvirt_conn.generate_domain_xml({
        "domain_name": domain_name,
        "memory_mb": 512,
        "vcpus": 4,
        "arch": "aarch64",
        "kernel_path": "/var/lib/libvirt/boot/Image",
        "initrd_path": "",
        "kernel_args": "console=ttyAMA0 rdinit=/bin/sh panic=0",
    })

    try:
        if not libvirt_conn.define_and_start(xml_config):
            GLTCore.GLTTestResult.fail(f"Failed to start {domain_name}")

        success, output, _ = libvirt_conn.run_virsh_command(["domstate", domain_name])
        if not success or "running" not in output.lower():
            GLTCore.GLTTestResult.fail(f"{domain_name} is not running: {output}")
    finally:
        libvirt_conn.cleanup(domain_name)
```

Guidelines:

- Always clean up domains in a `finally` block. A failed test should not leave
  `glt-*` domains behind for later test runs.
- Use `define_and_start()` for the normal single-domain flow. Use `define()` and
  `start()` separately only when a test needs to inspect or modify the domain
  between those steps.
- If a test creates many domains, destroy all domains first and then undefine
  them. This mirrors stress scenarios where teardown fanout is part of the
  behavior under test.
- Keep generated kernel/initrd artifacts in a path libvirt can read, such as
  `/var/lib/libvirt/boot`, and remove staged artifacts after the test.
- The helper manages host-side libvirt only. Do not add guest command execution
  APIs until a real test case needs them and the guest access method is clear.

### Test Results: FAILED, SKIPPED, or WARNING

Use `glt_core`'s test result methods to tell GreenLiT to fail, skip, or warn from a test:

```python
GLTCore.GLTTestResult.skip("<skip reason>")
GLTCore.GLTTestResult.fail("<error message>")
GLTCore.GLTTestResult.warn("<advisory message>")
```

Use `warn()` when the test detects an advisory condition — something worth surfacing in the report but not a hard failure (e.g., a deprecated configuration, a suboptimal-but-functional hardware state). Like `skip()` and `fail()`, `warn()` stops the test immediately and must only be called from a function decorated with `@GLTCore.GLTTestAttr()`.

**By default**, if none of the conditions required to skip, fail, or warn are met, the test case **passes**.

**See also**: [Appendix: Allowing test result exceptions](#appendix-allowing-test-result-exceptions) for how to dynamically allow certain cases to be skipped and failed cases to not count towards fails. Note that these circumstances are usually special-case scenarios where the fail/skip criteria is not a long term logic that should go into the main test case function (e.g., ignore failure during bug fix period).

### Managing Test Case Dependencies

#### Must Pass Dependency

`@GLTCore.GLTRequires.must_pass("<category>", "<test case>")`

If your test case should only be run when other test case(s) have passed, wrap your test case with this decorator, where:
- `<category>` is the category of the case that must pass first
- `<test case>` is the name of the test case that must pass first

#### Must Fail Dependency

`@GLTCore.GLTRequires.must_fail("<category>", "<test case>")`

If your test case should only be run when other test case(s) have failed, use this decorator instead.

#### Example

```python
@GLTCore.GLTRequires.must_pass("hwtracing", "test_perf_coresight")
@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("sanity")
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_nvidia_perf_events(_, logger):
    """Test NVIDIA perf events"""
    # ...(section omitted for brevity)
    
    # FIXME: perf seems not enable coresight on ubuntu.
    if NvDev.get_release_info().get("NAME") == "Ubuntu":
        GLTCore.GLTTestResult.skip("Perf doesn't enable coresight on Ubuntu.")
    
    # ...(section omitted for brevity)
```

### Post-Test Actions with Post Hooks

When a test needs to run something after it completes (e.g. cleanup, teardown, or restore), use **post hooks**. The framework supports four hooks currently, one per outcome: `postsuccess` (test passed), `postskip` (test skipped), `postfail` (test failed), `postafail` (allowed failure). After the test, the runner invokes the matching hook.

**How to use:** Create a `PostHooks` instance with `GLTCore.PostHooks.create(postsuccess=..., postskip=..., postfail=..., postafail=...)` and assign it to your test's **attr**: `test_something.attr.post_hooks = ...`. Pass callables with signature `(scheduler, attr, logger)` for the outcomes you need. Any slot you omit is a no-op. Each test has its own post hooks on its own attr; only tests that set `attr.post_hooks` get post hooks run. Example:

```python
def test_something(sd, logger):
    ...

test_something.attr.post_hooks = GLTCore.PostHooks.create(postsuccess=my_cleanup_fn)
```

You can pass the same or different callables for each outcome. For a full example that uses hooks to restore after a reboot, see [Create a Test Case that Requires Reboots](#create-a-test-case-that-requires-reboots).

## Create a Test Case that Requires Reboots

**TBD** - To be documented. Introduce the usage of general reboot case

### Create a Test Case that Requires kexec Reboots

Some tests need to kexec into a modified kernel command line, run the
actual test logic after kexec, and then reboot back to a clean state so
remaining tests can proceed. Use `glt_utils.handle_kexec_flow()` to
manage the three-phase kexec lifecycle:

1. **Verify prior attempt** — If a previous kexec was triggered but the
   system did not kexec-boot, the function returns `(False, reason)`.
2. **Initiate kexec** — If the system has not kexec-booted yet, the
   function calls `reboot_with_kexec()`. The process exits and the
   framework does not run more tests in that session.
3. **Resume after kexec** — On the next run, the framework resumes and
   `handle_kexec_flow()` detects the kexec boot. It disables the systemd
   service and returns `(True, None)`, signalling the test to proceed
   with its actual logic.

When `handle_kexec_flow()` returns `(False, reason)`, pass `reason` to
`GLTTestResult.fail()`. When it returns `(True, None)`, the kexec boot
was successful and the test should proceed with its actual logic.

#### Conditional reboot after kexec

After the test completes (pass, fail, or allowed-failure), the system
should reboot back to normal boot. Use
`PostHooks.for_conditional_reboot("TriggerKexec", glt_utils.is_kexec_boot)`
to register a post-hook that triggers a reboot only when the system is
in a kexec-booted state. Register it for `postsuccess`, `postfail`, and
`postafail`. Omit `postskip` (default no-op) because a skip means kexec
was never triggered.

#### Example

See `test_efi_runtime_services_mask` in `src/modules/acpi/glt_test_efi.py`:

```python
_kexec_conditional_reboot_hook = GLTCore.PostHooks.for_conditional_reboot(
    "TriggerKexec",
    glt_utils.is_kexec_boot,
)

@GLTCore.GLTTestAttr()
@GLTCore.GLTPlan("standard")
@GLTCore.GLTPlan("stress")
def test_efi_runtime_services_mask(sd, logger):
    if not glt_utils.is_efi_system():
        GLTCore.GLTTestResult.skip("Skipping test: System is not using EFI")

    ok, reason = glt_utils.handle_kexec_flow(
        test_efi_runtime_services_mask.__name__, sd, logger,
    )
    if not ok:
        GLTCore.GLTTestResult.fail(reason)

    # Actual test logic runs here (only reached after kexec boot)
    ...

test_efi_runtime_services_mask.attr.post_hooks = GLTCore.PostHooks.create(
    postsuccess=_kexec_conditional_reboot_hook,
    postfail=_kexec_conditional_reboot_hook,
    postafail=_kexec_conditional_reboot_hook,
)
```

Pass `custom_cmdline` for tests that need extra kernel command line
arguments (e.g. `custom_cmdline="pci=config_acs=101xxxx@0000:00:01.0"`).

## Create an External Test Case

External tests are standalone executable scripts (shell, Python, or any executable) that run outside the standard GreenLiT test framework but are automatically discovered and executed by GreenLiT.

### When to Use External Tests

Use external tests when:
- You have existing test scripts that don't fit into the Python test framework
- You need to write tests in bash or other scripting languages
- You want to integrate third-party test tools
- Your test is a one-off or experimental test
- You prefer a simpler approach without Python decorators and GreenLiT utilities

### Directory Structure

External tests are placed in the `external_tests/` directory:

```
greenlit/
└── external_tests/
    ├── tpm/
    │   └── test_config_tcg_tis_spi.sh
    └── your_category/
        └── your_test_script.sh
```

### Exit Codes

External tests communicate their results through exit codes:

| Exit Code | Status | Description |
|-----------|--------|-------------|
| `0` | PASS | Test passed successfully |
| `1` | FAIL | Test failed |
| `2` | XFAIL | Expected failure (test is expected to fail and did fail) |
| `3` | XPASS | Unexpected pass (test is expected to fail but passed - counts as FAIL) |
| `4` | SKIP | Test should be skipped |

### Creating an External Test

#### Step 1: Create Your Test Script

Create an executable script in `external_tests/`:

```bash
#!/bin/bash
# external_tests/my_category/test_my_feature.sh

# Perform your test
if some_command_that_tests_your_feature; then
    echo "Test passed: Feature is working correctly"
    exit 0  # PASS
else
    echo "Test failed: Feature is not working"
    exit 1  # FAIL
fi
```

#### Step 2: Make It Executable

```bash
chmod +x external_tests/my_category/test_my_feature.sh
```

#### Step 3: Test Your Script

Run it manually to verify:

```bash
./external_tests/my_category/test_my_feature.sh
echo $?  # Check exit code
```

### Example External Test Scripts

#### Example 1: Simple Bash Test

```bash
#!/bin/bash
# external_tests/storage/test_disk_space.sh

# Check if we have at least 1GB free space
free_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')

if [ "$free_space" -ge 1 ]; then
    echo "PASS: Free space is ${free_space}GB"
    exit 0
else
    echo "FAIL: Insufficient free space: ${free_space}GB"
    exit 1
fi
```

#### Example 2: Python External Test

```python
#!/usr/bin/env python3
# external_tests/network/test_dns_resolution.sh

import socket
import sys

def test_dns():
    try:
        socket.gethostbyname('www.google.com')
        print("PASS: DNS resolution working")
        return 0
    except socket.gaierror:
        print("FAIL: DNS resolution failed")
        return 1

if __name__ == "__main__":
    sys.exit(test_dns())
```

#### Example 3: Test with Expected Failure

```bash
#!/bin/bash
# external_tests/experimental/test_new_feature.sh

# This feature is known to be broken on certain kernels
if test_new_feature; then
    echo "Feature is working"
    exit 0  # PASS
else
    # Check if this is a known issue
    if uname -r | grep -q "5.15"; then
        echo "Expected failure on kernel 5.15"
        exit 2  # XFAIL (expected failure)
    else
        echo "Unexpected failure"
        exit 1  # FAIL
    fi
fi
```

#### Example 4: Test with Skip Logic

```bash
#!/bin/bash
# external_tests/hardware/test_tpm_device.sh

# Check if TPM hardware is available
if [ ! -e /dev/tpm0 ]; then
    echo "SKIP: TPM device not found"
    exit 4  # SKIP
fi

# Run the actual test
if tpm_test_command; then
    echo "PASS: TPM test successful"
    exit 0
else
    echo "FAIL: TPM test failed"
    exit 1
fi
```

### Test Discovery

GreenLiT automatically discovers external tests by:

1. Recursively scanning the `external_tests/` directory
2. Finding all executable files
3. Processing subdirectories first (depth-first search)
4. Executing tests in alphabetical order

**Test naming convention:**
- The test name is derived from the relative path
- Example: `external_tests/tpm/test_config.sh` becomes `external:tpm_test_config_sh`

### Environment Variables

External tests can access the following environment variable:

- **`GTL_WORKDIR`**: Working directory for the test to store temporary files or logs

**Example usage:**

```bash
#!/bin/bash
# Use GTL_WORKDIR for temporary files
temp_file="${GTL_WORKDIR}/temp_data.txt"
echo "Test data" > "$temp_file"
# Perform test using temp_file
```

### Best Practices

1. **Use descriptive names**: Name your test scripts clearly (e.g., `test_feature_name.sh`)

2. **Provide clear output**: Always echo what the test is doing and why it passed/failed

3. **Check prerequisites**: Use exit code 4 (SKIP) if required hardware/software is not available

4. **Handle errors gracefully**: Use proper error checking and return appropriate exit codes

5. **Keep tests focused**: One test per script, testing one specific feature

6. **Document your test**: Add comments explaining what the test does and any prerequisites

7. **Test locally first**: Always run your external test manually before committing

### Organizing External Tests

You can organize external tests in subdirectories by category:

```
external_tests/
├── hardware/
│   ├── test_cpu.sh
│   └── test_memory.sh
├── network/
│   ├── test_connectivity.sh
│   └── test_dns.sh
└── storage/
    └── test_disk_io.sh
```

### Limitations

External tests have some limitations compared to Python test cases:

- Cannot use GreenLiT utilities (`glt_utils`, `nv_device`, etc.)
- Cannot use test dependencies (`@GLTRequires`)
- Limited to single test plan (`ext_executable`)
- No direct access to the scheduler for reboot testing
- Must manage their own error handling and logging

If your test needs these features, consider creating a [standard Python test case](#create-a-test-case-that-does-not-require-reboots) instead.

## Inclusion of Additional Packages

If your test case requires additional packages that aren't available by default in the device-under-test:

### Case A: Packages Installable via Package Manager

- Install packages via the test device's package manager directly(e.g., `apt`, `dnf`, `zypper`).

### Case B: Executables Not Available via Package Manager

Executables that cannot be installed through a distro's software management system.

**Required steps:**

1. Put all files under the same parent as your `glt_test_<feature>.py` file, i.e., the root directory of your selected category (e.g., `/src/modules/i2c`).

2. Update GreenLiT `pyproject.toml` to include your package(s) and files if they are not Python-based.

## Licensing Considerations

**TBD** - Beware of potential licensing issues when including third-party packages or code.

## Follow Ups

Your test case should make it into the next GreenLiT release, which roughly follows a **1-month cadence**, after an MR is merged.

## Miscellaneous Q&A, Issues and Risks

N/A

---

## Appendix: Allowing Test Result Exceptions

While each test case contains code that controls when to skip or fail a test in "normal" situations, there are times where one needs to temporarily allow a case to either be skipped or the failure masked.

### Use Case Example

A bug that causes test case `test_all_pmu_events` to fail was introduced in the latest release of Ubuntu, but won't be fixed until the next release.

We might not want that case to keep showing up as failing during this bug fix period as it:
- Introduces unnecessary noise to test reports, or
- Blocks test cycles when the failure is a known, tracked issue awaiting an upstream fix

### Solution: Configuration YAML Files

`.yaml` files following the `00_default.yaml` format allow you to solve this problem, **without touching the actual test cases themselves**.

#### Step 1: Create or Update Configuration

Update the contents of `00_default.yaml` directly, or maintain your own set of separate `.yaml` files:

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
    - condition:
        name: "Assume NFS is present in kernel command line"
        script: |
          #!/usr/bin/python3
          return 'nfs' in open('/proc/cmdline').read()
  conditional-skip:
    - condition:
        name: "Skip because is not ready on Ubuntu 20.04"
        distro: "Ubuntu"
        release: "20.04"
    - condition:
        name: "Just a test"
        script: |
          #!/bin/bash
          grep -q 'nfs' /proc/cmdline
```

### Configuration Breakdown

**Format:** `<category>:<test_case_name>`
- Example: `arm64:test_all_pmu_events`
- This identifies the test case that requires conditional result changes

**Alternative Results:**
There are 2 valid alternatives:
- **`allow-failure`**: Keeps the failed outcome of the test case, but does not count towards "failed"
- **`conditional-skip`**: Will result in the test case not being run

**Condition Fields:**
- **`name`**: Please put in the reason you're applying "allow-failure" or "conditional-skip"
- **Conditions can be based on:**
  - **Distribution** (`distro` and `release`)
  - **Script execution** (`script`) - whether a script returns True

#### Step 2: Deploy Configuration

Copy your `.yaml` file(s) to the device under test's `/usr/share/greenlit/confs.d` directory.

- Multiple files are allowed and will all be detected and applied
- Your next rounds of tests' test results should reflect the configurations set in these `.yaml` files

For more details on configuration options, see the [Configuration Guide](../user-guides/configuration).
