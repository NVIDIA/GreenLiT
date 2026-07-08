# GreenLiT Troubleshooting Guide

## Executing GreenLiT

### `sudo glt-test-suite` results in `command not found`

This is likely to happen if you've performed a `pip install` on RHEL installations. Use the command `sudo /usr/local/bin/glt-test-suite` instead.

### Tests fail, skip, or behave oddly when run without root or without the target hardware

`glt-test-suite` must be run as root and interacts directly with kernel interfaces and physical devices (TPM, GPU, IPMI, etc.) on the actual device under test. If a required device isn't present, the affected test case should skip itself (or, for a kexec-dependent test, print a message and skip if `kexec` isn't installed/working) rather than fail outright — a hard failure without the expected hardware usually means a prerequisite is missing rather than a real product bug.

### A `conditional-skip` / `allow-failure` YAML entry doesn't seem to take effect

The most common cause is the entry's key not matching the real test identifier. Configuration entries must be keyed exactly as `category:test_case_name` (the same identifier printed by `sudo glt-test-suite --list`), with the operation and condition list nested beneath it — not wrapped in an extra top-level key, and not using a plain string as the condition. See [Configuration Guide](../user-guides/configuration.md) for the exact structure and the [Skipping Tests](../user-guides/user-manual.md#skipping-tests) section for a worked example. Unrecognized keys (like a `skip-reason` field) are silently ignored rather than raising an error, so a config that "does nothing" is often just a shape mismatch — check it against the structure in the Configuration Guide.

### A helper function's call to `GLTTestResult.fail()` / `.skip()` / `.warn()` doesn't change the reported result

These methods record the result on the **immediate caller's** `attr` object, which only exists if the immediate caller is a function decorated with `@GLTCore.GLTTestAttr()`. Calling them from a nested helper (not the decorated test function itself) raises the corresponding exception but silently fails to update the test's recorded result, since the helper has no `attr`. Either call `fail()`/`skip()`/`warn()` directly from the decorated test function, or pass the decorated test function explicitly via the `func=` keyword argument when calling from a helper.

### An external test with exit code 3 (XPASS) shows up as FAILED, not PASSED

This is expected: XPASS means the test was expected to fail (per its own script logic) but passed instead, and GreenLiT treats an unexpected pass as a hard failure. See the exit code table in [Create an External Test Case](../developer-docs/writing-test-cases.md#exit-codes).

## Getting Help

For issues not covered here, check the relevant guide first — [User Manual](../user-guides/user-manual.md), [Configuration Guide](../user-guides/configuration.md), or [Writing Test Cases](../developer-docs/writing-test-cases.md) — then file a bug or feature request as described in the [README's Support section](../../README.md#support).

