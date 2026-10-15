# AGENTS.md

Guidance for AI coding agents working in this repository.

**Read the docs first.** Before making changes, consult the relevant documentation in `docs/`:

| Topic | Doc |
|-------|-----|
| Project overview & usage | `docs/user-guides/user-manual.md` |
| Architecture & source layout | `docs/developer-docs/architecture.md` |
| Writing test cases | `docs/developer-docs/writing-test-cases.md` |
| Development setup & tooling | `docs/developer-docs/development-setup.md` |
| Test configuration (YAML) | `docs/user-guides/configuration.md` |
| Commit conventions | `CONTRIBUTING.md` |

## Local Checks

**GLT001 is mandatory on every agent run.** `GLTCore.GLTTestResult.fail()`,
`.skip()`, and `.warn()` must not exist in any function except the body of a
function decorated with `@GLTCore.GLTTestAttr(...)`. Helpers return information
to that test body. Nested helpers, lambdas, comprehensions, class or module
bodies, and definition expressions do not inherit this permission. The
generated `test_func` in `src/common/glt_ext_launcher.py` is the only
production exception.

Before finishing any task, run `python3 tools/check_glt_contracts.py` from the
repository root after your final edits, even for documentation-only changes.
This check requires only Python's standard library and must not be skipped.
Exit codes: **0** means no violations, **1** means contract violations were
found, and **2** means checking failed (for example, an unreadable or
unparseable file). Resolve failures caused by your changes and rerun. If an
unrelated violation or environment problem prevents a clean run, report the
blocker and exit code explicitly; do not claim the check passed.

Before creating a commit or MR, follow [Running Local Checks](docs/developer-docs/development-setup.md#running-local-checks).
Skip checks whose tools are unavailable, except the contract check.

## Boundaries

- Do **not** create or commit `poetry.lock` — GreenLiT intentionally does not lock package versions.
- Do **not** run `glt-test-suite` or anything in `external_tests/` without root and real hardware — these tests interact with kernel interfaces and physical devices.
- Do **not** add new test categories without also updating `docs/developer-docs/architecture.md`.
- In non-root/CI environments, only run unit tests and lint.
- Obey **GLT001** on every run: `GLTCore.GLTTestResult.fail()`, `.skip()`, and `.warn()` may exist only in a `@GLTCore.GLTTestAttr(...)` test body. `python3 tools/check_glt_contracts.py` enforces GLT001 and must be run before the task is finished.
- If the test validates a kernel, firmware, or userspace patch, inspect the related patch or source repository when available. If it is not available, ask for the relevant link, commit, or repository before guessing test behavior.
- If a test may contain non-public or sensitive details, stop and ask whether it belongs in the public GreenLiT repository.
- **Must** follow the [Recommended workflow](docs/user-guides/user-manual.md#recommended-workflow) in order when running GreenLiT on a target system, unless the user requests otherwise.
- Do **not** guess or assume platform, paths, or test plan; ask the user when uncertain.
- Do **not** search for kernel source trees; ask the user for the kernel source path.
- Do **not** run `glt-test-suite` until the user explicitly asks you to execute the test suite.
- **Stop** and report to the user if any workflow step fails.
