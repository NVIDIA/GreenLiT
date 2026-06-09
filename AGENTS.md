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

Always run before creating a commit or MR. Skip any check whose tool is not installed.

- `shellcheck` and `jq` are system tools.
- `pylint` and `pytest` require the `gltenv` venv below.

One-time venv setup:

```bash
python3 -m venv gltenv && source gltenv/bin/activate && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org . pylint pytest pytest-cov
```

Checks:

| Tool | Command |
|------|---------|
| Python lint | `source gltenv/bin/activate && find . -not -path "./gltenv/*" -name "*.py" -print0 \| xargs -0 pylint --init-hook="import sys; sys.path.extend(['src', '.'])"` |
| Shell lint | `find . -name "*.sh" -print0 \| xargs -0 shellcheck` |
| JSON validation | `find . -name "*.json" -print0 \| xargs -0 jq -r . > /dev/null` |
| Unit tests | `source gltenv/bin/activate && python -m pytest tests/ -v --cov=src --cov-report=term-missing` |

## Boundaries

- Do **not** create or commit `poetry.lock` — GreenLiT intentionally does not lock package versions.
- Do **not** run `glt-test-suite` or anything in `external_tests/` without root and real hardware — these tests interact with kernel interfaces and physical devices.
- Do **not** add new test categories without also updating `docs/developer-docs/architecture.md`.
- In non-root/CI environments, only run unit tests and lint.
- Do **not** call `GLTCore.GLTTestResult.fail()` or `GLTCore.GLTTestResult.skip()` from any function that is not decorated with `@GLTCore.GLTTestAttr()`.
- **Must** follow the [Recommended workflow](docs/user-guides/user-manual.md#recommended-workflow) in order when running GreenLiT on a target system, unless the user requests otherwise.
- Do **not** continue that workflow on macOS — stop after OS detection and tell the user Linux is required.
- Do **not** guess or assume platform, paths, or test plan; ask the user when uncertain.
- Do **not** search for kernel source trees; ask the user for the kernel source path.
- Do **not** run `glt-test-suite` until the user explicitly asks you to execute the test suite.
- **Stop** and report to the user if any workflow step fails.
