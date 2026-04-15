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

## Boundaries

- Do **not** create or commit `poetry.lock` — GreenLiT intentionally does not lock package versions.
- Do **not** run `glt-test-suite` or anything in `external_tests/` without root and real hardware — these tests interact with kernel interfaces and physical devices.
- Do **not** add new test categories without also updating `docs/developer-docs/architecture.md`.
- In non-root/CI environments, only run unit tests and lint.
