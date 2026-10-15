# Development Setup Guide

This guide covers setting up a local environment to work **on** GreenLiT itself
(running lint, running unit tests, iterating on source). If you instead want to
**install and use** GreenLiT to run tests against a target system, see the
[README](../../README.md#getting-started) — Option A (`install.sh`) for a
normal install, or Option B for an offline/wheel-based install.

## Index

- [Requirements](#requirements)
- [Setting up a Development Environment](#setting-up-a-development-environment)
- [Running Local Checks](#running-local-checks)
- [Manual Runtime Configuration](#manual-runtime-configuration)
- [Building Release Wheels (Poetry)](#building-release-wheels-poetry)
- [Managing Declared Dependencies (Poetry)](#managing-declared-dependencies-poetry)
- [Coding Style](#coding-style)

## Requirements

- Python 3.10 or higher
- Clone the greenlit repository or untar the source tarball
- Root/sudo privileges, but only if you go on to do the optional
  [manual runtime configuration](#manual-runtime-configuration) below

## Setting up a Development Environment

The recommended day-to-day workflow uses a plain `venv` named `gltenv` and
`pip install .` — this mirrors the setup CI uses to lint and test every
change, so running it locally closely reproduces CI.

One-time setup:

```bash
python3 -m venv gltenv && source gltenv/bin/activate && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org . pylint pytest pytest-cov
```

The `--trusted-host` flags disable certificate verification for those hosts,
which is only safe on a network where the path to PyPI is otherwise trusted
(e.g. behind a corporate TLS-inspecting proxy). Drop them if your network
reaches PyPI directly.

Re-activate `gltenv` (`source gltenv/bin/activate`) in each new shell before
running the checks below.

## Running Local Checks

Before an agent finishes any task, it must run the contract check after its
final edits, including documentation-only edits. This run enforces **GLT001**:
`GLTTestResult.fail()`, `.skip()`, and `.warn()` may exist only in a function
decorated with `@GLTCore.GLTTestAttr(...)`.

```bash
python3 tools/check_glt_contracts.py
```

This standard-library-only check must not be skipped. It parses all Python
files under the checkout's `src/` without importing or executing them.
Exit status **0** means no contract violations, **1** means violations were
found, and **2** means checking failed (for example, unreadable or unparseable
source). Checking failures take precedence if violations are also found.
Fix findings introduced by your changes and rerun; explicitly report any
unrelated findings or environment blockers that prevent a clean result.

For faster feedback while editing, pass individual files or directories:
`python3 tools/check_glt_contracts.py src/modules/cpu/`.
The final check must still use the default full-source scan.
`tests/test_glt_contracts.py` also runs this scan as part of the ordinary unit
tests, so CI jobs running `pytest tests/` enforce the same contract.
See [the result-call contract](writing-test-cases.md#result-call-contract-glt001)
for its scope and limitations.

Run these before creating a commit or MR. Skip any check whose tool isn't
installed, except the mandatory contract check above. `shellcheck` and `jq` are system tools; `pylint` and `pytest` come
from the `gltenv` venv set up above. [AGENTS.md](../../AGENTS.md) points here
for the canonical local-check commands.

| Tool | Command |
|------|---------|
| Contract check | `python3 tools/check_glt_contracts.py` |
| Python lint | `source gltenv/bin/activate && find . -not -path "./gltenv/*" -name "*.py" -print0 \| xargs -0 pylint --init-hook="import sys; sys.path.extend(['src', '.'])"` |
| Shell lint | `find . -name "*.sh" -print0 \| xargs -0 shellcheck` |
| JSON validation | `find . -name "*.json" -print0 \| xargs -0 jq -r . > /dev/null` |
| Unit tests | `source gltenv/bin/activate && python -m pytest tests/ -v --cov=src --cov-report=term-missing` |

## Manual Runtime Configuration

`pip install .` only installs the Python package — it does not stage private
runtime helpers, runtime configuration files, or systemd units that a real
GreenLiT install needs. `install.sh` does all of this automatically for end
users (see the [README](../../README.md)); if you're testing runtime/service
integration against a `pip install .`-based dev checkout without running the
full installer, stage it manually:

```bash
sudo mkdir -p /usr/share/greenlit/{bin,confs.d,executing_dir}
sudo cp -R src/bin/. /usr/share/greenlit/bin/
sudo cp confs/hardware-manifest.json /usr/share/greenlit/
sudo cp confs/glt-confs.d/* /usr/share/greenlit/confs.d/
sudo cp -r external_tests/* /usr/share/greenlit/executing_dir/

# Set up systemd services
sudo mkdir -p /usr/local/lib/systemd/system
sudo cp confs/greenlit.service /usr/local/lib/systemd/system/
sudo cp confs/greenlit.timer /usr/local/lib/systemd/system/
sudo systemctl daemon-reload
```

## Building Release Wheels (Poetry)

Poetry is used for exporting dependency metadata when packaging distributable
wheels (e.g. the offline install bundle described in the README's
[Option B](../../README.md#option-b-offline-installation)) — it is **not**
part of the day-to-day dev/lint/test loop above.

```bash
# Install Poetry system-wide (download, inspect, then run — avoid piping
# an unverified remote installer straight into sudo)
curl -fsSL -o /tmp/install-poetry.py https://install.python-poetry.org
less /tmp/install-poetry.py
sudo POETRY_HOME=/usr/local python3 /tmp/install-poetry.py
rm -f /tmp/install-poetry.py

# Verify Poetry installation
poetry --version

# Build the GreenLiT wheel (via pip, matching CI flow)
pip wheel . --wheel-dir glt-wheels/

# Export and download runtime dependency wheels
sudo poetry self add poetry-plugin-export
poetry export -f requirements.txt --output temp-requirements.txt --without-hashes
pip download -r temp-requirements.txt --only-binary=:all: --no-deps --dest glt-wheels/
rm temp-requirements.txt
```

See the README for the full offline packaging and installation flow.

## Managing Declared Dependencies (Poetry)

GreenLiT's runtime and packaging dependencies are declared in
`pyproject.toml`. Use Poetry to manage that file — these commands affect what
gets packaged (and what `pip install .` pulls in), not a separate Poetry-based
dev environment:

```bash
# Add a runtime dependency
poetry add pyyaml

# Update dependencies
poetry update

# Show current dependencies
poetry show
```

**Note**: `poetry add`, `poetry remove`, and `poetry update` automatically
generate `poetry.lock` — remove it manually, since GreenLiT intentionally does
not lock package versions.

## Coding Style

When adding or refactoring functions, keep them in alphabetical order within
their local section unless lifecycle, dependency, or framework ordering is
clearer. This applies to source files and related unit tests.
