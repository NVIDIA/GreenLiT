# Development Setup Guide

GreenLiT uses Poetry for dependency management and packaging. Here's how to set up a development environment:

## Index

- [Requirements](#requirements)
- [Setting up Development Environment](#setting-up-development-environment)
- [Managing Dependencies](#managing-dependencies)

## Requirements

- Python 3.9 or higher
- Root/sudo privileges for system-level installation
- Clone greenlit repository or untar source tarball

## Setting up Development Environment

```bash
# Install Poetry system-wide
curl -sSL https://install.python-poetry.org | sudo POETRY_HOME=/usr/local python3 -

# Verify Poetry installation
poetry --version

# Install including development dependencies
poetry install --with dev

# Run linting
poetry run pylint src/
```

If you want to run your code in a Python virtual environment within the repository, please modify `poetry.toml` as follows:

```toml
[virtualenvs]
# Enable virtual environment creation for development
create = true
in-project = true
system-site-packages = false
```

### Manual Configuration Setup

After installing via Poetry, the following configuration needs to be done manually:

```bash
sudo mkdir -p /usr/share/greenlit/{confs.d,data,executing_dir}
sudo cp confs/hardware-manifest.json /usr/share/greenlit/
sudo cp confs/glt-confs.d/* /usr/share/greenlit/confs.d/
sudo cp data/* /usr/share/greenlit/data/
sudo cp -r external_tests/* /usr/share/greenlit/executing_dir/

# Set up systemd services
sudo mkdir -p /usr/local/lib/systemd/system
sudo cp confs/greenlit.service /usr/local/lib/systemd/system/
sudo cp confs/greenlit.timer /usr/local/lib/systemd/system/
sudo systemctl daemon-reload
```

## Managing Dependencies

```bash
# Add a runtime dependency
poetry add pyyaml

# Add a development dependency
poetry add --group dev pylint

# Update dependencies
poetry update

# Show current dependencies
poetry show
```

**Note**: `poetry add` and `poetry remove` automatically generate `poetry.lock`, please remove it manually since greenlit does not lock package versions.

