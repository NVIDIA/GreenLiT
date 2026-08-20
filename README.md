# GreenLiT

GreenLiT is a comprehensive test framework designed for Linux-based distros, specifically tailored for NVIDIA BaseOSes.

## Overview
GreenLiT provides an automated, reliable, and extensible testing framework for validating Linux-based distributions on NVIDIA hardware. It simplifies the process of running system-level tests, hardware validation, and OS-level checks.

## Getting Started

### Option A: Direct Installation
Clone this repository and run the installer:

```bash
git clone https://github.com/NVIDIA/greenlit.git
cd greenlit
sudo ./install.sh
```

### Option B: Offline Installation
For DUTs (Device Under Test) without internet access, pre-build wheels on a connected system that matches the target DUT (same architecture, Python version, and compatible OS):

```bash
mkdir -p glt-wheels

# Build GreenLiT wheel
pip wheel . --wheel-dir glt-wheels/

# Download runtime dependencies
sudo poetry self add poetry-plugin-export
poetry export -f requirements.txt --output temp-requirements.txt --without-hashes
pip download -r temp-requirements.txt --only-binary=:all: --no-deps --dest glt-wheels/
rm temp-requirements.txt

# Package wheels
tar -czf glt-wheels-custom.tar.gz glt-wheels/
```

Then on the offline DUT:

```bash
tar -xzf greenlit-*.tar.gz --strip-components=1 -C greenlit/
tar -xzf glt-wheels-*.tar.gz
cd greenlit/
sudo ./install.sh --local-whl ../glt-wheels/
```

## Requirements
- **OS:** Linux-based distributions (Debian/Ubuntu, RHEL/CentOS, SUSE)
- **Hardware:** NVIDIA platforms (e.g., Grace, Vera, Tegra, DGX systems)
- **Runtime:** Python 3 with `pip`
- **Dependencies:** `build-essential` (or equivalent development tools), Linux kernel headers, `ipmitool`, `tar`, `perf`, and `freeipmi`. *(Note: The installer script will attempt to install these automatically).*

## Usage
Once installed, you can run the test suite directly.

```bash
# Run the default standard test plan
sudo glt-test-suite

# Run a specific test plan (standard or stress)
sudo glt-test-suite --standard

# Run specific test categories
sudo glt-test-suite --tpm --arm64 --ipmi
```
- More examples and tutorials: [User Manual](docs/user-guides/user-manual.md)

## Releases & Roadmap
- **Releases:** See the [GitHub Releases](https://github.com/NVIDIA/greenlit/releases) page for the latest versions and changelogs.

## Documentation
Comprehensive documentation is available in the `docs/` directory:
- [User Manual](docs/user-guides/user-manual.md)
- [Development Setup Guide](docs/developer-docs/development-setup.md)

## Contribution Guidelines
We welcome contributions!
- Start here: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Code of Conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

GreenLiT also supports Poetry for development installation. See the [Development Setup Guide](docs/developer-docs/development-setup.md) for details.

## Governance & Maintainers
- **Maintainers:** The NVIDIA BaseOS Team

## Security
- Vulnerability disclosure: [`SECURITY.md`](SECURITY.md)
- **Do not file public issues for security reports.** Please follow the instructions in `SECURITY.md`.

## Support
- **Level:** Maintained
- **How to get help:** Please report bugs or request features via [GitHub Issues](https://github.com/NVIDIA/greenlit/issues/new).

## License
This project is licensed under the Apache 2.0 License - see the [`LICENSE`](LICENSE) file for details.
