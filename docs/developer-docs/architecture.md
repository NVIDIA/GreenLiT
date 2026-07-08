# GreenLiT Architecture Overview

This document provides an overview of the GreenLiT architecture, explaining how the different components work together.

Here's a high-level overview of the directory structure:

```
greenlit/
├── confs/                  # Test configuration files
├── data/                   # Test data files
├── docs/                   # Documentation
├── external_tests/         # Packaged standalone executable tests
├── install.sh              # Installation script
├── poetry.toml             # Poetry configuration file
├── pyproject.toml          # Project build configuration
├── src/                    # Test framework source code
│   ├── common/             # Common utilities
│   │   └── virt/           # Libvirt/QEMU helper APIs for tests
│   ├── greenlit/           # Main package
│   ├── modules/            # Test modules
│   │   ├── acpi/           # ACPI and EFI tests
│   │   ├── arm64/          # ARM64 architecture tests
│   │   ├── cpu/            # CPU-related tests
│   │   ├── cxl/            # CXL device tests
│   │   ├── gpio/           # GPIO tests
│   │   ├── gpu/            # GPU and framebuffer tests
│   │   ├── hwtracing/      # Hardware tracing (perf, coresight)
│   │   ├── i2c/            # I2C bus tests
│   │   ├── iommu/          # IOMMU tests
│   │   ├── ipmi/           # IPMI interface tests
│   │   ├── irq/            # Interrupt tests
│   │   ├── irqchip/        # IRQ chip tests
│   │   ├── mm/             # Memory management tests
│   │   ├── mtd/            # MTD tests
│   │   ├── pci/            # PCI subsystem tests
│   │   ├── power/          # Power management tests
│   │   ├── rtc/            # Real-time clock tests
│   │   ├── system/         # System state and configuration tests
│   │   ├── tpm/            # TPM tests
│   │   └── virt/           # Virtualization tests
│   └── scripts/            # Standalone checker scripts (poetry entry points, e.g. check_kernel_config.py)
├── tests                   # Unit tests files
│   ├── common/             # Unit tests for common helpers
│   └── modules/            # Unit tests for module test logic
├── LICENSE                 # Apache License 2.0
├── README.md               # Project README
└── VERSION                 # Project version
```

## Test modules

Every in-tree test case belongs to exactly one **module** (also called a **category**): a subdirectory under `src/modules/`. The directory name is the module name (for example `cxl`, `pci`, `mm`). It is used for discovery, the `glt-test-suite --<module>` CLI flag, and the `category` field on `@GLTTestAttr` (inferred from the file path when not set explicitly).

When adding or moving a test case, if the user has **not** said which module it should live in:

1. **Ask** which module the test should belong to. List the plausible options from `src/modules/` (see the tree above).
2. **Recommend** one module yourself and briefly say why (for example a CXL memdev check → `cxl`; a generic PCI link test → `pci`).
3. If **no existing module** is a good fit, **ask the user** whether to add a new module. Do not create `src/modules/<new>/` without confirmation. A new module requires a new directory, updating the module list in this document, and the rules in [AGENTS.md](../../AGENTS.md).

Put the test in `src/modules/<module>/glt_test_<feature>.py` (new file or an existing one in that module). See [writing-test-cases.md](writing-test-cases.md) for how to implement the test function.

## Where to put changes

Use this table when modifying the repository. For how to run the suite on a system, see the user manual; for how to author tests, see the writing guide.

| Change | Location | Also see |
|--------|----------|----------|
| New in-tree test case | `src/modules/<module>/glt_test_<feature>.py` | [Test modules](#test-modules), [writing-test-cases.md](writing-test-cases.md) |
| New test module (category) | `src/modules/<module>/` (directory name = CLI flag) | Confirm with user first; update the module list in this doc; [AGENTS.md](../../AGENTS.md) |
| Shared helper used by tests | `src/common/` or `src/common/virt/` for libvirt/QEMU | Unit test under `tests/common/` |
| CLI, discovery, or scheduling | `src/greenlit/greenlit.py`, `src/common/glt_core.py` | — |
| External executable test | `external_tests/` (installed to `/usr/share/greenlit/executing_dir/`) | [user-manual.md](../user-guides/user-manual.md), [writing-test-cases.md](writing-test-cases.md) |
| Per-test skip or config YAML | `confs/glt-confs.d/` (installed to `/usr/share/greenlit/confs.d/`) | [configuration.md](../user-guides/configuration.md) |
| Hardware manifest | `confs/hardware-manifest.json` | — |
| Reference data (patches, kernel config lists, etc.) | `data/` (installed to `/usr/share/greenlit/data/`) | — |
| Standalone checker entry points (e.g. `check_kernel_config.py`, `check_kernel_patches.py`) | `src/scripts/` | `pyproject.toml` `[tool.poetry.scripts]` |
| Framework unit test | `tests/` (typically `tests/common/` for helpers or `tests/modules/` for module test logic) | [development-setup.md](development-setup.md), [AGENTS.md](../../AGENTS.md) |

Unit tests under `tests/modules/` validate module test logic with mocks. They do not define
new GreenLiT runtime test categories; runtime categories remain under `src/modules/`.

## Read also
* [user-manual.md](../user-guides/user-manual.md) — Runtime, CLI, automation on hardware
* [writing-test-cases.md](writing-test-cases.md) — Adding/editing tests
