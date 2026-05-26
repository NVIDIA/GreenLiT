# GreenLiT Architecture Overview

This document provides an overview of the GreenLiT architecture, explaining how the different components work together.

**Under Construction**

This document is currently being revised. Please check back later for the complete architecture guide. In the meantime, here's a high-level overview of the directory structure:

```
greenlit/
├── confs/                  # Test configuration files
├── data/                   # Test data files
├── docs/                   # Wiki Documentation
├── external_tests/         # External category of test scripts
├── install.sh              # Installation script
├── poetry.toml             # Poetry configuration file
├── pyproject.toml          # Project build configuration
├── src/                    # Test framework source code
│   ├── common/             # Common utilities
│   │   └── virt/           # Libvirt/QEMU helper APIs for tests
│   ├── greenlit/           # Main package
│   └── modules/            # Test modules
│       ├── acpi/           # ACPI and EFI tests
│       ├── arm64/          # ARM64 architecture tests
│       ├── cpu/            # CPU-related tests
│       ├── cxl/            # CXL device tests
│       ├── gpio/           # GPIO tests
│       ├── gpu/            # GPU and framebuffer tests
│       ├── hwtracing/      # Hardware tracing (perf, coresight)
│       ├── i2c/            # I2C bus tests
│       ├── iommu/          # IOMMU tests
│       ├── ipmi/           # IPMI interface tests
│       ├── irq/            # Interrupt tests
│       ├── irqchip/        # IRQ chip tests
│       ├── kconfig/        # Kernel configuration validation
│       ├── mm/             # Memory management tests
│       ├── mtd/            # MTD tests
│       ├── pci/            # PCI subsystem tests
│       ├── power/          # Power management tests
│       ├── rtc/            # Real-time clock tests
│       ├── system/         # System state and configuration tests
│       ├── tpm/            # TPM tests
│       └── virt/           # Virtualization tests
├── tests                   # Unit tests files
├── COPYRIGHT               # Copyright information
├── README.md               # Project README
└── VERSION                 # Project version
```
