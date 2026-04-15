# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Makefile for ARM SMMU v3 TLB Latency Test (DGX-8297)

# CUDA Compiler
NVCC := nvcc

# Check if NVCC is in PATH, otherwise use common paths
ifeq ($(shell command -v nvcc 2> /dev/null),)
    ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
        NVCC := /usr/local/cuda/bin/nvcc
    else ifneq ($(wildcard /usr/local/cuda-12/bin/nvcc),)
        NVCC := /usr/local/cuda-12/bin/nvcc
    else ifneq ($(wildcard /usr/local/cuda-11/bin/nvcc),)
        NVCC := /usr/local/cuda-11/bin/nvcc
    else ifneq ($(wildcard /opt/nvidia/cuda/bin/nvcc),)
        NVCC := /opt/nvidia/cuda/bin/nvcc
    else
        $(error CUDA Toolkit not found. Please install CUDA Toolkit or add nvcc to PATH)
    endif
endif

# Compiler flags
NVCC_FLAGS := -O2

# Target binary
TARGET := test_tlb_shootdown

# Source files
SOURCES := test_tlb_shootdown.cu

# Default target
all: $(TARGET)

# Build the CUDA test binary
$(TARGET): $(SOURCES)
	@echo "Compiling CUDA test program..."
	@echo "Using compiler: $(NVCC)"
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)
	@echo "Build successful: $(TARGET)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	@echo "Clean complete"

# Run the test with default parameters
test: $(TARGET)
	@echo "Running baseline test (100 iterations)..."
	./$(TARGET) 100 8e9
	@echo ""
	@echo "Running stress test (10000 iterations)..."
	./$(TARGET) 10000 8e9

# Display help information
help:
	@echo "Makefile for ARM SMMU v3 TLB Latency Test (DGX-8297)"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the test binary (default)"
	@echo "  clean      - Remove build artifacts"
	@echo "  test       - Build and run the test with default parameters"
	@echo "  help       - Display this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build the test"
	@echo "  make clean        # Clean build artifacts"
	@echo "  make test         # Build and run with default parameters"
	@echo "  ./$(TARGET) <iterations> <size>  # Run manually"
	@echo ""
	@echo "Example:"
	@echo "  ./$(TARGET) 100 8e9"

.PHONY: all clean test help

