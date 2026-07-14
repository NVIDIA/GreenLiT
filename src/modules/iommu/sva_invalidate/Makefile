# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Makefile for the SVA invalidate-range size regression workload

# Detect host architecture
HOST_ARCH := $(shell uname -m)

# Compiler selection
ifeq ($(HOST_ARCH),aarch64)
    # Native ARM64 build
    CC ?= gcc
    $(info Building natively for aarch64)
else
    # Cross-compile for ARM64
    CC ?= aarch64-linux-gnu-gcc
    $(info Cross-compiling from $(HOST_ARCH) to aarch64)
endif

CFLAGS = -Wall -Wextra -O2 -g
LDFLAGS =
LDLIBS = -ldl

# Target binary
TARGET = sva_trigger

# Source files
SRC = sva_trigger.c

# Object files
OBJ = $(SRC:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ)
