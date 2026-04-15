# Makefile for FEAT_LS64 test programs

# Detect host architecture
HOST_ARCH := $(shell uname -m)

# Compiler selection
ifeq ($(HOST_ARCH),aarch64)
    # Native ARM64 build
    CC ?= gcc
    $(info Building natively for aarch64)
else
    # Cross-compile for ARM64
    CC := aarch64-linux-gnu-gcc
    $(info Cross-compiling from $(HOST_ARCH) to aarch64)
endif

CFLAGS = -Wall -Wextra -O2 -g
LDFLAGS =

# Target binary
TARGET = test_ls64

# Source files
SRC = test_ls64.c

# Object files
OBJ = $(SRC:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ)
