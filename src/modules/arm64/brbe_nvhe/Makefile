# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

HOST_ARCH := $(shell uname -m)
ifeq ($(origin CC), default)
ifeq ($(HOST_ARCH),aarch64)
CC := gcc
else
CC := aarch64-linux-gnu-gcc
endif
endif

ifeq ($(strip $(KVM_MICROVM_DIR)),)
$(error KVM_MICROVM_DIR must name the common Arm64 KVM source directory)
endif

COMMON_SOURCE := $(KVM_MICROVM_DIR)/glt_arm64_kvm.c
CFLAGS += -std=gnu11 -O2 -g -Wall -Wextra -Werror -fPIE
CPPFLAGS += -I$(KVM_MICROVM_DIR)
ASFLAGS += -g
LDFLAGS += -pie

.PHONY: all

all:
	@set -eu; \
	output='$(OUTPUT)'; \
	case "$$output" in \
	/*) ;; \
	*) echo "OUTPUT must be an absolute executable path" >&2; exit 2 ;; \
	esac; \
	output_dir=$$(dirname -- "$$output"); \
	mkdir -p -- "$$output_dir"; \
	build_dir=$$(mktemp -d "$$output_dir/.glt-brbe-nvhe.XXXXXX"); \
	trap 'rm -rf -- "$$build_dir"' EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o "$$build_dir/harness.o" \
		glt_brbe_nvhe_harness.c; \
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o "$$build_dir/kvm.o" \
		$(COMMON_SOURCE); \
	$(CC) $(CPPFLAGS) $(ASFLAGS) -c -o "$$build_dir/guest.o" guest.S; \
	$(CC) $(LDFLAGS) -o "$$build_dir/glt_brbe_nvhe_harness" \
		"$$build_dir/harness.o" "$$build_dir/kvm.o" \
		"$$build_dir/guest.o"; \
	mv -f -- "$$build_dir/glt_brbe_nvhe_harness" "$$output"
