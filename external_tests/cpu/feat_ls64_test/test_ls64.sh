#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ARM64 FEAT_LS64 Test (LD64B/ST64B instructions)
# Tests kernel support for LS64 feature detection and instruction execution

set -e

EXIT_PASS=0
EXIT_FAIL=1
EXIT_SKIP=4

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $*"; exit $EXIT_SKIP; }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; exit $EXIT_FAIL; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; exit $EXIT_PASS; }

check_arm64_platform() {
    log_info "Checking platform architecture..."
    local arch
    arch=$(uname -m)
    if [ "$arch" != "aarch64" ]; then
        log_skip "Not ARM64 platform (detected: $arch, requires: aarch64)"
    fi
    log_info "ARM64 platform detected"
}

get_script_dir() {
    local script_path="${BASH_SOURCE[0]}"
    while [ -L "$script_path" ]; do
        local script_dir
        script_dir=$(dirname "$script_path")
        script_path=$(readlink "$script_path")
        [[ $script_path != /* ]] && script_path="$script_dir/$script_path"
    done
    dirname "$script_path"
}

build_test() {
    log_info "Building test program..."
    make -C "$1" 2>&1 || log_fail "Build failed"
    [ -f "$1/test_ls64" ] || log_fail "Binary not found"
    cp "$1/test_ls64" "$2"
    log_info "Build successful"
}

run_kvm_test() {
    local test_binary="$1"
    local kvm_output=""
    local kvm_rc=0

    log_info "Running KVM LS64 guest test..."
    kvm_output=$("$test_binary" kvm 2>&1) || kvm_rc=$?
    echo "$kvm_output"
    if [ "$kvm_rc" -ne 0 ]; then
        log_fail "KVM LS64 guest test failed (exit code: $kvm_rc); check kernel patches f174a9ffcd48 and 151b92c92a45"
    fi
    log_info "KVM LS64 guest test observed KVM_EXIT_ARM_LDST64B"
}

main() {
    log_info "ARM64 FEAT_LS64 Test"
    check_arm64_platform

    build_dir=$(get_script_dir)
    tmpdir=$(mktemp -d)
    trap 'make -C "$build_dir" clean &>/dev/null; rm -rf "$tmpdir"' EXIT

    build_test "$build_dir" "${tmpdir}/test_ls64"

    log_info "Running normal_memory test..."
    normal_memory_output=""
    normal_memory_test_rc=0
    normal_memory_output=$("${tmpdir}/test_ls64" normal_memory 2>&1) || normal_memory_test_rc=$?
    echo "$normal_memory_output"

    log_info "Running alignment test..."
    alignment_test_rc=0
    "${tmpdir}/test_ls64" alignment >/dev/null 2>&1 || alignment_test_rc=$?

    log_info "normal_memory test exit code: $normal_memory_test_rc"
    log_info "alignment test exit code: $alignment_test_rc"

    # Extract HWCAP3_LS64 status from output
    hwcap3_ls64_set=""
    if echo "$normal_memory_output" | grep -q "HWCAP3_LS64: SET"; then
        hwcap3_ls64_set=1
    fi

    # Exit code interpretation:
    #   0 - Test completed (no signal) - unexpected, should get SIGBUS
    #   1 - Generic error (usage, allocation failure, etc.)
    #   2 - SIGBUS received - expected for normal memory
    #   3 - SIGILL received - no hardware support
    if [ $normal_memory_test_rc -eq 3 ] || [ $alignment_test_rc -eq 3 ]; then
        # SIGILL - no hardware support
        if [ -n "$hwcap3_ls64_set" ]; then
            log_fail "HWCAP3_LS64 is set but instructions trigger SIGILL (kernel may not have enabled LS64)"
        else
            log_skip "No LS64 hardware support"
        fi
    elif [ $normal_memory_test_rc -eq 2 ]; then
        # SIGBUS on normal memory - expected behavior
        if [ -n "$hwcap3_ls64_set" ]; then
            run_kvm_test "${tmpdir}/test_ls64"
            log_pass "LS64 userspace and KVM support are functional"
        else
            log_fail "LS64 instructions work but HWCAP3_LS64 is not set (kernel detection issue)"
        fi
    elif [ $normal_memory_test_rc -eq 0 ]; then
        # No signal - unexpected, might have FEAT_LS64_WB
        if [ -n "$hwcap3_ls64_set" ]; then
            run_kvm_test "${tmpdir}/test_ls64"
            log_pass "LS64 userspace and KVM support are functional (FEAT_LS64_WB may be supported)"
        else
            log_fail "LS64 instructions executed without signal but HWCAP3_LS64 is not set"
        fi
    elif [ $normal_memory_test_rc -eq 1 ]; then
        # Generic error
        log_fail "Test program error (exit code: 1)"
    else
        log_fail "Unexpected test result (exit code: $normal_memory_test_rc)"
    fi
}

main "$@"
