#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Description: Verify that the kernel patch for T241-FABRIC-4 is present
# JIRA: DGX-8632
# Commit: https://git.kernel.org/torvalds/c/35727af2b15d

set -e

EXIT_PASS=0
EXIT_FAIL=1
EXIT_SKIP=4

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $*"; exit $EXIT_SKIP; }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; exit $EXIT_FAIL; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; exit $EXIT_PASS; }

check_grace_platform() {
    log_info "Checking Grace platform..."
    local grace_soc_id="jep106:036b:0241"
    for soc_dir in /sys/devices/soc*; do
        [ -f "$soc_dir/soc_id" ] || continue
        local soc_id
        soc_id=$(cat "$soc_dir/soc_id" 2>/dev/null)
        if [ "$soc_id" = "$grace_soc_id" ]; then
            log_info "Grace detected (SoC: $soc_id)"
            return 0
        fi
    done
    log_skip "Not Grace platform (requires SoC: $grace_soc_id)"
}

check_multi_socket() {
    log_info "Checking socket count..."

    if ! command -v lscpu &> /dev/null; then
        log_warn "lscpu command not found, skipping socket count check"
        return 0
    fi

    local socket_count
    socket_count=$(lscpu | grep -i "^Socket(s):" | awk '{print $2}')

    if [ -z "$socket_count" ]; then
        log_warn "Unable to determine socket count from lscpu"
        return 0
    fi

    if [ "$socket_count" -lt 2 ]; then
        log_info "Single socket system detected (Sockets: $socket_count)"
        log_info "Erratum T241-FABRIC-4 only affects multi-socket systems"
        log_skip "Test does not apply to this system"
    else
        log_info "Multi-socket system detected (Sockets: $socket_count)"
    fi
}

check_kallsyms() {
    log_info "Checking /proc/kallsyms availability..."
    if [ ! -f /proc/kallsyms ]; then
        log_skip "/proc/kallsyms not available"
    fi

    if [ ! -r /proc/kallsyms ]; then
        log_skip "/proc/kallsyms not readable (may require root permissions)"
    fi
}

check_arm_smccc_symbols() {
    if grep -q "arm_smccc_get_soc_id" /proc/kallsyms 2>/dev/null; then
        log_info "Found arm_smccc_get_soc_id symbols."
        log_pass "Kernel patch for T241-FABRIC-4 is present"
    else
        log_error "No arm_smccc_get_soc_id symbols found in /proc/kallsyms"
        log_error "This indicates the kernel patch for T241-FABRIC-4 is not applied"
        log_error "Patch: https://git.kernel.org/torvalds/c/35727af2b15d"
        log_fail "Kernel patch for T241-FABRIC-4 is not applied"
    fi
}

main() {
    log_info "Testing for kernel patch: https://git.kernel.org/torvalds/c/35727af2b15d"

    check_grace_platform
    check_multi_socket
    check_kallsyms
    check_arm_smccc_symbols
}

main "$@"
