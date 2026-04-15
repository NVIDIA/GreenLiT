#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

EXIT_PASS=0
EXIT_FAIL=1
EXIT_SKIP=4

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
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
        [ "$soc_id" = "$grace_soc_id" ] || continue
        log_info "Grace detected (SoC: $soc_id)"
        return 0
    done

    log_skip "Not Grace platform (requires SoC: $grace_soc_id)"
}

CPPC_WAS_LOADED=0

unload_cppc_cpufreq() {
    log_info "Checking cppc_cpufreq module status..."

    if lsmod | grep -q cppc_cpufreq; then
        log_info "cppc_cpufreq module is loaded, unloading for accurate test..."
        if modprobe -r cppc_cpufreq 2>&1 | grep -v "^$" >&2; then
            log_warn "Failed to unload cppc_cpufreq, continuing anyway..."
            return 1
        fi
        CPPC_WAS_LOADED=1
        log_info "cppc_cpufreq unloaded successfully"

        log_info "Cycling CPUs offline/online to restore firmware speed..."
        for cpu_dir in /sys/devices/system/cpu/cpu*/online; do
            [ -f "$cpu_dir" ] || continue
            cpu_num=$(echo "$cpu_dir" | grep -oP 'cpu\K[0-9]+')
            [ "$cpu_num" -eq 0 ] && continue
            echo 0 > "$cpu_dir" 2>/dev/null || true
            echo 1 > "$cpu_dir" 2>/dev/null || true
        done
        log_info "CPU cycle complete"

        return 0
    fi

    log_info "cppc_cpufreq module is not loaded"
    return 0
}

# shellcheck disable=SC2317
reload_cppc_cpufreq() {
    if [ "$CPPC_WAS_LOADED" -eq 1 ]; then
        log_info "Reloading cppc_cpufreq module..."
        if modprobe cppc_cpufreq 2>&1 | grep -v "^$" >&2; then
            log_warn "Failed to reload cppc_cpufreq"
        else
            log_info "cppc_cpufreq reloaded successfully"
        fi
    fi
}

run_mhz_test() {
    local mhz_binary=$1

    log_info "=========================================="
    log_info "CPU Clock Rate Test"
    log_info "=========================================="

    log_info "Running mhz binary: $mhz_binary"
    local mhz_output
    if ! mhz_output=$("$mhz_binary" 2>&1); then
        log_error "Failed to run mhz binary"
        log_error "Output: $mhz_output"
        log_fail "mhz execution failed"
    fi

    log_info "mhz output:"
    echo "$mhz_output"

    if echo "$mhz_output" | grep -q "81 MHz, 12.3457 nanosec clock"; then
        log_error "CPU is operating at 81 MHz!"
        log_error "This indicates the firmware regression has occurred."
        log_fail "CPU clock rate is invalid (81 MHz detected)"
    fi

    echo "$mhz_output" | grep -q "MHz" || log_fail "Unable to determine CPU frequency"

    local freq
    freq=$(echo "$mhz_output" | grep -oP '^\s*\d+' | head -1)
    [ -n "$freq" ] || log_fail "Unable to parse CPU frequency"

    log_info "Detected CPU frequency: ${freq} MHz"

    if [ "$freq" -lt 100 ]; then
        log_error "CPU frequency is too low: ${freq} MHz"
        log_error "Expected frequency should be much higher (typically > 1000 MHz)"
        log_fail "CPU clock rate is suspiciously low"
    fi

    log_pass "CPU clock rate is valid: ${freq} MHz"
}

main() {
    log_info "This test verifies that the CPU does not operate at 81MHz"

    trap reload_cppc_cpufreq EXIT INT

    check_grace_platform

    local mhz_binary="/usr/lib/lmbench/bin/mhz"
    if [ ! -x "$mhz_binary" ]; then
        log_skip "mhz binary not found at $mhz_binary. Please install lmbench package."
    fi

    unload_cppc_cpufreq
    run_mhz_test "$mhz_binary"
}

main "$@"
