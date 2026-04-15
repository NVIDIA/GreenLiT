#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ARM SMMU v3 TLB Latency Test
# Patch: https://git.kernel.org/torvalds/c/d5afb4b47e13

set -e

EXIT_PASS=0
EXIT_FAIL=1
EXIT_SKIP=4

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
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

check_nvidia_drivers() {
    log_info "Checking NVIDIA drivers..."

    # Get all device blocks with DRIVER=nvidia using udevadm
    local nvidia_info
    nvidia_info=$(udevadm info -e 2>/dev/null | awk -v RS= '/DRIVER=nvidia/')

    if [ -z "$nvidia_info" ]; then
        log_skip "NVIDIA drivers not loaded or not bound to any GPU device"
    fi

    # Count devices by counting device paths (P: lines)
    local nvidia_devices
    nvidia_devices=$(echo "$nvidia_info" | grep -c "^P:")
    log_info "NVIDIA driver bound to $nvidia_devices GPU device(s)"
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

build_cuda_test() {
    log_info "Building test program..."
    make -C "$1" 2>&1 || log_fail "Build failed"
    [ -f "$1/test_tlb_shootdown" ] || log_fail "Binary not found"
    cp "$1/test_tlb_shootdown" "$2"
    log_info "Build successful"
}

parse_perf_metric() {
    echo "$1" | grep "&&&& PERF $2" | awk '{print $4}'
}

run_tlb_test() {
    log_info "=========================================="
    log_info "TLB Shootdown Performance Test"
    log_info "=========================================="

    local baseline_iterations=100 stress_iterations=10000 alloc_size="8e9"

    log_info "Baseline: $baseline_iterations iterations x $alloc_size bytes..."
    baseline_output=$("$1" "$baseline_iterations" "$alloc_size" 2>&1) || log_fail "Baseline failed"
    echo "$baseline_output"
    baseline_release=$(parse_perf_metric "$baseline_output" "new_no_touch_release")
    [ -n "$baseline_release" ] || log_fail "Failed to parse baseline"
    log_info "Baseline: $baseline_release ms/iter"

    log_info ""
    log_info "Stress: $stress_iterations iterations x $alloc_size bytes..."
    stress_output=$("$1" "$stress_iterations" "$alloc_size" 2>&1) || log_fail "Stress failed"
    echo "$stress_output"
    stress_release=$(parse_perf_metric "$stress_output" "new_no_touch_release")
    [ -n "$stress_release" ] || log_fail "Failed to parse stress"
    log_info "Stress: $stress_release ms/iter"

    log_info ""
    log_info "Analyzing results..."
    iteration_ratio=$(awk "BEGIN {printf \"%.1f\", $stress_iterations / $baseline_iterations}")
    time_ratio=$(awk "BEGIN {printf \"%.1f\", $stress_release / $baseline_release}")
    max_expected_ratio=$(awk "BEGIN {printf \"%.1f\", $iteration_ratio * 1.5}")

    log_info "Iteration ratio: ${iteration_ratio}x"
    log_info "Time ratio: ${time_ratio}x"
    log_info "Max expected: ${max_expected_ratio}x"

    if [ "$(awk "BEGIN {print ($time_ratio > $max_expected_ratio)}")" -eq 1 ]; then
        log_error "Performance issue: ${time_ratio}x > ${max_expected_ratio}x"
        log_fail "TLB shootdown regression detected"
    fi

    log_pass "Performance acceptable: ${time_ratio}x <= ${max_expected_ratio}x"
}

main() {
    log_info "ARM SMMU v3 TLB Latency Test"
    check_grace_platform
    check_nvidia_drivers

    build_dir=$(get_script_dir)
    tmpdir=$(mktemp -d)
    trap 'make -C "$build_dir" clean &>/dev/null; rm -rf "$tmpdir"' EXIT

    build_cuda_test "$build_dir" "${tmpdir}/test_tlb_shootdown"
    run_tlb_test "${tmpdir}/test_tlb_shootdown"
}

main "$@"
