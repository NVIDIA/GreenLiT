#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# Basic functional test for NVIDIA Tegra410 PMU drivers
# Tests presence and operational status of all PMU types
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PERF_EVENT_DIR="/sys/bus/event_source/devices"
TEST_DURATION=0.1  # seconds for perf stat tests
FAILED_TESTS=0
PASSED_TESTS=0
SKIPPED_TESTS=0

# Exit codes
EXIT_PASS=0    # Test passed successfully
EXIT_FAIL=1    # Test failed
EXIT_SKIP=4    # Test should be skipped

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED_TESTS++))
}

# Check if running as root (needed for some perf operations)
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_warn "Not running as root. Some tests may fail due to permissions."
        log_warn "Consider running with: sudo $0"
        return 1
    fi
    return 0
}

# Check if running on Tegra410 (Vera) platform
check_t410_platform() {
    log_info "Checking T410 (Vera) platform..."
    local t410_soc_id="jep106:036b:0410"

    for soc_dir in /sys/devices/soc*; do
        [ -r "$soc_dir/soc_id" ] || continue
        local soc_id
        soc_id=$(cat "$soc_dir/soc_id" 2>/dev/null)
        [ "$soc_id" = "$t410_soc_id" ] || continue
        log_info "T410 (Vera) detected (SoC: $soc_id)"
        return 0
    done

    log_skip "Not T410 platform (requires SoC: $t410_soc_id)"
    return 1
}

# Check if perf tool is available
check_perf() {
    # Allow override via PERF environment variable
    if [ -n "$PERF" ]; then
        if [ ! -x "$PERF" ]; then
            log_error "perf tool not found at: $PERF"
            exit 1
        fi
        log_info "Using perf: $PERF ($($PERF --version))"
    elif command -v perf &> /dev/null; then
        PERF="perf"
        log_info "Found perf: $(perf --version)"
    else
        log_error "perf tool not found."
        log_error "Either install linux-tools package or set PERF environment variable:"
        log_error "  export PERF=/path/to/kernel/tools/perf/perf"
        log_error "  sudo -E ./test_tegra410_pmu.sh"
        exit 1
    fi
}

# Test if a PMU device exists in sysfs
test_pmu_present() {
    local pmu_pattern=$1
    local description=$2
    
    log_info "Testing $description..."
    
    # Use glob pattern directly instead of ls
    local pmu_devices=()
    for pmu_dev in "${PERF_EVENT_DIR}"/${pmu_pattern}; do
        # Check if the glob matched any files
        [ -e "$pmu_dev" ] || continue
        pmu_devices+=("$pmu_dev")
    done
    
    if [ ${#pmu_devices[@]} -eq 0 ]; then
        log_skip "$description - No devices found matching pattern: $pmu_pattern"
        return 1
    fi
    
    log_pass "$description - Found ${#pmu_devices[@]} device(s)"
    
    # Print details for each device
    for pmu_dev in "${pmu_devices[@]}"; do
        local pmu_name
        pmu_name=$(basename "$pmu_dev")
        echo "  Device: $pmu_name"
        
        # Check for key sysfs files
        if [ -r "$pmu_dev/type" ]; then
            local pmu_type
            pmu_type=$(cat "$pmu_dev/type")
            echo "    Type: $pmu_type"
        fi
        
        if [ -r "$pmu_dev/cpumask" ]; then
            local cpumask
            cpumask=$(cat "$pmu_dev/cpumask")
            echo "    CPU mask: $cpumask"
        fi
        
        if [ -d "$pmu_dev/events" ]; then
            local num_events
            num_events=$(find "$pmu_dev/events" -maxdepth 1 -type f 2>/dev/null | wc -l)
            echo "    Events: $num_events available"
        fi
        
        if [ -d "$pmu_dev/format" ]; then
            local num_formats
            num_formats=$(find "$pmu_dev/format" -maxdepth 1 -type f 2>/dev/null | wc -l)
            echo "    Format attributes: $num_formats"
        fi
    done
    
    return 0
}

# Get first available event for a PMU
get_first_event() {
    local pmu_name=$1
    local pmu_path="${PERF_EVENT_DIR}/${pmu_name}"
    
    # Try to find 'cycles' event first as it's most common
    if [ -f "$pmu_path/events/cycles" ]; then
        echo "cycles"
        return 0
    fi
    
    # Otherwise, get the first available event
    local first_event
    first_event=$(find "$pmu_path/events/" -maxdepth 1 -type f 2>/dev/null | head -1)
    if [ -n "$first_event" ]; then
        basename "$first_event"
        return 0
    fi
    
    return 1
}

# Test if we can count events with perf stat
test_pmu_operational() {
    local pmu_name=$1
    local description=$2
    
    if [ ! -d "${PERF_EVENT_DIR}/${pmu_name}" ]; then
        log_skip "$description - PMU $pmu_name not present"
        return 1
    fi
    
    # Get first available event
    local event
    event=$(get_first_event "$pmu_name")
    if [ -z "$event" ]; then
        log_skip "$description - No events available"
        return 1
    fi
    
    log_info "Testing $description: $PERF stat -a -e ${pmu_name}/${event}/ sleep $TEST_DURATION"
    
    # Run perf stat and capture output
    local perf_output
    if perf_output=$($PERF stat -a -e "${pmu_name}/${event}/" sleep $TEST_DURATION 2>&1); then
        # Check if we got a valid count (not <not counted> or <not supported>)
        if echo "$perf_output" | grep -q "not counted\|not supported"; then
            log_fail "$description - Event not counted"
            echo "$perf_output" | grep -A 2 -B 2 "$pmu_name"
            return 1
        else
            log_pass "$description - Event counted successfully (${event})"
            # Show the counter value
            local count_line
            count_line=$(echo "$perf_output" | grep "$pmu_name" || echo "")
            if [ -n "$count_line" ]; then
                echo "  $count_line"
            fi
            return 0
        fi
    else
        log_fail "$description - perf stat failed"
        echo "$perf_output" | head -20
        return 1
    fi
}

# List all events for a PMU
list_pmu_events() {
    local pmu_name=$1
    local pmu_path="${PERF_EVENT_DIR}/${pmu_name}"
    
    if [ ! -d "$pmu_path/events" ]; then
        return
    fi
    
    echo "  Available events for $pmu_name:"
    for event_file in "$pmu_path/events"/*; do
        if [ -r "$event_file" ]; then
            local event_name
            event_name=$(basename "$event_file")
            local event_config
            event_config=$(cat "$event_file")
            printf "    %-30s : %s\n" "$event_name" "$event_config"
        fi
    done
}

# Test CPU core vendor events (Olympus)
test_vendor_events() {
    log_info "Testing CPU core vendor events..."
    
    # Check if we're running on NVIDIA Olympus CPU
    local cpu_partnum=""
    local cpu_implementer=""
    
    if [ -r "/proc/cpuinfo" ]; then
        cpu_implementer=$(grep -m1 "CPU implementer" /proc/cpuinfo | awk '{print $4}')
        cpu_partnum=$(grep -m1 "CPU part" /proc/cpuinfo | awk '{print $4}')
    fi
    
    # Check for NVIDIA implementer (0x4e) and Olympus part (0x100 or 0x010 depending on format)
    if [ "$cpu_implementer" != "0x4e" ]; then
        log_skip "CPU vendor events - Not running on NVIDIA CPU (implementer=$cpu_implementer)"
        return 1
    fi
    
    # Part number can be reported as 0x100 (12-bit) or 0x010 (short form)
    if [ "$cpu_partnum" != "0x100" ] && [ "$cpu_partnum" != "0x010" ]; then
        log_skip "CPU vendor events - Not Olympus CPU (part=$cpu_partnum, expected 0x100/0x010)"
        return 1
    fi
    
    log_info "Detected NVIDIA Olympus CPU (implementer=$cpu_implementer, part=$cpu_partnum)"
    
    # Check if perf can list vendor events
    local vendor_events_output
    if ! vendor_events_output=$($PERF list 2>&1); then
        log_warn "perf list failed, skipping vendor events test"
        return 1
    fi
    
    # Look for some Olympus-specific or common vendor events
    local test_events=("CPU_CYCLES" "INST_RETIRED" "L1D_CACHE_REFILL" "BR_MIS_PRED")
    local found_events=0
    
    echo "  Checking for vendor events:"
    for event in "${test_events[@]}"; do
        # Check both raw event name and potential aliases
        if echo "$vendor_events_output" | grep -qi "$event"; then
            echo "    ✓ $event - available"
            found_events=$((found_events + 1))
        fi
    done
    
    if [ $found_events -eq 0 ]; then
        log_skip "CPU vendor events - No vendor events found in perf list"
        return 1
    fi
    
    # Try to count a basic CPU event
    log_info "Testing CPU core event counting: $PERF stat -e cycles -a sleep $TEST_DURATION"
    
    local perf_output
    set +e  # Temporarily disable exit on error
    perf_output=$($PERF stat -e cycles -a sleep $TEST_DURATION 2>&1)
    local perf_exit=$?
    set -e  # Re-enable exit on error
    
    if [ $perf_exit -ne 0 ]; then
        log_fail "CPU vendor events - perf stat failed (exit code: $perf_exit)"
        echo "$perf_output" | head -10
        return 1
    fi
    
    if echo "$perf_output" | grep -q "not counted\|not supported"; then
        log_fail "CPU vendor events - cycles event not counted"
        return 1
    else
        log_pass "CPU vendor events - Core PMU operational"
        local count_line
        count_line=$(echo "$perf_output" | grep "cycles" | head -1 || echo "")
        if [ -n "$count_line" ]; then
            echo "  $count_line"
        fi
        
        # Check if vendor-specific metrics are loaded (informational only)
        echo ""
        echo "  Checking for vendor-specific metrics (from JSON files):"
        set +e  # Don't fail on grep mismatches
        local has_metrics=0
        
        # Check for some Tegra410 Olympus specific metrics
        if $PERF list metric 2>/dev/null | grep -qi "backend_bound"; then
            echo "    ✓ backend_bound metric found"
            has_metrics=1
        fi
        
        if $PERF list metric 2>/dev/null | grep -qi "ipc\|instructions.*per.*cycle"; then
            echo "    ✓ IPC metric found"
            has_metrics=1
        fi
        
        if $PERF list metric 2>/dev/null | grep -qi "frontend_bound"; then
            echo "    ✓ frontend_bound metric found"
            has_metrics=1
        fi
        
        set -e  # Re-enable exit on error
        
        if [ $has_metrics -eq 1 ]; then
            echo "  ✓ Vendor events/metrics loaded from JSON files"
        else
            echo "  ○ Vendor events not loaded (using standard ARM events only)"
        fi
        
        return 0
    fi
}

# Main test execution
main() {
    echo "========================================"
    echo "Tegra410 PMU Functional Test"
    echo "========================================"
    echo ""
    
    # Check platform first
    if ! check_t410_platform; then
        echo ""
        echo "========================================"
        echo "Test Summary"
        echo "========================================"
        echo "Passed:  0"
        echo "Failed:  0"
        echo "Skipped: 1"
        echo ""
        log_warn "Test suite skipped - not running on T410 platform."
        exit $EXIT_SKIP
    fi
    echo ""
    
    check_perf
    check_root
    echo ""
    
    # Test 1: UCF PMU
    echo "----------------------------------------"
    echo "Test 1: UCF (Unified Coherence Fabric) PMU"
    echo "----------------------------------------"
    if test_pmu_present "nvidia_ucf_pmu_*" "UCF PMU"; then
        # List available UCF PMUs and test each
        for pmu_dev in "${PERF_EVENT_DIR}"/nvidia_ucf_pmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            list_pmu_events "$pmu_name"
            test_pmu_operational "$pmu_name" "$pmu_name basic event"
        done
    fi
    echo ""
    
    # Test 2: PCIE PMU
    echo "----------------------------------------"
    echo "Test 2: PCIE PMU"
    echo "----------------------------------------"
    if test_pmu_present "nvidia_pcie_pmu_*" "PCIE PMU"; then
        for pmu_dev in "${PERF_EVENT_DIR}"/nvidia_pcie_pmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            list_pmu_events "$pmu_name"
            test_pmu_operational "$pmu_name" "$pmu_name basic event"
        done
    fi
    echo ""
    
    # Test 3: PCIE-TGT PMU
    echo "----------------------------------------"
    echo "Test 3: PCIE-TGT PMU"
    echo "----------------------------------------"
    if test_pmu_present "nvidia_pcie_tgt_pmu_*" "PCIE-TGT PMU"; then
        for pmu_dev in "${PERF_EVENT_DIR}"/nvidia_pcie_tgt_pmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            list_pmu_events "$pmu_name"
            test_pmu_operational "$pmu_name" "$pmu_name basic event"
        done
    fi
    echo ""
    
    # Test 4: CMEM Latency PMU
    echo "----------------------------------------"
    echo "Test 4: CPU Memory (CMEM) Latency PMU"
    echo "----------------------------------------"
    if test_pmu_present "nvidia_cmem_latency_pmu_*" "CMEM Latency PMU"; then
        for pmu_dev in "${PERF_EVENT_DIR}"/nvidia_cmem_latency_pmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            list_pmu_events "$pmu_name"
            test_pmu_operational "$pmu_name" "$pmu_name basic event"
        done
    fi
    echo ""
    
    # Test 5: NVLink-C2C PMU
    echo "----------------------------------------"
    echo "Test 5: NVLink-C2C PMU"
    echo "----------------------------------------"
    if test_pmu_present "nvidia_nvlink_c2c_pmu_*" "NVLink-C2C PMU"; then
        for pmu_dev in "${PERF_EVENT_DIR}"/nvidia_nvlink_c2c_pmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            
            # Check peer information if available
            if [ -r "$pmu_dev/peer" ]; then
                echo "  Peer info: $(cat "$pmu_dev/peer")"
            fi
            
            list_pmu_events "$pmu_name"
            test_pmu_operational "$pmu_name" "$pmu_name basic event"
        done
    fi
    echo ""
    
    # Test 6: NV-CLink PMU
    echo "----------------------------------------"
    echo "Test 6: NV-CLink PMU"
    echo "----------------------------------------"
    if test_pmu_present "nvidia_nvclink_pmu_*" "NV-CLink PMU"; then
        for pmu_dev in "${PERF_EVENT_DIR}"/nvidia_nvclink_pmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            list_pmu_events "$pmu_name"
            test_pmu_operational "$pmu_name" "$pmu_name basic event"
        done
    fi
    echo ""
    
    # Test 7: NV-DLink PMU
    echo "----------------------------------------"
    echo "Test 7: NV-DLink PMU"
    echo "----------------------------------------"
    if test_pmu_present "nvidia_nvdlink_pmu_*" "NV-DLink PMU"; then
        for pmu_dev in "${PERF_EVENT_DIR}"/nvidia_nvdlink_pmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            list_pmu_events "$pmu_name"
            test_pmu_operational "$pmu_name" "$pmu_name basic event"
        done
    fi
    echo ""
    
    # Test 8: Check for arm_cspmu entries (used by some Tegra410 PMUs)
    echo "----------------------------------------"
    echo "Test 8: ARM CoreSight PMU (used by Tegra410)"
    echo "----------------------------------------"
    if test_pmu_present "arm_cspmu_*" "ARM CoreSight PMU"; then
        # Just list them, don't test as they may be the same as above
        for pmu_dev in "${PERF_EVENT_DIR}"/arm_cspmu_*; do
            [ -e "$pmu_dev" ] || continue
            local pmu_name
            pmu_name=$(basename "$pmu_dev")
            echo "  Found: $pmu_name"
            if [ -r "$pmu_dev/identifier" ]; then
                echo "    Identifier: $(cat "$pmu_dev/identifier")"
            fi
        done
    fi
    echo ""
    
    # Test 9: CPU Core Vendor Events (Tegra410 Olympus)
    echo "----------------------------------------"
    echo "Test 9: CPU Core Vendor Events (Tegra410 Olympus)"
    echo "----------------------------------------"
    set +e  # Temporarily disable exit on error for vendor events test
    test_vendor_events
    set -e  # Re-enable exit on error
    echo ""
    
    # Summary
    echo "========================================"
    echo "Test Summary"
    echo "========================================"
    echo "Passed:  $PASSED_TESTS"
    echo "Failed:  $FAILED_TESTS"
    echo "Skipped: $SKIPPED_TESTS"
    echo ""
    
    # Determine exit code based on results
    if [ $FAILED_TESTS -gt 0 ]; then
        log_error "Some tests failed. Check output above for details."
        exit $EXIT_FAIL
    elif [ $PASSED_TESTS -eq 0 ] && [ $SKIPPED_TESTS -gt 0 ]; then
        log_warn "All tests were skipped - no PMUs found on this system."
        exit $EXIT_SKIP
    elif [ $PASSED_TESTS -gt 0 ]; then
        log_info "All tests passed!"
        exit $EXIT_PASS
    else
        log_warn "No tests were run."
        exit $EXIT_SKIP
    fi
}

# Run main function
main "$@"
