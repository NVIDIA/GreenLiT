#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# Tegra410 I2C Functional Test Script
#
# Tests the NVIDIA Tegra410 I2C driver functionality
# Verifies: bus detection, device scanning, read operations, error handling
#

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
TESTS_WARNING=0

# Configuration
VERBOSE=0
SPECIFIC_BUS=""
TEST_LEVEL=3  # Default to level 3 (basic read testing)

# Expected ACPI ID for Tegra410
TEGRA410_ACPI_ID="NVDA2017"

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to print test result
print_result() {
    local status=$1
    local message=$2
    
    case "$status" in
        PASS)
            print_color "$GREEN" "[PASS] $message"
            ((TESTS_PASSED++))
            ;;
        FAIL)
            print_color "$RED" "[FAIL] $message"
            ((TESTS_FAILED++))
            ;;
        SKIP)
            print_color "$YELLOW" "[SKIP] $message"
            ((TESTS_SKIPPED++))
            ;;
        WARN)
            print_color "$YELLOW" "[WARN] $message"
            ((TESTS_WARNING++))
            ;;
        INFO)
            print_color "$BLUE" "[INFO] $message"
            ;;
    esac
    ((TESTS_RUN++))
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_color "$RED" "Error: This script must be run as root (use sudo)"
        exit 1
    fi
}

# Function to check if running on Tegra410 (Vera) platform
check_t410_platform() {
    print_result INFO "Checking T410 (Vera) platform..."
    local t410_soc_id="jep106:036b:0410"

    for soc_dir in /sys/devices/soc*; do
        [ -f "$soc_dir/soc_id" ] || continue
        local soc_id
        soc_id=$(cat "$soc_dir/soc_id" 2>/dev/null)
        [ "$soc_id" = "$t410_soc_id" ] || continue
        print_result INFO "T410 (Vera) detected (SoC: $soc_id)"
        return 0
    done

    print_result SKIP "Not T410 platform (requires SoC: $t410_soc_id)"
    return 1
}

# Function to check for required tools
check_tools() {
    local missing_tools=0
    
    for tool in i2cdetect i2cget i2cdump; do
        if ! command -v "$tool" &> /dev/null; then
            print_result FAIL "Required tool not found: $tool"
            missing_tools=1
        fi
    done
    
    if [ $missing_tools -eq 1 ]; then
        print_color "$RED" "Please install i2c-tools: sudo apt-get install i2c-tools"
        exit 1
    fi
    
    print_result PASS "All required tools found (i2c-tools)"
}

# Function to find Tegra410 I2C buses
find_tegra410_buses() {
    local buses=()
    
    # Method 1: Check sysfs for ACPI ID
    for i2c_dev in /sys/bus/i2c/devices/i2c-*; do
        if [ -e "$i2c_dev/firmware_node/path" ]; then
            local acpi_path
            acpi_path=$(cat "$i2c_dev/firmware_node/path" 2>/dev/null)
            if [[ "$acpi_path" == *"$TEGRA410_ACPI_ID"* ]]; then
                local bus_num
                bus_num=$(basename "$i2c_dev" | sed 's/i2c-//')
                buses+=("$bus_num")
            fi
        fi
    done
    
    # Method 2: Check i2cdetect -l output
    if [ ${#buses[@]} -eq 0 ]; then
        while read -r line; do
            if [[ "$line" == *"$TEGRA410_ACPI_ID"* ]]; then
                local bus_num
                bus_num=$(echo "$line" | grep -oP 'i2c-\K[0-9]+')
                if [ -n "$bus_num" ]; then
                    buses+=("$bus_num")
                fi
            fi
        done < <(i2cdetect -l 2>/dev/null)
    fi
    
    # Return unique sorted bus numbers
    printf '%s\n' "${buses[@]}" | sort -u
}

# Level 1: Basic bus detection
test_level1_detection() {
    print_color "$BLUE" "\n=== Level 1: Bus Detection ==="
    
    # Test: i2cdetect -l works
    if ! i2cdetect -l &> /dev/null; then
        print_result FAIL "i2cdetect -l command failed"
        return 1
    fi
    print_result PASS "i2cdetect command executes successfully"
    
    # Find Tegra410 buses
    local buses=()
    while IFS= read -r bus; do
        buses+=("$bus")
    done < <(find_tegra410_buses)
    
    if [ ${#buses[@]} -eq 0 ]; then
        print_result FAIL "No Tegra410 I2C buses found (expected ACPI ID: $TEGRA410_ACPI_ID)"
        print_color "$YELLOW" "i2cdetect -l output:"
        i2cdetect -l
        return 1
    fi
    
    print_result PASS "Found ${#buses[@]} Tegra410 I2C bus(es): ${buses[*]}"
    
    # Display bus information (to stderr so it doesn't interfere with return value)
    print_color "$BLUE" "\nI2C Bus Information:" >&2
    i2cdetect -l | grep -E "$(IFS=\|; echo "${buses[*]/#/i2c-}")" >&2
    
    # Return buses for other tests (only this, nothing else)
    echo "${buses[@]}"
    return 0
}

# Level 2: Device scanning
test_level2_scanning() {
    local buses=("$@")
    print_color "$BLUE" "\n=== Level 2: Device Scanning ==="
    
    if [ ${#buses[@]} -eq 0 ]; then
        print_result SKIP "No buses to scan"
        return 0
    fi
    
    local scan_errors=0
    local devices_found=0
    
    for bus in "${buses[@]}"; do
        print_color "$BLUE" "\nScanning i2c-$bus..."
        
        # Run i2cdetect and capture output
        local scan_output
        scan_output=$(i2cdetect -y -r "$bus" 2>&1)
        local scan_result=$?
        
        if [ $scan_result -ne 0 ]; then
            print_result FAIL "Bus scan failed for i2c-$bus"
            echo "$scan_output"
            ((scan_errors++))
            continue
        fi
        
        echo "$scan_output"
        
        # Count detected devices (not "--" or "UU")
        local device_count
        device_count=$(echo "$scan_output" | grep -oE '[0-9a-f]{2}' | grep -vE '^(00|01|02|03|04|05|06|07|UU)$' | grep -c '^' || true)
        
        if [ "$device_count" -gt 0 ]; then
            print_result PASS "i2c-$bus: Scan completed, $device_count device(s) detected"
            ((devices_found += device_count))
        else
            print_result WARN "i2c-$bus: Scan completed, no devices detected (bus may be empty)"
        fi
    done
    
    if [ $scan_errors -gt 0 ]; then
        print_result FAIL "Bus scanning failed on $scan_errors bus(es)"
        return 1
    fi
    
    if [ $devices_found -eq 0 ]; then
        print_result WARN "No I2C devices detected on any bus (buses may be empty)"
    else
        print_result PASS "Total devices detected across all buses: $devices_found"
    fi
    
    return 0
}

# Level 3: Basic read operations
test_level3_read() {
    local buses=("$@")
    print_color "$BLUE" "\n=== Level 3: Read Operations ==="
    
    if [ ${#buses[@]} -eq 0 ]; then
        print_result SKIP "No buses to test"
        return 0
    fi
    
    local read_tested=0
    local read_success=0
    
    for bus in "${buses[@]}"; do
        # Find devices on this bus
        local devices
        devices=$(i2cdetect -y -r "$bus" 2>/dev/null | grep -oE ' [0-9a-f]{2} ' | tr -d ' ' | grep -vE '^(UU)$')
        
        if [ -z "$devices" ]; then
            [ $VERBOSE -eq 1 ] && print_result SKIP "i2c-$bus: No devices to read from"
            continue
        fi
        
        # Try reading from each device
        for addr in $devices; do
            local addr_dec=$((16#$addr))
            
            [ $VERBOSE -eq 1 ] && print_color "$BLUE" "Attempting read from i2c-$bus address 0x$addr..."
            
            # Try reading a byte
            local read_output
            read_output=$(i2cget -y "$bus" "$addr_dec" 0x00 2>&1)
            local read_result=$?
            
            ((read_tested++))
            
            if [ $read_result -eq 0 ]; then
                print_result PASS "i2c-$bus: Read from 0x$addr succeeded (value: $read_output)"
                ((read_success++))
            else
                # Some devices may not respond to address 0x00, this is not necessarily a driver failure
                [ $VERBOSE -eq 1 ] && print_result WARN "i2c-$bus: Read from 0x$addr failed (device may not support this register)"
            fi
        done
    done
    
    if [ $read_tested -eq 0 ]; then
        print_result WARN "No devices available to test read operations"
        return 0
    fi
    
    if [ $read_success -gt 0 ]; then
        print_result PASS "Read operations: $read_success/$read_tested successful"
        return 0
    else
        print_result WARN "Read operations: No successful reads (devices may not support test register)"
        return 0
    fi
}

# Check kernel logs for I2C errors
test_kernel_logs() {
    print_color "$BLUE" "\n=== Kernel Log Analysis ==="
    
    # Check for I2C initialization messages
    # Use journalctl to avoid dmesg ring buffer issues
    local init_msgs
    init_msgs=$(journalctl -k -b0 -g "$TEGRA410_ACPI_ID|i2c.*tegra" --no-pager 2>/dev/null | grep -i "probe\|init\|register" | tail -10)
    if [ -n "$init_msgs" ]; then
        print_result INFO "I2C driver initialization messages found"
        [ $VERBOSE -eq 1 ] && echo "$init_msgs"
    fi
    
    # Check for errors
    local error_msgs
    error_msgs=$(journalctl -k -b0 -g "i2c" --no-pager 2>/dev/null | grep -iE "error|timeout|arbitration|fail" | grep -v "i2c read failed" | tail -10)
    if [ -n "$error_msgs" ]; then
        print_result WARN "I2C error messages found in kernel log:"
        echo "$error_msgs"
    else
        print_result PASS "No critical I2C errors in kernel log"
    fi
    
    # Check DMA status (should NOT be enabled for Tegra410)
    local dma_msgs
    dma_msgs=$(journalctl -k -b0 -g "i2c.*dma" --no-pager 2>/dev/null | tail -5)
    if echo "$dma_msgs" | grep -iq "dma.*support.*not.*available\|dma.*not.*supported"; then
        print_result PASS "DMA correctly disabled for Tegra410"
        [ $VERBOSE -eq 1 ] && echo "$dma_msgs"
    elif echo "$dma_msgs" | grep -iq "dma.*enable\|dma.*init"; then
        print_result WARN "DMA appears to be enabled (expected disabled for Tegra410)"
        echo "$dma_msgs"
    else
        print_result INFO "No DMA-related messages (expected for Tegra410)"
    fi
}

# Verify driver features
test_driver_features() {
    print_color "$BLUE" "\n=== Driver Feature Verification ==="
    
    # Check for software mutex support
    if journalctl -k -b0 -g "i2c.*mutex" --no-pager 2>/dev/null | grep -iq "i2c.*mutex"; then
        print_result INFO "SW mutex messages found in logs"
    fi
    
    # Check for HS mode support
    if journalctl -k -b0 -g "i2c.*hs.*mode|i2c.*high.*speed" --no-pager 2>/dev/null | grep -iq "i2c.*hs.*mode\|i2c.*high.*speed"; then
        print_result INFO "HS mode support messages found"
    fi
    
    # Check i2c-dev module (needed for userspace access)
    if lsmod | grep -q i2c_dev; then
        print_result PASS "i2c-dev module loaded"
    elif [ -e /sys/module/i2c_dev ]; then
        print_result PASS "i2c-dev built-in to kernel"
    else
        print_result WARN "i2c-dev not available (may need: modprobe i2c-dev)"
    fi
    
    # Check tegra i2c driver
    if lsmod | grep -q i2c_tegra; then
        print_result PASS "i2c_tegra module loaded"
    elif [ -e /sys/module/i2c_tegra ]; then
        print_result PASS "i2c_tegra built-in to kernel"
    else
        print_result FAIL "i2c_tegra driver not found"
    fi
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -l, --level N       Run tests up to level N (1-3, default: 3)
    -b, --bus N         Test only specific bus number N
    
Test Levels:
    1 - Bus detection only
    2 - Bus detection + device scanning
    3 - Bus detection + scanning + read operations (default)

Examples:
    $0                  # Run all tests (level 3)
    $0 --level 1        # Only test bus detection
    $0 --bus 0          # Test only i2c-0
    $0 -v               # Verbose output

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -l|--level)
            TEST_LEVEL="$2"
            shift 2
            ;;
        -b|--bus)
            SPECIFIC_BUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Main test execution
main() {
    print_color "$GREEN" "========================================"
    print_color "$GREEN" "  Tegra410 I2C Functional Test Suite"
    print_color "$GREEN" "========================================"
    echo ""
    
    # Prerequisites
    check_root
    
    # Check platform first
    if ! check_t410_platform; then
        echo ""
        print_summary
    fi
    
    check_tools
    
    # Level 1: Detection - capture output and parse for bus numbers only
    local temp_file
    temp_file=$(mktemp)
    test_level1_detection > "$temp_file"
    
    # Extract only the numeric bus IDs from the output
    local buses=()
    while IFS= read -r bus; do
        buses+=("$bus")
    done < <(tail -1 "$temp_file" | tr ' ' '\n' | grep -E '^[0-9]+$')
    rm -f "$temp_file"
    
    if [ ${#buses[@]} -eq 0 ]; then
        print_color "$RED" "\nNo Tegra410 I2C buses found. Cannot continue testing."
        print_summary
    fi
    
    # Filter to specific bus if requested
    if [ -n "$SPECIFIC_BUS" ]; then
        local found=0
        for bus in "${buses[@]}"; do
            if [ "$bus" = "$SPECIFIC_BUS" ]; then
                found=1
                break
            fi
        done
        if [ $found -eq 1 ]; then
            buses=("$SPECIFIC_BUS")
            print_color "$BLUE" "\nTesting only bus i2c-$SPECIFIC_BUS"
        else
            print_color "$RED" "Requested bus i2c-$SPECIFIC_BUS not found in Tegra410 buses"
            exit 1
        fi
    fi
    
    # Level 2: Scanning
    if [ "$TEST_LEVEL" -ge 2 ]; then
        test_level2_scanning "${buses[@]}"
    fi
    
    # Level 3: Read operations
    if [ "$TEST_LEVEL" -ge 3 ]; then
        test_level3_read "${buses[@]}"
    fi
    
    # Additional checks
    test_kernel_logs
    test_driver_features
    
    # Summary
    print_summary
}

# Print test summary
print_summary() {
    echo ""
    print_color "$GREEN" "========================================"
    print_color "$GREEN" "  Test Summary"
    print_color "$GREEN" "========================================"
    echo "Total tests run:    $TESTS_RUN"
    print_color "$GREEN" "Passed:             $TESTS_PASSED"
    print_color "$RED" "Failed:             $TESTS_FAILED"
    print_color "$YELLOW" "Warnings:           $TESTS_WARNING"
    print_color "$YELLOW" "Skipped:            $TESTS_SKIPPED"
    echo ""
    
    # Determine exit code
    if [ $TESTS_FAILED -gt 0 ]; then
        print_color "$RED" "Result: FAIL"
        exit 1
    elif [ $TESTS_PASSED -gt 0 ]; then
        if [ $TESTS_WARNING -gt 0 ]; then
            print_color "$GREEN" "Result: PASS (with warnings)"
        else
            print_color "$GREEN" "Result: PASS"
        fi
        exit 0
    else
        print_color "$YELLOW" "Result: SKIP (no tests completed)"
        exit 4
    fi
}

# Run main function
main
