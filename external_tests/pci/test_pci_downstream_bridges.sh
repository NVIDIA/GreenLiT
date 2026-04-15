#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: GPL-2.0
#
# Verify patch PCI: Use downstream bridges for distributing resources
# at systems with NVIDIA BF3 switch or CX8 DMA controller
#
run_pci_realloc_test() {
    SEARCH_STRING="Some PCI device resources are unassigned, try booting with pci=realloc"

    local found=0
    if command -v journalctl >/dev/null 2>&1 && journalctl -b0 -k -g "$SEARCH_STRING" --no-pager -q >/dev/null 2>&1; then
        found=1
    elif dmesg 2>/dev/null | grep -qF "$SEARCH_STRING"; then
        found=1
    fi

    # Check journalctl or dmesg for the message
    if [[ $found -eq 1 ]]; then
        echo "FOUND: PCI device resources unassigned message"
        echo ""
        echo "Message found:"
        echo "─────────────────────────────────────────────────────────────"
        if command -v journalctl >/dev/null 2>&1; then
           journalctl -b0 -k -g "$SEARCH_STRING" --no-pager 2>/dev/null || dmesg | grep -F "$SEARCH_STRING"
        else
           dmesg | grep -F "$SEARCH_STRING"
        fi
        echo ""
        DISABLED_REGIONS=$(lspci -vv 2>/dev/null | grep -E "Region.*Memory.*\[disabled\]")
        if [ -n "$DISABLED_REGIONS" ]; then
            echo "Found disabled memory regions:"
            echo "$DISABLED_REGIONS"
            echo ""
            echo "Context (10 lines before each disabled region):"
            echo "─────────────────────────────────────────────────────────────"
            lspci -vv 2>/dev/null | grep -B 10 -E "Region.*Memory.*\[disabled\]"
            echo ""
        fi
        return 1
    else
        echo "No PCI realloc message found."
        echo ""
        return 0
    fi
}

check_for_bf3_switch_or_cx8_dma_controller() {

    if ! command -v lspci >/dev/null 2>&1; then
        echo "Error: lspci not found"
        echo "Overall: SKIPPED"
        exit 4
    fi

    # Check if system has NVIDIA BF3 switch or CX8 DMA controller
    BF3=$(lspci -nD -d 15b3:197b 2>/dev/null || true)
    CX8=$(lspci -nD -d 15b3:2100 2>/dev/null || true)
    if [ -n "$BF3" ]; then
        echo "Detected: NVIDIA BF3 switch"
        echo ""
    elif [ -n "$CX8" ]; then
        echo "Detected: NVIDIA CX8 DMA controller"
        echo ""
    else
        echo "System does not have NVIDIA BlueField3 Switch or CX8 DMA Controller"
        echo "Overall: SKIPPED"
        exit 4
    fi
}

check_kernel_cmdline() {
    # Check if pci=realloc is already in kernel cmdline
    echo "Checking kernel command line..."
    KERNEL_CMDLINE=$(cat /proc/cmdline 2>/dev/null)

    if echo "$KERNEL_CMDLINE" | grep -qE "(^| )pci=realloc(=|[^=]|$)"; then
        echo "pci=realloc is in kernel command line"
        echo ""
        echo "Remove pci=realloc from kernel boot parameters"
        echo ""
        echo "Overall: SKIPPED"
        exit 4
    fi

    echo "No pci=realloc at kernel command line"
    echo ""
}

require_root() {
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        echo "must run as root"
        exit 4
    fi
}

main() {
    require_root
    check_for_bf3_switch_or_cx8_dma_controller
    check_kernel_cmdline

    echo "=== Checking for PCI Realloc Message ==="
    run_pci_realloc_test
    local ret=$?
    pci_realloc_test_result="SKIPPED"
    if [[ ${ret} -eq 0 ]]; then
        pci_realloc_test_result="PASSED"
    elif [[ ${ret} -eq 1 ]]; then
        pci_realloc_test_result="FAILED"
    fi


    echo ""
    echo "=== Test Summary ==="
    echo "Check for pci realloc test:  ${pci_realloc_test_result}"

    if [[ "${pci_realloc_test_result}" == "PASSED" ]]; then
        echo "Overall: PASSED"
        exit 0
    elif [[ "${pci_realloc_test_result}" == "SKIPPED" ]]; then
        echo "Overall: SKIPPED"
        exit 4
    else
        echo "Overall: FAILED"
        exit 1
    fi
}

main "$@"
