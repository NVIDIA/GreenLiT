#!/usr/bin/env bash
set -uo pipefail

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# MAIR_EL1 WAR Test Suite
#
# Purpose:
# Validates ARM64 MAIR_EL1 (Memory Attribute Indirection Register) configuration
# fixes. Incorrect MAIR_EL1 configuration can cause wrong memory attributes for
# MMIO mappings, leading to data corruption and incorrect memory ordering.
#
# Current test cases:
# 1. ioremap_wc() API validation - Tests write-combine mapping correctness
#
#
#

# Exit codes:
# 0 = PASS
# 1 = FAIL
# 4 = SKIP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOREMAP_MODULE_NAME="test_ioremap_wc_api"
IOREMAP_MODULE_DIR="${SCRIPT_DIR}/${IOREMAP_MODULE_NAME}"
PGPROT_MODULE_NAME="test_pgprot_api"
PGPROT_MODULE_DIR="${SCRIPT_DIR}/${PGPROT_MODULE_NAME}"
PGPROT_USER_BIN="${SCRIPT_DIR}/${PGPROT_MODULE_NAME}/test_pgprot_api_user"
DMA_ALLOC_MODULE_NAME="test_dma_alloc"
DMA_ALLOC_MODULE_DIR="${SCRIPT_DIR}/${DMA_ALLOC_MODULE_NAME}"

# Default test configuration (will be overridden by auto-discovery)
PHYS_ADDR="0x0"
MAP_SIZE="0x0"
RW_LEN="0x1000"           # touch first 4 KiB
RW_PATTERN="0xa5a5a5a5"   # 32-bit test pattern

PCI_BDF=""
PCI_BAR=""

TMP_KO_DIR=""

# MOK key path (Ubuntu default, shared with dkms and shim-signed)
MOK_KEY_DIR="/var/lib/shim-signed/mok"
MOK_PRIV="${MOK_KEY_DIR}/MOK.priv"
MOK_DER="${MOK_KEY_DIR}/MOK.der"

is_secure_boot_enabled() {
    local sb_var="/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c"
    [[ -f "${sb_var}" ]] || return 1
    # Last byte: 1 = enabled
    local val
    val=$(od -An -tx1 -j4 -N1 "${sb_var}" 2>/dev/null | tr -d ' ')
    [[ "${val}" == "01" ]]
}

# Well-known path where users can place sign-file manually
# (e.g. from Buildroot or a custom kernel build).
SIGN_FILE_OVERRIDE="/usr/local/bin/sign-file"

find_sign_file() {
    local kver
    kver="$(uname -r)"

    # 1. User override (for Buildroot / custom kernels)
    [[ -x "${SIGN_FILE_OVERRIDE}" ]] && { echo "${SIGN_FILE_OVERRIDE}"; return 0; }

    # 2. PATH lookup
    local path_sign_file
    path_sign_file="$(command -v sign-file 2>/dev/null)" && { echo "${path_sign_file}"; return 0; }

    # 3. Distro-specific paths
    local candidates=(
        "/usr/src/linux-headers-${kver}/scripts/sign-file"
        "/usr/src/kernels/${kver}/scripts/sign-file"
        "/usr/lib/linux-kbuild/scripts/sign-file"
    )
    for c in "${candidates[@]}"; do
        [[ -x "${c}" ]] && { echo "${c}"; return 0; }
    done

    # 4. Glob fallback across all installed header versions
    local m
    for m in /usr/src/linux-headers-*/scripts/sign-file \
             /usr/src/kernels/*/scripts/sign-file; do
        [[ -x "${m}" ]] && { echo "${m}"; return 0; }
    done

    return 1
}

# Sign a .ko module if Secure Boot is enabled.
# Returns 0 on success or if signing is not needed.
# Returns 1 if signing is needed but failed.
sign_module_if_needed() {
    local ko_path="$1"

    if ! is_secure_boot_enabled; then
        return 0
    fi

    if [[ ! -f "${MOK_PRIV}" || ! -f "${MOK_DER}" ]]; then
        echo "ERROR: Secure Boot is enabled but MOK keys not found at ${MOK_KEY_DIR}" >&2
        echo "See docs/user-guides/secure-boot.md for setup instructions." >&2
        return 1
    fi

    local sign_file
    if ! sign_file="$(find_sign_file)"; then
        echo "ERROR: Cannot find 'sign-file' utility. Either:" >&2
        echo "  1. Install kernel headers:" >&2
        echo "     apt install linux-headers-\$(uname -r)  OR" >&2
        echo "     dnf install kernel-devel-\$(uname -r)" >&2
        echo "  2. Copy sign-file from your kernel build tree to ${SIGN_FILE_OVERRIDE}" >&2
        return 1
    fi

    echo "Signing module ${ko_path} for Secure Boot..."
    if ! "${sign_file}" sha256 "${MOK_PRIV}" "${MOK_DER}" "${ko_path}"; then
        echo "ERROR: Failed to sign ${ko_path}" >&2
        return 1
    fi

    return 0
}

# shellcheck disable=SC2317
cleanup() {
    set +e

    if lsmod | awk '{print $1}' | grep -qx "${IOREMAP_MODULE_NAME}"; then
        rmmod "${IOREMAP_MODULE_NAME}" >/dev/null 2>&1
    fi

    if lsmod | awk '{print $1}' | grep -qx "${PGPROT_MODULE_NAME}"; then
        rmmod "${PGPROT_MODULE_NAME}" >/dev/null 2>&1
    fi

    if lsmod | awk '{print $1}' | grep -qx "${DMA_ALLOC_MODULE_NAME}"; then
        rmmod "${DMA_ALLOC_MODULE_NAME}" >/dev/null 2>&1
    fi

    if [[ -d "${IOREMAP_MODULE_DIR}" && -f "${IOREMAP_MODULE_DIR}/Makefile" ]]; then
        (cd "${IOREMAP_MODULE_DIR}" && make clean >/dev/null 2>&1)
    fi

    if [[ -d "${PGPROT_MODULE_DIR}" && -f "${PGPROT_MODULE_DIR}/Makefile" ]]; then
        (cd "${PGPROT_MODULE_DIR}" && make clean >/dev/null 2>&1)
    fi

    if [[ -d "${DMA_ALLOC_MODULE_DIR}" && -f "${DMA_ALLOC_MODULE_DIR}/Makefile" ]]; then
        (cd "${DMA_ALLOC_MODULE_DIR}" && make clean >/dev/null 2>&1)
    fi

    if [[ -n "${TMP_KO_DIR}" && -d "${TMP_KO_DIR}" ]]; then
        rm -rf "${TMP_KO_DIR}"
    fi

    rm -f "${PGPROT_USER_BIN}"
}
trap cleanup EXIT


require_root() {
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        echo "must run as root"
        exit 4
    fi
}

get_last_result() {
    # $1: dmesg line count *before* this run started
    local start_line=${1:-0}

    # Show only lines added after start_line, then search newest→oldest
    dmesg | tail -n +"$((start_line + 1))" | tac | grep -m1 "${IOREMAP_MODULE_NAME}:" || true
}

# Find a suitable unbound PCI device and MEM BAR via /sys/bus/pci/devices
# On success, sets: PCI_BDF, PCI_BAR, PHYS_ADDR, MAP_SIZE and returns 0.
# On failure, returns 1 (caller will SKIP).
find_free_pci_bar() {
    local dev bdf resource_file
    local idx start_hex end_hex flags_hex
    local start_dec end_dec size_dec flags_dec
    local bar_val
    local vendor class class_major

    for dev in /sys/bus/pci/devices/*; do
        [[ -e "${dev}" ]] || continue

        # Skip devices already bound to a driver
        if [[ -e "${dev}/driver" ]]; then
            continue
        fi

        # Restrict to NVIDIA GPU or NVIDIA/Mellanox (BlueField / ConnectX) NICs
        if [[ ! -r "${dev}/vendor" || ! -r "${dev}/class" ]]; then
            continue
        fi
        vendor=$(<"${dev}/vendor")
        class=$(<"${dev}/class")

        case "${vendor}" in
            0x10de)
                # NVIDIA GPU: must be display/3D controller (class 0x03xxxx)
                if [[ "${class}" != 0x03* ]]; then
                    continue
                fi
                ;;
            0x15b3)
                # NVIDIA/Mellanox NIC / BlueField: require network- or IB-class
                # class_major = "0x02" or "0x0b"
                class_major=${class:0:4}
                if [[ "${class_major}" != 0x02 && "${class_major}" != 0x0b ]]; then
                    continue
                fi
                ;;
            *)
                # Not NVIDIA GPU, not Mellanox/BlueField NIC, skip
                continue
                ;;
        esac

        bdf="${dev##*/}"
        resource_file="${dev}/resource"
        [[ -r "${resource_file}" ]] || continue

        idx=0
        while read -r start_hex end_hex flags_hex; do
            # Only consider the first 6 entries as BAR0–BAR5; idx 6 is ROM
            if (( idx >= 6 )); then
                break
            fi

            # Empty line (no BAR) or past last entry
            [[ -z "${start_hex}" ]] && { ((idx++)); continue; }

            start_dec=$((start_hex))
            end_dec=$((end_hex))
            flags_dec=$((flags_hex))

            # IORESOURCE_MEM is 0x00000200
            if (( (flags_dec & 0x200) == 0 )); then
                ((idx++))
                continue
            fi

            if (( end_dec <= start_dec )); then
                ((idx++))
                continue
            fi

            size_dec=$((end_dec - start_dec + 1))

            # Require at least 4 KiB
            if (( size_dec < 0x1000 )); then
                ((idx++))
                continue
            fi

            # Read BAR value from config space for this index
            bar_val="$(setpci -s "${bdf}" "BASE_ADDRESS_${idx}.L" 2>/dev/null || true)"
            if [[ -z "${bar_val}" ]]; then
                ((idx++))
                continue
            fi

            # bar_val is hex without 0x; convert and check prefetchable bit (bit 3 = 0x8)
            if (( (0x${bar_val} & 0x8) == 0 )); then
                ((idx++))
                continue
            fi

            # Found a suitable prefetchable MEM BAR on a GPU or BlueField NIC
            PCI_BDF="${bdf}"
            PCI_BAR="${idx}"
            PHYS_ADDR=$(printf "0x%x" "${start_dec}")
            MAP_SIZE=$(printf "0x%x" "${size_dec}")

            if ! [[ "${PCI_BAR}" =~ ^[0-9]+$ ]] || (( PCI_BAR < 0 || PCI_BAR > 5 )); then
                echo "${IOREMAP_MODULE_NAME}: SKIPPED (invalid PCI_BAR index: ${PCI_BAR})"
                return 4
            fi

            echo "${IOREMAP_MODULE_NAME}: selected ${PCI_BDF} BAR${PCI_BAR} phys=${PHYS_ADDR} size=${MAP_SIZE} (prefetchable)" >&2
            return 0

        done < "${resource_file}"
    done

    return 1
}


run_ioremap_wc_api_test() {
    # Returns: "PASS", "FAIL", or "SKIP"
    
    if [[ ! -d "${IOREMAP_MODULE_DIR}" ]]; then
        echo "${IOREMAP_MODULE_NAME}: SKIPPED (module directory not found: ${IOREMAP_MODULE_DIR})"
        return 4
    fi

    if ! command -v make >/dev/null 2>&1; then
        echo "${IOREMAP_MODULE_NAME}: SKIPPED (make not available)"
        return 4
    fi

    # Auto-select an unbound PCI device and MEM BAR
    if ! find_free_pci_bar; then
        echo "${IOREMAP_MODULE_NAME}: SKIPPED (no suitable unbound prefetchable MEM BAR: all PCIe devices may be bound to drivers, no BAR meets the minimum size requirement (4 KiB)," \
            " or no available PCIe device with prefetchable memory)"
        return 4
    fi

    echo "${IOREMAP_MODULE_NAME}: using phys_addr=${PHYS_ADDR} map_size=${MAP_SIZE}"
    echo "${IOREMAP_MODULE_NAME}: RW_LEN=${RW_LEN} pattern=${RW_PATTERN}"
    echo "${IOREMAP_MODULE_NAME}: PCI_BDF=${PCI_BDF} PCI_BAR=${PCI_BAR}"
    echo "${IOREMAP_MODULE_NAME}: building ${IOREMAP_MODULE_NAME} in ${IOREMAP_MODULE_DIR}"

    if ! (cd "${IOREMAP_MODULE_DIR}" && make); then
        echo "${IOREMAP_MODULE_NAME}: FAILED (build error)"
        return 1
    fi

    local ko_src="${IOREMAP_MODULE_DIR}/${IOREMAP_MODULE_NAME}.ko"
    if [[ ! -f "${ko_src}" ]]; then
        echo "${IOREMAP_MODULE_NAME}: FAILED (built module not found: ${ko_src})"
        return 1
    fi

    TMP_KO_DIR="$(mktemp -d "/tmp/${IOREMAP_MODULE_NAME}.XXXXXX")"
    local ko="${TMP_KO_DIR}/${IOREMAP_MODULE_NAME}.ko"
    cp "${ko_src}" "${ko}"

    echo "${IOREMAP_MODULE_NAME}: loading module ${IOREMAP_MODULE_NAME} with R/W + devm + PCI BAR test"

    if ! sign_module_if_needed "${ko}"; then
        echo "${IOREMAP_MODULE_NAME}: FAILED (module signing failed for Secure Boot)"
        return 1
    fi

    # Record dmesg line count before insmod so we only search this run's output
    local dmesg_start
    dmesg_start=$(dmesg | wc -l)

    if ! [[ "${PCI_BAR}" =~ ^[0-9]+$ ]] || (( PCI_BAR < 0 || PCI_BAR > 5 )); then
        echo "${IOREMAP_MODULE_NAME}: SKIPPED (invalid PCI_BAR index: ${PCI_BAR})"
        return 4
    fi

    if ! insmod "${ko}" \
        phys_addr_param="${PHYS_ADDR}" \
        map_size="${MAP_SIZE}" \
        rw_len="${RW_LEN}" \
        rw_pattern="${RW_PATTERN}" \
        pci_bdf="${PCI_BDF}" \
        pci_bar="${PCI_BAR}"; then
        echo "${IOREMAP_MODULE_NAME}: insmod returned error (will inspect dmesg)" >&2
    fi

    local last_line
    last_line="$(get_last_result "${dmesg_start}")"

    if [[ -z "${last_line}" ]]; then
        dmesg | tail -n 50 || true
        echo "${IOREMAP_MODULE_NAME}: FAILED (no result line found in dmesg)"
        return 1
    elif [[ "${last_line}" == *"PASSED"* ]]; then
        echo "${IOREMAP_MODULE_NAME}: PASSED"
        return 0
    elif [[ "${last_line}" == *"FAIL"* ]]; then
        echo "${IOREMAP_MODULE_NAME}: ${last_line}"
        return 1
    else
        dmesg | tail -n 50 || true
        echo "${IOREMAP_MODULE_NAME}: FAILED (unknown result: ${last_line})"
        return 1
    fi
}


run_pgprot_test() {
    # Build and load the pgprot module, then run the user-space tests
    # Returns: 0 = PASS, 1 = FAIL, 4 = SKIP

    if [[ ! -d "${PGPROT_MODULE_DIR}" ]]; then
        echo "${PGPROT_MODULE_NAME}: SKIPPED (module directory not found: ${PGPROT_MODULE_DIR})"
        return 4
    fi

    if ! command -v make >/dev/null 2>&1; then
        echo "${PGPROT_MODULE_NAME}: SKIPPED (make not available)"
        return 4
    fi

    if ! command -v gcc >/dev/null 2>&1; then
        echo "${PGPROT_MODULE_NAME}: SKIPPED (gcc not available)"
        return 4
    fi

    echo "${PGPROT_MODULE_NAME}: building pgprot module in ${PGPROT_MODULE_DIR}"
    if ! (cd "${PGPROT_MODULE_DIR}" && make); then
        echo "${PGPROT_MODULE_NAME}: FAILED (module build error)"
        return 1
    fi

    local pgprot_ko="${PGPROT_MODULE_DIR}/${PGPROT_MODULE_NAME}.ko"
    if [[ ! -f "${pgprot_ko}" ]]; then
        echo "${PGPROT_MODULE_NAME}: FAILED (built module not found: ${pgprot_ko})"
        return 1
    fi

    echo "${PGPROT_MODULE_NAME}: loading pgprot module"
    if ! sign_module_if_needed "${pgprot_ko}"; then
        echo "${PGPROT_MODULE_NAME}: FAILED (module signing failed for Secure Boot)"
        return 1
    fi
    if ! insmod "${pgprot_ko}"; then
        echo "${PGPROT_MODULE_NAME}: FAILED (insmod failed)"
        dmesg | tail -n 20 || true
        return 1
    fi

    echo "${PGPROT_MODULE_NAME}: building user-space test binary"
    if ! gcc -O2 -Wall -o "${PGPROT_USER_BIN}" "${PGPROT_USER_BIN}.c"; then
        echo "${PGPROT_MODULE_NAME}: FAILED (build error)"
        return 1
    fi

    echo "${PGPROT_MODULE_NAME}: testing /dev/pgprot_wc and /dev/pgprot_dma"
    local ret=0
    # device, expect_sigbus (0|1), test (1=aligned 0=unaligned)
    if ! "${PGPROT_USER_BIN}" "/dev/pgprot_wc" "1" "1" || \
       ! "${PGPROT_USER_BIN}" "/dev/pgprot_wc" "1" "0"; then
        ret=1
    fi
    if ! "${PGPROT_USER_BIN}" "/dev/pgprot_dma" "0" "1" || \
       ! "${PGPROT_USER_BIN}" "/dev/pgprot_dma" "0" "0"; then
        ret=1
    fi

    if [[ ${ret} -eq 0 ]]; then
        echo "${PGPROT_MODULE_NAME}: PASSED"
        return 0
    else
        echo "ERROR: pgprot user-space test failed" >&2
        echo "${PGPROT_MODULE_NAME}: FAILED"
        return 1
    fi
}


run_dma_alloc_test() {

    if [[ ! -d "${DMA_ALLOC_MODULE_DIR}" ]]; then
        echo "${DMA_ALLOC_MODULE_NAME}: SKIPPED (module directory not found: ${DMA_ALLOC_MODULE_DIR})"
        return 4
    fi

    echo "${DMA_ALLOC_MODULE_NAME}: building module in ${DMA_ALLOC_MODULE_DIR}"
    if ! (cd "${DMA_ALLOC_MODULE_DIR}" && make); then
        echo "${DMA_ALLOC_MODULE_NAME}: FAILED (module build error)"
        return 1
    fi

    local dma_alloc_ko="${DMA_ALLOC_MODULE_DIR}/${DMA_ALLOC_MODULE_NAME}.ko"
    if [[ ! -f "${dma_alloc_ko}" ]]; then
        echo "${DMA_ALLOC_MODULE_NAME}: FAILED (built module not found: ${dma_alloc_ko})"
        return 1
    fi

    echo "${DMA_ALLOC_MODULE_NAME}: loading module"
    if ! sign_module_if_needed "${dma_alloc_ko}"; then
        echo "${DMA_ALLOC_MODULE_NAME}: FAILED (module signing failed for Secure Boot)"
        return 1
    fi
    if ! insmod "${dma_alloc_ko}"; then
        if dmesg | tail -n 30 | grep -i "${DMA_ALLOC_MODULE_NAME}: SKIPPED"; then
            echo "${DMA_ALLOC_MODULE_NAME}: SKIPPED (no suitable PCI device found)."
            echo "  Possible reasons: all PCIe devices are bound to drivers,"
            echo "  bus master is not enabled, or no non-bridge PCI device exists."
            return 4
        fi
        echo "${DMA_ALLOC_MODULE_NAME}: FAILED (insmod failed)"
        dmesg | tail -n 50 || true
        return 1
    fi

    if ! rmmod -v ${DMA_ALLOC_MODULE_NAME}; then
        echo "${DMA_ALLOC_MODULE_NAME}: WARNING (rmmod failed; leaving module loaded)"
    fi

    echo "${DMA_ALLOC_MODULE_NAME}: PASSED"
    return 0
}


check_prerequisites() {
    if ! command -v gcc >/dev/null 2>&1; then
        echo "ERROR: gcc is required but not found in PATH"
        return 4
    fi

    if ! command -v make >/dev/null 2>&1; then
        echo "ERROR: make is required but not found in PATH"
        return 4
    fi

    return 0
}


main() {
    require_root
    check_prerequisites

    # Check if NVIDIA PCIE Ordering Workaround is enabled in kernel log
    if ! journalctl -k | grep -i "CPU features: detected: MAIR_EL1 NC to nGnRE"; then
        echo "${IOREMAP_MODULE_NAME}: SKIPPED (NVIDIA PCIE Ordering Workaround not enabled in kernel log)"
        exit 4
    fi

    echo "${IOREMAP_MODULE_NAME}: NVIDIA PCIE Ordering Workaround detected in kernel log"

    echo "=== Running ioremap_wc() APIs test ==="
    local ioremap_wc_api_test_result="SKIPPED"
    set +e  # Temporarily disable exit on error
    run_ioremap_wc_api_test
    local ret=$?
    set -e  # Re-enable exit on error
    if [[ ${ret} -eq 0 ]]; then
        ioremap_wc_api_test_result="PASSED"
    elif [[ ${ret} -eq 1 ]]; then
        ioremap_wc_api_test_result="FAILED"
    else
        ioremap_wc_api_test_result="SKIPPED"
    fi

    local pgprot_api_test_result="SKIPPED"
    set +e
    run_pgprot_test
    ret=$?
    set -e
    if [[ ${ret} -eq 0 ]]; then
        pgprot_api_test_result="PASSED"
    elif [[ ${ret} -eq 1 ]]; then
        pgprot_api_test_result="FAILED"
    else
        pgprot_api_test_result="SKIPPED"
    fi

    local dma_alloc_api_test_result="SKIPPED"
    set +e
    run_dma_alloc_test
    ret=$?
    set -e
    if [[ ${ret} -eq 0 ]]; then
        dma_alloc_api_test_result="PASSED"
    elif [[ ${ret} -eq 1 ]]; then
        dma_alloc_api_test_result="FAILED"
    else
        dma_alloc_api_test_result="SKIPPED"
    fi

    echo ""
    echo "=== Test Summary ==="
    echo "ioremap_wc() APIs test:  ${ioremap_wc_api_test_result}"
    echo "pgprot APIs test:     ${pgprot_api_test_result}"
    echo "dma_alloc APIs test   : ${dma_alloc_api_test_result}"

    if [[ "${ioremap_wc_api_test_result}" == "FAILED" || "${pgprot_api_test_result}" == "FAILED" || "${dma_alloc_api_test_result}" == "FAILED" ]]; then
        echo "Overall: FAILED"
        exit 1
    elif [[ "${ioremap_wc_api_test_result}" == "SKIPPED" && "${pgprot_api_test_result}" == "SKIPPED" && "${dma_alloc_api_test_result}" == "SKIPPED" ]]; then
        echo "Overall: SKIPPED"
        exit 4
    else
        echo "Overall: PASSED"
        exit 0
    fi
}

main "$@"