#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DGX-8273: CONFIG_TCG_TIS_SPI=y Test
# JIRA: DGX-8273
# NVBug: 3926738, 3930044

# shellcheck disable=SC2002,SC2317

set -e

EXIT_PASS=0
EXIT_FAIL=1
EXIT_SKIP=4

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODULE_LOADED_BY_US=0
HASH_ALG=""  # Detected hash algorithm (sha256 or sha384)

# SOC ID to platform name mapping
declare -A SOC_ID_TO_NAME
SOC_ID_TO_NAME["jep106:036b:0241"]="Grace"
SOC_ID_TO_NAME["jep106:036b:0410"]="Vera"

# PCR extend values for seal/unseal test
PCR16_SHA384_SEAL_EXTEND="09eac24fcc154444019f7d1135059d41e61897b30bb5de39384e405ca159d15aa5d02654fe9fa078a58c3a73c83751db"
PCR16_SHA256_SEAL_EXTEND="232803363ace3a7ce76a094885527297395e8a6dd1198c5b365f6bf59d5ef8ae"

# PCR extend values for PCR reset/extend test
PCR16_SHA384_TEST_EXTEND="21ac2468cfebea2806c8407805239c78b263d2350b3c0687e498e9511483e6192f6bf923345db2b19f6ca0a370b00f08"
PCR16_SHA256_TEST_EXTEND="232803363ace3a7ce76a094885527297395e8a6dd1198c5b365f6bf59d5ef8ae"

# Expected PCR values after reset
PCR16_SHA384_RESET="0x000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
PCR16_SHA256_RESET="0x0000000000000000000000000000000000000000000000000000000000000000"

# Expected PCR values after extend (for PCR reset/extend test)
PCR16_SHA384_TEST_EXTENDED="0xB5A2B1E6633D1F1CBB59806CB0C590C05B630A95394B1588E4CE6352CFD5AB9FF8B5AE9D8C006D1FE7A3FEC9FA29E1FD"
PCR16_SHA256_TEST_EXTENDED="0x208F4312FF6518B1F222DE5B2284E1A6424C067DC81B88C1416C553A34892A39"

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $*"; exit $EXIT_SKIP; }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; exit $EXIT_FAIL; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; exit $EXIT_PASS; }

check_nvidia_platform() {
    log_info "Checking for NVIDIA platform..."
    
    for soc_dir in /sys/devices/soc*; do
        [ -f "$soc_dir/soc_id" ] || continue
        local soc_id
        soc_id=$(cat "$soc_dir/soc_id" 2>/dev/null)
        
        # Check if this SOC ID is in our supported platforms
        if [[ -n "${SOC_ID_TO_NAME[$soc_id]+x}" ]]; then
            log_info "${SOC_ID_TO_NAME[$soc_id]} platform detected (SoC: $soc_id)"
            return 0
        fi
    done
    
    # Build list of supported platforms for error message
    local supported_platforms=""
    for soc_id in "${!SOC_ID_TO_NAME[@]}"; do
        supported_platforms+="${SOC_ID_TO_NAME[$soc_id]} ($soc_id), "
    done
    supported_platforms=${supported_platforms%, }  # Remove trailing comma
    
    log_skip "Not a supported NVIDIA platform. Supported: $supported_platforms"
}

check_tpm_config() {
    local kernel_version
    kernel_version=$(uname -r)
    local config_file="/boot/config-${kernel_version}"

    if [ ! -f "$config_file" ]; then
        log_fail "Kernel config file not found: $config_file"
    fi

    local config_value
    config_value=$(grep "^CONFIG_TCG_TIS_SPI=" "$config_file" || echo "not_found")

    if [ "$config_value" = "not_found" ]; then
        log_fail "CONFIG_TCG_TIS_SPI not found in kernel config"
    elif [ "$config_value" = "CONFIG_TCG_TIS_SPI=y" ]; then
        log_info "CONFIG_TCG_TIS_SPI=y (built-in)"
    elif [ "$config_value" = "CONFIG_TCG_TIS_SPI=m" ]; then
        log_info "CONFIG_TCG_TIS_SPI=m (module)"
        # Load the module if not already loaded
        if ! lsmod | grep -q "^tpm_tis_spi"; then
            if modprobe tpm_tis_spi 2>/dev/null; then
                MODULE_LOADED_BY_US=1
            fi
        fi
    else
        log_fail "CONFIG_TCG_TIS_SPI has unexpected value: $config_value"
    fi
}

unload_tpm_module() {
    if [ "$MODULE_LOADED_BY_US" -eq 1 ]; then
        log_info "Unloading tpm_tis_spi module (loaded by test)..."
        rmmod tpm_tis_spi 2>/dev/null || log_warn "Failed to unload tpm_tis_spi module"
    fi
}

check_tpm_devices() {
    # Check for expected devices
    if [ ! -c "/dev/tpm0" ]; then
        log_skip "No TPM devices found in /dev/ (TPM not present or not enabled in BIOS)"
    fi

    if [ ! -c "/dev/tpmrm0" ]; then
        log_warn "/dev/tpmrm0 device not found (optional)"
    fi

    log_info "Found TPM devices:"
    ls -1 /dev/tpm* 2>/dev/null

    log_info "TPM devices verified"
}

check_tpm_tools() {
    if ! command -v tpm2_pcrread &>/dev/null; then
        log_fail "tpm2-tools not found. Please install tpm2-tools package."
    fi
}

detect_hash_algorithm() {
    # Detect which hash algorithm the TPM supports
    local algos
    algos=$(tpm2_getcap algorithms 2>/dev/null) || log_fail "Failed to get TPM algorithms"

    if echo "$algos" | grep -qw "sha384"; then
        HASH_ALG="sha384"
        log_info "Using SHA384 hash algorithm"
    elif echo "$algos" | grep -qw "sha256"; then
        HASH_ALG="sha256"
        log_info "Using SHA256 hash algorithm (SHA384 not supported)"
    else
        log_fail "TPM does not support SHA256 or SHA384"
    fi
}

test_tpm_access() {
    local pcr_output
    pcr_output=$(tpm2_pcrread 2>&1) || log_fail "tpm2_pcrread failed"

    echo "$pcr_output"

    # Verify that we got hash algorithm output
    if echo "$pcr_output" | grep -q "sha256:"; then
        log_info "TPM supports SHA256"
    fi

    if echo "$pcr_output" | grep -q "sha384:"; then
        log_info "TPM supports SHA384"
    fi

    log_info "TPM device access verified"
}

cleanup_tpm_files() {
    log_info "Cleaning up TPM test files..."
    rm -f /tmp/primary.ctx /tmp/message.dat /tmp/secret.dat /tmp/rsa.pub /tmp/rsa.priv /tmp/rsa.ctx
    rm -f /tmp/sig.rssa /tmp/msg.enc /tmp/msg.ptext /tmp/seal.dat /tmp/pcr.bin /tmp/pcr.policy
    rm -f /tmp/seal.pub /tmp/seal.priv /tmp/seal.ctx /tmp/unsealed.dat /tmp/ecc.pub /tmp/ecc.priv /tmp/ecc.ctx
    rm -f /tmp/sig.ecc /tmp/hmac.key
}

run_qspi_tpm_tests() {
    # Exact content from qspi-tpm-test.sh
    #    clear tpm
    tpm2_clear
    #    test.tpm20_start()
    tpm2_startup -c

    #    test.tpm20_selftest()
    #TP-TH500-APT-L0-2: TPM2_SelfTest
    tpm2_selftest

    #    test.tpm20_getrandom()
    #TP-TH500-APT-L0-1: TPM2_GetRandom
    tpm2_getrandom 32

    #    test.tpm20_capabilities()
    #TP-TH500-APT-L0-3: TPM2_GetCapability - capabilities
    tpm2_getcap algorithms
    #TP-TH500-APT-L0-4: TPM2_GetCapability - commands
    tpm2_getcap commands
    #TP-TH500-APT-L0-5: TPM2_GetCapability - pcrs
    tpm2_getcap pcrs

    #    test.tpm20_rsa()
    tpm2_createprimary -c /tmp/primary.ctx
    echo "data to be signed" > /tmp/message.dat
    echo "Secret data" > /tmp/secret.dat
    #TP-TH500-APT-L0-8: RSA key generate
    tpm2_create -C /tmp/primary.ctx -Grsa2048 -u /tmp/rsa.pub -r /tmp/rsa.priv
    #TP-TH500-APT-L0-10: RSA key load
    tpm2_load -C /tmp/primary.ctx -u /tmp/rsa.pub -r /tmp/rsa.priv -c /tmp/rsa.ctx
    #TP-TH500-APT-L0-25: RSA sign with PKCS#1 v1.5 padding
    tpm2_sign -c /tmp/rsa.ctx -g "$HASH_ALG" -o /tmp/sig.rssa /tmp/message.dat -s rsassa
    #TP-TH500-APT-L0-22: RSA verify with with PKCS#1 v1.5 padding
    tpm2_verifysignature -c /tmp/rsa.ctx -g "$HASH_ALG" -s /tmp/sig.rssa -m /tmp/message.dat
    #TP-TH500-APT-L0-26: RSA sign with PSS padding
    tpm2_sign -c /tmp/rsa.ctx -g "$HASH_ALG" -o /tmp/sig.rssa /tmp/message.dat -s rsapss
    #TP-TH500-APT-L0-23: RSA verify with with PSS padding
    tpm2_verifysignature -c /tmp/rsa.ctx -g "$HASH_ALG" -s /tmp/sig.rssa -m /tmp/message.dat
    #TP-TH500-APT-L0-16: RSA encrypt
    tpm2_rsaencrypt -c /tmp/rsa.ctx -o /tmp/msg.enc /tmp/secret.dat
    tpm2_rsadecrypt -c /tmp/rsa.ctx -o /tmp/msg.ptext /tmp/msg.enc
    # Verify encrypt and decrypt message match
    if ! diff /tmp/secret.dat /tmp/msg.ptext; then
        log_fail "Decrypted message does not match original"
    fi
    log_info "Encrypt and decrypt message match verified"

    #    test.tpm20_seal_unseal()
    # TPM 2.0 seal and unseal tests
    tpm2_createprimary -c /tmp/primary.ctx
    echo "Data to be sealed" > /tmp/seal.dat
    tpm2_pcrreset 16
    
    # Select PCR extend value based on hash algorithm (for seal/unseal test)
    local pcr_extend_value
    case "$HASH_ALG" in
        sha384)
            pcr_extend_value="$PCR16_SHA384_SEAL_EXTEND"
            ;;
        sha256)
            pcr_extend_value="$PCR16_SHA256_SEAL_EXTEND"
            ;;
        *)
            log_fail "Unsupported hash algorithm: $HASH_ALG"
            ;;
    esac
    
    tpm2_pcrextend "16:${HASH_ALG}=${pcr_extend_value}"
    tpm2_pcrread -Q -o /tmp/pcr.bin "${HASH_ALG}:16"
    tpm2_createpolicy -Q --policy-pcr -l "${HASH_ALG}:16" -f /tmp/pcr.bin -L /tmp/pcr.policy
    #TP-TH500-APT-L0-12: TPM seal data
    cat /tmp/seal.dat | tpm2_create -C /tmp/primary.ctx -L /tmp/pcr.policy -i- -u /tmp/seal.pub -r /tmp/seal.priv -c /tmp/seal.ctx -Q
    tpm2_unseal -c /tmp/seal.ctx -p "pcr:${HASH_ALG}:16" > /tmp/unsealed.dat 2>/dev/null
    # Verify seal and unseal message match
    if ! diff /tmp/seal.dat /tmp/unsealed.dat; then
        log_fail "Unsealed message does not match original"
    fi
    log_info "Seal and unseal message match verified"

    #    test.tpm20_ecc()
    #TP-TH500-APT-L0-9 ECC (P256) key generate
    tpm2_create -C /tmp/primary.ctx -Gecc -u /tmp/ecc.pub -r /tmp/ecc.priv
    #TP-TH500-APT-L0-11: ECC (P256) key load
    tpm2_load -C /tmp/primary.ctx -u /tmp/ecc.pub -r /tmp/ecc.priv -c /tmp/ecc.ctx
    #TP-TH500-APT-L0-27: ECDSA sign
    tpm2_sign -c /tmp/ecc.ctx -g "$HASH_ALG" -o /tmp/sig.ecc /tmp/message.dat -s ecdsa
    #TP-TH500-APT-L0-24: ECDSA verify
    tpm2_verifysignature -c /tmp/ecc.ctx -g "$HASH_ALG" -s /tmp/sig.ecc -m /tmp/message.dat

    #    test.tpm20_hmac()
    tpm2_createprimary -c /tmp/primary.ctx
    tpm2_create -C /tmp/primary.ctx -Ghmac -c /tmp/hmac.key
    #TP-TH500-APT-L0-19: Generate HMAC
    tpm2_hmac -c /tmp/hmac.key --hex /tmp/message.dat

    #    test.tpm20_pcr()
    #TP-TH500-APT-L0-43/32: PCR16 reset
    tpm2_pcrreset 16
    pcr_value=$(tpm2_pcrread "${HASH_ALG}:16" 2>/dev/null | grep -A1 "${HASH_ALG}:" | tail -1 | awk '{print $2}')
    
    # Select expected reset value based on hash algorithm
    local expected_reset
    case "$HASH_ALG" in
        sha384)
            expected_reset="$PCR16_SHA384_RESET"
            ;;
        sha256)
            expected_reset="$PCR16_SHA256_RESET"
            ;;
        *)
            log_fail "Unsupported hash algorithm: $HASH_ALG"
            ;;
    esac
    
    if [ "$pcr_value" != "$expected_reset" ]; then
        log_fail "PCR16 after reset is $pcr_value, expected $expected_reset"
    fi
    log_info "PCR16 reset value verified: $pcr_value"

    # Select PCR extend value and expected result based on hash algorithm
    local pcr_test_extend expected_extended
    case "$HASH_ALG" in
        sha384)
            pcr_test_extend="$PCR16_SHA384_TEST_EXTEND"
            expected_extended="$PCR16_SHA384_TEST_EXTENDED"
            ;;
        sha256)
            pcr_test_extend="$PCR16_SHA256_TEST_EXTEND"
            expected_extended="$PCR16_SHA256_TEST_EXTENDED"
            ;;
        *)
            log_fail "Unsupported hash algorithm: $HASH_ALG"
            ;;
    esac
    
    tpm2_pcrextend "16:${HASH_ALG}=${pcr_test_extend}"
    pcr_value=$(tpm2_pcrread "${HASH_ALG}:16" 2>/dev/null | grep -A1 "${HASH_ALG}:" | tail -1 | awk '{print $2}')
    if [ "$pcr_value" != "$expected_extended" ]; then
        log_fail "PCR16 after extend is $pcr_value, expected $expected_extended"
    fi
    log_info "PCR16 extended value verified: $pcr_value"
}

cleanup_on_exit() {
    cleanup_tpm_files
    unload_tpm_module
}

main() {
    trap cleanup_on_exit EXIT INT

    check_nvidia_platform
    check_tpm_config
    check_tpm_devices
    check_tpm_tools
    detect_hash_algorithm
    test_tpm_access
    run_qspi_tpm_tests

    log_info "=========================================="
    log_pass "All TPM tests passed successfully"
}

main "$@"
