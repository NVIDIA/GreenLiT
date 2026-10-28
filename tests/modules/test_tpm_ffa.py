# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked coverage for TPM CRB over FF-A applicability and command results."""
# pylint: disable=protected-access

import subprocess
from unittest import mock

import pytest

from common import glt_core as GLTCore
from common.tpm import ARM_FFA_TPM_UUID
from src.modules.tpm import glt_test_tpm_ffa as tpm_ffa
from tests.common.tpm import make_tpm2_table

GOOD_PROPERTIES = 'TPM2_PT_FAMILY_INDICATOR:\n  raw: 0x322E3000\n  value: "2.0"\n'


@pytest.fixture(name="runtime")
def tpm_runtime(tmp_path, monkeypatch):
    """Create synthetic sysfs and mock every command; never touch a TPM device."""
    paths = {
        "table": tmp_path / "TPM2",
        "ffa": tmp_path / "arm_ffa",
        "tpm": tmp_path / "tpm",
        "dev": tmp_path / "dev",
    }
    paths["table"].write_bytes(make_tpm2_table())
    for name in ("ffa", "tpm", "dev"):
        paths[name].mkdir()
    endpoint = paths["ffa"] / "arm-ffa-0"
    endpoint.mkdir()
    (endpoint / "uuid").write_text(str(ARM_FFA_TPM_UUID), encoding="ascii")
    (endpoint / "partition_id").write_text("0x8001\n", encoding="ascii")
    ffa_driver = tmp_path / "ffa-crb"
    ffa_driver.mkdir()
    (endpoint / "driver").symlink_to(ffa_driver, target_is_directory=True)
    tpm = paths["tpm"] / "tpm0"
    device = tpm / "device"
    device.mkdir(parents=True)
    (device / "hid").write_text("MSFT0101\n", encoding="ascii")
    (tpm / "tpm_version_major").write_text("2\n", encoding="ascii")
    crb_driver = tmp_path / "tpm_crb"
    crb_driver.mkdir()
    (device / "driver").symlink_to(crb_driver, target_is_directory=True)
    # This character device is never opened: subprocess.run is mocked below.
    (paths["dev"] / "tpmrm0").symlink_to("/dev/null")
    monkeypatch.setattr(tpm_ffa, "TPM2_ACPI_TABLE", paths["table"])
    monkeypatch.setattr(tpm_ffa, "FFA_DEVICES", paths["ffa"])
    monkeypatch.setattr(tpm_ffa, "TPM_CLASS", paths["tpm"])
    monkeypatch.setattr(tpm_ffa, "DEVICE_DIRECTORY", paths["dev"])
    monkeypatch.setattr(tpm_ffa.platform, "machine", lambda: "aarch64")
    cli = mock.Mock(return_value=True)
    config = mock.Mock(return_value={"CONFIG_TCG_ARM_CRB_FFA": "y", "CONFIG_TCG_CRB": "y"})
    process = mock.Mock(return_value=subprocess.CompletedProcess([], 0, GOOD_PROPERTIES, ""))
    monkeypatch.setattr(tpm_ffa.glt_utils, "is_cli_available", cli)
    monkeypatch.setattr(tpm_ffa.tpm_runtime.check_kernel_config, "parse_kconfig", config)
    monkeypatch.setattr(tpm_ffa.glt_utils.subprocess, "run", process)
    return {**paths, "endpoint": endpoint, "device": device, "process": process,
            "cli": cli, "config": config, "logger": mock.Mock()}


def test_acpi_checksum_failure_is_not_skipped(runtime):
    """Malformed advertised ACPI must not hide a missing FF-A command path."""
    table = bytearray(make_tpm2_table())
    table[9] ^= 1
    runtime["table"].write_bytes(table)
    with pytest.raises(GLTCore.GLTTestFail, match="checksum"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


@pytest.mark.parametrize("table, reason", [
    (b"TPM2", "truncated"),
    (make_tpm2_table()[1:], "signature"),
    (make_tpm2_table()[:-1], "length"),
    (make_tpm2_table(length=52), "parameters"),
])
def test_acpi_malformed_fails(runtime, table, reason):
    """Reject bad signatures, declared lengths and short FF-A parameter blocks."""
    runtime["table"].write_bytes(table)
    with pytest.raises(GLTCore.GLTTestFail, match=reason):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


def test_acpi_missing_skips(runtime):
    """Firmware without any TPM2 table is outside applicability."""
    runtime["table"].unlink()
    with pytest.raises(GLTCore.GLTTestSkip, match="does not expose"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


def test_acpi_unreadable_fails(runtime, monkeypatch):
    """Permission errors cannot be treated as absent firmware support."""
    monkeypatch.setattr(type(runtime["table"]), "read_bytes", mock.Mock(side_effect=PermissionError("denied")))
    with pytest.raises(GLTCore.GLTTestFail, match="denied"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


@pytest.mark.parametrize("error", [
    subprocess.TimeoutExpired("tpm2_getcap", 30),
    subprocess.CalledProcessError(1, "tpm2_getcap", stderr="TPM failure"),
    OSError("cannot execute"),
    UnicodeDecodeError("ascii", b"\xff", 0, 1, "invalid"),
])
def test_command_errors_fail(runtime, error):
    """Timeouts, command errors and decode errors are runtime failures."""
    runtime["process"].side_effect = error
    with pytest.raises(GLTCore.GLTTestFail, match="TPM CRB FF-A command failed"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


@pytest.mark.parametrize("output", [
    "", "[]", "[unterminated", "TPM2_PT_FAMILY_INDICATOR: invalid",
    'TPM2_PT_FAMILY_INDICATOR: {raw: 0x312E3200, value: "1.2"}',
    'TPM2_PT_FAMILY_INDICATOR: {raw: 0x312E3200, value: "2.0"}',
    'TPM2_PT_MANUFACTURER: {raw: 0x41524D00, value: "ARM"}',
])
def test_command_malformed_results_fail(runtime, output):
    """Exit status zero alone cannot pass a missing or inconsistent TPM family."""
    runtime["process"].return_value.stdout = output
    with pytest.raises(GLTCore.GLTTestFail, match="TPM CRB FF-A command failed"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


def test_discrete_tpm_cannot_false_pass(runtime):
    """A TPM on another interface cannot stand in for the ACPI FF-A TPM."""
    (runtime["device"] / "hid").write_text("OTHER0001\n", encoding="ascii")
    with pytest.raises(GLTCore.GLTTestFail, match="no MSFT0101"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


@pytest.mark.parametrize("attribute, value, reason", [
    ("uuid", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "No architected TPM"),
    ("uuid", "invalid", "UUID"),
    ("partition_id", "0x8002", "partition ID 0x8001"),
    ("partition_id", "65536", "Invalid FF-A partition"),
])
def test_ffa_endpoint_identity_fails(runtime, attribute, value, reason):
    """Require a coherent UUID and the partition advertised by the ACPI table."""
    (runtime["endpoint"] / attribute).write_text(value, encoding="ascii")
    with pytest.raises(GLTCore.GLTTestFail, match=reason):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


@pytest.mark.parametrize("path", ["uuid", "partition_id", "driver"])
def test_ffa_endpoint_missing_attributes_fail(runtime, path):
    """An incomplete or unbound endpoint fails after firmware advertises FF-A."""
    (runtime["endpoint"] / path).unlink()
    with pytest.raises(GLTCore.GLTTestFail, match="Advertised TPM CRB FF-A"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


def test_firmware_node_hid_is_supported(runtime):
    """Kernel ACPI layouts can also expose the hardware ID through firmware_node."""
    (runtime["device"] / "hid").unlink()
    firmware_node = runtime["device"] / "firmware_node"
    firmware_node.mkdir()
    (firmware_node / "hid").write_text("MSFT0101\n", encoding="ascii")
    tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    assert runtime["process"].call_count == 1


def test_missing_cli_does_not_mask_unbound_device(runtime):
    """Validate advertised hardware before applying tool availability skips."""
    runtime["cli"].return_value = False
    (runtime["device"] / "driver").unlink()
    with pytest.raises(GLTCore.GLTTestFail, match="Advertised TPM CRB FF-A"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


def test_missing_resource_manager_fails(runtime):
    """A bound TPM without its kernel resource manager cannot pass the query."""
    (runtime["dev"] / "tpmrm0").unlink()
    with pytest.raises(GLTCore.GLTTestFail, match="resource manager character device"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


@pytest.mark.parametrize("directory", ["ffa", "tpm"])
def test_missing_runtime_bus_fails(runtime, monkeypatch, tmp_path, directory):
    """Advertised FF-A hardware cannot skip a missing transport or TPM class bus."""
    monkeypatch.setattr(tpm_ffa, "FFA_DEVICES" if directory == "ffa" else "TPM_CLASS", tmp_path / "missing")
    with pytest.raises(GLTCore.GLTTestFail, match="Advertised TPM CRB FF-A"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


def test_missing_tool_skips_after_valid_binding(runtime):
    """A missing userspace query utility is infrastructure outside the oracle."""
    runtime["cli"].return_value = False
    with pytest.raises(GLTCore.GLTTestSkip, match="Install tpm2-tools"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


def test_modprobe_error_fails(runtime):
    """A failed module load is detected before attempting a TPM capability query."""
    runtime["config"].return_value = {"CONFIG_TCG_ARM_CRB_FFA": "m"}
    runtime["process"].side_effect = subprocess.CalledProcessError(1, "modprobe")
    with pytest.raises(GLTCore.GLTTestFail, match="Failed to load TPM module"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    assert runtime["process"].call_count == 1


def test_modprobe_missing_fails_for_modular_driver(runtime):
    """Configured modular driver setup cannot silently pass when loading is impossible."""
    runtime["config"].return_value = {"CONFIG_TCG_ARM_CRB_FFA": "m"}
    runtime["cli"].return_value = False
    with pytest.raises(GLTCore.GLTTestFail, match="modprobe is required"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


def test_modular_drivers_load_before_command(runtime):
    """Modular kernels need both configured drivers before the runtime oracle."""
    runtime["config"].return_value = {"CONFIG_TCG_ARM_CRB_FFA": "m", "CONFIG_TCG_CRB": "m"}
    tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    assert [call.args[0] for call in runtime["process"].call_args_list[:2]] == [
        ["modprobe", "tpm_crb_ffa"], ["modprobe", "tpm_crb"],
    ]


@pytest.mark.parametrize("hardware_id, ambiguous", [("MSFT0101", True), ("OTHER0001", False)])
def test_multiple_tpm_devices_require_unique_eligible_resource(runtime, hardware_id, ambiguous):
    """Do not query ambiguous CRB resources or count unrelated TPM interfaces."""
    tpm = runtime["tpm"] / "tpm1"
    device = tpm / "device"
    device.mkdir(parents=True)
    (device / "hid").write_text(hardware_id, encoding="ascii")
    (tpm / "tpm_version_major").write_text("2", encoding="ascii")
    (device / "driver").symlink_to((runtime["device"] / "driver").resolve(), target_is_directory=True)
    (runtime["dev"] / "tpmrm1").symlink_to("/dev/zero")
    if ambiguous:
        with pytest.raises(GLTCore.GLTTestSkip, match="uniquely identify"):
            tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
        runtime["process"].assert_not_called()
    else:
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
        runtime["process"].assert_called_once()
        assert runtime["process"].call_args.args[0][2] == f"device:{runtime['dev'] / 'tpmrm0'}"


def test_no_boot_config_can_validate_bound_runtime(runtime):
    """A missing optional config file does not discard direct runtime evidence."""
    runtime["config"].side_effect = FileNotFoundError("no config")
    tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    assert runtime["process"].call_count == 1


@pytest.mark.parametrize("architecture", ["x86_64", "armv7l"])
def test_other_architecture_skips(runtime, monkeypatch, architecture):
    """Non-arm64 platforms do not touch TPM firmware or run a command."""
    monkeypatch.setattr(tpm_ffa.platform, "machine", lambda: architecture)
    with pytest.raises(GLTCore.GLTTestSkip, match="requires arm64"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["config"].assert_not_called()
    runtime["process"].assert_not_called()


def test_pass_uses_explicit_device_and_bounded_query(runtime):
    """The healthy path reads properties from the correlated kernel TPM device."""
    tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_called_once_with(
        ["tpm2_getcap", "-T", f"device:{runtime['dev'] / 'tpmrm0'}", "properties-fixed"],
        capture_output=True, text=True, check=True, timeout=30,
    )


def test_regular_file_is_not_resource_manager(runtime):
    """An arbitrary path named tpmrm0 is not sufficient device evidence."""
    resource = runtime["dev"] / "tpmrm0"
    resource.unlink()
    resource.write_text("not a device", encoding="ascii")
    with pytest.raises(GLTCore.GLTTestFail, match="character device"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


def test_uppercase_endpoint_uuid_passes(runtime):
    """UUID text casing does not change the architected service identity."""
    (runtime["endpoint"] / "uuid").write_text(str(ARM_FFA_TPM_UUID).upper(), encoding="ascii")
    tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])


@pytest.mark.parametrize("start_method", [2, 6, 7, 8, 11, 13])
def test_valid_other_start_method_skips(runtime, start_method):
    """A valid ACPI table for another TPM protocol is outside applicability."""
    runtime["table"].write_bytes(make_tpm2_table(start_method=start_method, length=52))
    with pytest.raises(GLTCore.GLTTestSkip, match="not CRB over Arm FF-A"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


@pytest.mark.parametrize("kind, expected", [("ffa", "ffa-crb"), ("tpm", "tpm_crb")])
def test_wrong_driver_fails(runtime, tmp_path, kind, expected):
    """Advertised devices require both the transport and CRB driver bindings."""
    device = runtime["endpoint"] if kind == "ffa" else runtime["device"]
    wrong_driver = tmp_path / "wrong-driver"
    wrong_driver.mkdir()
    (device / "driver").unlink()
    (device / "driver").symlink_to(wrong_driver, target_is_directory=True)
    with pytest.raises(GLTCore.GLTTestFail, match=f"expected '{expected}'"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()


def test_wrong_tpm_version_fails(runtime):
    """The ACPI TPM class device must expose version 2 before it is queried."""
    (runtime["tpm"] / "tpm0" / "tpm_version_major").write_text("1\n", encoding="ascii")
    with pytest.raises(GLTCore.GLTTestFail, match="expected TPM2"):
        tpm_ffa.test_tpm_crb_ffa_command_path(None, runtime["logger"])
    runtime["process"].assert_not_called()
