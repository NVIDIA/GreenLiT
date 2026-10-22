# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Portable tests of shared TPM identity and capability-response helpers."""

from pathlib import Path
from unittest import mock
import uuid

import pytest
import yaml

from common.tpm import ARM_FFA_TPM_UUID
from common.tpm import runtime as tpm_runtime

GOOD_PROPERTIES = 'TPM2_PT_FAMILY_INDICATOR: {raw: 0x322E3000, value: "2.0"}'


def _endpoint(paths, name, partition_id="0x8123", service_uuid=ARM_FFA_TPM_UUID, driver="ffa-crb"):
    """Create an FF-A service with an ordinary temporary driver symlink."""
    endpoint = paths["ffa"] / name
    endpoint.mkdir()
    (endpoint / "uuid").write_text(f"{service_uuid}\n", encoding="ascii")
    (endpoint / "partition_id").write_text(f"{partition_id}\n", encoding="ascii")
    driver_path = paths["drivers"] / driver
    driver_path.mkdir(exist_ok=True)
    (endpoint / "driver").symlink_to(driver_path, target_is_directory=True)
    return endpoint


def _tpm(paths, name, hardware_id="MSFT0101", version="2", driver="tpm_crb"):
    """Create a class entry and a fake resource node that is never opened."""
    tpm = paths["tpm"] / name
    device = tpm / "device"
    device.mkdir(parents=True)
    (device / "hid").write_text(f"{hardware_id}\n", encoding="ascii")
    (tpm / "tpm_version_major").write_text(f"{version}\n", encoding="ascii")
    driver_path = paths["drivers"] / driver
    driver_path.mkdir(exist_ok=True)
    (device / "driver").symlink_to(driver_path, target_is_directory=True)
    resource = paths["dev"] / f"tpmrm{name[3:]}"
    resource.touch()
    paths["characters"].add(resource)
    return tpm, resource


@pytest.fixture(name="module_loader")
def module_loader_fixture(tmp_path, monkeypatch):
    """Mock configuration and every module-management command without side effects."""
    loader = {
        "config": tmp_path / "running-kernel-config",
        "reader": mock.Mock(return_value={}),
        "logger": mock.Mock(),
        "available": mock.Mock(return_value=True),
        "command": mock.Mock(),
    }
    monkeypatch.setattr(tpm_runtime.check_kernel_config, "parse_kconfig", loader["reader"])
    monkeypatch.setattr(tpm_runtime.glt_utils, "is_cli_available", loader["available"])
    monkeypatch.setattr(tpm_runtime.glt_utils, "run_command", loader["command"])
    return loader


@pytest.fixture(name="runtime_paths")
def runtime_paths_fixture(tmp_path, monkeypatch):
    """Supply caller-selected paths and virtual character-device metadata only."""
    paths = {name: tmp_path / name for name in ("ffa", "tpm", "dev", "drivers")}
    for path in paths.values():
        path.mkdir()
    paths["characters"] = set()
    monkeypatch.setattr(
        Path, "is_char_device",
        lambda path: path in paths["characters"] and path.exists(),
    )
    return paths


def test_endpoint_uses_architected_uuid_partition_and_directory(runtime_paths):
    """Neither another UUID nor the right UUID on another partition can match."""
    _endpoint(runtime_paths, "arm-ffa-1", service_uuid=uuid.UUID(int=1), driver="unrelated")
    _endpoint(runtime_paths, "arm-ffa-2", partition_id="0x8124", driver="unrelated")
    expected = _endpoint(runtime_paths, "arm-ffa-3")
    assert tpm_runtime.find_ffa_tpm_endpoint(0x8123, runtime_paths["ffa"]) == expected


def test_endpoint_ignores_nonmatching_service_attributes(runtime_paths):
    """Other service UUIDs do not need the TPM partition or binding attributes."""
    other = _endpoint(runtime_paths, "arm-ffa-1", service_uuid=uuid.UUID(int=1))
    (other / "partition_id").unlink()
    (other / "driver").unlink()
    expected = _endpoint(runtime_paths, "arm-ffa-2")
    assert tpm_runtime.find_ffa_tpm_endpoint(0x8123, runtime_paths["ffa"]) == expected


@pytest.mark.parametrize("attribute,value", [
    ("uuid", str(uuid.UUID(int=1))),
    ("partition_id", "0x8124"),
])
def test_endpoint_identity_mismatch_has_no_match(runtime_paths, attribute, value):
    """The architected TPM service and the selected partition must both match."""
    endpoint = _endpoint(runtime_paths, "arm-ffa-1")
    (endpoint / attribute).write_text(value, encoding="ascii")
    with pytest.raises(ValueError, match="No architected TPM.*0x8123"):
        tpm_runtime.find_ffa_tpm_endpoint(0x8123, runtime_paths["ffa"])


@pytest.mark.parametrize("partition_id", ["8123", "0x81230", "-1"])
def test_endpoint_rejects_malformed_partition_id(runtime_paths, partition_id):
    """Malformed advertised IDs remain errors rather than absent matches."""
    _endpoint(runtime_paths, "arm-ffa-1", partition_id=partition_id)
    with pytest.raises(ValueError, match="Invalid FF-A partition ID"):
        tpm_runtime.find_ffa_tpm_endpoint(0x8123, runtime_paths["ffa"])


def test_endpoint_rejects_malformed_uuid(runtime_paths):
    """Malformed service identity propagates as an ordinary parsing error."""
    endpoint = _endpoint(runtime_paths, "arm-ffa-1")
    (endpoint / "uuid").write_text("not-a-uuid", encoding="ascii")
    with pytest.raises(ValueError, match="UUID"):
        tpm_runtime.find_ffa_tpm_endpoint(0x8123, runtime_paths["ffa"])


def test_endpoint_rejects_wrong_driver(runtime_paths):
    """A matching service with another driver cannot substitute for ffa-crb."""
    _endpoint(runtime_paths, "arm-ffa-1", driver="another-client")
    with pytest.raises(ValueError, match="binds 'another-client', expected 'ffa-crb'"):
        tpm_runtime.find_ffa_tpm_endpoint(0x8123, runtime_paths["ffa"])


@pytest.mark.parametrize("attribute", ["uuid", "partition_id", "driver"])
def test_endpoint_preserves_missing_attribute_error(runtime_paths, attribute):
    """Missing runtime identity and binding attributes remain distinguishable."""
    endpoint = _endpoint(runtime_paths, "arm-ffa-1")
    (endpoint / attribute).unlink()
    with pytest.raises(FileNotFoundError):
        tpm_runtime.find_ffa_tpm_endpoint(0x8123, runtime_paths["ffa"])


def test_endpoint_preserves_missing_directory_error(tmp_path):
    """An unavailable caller-selected bus is not rewritten as an empty bus."""
    with pytest.raises(FileNotFoundError):
        tpm_runtime.find_ffa_tpm_endpoint(0x8123, tmp_path / "missing")


def test_resource_managers_filter_and_sort_caller_paths(runtime_paths):
    """Return every ACPI TPM2 CRB resource, excluding other interfaces and names."""
    _, resource_two = _tpm(runtime_paths, "tpm2")
    _, resource_ten = _tpm(runtime_paths, "tpm10")
    _tpm(runtime_paths, "tpm1", hardware_id="PNP0C31", version="1", driver="tpm_tis")
    (runtime_paths["tpm"] / "tpmrm0").mkdir()
    (runtime_paths["tpm"] / "unrelated").mkdir()
    unidentified = runtime_paths["tpm"] / "tpm3" / "device"
    unidentified.mkdir(parents=True)
    assert tpm_runtime.find_tpm_resource_managers(runtime_paths["tpm"], runtime_paths["dev"]) == [
        resource_ten, resource_two,
    ]


def test_resource_manager_accepts_firmware_node_hid(runtime_paths):
    """The alternate ACPI firmware-node layout resolves the same resource."""
    tpm, resource = _tpm(runtime_paths, "tpm4")
    (tpm / "device" / "hid").unlink()
    firmware = tpm / "device" / "firmware_node"
    firmware.mkdir()
    (firmware / "hid").write_text("MSFT0101\n", encoding="ascii")
    assert tpm_runtime.find_tpm_resource_managers(runtime_paths["tpm"], runtime_paths["dev"]) == [resource]


def test_resource_manager_prefers_direct_hid(runtime_paths):
    """A different direct ACPI ID cannot be overridden by a fallback ID."""
    tpm, _ = _tpm(runtime_paths, "tpm0", hardware_id="PNP0C31")
    firmware = tpm / "device" / "firmware_node"
    firmware.mkdir()
    (firmware / "hid").write_text("MSFT0101", encoding="ascii")
    with pytest.raises(ValueError, match="no MSFT0101 TPM2 CRB device"):
        tpm_runtime.find_tpm_resource_managers(runtime_paths["tpm"], runtime_paths["dev"])


@pytest.mark.parametrize("attribute,value,reason", [
    ("tpm_version_major", "1", "expected TPM2"),
    ("driver", "tpm_tis", "expected 'tpm_crb'"),
])
def test_resource_manager_rejects_wrong_acpi_interface(runtime_paths, attribute, value, reason):
    """Another healthy TPM must not mask a broken advertised ACPI TPM entry."""
    _tpm(runtime_paths, "tpm0")
    tpm, _ = _tpm(runtime_paths, "tpm1")
    if attribute == "driver":
        link = tpm / "device" / "driver"
        link.unlink()
        driver = runtime_paths["drivers"] / value
        driver.mkdir()
        link.symlink_to(driver, target_is_directory=True)
    else:
        (tpm / attribute).write_text(value, encoding="ascii")
    with pytest.raises(ValueError, match=reason):
        tpm_runtime.find_tpm_resource_managers(runtime_paths["tpm"], runtime_paths["dev"])


@pytest.mark.parametrize("attribute", ["tpm_version_major", "device/driver"])
def test_resource_manager_preserves_missing_attribute_error(runtime_paths, attribute):
    """Incomplete advertised ACPI devices preserve the underlying I/O error."""
    tpm, _ = _tpm(runtime_paths, "tpm0")
    (tpm / attribute).unlink()
    with pytest.raises(FileNotFoundError):
        tpm_runtime.find_tpm_resource_managers(runtime_paths["tpm"], runtime_paths["dev"])


@pytest.mark.parametrize("missing", [False, True])
def test_resource_manager_requires_character_node(runtime_paths, missing):
    """Neither an ordinary file nor an absent node can be a resource manager."""
    _, resource = _tpm(runtime_paths, "tpm5")
    if missing:
        resource.unlink()
    else:
        runtime_paths["characters"].remove(resource)
    with pytest.raises(ValueError, match="resource manager character device.*tpmrm5.*missing"):
        tpm_runtime.find_tpm_resource_managers(runtime_paths["tpm"], runtime_paths["dev"])


def test_resource_manager_preserves_missing_class_error(tmp_path):
    """Callers receive the missing-class error rather than a false device list."""
    with pytest.raises(FileNotFoundError):
        tpm_runtime.find_tpm_resource_managers(tmp_path / "missing", tmp_path / "devices")


def test_modular_drivers_configured_modules_use_order_and_timeout(module_loader):
    """Both configured modules load in dependency order using the caller timeout."""
    module_loader["reader"].return_value = {"CONFIG_TCG_ARM_CRB_FFA": "m", "CONFIG_TCG_CRB": "m"}
    tpm_runtime.load_modular_tpm_drivers(
        module_loader["logger"], module_loader["config"], 17,
    )
    module_loader["reader"].assert_called_once_with(module_loader["config"])
    assert module_loader["command"].call_args_list == [
        mock.call(
            ["modprobe", "tpm_crb_ffa"],
            "Failed to load TPM module tpm_crb_ffa within 17 seconds", timeout=17,
        ),
        mock.call(
            ["modprobe", "tpm_crb"],
            "Failed to load TPM module tpm_crb within 17 seconds", timeout=17,
        ),
    ]
    assert module_loader["available"].call_args_list == [mock.call("modprobe"), mock.call("modprobe")]
    module_loader["logger"].info.assert_not_called()


@pytest.mark.parametrize("failure_index", [0, 1])
def test_modular_drivers_command_error_propagates(module_loader, failure_index):
    """A failed load stops the sequence and preserves the command helper error."""
    module_loader["reader"].return_value = {"CONFIG_TCG_ARM_CRB_FFA": "m", "CONFIG_TCG_CRB": "m"}
    error = RuntimeError("controlled module-load failure")
    module_loader["command"].side_effect = [None] * failure_index + [error]
    with pytest.raises(RuntimeError) as raised:
        tpm_runtime.load_modular_tpm_drivers(
            module_loader["logger"], module_loader["config"], 9,
        )
    assert raised.value is error
    assert module_loader["command"].call_count == failure_index + 1


@pytest.mark.parametrize("config,module", [
    ({"CONFIG_TCG_ARM_CRB_FFA": "m", "CONFIG_TCG_CRB": "y"}, "tpm_crb_ffa"),
    ({"CONFIG_TCG_ARM_CRB_FFA": "y", "CONFIG_TCG_CRB": "m"}, "tpm_crb"),
])
def test_modular_drivers_load_only_modular_entry(module_loader, config, module):
    """A built-in peer does not cause an extra modprobe invocation."""
    module_loader["reader"].return_value = config
    tpm_runtime.load_modular_tpm_drivers(
        module_loader["logger"], module_loader["config"], 11,
    )
    module_loader["command"].assert_called_once_with(
        ["modprobe", module], f"Failed to load TPM module {module} within 11 seconds", timeout=11,
    )
    module_loader["available"].assert_called_once_with("modprobe")


def test_modular_drivers_missing_modprobe_is_error(module_loader):
    """A configured modular driver cannot fall back when its loader is unavailable."""
    module_loader["reader"].return_value = {"CONFIG_TCG_ARM_CRB_FFA": "m"}
    module_loader["available"].return_value = False
    with pytest.raises(RuntimeError, match="modprobe is required.*tpm_crb_ffa"):
        tpm_runtime.load_modular_tpm_drivers(
            module_loader["logger"], module_loader["config"], 13,
        )
    module_loader["command"].assert_not_called()


@pytest.mark.parametrize("error", [FileNotFoundError("absent config"), PermissionError("unreadable config")])
def test_modular_drivers_missing_config_uses_runtime_fallback(module_loader, error):
    """Configuration I/O errors log the binding fallback and run no commands."""
    module_loader["reader"].side_effect = error
    tpm_runtime.load_modular_tpm_drivers(
        module_loader["logger"], module_loader["config"], 19,
    )
    module_loader["reader"].assert_called_once_with(module_loader["config"])
    module_loader["logger"].info.assert_called_once_with(
        "Cannot read %s (%s); validating already-bound TPM drivers", module_loader["config"], error,
    )
    module_loader["available"].assert_not_called()
    module_loader["command"].assert_not_called()


@pytest.mark.parametrize("config", [{}, {"CONFIG_TCG_ARM_CRB_FFA": "y", "CONFIG_TCG_CRB": "y"}])
def test_modular_drivers_without_modules_need_no_modprobe(module_loader, config):
    """Absent or built-in options require neither command execution nor a loader."""
    module_loader["reader"].return_value = config
    tpm_runtime.load_modular_tpm_drivers(
        module_loader["logger"], module_loader["config"], 7,
    )
    module_loader["available"].assert_not_called()
    module_loader["command"].assert_not_called()


def test_properties_accepts_complete_tpm2_family():
    """A consistent capability response validates without a GreenLiT result call."""
    assert tpm_runtime.validate_tpm_properties(GOOD_PROPERTIES) is None


def test_properties_wraps_yaml_error_as_value_error():
    """Malformed output has a useful ordinary error while retaining its cause."""
    with pytest.raises(ValueError, match="malformed YAML") as error:
        tpm_runtime.validate_tpm_properties("[unterminated")
    assert isinstance(error.value.__cause__, yaml.YAMLError)


@pytest.mark.parametrize("output", ["", "[]", "unexpected scalar", "23"])
def test_properties_requires_mapping(output):
    """Scalar, sequence and empty YAML are not capability mappings."""
    with pytest.raises(ValueError, match="did not return a properties mapping"):
        tpm_runtime.validate_tpm_properties(output)


@pytest.mark.parametrize("family", [
    "{}",
    "{TPM2_PT_FAMILY_INDICATOR: invalid}",
    'TPM2_PT_FAMILY_INDICATOR: {raw: 0x322E3000, value: "1.2"}',
    "TPM2_PT_FAMILY_INDICATOR: {raw: 0x322E3000, value: 2.0}",
])
def test_properties_requires_tpm2_family_string(family):
    """The required family must be a mapping with the explicit string '2.0'."""
    with pytest.raises(ValueError, match="TPM2_PT_FAMILY_INDICATOR value '2.0'"):
        tpm_runtime.validate_tpm_properties(family)


@pytest.mark.parametrize("raw", ["0x312E3200", '"0x322E3000"', "null"])
def test_properties_requires_consistent_numeric_raw(raw):
    """A family label cannot mask an absent, stringified or contradictory raw ID."""
    output = f'TPM2_PT_FAMILY_INDICATOR: {{raw: {raw}, value: "2.0"}}'
    with pytest.raises(ValueError, match="family raw value is inconsistent"):
        tpm_runtime.validate_tpm_properties(output)
