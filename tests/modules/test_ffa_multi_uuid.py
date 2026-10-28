# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Mocked public FF-A multi-UUID messaging, strict evidence and owned cleanup."""
# pylint: disable=protected-access

from pathlib import Path
import shutil
import stat
import subprocess
from types import SimpleNamespace
from unittest import mock
import uuid

import pytest

from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_ffa_multi_uuid as multi

TOKEN = "a" * 32
RELEASE = "unit-kernel-release"


def _output(profile, **updates):
    """Serialize fresh, address-free synthetic evidence without hardware values."""
    values = {
        "schema_version": "2", "evidence_kind": "running_kernel_ffa_multi_uuid", "run_id": TOKEN,
        "kernel_release": RELEASE, "device": profile.buffers_endpoint.name, "uuid": str(multi.BUFFERS_UUID),
        "partition_id": f"0x{profile.partition_id:04x}", "properties": "0x00000200", "mode_32bit": "0",
        "driver_bound": "0", "req2_supported": "1", "api_available": "1", "query_attempted": "1", "api_rc": "0",
        "response_received": "1", "service_status": "0x0", "base_nonzero": "1", "buffer_size": "4096",
        "minimum_size": "25", "stage": "complete",
    }
    values.update(updates)
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _parse(profile, **updates):
    """Parse one controlled result through the production strict parser."""
    return multi._parse_probe_result(_output(profile, **updates), TOKEN, RELEASE, profile)


def _prequery(**updates):
    """Describe a pre-query result with no response or interpreted metadata."""
    values = {
        "query_attempted": "0", "api_rc": "0", "response_received": "0",
        "service_status": "0x0", "base_nonzero": "0", "buffer_size": "0",
    }
    values.update(updates)
    return values


def _service(directory, name, service_uuid, partition_id="0x8123"):
    """Write only temporary sysfs identity attributes for profile discovery."""
    endpoint = directory / name
    endpoint.mkdir()
    (endpoint / "uuid").write_text(str(service_uuid), encoding="ascii")
    (endpoint / "partition_id").write_text(partition_id, encoding="ascii")
    return endpoint


@pytest.fixture(name="profile")
def profile_fixture(tmp_path):
    """Select synthetic paths without opening any TPM or kernel interface."""
    return multi._Profile(
        0x8123, tmp_path / "arm-ffa-1", tmp_path / "dev" / "tpmrm2", tmp_path / "arm-ffa-2",
    )


@pytest.fixture(name="discovery")
def discovery_fixture(tmp_path, monkeypatch, profile):
    """Mock the advertised TPM while exercising actual temporary Buffers lookup."""
    devices = tmp_path / "ffa-devices"
    devices.mkdir()
    endpoint = mock.Mock(return_value=profile.tpm_endpoint)
    resources = mock.Mock(return_value=[profile.resource])
    monkeypatch.setattr(multi, "FFA_DEVICES", devices)
    monkeypatch.setattr(multi.tpm_runtime, "find_ffa_tpm_endpoint", endpoint)
    monkeypatch.setattr(multi.tpm_runtime, "find_tpm_resource_managers", resources)
    return SimpleNamespace(devices=devices, endpoint=endpoint, resources=resources)


@pytest.fixture(name="sequence")
def sequence_fixture(monkeypatch, profile):
    """Separate retained-descriptor orchestration from temporary module builds."""
    events = []
    errors = {}
    identity = ("endpoint", "ffa-crb", (1, 2, 3))
    opened = mock.Mock(side_effect=lambda *_: events.append("open") or 37)
    closed = mock.Mock(side_effect=lambda *_: events.append("close"))
    identities = mock.Mock(side_effect=lambda *_: events.append("identity") or identity)

    def query(_resource, phase, _logger):
        events.append(phase)
        if phase in errors:
            raise errors[phase]

    result = ["PASS", "synthetic Buffers success"]

    def buffers(*_args):
        events.append("B")
        if "B" in errors:
            raise errors["B"]
        return tuple(result)

    queried = mock.Mock(side_effect=query)
    probe = mock.Mock(side_effect=buffers)
    monkeypatch.setattr(multi.os, "open", opened)
    monkeypatch.setattr(multi.os, "close", closed)
    monkeypatch.setattr(multi, "_tpm_identity", identities)
    monkeypatch.setattr(multi, "_query_tpm", queried)
    monkeypatch.setattr(multi, "_run_buffers_probe", probe)
    return SimpleNamespace(
        profile=profile, events=events, errors=errors, opened=opened, closed=closed,
        identities=identities, queried=queried, probe=probe, result=result,
    )


@pytest.fixture(name="probe_runtime")
def probe_runtime_fixture(tmp_path, monkeypatch, profile):
    """Copy real payload into temporary paths and mock all build/load commands."""
    headers_root = tmp_path / "installed-headers"
    headers = headers_root / RELEASE / "build"
    headers.mkdir(parents=True)
    (headers / "Makefile").touch()
    payload = tmp_path / "payload"
    payload.mkdir()
    for name in ("Makefile", multi.PROBE_MODULE + ".c"):
        shutil.copy2(multi.PROBE_PAYLOAD / name, payload / name)
    result = tmp_path / "owned-proc-result"
    module_state = tmp_path / "owned-module-state"
    state = SimpleNamespace(
        profile=profile, headers=headers, result=result, module_state=module_state, payload=payload,
        build_directories=[], release_directories=[], output=_output(profile),
        header_release=RELEASE, build_error=False,
        load_error=False, create_on_load_error=False, unload_error=False, remain_after_unload=False,
    )

    def command(argv, _failure, **_kwargs):
        if argv[-1] == "kernelrelease":
            directory = Path(next(arg[2:] for arg in argv if arg.startswith("M=")))
            assert directory.is_dir() and directory != headers
            assert (directory / "Makefile").is_file()
            state.release_directories.append(directory)
            return subprocess.CompletedProcess(argv, 0, state.header_release + "\n", "")
        if argv[0] == "make":
            directory = Path(next(arg[2:] for arg in argv if arg.startswith("M=")))
            state.build_directories.append(directory)
            assert (directory / "Makefile").read_bytes() == (payload / "Makefile").read_bytes()
            assert (directory / (multi.PROBE_MODULE + ".c")).read_bytes() == (
                payload / (multi.PROBE_MODULE + ".c")
            ).read_bytes()
            if state.build_error:
                raise RuntimeError("controlled build failure")
            (directory / (multi.PROBE_MODULE + ".ko")).touch()
        elif argv[0] == "insmod":
            if not state.load_error or state.create_on_load_error:
                module_state.mkdir()
                result.write_text(state.output, encoding="ascii")
            if state.load_error:
                raise RuntimeError("controlled uncertain load failure")
        elif argv[0] == "rmmod":
            if state.unload_error:
                raise RuntimeError("controlled unload failure")
            if not state.remain_after_unload:
                result.unlink()
                module_state.rmdir()
        else:
            raise AssertionError(f"Unexpected mocked command: {argv}")
        return subprocess.CompletedProcess(argv, 0, "", "")

    state.config = mock.Mock(return_value={"CONFIG_MODULES": "y", "CONFIG_MODULE_UNLOAD": "y"})
    monkeypatch.setattr(multi.glt_utils.check_kernel_config, "parse_kconfig", state.config)
    state.command = mock.Mock(side_effect=command)
    state.sign = mock.Mock(return_value=True)
    state.read = mock.Mock(wraps=multi.glt_utils.read_sysfs_value)
    monkeypatch.setattr(multi, "HEADERS_ROOT", headers_root)
    monkeypatch.setattr(multi, "PROBE_PAYLOAD", payload)
    monkeypatch.setattr(multi, "PROBE_RESULT", result)
    monkeypatch.setattr(multi, "PROBE_MODULE_STATE", module_state)
    monkeypatch.setattr(multi.platform, "release", lambda: RELEASE)
    monkeypatch.setattr(multi.uuid, "uuid4", lambda: SimpleNamespace(hex=TOKEN))
    monkeypatch.setattr(multi.glt_utils, "run_command", state.command)
    monkeypatch.setattr(multi.glt_utils, "sign_kernel_module", state.sign)
    monkeypatch.setattr(multi.glt_utils, "read_sysfs_value", state.read)
    return state


@pytest.fixture(name="native")
def native_fixture(monkeypatch, profile):
    """Mock every native applicability boundary and orchestration entry point."""
    state = SimpleNamespace(
        logger=mock.Mock(), acpi=mock.Mock(return_value=profile.partition_id),
        load=mock.Mock(), discover=mock.Mock(return_value=(profile, "")),
        prerequisite=mock.Mock(return_value=""), exercise=mock.Mock(return_value=("PASS", "success")),
        real_prerequisite=multi._probe_prerequisite_reason,
    )
    monkeypatch.setattr(multi.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(multi, "read_tpm2_ffa_partition", state.acpi)
    monkeypatch.setattr(multi.tpm_runtime, "load_modular_tpm_drivers", state.load)
    monkeypatch.setattr(multi, "_discover_profile", state.discover)
    monkeypatch.setattr(multi, "_probe_prerequisite_reason", state.prerequisite)
    monkeypatch.setattr(multi, "_exercise_profile", state.exercise)
    return state


def test_parser_accepts_exact_success_and_minimum(profile):
    """Complete eligible API/service success accepts the packed 25-byte minimum."""
    assert len(multi.RESULT_FIELDS) == 20
    assert _parse(profile)[0] == "PASS"
    assert _parse(profile, buffer_size="25")[0] == "PASS"


@pytest.mark.parametrize("damage", ["missing", "extra", "address", "duplicate", "separator"])
def test_parser_rejects_inexact_field_set(profile, damage):
    """Truncated, extended, address-bearing and ambiguous records cannot pass."""
    output = _output(profile)
    if damage == "missing":
        output = output.replace("base_nonzero=1\n", "")
    elif damage == "extra":
        output += "unexpected=1\n"
    elif damage == "address":
        output += "buffer_base=0xdeadbeef\n"
    elif damage == "duplicate":
        output += "run_id=" + TOKEN + "\n"
    else:
        output += "unstructured line\n"
    with pytest.raises(RuntimeError):
        multi._parse_probe_result(output, TOKEN, RELEASE, profile)


@pytest.mark.parametrize("key,value", [
    ("run_id", "b" * 32), ("kernel_release", "different-release"), ("device", "arm-ffa-99"),
    ("uuid", str(uuid.UUID(int=1))), ("partition_id", "0x8124"), ("schema_version", "1"),
    ("evidence_kind", "source-only"), ("minimum_size", "24"),
])
def test_parser_rejects_stale_or_other_identity(profile, key, value):
    """Run identity and the selected real service are mandatory evidence fields."""
    with pytest.raises(RuntimeError):
        _parse(profile, **{key: value})


@pytest.mark.parametrize("key", [
    "mode_32bit", "driver_bound", "req2_supported", "api_available", "query_attempted",
    "response_received", "base_nonzero",
])
def test_parser_rejects_nonbinary_flags(profile, key):
    """Text booleans are not accepted as kernel numeric execution flags."""
    with pytest.raises(RuntimeError, match="invalid"):
        _parse(profile, **{key: "true"})


@pytest.mark.parametrize("key,value", [
    ("properties", "0x200"), ("partition_id", "33059"), ("api_rc", "+0"),
    ("service_status", "0"), ("buffer_size", "-1"), ("uuid", "invalid"),
])
def test_parser_rejects_malformed_numeric_or_uuid_fields(profile, key, value):
    """Strict wire types prevent ambiguous status, size and device records."""
    with pytest.raises(RuntimeError, match="invalid"):
        _parse(profile, **{key: value})


@pytest.mark.parametrize("key,value", [
    ("mode_32bit", "1"), ("driver_bound", "1"), ("req2_supported", "0"),
    ("api_available", "0"), ("query_attempted", "0"), ("api_rc", "-95"),
    ("response_received", "0"), ("service_status", "0x1"), ("properties", "0x00000000"),
])
def test_parser_pass_requires_actual_eligible_request(profile, key, value):
    """A complete stage cannot replace eligibility and a received successful reply."""
    with pytest.raises(RuntimeError, match="eligibility|complete stage"):
        _parse(profile, **{key: value})


@pytest.mark.parametrize("stage", ["api_missing", "api_error", "service_error", "invalid_metadata"])
def test_parser_retains_semantic_failure_with_runtime_guidance(profile, stage):
    """Semantic failures retain their stage without attributing an admission patch."""
    updates = _prequery(stage="complete" if stage == "invalid_metadata" else stage)
    if stage == "api_missing":
        updates["api_available"] = "0"
    else:
        updates["query_attempted"] = "1"
        if stage == "api_error":
            updates["api_rc"] = "-5"
        else:
            updates["response_received"] = "1"
            if stage == "service_error":
                updates["service_status"] = "0x1"
    status, reason = _parse(profile, **updates)
    assert status == "FAIL"
    assert stage in reason
    assert "Linux FF-A driver/API support" in reason
    assert "firmware service protocol" in reason
    assert "kernel logs" in reason
    assert "be61da938576" not in reason


@pytest.mark.parametrize("stage", ["device_missing", "device_identity_changed", "device_bound", "req2_unsupported"])
def test_parser_prequery_skip_has_no_success_claim(profile, stage):
    """Unavailable profile cases skip only before attempting a firmware request."""
    updates = _prequery(stage=stage)
    if stage == "device_identity_changed":
        updates.update(uuid=str(uuid.UUID(int=1)), partition_id="0x8124")
    elif stage == "device_bound":
        updates["driver_bound"] = "1"
    elif stage == "req2_unsupported":
        updates.update(req2_supported="0", properties="0x00000000")
    status, reason = _parse(profile, **updates)
    assert status == "SKIP"
    assert stage in reason
    assert "inconclusive" in reason


@pytest.mark.parametrize("key,value", [
    ("query_attempted", "1"), ("api_rc", "-1"), ("response_received", "1"),
    ("service_status", "0x1"), ("base_nonzero", "1"), ("buffer_size", "1"),
])
def test_parser_skip_rejects_query_evidence(profile, key, value):
    """A performed request or metadata cannot be hidden behind pre-query SKIP."""
    updates = _prequery(stage="device_missing")
    updates[key] = value
    with pytest.raises(RuntimeError, match="unexpected query execution"):
        _parse(profile, **updates)


def test_parser_bound_skip_still_requires_selected_identity(profile):
    """Only disappearance and identity-change skips may report another identity."""
    with pytest.raises(RuntimeError, match="inconsistent service identity"):
        _parse(profile, **_prequery(stage="device_bound", partition_id="0x8124"))


@pytest.mark.parametrize("stage", ["unknown", "invalid_metadata"])
def test_parser_rejects_unknown_raw_stage(profile, stage):
    """Only documented raw execution stages can be interpreted by Python."""
    with pytest.raises(RuntimeError, match="invalid stage"):
        _parse(profile, stage=stage)


@pytest.mark.parametrize("status", ["PASS", "FAIL", "SKIP"])
def test_parser_rejects_kernel_verdict_metadata(profile, status):
    """The schema contains raw evidence, never a kernel-supplied GLT verdict."""
    with pytest.raises(RuntimeError, match="missing or unexpected"):
        _parse(profile, status=status)


@pytest.mark.parametrize("updates", [{"base_nonzero": "0"}, {"buffer_size": "24"}])
def test_parser_derives_metadata_failure_from_raw_reply(profile, updates):
    """A successful transport reply can still fail Python's metadata policy."""
    status, reason = _parse(profile, **updates)
    assert status == "FAIL" and "invalid_metadata" in reason


@pytest.mark.parametrize("stage,updates", [
    ("api_missing", {"query_attempted": "1"}),
    ("api_error", {"api_rc": "0"}),
    ("service_error", {"service_status": "0x0"}),
])
def test_parser_rejects_inconsistent_failure_stage(profile, stage, updates):
    """A failure stage must agree with the raw API and response evidence."""
    values = _prequery(stage=stage, api_available="0" if stage == "api_missing" else "1")
    if stage != "api_missing":
        values.update(query_attempted="1", api_rc="-5" if stage == "api_error" else "0")
    if stage == "service_error":
        values.update(response_received="1", service_status="0x1")
    values.update(updates)
    with pytest.raises(RuntimeError, match="inconsistent query evidence"):
        _parse(profile, **values)


def test_discovery_validates_advertised_tpm_before_missing_buffers(discovery):
    """A missing Buffers profile cannot mask a broken advertised TPM binding."""
    discovery.endpoint.side_effect = ValueError("advertised TPM driver is wrong")
    with pytest.raises(ValueError, match="advertised TPM driver"):
        multi._discover_profile(0x8123)
    discovery.resources.assert_not_called()


def test_discovery_rejects_zero_resource_managers(discovery):
    """Advertised TPM hardware with no resource-manager device is a failure."""
    error = ValueError("ACPI advertises TPM CRB over FF-A but no MSFT0101 TPM2 CRB device exists")
    discovery.resources.side_effect = error
    with pytest.raises(ValueError) as failure:
        multi._discover_profile(0x8123)
    assert failure.value is error


def test_discovery_multiple_resource_managers_skip_ambiguity(discovery, profile):
    """Do not choose an arbitrary TPM resource manager when several match."""
    discovery.resources.return_value = [profile.resource, profile.resource.with_name("tpmrm3")]
    selected, reason = multi._discover_profile(profile.partition_id)
    assert selected is None
    assert "uniquely" in reason


@pytest.mark.parametrize("case", ["missing", "wrong_partition", "bound", "dangling_driver", "multiple"])
def test_discovery_unavailable_buffers_profile_skips(discovery, case):
    """Missing or ambiguous Linux profiles cannot establish firmware service absence."""
    if case != "missing":
        endpoint = _service(
            discovery.devices, "arm-ffa-2", multi.BUFFERS_UUID,
            "0x8124" if case == "wrong_partition" else "0x8123",
        )
        if case == "bound":
            (endpoint / "driver").mkdir()
        elif case == "dangling_driver":
            (endpoint / "driver").symlink_to(discovery.devices / "absent-driver")
        elif case == "multiple":
            _service(discovery.devices, "arm-ffa-3", multi.BUFFERS_UUID)
    selected, reason = multi._discover_profile(0x8123)
    assert selected is None
    if case in ("bound", "dangling_driver"):
        assert "bound to another client" in reason
    else:
        assert "admission is inconclusive" in reason
        assert "firmware absence" in reason
        assert "Linux admission loss" in reason
        assert "bounded independent firmware inventory" in reason


def test_discovery_selects_same_partition_unbound_buffers(discovery, profile):
    """Preserve the genuine TPM and select only the public Buffers UUID."""
    expected = _service(discovery.devices, "arm-ffa-2", multi.BUFFERS_UUID)
    _service(discovery.devices, "arm-ffa-3", uuid.UUID(int=1))
    selected, reason = multi._discover_profile(profile.partition_id)
    assert selected == multi._Profile(profile.partition_id, profile.tpm_endpoint, profile.resource, expected)
    assert not reason


def test_discovery_uses_shared_sysfs_reader(discovery, monkeypatch):
    """Read the candidate's UUID and partition through the existing utility."""
    endpoint = _service(discovery.devices, "arm-ffa-2", multi.BUFFERS_UUID)
    reader = mock.Mock(wraps=multi.glt_utils.read_sysfs_value)
    monkeypatch.setattr(multi.glt_utils, "read_sysfs_value", reader)
    selected, _ = multi._discover_profile(0x8123)
    assert selected.buffers_endpoint == endpoint
    assert reader.call_args_list == [mock.call(endpoint / "uuid"), mock.call(endpoint / "partition_id")]


def test_sequence_retains_client_across_a1_b_a2(sequence):
    """A single real client descriptor spans the independent UUID requests."""
    assert multi._exercise_profile(sequence.profile, mock.Mock())[0] == "PASS"
    assert sequence.events == ["open", "identity", "A1", "B", "identity", "A2", "identity", "close"]
    sequence.opened.assert_called_once_with(
        sequence.profile.resource, multi.os.O_RDWR | multi.os.O_CLOEXEC | multi.os.O_NOFOLLOW,
    )
    sequence.closed.assert_called_once_with(37)


def test_sequence_a1_failure_never_queries_buffers(sequence):
    """A failed initial legitimate TPM query prevents the optional firmware call."""
    sequence.errors["A1"] = RuntimeError("A1 failed")
    with pytest.raises(RuntimeError, match="A1 failed"):
        multi._exercise_profile(sequence.profile, mock.Mock())
    assert sequence.events == ["open", "identity", "A1", "close"]
    sequence.probe.assert_not_called()


@pytest.mark.parametrize("status", ["FAIL", "SKIP"])
def test_sequence_buffers_verdict_still_runs_a2(sequence, status):
    """The recovery query runs even when Buffers reports a failure or missing profile."""
    sequence.result[:] = [status, "controlled B result"]
    assert multi._exercise_profile(sequence.profile, mock.Mock()) == (status, "controlled B result")
    assert sequence.events[-4:] == ["identity", "A2", "identity", "close"]


def test_sequence_buffers_exception_still_runs_a2(sequence):
    """A probe exception becomes FAIL while preserving the TPM recovery attempt."""
    sequence.errors["B"] = RuntimeError("B failed")
    status, reason = multi._exercise_profile(sequence.profile, mock.Mock())
    assert status == "FAIL" and "B failed" in reason
    assert sequence.events[-4:] == ["identity", "A2", "identity", "close"]


@pytest.mark.parametrize("prior", ["SKIP", "FAIL"])
def test_sequence_a2_failure_overrides_skip_and_retains_b_reason(sequence, prior):
    """TPM recovery failure has priority and retains the preceding probe evidence."""
    sequence.result[:] = [prior, "controlled B reason"]
    sequence.errors["A2"] = ValueError("recovery family invalid")
    status, reason = multi._exercise_profile(sequence.profile, mock.Mock())
    assert status == "FAIL"
    assert "controlled B reason" in reason and "recovery family invalid" in reason
    assert sequence.events[-1] == "close"


def test_sequence_identity_change_refuses_replacement_query(sequence):
    """Do not issue A2 to a new binding after the original identity changes."""
    sequence.identities.side_effect = [("original",), ("replacement",)]
    status, reason = multi._exercise_profile(sequence.profile, mock.Mock())
    assert status == "FAIL" and "identity changed" in reason
    assert [call.args[1] for call in sequence.queried.call_args_list] == ["A1"]
    sequence.closed.assert_called_once_with(37)


def test_sequence_identity_replacement_during_a2_fails(sequence):
    """A2 success is insufficient when the retained identity changes during it."""
    sequence.identities.side_effect = [("original",), ("original",), ("replacement",)]
    status, reason = multi._exercise_profile(sequence.profile, mock.Mock())
    assert status == "FAIL" and "second TPM query" in reason
    assert [call.args[1] for call in sequence.queried.call_args_list] == ["A1", "A2"]
    sequence.closed.assert_called_once_with(37)


@pytest.mark.parametrize("case", ["healthy", "retained_file", "current_file", "new_inode", "resource", "endpoint"])
def test_tpm_identity_requires_same_bound_character_device(profile, monkeypatch, case):
    """Both retained and current metadata must identify the selected character node."""
    profile.tpm_endpoint.mkdir()
    driver = profile.tpm_endpoint.parent / "ffa-crb"
    driver.mkdir()
    (profile.tpm_endpoint / "driver").symlink_to(driver, target_is_directory=True)
    profile.resource.parent.mkdir()
    profile.resource.touch()
    replacement = profile.tpm_endpoint.parent / "replacement-endpoint"
    replacement.mkdir()
    retained = SimpleNamespace(st_dev=1, st_ino=2, st_rdev=3, st_mode=stat.S_IFCHR)
    current = SimpleNamespace(st_dev=1, st_ino=2, st_rdev=3, st_mode=stat.S_IFCHR)
    if case == "retained_file":
        retained.st_mode = stat.S_IFREG
    elif case == "current_file":
        current.st_mode = stat.S_IFREG
    elif case == "new_inode":
        current.st_ino = 9
    endpoint = replacement if case == "endpoint" else profile.tpm_endpoint
    resources = [profile.resource.with_name("tpmrm9")] if case == "resource" else [profile.resource]
    monkeypatch.setattr(multi.tpm_runtime, "find_ffa_tpm_endpoint", mock.Mock(return_value=endpoint))
    monkeypatch.setattr(multi.tpm_runtime, "find_tpm_resource_managers", mock.Mock(return_value=resources))
    # Keep fake descriptor metadata local; process-wide fstat is used by coverage.
    monkeypatch.setattr(
        multi, "os", mock.Mock(wraps=multi.os, fstat=mock.Mock(return_value=retained)),
    )
    original_stat = Path.stat

    def controlled_stat(path, *args, **kwargs):
        if path == profile.resource:
            return current
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", controlled_stat)
    if case == "healthy":
        assert multi._tpm_identity(profile, 37) == (profile.tpm_endpoint, driver, (1, 2, 3))
    else:
        with pytest.raises(RuntimeError, match="identity changed|retained real character device"):
            multi._tpm_identity(profile, 37)


def test_query_tpm_uses_explicit_resource_timeout_and_validator(profile, monkeypatch):
    """Each A query uses the legitimate device TCTI and validates its returned text."""
    command = mock.Mock(return_value=SimpleNamespace(stdout="controlled capability response"))
    validator = mock.Mock()
    logger = mock.Mock()
    monkeypatch.setattr(multi.glt_utils, "run_command", command)
    monkeypatch.setattr(multi.tpm_runtime, "validate_tpm_properties", validator)
    multi._query_tpm(profile.resource, "A1", logger)
    command.assert_called_once_with(
        ["tpm2_getcap", "-T", f"device:{profile.resource}", "properties-fixed"],
        f"TPM capability query A1 failed or exceeded {multi.COMMAND_TIMEOUT} seconds",
        timeout=multi.COMMAND_TIMEOUT,
    )
    validator.assert_called_once_with("controlled capability response")


@pytest.mark.parametrize("case", ["pass", "semantic_fail", "malformed", "build", "load", "unload", "remaining"])
def test_probe_lifecycle_outcomes_and_temporary_cleanup(probe_runtime, case):
    """Owned resources unload on semantic/readback errors and temporary builds vanish."""
    if case == "semantic_fail":
        probe_runtime.output = _output(probe_runtime.profile, **_prequery(
            stage="api_error", query_attempted="1", api_rc="-5",
        ))
    elif case == "malformed":
        probe_runtime.output = "unstructured output\n"
    elif case == "build":
        probe_runtime.build_error = True
    elif case == "load":
        probe_runtime.load_error = True
    elif case == "unload":
        probe_runtime.unload_error = True
    elif case == "remaining":
        probe_runtime.remain_after_unload = True
    status, reason = multi._run_buffers_probe(probe_runtime.profile, mock.Mock())
    assert status == ("PASS" if case == "pass" else "FAIL")
    assert reason
    assert probe_runtime.build_directories
    assert all(not directory.exists() for directory in probe_runtime.build_directories)
    commands = [call.args[0][0] for call in probe_runtime.command.call_args_list]
    assert ("rmmod" in commands) == (case not in ("build", "load"))
    if case not in ("unload", "remaining"):
        assert not probe_runtime.result.exists()
        assert not probe_runtime.module_state.exists()
    else:
        assert "cleanup failed" in reason


@pytest.mark.parametrize("owned", [True, False])
def test_probe_uncertain_load_uses_nonce_before_cleanup(probe_runtime, owned):
    """A failed insmod permits cleanup only after matching this invocation's evidence."""
    probe_runtime.load_error = True
    probe_runtime.create_on_load_error = True
    if not owned:
        probe_runtime.output = _output(probe_runtime.profile, run_id="b" * 32)
    status, reason = multi._run_buffers_probe(probe_runtime.profile, mock.Mock())
    assert status == "FAIL"
    probe_runtime.read.assert_called_once_with(probe_runtime.result)
    commands = [call.args[0][0] for call in probe_runtime.command.call_args_list]
    assert ("rmmod" in commands) == owned
    assert probe_runtime.result.exists() is not owned
    assert probe_runtime.module_state.exists() is not owned
    assert all(not directory.exists() for directory in probe_runtime.build_directories)
    if not owned:
        assert "cannot establish module ownership" in reason
        assert "run_id=" + "b" * 32 in probe_runtime.result.read_text(encoding="ascii")


def test_probe_rejects_mismatched_installed_header_release(probe_runtime):
    """The running kernel release must match the installed header build release."""
    probe_runtime.header_release = "another-kernel"
    status, reason = multi._run_buffers_probe(probe_runtime.profile, mock.Mock())
    assert status == "SKIP" and "Installed headers identify" in reason
    assert not probe_runtime.build_directories
    assert len(probe_runtime.release_directories) == 1
    assert not probe_runtime.release_directories[0].exists()
    assert probe_runtime.command.call_count == 1


def test_probe_reads_result_with_shared_sysfs_reader(probe_runtime):
    """Normal readback uses the same utility as uncertain-load ownership checks."""
    assert multi._run_buffers_probe(probe_runtime.profile, mock.Mock())[0] == "PASS"
    probe_runtime.read.assert_called_once_with(probe_runtime.result)


def test_probe_kernelrelease_uses_external_module_directory(probe_runtime):
    """Release discovery and compilation share one existing temporary payload."""
    assert multi._run_buffers_probe(probe_runtime.profile, mock.Mock())[0] == "PASS"
    assert probe_runtime.release_directories == probe_runtime.build_directories
    commands = [call.args[0] for call in probe_runtime.command.call_args_list]
    assert commands[0][-1] == "kernelrelease"
    assert commands[1][-1] == "modules"
    for command in commands[:2]:
        assert f"M={probe_runtime.build_directories[0]}" in command
    assert not probe_runtime.release_directories[0].exists()


@pytest.mark.skipif(shutil.which("make") is None, reason="make is needed for the temporary Kbuild fixture")
def test_probe_release_query_preserves_header_tree(probe_runtime):
    """A distro-like Makefile selects cached ABI with M= and never rewrites headers."""
    cached = probe_runtime.headers / "include" / "config" / "kernel.release"
    cached.parent.mkdir(parents=True)
    cached.write_text(RELEASE + "\n", encoding="ascii")
    (probe_runtime.headers / "Makefile").write_text(
        ".PHONY: kernelrelease\n"
        "kernelrelease:\n"
        "ifeq ($(strip $(M)),)\n"
        "\t@printf 'base-version\\n' > include/config/kernel.release\n"
        "\t@printf 'base-version\\n'\n"
        "else\n"
        "\t@cat include/config/kernel.release\n"
        "\t@printf 'external-module\\n' > \"$(M)/release-query-marker\"\n"
        "endif\n", encoding="ascii",
    )
    before = {
        path.relative_to(probe_runtime.headers): (
            path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None,
        )
        for path in probe_runtime.headers.rglob("*")
    }
    simulated_command = probe_runtime.command.side_effect

    def external_release(argv, failure, **kwargs):
        if argv[-1] == "kernelrelease":
            directory = Path(next(argument[2:] for argument in argv if argument.startswith("M=")))
            assert directory.is_dir() and (directory / "Makefile").is_file()
            probe_runtime.release_directories.append(directory)
            completed = subprocess.run(argv, check=True, text=True, capture_output=True, timeout=10)
            assert (directory / "release-query-marker").read_text(encoding="ascii") == "external-module\n"
            return completed
        return simulated_command(argv, failure, **kwargs)

    probe_runtime.command.side_effect = external_release
    assert multi._run_buffers_probe(probe_runtime.profile, mock.Mock())[0] == "PASS"
    after = {
        path.relative_to(probe_runtime.headers): (
            path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None,
        )
        for path in probe_runtime.headers.rglob("*")
    }
    assert before == after
    assert probe_runtime.release_directories == probe_runtime.build_directories
    assert not probe_runtime.release_directories[0].exists()


def test_probe_signing_unavailable_cleans_temporary_build(probe_runtime):
    """An unavailable Secure Boot signing path skips before any module load."""
    probe_runtime.sign.return_value = False
    status, reason = multi._run_buffers_probe(probe_runtime.profile, mock.Mock())
    assert status == "SKIP" and "sign" in reason
    assert all(not directory.exists() for directory in probe_runtime.build_directories)
    assert all(call.args[0][0] != "insmod" for call in probe_runtime.command.call_args_list)


@pytest.mark.parametrize("case", ["headers", "tool", "existing_result", "existing_module"])
def test_probe_prerequisites_without_kernel_source(probe_runtime, monkeypatch, case):
    """Headers, tools, unloading configuration, and vacant probe resources suffice."""
    available = mock.Mock(return_value=True)
    monkeypatch.setattr(multi.glt_utils, "is_cli_available", available)
    assert multi._probe_prerequisite_reason() == ""
    if case == "headers":
        (probe_runtime.headers / "Makefile").unlink()
    elif case == "tool":
        available.side_effect = lambda command: command != "tpm2_getcap"
    elif case == "existing_result":
        probe_runtime.result.touch()
    else:
        probe_runtime.module_state.mkdir()
    assert multi._probe_prerequisite_reason()
    probe_runtime.command.assert_not_called()


@pytest.mark.parametrize("case", ["architecture", "acpi_absent", "other_method", "prerequisite", "profile"])
def test_native_applicability_skips_before_exercise(native, monkeypatch, case):
    """Unsupported applicability and missing runtime prerequisites never exercise hardware."""
    if case == "architecture":
        monkeypatch.setattr(multi.platform, "machine", lambda: "x86_64")
    elif case == "acpi_absent":
        native.acpi.side_effect = FileNotFoundError("no table")
    elif case == "other_method":
        native.acpi.return_value = None
    elif case == "prerequisite":
        native.prerequisite.return_value = "Matching installed headers are unavailable"
    else:
        native.discover.return_value = (None, "No compatible Buffers profile")
    with pytest.raises(GLTCore.GLTTestSkip):
        multi.test_arm_ffa_multi_uuid_messaging(None, native.logger)
    native.exercise.assert_not_called()


@pytest.mark.parametrize("case", ["disabled", "missing", "unreadable"])
def test_native_unavailable_unloading_skips_before_messaging(probe_runtime, native, monkeypatch, case):
    """No messaging or module load occurs without confirmed kernel unloading."""
    monkeypatch.setattr(multi, "_probe_prerequisite_reason", native.real_prerequisite)
    monkeypatch.setattr(multi.glt_utils, "is_cli_available", mock.Mock(return_value=True))
    if case == "disabled":
        probe_runtime.config.return_value["CONFIG_MODULE_UNLOAD"] = "n"
    elif case == "missing":
        probe_runtime.config.return_value.pop("CONFIG_MODULE_UNLOAD")
    else:
        probe_runtime.config.side_effect = PermissionError("unreadable config")
    with pytest.raises(GLTCore.GLTTestSkip) as failure:
        multi.test_arm_ffa_multi_uuid_messaging(None, native.logger)
    assert "unload" in str(failure.value).lower()
    native.exercise.assert_not_called()
    probe_runtime.command.assert_not_called()
    assert not probe_runtime.result.exists()
    assert not probe_runtime.module_state.exists()


@pytest.mark.parametrize("stage", ["acpi", "discover", "exercise"])
def test_native_advertised_or_runtime_errors_fail(native, stage):
    """Broken advertised identities and runtime errors cannot become profile skips."""
    getattr(native, stage).side_effect = ValueError("controlled broken identity")
    with pytest.raises(GLTCore.GLTTestFail, match="controlled broken identity") as caught:
        multi.test_arm_ffa_multi_uuid_messaging(None, native.logger)
    reason = str(caught.value)
    assert "be61da938576" not in reason
    if stage in ("discover", "exercise"):
        assert "Linux FF-A driver/API support" in reason
        assert "firmware service protocol" in reason
        assert "kernel logs" in reason


@pytest.mark.parametrize("status,exception", [("FAIL", GLTCore.GLTTestFail), ("SKIP", GLTCore.GLTTestSkip)])
def test_native_preserves_sequence_verdict(native, status, exception):
    """The native wrapper maps ordinary helper results in its decorated body."""
    native.exercise.return_value = (status, "controlled sequence reason")
    with pytest.raises(exception, match="controlled sequence reason"):
        multi.test_arm_ffa_multi_uuid_messaging(None, native.logger)


def test_native_success_runs_profile_sequence_once(native):
    """A native success requires the independently exercised real-service profile."""
    multi.test_arm_ffa_multi_uuid_messaging(None, native.logger)
    native.exercise.assert_called_once()
    native.load.assert_called_once()
