# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Arm BRBE nVHE host-isolation validation."""
# pylint: disable=protected-access,too-many-lines

import hashlib
import json
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from common import glt_core as GLTCore
from greenlit import greenlit
from modules.arm64 import glt_test_brbe
from modules.arm64 import glt_test_brbe_nvhe as brbe_nvhe
from tests.test_base import GLTTestBase


TEST_NAME = "test_brbe_nvhe_host_isolation"
TEST_KERNEL_NOTES_SHA256 = "a" * 64


def _kernel_identity(
    release="6.14-fixed", notes_sha256=TEST_KERNEL_NOTES_SHA256,
):
    return {
        "release": release,
        "notes_sha256": notes_sha256,
    }


class _ProgressScheduler:
    """Small in-memory scheduler progress model for reboot unit tests."""

    def __init__(self, values=None):
        self.log_dir = "/session"
        self.progress = [{TEST_NAME: dict(values or {})}]
        self.test_categories = set()
        self.test_tags = set()
        self.test_list = []

    @property
    def values(self):
        """Return mutable progress values for the BRBE test."""
        return self.progress[0][TEST_NAME]

    def progress_check(self, item, key):
        """Return whether a progress token exists."""
        return key in self.progress[0].get(item, {})

    def progress_delete_time(self, item, key):
        """Delete a progress token without filesystem persistence."""
        self.progress[0].get(item, {}).pop(key, None)

    def progress_record_time(self, item, key):
        """Record a deterministic marker in place of wall-clock time."""
        self.progress[0].setdefault(item, {})[key] = "recorded"

    def progress_update(self, item, key, value):
        """Store a progress value without filesystem persistence."""
        self.progress[0].setdefault(item, {})[key] = value


# pylint: disable=too-many-arguments
def _parameter_status(
    *, ok=True, active=False, configured=False, owned=False,
    reboot_required=False, reason="",
):
    return SimpleNamespace(
        ok=ok,
        active=active,
        configured=configured,
        added_by_glt=owned,
        reboot_required=reboot_required,
        reason=reason,
    )


def _parameter_statuses(**state):
    return {
        parameter: _parameter_status(**state)
        for parameter in brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
    }


class TestBrbeNvheMetadata(GLTTestBase):
    """Verify discovery and dependency metadata."""

    def test_harness_parameters_force_guest_instruction_overflow(self):
        """Pin the one-shot period and the larger dependent guest workload."""
        self.assertEqual(brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD, 10_000_000)
        self.assertEqual(brbe_nvhe.BRBE_NVHE_GUEST_BRANCHES, 20_000_000)

    def test_test_case_is_disruptive_and_stress_without_cross_plan_dependency(self):
        """The harness supplies host controls without a standard-only dependency."""
        test_case = brbe_nvhe.test_brbe_nvhe_host_isolation

        self.assertCountEqual(test_case.plans, ["disruptive", "stress"])
        self.assertEqual(
            GLTCore.GLTScheduler.get_test_case_name(test_case),
            "arm64:test_brbe_nvhe_host_isolation",
        )
        self.assertFalse(
            getattr(test_case.__wrapped__, "requires", GLTCore.GLTRequires()).all_true
        )
        hooks = brbe_nvhe.test_brbe_nvhe_host_isolation.attr.post_hooks
        for outcome in ("postsuccess", "postskip", "postafail", "postfail", "postwarn"):
            self.assertIs(getattr(hooks, outcome), brbe_nvhe._restore_nvhe_boot)

    def test_brbe_catalogue_is_valid_for_every_ci_plan(self):
        """Use the real CI planner without executing hardware test bodies."""
        host_id = "arm64:test_brbe_branch_stack"
        nvhe_id = "arm64:test_brbe_nvhe_host_isolation"
        catalogue = [
            glt_test_brbe.test_brbe_branch_stack,
            brbe_nvhe.test_brbe_nvhe_host_isolation,
        ]
        expected = {
            "standard": {host_id},
            "disruptive": {nvhe_id},
            "stress": {host_id, nvhe_id},
        }
        for plan_name, expected_ids in expected.items():
            with self.subTest(plan=plan_name):
                args = SimpleNamespace(**{
                    plan.value: plan.value == plan_name
                    for plan in brbe_nvhe.GLTCore.GLTPlanType
                })
                active_tags = greenlit.get_active_plan_tags(args)
                with self.mock.patch.object(
                    greenlit, "test_case_discover", return_value=catalogue
                ):
                    selected, error, exit_code = greenlit.discover_target_test_cases(
                        ["arm64"], None, active_tags)
                self.assertEqual(exit_code, 0, error)
                self.assertEqual(error, "")
                self.assertEqual(
                    {brbe_nvhe.GLTCore.GLTScheduler.get_test_case_name(case)
                     for case in selected},
                    expected_ids,
                )


class TestBrbeNvheCpuSelection(GLTTestBase):
    """Verify deterministic selection of an online, allowed CPU."""

    def test_selects_highest_online_cpu_in_process_affinity(self):
        """Intersect sysfs online CPUs with the process affinity mask."""
        online_path = self.mock.Mock()
        online_path.read_text.return_value = "0-3,8\n"
        with self.mock.patch.object(
            brbe_nvhe, "CPU_ONLINE_PATH", online_path
        ), self.mock.patch.object(
            brbe_nvhe.os, "sched_getaffinity", return_value={1, 3, 9}
        ):
            cpu, reason = brbe_nvhe._select_test_cpu()

        self.assertEqual((cpu, reason), (3, ""))

    def test_rejects_empty_online_affinity_intersection(self):
        """Do not silently run the pinned-vCPU probe on an unavailable CPU."""
        online_path = self.mock.Mock()
        online_path.read_text.return_value = "0-3\n"
        with self.mock.patch.object(
            brbe_nvhe, "CPU_ONLINE_PATH", online_path
        ), self.mock.patch.object(
            brbe_nvhe.os, "sched_getaffinity", return_value={8, 9}
        ):
            cpu, reason = brbe_nvhe._select_test_cpu()

        self.assertEqual(cpu, -1)
        self.assertIn("No online CPU", reason)

    def test_reports_unreadable_online_mask(self):
        """Surface CPU-topology inspection errors as inconclusive setup."""
        online_path = self.mock.Mock()
        online_path.read_text.side_effect = OSError("permission denied")
        with self.mock.patch.object(
            brbe_nvhe, "CPU_ONLINE_PATH", online_path
        ):
            cpu, reason = brbe_nvhe._select_test_cpu()

        self.assertEqual(cpu, -1)
        self.assertIn("permission denied", reason)



class TestBrbeNvhePreflight(GLTTestBase):
    """Lightweight reads must precede boot mutation, without a perf probe."""

    def setUp(self):
        """Provide readable Vera capabilities and stop at the setup boundary."""
        super().setUp()
        self.scheduler = _ProgressScheduler()
        self.nvdev = self.mock.Mock()
        self.nvdev.is_cpu_type.return_value = True
        self.kernel_config = self.mock.Mock(return_value={"CONFIG_ARM64_BRBE": "y"})
        self.pmu = self.mock.MagicMock()
        self.pmu.is_dir.return_value = True
        self.capability = self.pmu.__truediv__.return_value.__truediv__.return_value
        self.capability.read_text.return_value = "32\n"
        self.event_source = self.mock.Mock()
        self.event_source.glob.return_value = [self.pmu]
        self.operations = SimpleNamespace(
            setup=self.mock.Mock(return_value=(False, "stop after preflight", True)),
            append=self.mock.Mock(), remove=self.mock.Mock(), reboot=self.mock.Mock(),
            build=self.mock.Mock(), harness=self.mock.Mock(),
        )
        patches = [
            self.mock.patch.object(
                brbe_nvhe.glt_utils, "is_cli_available", return_value=True
            ),
            self.mock.patch.object(Path, "is_file", return_value=True),
            self.mock.patch.object(brbe_nvhe, "_select_test_cpu", return_value=(7, "")),
            self.mock.patch.object(brbe_nvhe, "NvDev", return_value=self.nvdev),
            self.mock.patch.object(
                brbe_nvhe.check_kernel_config, "parse_kconfig", self.kernel_config
            ),
            self.mock.patch.object(
                brbe_nvhe.perf_utils, "Path", return_value=self.event_source
            ),
            self.mock.patch.object(brbe_nvhe, "_ensure_nvhe_environment", self.operations.setup),
            self.mock.patch.object(
                brbe_nvhe.glt_utils, "append_kernel_parameter", self.operations.append
            ),
            self.mock.patch.object(
                brbe_nvhe.glt_utils, "remove_kernel_parameter", self.operations.remove
            ),
            self.mock.patch.object(brbe_nvhe.glt_utils, "reboot", self.operations.reboot),
            self.mock.patch.object(brbe_nvhe.glt_utils, "run_command", self.operations.build),
            self.mock.patch.object(brbe_nvhe.subprocess, "run", self.operations.harness),
        ]
        for patcher in patches:
            self.addCleanup(patcher.stop)
            patcher.start()

    def test_non_vera_skips_before_kernel_reads_or_boot_changes(self):
        """An unsupported target cannot acquire ownership or execute workloads."""
        self.nvdev.is_cpu_type.return_value = False
        with self.assertRaisesRegex(brbe_nvhe.GLTCore.GLTTestSkip, "Vera"):
            brbe_nvhe.test_brbe_nvhe_host_isolation(self.scheduler, self.mock_logger)
        self.kernel_config.assert_not_called()
        self.event_source.glob.assert_not_called()
        for operation in vars(self.operations).values():
            operation.assert_not_called()
        self.assertNotIn(brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN, self.scheduler.values)

    def test_disabled_or_missing_brbe_config_skips_before_boot_changes(self):
        """No observer means no reason to switch the system into nVHE."""
        for config in ({}, {"CONFIG_ARM64_BRBE": "n"}):
            with self.subTest(config=config):
                self.kernel_config.return_value = config
                with self.assertRaisesRegex(
                    brbe_nvhe.GLTCore.GLTTestSkip, "CONFIG_ARM64_BRBE"
                ):
                    brbe_nvhe.test_brbe_nvhe_host_isolation(
                        self.scheduler, self.mock_logger
                    )
        for operation in vars(self.operations).values():
            operation.assert_not_called()

    def test_unreadable_kernel_config_skips_without_claiming_missing_support(self):
        """An uninspectable configuration is not proof of a missing patch."""
        self.kernel_config.side_effect = OSError("config unavailable")
        with self.assertRaisesRegex(
            brbe_nvhe.GLTCore.GLTTestSkip, "Unable to inspect kernel config"
        ):
            brbe_nvhe.test_brbe_nvhe_host_isolation(self.scheduler, self.mock_logger)
        for operation in vars(self.operations).values():
            operation.assert_not_called()

    def test_no_arm_pmu_skips_before_boot_changes(self):
        """An absent Arm PMU cannot supply the isolation observer."""
        self.event_source.glob.return_value = []
        with self.assertRaisesRegex(brbe_nvhe.GLTCore.GLTTestSkip, "armv8_pmuv3 PMU"):
            brbe_nvhe.test_brbe_nvhe_host_isolation(self.scheduler, self.mock_logger)
        for operation in vars(self.operations).values():
            operation.assert_not_called()

    def test_invalid_brbe_capacities_skip_before_boot_changes(self):
        """Malformed or unsupported capacities cannot justify disruptive setup."""
        for count in ("0", "7", "128", "malformed"):
            with self.subTest(count=count):
                self.capability.read_text.return_value = count
                with self.assertRaisesRegex(
                    brbe_nvhe.GLTCore.GLTTestSkip, "BRBE"
                ):
                    brbe_nvhe.test_brbe_nvhe_host_isolation(
                        self.scheduler, self.mock_logger
                    )
        for operation in vars(self.operations).values():
            operation.assert_not_called()

    def test_unreadable_capability_skips_before_boot_changes(self):
        """Missing or unreadable capability files do not trigger reboot."""
        self.capability.read_text.side_effect = OSError("capability unavailable")
        with self.assertRaisesRegex(brbe_nvhe.GLTCore.GLTTestSkip, "BRBE"):
            brbe_nvhe.test_brbe_nvhe_host_isolation(self.scheduler, self.mock_logger)
        for operation in vars(self.operations).values():
            operation.assert_not_called()

    def test_valid_capabilities_reach_setup_without_running_perf(self):
        """Successful static checks permit setup, not a premature PASS verdict."""
        with self.assertRaisesRegex(
            brbe_nvhe.GLTCore.GLTTestSkip, "stop after preflight"
        ):
            brbe_nvhe.test_brbe_nvhe_host_isolation(self.scheduler, self.mock_logger)
        self.operations.setup.assert_called_once_with(TEST_NAME, self.scheduler, self.mock_logger)
        self.event_source.glob.assert_called_once_with("armv8_pmuv3*")
        self.capability.read_text.assert_called_once_with(encoding="utf-8")
        for name in ("append", "remove", "reboot", "build", "harness"):
            getattr(self.operations, name).assert_not_called()

    def test_skip_gates_take_priority_over_cpu_inspection_failure(self):
        """Unsupported targets SKIP even when CPU inspection would fail."""
        cases = (
            ("platform", False, True, True, "32"),
            ("config", True, False, True, "32"),
            ("pmu", True, True, False, "32"),
            ("capacity", True, True, True, "0"),
        )
        for name, vera, configured, has_pmu, capacity in cases:
            with self.subTest(gate=name), self.mock.patch.object(
                brbe_nvhe, "_select_test_cpu",
                return_value=(-1, "online or affinity inspection unavailable"),
            ) as select:
                self.nvdev.is_cpu_type.return_value = vera
                self.kernel_config.return_value = {"CONFIG_ARM64_BRBE": "y"} if configured else {}
                self.event_source.glob.return_value = [self.pmu] if has_pmu else []
                self.capability.read_text.return_value = capacity
                with self.assertRaises(brbe_nvhe.GLTCore.GLTTestSkip):
                    brbe_nvhe.test_brbe_nvhe_host_isolation(
                        self.scheduler, self.mock_logger
                    )
                select.assert_not_called()
                for operation in vars(self.operations).values():
                    operation.assert_not_called()

    def test_cpu_selection_failure_does_not_prepare_boot(self):
        """CPU selection remains necessary even after the TODO is resolved."""
        with self.mock.patch.object(
            brbe_nvhe, "_select_test_cpu", return_value=(-1, "no usable CPU")
        ):
            with self.assertRaises(brbe_nvhe.GLTCore.GLTTestFail):
                brbe_nvhe.test_brbe_nvhe_host_isolation(
                    self.scheduler, self.mock_logger
                )
        self.nvdev.is_cpu_type.assert_called_once_with("vera")
        self.capability.read_text.assert_called_once_with(encoding="utf-8")
        for operation in vars(self.operations).values():
            operation.assert_not_called()


class TestBrbeNvheModeAlias(GLTTestBase):
    """Accept user aliases without weakening exact GLT ownership checks."""

    def test_nvhe_alias_is_not_a_conflict(self):
        """Both kernel-supported spellings accept the desired nVHE value."""
        cmdline_path = self.mock.Mock()
        cmdline_path.read_text.return_value = (
            "kvm_arm.mode=nvhe arm64_sw.hvhe=0 id_aa64mmfr1.vh=0"
        )
        with self.mock.patch.object(brbe_nvhe, "Path", return_value=cmdline_path):
            self.assertEqual(brbe_nvhe._get_active_parameter_conflict(), ("", False))

    def test_active_persistent_user_alias_needs_no_setup_reboot(self):
        """An already active alias satisfies the user-owned mode requirement."""
        scheduler = _ProgressScheduler()
        statuses = (
            _parameter_status(),
            _parameter_status(active=True, configured=True),
            _parameter_status(active=True, configured=True),
            _parameter_status(active=True, configured=True),
        )
        with self.mock.patch.object(
            brbe_nvhe.glt_utils, "check_kernel_parameter", side_effect=statuses
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "append_kernel_parameter"
        ) as append, self.mock.patch.object(
            brbe_nvhe.glt_utils, "remove_kernel_parameter"
        ) as remove, self.mock.patch.object(
            brbe_nvhe.glt_utils, "reboot"
        ) as reboot:
            self.assertEqual(
                brbe_nvhe._ensure_nvhe_environment(TEST_NAME, scheduler, self.mock_logger),
                (True, "", False),
            )
        append.assert_not_called()
        remove.assert_not_called()
        reboot.assert_not_called()
        self.assertNotIn(brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN, scheduler.values)

    def test_user_alias_does_not_mask_an_owned_canonical_token(self):
        """Owned setup and cleanup inspect the exact token GLT added."""
        scheduler = _ProgressScheduler({
            brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: {
                "added_parameters": ["kvm-arm.mode=nvhe"],
            },
        })
        with self.mock.patch.object(
            brbe_nvhe.glt_utils, "check_kernel_parameter",
            side_effect=(_parameter_status(), _parameter_status(), _parameter_status()),
        ) as check:
            statuses, reason = brbe_nvhe._check_nvhe_parameters(TEST_NAME, scheduler)
        self.assertEqual(reason, "")
        self.assertFalse(statuses["kvm-arm.mode=nvhe"].active)
        self.assertFalse(statuses["kvm-arm.mode=nvhe"].configured)
        self.assertEqual(check.call_count, 3)

    def test_alias_inspection_error_is_not_ignored(self):
        """An unreadable alias state is reported rather than treated as absent."""
        with self.mock.patch.object(
            brbe_nvhe.glt_utils, "check_kernel_parameter",
            side_effect=(
                _parameter_status(), _parameter_status(), _parameter_status(),
                _parameter_status(ok=False, reason="alias backend unavailable"),
            ),
        ):
            _, reason = brbe_nvhe._check_nvhe_parameters(TEST_NAME, _ProgressScheduler())
        self.assertIn("kvm_arm.mode=nvhe: alias backend unavailable", reason)


class TestBrbeNvheBootEvidence(GLTTestBase):
    """Verify that only exact current-boot nVHE evidence is accepted."""

    def test_accepts_exact_nvhe_initialization_message(self):
        """Recognize the kernel's production nVHE initialization record."""
        with self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "get_kernel_log",
            return_value=(
                "[0.000] boot\n"
                f"[0.100] {brbe_nvhe.BRBE_NVHE_LOG_MESSAGE}\n"
            ),
        ):
            self.assertEqual(brbe_nvhe._get_nvhe_log_failure(), "")

    def test_rejects_vhe_or_approximate_log_messages(self):
        """Do not infer nVHE from ordinary VHE or similar-looking text."""
        messages = (
            "VHE mode initialized successfully",
            "Hyp nVHE mode initialized",
            "nVHE mode initialized successfully",
        )
        for message in messages:
            with self.subTest(message=message), self.mock.patch.object(
                brbe_nvhe.glt_utils, "get_kernel_log", return_value=message
            ):
                reason = brbe_nvhe._get_nvhe_log_failure()
                self.assertIn(brbe_nvhe.BRBE_NVHE_LOG_MESSAGE, reason)

    def test_reports_unavailable_current_boot_log(self):
        """An unreadable journal cannot prove the requested hypervisor mode."""
        with self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "get_kernel_log",
            side_effect=RuntimeError("journal unavailable"),
        ):
            reason = brbe_nvhe._get_nvhe_log_failure()

        self.assertIn("journal unavailable", reason)


class TestBrbeNvheKernelIdentity(GLTTestBase):
    """Verify exact running-image identity across the test's reboots."""

    def test_hashes_running_kernel_notes_with_release(self):
        """Use binary running-kernel notes, not only the release string."""
        notes = b"kernel-note-one\x00kernel-note-two"
        with tempfile.TemporaryDirectory() as temp_dir:
            notes_path = Path(temp_dir) / "notes"
            notes_path.write_bytes(notes)
            with self.mock.patch.object(
                brbe_nvhe.platform, "release", return_value="6.14-test"
            ):
                identity, reason = brbe_nvhe._get_running_kernel_identity(
                    notes_path)

        self.assertEqual(reason, "")
        self.assertEqual(
            identity,
            {
                "release": "6.14-test",
                "notes_sha256": hashlib.sha256(notes).hexdigest(),
            },
        )

    def test_reports_unreadable_running_kernel_notes(self):
        """Fail closed when the running image cannot be fingerprinted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            identity, reason = brbe_nvhe._get_running_kernel_identity(
                Path(temp_dir) / "missing-notes")

        self.assertEqual(identity, {})
        self.assertIn("Unable to fingerprint", reason)

    def test_same_release_with_different_notes_is_rejected(self):
        """A same-version package replacement is still a kernel change."""
        scheduler = _ProgressScheduler(
            {brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: _kernel_identity()}
        )
        with self.mock.patch.object(
            brbe_nvhe,
            "_get_running_kernel_identity",
            return_value=(_kernel_identity(notes_sha256="b" * 64), ""),
        ):
            reason = brbe_nvhe._verify_owned_kernel(
                TEST_NAME, scheduler, "setup reboot")

        self.assertIn("changed kernels", reason)
        self.assertIn(TEST_KERNEL_NOTES_SHA256, reason)
        self.assertIn("b" * 64, reason)


    def test_owned_parameter_list_rejects_invalid_saved_values(self):
        """Reject malformed parameter ownership before using saved state."""
        parameter = brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS[0]
        for added in ([], None, "nvhe", ["unknown"], [parameter, parameter], [{}]):
            with self.subTest(added=added):
                scheduler = _ProgressScheduler({
                    brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: {
                        **_kernel_identity(), "added_parameters": added,
                    },
                })
                reason = brbe_nvhe._verify_owned_kernel(
                    TEST_NAME, scheduler, "setup reboot")
                self.assertIn("invalid saved parameter list", reason)


class TestBrbeNvheBootLifecycle(GLTTestBase):
    """Verify atomic setup, resume, and restoration of nVHE boot state."""

    def test_parameter_tuple_and_checks_use_one_transaction(self):
        """Inspect the exact three tokens under the test's transaction ID."""
        self.assertEqual(
            brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS,
            (
                "kvm-arm.mode=nvhe",
                "arm64_sw.hvhe=0",
                "id_aa64mmfr1.vh=0",
            ),
        )
        scheduler = _ProgressScheduler()
        status_values = (
            _parameter_status(active=True),
            _parameter_status(ok=False, reason="backend unavailable"),
            _parameter_status(configured=True),
        )
        with self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "check_kernel_parameter",
            side_effect=(*status_values, _parameter_status()),
        ) as mock_check:
            statuses, reason = brbe_nvhe._check_nvhe_parameters(
                TEST_NAME, scheduler
            )

        self.assertEqual(
            tuple(statuses), brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        )
        self.assertIn("arm64_sw.hvhe=0: backend unavailable", reason)
        self.assertEqual(
            mock_check.call_args_list,
            [
                self.mock.call(
                    parameter,
                    test_case_name=TEST_NAME,
                    session_dir=scheduler.log_dir,
                )
                for parameter in (*brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS, "kvm_arm.mode=nvhe")
            ],
        )

    def test_conflict_parser_recognizes_values_and_kvm_arm_alias(self):
        """Do not append canonical tokens beside user-selected conflicts."""
        cases = {
            "kvm_value": "kvm-arm.mode=vhe",
            "hvhe_value": "arm64_sw.hvhe=1",
            "vh_value": "id_aa64mmfr1.vh=1",
            "kvm_alias": "kvm_arm.mode=vhe",
        }
        for name, cmdline in cases.items():
            cmdline_path = self.mock.Mock()
            cmdline_path.read_text.return_value = cmdline
            with self.subTest(name=name), self.mock.patch.object(
                brbe_nvhe, "Path", return_value=cmdline_path
            ):
                reason, should_skip = (
                    brbe_nvhe._get_active_parameter_conflict()
                )
                self.assertTrue(should_skip)
                self.assertIn(cmdline, reason)

        cmdline_path = self.mock.Mock()
        cmdline_path.read_text.return_value = " ".join(
            brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        )
        with self.mock.patch.object(
            brbe_nvhe, "Path", return_value=cmdline_path
        ):
            self.assertEqual(
                brbe_nvhe._get_active_parameter_conflict(), ("", False)
            )

    def test_resume_disables_service_only_after_greenlit_reboot(self):
        """Only a recorded setup reboot disables the resume service."""
        for rebooted in (False, True):
            with self.subTest(rebooted=rebooted):
                scheduler = _ProgressScheduler(
                    {brbe_nvhe.GLT_REBOOT_TOKEN: "saved"} if rebooted else {}
                )
                with self.mock.patch.object(
                    brbe_nvhe.glt_utils, "systemd_enable"
                ) as enable, self.mock.patch.object(
                    brbe_nvhe.glt_utils, "is_cli_available",
                    side_effect=RuntimeError("stop after service check")
                ):
                    with self.assertRaisesRegex(RuntimeError, "stop after service check"):
                        brbe_nvhe.test_brbe_nvhe_host_isolation(
                            scheduler, self.mock.Mock()
                        )
                if rebooted:
                    enable.assert_called_once_with(enable=False)
                else:
                    enable.assert_not_called()

    def test_all_active_user_configuration_runs_without_mutation(self):
        """Reuse a complete user-selected nVHE boot without claiming it."""
        scheduler = _ProgressScheduler()
        statuses = _parameter_statuses(active=True, configured=True)
        with self.mock.patch.object(
            brbe_nvhe, "_check_nvhe_parameters", return_value=(statuses, "")
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "append_kernel_parameter"
        ) as mock_append, self.mock.patch.object(
            brbe_nvhe.glt_utils, "remove_kernel_parameter"
        ) as mock_remove, self.mock.patch.object(
            brbe_nvhe.glt_utils, "reboot"
        ) as mock_reboot:
            result = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertEqual(result, (True, "", False))
        mock_append.assert_not_called()
        mock_remove.assert_not_called()
        mock_reboot.assert_not_called()
        self.assertNotIn(
            brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN, scheduler.values
        )

    def test_all_absent_appends_all_parameters_then_reboots(self):
        """Configure all three tokens with the same transaction before reboot."""
        scheduler = _ProgressScheduler()
        absent = _parameter_statuses()
        configured = _parameter_statuses(
            configured=True, owned=True, reboot_required=True
        )
        check_results = [(absent, "")] + [(configured, "")] * 4
        with self.mock.patch.object(
            brbe_nvhe, "_check_nvhe_parameters", side_effect=check_results
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe,
            "_get_running_kernel_identity",
            return_value=(_kernel_identity(), ""),
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "append_kernel_parameter",
            return_value=(True, ""),
        ) as mock_append, self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "remove_kernel_parameter",
            return_value=(True, ""),
        ) as mock_remove, self.mock.patch.object(
            brbe_nvhe.glt_utils, "reboot"
        ) as mock_reboot:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertFalse(ready)
        self.assertFalse(should_skip)
        self.assertIn("process did not exit", reason)
        expected_calls = [
            self.mock.call(
                parameter,
                test_case_name=TEST_NAME,
                session_dir=scheduler.log_dir,
            )
            for parameter in brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        ]
        self.assertEqual(mock_append.call_args_list, expected_calls)
        self.assertEqual(mock_remove.call_args_list, expected_calls)
        mock_reboot.assert_called_once_with(TEST_NAME, scheduler)
        self.assertEqual(
            scheduler.values[brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN],
            {
                **_kernel_identity(),
                "added_parameters": list(brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS),
            },
        )

    def test_nonpersistent_or_fully_pending_user_state_skips_without_mutation(self):
        """Leave user-only state unchanged when a safe setup is unavailable."""
        nonpersistent = _parameter_statuses()
        nonpersistent[brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS[0]] = (
            _parameter_status(active=True))
        pending = _parameter_statuses(configured=True)
        for states in (nonpersistent, pending):
            with self.subTest(states=states):
                scheduler = _ProgressScheduler()
                with self.mock.patch.object(
                    brbe_nvhe, "_check_nvhe_parameters", return_value=(states, "")
                ), self.mock.patch.object(
                    brbe_nvhe, "_get_active_parameter_conflict",
                    return_value=("", False)
                ), self.mock.patch.object(
                    brbe_nvhe.glt_utils, "append_kernel_parameter"
                ) as append, self.mock.patch.object(
                    brbe_nvhe.glt_utils, "remove_kernel_parameter"
                ) as remove, self.mock.patch.object(
                    brbe_nvhe.glt_utils, "reboot"
                ) as reboot:
                    ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                        TEST_NAME, scheduler, self.mock.Mock())

                self.assertFalse(ready)
                self.assertTrue(should_skip)
                self.assertIn("User nVHE", reason)
                append.assert_not_called()
                remove.assert_not_called()
                reboot.assert_not_called()
                self.assertEqual(scheduler.values, {})

    def test_conflicting_user_state_skips_without_mutation(self):
        """Map both conflicting values and the alias to a non-mutating skip."""
        for conflict in ("kvm-arm.mode=vhe", "kvm_arm.mode=nvhe"):
            scheduler = _ProgressScheduler()
            with self.subTest(conflict=conflict), self.mock.patch.object(
                brbe_nvhe,
                "_check_nvhe_parameters",
                return_value=(_parameter_statuses(), ""),
            ), self.mock.patch.object(
                brbe_nvhe,
                "_get_active_parameter_conflict",
                return_value=(conflict, True),
            ), self.mock.patch.object(
                brbe_nvhe.glt_utils, "append_kernel_parameter"
            ) as mock_append, self.mock.patch.object(
                brbe_nvhe.glt_utils, "remove_kernel_parameter"
            ) as mock_remove:
                result = brbe_nvhe._ensure_nvhe_environment(
                    TEST_NAME, scheduler, self.mock_logger
                )

            self.assertEqual(result, (False, conflict, True))
            mock_append.assert_not_called()
            mock_remove.assert_not_called()

    def test_orphan_owned_metadata_fails_without_mutation(self):
        """Do not remove a transaction if its progress ownership is lost."""
        statuses = _parameter_statuses()
        statuses[brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS[0]] = (
            _parameter_status(configured=True, owned=True)
        )
        scheduler = _ProgressScheduler()
        with self.mock.patch.object(
            brbe_nvhe, "_check_nvhe_parameters", return_value=(statuses, "")
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "append_kernel_parameter"
        ) as mock_append, self.mock.patch.object(
            brbe_nvhe.glt_utils, "remove_kernel_parameter"
        ) as mock_remove:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertFalse(ready)
        self.assertFalse(should_skip)
        self.assertIn("orphan GreenLiT", reason)
        mock_append.assert_not_called()
        mock_remove.assert_not_called()

    def test_append_failure_removes_all_and_preserves_cleanup_error(self):
        """Roll back the shared transaction even after a partial append."""
        scheduler = _ProgressScheduler()
        partial = _parameter_statuses()
        partial[brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS[0]] = (
            _parameter_status(configured=True, owned=True)
        )
        with self.mock.patch.object(
            brbe_nvhe,
            "_check_nvhe_parameters",
            side_effect=[(_parameter_statuses(), ""), (partial, "")],
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe,
            "_get_running_kernel_identity",
            return_value=(_kernel_identity(), ""),
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "append_kernel_parameter",
            side_effect=[(True, ""), (False, "write failed")],
        ) as mock_append, self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "remove_kernel_parameter",
            side_effect=[(True, ""), (False, "cleanup denied"), (True, "")],
        ) as mock_remove:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertFalse(ready)
        self.assertFalse(should_skip)
        self.assertIn("write failed", reason)
        self.assertIn("cleanup denied", reason)
        self.assertEqual(mock_append.call_count, 2)
        self.assertEqual(mock_remove.call_count, 3)

    def test_post_append_check_failure_rolls_back_with_cleanup_error(self):
        """A metadata read failure after writing one token rolls back all."""
        scheduler = _ProgressScheduler()
        with self.mock.patch.object(
            brbe_nvhe,
            "_check_nvhe_parameters",
            side_effect=[
                (_parameter_statuses(), ""),
                ({}, "metadata read failed"),
            ],
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe,
            "_get_running_kernel_identity",
            return_value=(_kernel_identity(), ""),
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "append_kernel_parameter",
            return_value=(True, ""),
        ) as mock_append, self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "remove_kernel_parameter",
            side_effect=[(False, "cleanup failed"), (True, ""), (True, "")],
        ) as mock_remove:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertFalse(ready)
        self.assertFalse(should_skip)
        self.assertIn("metadata read failed", reason)
        self.assertIn("cleanup failed", reason)
        mock_append.assert_called_once()
        self.assertEqual(mock_remove.call_count, 3)

    def test_owned_resume_requires_same_kernel_and_cleans_persistent_state(self):
        """Keep tokens active but remove GLT config before running workload."""
        scheduler = _ProgressScheduler(
            {brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: _kernel_identity()}
        )
        before_cleanup = _parameter_statuses(
            active=True, configured=True, owned=True
        )
        after_cleanup = _parameter_statuses(active=True)
        with self.mock.patch.object(
            brbe_nvhe,
            "_check_nvhe_parameters",
            side_effect=[(before_cleanup, ""), (after_cleanup, "")],
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe,
            "_get_running_kernel_identity",
            return_value=(_kernel_identity(), ""),
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "remove_kernel_parameter",
            return_value=(True, ""),
        ) as mock_remove, self.mock.patch.object(
            brbe_nvhe.glt_utils, "append_kernel_parameter"
        ) as mock_append, self.mock.patch.object(
            brbe_nvhe.glt_utils, "reboot"
        ) as mock_reboot:
            result = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertEqual(result, (True, "", False))
        self.assertEqual(mock_remove.call_count, 3)
        mock_append.assert_not_called()
        mock_reboot.assert_not_called()

    def test_owned_resume_wrong_kernel_fails_and_cleans_all(self):
        """Never validate a different kernel from the one setup recorded."""
        scheduler = _ProgressScheduler(
            {brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: _kernel_identity()}
        )
        with self.mock.patch.object(
            brbe_nvhe,
            "_check_nvhe_parameters",
            return_value=(
                _parameter_statuses(active=True, configured=True, owned=True),
                "",
            ),
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe,
            "_get_running_kernel_identity",
            return_value=(_kernel_identity(notes_sha256="b" * 64), ""),
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "remove_kernel_parameter",
            return_value=(True, ""),
        ) as mock_remove:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertFalse(ready)
        self.assertFalse(should_skip)
        self.assertIn("changed kernels", reason)
        self.assertEqual(mock_remove.call_count, 3)

    def test_owned_resume_incomplete_activation_fails_and_cleans_all(self):
        """The setup reboot must activate every required token."""
        scheduler = _ProgressScheduler(
            {brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: _kernel_identity()}
        )
        statuses = _parameter_statuses(configured=True, owned=True)
        with self.mock.patch.object(
            brbe_nvhe, "_check_nvhe_parameters", return_value=(statuses, "")
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe,
            "_get_running_kernel_identity",
            return_value=(_kernel_identity(), ""),
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils,
            "remove_kernel_parameter",
            return_value=(True, ""),
        ) as mock_remove:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock_logger
            )

        self.assertFalse(ready)
        self.assertFalse(should_skip)
        self.assertIn("did not activate", reason)
        self.assertEqual(mock_remove.call_count, 3)

    def test_runner_saves_result_before_posthook_reboot_and_skips_on_resume(self):
        """Verdicts survive reboot or best-effort cleanup errors and resume."""

        core = brbe_nvhe.GLTCore
        test_case = brbe_nvhe.test_brbe_nvhe_host_isolation
        attr = test_case.attr
        outcomes = (
            (core.GLTTestResult.PASS, None),
            (core.GLTTestResult.FAIL, core.GLTTestFail("contamination")),
            (core.GLTTestResult.SKIP, core.GLTTestSkip("unavailable")),
            (core.GLTTestResult.AFAIL, core.GLTTestFail("allowed failure")),
            (core.GLTTestResult.WARN, core.GLTTestWarn("advisory")),
        )
        for verdict, exception, cleanup_ok in (
            (verdict, exception, cleanup_ok)
            for verdict, exception in outcomes for cleanup_ok in (True, False)
        ):
            with self.subTest(verdict=verdict, cleanup_ok=cleanup_ok):
                scheduler = _ProgressScheduler({
                    brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: _kernel_identity(),
                })
                scheduler.test_categories = {attr.category}
                scheduler.test_tags = set(attr.plans)
                scheduler.test_list = [test_case]
                scheduler.log_dir = "/session"
                def measurement_result(_sd, _logger, outcome=verdict, error=exception):
                    # Decorated tests store their verdict before raising an outcome.
                    if error:
                        status = (
                            core.GLTTestResult.FAIL
                            if outcome == core.GLTTestResult.AFAIL else outcome
                        )
                        attr.result = (status, str(error))
                        raise error

                body = self.mock.Mock(spec=[], side_effect=measurement_result)
                body.attr = attr
                if verdict == core.GLTTestResult.AFAIL:
                    body.allow_failure = self.mock.Mock(spec=["can_fail"])
                    body.allow_failure.can_fail.return_value = (False, str(exception))

                def request_reboot(
                    test_name, sd, current=scheduler, outcome=verdict, error=exception
                ):
                    self.assertEqual(test_name, TEST_NAME)
                    self.assertIs(sd, current)
                    self.assertEqual(sd.values["Result"], outcome.value)
                    self.assertEqual(
                        sd.values["ResultReason"], str(error) if error else ""
                    )
                    self.assertIn("Completed", sd.values)
                    self.assertNotIn(brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN, sd.values)
                    raise SystemExit(0)

                with self.mock.patch.object(
                brbe_nvhe.GLTCore.GLTRequires, "can_start", return_value=(True, "")
                ), self.mock.patch.object(
                    attr, "func", body
                ), self.mock.patch.object(
                    attr, "result", (core.GLTTestResult.PASS, "")
                ), self.mock.patch.object(
                    brbe_nvhe, "_get_running_kernel_identity",
                    return_value=(_kernel_identity(), "")
                ), self.mock.patch.object(
                    brbe_nvhe.glt_utils, "remove_kernel_parameter",
                    return_value=(cleanup_ok, "" if cleanup_ok else "remove failed")
                ) as remove, self.mock.patch.object(
                    brbe_nvhe, "_check_nvhe_parameters",
                    return_value=(_parameter_statuses(active=True), "")
                ), self.mock.patch.object(
                    brbe_nvhe.glt_utils, "reboot", side_effect=request_reboot
                ) as reboot, self.mock.patch.object(
                    greenlit.glt_utils, "collect_logs", return_value=(True, "")
                ), self.mock.patch.object(greenlit, "show_report") as report:
                    self.mock_logger.getChild.return_value.error.reset_mock()
                    if cleanup_ok:
                        with self.assertRaises(SystemExit):
                            greenlit.perform_tests(self.mock_logger, scheduler)
                    else:
                        greenlit.perform_tests(self.mock_logger, scheduler)
                        self.mock_logger.getChild.return_value.error.assert_called()
                        self.assertIn(brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN, scheduler.values)
                        report.assert_called_once()
                    body.assert_called_once()
                    self.assertEqual(remove.call_count, 3)
                    self.assertEqual(reboot.call_count, int(cleanup_ok))

                    resumed = _ProgressScheduler(scheduler.values.copy())
                    resumed.test_categories = scheduler.test_categories
                    resumed.test_tags = scheduler.test_tags
                    resumed.test_list = scheduler.test_list
                    resumed.log_dir = scheduler.log_dir
                    greenlit.perform_tests(self.mock_logger, resumed)
                    body.assert_called_once()
                    self.assertEqual(reboot.call_count, int(cleanup_ok))
                    self.assertEqual(resumed.values["Result"], verdict.value)

    def test_posthook_without_ownership_does_not_mutate_or_reboot(self):
        """A user-owned environment needs no posthook mutation or reboot."""
        scheduler = _ProgressScheduler({
            "Result": "PASSED", "ResultReason": "", "Completed": "saved",
        })
        before = scheduler.values.copy()
        with self.mock.patch.object(
            brbe_nvhe, "_remove_owned_nvhe_parameters"
        ) as remove, self.mock.patch.object(brbe_nvhe.glt_utils, "reboot") as reboot:
            brbe_nvhe._restore_nvhe_boot(
                scheduler, brbe_nvhe.test_brbe_nvhe_host_isolation.attr, self.mock_logger
            )
        remove.assert_not_called()
        reboot.assert_not_called()
        self.assertEqual(scheduler.values, before)

    def test_posthook_cleanup_errors_do_not_rewrite_completed_verdict(self):
        """Cleanup errors are logged without replacing the BRBE verdict."""
        for kernel_changed, remove_ok, check_reason, configured, owned in (
            (True, True, "", False, False),
            (False, False, "", False, False),
            (False, True, "status unavailable", False, False),
            (False, True, "", True, False),
            (False, True, "", False, True),
        ):
            with self.subTest(
                kernel_changed=kernel_changed, remove_ok=remove_ok,
                check_reason=check_reason, configured=configured, owned=owned
            ):
                scheduler = _ProgressScheduler({
                    brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: _kernel_identity(),
                    "Result": "PASSED", "ResultReason": "", "Completed": "saved",
                })
                identity = _kernel_identity(
                    notes_sha256="b" * 64 if kernel_changed else TEST_KERNEL_NOTES_SHA256
                )
                with self.mock.patch.object(
                    brbe_nvhe, "_get_running_kernel_identity", return_value=(identity, "")
                ), self.mock.patch.object(
                    brbe_nvhe.glt_utils, "remove_kernel_parameter",
                    return_value=(remove_ok, "" if remove_ok else "remove failed")
                ) as remove, self.mock.patch.object(
                    brbe_nvhe, "_check_nvhe_parameters",
                    return_value=(
                        _parameter_statuses(
                            active=True, configured=configured, owned=owned
                        ), check_reason
                    )
                ), self.mock.patch.object(brbe_nvhe.glt_utils, "reboot") as reboot:
                    self.mock_logger.error.reset_mock()
                    brbe_nvhe._restore_nvhe_boot(
                        scheduler, brbe_nvhe.test_brbe_nvhe_host_isolation.attr,
                        self.mock_logger
                    )
                    self.mock_logger.error.assert_called_once()
                    if kernel_changed:
                        remove.assert_not_called()
                    reboot.assert_not_called()
                self.assertEqual(scheduler.values["Result"], "PASSED")
                self.assertEqual(scheduler.values["ResultReason"], "")
                self.assertEqual(scheduler.values["Completed"], "saved")


    def test_posthook_reboot_failure_retains_ownership_and_verdict(self):
        """A failed or returning reboot logs an error and retains recovery evidence."""
        for reboot_error in (None, OSError("reboot denied")):
            with self.subTest(reboot_error=reboot_error):
                ownership = {
                    **_kernel_identity(),
                    "added_parameters": list(brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS),
                }
                scheduler = _ProgressScheduler({
                    brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: ownership,
                    "Result": "PASSED", "ResultReason": "", "Completed": "saved",
                })
                with (
                    self.mock.patch.object(
                        brbe_nvhe, "_get_running_kernel_identity",
                        return_value=(_kernel_identity(), ""),
                    ),
                    self.mock.patch.object(
                        brbe_nvhe.glt_utils, "remove_kernel_parameter",
                        return_value=(True, ""),
                    ),
                    self.mock.patch.object(
                        brbe_nvhe, "_check_nvhe_parameters",
                        return_value=(_parameter_statuses(active=True), ""),
                    ),
                    self.mock.patch.object(
                        brbe_nvhe.glt_utils, "reboot", return_value=None,
                        side_effect=reboot_error,
                    ) as reboot,
                ):
                    self.mock_logger.error.reset_mock()
                    self.assertIsNone(brbe_nvhe._restore_nvhe_boot(
                        scheduler, brbe_nvhe.test_brbe_nvhe_host_isolation.attr,
                        self.mock_logger,
                    ))
                reboot.assert_called_once_with(TEST_NAME, scheduler)
                self.mock_logger.error.assert_called_once()
                self.assertIn(
                    "Reboot manually", self.mock_logger.error.call_args[0][0]
                )
                self.assertIs(
                    scheduler.values[brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN], ownership
                )
                self.assertIn(brbe_nvhe.BRBE_NVHE_RESTORE_REBOOT_TOKEN, scheduler.values)
                self.assertEqual(scheduler.values["Result"], "PASSED")
                self.assertEqual(scheduler.values["ResultReason"], "")
                self.assertEqual(scheduler.values["Completed"], "saved")


class TestBrbeNvheMixedOwnership(GLTTestBase):
    """Verify setup and restoration preserve pre-existing user parameters."""

    def test_partial_user_configuration_appends_only_missing_parameters(self):
        """Complete active or pending user subsets without claiming them."""
        parameters = brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        for existing_count in (1, 2):
            for active in (False, True):
                with self.subTest(existing_count=existing_count, active=active):
                    states = _parameter_statuses()
                    for parameter in parameters[:existing_count]:
                        states[parameter] = _parameter_status(
                            active=active, configured=True,
                            reboot_required=not active)
                    added = list(parameters[existing_count:])
                    scheduler = _ProgressScheduler()

                    def append_parameter(parameter, parameter_states=states, **_kwargs):
                        parameter_states[parameter] = _parameter_status(
                            configured=True, owned=True, reboot_required=True)
                        return True, ""

                    with self.mock.patch.object(
                        brbe_nvhe, "_check_nvhe_parameters", return_value=(states, "")
                    ), self.mock.patch.object(
                        brbe_nvhe, "_get_active_parameter_conflict",
                        return_value=("", False)
                    ), self.mock.patch.object(
                        brbe_nvhe, "_get_running_kernel_identity",
                        return_value=(_kernel_identity(), "")
                    ), self.mock.patch.object(
                        brbe_nvhe.glt_utils, "append_kernel_parameter",
                        side_effect=append_parameter
                    ) as append, self.mock.patch.object(
                        brbe_nvhe.glt_utils, "remove_kernel_parameter",
                        return_value=(True, "")
                    ) as remove, self.mock.patch.object(
                        brbe_nvhe.glt_utils, "reboot", side_effect=SystemExit
                    ) as reboot:
                        with self.assertRaises(SystemExit):
                            brbe_nvhe._ensure_nvhe_environment(
                                TEST_NAME, scheduler, self.mock.Mock())

                    self.assertEqual(
                        [call.args[0] for call in append.call_args_list], added)
                    self.assertEqual(
                        scheduler.values[brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN],
                        {**_kernel_identity(), "added_parameters": added})
                    remove.assert_not_called()
                    reboot.assert_called_once_with(TEST_NAME, scheduler)
                    for parameter in parameters[:existing_count]:
                        self.assertTrue(states[parameter].configured)
                        self.assertFalse(states[parameter].added_by_glt)

    def test_mixed_owned_resume_preserves_user_persistent_parameter(self):
        """Clean GLT boot settings while retaining the user's persistent value."""
        parameters = brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        ownership = {
            **_kernel_identity(), "added_parameters": list(parameters[1:]),
        }
        scheduler = _ProgressScheduler({
            brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: ownership,
        })
        before = _parameter_statuses(active=True, configured=True, owned=True)
        before[parameters[0]] = _parameter_status(active=True, configured=True)
        after = _parameter_statuses(active=True)
        after[parameters[0]] = _parameter_status(active=True, configured=True)
        with self.mock.patch.object(
            brbe_nvhe, "_check_nvhe_parameters",
            side_effect=((before, ""), (after, ""))
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe, "_get_running_kernel_identity",
            return_value=(_kernel_identity(), "")
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "remove_kernel_parameter", return_value=(True, "")
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "append_kernel_parameter"
        ) as append, self.mock.patch.object(
            brbe_nvhe.glt_utils, "reboot"
        ) as reboot:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock.Mock())

        self.assertEqual((ready, reason, should_skip), (True, "", False))
        self.assertEqual(
            scheduler.values[brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN], ownership)
        append.assert_not_called()
        reboot.assert_not_called()

    def test_mixed_posthook_preserves_user_parameter_and_completed_verdict(self):
        """Only GLT additions are removed and the saved verdict is unchanged."""
        parameters = brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        for verdict, reason in (("PASSED", ""), ("FAILED", "contamination")):
            with self.subTest(verdict=verdict):
                scheduler = _ProgressScheduler({
                    brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: {
                        **_kernel_identity(), "added_parameters": list(parameters[1:]),
                    },
                    "Result": verdict, "ResultReason": reason, "Completed": "saved",
                })
                states = _parameter_statuses(active=True)
                states[parameters[0]] = _parameter_status(
                    active=True, configured=True, owned=False
                )
                with self.mock.patch.object(
                    brbe_nvhe, "_get_running_kernel_identity",
                    return_value=(_kernel_identity(), "")
                ), self.mock.patch.object(
                    brbe_nvhe, "_check_nvhe_parameters", return_value=(states, "")
                ), self.mock.patch.object(
                    brbe_nvhe.glt_utils, "remove_kernel_parameter",
                    return_value=(True, "")
                ) as remove, self.mock.patch.object(
                    brbe_nvhe.glt_utils, "reboot", side_effect=SystemExit(0)
                ) as reboot:
                    with self.assertRaises(SystemExit):
                        brbe_nvhe._restore_nvhe_boot(
                            scheduler, brbe_nvhe.test_brbe_nvhe_host_isolation.attr,
                            self.mock.Mock()
                        )
                    self.assertEqual(
                        [call.args[0] for call in remove.call_args_list],
                        list(parameters[1:])
                    )
                    reboot.assert_called_once_with(TEST_NAME, scheduler)
                self.assertEqual(scheduler.values["Result"], verdict)
                self.assertEqual(scheduler.values["ResultReason"], reason)
                self.assertEqual(scheduler.values["Completed"], "saved")
                self.assertNotIn(brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN, scheduler.values)
                self.assertTrue(states[parameters[0]].configured)

    def test_mixed_posthook_does_not_reboot_for_user_parameter_alone(self):
        """An active user parameter alone does not request a cleanup reboot."""
        parameters = brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        scheduler = _ProgressScheduler({
            brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: {
                **_kernel_identity(), "added_parameters": list(parameters[1:]),
            },
        })
        states = _parameter_statuses()
        states[parameters[0]] = _parameter_status(
            active=True, configured=True, owned=False
        )
        with self.mock.patch.object(
            brbe_nvhe, "_get_running_kernel_identity",
            return_value=(_kernel_identity(), "")
        ), self.mock.patch.object(
            brbe_nvhe, "_check_nvhe_parameters", return_value=(states, "")
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "remove_kernel_parameter", return_value=(True, "")
        ), self.mock.patch.object(brbe_nvhe.glt_utils, "reboot") as reboot:
            self.assertIsNone(
                brbe_nvhe._restore_nvhe_boot(
                    scheduler, brbe_nvhe.test_brbe_nvhe_host_isolation.attr,
                    self.mock.Mock()
                )
            )
        reboot.assert_not_called()
        self.assertEqual(scheduler.values, {})
        self.assertTrue(states[parameters[0]].configured)

    def test_mixed_posthook_logs_owned_parameter_residue(self):
        """Persistent GLT-owned residue aborts cleanup before reboot."""
        parameters = brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        for state in ({"configured": True}, {"owned": True}):
            with self.subTest(state=state):
                scheduler = _ProgressScheduler({
                    brbe_nvhe.BRBE_NVHE_OWNERSHIP_TOKEN: {
                        **_kernel_identity(), "added_parameters": list(parameters[1:]),
                    },
                })
                states = _parameter_statuses()
                states[parameters[0]] = _parameter_status(
                    active=True, configured=True, owned=False
                )
                states[parameters[1]] = _parameter_status(**state)
                with self.mock.patch.object(
                    brbe_nvhe, "_get_running_kernel_identity",
                    return_value=(_kernel_identity(), "")
                ), self.mock.patch.object(
                    brbe_nvhe, "_check_nvhe_parameters", return_value=(states, "")
                ), self.mock.patch.object(
                    brbe_nvhe.glt_utils, "remove_kernel_parameter",
                    return_value=(True, "")
                ), self.mock.patch.object(brbe_nvhe.glt_utils, "reboot") as reboot:
                    logger = self.mock.Mock()
                    brbe_nvhe._restore_nvhe_boot(
                        scheduler, brbe_nvhe.test_brbe_nvhe_host_isolation.attr,
                        logger
                    )
                    logger.error.assert_called_once()
                    self.assertIn("remain configured", logger.error.call_args[0][0])
                    reboot.assert_not_called()

    def test_partial_setup_failure_removes_only_glt_additions(self):
        """Roll back a partially failed setup without changing user settings."""
        parameters = brbe_nvhe.BRBE_NVHE_KERNEL_PARAMETERS
        states = _parameter_statuses()
        states[parameters[0]] = _parameter_status(active=True, configured=True)
        scheduler = _ProgressScheduler()

        def append_parameter(parameter, **_kwargs):
            if parameter == parameters[2]:
                return False, "injected append failure"
            states[parameter] = _parameter_status(
                configured=True, owned=True, reboot_required=True)
            return True, ""

        def remove_parameter(parameter, **_kwargs):
            if states[parameter].added_by_glt:
                states[parameter] = _parameter_status()
            return True, ""

        with self.mock.patch.object(
            brbe_nvhe, "_check_nvhe_parameters", return_value=(states, "")
        ), self.mock.patch.object(
            brbe_nvhe, "_get_active_parameter_conflict", return_value=("", False)
        ), self.mock.patch.object(
            brbe_nvhe, "_get_running_kernel_identity",
            return_value=(_kernel_identity(), "")
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "append_kernel_parameter",
            side_effect=append_parameter
        ) as append, self.mock.patch.object(
            brbe_nvhe.glt_utils, "remove_kernel_parameter",
            side_effect=remove_parameter
        ), self.mock.patch.object(
            brbe_nvhe.glt_utils, "reboot"
        ) as reboot:
            ready, reason, should_skip = brbe_nvhe._ensure_nvhe_environment(
                TEST_NAME, scheduler, self.mock.Mock())

        self.assertFalse(ready)
        self.assertFalse(should_skip)
        self.assertIn("injected append failure", reason)
        self.assertEqual(
            [call.args[0] for call in append.call_args_list], list(parameters[1:]))
        self.assertTrue(states[parameters[0]].active)
        self.assertTrue(states[parameters[0]].configured)
        self.assertFalse(states[parameters[0]].added_by_glt)
        self.assertFalse(states[parameters[1]].configured)
        self.assertEqual(scheduler.values, {})
        reboot.assert_not_called()


class _BrbeNvheResultMixin:
    """Provide one internally consistent harness result."""

    @staticmethod
    def valid_result(**updates):
        """Return valid harness evidence with optional field overrides."""
        result = {
            "schema_version": 3,
            "verdict": "fixed",
            "exit_code": 0,
            "cpu": 7,
            "iterations": brbe_nvhe.BRBE_NVHE_ITERATIONS,
            "iterations_completed": brbe_nvhe.BRBE_NVHE_ITERATIONS,
            "guest_branches": brbe_nvhe.BRBE_NVHE_GUEST_BRANCHES,
            "sample_period": brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD,
            "max_pre_guest_count": (
                brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD // 4 - 1
            ),
            "guest_samples": brbe_nvhe.BRBE_NVHE_ITERATIONS,
            "guest_windows_verified": brbe_nvhe.BRBE_NVHE_ITERATIONS,
            "min_guest_instruction_delta": brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD,
            "nvhe_cmdline": True,
            "samples": brbe_nvhe.BRBE_NVHE_ITERATIONS + 2,
            "entries": (brbe_nvhe.BRBE_NVHE_ITERATIONS + 2) * 32,
            "guest_range_start": "0x400000",
            "guest_range_end": "0x410000",
            "guest_hits": 0,
            "host_pre_hits": 2,
            "host_post_hits": 2,
            "guest_brbe_id": 0,
            "guest_brbcr_undef": True,
            "guest_unexpected_exception": False,
            "lost_records": 0,
            "throttle_records": 0,
            "context_switches": 128,
            "host_pre_range": "0x420000-0x421000",
            "host_post_range": "0x430000-0x431000",
            "reason": "",
        }
        result.update(updates)
        return result


class TestBrbeNvheResultJson(_BrbeNvheResultMixin, GLTTestBase):
    """Verify fail-closed parsing of harness evidence."""

    def test_rejects_legacy_or_inconsistent_guest_window_evidence(self):
        """Reject old results and impossible per-window witness counts."""
        cases = (
            self.valid_result(schema_version=2),
            self.valid_result(guest_samples=-1),
            self.valid_result(guest_windows_verified=-1),
            self.valid_result(min_guest_instruction_delta=-1),
            self.valid_result(guest_samples=131),
            self.valid_result(guest_samples=0, guest_windows_verified=1),
            self.valid_result(guest_windows_verified=129),
            self.valid_result(entries=4, guest_windows_verified=5),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for result in cases:
                with self.subTest(result=result):
                    path = self._write_result(temp_dir, result)
                    _, reason = brbe_nvhe._validate_result_json(path, 7)
                    self.assertTrue(reason)

    def _write_result(self, directory, result):
        path = Path(directory) / "result.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def test_accepts_complete_consistent_result(self):
        """Return typed evidence only after validating its entire schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = self.valid_result()
            path = self._write_result(temp_dir, expected)

            result, reason = brbe_nvhe._validate_result_json(path, 7)

        self.assertEqual(result, expected)
        self.assertEqual(reason, "")

    def test_rejects_missing_extra_and_wrongly_typed_fields(self):
        """Reject schema drift, incomplete data, and bool-as-int ambiguity."""
        cases = {
            "missing": (
                {
                    key: value for key, value in self.valid_result().items()
                    if key != "max_pre_guest_count"
                },
                "missing 'max_pre_guest_count'",
            ),
            "extra": (
                self.valid_result(unexpected_metric=1),
                "unknown fields: 'unexpected_metric'",
            ),
            "bool_for_count": (
                self.valid_result(max_pre_guest_count=False), "must be int"
            ),
            "integer_for_bool": (
                self.valid_result(guest_brbcr_undef=1), "must be bool"
            ),
            "non_string_reason": (
                self.valid_result(reason=None), "must be str"
            ),
            "integer_for_exception_bool": (
                self.valid_result(guest_unexpected_exception=0), "must be bool"
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, (payload, expected_reason) in cases.items():
                with self.subTest(name=name):
                    path = self._write_result(temp_dir, payload)
                    result, reason = brbe_nvhe._validate_result_json(path, 7)
                    self.assertEqual(result, {})
                    self.assertIn(expected_reason, reason)

    def test_rejects_non_object_and_invalid_json(self):
        """Require one JSON object instead of accepting partial output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            cases = (("[]", "must contain an object"), ("{", "Unable to read"))
            for payload, expected_reason in cases:
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    result, reason = brbe_nvhe._validate_result_json(path, 7)
                    self.assertEqual(result, {})
                    self.assertIn(expected_reason, reason)

    def test_rejects_identity_count_and_range_inconsistencies(self):
        """Reject evidence that cannot describe the requested harness run."""
        cases = {
            "wrong_schema": (
                self.valid_result(schema_version=1), "schema_version=1"
            ),
            "wrong_cpu": (self.valid_result(cpu=6), "expected 7"),
            "wrong_iterations": (
                self.valid_result(iterations=1),
                f"expected {brbe_nvhe.BRBE_NVHE_ITERATIONS}",
            ),
            "incomplete_iterations": (
                self.valid_result(iterations_completed=127),
                "iterations_completed=127",
            ),
            "wrong_guest_branches": (
                self.valid_result(guest_branches=1), "guest_branches=1"
            ),
            "wrong_sample_period": (
                self.valid_result(sample_period=2), "sample_period=2"
            ),
            "missing_nvhe_cmdline": (
                self.valid_result(nvhe_cmdline=False), "did not confirm"
            ),
            "negative_count": (
                self.valid_result(lost_records=-1), "cannot be negative"
            ),
            "negative_pre_guest_count": (
                self.valid_result(max_pre_guest_count=-1),
                "cannot be negative",
            ),
            "pre_guest_count_at_limit": (
                self.valid_result(
                    max_pre_guest_count=(
                        brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD // 4
                    )
                ),
                "expected less than 2500000",
            ),
            "invalid_start": (
                self.valid_result(guest_range_start="400000"),
                "is not hexadecimal",
            ),
            "empty_range": (
                self.valid_result(
                    guest_range_start="0x410000",
                    guest_range_end="0x410000",
                ),
                "empty or reversed",
            ),
            "invalid_host_range": (
                self.valid_result(host_pre_range="0x420000"),
                "start-end range",
            ),
            "reversed_host_range": (
                self.valid_result(host_post_range="0x431000-0x430000"),
                "empty or reversed",
            ),
            "hit_exceeds_entries": (
                self.valid_result(entries=1, host_post_hits=2),
                "exceeds the decoded entry count",
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, (payload, expected_reason) in cases.items():
                with self.subTest(name=name):
                    path = self._write_result(temp_dir, payload)
                    result, reason = brbe_nvhe._validate_result_json(path, 7)
                    self.assertEqual(result, {})
                    self.assertIn(expected_reason, reason)


class TestBrbeNvheClassification(_BrbeNvheResultMixin, GLTTestBase):
    """Verify the causal fixed/regressed result oracle."""

    def test_incomplete_guest_windows_cannot_pass(self):
        """Missing window proof is SKIP, unless contradicted by a fixed verdict."""
        remaining_period = (
            brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD
            - self.valid_result()["max_pre_guest_count"]
        )
        cases = (
            {"guest_samples": 0, "guest_windows_verified": 0},
            {"guest_samples": brbe_nvhe.BRBE_NVHE_ITERATIONS - 1},
            {"guest_windows_verified": brbe_nvhe.BRBE_NVHE_ITERATIONS - 1},
            {"min_guest_instruction_delta": remaining_period - 1},
        )
        for changes in cases:
            for exit_code, verdict in ((0, "fixed"), (2, "inconclusive")):
                with self.subTest(changes=changes, exit_code=exit_code):
                    result = self.valid_result(
                        **changes, exit_code=exit_code, verdict=verdict
                    )
                    reason, should_skip = brbe_nvhe._classify_harness_result(
                        result, exit_code
                    )
                    self.assertIn("lacks verified guest-window sampling", reason)
                    self.assertEqual(should_skip, exit_code == 2)

    def test_guest_contamination_is_decisive_without_window_witness(self):
        """Guest branches must fail even when window sampling was inconclusive."""
        result = self.valid_result(
            guest_hits=1, guest_samples=0, guest_windows_verified=0,
            min_guest_instruction_delta=0, exit_code=2, verdict="inconclusive",
        )
        reason, should_skip = brbe_nvhe._classify_harness_result(result, 2)
        self.assertIn("host isolation failed", reason)
        self.assertFalse(should_skip)

    def test_guest_window_threshold_and_context_switches_are_accepted(self):
        """Accept the exact counter threshold without vetoing global switches."""
        result = self.valid_result(context_switches=6)
        result["min_guest_instruction_delta"] = (
            result["sample_period"] - result["max_pre_guest_count"]
        )
        self.assertEqual(brbe_nvhe._classify_harness_result(result, 0), ("", False))

    def test_accepts_only_fixed_host_isolation_evidence(self):
        """Zero guest hits plus both host controls and guest hiding pass."""
        self.assertEqual(
            brbe_nvhe._classify_harness_result(self.valid_result(), 0)[0], ""
        )

    def test_guest_range_entries_are_contamination(self):
        """Any guest address decoded in the host branch stack is a failure."""
        reason = brbe_nvhe._classify_harness_result(
            self.valid_result(
                verdict="contaminated",
                exit_code=1,
                guest_hits=3,
                reason="guest range observed",
            ),
            1,
        )[0]

        self.assertIn("host isolation failed", reason)
        self.assertIn("3 guest-range", reason)
        self.assertIn("guest range observed", reason)

    def test_rejects_guest_visibility_control_failures(self):
        """Guest BRBE ID and BRBCR trapping prove the guest-side setup."""
        cases = (
            self.valid_result(guest_brbe_id=1),
            self.valid_result(guest_brbcr_undef=False),
            self.valid_result(guest_unexpected_exception=True),
        )
        for result in cases:
            with self.subTest(result=result):
                reason = brbe_nvhe._classify_harness_result(result, 0)[0]
                self.assertIn("prerequisites are invalid", reason)

    def test_rejects_missing_samples_entries_and_host_controls(self):
        """A quiet negative cannot be mistaken for fixed isolation."""
        cases = {
            "samples": (self.valid_result(samples=0), "no usable evidence"),
            "entries": (self.valid_result(entries=0), "no usable evidence"),
            "pre": (self.valid_result(host_pre_hits=0), "before and after"),
            "post": (self.valid_result(host_post_hits=0), "before and after"),
        }
        for name, (result, expected_reason) in cases.items():
            with self.subTest(name=name):
                reason = brbe_nvhe._classify_harness_result(result, 0)[0]
                self.assertIn(expected_reason, reason)

    def test_lost_or_throttled_records_are_inconclusive(self):
        """A clean negative requires complete, unthrottled perf evidence."""
        cases = {
            "lost": self.valid_result(
                verdict="inconclusive", exit_code=2, lost_records=1,
                reason="lost perf records",
            ),
            "throttled": self.valid_result(
                verdict="inconclusive", exit_code=2, throttle_records=1,
                reason="perf throttled",
            ),
        }
        for name, result in cases.items():
            with self.subTest(name=name):
                reason = brbe_nvhe._classify_harness_result(result, 2)[0]
                self.assertIn("inconclusive", reason)
                self.assertIn("perf evidence was incomplete", reason)

    def test_guest_contamination_remains_decisive_with_lost_records(self):
        """Observed guest entries cannot be downgraded to inconclusive."""
        result = self.valid_result(
            verdict="contaminated",
            exit_code=1,
            guest_hits=3,
            lost_records=1,
            throttle_records=1,
            reason="guest range observed",
        )

        reason = brbe_nvhe._classify_harness_result(result, 1)[0]

        self.assertIn("host isolation failed", reason)
        self.assertNotIn("perf evidence was incomplete", reason)

    def test_rejects_exit_status_and_verdict_contradictions(self):
        """Require process status, JSON status, and verdict to agree exactly."""
        cases = (
            (self.valid_result(), 2, "does not match JSON exit_code"),
            (
                self.valid_result(verdict="contaminated"),
                0,
                "does not match exit_code",
            ),
            (
                self.valid_result(exit_code=3, verdict="unknown"),
                3,
                "unsupported exit_code",
            ),
        )
        for result, returncode, expected_reason in cases:
            with self.subTest(returncode=returncode, result=result):
                reason = brbe_nvhe._classify_harness_result(result, returncode)[0]
                self.assertIn(expected_reason, reason)

    def test_rejects_nonfixed_exit_or_reason_with_clean_counts(self):
        """Do not trust clean-looking counters when the harness reports error."""
        cases = (
            (
                self.valid_result(
                    verdict="inconclusive", exit_code=2,
                    reason="KVM_RUN failed",
                ),
                2,
                "classified the result as 'inconclusive'",
            ),
            (
                self.valid_result(reason="KVM_RUN failed"),
                0,
                "failure reason",
            ),
            (self.valid_result(reason=" "), 0, "failure reason"),
        )
        for result, returncode, expected_reason in cases:
            with self.subTest(returncode=returncode, result=result):
                reason = brbe_nvhe._classify_harness_result(result, returncode)[0]
                self.assertIn(expected_reason, reason)

    def test_inconclusive_and_guest_preconditions_are_skipped(self):
        """Valid observer misses are SKIP, never false isolation failures."""
        cases = (
            {"lost_records": 1},
            {"throttle_records": 1},
            {"samples": 0, "entries": 0,
             "guest_samples": 0, "guest_windows_verified": 0, "host_pre_hits": 0, "host_post_hits": 0},
            {"entries": 0, "guest_windows_verified": 0, "host_pre_hits": 0, "host_post_hits": 0},
            {"host_pre_hits": 0},
            {"host_post_hits": 0},
            {"max_pre_guest_count": brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD // 4},
            {"guest_brbe_id": 1},
            {"guest_brbcr_undef": False},
            {"guest_unexpected_exception": True},
            {"reason": "one-shot observer did not fire"},
        )
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "inconclusive.json"
            for changes in cases:
                with self.subTest(changes=changes):
                    result = self.valid_result(
                        exit_code=2, verdict="inconclusive", **changes)
                    result_path.write_text(json.dumps(result), encoding="utf-8")
                    parsed, validation_error = brbe_nvhe._validate_result_json(
                        result_path, result["cpu"])
                    self.assertEqual(validation_error, "")
                    reason, should_skip = brbe_nvhe._classify_harness_result(parsed, 2)
                    self.assertTrue(reason)
                    self.assertTrue(should_skip)

    def test_contamination_and_protocol_errors_are_not_skipped(self):
        """Missing side controls cannot hide real contamination or bad JSON."""
        for changes in (
            {"lost_records": 1},
            {"guest_brbe_id": 1},
            {"guest_brbcr_undef": False},
            {"guest_unexpected_exception": True},
        ):
            with self.subTest(changes=changes):
                result = self.valid_result(
                    exit_code=2, verdict="inconclusive", guest_hits=32, **changes)
                reason, should_skip = brbe_nvhe._classify_harness_result(result, 2)
                self.assertIn("guest-range", reason)
                self.assertIn("d7567e9b9ba5", reason)
                self.assertFalse(should_skip)
        for result, returncode in (
            (self.valid_result(exit_code=2, verdict="inconclusive"), 0),
            (self.valid_result(exit_code=2, verdict="fixed"), 2),
            (self.valid_result(samples=0), 0),
            (self.valid_result(host_pre_hits=0), 0),
            (self.valid_result(reason="contradictory fixed result"), 0),
        ):
            with self.subTest(result=result, returncode=returncode):
                reason, should_skip = brbe_nvhe._classify_harness_result(
                    result, returncode)
                self.assertTrue(reason)
                self.assertFalse(should_skip)



class TestBrbeNvheOutputCleanup(GLTTestBase):
    """Verify idempotent harness-output cleanup."""

    def test_remove_files_deletes_outputs_and_ignores_missing_paths(self):
        """Cleanup is idempotent across interrupted and resumed runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing = Path(temp_dir) / "harness"
            missing = Path(temp_dir) / "missing"
            existing.touch()

            reason = brbe_nvhe._remove_files((existing, missing))

            self.assertEqual(reason, "")
            self.assertFalse(existing.exists())

    def test_remove_files_reports_cleanup_failure(self):
        """Surface stale executable cleanup failures in the final result."""
        path = self.mock.Mock()
        path.unlink.side_effect = OSError("read-only filesystem")

        reason = brbe_nvhe._remove_files((path,))

        self.assertIn("read-only filesystem", reason)


class TestBrbeNvheRuntimePersistence(_BrbeNvheResultMixin, GLTTestBase):
    """Verify orchestration retains evidence while removing the binary."""

    def setUp(self):
        """Supply static capabilities while testing the full harness lifecycle."""
        super().setUp()
        nvdev = self.mock.Mock()
        nvdev.is_cpu_type.return_value = True
        pmu = self.mock.MagicMock()
        pmu.is_dir.return_value = True
        pmu.__truediv__.return_value.__truediv__.return_value.read_text.return_value = "32\n"
        event_source = self.mock.Mock()
        event_source.glob.return_value = [pmu]
        patches = [
            self.mock.patch.object(brbe_nvhe, "NvDev", return_value=nvdev),
            self.mock.patch.object(
                brbe_nvhe.check_kernel_config, "parse_kconfig",
                return_value={"CONFIG_ARM64_BRBE": "y"},
            ),
            self.mock.patch.object(brbe_nvhe.perf_utils, "Path", return_value=event_source),
        ]
        for patcher in patches:
            self.addCleanup(patcher.stop)
            patcher.start()

    def test_pass_and_skip_keep_results_and_remove_harness(self):
        """Session evidence survives cleanup and the restore phase."""
        for should_skip in (False, True):
            with self.subTest(should_skip=should_skip):
                self.mock_sd.log_dir = "/session"
                self.mock_sd.progress_check.return_value = False
                valid_result = self.valid_result(
                    cpu=7,
                    exit_code=2 if should_skip else 0,
                    verdict="inconclusive" if should_skip else "fixed",
                    lost_records=int(should_skip),
                )
                remove_calls = []
                harness = Path("/session/glt_brbe_nvhe_harness")
                result_path = Path("/session/brbe_nvhe_result.json")
                entries_path = Path("/session/brbe_nvhe_entries.txt")
                makefile_path = self.mock.Mock()
                makefile_path.is_file.return_value = True
                build_result = subprocess.CompletedProcess(
                    ["make"], 0, stdout="built\n", stderr=""
                )
                harness_result = subprocess.CompletedProcess(
                    [str(harness)], 2 if should_skip else 0, stdout="fixed\n", stderr=""
                )

                def record_remove(paths, record_removal=remove_calls.append):
                    record_removal(tuple(paths))
                    return ""

                with self.mock.patch.object(
                    brbe_nvhe.glt_utils, "is_cli_available", return_value=True
                ), self.mock.patch.object(
                    brbe_nvhe, "HARNESS_MAKEFILE_PATH", makefile_path
                ), self.mock.patch.object(
                    brbe_nvhe, "_ensure_nvhe_environment", return_value=(True, "", False)
                ), self.mock.patch.object(
                    brbe_nvhe, "_get_nvhe_log_failure", return_value=""
                ), self.mock.patch.object(
                    brbe_nvhe.os, "makedirs"
                ), self.mock.patch.object(
                    brbe_nvhe, "_remove_files", side_effect=record_remove
                ), self.mock.patch.object(
                    brbe_nvhe, "_select_test_cpu", return_value=(7, "")
                ), self.mock.patch.object(
                    brbe_nvhe.glt_utils,
                    "run_command",
                    return_value=build_result,
                ) as mock_build, self.mock.patch.object(
                    brbe_nvhe.subprocess,
                    "run",
                    return_value=harness_result,
                ) as mock_run, self.mock.patch.object(
                    brbe_nvhe.Path, "is_file", return_value=True
                ), self.mock.patch.object(
                    brbe_nvhe.os, "access", return_value=True
                ), self.mock.patch.object(
                    brbe_nvhe, "_validate_result_json", return_value=(valid_result, "")
                ), self.mock.patch.object(
                    brbe_nvhe, "_restore_nvhe_boot", return_value=(True, "")
                ) as mock_restore:
                    with (
                        self.assertRaises(brbe_nvhe.GLTCore.GLTTestSkip)
                        if should_skip else nullcontext()
                    ):
                        brbe_nvhe.test_brbe_nvhe_host_isolation(
                            self.mock_sd, self.mock_logger
                        )

                self.assertEqual(
                    remove_calls,
                    [(harness, result_path, entries_path), (harness,)],
                )
                self.assertEqual(
                    mock_build.call_args,
                    self.mock.call(
                        [
                            "make", "-C", str(brbe_nvhe.HARNESS_SOURCE_DIR),
                            f"OUTPUT={harness}",
                            "KVM_MICROVM_DIR="
                            f"{brbe_nvhe.KVM_MICROVM_SOURCE_DIR}",
                        ],
                        "BRBE nVHE harness build failed",
                        timeout=brbe_nvhe.BRBE_NVHE_BUILD_TIMEOUT_SECONDS,
                    ),
                )
                self.assertEqual(
                    mock_run.call_args,
                    self.mock.call(
                        [
                            str(harness),
                            "--cpu", "7",
                            "--iterations", str(brbe_nvhe.BRBE_NVHE_ITERATIONS),
                            "--guest-branches",
                            str(brbe_nvhe.BRBE_NVHE_GUEST_BRANCHES),
                            "--sample-period",
                            str(brbe_nvhe.BRBE_NVHE_SAMPLE_PERIOD),
                            "--result-json", str(result_path),
                            "--entries", str(entries_path),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=brbe_nvhe.BRBE_NVHE_HARNESS_TIMEOUT_SECONDS,
                    ),
                )
                mock_restore.assert_not_called()


if __name__ == "__main__":
    GLTTestBase.unittest.main()
