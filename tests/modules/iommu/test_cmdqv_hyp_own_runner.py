"""Unit tests for CMDQV hyp_own runner helpers."""

# pylint: disable=protected-access

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.iommu import cmdqv_hyp_own_runner as runner


class TestEvaluateResult(GLTTestBase):
    """Verdict core for the dual-function-probe COUNT discriminator."""

    def test_fails_when_guest_supports_cmd_called(self):
        """guest_supports_cmd > 0 with get_cmdq_nonnull == 0 is the textbook unpatched case."""
        result = runner.CmdqvCounts(get_cmdq_nonnull=0, guest_supports_cmd=323)
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.evaluate_result(result)
        self.assertIn("tegra241_guest_vcmdq_supports_cmd", str(context.exception))
        self.assertIn("323", str(context.exception))

    def test_fails_when_both_counts_are_nonzero(self):
        """Any scoped guest callback on the pre-gated bare-metal host fails."""
        result = runner.CmdqvCounts(get_cmdq_nonnull=5, guest_supports_cmd=1)
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.evaluate_result(result)
        self.assertIn("tegra241_guest_vcmdq_supports_cmd", str(context.exception))

    def test_passes_on_nonzero_get_cmdq_nonnull_and_zero_guest_supports_cmd(self):
        """get_cmdq_nonnull > 0 with guest_supports_cmd == 0 is a fixed kernel."""
        result = runner.CmdqvCounts(get_cmdq_nonnull=321, guest_supports_cmd=0)
        runner.evaluate_result(result)

    def test_liveness_not_met_when_both_counts_zero(self):
        """Both counts staying 0 is ambiguous and retryable, not a verdict."""
        result = runner.CmdqvCounts(get_cmdq_nonnull=0, guest_supports_cmd=0)
        with self.assertRaises(runner._CandidateNotLive) as context:
            runner.evaluate_result(result)
        self.assertIn("both get_cmdq_nonnull and guest_supports_cmd are 0", str(context.exception))

    def test_liveness_not_met_when_no_counts_observed(self):
        """No CMDQV_COUNTS line at all is a retryable liveness gap, not a verdict."""
        with self.assertRaises(runner._CandidateNotLive) as context:
            runner.evaluate_result(None)
        self.assertIn("no CMDQV_COUNTS observed", str(context.exception))

class TestObserveTemplateContract(GLTTestBase):
    """Pin observe.bt.in to the runner parser and DISP-005 scoping predicate."""

    def test_printf_format_matches_counts_re(self):
        """The observer output format stays parseable by the runner."""
        with open(runner.OBSERVE_TEMPLATE, encoding="utf-8") as handle:
            template = handle.read()
        self.assertIn("CMDQV_COUNTS get_cmdq_calls=%d", template)
        self.assertIn("get_cmdq_nonnull=%d", template)
        self.assertIn("guest_supports_cmd=%d", template)
        self.assertRegex(
            "CMDQV_COUNTS get_cmdq_calls=1 get_cmdq_nonnull=2 guest_supports_cmd=3",
            runner.COUNTS_RE,
        )

    def test_guest_probe_requires_in_scope_filter(self):
        """Guest callback counts are scoped to the selected command path."""
        with open(runner.OBSERVE_TEMPLATE, encoding="utf-8") as handle:
            template = handle.read()
        self.assertIn("kprobe:tegra241_guest_vcmdq_supports_cmd", template)
        self.assertIn("/@in_scope[tid]/", template)


class TestParseCmdqvCounts(GLTTestBase):
    """Parse the CMDQV_COUNTS line emitted by the dual-function-probe observer.

    Format is ``CMDQV_COUNTS get_cmdq_calls=N get_cmdq_nonnull=N
    guest_supports_cmd=N`` (get_cmdq_calls added as a diagnostic-only field
    -- see observe.bt.in's END{} block and CmdqvCounts' docstring).
    """

    def test_parses_fixed_pattern(self):
        """Non-zero get_cmdq_nonnull with zero guest_supports_cmd parses cleanly."""
        output = (
            "PROBE_READY\n"
            "CMDQV_COUNTS get_cmdq_calls=321 get_cmdq_nonnull=321 guest_supports_cmd=0\n"
        )
        result = runner.parse_cmdqv_counts(output)
        self.assertIsNotNone(result)
        self.assertEqual(result.get_cmdq_calls, 321)
        self.assertEqual(result.get_cmdq_nonnull, 321)
        self.assertEqual(result.guest_supports_cmd, 0)

    def test_parses_unfixed_pattern(self):
        """The unfixed observer pattern preserves every discriminator count."""
        output = (
            "CMDQV_COUNTS get_cmdq_calls=323 get_cmdq_nonnull=0 "
            "guest_supports_cmd=321\n"
        )
        result = runner.parse_cmdqv_counts(output)
        self.assertEqual(result.get_cmdq_calls, 323)
        self.assertEqual(result.get_cmdq_nonnull, 0)
        self.assertEqual(result.guest_supports_cmd, 321)

    def test_returns_none_for_missing_counts_line(self):
        """No CMDQV_COUNTS line anywhere in output returns None."""
        self.assertIsNone(runner.parse_cmdqv_counts("PROBE_READY\nsome noise\n"))

    def test_rejects_malformed_counts_line(self):
        """A line that only partially matches the expected format is rejected."""
        self.assertIsNone(
            runner.parse_cmdqv_counts("CMDQV_COUNTS get_cmdq_nonnull=1\n")
        )

    def test_uses_last_valid_counts_line(self):
        """The last complete line wins when observer output contains retries."""
        output = (
            "CMDQV_COUNTS get_cmdq_calls=1 get_cmdq_nonnull=1 guest_supports_cmd=0\n"
            "CMDQV_COUNTS get_cmdq_calls=5 get_cmdq_nonnull=0 guest_supports_cmd=4\n"
        )
        result = runner.parse_cmdqv_counts(output)
        self.assertEqual(result.get_cmdq_calls, 5)
        self.assertEqual(result.get_cmdq_nonnull, 0)
        self.assertEqual(result.guest_supports_cmd, 4)


class TestKallsymsHas(GLTTestBase):
    """Resolve kernel symbols from /proc/kallsyms."""

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.open",
        new_callable=GLTTestBase.unittest.mock.mock_open,
        read_data="ffffffff80123456 t tegra241_cmdqv_get_cmdq\t[arm_smmu_v3]\n",
    )
    def test_matches_module_qualified_symbol(self, _mock_open):
        """Module-suffixed kallsyms lines still match the bare symbol name."""
        self.assertTrue(runner.kallsyms_has("tegra241_cmdqv_get_cmdq"))


class TestCheckPreconditionsOrdering(GLTTestBase):
    """Applicability SKIP must precede capability FAIL checks."""

    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.geteuid")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.shutil.which")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.platform")
    def test_non_aarch64_skips_before_root_or_bpftrace(
        self, mock_platform, mock_which, mock_geteuid
    ):
        """Non-aarch64 hosts skip before root or bpftrace capability checks."""
        mock_platform.machine.return_value = "x86_64"
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            runner.check_preconditions(self.mock_logger)
        self.assertIn("aarch64", str(context.exception))
        mock_geteuid.assert_not_called()
        mock_which.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.geteuid",
                                      return_value=1000)
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.platform")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.acpi_cmdqv_present")
    def test_non_root_skips_not_fails(self, mock_acpi, mock_platform, _geteuid):
        """Not running as root is environment-unsuitable, not a regression (DISP-001)."""
        mock_platform.machine.return_value = "aarch64"
        mock_acpi.return_value = True
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            runner.check_preconditions(self.mock_logger)
        self.assertIn("root", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.shutil.which", return_value=None)
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.geteuid",
                                      return_value=0)
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.platform")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.acpi_cmdqv_present")
    def test_missing_bpftrace_skips_not_fails(
        self, mock_acpi, mock_platform, _geteuid, _which
    ):
        """Missing bpftrace is environment-unsuitable, not a regression (DISP-001)."""
        mock_platform.machine.return_value = "aarch64"
        mock_acpi.return_value = True
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            runner.check_preconditions(self.mock_logger)
        self.assertIn("bpftrace", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.kallsyms_has", return_value=False)
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_global_iommu_translated")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.running_as_guest", return_value=False)
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_btf_available")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.cmdqv_admin_disabled_reason",
        return_value="")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.os.path.isfile", return_value=True)
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.geteuid",
                                      return_value=0)
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.shutil.which", return_value="/usr/bin/bpftrace")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.platform")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.acpi_cmdqv_present")
    def test_fails_on_absent_tegra241_liveness_symbol(
        self, mock_acpi, mock_platform, _which, _geteuid, _isfile, _disabled,
        _btf, _guest, _translated, _kallsyms
    ):
        """A missing tegra241_cmdqv_get_cmdq symbol on a confirmed NVDA200C platform
        is a hard FAIL: without it the discriminator cannot establish the get_cmdq path."""
        mock_platform.machine.return_value = "aarch64"
        mock_acpi.return_value = True
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.check_preconditions(self.mock_logger)
        self.assertIn("tegra241_cmdqv_get_cmdq", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.kallsyms_has",
        side_effect=lambda symbol: symbol == runner.SYMBOL_LIVENESS)
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_global_iommu_translated")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.running_as_guest", return_value=False)
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_btf_available")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.cmdqv_admin_disabled_reason",
        return_value="")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.os.path.isfile", return_value=True)
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.geteuid",
                                      return_value=0)
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.shutil.which", return_value="/usr/bin/bpftrace")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.platform")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.acpi_cmdqv_present")
    def test_fails_on_absent_guest_supports_cmd_symbol(
        self, mock_acpi, mock_platform, _which, _geteuid, _isfile, _disabled,
        _btf, _guest, _translated, _kallsyms
    ):
        """The second (dual-probe) symbol is validated too, not just SYMBOL_LIVENESS.

        If tegra241_guest_vcmdq_supports_cmd() were silently absent, the
        discriminator's FAIL branch could never fire and an unpatched
        kernel could read as a false PASS -- so this must FAIL up front,
        the same as a missing SYMBOL_LIVENESS.
        """
        mock_platform.machine.return_value = "aarch64"
        mock_acpi.return_value = True
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.check_preconditions(self.mock_logger)
        self.assertIn("tegra241_guest_vcmdq_supports_cmd", str(context.exception))

class TestWaitForProbeReady(GLTTestBase):
    """The real observer launch is the single attach-readiness check."""

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.time.monotonic",
        side_effect=[0.0, 0.1, 0.2],
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.select.select",
        return_value=([3], [], []),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.os.read",
        side_effect=[b"Attaching 4 probes...\nPROBE_", b"READY\n"],
    )
    def test_returns_combined_output_after_split_ready_marker(
        self, _mock_read, _mock_select, _mock_mono
    ):
        """Readiness detection works when the marker spans pipe reads."""
        mock_proc = GLTTestBase.unittest.mock.MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.fileno.return_value = 3
        output = runner.wait_for_probe_ready(mock_proc, timeout_sec=30)
        self.assertEqual(output, "Attaching 4 probes...\nPROBE_READY\n")

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.time.monotonic",
        side_effect=[0.0, 2.0],
    )
    def test_timeout_reports_configured_bound(self, _mock_mono):
        """Timeout failures identify the configured readiness bound."""
        mock_proc = GLTTestBase.unittest.mock.MagicMock()
        mock_proc.stdout.fileno.return_value = 3
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.wait_for_probe_ready(mock_proc, timeout_sec=1)
        self.assertIn("within 1s", str(context.exception))

    def test_process_exit_with_btf_error_skips(self):
        """Known kernel BTF/type gaps are environment skips with full output."""
        for marker in (
            "Unknown struct/union",
            "unknown type",
            "No BTF found",
            "failed to resolve",
        ):
            with self.subTest(marker=marker):
                mock_proc = GLTTestBase.unittest.mock.MagicMock()
                mock_proc.stdout.fileno.return_value = 3
                mock_proc.poll.return_value = 1
                output = f"observe.bt:36:12: ERROR: {marker}: detail\n"
                mock_proc.communicate.return_value = (output.encode(), None)
                mock_proc.returncode = 1
                with self.assertRaises(GLTCore.GLTTestSkip) as context:
                    runner.wait_for_probe_ready(mock_proc, timeout_sec=30)
                self.assertIn(output.strip(), str(context.exception))

    def test_process_exit_with_attach_error_fails(self):
        """Generic attach errors are hard failures with full output."""
        mock_proc = GLTTestBase.unittest.mock.MagicMock()
        mock_proc.stdout.fileno.return_value = 3
        mock_proc.poll.return_value = 1
        output = "ERROR: failed to attach kprobe\n"
        mock_proc.communicate.return_value = (output.encode(), None)
        mock_proc.returncode = 1
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.wait_for_probe_ready(mock_proc, timeout_sec=30)
        self.assertIn(output.strip(), str(context.exception))


class TestResolveSmmuIdentityForBdf(GLTTestBase):
    """SMMU platform name / MMIO node resolution across sysfs layouts.

    The ``iommu`` symlink resolves to the SMMU MMIO node at different depths
    depending on the kernel, so resolution must walk upward to the
    ``arm-smmu-v3.*`` platform component rather than count a fixed number of
    path hops.
    """

    # Real Tegra241 layout nests the MMIO node under an intermediate iommu/
    # directory; the reviewer's assumed layout does not. Both must resolve.
    _NESTED = (
        "/sys/devices/platform/bus@0/9000000.arm-smmu-v3/"
        "arm-smmu-v3.0.auto/iommu/smmu3.0x000000000d000000"
    )
    _FLAT = "/sys/devices/platform/arm-smmu-v3.1/smmu3.0x00000000ff400000"

    def _resolve(self, iommu_path):
        with GLTTestBase.unittest.mock.patch(
            "src.modules.iommu.cmdqv_hyp_own_runner.os.path.exists",
            return_value=True,
        ), GLTTestBase.unittest.mock.patch(
            "src.modules.iommu.cmdqv_hyp_own_runner.os.path.realpath",
            return_value=iommu_path,
        ):
            return runner.resolve_smmu_identity_for_bdf("0000:01:00.0")

    def test_resolves_nested_iommu_dir_layout(self):
        """`.../arm-smmu-v3.N.auto/iommu/smmu3.0x*` resolves correctly."""
        self.assertEqual(
            self._resolve(self._NESTED),
            ("arm-smmu-v3.0.auto", "smmu3.0x000000000d000000"),
        )

    def test_resolves_flat_layout(self):
        """`.../arm-smmu-v3.N/smmu3.0x*` (no intermediate dir) also resolves."""
        self.assertEqual(
            self._resolve(self._FLAT),
            ("arm-smmu-v3.1", "smmu3.0x00000000ff400000"),
        )

    def test_resolves_dt_only_iommu_node_via_driver_symlink(self):
        """DT nodes named <reg>.iommu resolve via driver=arm-smmu-v3 identity."""
        platform = "/sys/devices/platform/bus@0/9000000.iommu"
        iommu = f"{platform}/iommu/smmu3.0x9000000"

        def _exists(_path):
            return True

        def _islink(path):
            return path == f"{platform}/driver"

        def _realpath(path):
            if path == f"{platform}/driver":
                return "/sys/bus/platform/drivers/arm-smmu-v3"
            if path.endswith("/iommu") or path == iommu:
                return iommu
            return path

        with GLTTestBase.unittest.mock.patch(
            "src.modules.iommu.cmdqv_hyp_own_runner.os.path.exists",
            side_effect=_exists,
        ), GLTTestBase.unittest.mock.patch(
            "src.modules.iommu.cmdqv_hyp_own_runner.os.path.islink",
            side_effect=_islink,
        ), GLTTestBase.unittest.mock.patch(
            "src.modules.iommu.cmdqv_hyp_own_runner.os.path.realpath",
            side_effect=_realpath,
        ):
            platform_name, mmio = runner.resolve_smmu_identity_for_bdf("0000:01:00.0")
        self.assertEqual(platform_name, "9000000.iommu")
        self.assertEqual(mmio, "smmu3.0x9000000")

    def test_missing_iommu_symlink_fails(self):
        """A device without an iommu symlink is a hard fail."""
        with GLTTestBase.unittest.mock.patch(
            "src.modules.iommu.cmdqv_hyp_own_runner.os.path.exists",
            return_value=False,
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                runner.resolve_smmu_identity_for_bdf("0000:01:00.0")
        self.assertIn("no iommu symlink", str(ctx.exception))

    def test_non_arm_smmu_v3_fails(self):
        """An SMMU that is not arm-smmu-v3 fails rather than misreporting."""
        path = "/sys/devices/platform/other-smmu.0/iommu/smmu3.0x1000"
        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            self._resolve(path)
        self.assertIn("not arm-smmu-v3", str(ctx.exception))

    def test_unexpected_mmio_node_fails(self):
        """A non-smmu3.0x MMIO leaf fails even when the platform matches."""
        path = "/sys/devices/platform/arm-smmu-v3.0/unexpected-node"
        with self.assertRaises(GLTCore.GLTTestFail) as ctx:
            self._resolve(path)
        self.assertIn("unexpected SMMU MMIO node", str(ctx.exception))

class TestRunTriggerDma(GLTTestBase):
    """Pure-Python O_DIRECT read trigger (ported from trigger_dma.c)."""

    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.mmap.mmap")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.readv")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.close")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.open")
    def test_raises_when_too_few_reads_succeed(
        self, mock_open, _mock_close, mock_readv, mock_mmap
    ):
        """Fewer than min_reads successful reads is a hard failure."""
        mock_open.return_value = 7
        mock_readv.return_value = 4096
        mock_mmap.return_value.__enter__.return_value = GLTTestBase.unittest.mock.MagicMock()
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.run_trigger_dma("/dev/nvme0n1", num_reads=2, min_reads=5, max_sec=0)
        self.assertIn("successful reads", str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.mmap.mmap")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.readv")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.close")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.open")
    def test_passes_when_min_reads_reached(
        self, mock_open, mock_close, mock_readv, mock_mmap
    ):
        """Reaching min_reads within num_reads succeeds and closes the fd."""
        mock_open.return_value = 7
        mock_readv.return_value = 4096
        mock_mmap.return_value.__enter__.return_value = GLTTestBase.unittest.mock.MagicMock()
        runner.run_trigger_dma("/dev/nvme0n1", num_reads=10, min_reads=5, max_sec=0)
        mock_close.assert_called_once_with(7)

    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.os.open")
    def test_raises_when_device_cannot_be_opened(self, mock_open):
        """An OSError opening the device is surfaced as GLTCore.GLTTestFail."""
        mock_open.side_effect = OSError("no such device")
        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.run_trigger_dma("/dev/nvme99n1", num_reads=1, min_reads=1, max_sec=0)
        self.assertIn("open", str(context.exception))


class TestRunCmdqvHypOwnTestVerdictWiring(GLTTestBase):
    """End-to-end wiring: run_cmdqv_hyp_own_test applies evaluate_result per candidate."""

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:01:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        return_value=set())
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_pass_result_returns_without_raising(
        self, _preconditions, _collect, _candidates, mock_run_observed
    ):
        """A fixed-kernel CMDQV_COUNTS on the first candidate is a clean pass."""
        mock_run_observed.return_value = runner.CmdqvCounts(
            get_cmdq_nonnull=321, guest_supports_cmd=0
        )
        runner.run_cmdqv_hyp_own_test(self.mock_logger)

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:01:00.0", "0000:02:00.0", "0000:00:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        return_value={"0000:01:00.0"})
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_enumerates_once_and_prefers_bdf_ordered_non_root_candidate(
        self, _preconditions, mock_root, mock_enumerate, mock_run_observed
    ):
        """Automatic discovery scans once and tries non-root NVMe first."""
        mock_run_observed.return_value = runner.CmdqvCounts(
            get_cmdq_nonnull=1, guest_supports_cmd=0
        )
        runner.run_cmdqv_hyp_own_test(self.mock_logger)
        mock_enumerate.assert_called_once_with()
        mock_root.assert_called_once_with("/")
        self.assertEqual(mock_run_observed.call_args.args[0], "0000:00:00.0")

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger",
        return_value=runner.CmdqvCounts(get_cmdq_nonnull=1, guest_supports_cmd=0))
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:02:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        return_value={"0000:02:00.0"})
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_root_only_candidate_remains_usable(
        self, _preconditions, mock_root, mock_enumerate, mock_run_observed
    ):
        """Read-only triggering falls back to the root NVMe when it is the only choice."""
        runner.run_cmdqv_hyp_own_test(self.mock_logger)
        mock_enumerate.assert_called_once_with()
        mock_root.assert_called_once_with("/")
        self.assertEqual(mock_run_observed.call_args.args[0], "0000:02:00.0")

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger",
        return_value=runner.CmdqvCounts(get_cmdq_nonnull=1, guest_supports_cmd=0))
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:02:00.0", "0000:01:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        side_effect=OSError("unreadable"))
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_root_discovery_error_fails_open_with_stable_bdf_order(
        self, _preconditions, mock_root, mock_enumerate, mock_run_observed
    ):
        """Unreadable mount data does not block the read-only trigger."""
        runner.run_cmdqv_hyp_own_test(self.mock_logger)
        mock_enumerate.assert_called_once_with()
        mock_root.assert_called_once_with("/")
        self.assertEqual(mock_run_observed.call_args.args[0], "0000:01:00.0")

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:01:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        return_value=set())
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_fail_result_propagates_as_fail(
        self, _preconditions, _collect, _candidates, mock_run_observed
    ):
        """An unpatched CMDQV_COUNTS (guest_supports_cmd > 0) propagates as GLTCore.GLTTestFail."""
        mock_run_observed.return_value = runner.CmdqvCounts(
            get_cmdq_nonnull=0, guest_supports_cmd=323
        )
        with self.assertRaises(GLTCore.GLTTestFail):
            runner.run_cmdqv_hyp_own_test(self.mock_logger)

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:01:00.0", "0000:02:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        return_value=set())
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_no_result_on_all_candidates_raises_skip(
        self, _preconditions, _collect, _candidates, mock_run_observed
    ):
        """No CMDQV_COUNTS on any candidate exhausts retries into a SKIP, not a FAIL (DISP-001)."""
        mock_run_observed.return_value = None
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            runner.run_cmdqv_hyp_own_test(self.mock_logger)
        self.assertEqual(mock_run_observed.call_count, 2)
        self.assertIn("no NVMe could drive", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:01:00.0", "0000:02:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        return_value=set())
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_both_zero_on_all_candidates_raises_skip(
        self, _preconditions, _collect, _candidates, mock_run_observed
    ):
        """Both counts staying 0 on every candidate exhausts retries into a SKIP (DISP-001)."""
        mock_run_observed.return_value = runner.CmdqvCounts(
            get_cmdq_nonnull=0, guest_supports_cmd=0
        )
        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            runner.run_cmdqv_hyp_own_test(self.mock_logger)
        self.assertEqual(mock_run_observed.call_count, 2)
        self.assertIn("no NVMe could drive", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_observed_trigger")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.enumerate_nvme_candidates",
        return_value=["0000:01:00.0", "0000:02:00.0"])
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.pci_utils.pci_bdfs_backing_mount",
        return_value=set())
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.check_preconditions")
    def test_liveness_gap_retries_next_candidate(
        self, _preconditions, _collect, _candidates, mock_run_observed
    ):
        """A no-result first candidate is retried; a passing second candidate returns clean."""
        mock_run_observed.side_effect = [
            None,
            runner.CmdqvCounts(get_cmdq_nonnull=321, guest_supports_cmd=0),
        ]
        runner.run_cmdqv_hyp_own_test(self.mock_logger)
        self.assertEqual(mock_run_observed.call_count, 2)


class TestRunObservedTriggerFailureMapping(GLTTestBase):
    """Candidate-local trigger I/O failures are retryable; attach failures are not (DISP-007)."""

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner._kill_bpftrace_proc")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_trigger_dma")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.wait_for_probe_ready",
        return_value="PROBE_READY\n")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.subprocess.Popen")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.write_observe_script")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.resolve_smmu_identity_for_bdf",
        return_value=("arm-smmu-v3.0", "smmu3.0x1000000"))
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.resolve_nvme_block_from_bdf",
        return_value="/dev/nvme0n1")
    def test_unexpected_post_ready_exit_cannot_pass_from_end_counts(
        self, _resolve_block, _resolve_smmu, _write_script, mock_popen,
        _wait_ready, _trigger, mock_kill,
    ):
        """An early runtime failure cannot turn unconditional END counts into PASS."""
        output = (
            b"ERROR: probe runtime failure\n"
            b"CMDQV_COUNTS get_cmdq_calls=1 get_cmdq_nonnull=1 "
            b"guest_supports_cmd=0\n"
        )
        mock_proc = GLTTestBase.unittest.mock.MagicMock(returncode=1)
        mock_proc.communicate.return_value = (output, None)
        mock_proc.poll.return_value = 1
        mock_popen.return_value = mock_proc

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            runner.run_observed_trigger("0000:01:00.0", self.mock_logger)
        self.assertIn("exited before the observation window ended", str(context.exception))
        self.assertIn("ERROR: probe runtime failure", str(context.exception))
        mock_kill.assert_not_called()

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner._kill_bpftrace_proc")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.run_trigger_dma",
        side_effect=GLTCore.GLTTestFail(
            "trigger_dma: open /dev/nvme0n1 failed: boom"))
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.wait_for_probe_ready",
        return_value="PROBE_READY\n")
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.subprocess.Popen")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.write_observe_script")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.resolve_smmu_identity_for_bdf",
        return_value=("arm-smmu-v3.0", "smmu3.0x1000000"))
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.resolve_nvme_block_from_bdf",
        return_value="/dev/nvme0n1")
    def test_trigger_dma_failure_is_mapped_to_liveness_not_met(
        self, _resolve_block, _resolve_smmu, _write_script, mock_popen,
        _wait_ready, _trigger, mock_kill,
    ):
        """A candidate-local trigger I/O failure is retryable, not a hard FAIL.

        Otherwise a single flaky NVMe candidate would abort the whole test
        even when a later candidate could still drive the CMDQV path.
        """
        mock_proc = GLTTestBase.unittest.mock.MagicMock()
        # Still running when the exception fires; reaped after killpg helper.
        mock_proc.poll.side_effect = [None, -9]
        mock_popen.return_value = mock_proc

        def _reap(_proc):
            mock_proc.poll.side_effect = None
            mock_proc.poll.return_value = -9

        mock_kill.side_effect = _reap
        with self.assertRaises(runner._CandidateNotLive):
            runner.run_observed_trigger("0000:01:00.0", self.mock_logger)
        self.assertIs(
            mock_popen.call_args.kwargs["stderr"], runner.subprocess.STDOUT
        )
        mock_kill.assert_called_once_with(mock_proc)

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner._kill_bpftrace_proc")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.wait_for_probe_ready",
        side_effect=GLTCore.GLTTestFail(
            "bpftrace observer exited before PROBE_READY (rc=1)"))
    @GLTTestBase.unittest.mock.patch("src.modules.iommu.cmdqv_hyp_own_runner.subprocess.Popen")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.write_observe_script")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.resolve_smmu_identity_for_bdf",
        return_value=("arm-smmu-v3.0", "smmu3.0x1000000"))
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.resolve_nvme_block_from_bdf",
        return_value="/dev/nvme0n1")
    def test_probe_attach_failure_stays_a_hard_fail(
        self, _resolve_block, _resolve_smmu, _write_script, mock_popen, _wait_ready,
        mock_kill,
    ):
        """A bpftrace attach failure is NOT candidate-local and must still hard-FAIL.

        Attach failures (e.g. a symbol that stopped resolving) affect every
        candidate equally, unlike a single flaky NVMe device -- these must
        not be swallowed into the retry path.
        """
        mock_proc = GLTTestBase.unittest.mock.MagicMock()
        mock_proc.poll.side_effect = [None, -9]
        mock_popen.return_value = mock_proc

        def _reap(_proc):
            mock_proc.poll.side_effect = None
            mock_proc.poll.return_value = -9

        mock_kill.side_effect = _reap
        with self.assertRaises(GLTCore.GLTTestFail):
            runner.run_observed_trigger("0000:01:00.0", self.mock_logger)
        mock_kill.assert_called_once_with(mock_proc)


class TestBpftraceCleanup(GLTTestBase):
    """Observer cleanup keeps process isolation and bounded local reaping."""

    def test_set_parent_death_signal_does_not_raise(self):
        """Best-effort PDEATHSIG setup never raises, even off-Linux (no libc.so.6)."""
        runner._set_parent_death_signal()  # pylint: disable=protected-access

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.os.killpg"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.os.getpgid", return_value=4242
    )
    def test_kill_helper_escalates_and_bounds_both_reaps(
        self, _getpgid, mock_killpg
    ):
        """A TERM timeout escalates to KILL with another bounded reap."""
        teardown = runner.BPFTRACE_TEARDOWN_TIMEOUT_SEC
        mock_proc = GLTTestBase.unittest.mock.MagicMock(pid=4242)
        mock_proc.communicate.side_effect = [
            runner.subprocess.TimeoutExpired("bpftrace", teardown),
            (b"", None),
        ]
        runner._kill_bpftrace_proc(  # pylint: disable=protected-access
            mock_proc
        )
        self.assertEqual(
            mock_killpg.call_args_list,
            [
                GLTTestBase.unittest.mock.call(4242, runner.signal.SIGTERM),
                GLTTestBase.unittest.mock.call(4242, runner.signal.SIGKILL),
            ],
        )
        self.assertEqual(
            mock_proc.communicate.call_args_list,
            [
                GLTTestBase.unittest.mock.call(timeout=teardown),
                GLTTestBase.unittest.mock.call(timeout=teardown),
            ],
        )


class TestRunningAsGuest(GLTTestBase):
    """Host-only gate must not treat containers as guests."""

    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.glt_utils.is_cli_available",
        return_value=True,
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.iommu.cmdqv_hyp_own_runner.subprocess.run")
    def test_running_as_guest_uses_vm_flag(self, mock_run, _avail):
        """systemd-detect-virt must be invoked with --vm (not containers)."""
        mock_run.return_value = GLTTestBase.unittest.mock.MagicMock(returncode=1)
        self.assertFalse(runner.running_as_guest())
        args = mock_run.call_args[0][0]
        self.assertEqual(args[:3], ["systemd-detect-virt", "--quiet", "--vm"])
