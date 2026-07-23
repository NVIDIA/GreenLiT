"""Unit tests for contpte access-flags discriminator helpers."""
# pylint: disable=wrong-import-position,protected-access

import sys
from pathlib import Path
from unittest import mock

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.mm import glt_test_contpte


class TestContpteAccessFlags(GLTTestBase):
    """Unit coverage for contpte false-noop skip/fail gates."""

    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte.platform.machine",
        return_value="x86_64",
    )
    def test_skips_on_non_arm64(self, _machine):
        """Non-arm64 hosts must SKIP."""
        with self.assertRaises(GLTCore.GLTTestSkip):
            glt_test_contpte.test_contpte_set_access_flags_no_op(None, mock.Mock())

    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte.platform.machine",
        return_value="aarch64",
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte._read_kernel_config_text",
        return_value="CONFIG_ARM64=y\n",
    )
    def test_skips_when_contpte_config_missing(self, _kconfig, _machine):
        """Missing CONT PTE config must SKIP."""
        with self.assertRaises(GLTCore.GLTTestSkip):
            glt_test_contpte.test_contpte_set_access_flags_no_op(None, mock.Mock())

    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte.platform.machine",
        return_value="aarch64",
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte._read_kernel_config_text",
        return_value="CONFIG_ARM64_CONTPTE=y\nCONFIG_ARM64=y\n",
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte.os.path.isdir",
        return_value=True,
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte.glt_utils.is_cli_available",
        return_value=True,
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte.glt_utils.chdir_managed",
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.mm.glt_test_contpte.glt_utils.run_command",
        side_effect=RuntimeError("Cannot build contpte access-flags probe module"),
    )
    def test_skips_when_make_raises(
        self,
        run_command,
        _chdir,
        _cli,
        _isdir,
        _kconfig,
        _machine,
    ):
        """make RuntimeError is an env gap and must SKIP."""
        with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
            glt_test_contpte.test_contpte_set_access_flags_no_op(None, mock.Mock())
        self.assertIn("Cannot build contpte access-flags probe module", str(ctx.exception))
        run_command.assert_any_call(
            ["make"],
            "Cannot build contpte access-flags probe module",
            timeout=120,
        )

    def test_probe_result_struct_size(self):
        """Ioctl struct layout must stay 28 bytes for _IOR encoding."""
        self.assertEqual(
            glt_test_contpte.ctypes.sizeof(glt_test_contpte.ContpteProbeResult),
            28,
        )

    def test_evaluate_probe_false_noop_fails(self):
        """false_noop must FAIL as the regression signal."""
        probe = glt_test_contpte.ContpteProbeResult(
            set_access_ret=0,
            target_rdonly_before=1,
            slots_rdonly_after=15,
            false_noop=1,
            cont_ptes=16,
            page_size=4096,
            gather_matches_entry=1,
        )
        outcome, reason = glt_test_contpte._evaluate_probe_result(probe)
        self.assertEqual(outcome, "fail")
        self.assertIn(glt_test_contpte.CONTPTE_FIX_SHA, reason)

    def test_evaluate_probe_missing_precondition_skips(self):
        """Failed craft of target RDONLY is inconclusive SKIP."""
        probe = glt_test_contpte.ContpteProbeResult(
            set_access_ret=1,
            target_rdonly_before=0,
            slots_rdonly_after=0,
            false_noop=0,
            cont_ptes=16,
            page_size=4096,
            gather_matches_entry=0,
        )
        outcome, _reason = glt_test_contpte._evaluate_probe_result(probe)
        self.assertEqual(outcome, "skip")

    def test_evaluate_probe_gather_mismatch_skips(self):
        """Unreachable buggy no-op predicate must SKIP (no false PASS)."""
        probe = glt_test_contpte.ContpteProbeResult(
            set_access_ret=1,
            target_rdonly_before=1,
            slots_rdonly_after=0,
            false_noop=0,
            cont_ptes=16,
            page_size=4096,
            gather_matches_entry=0,
        )
        outcome, _reason = glt_test_contpte._evaluate_probe_result(probe)
        self.assertEqual(outcome, "skip")

    def test_evaluate_probe_sibling_rdonly_fails(self):
        """Leftover PTE_RDONLY on any CONT slot must FAIL."""
        probe = glt_test_contpte.ContpteProbeResult(
            set_access_ret=1,
            target_rdonly_before=1,
            slots_rdonly_after=2,
            false_noop=0,
            cont_ptes=16,
            page_size=4096,
            gather_matches_entry=1,
        )
        outcome, _reason = glt_test_contpte._evaluate_probe_result(probe)
        self.assertEqual(outcome, "fail")

    def test_evaluate_probe_fixed_passes(self):
        """Fixed outcome (ret=1, no RDONLY leftover) does not skip or fail."""
        probe = glt_test_contpte.ContpteProbeResult(
            set_access_ret=1,
            target_rdonly_before=1,
            slots_rdonly_after=0,
            false_noop=0,
            cont_ptes=16,
            page_size=4096,
            gather_matches_entry=1,
        )
        outcome, reason = glt_test_contpte._evaluate_probe_result(probe)
        self.assertIsNone(outcome)
        self.assertIsNone(reason)
