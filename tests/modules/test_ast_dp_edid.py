"""Unit tests for the AST DisplayPort EDID msleep discriminator."""
# pylint: disable=wrong-import-position,protected-access

import sys
from pathlib import Path
from types import SimpleNamespace

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.gpu import glt_test_ast_dp_edid as ast_edid


_FIXED = """\
0000000000011be8 <ast_astdp_read_edid_block>:
   11be8:	d503201f 	nop
   11d84:	94000000 	bl	0 <msleep>
			11d84: R_AARCH64_CALL26	msleep
   11d88:	aa0003e0 	mov	x0, x0
0000000000011f40 <ast_astdp_is_connected>:
   11f40:	94000000 	bl	0 <msleep>
			11f40: R_AARCH64_CALL26	msleep
"""

_FIXED_RELOC_ONLY = """\
0000000000011be8 <ast_astdp_read_edid_block>:
   11d84:	94000000 	bl	0 <ast_astdp_read_edid_block+0x19c>
			11d84: R_AARCH64_CALL26	msleep
0000000000011f40 <ast_astdp_is_connected>:
			11f40: R_AARCH64_CALL26	msleep
"""

_UNFIXED = """\
0000000000011be8 <ast_astdp_read_edid_block>:
   11be8:	d503201f 	nop
   11d84:	94000000 	bl	0 <__const_udelay>
			11d84: R_AARCH64_CALL26	__const_udelay
0000000000011f40 <ast_astdp_is_connected>:
   11f40:	94000000 	bl	0 <msleep>
			11f40: R_AARCH64_CALL26	msleep
"""

_MIXED = """\
0000000000011be8 <ast_astdp_read_edid_block>:
   11d84:	94000000 	bl	0 <msleep>
			11d84: R_AARCH64_CALL26	msleep
   11d90:	94000000 	bl	0 <__const_udelay>
			11d90: R_AARCH64_CALL26	__const_udelay
0000000000011f40 <ast_astdp_is_connected>:
"""


class TestAstDpEdidOracle(GLTTestBase):
    """Parse one-function objdump bodies without later-symbol contamination."""

    def test_extracts_target_function_only(self):
        """Later msleep callers are outside the EDID reader."""
        body = ast_edid.extract_function_disassembly(
            _FIXED, ast_edid.AST_DP_EDID_FUNCTION
        )
        self.assertIn("<msleep>", body)
        self.assertNotIn("ast_astdp_is_connected", body)

    def test_fixed_function_calls_msleep(self):
        """A bl to msleep in the EDID reader is the fixed invariant."""
        body = ast_edid.extract_function_disassembly(
            _FIXED, ast_edid.AST_DP_EDID_FUNCTION
        )
        self.assertIsNotNone(ast_edid._MSLEEP_RE.search(body))
        self.assertIsNone(ast_edid._BUSY_WAIT_RE.search(body))

    def test_reloc_only_msleep_is_fixed(self):
        """Relocatable .ko may name the call as R_* msleep, not <msleep>."""
        body = ast_edid.extract_function_disassembly(
            _FIXED_RELOC_ONLY, ast_edid.AST_DP_EDID_FUNCTION
        )
        self.assertIsNotNone(ast_edid._MSLEEP_RE.search(body))
        self.assertIsNone(ast_edid._BUSY_WAIT_RE.search(body))

    def test_unfixed_function_does_not_call_msleep(self):
        """Busy-wait udelay in the EDID reader is unfixed even if later code sleeps."""
        body = ast_edid.extract_function_disassembly(
            _UNFIXED, ast_edid.AST_DP_EDID_FUNCTION
        )
        self.assertIsNone(ast_edid._MSLEEP_RE.search(body))
        self.assertIsNotNone(ast_edid._BUSY_WAIT_RE.search(body))

    def test_mixed_msleep_and_busy_wait_is_unfixed(self):
        """Residual busy-wait in the same function must not PASS."""
        body = ast_edid.extract_function_disassembly(
            _MIXED, ast_edid.AST_DP_EDID_FUNCTION
        )
        self.assertIsNotNone(ast_edid._MSLEEP_RE.search(body))
        self.assertIsNotNone(ast_edid._BUSY_WAIT_RE.search(body))

    def test_missing_symbol_returns_none(self):
        """A module without the EDID reader is a skip, not a pass."""
        self.assertIsNone(
            ast_edid.extract_function_disassembly(_FIXED, "no_such_symbol")
        )


class TestAstDpEdidTestcase(GLTTestBase):
    """Skip/fail gates for the decorated GreenLiT test."""

    def _patch_cli(self, **which):
        return self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.is_cli_available",
            side_effect=lambda name: which.get(name, True),
        )

    def _fake_run(self, dump, modinfo="/lib/modules/x/ast.ko\n"):
        def fake_run(command, _message, timeout=30.0):
            del timeout
            if command[:3] == ["modinfo", "-n", "ast"]:
                return SimpleNamespace(stdout=modinfo)
            if command[:2] == ["objdump", "-dr"]:
                return SimpleNamespace(stdout=dump)
            raise AssertionError(command)
        return fake_run

    def _patch_isfile(self, exists=True):
        return self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.os.path.isfile",
            return_value=exists,
        )

    def test_skips_when_ast_module_is_missing(self):
        """No ast.ko is a skip; a live ASPEED function is not required."""
        with self._patch_cli(), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.run_command",
            side_effect=RuntimeError("Cannot locate the ast kernel module"),
        ):
            with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
                ast_edid.test_ast_dp_edid_uses_msleep(None, self.mock_logger)
        self.assertIn("ast kernel module", str(ctx.exception))

    def test_skips_builtin_module(self):
        """Builtin ast has no .ko to disassemble."""
        with self._patch_cli(), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.run_command",
            side_effect=self._fake_run(_FIXED, "(builtin)\n"),
        ):
            with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
                ast_edid.test_ast_dp_edid_uses_msleep(None, self.mock_logger)
        self.assertIn("loadable kernel module", str(ctx.exception))

    def test_skips_compressed_module_without_decompressor(self):
        """Packed ast.ko SKIPs only when the decompressor is missing."""
        with self._patch_cli(zstd=False), self._patch_isfile(), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.run_command",
            side_effect=self._fake_run(_FIXED, "/lib/modules/x/ast.ko.zst\n"),
        ):
            with self.assertRaises(GLTCore.GLTTestSkip) as ctx:
                ast_edid.test_ast_dp_edid_uses_msleep(None, self.mock_logger)
        self.assertIn("zstd", str(ctx.exception))

    def test_decompresses_zst_module_then_passes(self):
        """Compressed ast.ko is still the observation after unpack."""
        def fake_decompress(command, stdout=None, **_kwargs):
            self.assertEqual(command[:2], ["zstd", "-dc"])
            stdout.write(b"fake-elf")
            return SimpleNamespace(returncode=0)

        with self._patch_cli(), self._patch_isfile(), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.run_command",
            side_effect=self._fake_run(_FIXED, "/lib/modules/x/ast.ko.zst\n"),
        ), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.subprocess.run",
            side_effect=fake_decompress,
        ):
            ast_edid.test_ast_dp_edid_uses_msleep(None, self.mock_logger)
        self.mock_logger.info.assert_called_once()

    def test_fails_when_edid_wait_busy_waits(self):
        """Unfixed disassembly must FAIL and cite the causal commit."""
        with self._patch_cli(), self._patch_isfile(), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.run_command",
            side_effect=self._fake_run(_UNFIXED),
        ):
            with self.assertRaises(GLTCore.GLTTestFail) as ctx:
                ast_edid.test_ast_dp_edid_uses_msleep(None, self.mock_logger)
        self.assertIn(ast_edid.AST_DP_EDID_SLEEP_COMMIT, str(ctx.exception))
        self.assertNotIn("DGX-", str(ctx.exception))

    def test_fails_when_busy_wait_remains_beside_msleep(self):
        """msleep plus leftover udelay in the same function is still unfixed."""
        with self._patch_cli(), self._patch_isfile(), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.run_command",
            side_effect=self._fake_run(_MIXED),
        ):
            with self.assertRaises(GLTCore.GLTTestFail):
                ast_edid.test_ast_dp_edid_uses_msleep(None, self.mock_logger)

    def test_passes_when_edid_wait_calls_msleep(self):
        """Fixed disassembly must PASS."""
        with self._patch_cli(), self._patch_isfile(), self.mock.patch(
            "src.modules.gpu.glt_test_ast_dp_edid.glt_utils.run_command",
            side_effect=self._fake_run(_FIXED),
        ):
            ast_edid.test_ast_dp_edid_uses_msleep(None, self.mock_logger)
        self.mock_logger.info.assert_called_once()
