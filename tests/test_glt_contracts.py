"""Contract checker behavior and the production-source policy gate."""

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from tools import check_glt_contracts as contracts


class TestGLTContracts(unittest.TestCase):
    """Check contracts using source strings; never execute hardware tests."""

    def check_source(self, source, path=None):
        """Return diagnostics for a synthetic module."""
        checker = contracts.ContractChecker(path or Path("example.py"))
        checker.visit(ast.parse(textwrap.dedent(source)))
        return checker.violations

    def run_checker(self, *paths):
        """Capture the CLI result and both diagnostic streams."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = contracts.main([str(path) for path in paths])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_aliases_and_decorated_bodies(self):
        """Absolute module imports and direct import aliases are recognized."""
        for imports, decorator, result in (
            ("from common import glt_core as Core", "Core.GLTTestAttr", "Core.GLTTestResult"),
            ("import common.glt_core", "common.glt_core.GLTTestAttr", "common.glt_core.GLTTestResult"),
            ("import common.glt_core as C", "C.GLTTestAttr", "C.GLTTestResult"),
            ("from common.glt_core import GLTTestAttr as A, GLTTestResult as R", "A", "R"),
        ):
            for method in ("fail", "skip", "warn"):
                with self.subTest(imports=imports, method=method):
                    source = (
                        f"{imports}\n@{decorator}()\nasync def test_case():\n"
                        f"    if True:\n        {result}.{method}('reason')"
                    )
                    self.assertEqual(self.check_source(source), [])
                    self.assertEqual(len(self.check_source(source.replace(f"@{decorator}()\n", ""))), 1)

    def test_definition_expressions(self):
        """Defaults, annotations, and decorators are outside the test body."""
        for header in (
            "@C.GLTTestAttr(name=C.GLTTestResult.fail('bad'))\ndef test_case(): pass",
            "@C.GLTTestAttr()\ndef test_case(x=C.GLTTestResult.skip('bad')): pass",
            "@C.GLTTestAttr()\ndef test_case() -> C.GLTTestResult.warn('bad'): pass",
        ):
            with self.subTest(header=header):
                self.assertEqual(len(self.check_source("from common import glt_core as C\n" + header)), 1)

    def test_encoding_and_parse_failures(self):
        """Python source encodings are honored; bad source returns 2."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_bytes(b"# coding: latin-1\nvalue = 'caf\xe9'\n")
            self.assertEqual(self.run_checker(path)[0], 0)
            for content in (b"def broken(:", b"\xff", b"# coding: unknown-encoding\n"):
                path.write_bytes(content)
                status, _, stderr = self.run_checker(path)
                self.assertEqual(status, 2)
                self.assertIn("checking failed", stderr)

    def test_exit_codes_and_error_precedence(self):
        """Findings return 1, missing inputs return 2, and both are reported."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.py"
            path.write_text("from common.glt_core import GLTTestResult\nGLTTestResult.fail('bad')\n", encoding="utf-8")
            missing = Path(directory) / "missing.py"
            self.assertEqual(self.run_checker(path)[0], 1)
            status, stdout, stderr = self.run_checker(path, missing)
            self.assertEqual(status, 2)
            self.assertEqual(contracts.GLT001, "GLT001")
            self.assertIn(f"{path}:2:1: {contracts.GLT001}", stdout)
            self.assertIn(str(missing), stderr)

    def test_generated_test_exception_is_narrow(self):
        """Only the known generated function with an explicit target is exempt."""
        source = """
            from common import glt_core as C
            def _generate_test_func():
                def test_func():
                    C.GLTTestResult.skip('reason', func=test_func)
        """
        self.assertEqual(self.check_source(source, contracts.GENERATED_TEST), [])
        self.assertEqual(len(self.check_source(source)), 1)
        for changed in (
            source.replace("func=test_func", "func=other"),
            source.replace(", func=test_func", ""),
            source.replace("def test_func", "def helper"),
            source.replace("def _generate_test_func", "def other_factory"),
        ):
            with self.subTest(source=changed):
                self.assertEqual(len(self.check_source(changed, contracts.GENERATED_TEST)), 1)

    def test_module_and_helper_calls(self):
        """A test-like name or explicit func target does not exempt a helper."""
        for method in ("fail", "skip", "warn"):
            source = f"""
                from common import glt_core as C
                C.GLTTestResult.{method}('module')
                def test_undecorated():
                    C.GLTTestResult.{method}('helper', func=test_undecorated)
            """
            self.assertEqual(len(self.check_source(source)), 2)

    def test_nested_scopes(self):
        """Nested helpers and implicit function scopes cannot inherit permission."""
        violations = self.check_source("""
            from common import glt_core as C
            @C.GLTTestAttr()
            def test_case():
                def helper():
                    C.GLTTestResult.fail('helper')
                callback = lambda: C.GLTTestResult.skip('lambda')
                class Nested:
                    C.GLTTestResult.warn('class')
                values = [C.GLTTestResult.fail('comprehension') for x in range(1)]
                @C.GLTTestAttr()
                def nested_test():
                    C.GLTTestResult.skip('allowed')
                C.GLTTestResult.warn('allowed')
        """)
        self.assertEqual(len(violations), 4)

    def test_production_source_contracts(self):
        """The ordinary unit-test command also gates all production source."""
        status, stdout, stderr = self.run_checker()
        self.assertEqual(status, 0, stdout + stderr)

    def test_script_from_another_directory(self):
        """Default scanning is anchored to the checkout, not the caller's cwd."""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(contracts.PROJECT_ROOT / "tools/check_glt_contracts.py")],
                cwd=directory, capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no violations", result.stdout)

    def test_unreadable_files_and_directories(self):
        """Do not silently succeed when file reading or traversal fails."""
        with mock.patch.object(contracts.tokenize, "open", side_effect=PermissionError("denied")):
            self.assertEqual(self.run_checker(Path("blocked.py"))[0], 2)

        def failed_walk(_path, onerror):
            onerror(PermissionError("directory denied"))
            return iter(())

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(contracts.os, "walk", side_effect=failed_walk):
                self.assertEqual(self.run_checker(Path(directory))[0], 2)

    def test_unrelated_and_shadowed_names(self):
        """Unrelated result-like APIs and shadowed imports are not GLT calls."""
        self.assertEqual(self.check_source("""
            from unrelated import GLTTestResult
            GLTTestResult.fail('unrelated')
            from common import glt_core as C
            def helper(C):
                C.GLTTestResult.warn('shadowed parameter')
            def another():
                C = object()
                C.GLTTestResult.skip('shadowed assignment')
            logger.warn('log message')
        """), [])

    def test_comprehension_targets_do_not_shadow_imports(self):
        """Comprehension targets are local and must not hide imported GLT calls."""
        violations = self.check_source("""
            from common import glt_core as C
            def helper():
                listed = [C.GLTTestResult.fail('inside') for x in range(1)]
                _ = [y for C in range(1)]
                _ = {y for C in range(1)}
                _ = {y: y for C in range(1)}
                _ = (y for C in range(1))
                C.GLTTestResult.skip('outside')
            def iterable_call():
                _ = [y for C in [C.GLTTestResult.warn('iterable')]]
            def same_target():
                _ = [C.GLTTestResult.fail('loop variable') for C in range(1)]
            def assigned():
                _ = [(C := object()) for item in range(1)]
                C.GLTTestResult.fail('assignment expression')
            @C.GLTTestAttr()
            def test_case():
                _ = [C.GLTTestResult.fail('inside') for x in range(1)]
                _ = [y for C in range(1)]
                C.GLTTestResult.skip('allowed')
        """)
        methods = [item.split("GLTTestResult.", 1)[1].split("()", 1)[0] for item in violations]
        self.assertEqual(methods, ["fail", "skip", "warn", "fail"])
