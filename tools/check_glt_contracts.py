#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check GreenLiT source contracts without importing or executing source.

Rules:
    GLT001 (mandatory on every agent run): GLTTestResult.fail(), .skip(), and
    .warn() may exist only in the body of a function decorated with
    @GLTCore.GLTTestAttr() (common.glt_core.GLTTestAttr). Any other function is
    forbidden, including helpers, nested functions, lambdas, comprehensions,
    class or module bodies, and definition expressions.

Exit codes: 0 = clean, 1 = contract violations, 2 = checking failed.
Analysis supports absolute imports and import aliases, not dynamic API lookup,
re-exports, assignment aliases, or interprocedural data flow.
"""

import argparse
import ast
import os
from pathlib import Path
import sys
import tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE = "common.glt_core"
# GLT001: result calls exist only in a real @GLTTestAttr test function.
GLT001 = "GLT001"
RESULT_METHODS = {f"{CORE}.GLTTestResult.{method}" for method in ("fail", "skip", "warn")}
# discover() decorates this generated function later. Calls must explicitly
# target test_func; no other helper in this file is exempt.
GENERATED_TEST = (PROJECT_ROOT / "src/common/glt_ext_launcher.py").resolve()


class Bindings(ast.NodeVisitor):
    """Collect lexical bindings without descending into nested scopes.

    Comprehension targets stay in the comprehension. Assignment expressions
    inside one still bind in this scope.
    """

    def __init__(self):
        self.names = {}

    def bind(self, name, target=None):
        """Keep ambiguous/reassigned names unresolved instead of guessing."""
        self.names.setdefault(name, set()).add(target)

    def visit_ClassDef(self, node):  # pylint: disable=invalid-name
        """A nested definition binds its name in this scope."""
        self.bind(node.name)

    visit_FunctionDef = visit_ClassDef
    visit_AsyncFunctionDef = visit_ClassDef

    def visit_Import(self, node):  # pylint: disable=invalid-name
        """Resolve module imports, including aliases."""
        for alias in node.names:
            self.bind(alias.asname or alias.name.split(".")[0],
                      alias.name if alias.asname else alias.name.split(".")[0])

    def visit_ImportFrom(self, node):  # pylint: disable=invalid-name
        """Resolve absolute from-imports."""
        for alias in node.names:
            target = f"{node.module}.{alias.name}" if not node.level and node.module else None
            self.bind(alias.asname or alias.name, target)

    def visit_Lambda(self, node):  # pylint: disable=invalid-name,unused-argument
        """Lambda bindings belong to a separate scope."""

    def visit_ListComp(self, node):  # pylint: disable=invalid-name
        """Do not let comprehension targets shadow names in this scope."""
        for generator in node.generators:
            self.visit(generator.iter)
            for clause in generator.ifs:
                self.visit(clause)
        for field in ("elt", "key", "value"):
            child = getattr(node, field, None)
            if child is not None:
                self.visit(child)

    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp
    visit_SetComp = visit_ListComp

    def visit_Name(self, node):  # pylint: disable=invalid-name
        """Assignments shadow imports."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id)


class ContractChecker(ast.NodeVisitor):
    """Check result calls against their immediate execution scope."""

    def __init__(self, path):
        self.path = path.resolve()
        self.bindings = []
        self.scopes = []
        self.allowed = False
        self.violations = []

    def check_body(self, node, allowed=False):
        """Visit a lexical scope with its own bindings and result permission."""
        bindings = Bindings()
        for statement in node.body:
            bindings.visit(statement)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg):
                    bindings.bind(arg.arg)
        self.bindings.append(bindings.names)
        self.scopes.append(getattr(node, "name", "<module>"))
        previous = self.allowed
        self.allowed = allowed
        for statement in node.body:
            self.visit(statement)
        self.allowed = previous
        self.scopes.pop()
        self.bindings.pop()

    def resolve(self, node):
        """Resolve attribute chains rooted at an unambiguous import."""
        if isinstance(node, ast.Attribute):
            parent = self.resolve(node.value)
            return f"{parent}.{node.attr}" if parent else None
        if isinstance(node, ast.Name):
            for bindings in reversed(self.bindings):
                if node.id in bindings:
                    targets = bindings[node.id]
                    return next(iter(targets)) if len(targets) == 1 else None
        return None

    def visit_Call(self, node):  # pylint: disable=invalid-name
        """Report a result call outside an allowed test body."""
        target = self.resolve(node.func)
        generated = (
            self.path == GENERATED_TEST
            and self.scopes == ["<module>", "_generate_test_func", "test_func"]
            and any(keyword.arg == "func" and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "test_func" for keyword in node.keywords)
        )
        if target in RESULT_METHODS and not self.allowed and not generated:
            method = target.rsplit(".", 1)[1]
            self.violations.append(
                f"{self.path}:{node.lineno}:{node.col_offset + 1}: {GLT001} "
                f"GLTTestResult.{method}() is outside a @GLTTestAttr(...) test body "
                f"(scope: {self.scopes[-1]}). Return information to the decorated "
                f"test and call {method}() there."
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node):  # pylint: disable=invalid-name
        """Class bodies and definition expressions are not test bodies."""
        self.visit_definition_expressions(node)
        self.check_body(node)

    def visit_definition_expressions(self, node):
        """Do not grant test-body permission to decorators/defaults/annotations."""
        previous = self.allowed
        self.allowed = False
        for field, value in ast.iter_fields(node):
            if field == "body":
                continue
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    self.visit(child)
        self.allowed = previous

    def visit_FunctionDef(self, node):  # pylint: disable=invalid-name
        """Only a recognized decorator grants permission to this function."""
        allowed = any(isinstance(decorator, ast.Call)
                      and self.resolve(decorator.func) == f"{CORE}.GLTTestAttr"
                      for decorator in node.decorator_list)
        self.visit_definition_expressions(node)
        self.check_body(node, allowed)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):  # pylint: disable=invalid-name
        """Implicit function scopes cannot inherit test-body permission."""
        previous = self.allowed
        self.allowed = False
        self.scopes.append(f"<{type(node).__name__}>")
        self.generic_visit(node)
        self.scopes.pop()
        self.allowed = previous

    def visit_ListComp(self, node):  # pylint: disable=invalid-name
        """Comprehension targets shadow imports only inside that comprehension."""
        previous = self.allowed
        self.allowed = False
        self.scopes.append(f"<{type(node).__name__}>")
        generators = node.generators
        if generators:
            # The leftmost iterable is evaluated in the enclosing scope.
            self.visit(generators[0].iter)
        local = Bindings()
        for generator in generators:
            local.visit(generator.target)
        self.bindings.append(local.names)
        for generator in generators[1:]:
            self.visit(generator.iter)
        for generator in generators:
            for clause in generator.ifs:
                self.visit(clause)
        for field in ("elt", "key", "value"):
            child = getattr(node, field, None)
            if child is not None:
                self.visit(child)
        self.bindings.pop()
        self.scopes.pop()
        self.allowed = previous

    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp
    visit_SetComp = visit_ListComp

    def visit_Module(self, node):  # pylint: disable=invalid-name
        """Start analysis at module scope."""
        self.check_body(node)


def check_file(path):
    """Read and parse one source file; propagate errors to the CLI."""
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))
    checker = ContractChecker(path)
    checker.visit(tree)
    return checker.violations


def collect_files(paths):
    """Expand explicit files/directories, reporting traversal failures."""
    files = set()
    errors = []
    for path in paths:
        try:
            if path.is_dir():
                for directory, _, names in os.walk(path, onerror=errors.append):
                    files.update(Path(directory) / name for name in names if name.endswith(".py"))
            else:
                # Reading below reports missing, unreadable, and non-regular files.
                files.add(path)
        except OSError as error:
            errors.append(error)
    return sorted(files), errors


def main(argv=None):
    """Run checks; checking failures take precedence over contract violations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Python files or directories (default: project src/)")
    args = parser.parse_args(argv)
    files, errors = collect_files(args.paths or [PROJECT_ROOT / "src"])
    violations = []
    for path in files:
        try:
            violations.extend(check_file(path))
        except (OSError, SyntaxError, UnicodeError, ValueError, RecursionError) as error:
            errors.append(f"{path}: {error}")
    for violation in violations:
        print(violation)
    for error in errors:
        print(f"check_glt_contracts: checking failed: {error}", file=sys.stderr)
    if errors:
        return 2
    if violations:
        return 1
    print(f"check_glt_contracts: no violations ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
