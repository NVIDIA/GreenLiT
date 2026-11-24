# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
The module hosts the logic of managing test cases
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime
from enum import Enum
from functools import wraps
from graphlib import CycleError, TopologicalSorter
import inspect
import json
import logging
import os
from typing import NoReturn, TypeAlias

from common import glt_logger
from common import glt_session
from common import glt_utils


# Signature of a post hook: hook(scheduler, attr, logger) -> None.
PostHook: TypeAlias = Callable[["GLTScheduler", "GLTTestAttr", logging.Logger], None]


class PostHooks:
    """
    Optional hooks run by the framework after a test completes.
    The runner invokes the matching hook: PASS → postsuccess, SKIP → postskip,
    FAIL → postfail, AFAIL → postafail, WARN → postwarn. Each slot is a callable(scheduler, attr, logger).
    """

    def __init__(self) -> None:
        self.postsuccess = None  # callable(scheduler, attr, logger)
        self.postskip = None
        self.postafail = None
        self.postfail = None
        self.postwarn = None

    # Pylint 3 does not know the Pylint 4 too-many-positional-arguments message.
    # pylint: disable=unknown-option-value
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @classmethod
    def create(
        cls,
        postsuccess: PostHook | None = None,
        postskip: PostHook | None = None,
        postafail: PostHook | None = None,
        postfail: PostHook | None = None,
        postwarn: PostHook | None = None,
    ) -> "PostHooks":
        """
        Build a PostHooks instance for the given callables. Assign the return value
        to your test's attr: my_test.attr.post_hooks = PostHooks.create(...).
        Pass callables with signature (scheduler, attr, logger). Omitted slots are
        no-ops. The runner invokes the hook that matches the test result.
        """
        hooks = cls()
        hooks.postsuccess = postsuccess if postsuccess is not None else (lambda _s, _a, _l: None)
        hooks.postskip = postskip if postskip is not None else (lambda _s, _a, _l: None)
        hooks.postafail = postafail if postafail is not None else (lambda _s, _a, _l: None)
        hooks.postfail = postfail if postfail is not None else (lambda _s, _a, _l: None)
        hooks.postwarn = postwarn if postwarn is not None else (lambda _s, _a, _l: None)
        return hooks
    # pylint: enable=too-many-arguments,too-many-positional-arguments
    # pylint: enable=unknown-option-value

    @staticmethod
    def for_conditional_reboot(
        trigger_key: str, condition_fn: Callable[[], bool],
    ) -> PostHook:
        """
        Return a post-hook callable that runs a normal reboot only when progress
        has trigger_key for this test and condition_fn() is true. Use as
        postsuccess=..., postskip=..., etc. in PostHooks.create() and assign the
        result to test_func.attr.post_hooks.
        """
        def hook(scheduler, attr, logger):
            if scheduler.progress_check(attr.name, trigger_key) and condition_fn():
                logger.info("[post-hook] Conditional reboot triggered")
                glt_utils.reboot(attr.name, scheduler)

        return hook


def _condition_description(condition: Callable[[], bool]) -> object:
    """Return a condition description or its callable name."""
    return getattr(condition, "desc", getattr(condition, "__name__", type(condition).__name__))


class GLTConditionalCheck:
    """The class to check the condition"""

    def __init__(self) -> None:
        self.all_true: list[Callable[[], bool]] = []  # Functions must all return true
        self.all_false: list[Callable[[], bool]] = []  # Functions must all return false
        self.any_true: list[Callable[[], bool]] = []  # At least one must return True
        self.any_false: list[Callable[[], bool]] = []  # At least one must return False

    def check(self) -> tuple[bool, str]:
        """Validate the conditions for all function groups."""
        allow = False
        reason = ""
        # Check all_true
        for func in self.all_true:
            if not func():
                return False, f"{getattr(func, '__name__')} in all_true returned False"
            allow = True
            reason += "(ALL TRUE)"

        # Check all_false
        for func in self.all_false:
            if func():
                return False, f"{getattr(func, '__name__')} in all_false returned True"
            allow = True
            reason += "(ALL FALSE)"

        # Check any_true
        if self.any_true:
            can_do = False
            for func in self.any_true:
                if func():
                    can_do = True
                    desc = _condition_description(func)
                    reason += f"(ANY TRUE: {desc})"
                    break
            if not can_do:
                return False, "No function in any_true returned True"
            allow = True

        # Check any_false
        if self.any_false:
            can_do = False
            for func in self.any_false:
                if not func():
                    can_do = True
                    desc = _condition_description(func)
                    reason += f"(ANY FALSE: {desc})"
                    break
            if not can_do:
                return False, "No function in any_false returned False"
            allow = True

        return allow, reason


class GLTAllowFailure(GLTConditionalCheck):
    """Check conditions to allow GLT test case failed or not"""

    def can_fail(self) -> tuple[bool, str]:
        """Validate the conditions for all function groups."""
        return self.check()


class GLTConditionalSkip(GLTConditionalCheck):
    """Allow GLT test case be skipped"""

    def can_skip(self) -> tuple[bool, str]:
        """Validate the conditions for all function groups."""
        return self.check()


def _get_requires(func: Callable[..., object]) -> "GLTRequires":
    """Return the requirement set stored on a decorated test, creating it if needed.

    Requirements stay on the wrapped function when one exists, which is where
    the scheduler reads them. A requirement decorator applied directly, or
    beneath @GLTTestAttr, has no wrapper yet and stores them on func.
    """
    target = getattr(func, "__wrapped__", func)
    requires = getattr(target, "requires", None)
    if not isinstance(requires, GLTRequires):
        requires = GLTRequires()
        setattr(target, "requires", requires)
    return requires


class GLTRequires:
    """To check reaching requirements before launch"""

    def __init__(self) -> None:
        self.all_true: list[str] = []  # category:name entries that must pass
        self.all_false: list[str] = []  # category:name entries that must fail
        self.any_true: list[str] = []  # at least one must pass
        self.any_false: list[str] = []  # at least one must fail

    def dependency_names(self) -> list[str]:
        """Return all dependency test IDs referenced by this requirement set."""
        return self.all_true + self.all_false + self.any_true + self.any_false

    def categories(self) -> set[str]:
        """Return the set of category prefixes from "category:name" entries."""
        return {
            dep.split(":", 1)[0]
            for dep in self.dependency_names()
            if ":" in dep
        }

    @staticmethod
    def any_fail(
        category: str, name: str,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorator for any_fail when discover function"""

        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            _get_requires(func).any_false.append(f"{category}:{name}")
            return func

        return decorator

    @staticmethod
    def any_pass(
        category: str, name: str,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorator for any_pass when discover function"""

        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            _get_requires(func).any_true.append(f"{category}:{name}")
            return func

        return decorator

    def can_start(self, sd: "GLTScheduler") -> tuple[bool, str]:  # pylint: disable=too-many-branches
        """Validate the conditions for all function groups."""
        # Check all_true
        for func in self.all_true:
            test_obj = sd.get_test_obj_by_name(func)
            if test_obj is None:
                raise AttributeError(f"{func} is not a scheduled test")
            attr = _get_wrapped_attr(test_obj)
            outcome = getattr(attr, "result", None)
            if outcome is None or outcome[0] != GLTTestResult.PASS:
                return False, f"{getattr(attr, 'name')} in all_true did not pass"

        # Check all_false
        for func in self.all_false:
            test_obj = sd.get_test_obj_by_name(func)
            if test_obj is None:
                raise AttributeError(f"{func} is not a scheduled test")
            attr = _get_wrapped_attr(test_obj)
            outcome = getattr(attr, "result", None)
            if outcome is None or outcome[0] != GLTTestResult.FAIL:
                return False, f"{getattr(attr, 'name')} in all_false did not fail"

        # Check any_true
        if self.any_true:
            result = False
            for func in self.any_true:
                test_obj = sd.get_test_obj_by_name(func)
                if test_obj is None:
                    raise AttributeError(f"{func} is not a scheduled test")
                attr = _get_wrapped_attr(test_obj)
                outcome = getattr(attr, "result", None)
                if outcome is not None and outcome[0] == GLTTestResult.PASS:
                    result = True
                    break
            if not result:
                return False, "No function in any_true passed"

        # Check any_false
        if self.any_false:
            result = False
            for func in self.any_false:
                test_obj = sd.get_test_obj_by_name(func)
                if test_obj is None:
                    raise AttributeError(f"{func} is not a scheduled test")
                attr = _get_wrapped_attr(test_obj)
                outcome = getattr(attr, "result", None)
                if outcome is not None and outcome[0] == GLTTestResult.FAIL:
                    result = True
                    break
            if not result:
                return False, "No function in any_false failed"

        return True, ""

    @staticmethod
    def must_fail(
        category: str, name: str,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorator for must_fail when discover function"""

        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            _get_requires(func).all_false.append(f"{category}:{name}")
            return func

        return decorator

    @staticmethod
    def must_pass(
        category: str, name: str,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorator for must_pass when discover function"""

        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            _get_requires(func).all_true.append(f"{category}:{name}")
            return func

        return decorator


class GLTPlanType(Enum):
    """GLT supported test plans"""

    STANDARD = (
        "standard",
        "(default) Standard test cases that do not intentionally reboot or kexec and "
        "are not reasonably expected to hang or crash the system.",
        ["ext_executable"],
    )
    DISRUPTIVE = (
        "disruptive",
        "Test cases that intentionally reboot or kexec the system, or might hang "
        "or crash it only if a required patch is missing. Manual recovery, such "
        "as a power cycle, might be required.",
        [],
    )
    STRESS = (
        "stress",
        "Extended stress and stability test cases; may include disruptive tests.",
        ["ext_executable"],
    )
    EXT_EXECUTABLE = ("ext_executable", "External test cases.", [])

    def __init__(self, value, description, nested):
        self._value_ = value
        self.description = description
        self.nested = nested

    @classmethod
    def get_all_nested(cls, plan_value: str | Iterable[str]) -> set[str]:
        """Return a set of all plans nested under the given plan (including itself)."""
        if isinstance(plan_value, str):
            to_process = [plan_value]
        else:
            to_process = list(plan_value)
        result = set()
        while to_process:
            current = to_process.pop()
            if current not in result:
                result.add(current)
                plan = next((p for p in cls if p.value == current), None)
                if plan:
                    to_process.extend(plan.nested)
        return result


class GLTPlan:
    """GLT Test Plan"""

    def __init__(self, plan_type: str) -> None:
        valid_values = [t.value for t in GLTPlanType]
        if plan_type not in valid_values:
            raise ValueError(
                f"Invalid plan type: {plan_type}. Must be one of {valid_values}."
            )
        self.plan_type = plan_type

    def __call__(self, func: Callable[..., object]) -> Callable[..., object]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        plans = getattr(wrapper, "plans", None)
        if not isinstance(plans, list):
            plans = []
            setattr(wrapper, "plans", plans)

        selected_plans = {plan for plan in plans if isinstance(plan, str)}
        selected_plans.add(self.plan_type)
        incompatible_plans = {
            GLTPlanType.STANDARD.value,
            GLTPlanType.DISRUPTIVE.value,
        }
        if incompatible_plans.issubset(selected_plans):
            raise ValueError(
                "A test case cannot belong to both standard and disruptive plans."
            )

        plans.append(self.plan_type)
        return wrapper


class GLTTestAttr:  # pylint: disable=too-many-instance-attributes
    """GLT test case attributes"""

    def __init__(
        self,
        cat: str | None = None,
        name: str | None = None,
        func: Callable[..., object] | None = None,
    ) -> None:
        self.plans: list[str] = []
        self.category = cat
        self.name = name
        self.func = func
        self.allow_failure = GLTAllowFailure()
        self.conditional_skip = GLTConditionalSkip()
        self.result: tuple["GLTTestResult", str] | None = None

    def __call__(self, func: Callable[..., object]) -> Callable[..., object]:
        if not self.name:
            function_name = getattr(func, "__name__", None)
            if not isinstance(function_name, str):
                raise TypeError(
                    "test function __name__ must be str, "
                    f"got {type(function_name).__name__}"
                )
            self.name = function_name

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        if not hasattr(func, "attr"):
            setattr(func, "attr", self)
        # The name bound in the test module is this wrapper, not the inner func.
        # Set wrapper.attr so the author can write test_xyz.attr.post_hooks = ...
        # without reaching through __wrapped__.
        setattr(wrapper, "attr", self)

        caller_frame = inspect.stack()[1]
        filepath = caller_frame.filename

        if not self.category:
            self.category = self._extract_cat(filepath)

        if not self.func:
            self.func = func

        if not self.result:
            self.result = (GLTTestResult.NA, "")

        plans = getattr(func, "plans", None)
        if isinstance(plans, list) and all(isinstance(plan, str) for plan in plans):
            self.plans = [plan for plan in plans if isinstance(plan, str)]

        return wrapper

    def _extract_cat(self, filepath):
        # Extract 'cat' (e.g., 'arm64') from the file path
        parts = filepath.split(os.path.sep)
        if "modules" in parts:
            idx = parts.index("modules")
            return parts[idx + 1]
        raise ValueError(f"Not able to detect category from {filepath}")

    # pylint: disable=too-many-branches
    def dump_attrs(self, logger: logging.Logger | None = None) -> None:
        """Dump test case attributes"""
        if logger is None:
            output = print
        else:
            output = logger.debug
        func = self.func
        output(f"[{self.category}:{self.name}]")
        output(f" - function: {func}")
        if func is None:
            return
        if self.plans:
            output(" - plans:")
            for plan in self.plans:
                output(f"  - {plan}")
        requires = getattr(func, "requires", None)
        if isinstance(requires, GLTRequires):
            output(" - requires:")
            if requires.all_true:
                output("  - all_true:")
                for item in requires.all_true:
                    output(f"   - [{item}]")
            if requires.all_false:
                output("  - all_false:")
                for item in requires.all_false:
                    output(f"   - [{item}]")
            if requires.any_true:
                output("  - any_true:")
                for item in requires.any_true:
                    output(f"   - [{item}]")
            if requires.any_false:
                output("  - any_false:")
                for item in requires.any_false:
                    output(f"   - [{item}]")
        conditional_skip = getattr(getattr(func, "attr", None), "conditional_skip", None)
        if isinstance(conditional_skip, GLTConditionalSkip):
            output(" - conditional_skip:")
            if conditional_skip.all_true:
                output("  - all_true:")
                for condition in conditional_skip.all_true:
                    output(f"   - [{_condition_description(condition)}]")
            if conditional_skip.all_false:
                output("  - all_false:")
                for condition in conditional_skip.all_false:
                    output(f"   - [{_condition_description(condition)}]")
            if conditional_skip.any_true:
                output("  - any_true:")
                for condition in conditional_skip.any_true:
                    output(f"   - [{_condition_description(condition)}]")
            if conditional_skip.any_false:
                output("  - any_false:")
                for condition in conditional_skip.any_false:
                    output(f"   - [{_condition_description(condition)}]")
        allow_failure = getattr(getattr(func, "attr", None), "allow_failure", None)
        if isinstance(allow_failure, GLTAllowFailure):
            output(" - allow_failure:")
            if allow_failure.all_true:
                output("  - all_true:")
                for condition in allow_failure.all_true:
                    output(f"   - [{_condition_description(condition)}]")
            if allow_failure.all_false:
                output("  - all_false:")
                for condition in allow_failure.all_false:
                    output(f"   - [{_condition_description(condition)}]")
            if allow_failure.any_true:
                output("  - any_true:")
                for condition in allow_failure.any_true:
                    output(f"   - [{_condition_description(condition)}]")
            if allow_failure.any_false:
                output("  - any_false:")
                for condition in allow_failure.any_false:
                    output(f"   - [{_condition_description(condition)}]")
        if self.result:
            output(f" - result: [{self.result[0].value}] {self.result[1]}")

    # pylint: enable=too-many-branches


class GLTTestSkip(Exception):
    """Exception class for skiped GLT test case"""


class GLTTestFail(Exception):
    """Exception class for failed GLT test case"""


class GLTTestWarn(Exception):
    """Exception class for warned GLT test case"""


def _record_test_result(target: object, result: tuple["GLTTestResult", str]) -> None:
    """Store result on a decorated test, preferring the wrapped function."""
    wrapped = getattr(target, "__wrapped__", None)
    if wrapped is not None:
        attr = getattr(wrapped, "attr", None)
        if isinstance(attr, GLTTestAttr):
            attr.result = result
            return
        raise AttributeError("wrapped test has no GLTTestAttr")
    attr = getattr(target, "attr", None)
    if isinstance(attr, GLTTestAttr):
        attr.result = result
        return
    raise AttributeError("test has no GLTTestAttr")


class GLTTestResult(Enum):
    """GLT Test results"""

    SKIP = "SKIPPED"
    PASS = "PASSED"
    AFAIL = "Allowed FAILURE"
    FAIL = "FAILED"
    WARN = "WARNED"
    NA = "N/A"

    @classmethod
    def colors(cls) -> dict["GLTTestResult", str]:
        """Get the color map"""
        return {
            cls.FAIL: "\033[31m",  # Red
            cls.PASS: "\033[32m",  # Green
            cls.SKIP: "\033[33m",  # Yellow
            cls.AFAIL: "\033[34m",  # Blue
            cls.NA: "\033[35m",  # Magenta
            cls.WARN: "\033[36m",  # Cyan
        }

    @classmethod
    def colorize(cls, result: "GLTTestResult") -> str:
        """Return the colored string"""
        colors = cls.colors()
        color = colors.get(result, "")
        return f"{color}{result.value}\033[0m"  # Reset color

    @staticmethod
    def skip(reason: str = "", func: Callable[..., object] | None = None) -> NoReturn:
        """Skip the test case and apply reason to function"""
        target: object = func
        if not func:
            caller_frame = inspect.stack()[1]
            caller_func = caller_frame.function
            target = caller_frame.frame.f_globals.get(caller_func)
        _record_test_result(target, (GLTTestResult.SKIP, reason))

        raise GLTTestSkip(reason)

    @staticmethod
    def fail(reason: str = "", func: Callable[..., object] | None = None) -> NoReturn:
        """Fail the test case and apply reason to function"""
        target: object = func
        if not func:
            caller_frame = inspect.stack()[1]
            caller_func = caller_frame.function
            target = caller_frame.frame.f_globals.get(caller_func)
        _record_test_result(target, (GLTTestResult.FAIL, reason))

        raise GLTTestFail(reason)

    @staticmethod
    def warn(reason: str = "", func: Callable[..., object] | None = None) -> NoReturn:
        """Warn from the test case and apply reason to function"""
        target: object = func
        if not func:
            caller_frame = inspect.stack()[1]
            caller_func = caller_frame.function
            target = caller_frame.frame.f_globals.get(caller_func)
        _record_test_result(target, (GLTTestResult.WARN, reason))

        raise GLTTestWarn(reason)


def _get_wrapped_attr(test_obj: object) -> object:
    """Return the attribute record attached to a decorated test."""
    wrapped = getattr(test_obj, "__wrapped__", None)
    attr = getattr(wrapped, "attr", None)
    if attr is None:
        raise AttributeError("scheduled test has no GLTTestAttr")
    return attr


# pylint: disable=too-many-instance-attributes
class GLTScheduler:
    """
    GLT Scheduler to control the test cases be performed and order.
    Test case will be stored as f"{category}:{test_case}".
    """

    def __init__(
        self, test_objs: Iterable[Callable[..., object]], logdir: str | None = None,
    ) -> None:
        self.dependencies = defaultdict(list)
        self.test_categories = []
        self.test_tags = set()
        self.test_list: list[Callable[..., object]] = []
        self.test_map: dict[str, Callable[..., object]] = {}
        self.progress: list[object] = []
        self.log_dir = logdir
        for test_obj in test_objs:
            name = self.get_test_case_name(test_obj)
            self.test_list.append(test_obj)
            wrapped = getattr(test_obj, "__wrapped__", None)
            reqs = getattr(wrapped, "requires", None)
            if isinstance(reqs, GLTRequires):
                if reqs.all_true:
                    for item in reqs.all_true:
                        self.add_dependency(name, item)
                if reqs.all_false:
                    for item in reqs.all_false:
                        self.add_dependency(name, item)
                if reqs.any_true:
                    for item in reqs.any_true:
                        self.add_dependency(name, item)
                if reqs.any_false:
                    for item in reqs.any_false:
                        self.add_dependency(name, item)
            elif reqs is not None:
                raise TypeError(f"{name}: requires must be GLTRequires, got {type(reqs).__name__}")

    def add_dependency(self, case: str, depends_on: str) -> None:
        """Add a dependency: the case depends on depends_on."""
        self.dependencies[depends_on].append(case)

    @staticmethod
    def get_test_case_name(test_obj: Callable[..., object]) -> str:
        """Return the test case name for structuring"""
        attr = _get_wrapped_attr(test_obj)
        return f"{getattr(attr, 'category')}:{getattr(attr, 'name')}"

    def get_test_obj_by_name(self, name: str) -> Callable[..., object] | None:
        """Return the test object by test case name.

        None means the scheduler has no test with that name.
        """
        return None if not self.test_map else self.test_map.get(name)

    def progress_delete_time(self, item: str, status: str, save: bool = True) -> None:
        """
        Delete time stamp for a test item.
        item: test case name in string.
        status: 'Start', 'Completed' or custom status for test case itself.
        """
        for entry in self.progress:
            if not isinstance(entry, dict):
                continue
            record = entry.get(item)
            if isinstance(record, dict):
                record.pop(status, None)

        if save:
            self.progress_save()

    def progress_check(self, item: str, k: str) -> bool:
        """Check if key exists in specific test item"""
        for entry in self.progress:
            if not isinstance(entry, dict):
                continue
            record = entry.get(item)
            if isinstance(record, dict) and k in record:
                return True
        return False

    def progress_update(
        self, item: str, k: str, v: object, save: bool = True,
    ) -> None:
        """
        Save test progress.
        item: test case name in string.
        k: key to save in dict.
        v: value to save in dict.
        """
        updated = False
        for entry in self.progress:
            if not isinstance(entry, dict):
                continue
            record = entry.get(item)
            if isinstance(record, dict):
                record[k] = v
                updated = True
        if not updated:
            self.progress.append({item: {k: v}})

        if save:
            self.progress_save()

    # pylint: disable=too-many-branches,too-many-return-statements
    def progress_read(self, session_dir: str | os.PathLike[str] | None = None) -> str:
        """Load the test progress from file."""
        if session_dir is None:
            if self.log_dir is None:
                return "Unable to read resume progress: scheduler log directory is unset"
            progress_path = os.path.join(
                self.log_dir, glt_session.SESSION_PROGRESS_FILE)
        else:
            progress_path = glt_session.get_session_progress_path(session_dir)
        try:
            session_progress = glt_session.read_progress_file(progress_path)
        except OSError as err:
            return f"Unable to read resume progress: {err}"
        except json.JSONDecodeError as err:
            return f"Unable to parse resume progress: {err}"

        summary_error = session_progress.summary_error()
        if summary_error:
            return summary_error

        progress = session_progress.progress
        if not isinstance(progress, list):
            return (
                f"Resume progress in {session_progress.progress_path} "
                "has invalid format (expected a JSON array)")

        # Validate all entries before mutating scheduler or test state.
        resume_test_list: list[Callable[..., object]] = []
        resume_results: list[tuple[object, tuple[GLTTestResult, str]]] = []
        # progress stays object until the cached payload is modeled.
        for item in progress:  # pylint: disable=not-an-iterable
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("Summary"), dict):
                continue

            for case, status in item.items():
                if not isinstance(status, dict) or not status.get("Category"):
                    continue
                category = status["Category"]
                if not isinstance(category, str):
                    continue
                obj_name = f"{category}:{case}"
                test_obj = self.get_test_obj_by_name(obj_name)
                if test_obj is None:
                    return (
                        "Resume progress references unknown test case "
                        f"{obj_name}")
                attr = _get_wrapped_attr(test_obj)
                if "Result" in status:
                    result_value = status["Result"]
                    result_reason = status.get("ResultReason")
                    if not isinstance(result_value, str):
                        return f"Resume progress Result for {obj_name} is not a string"
                    if not isinstance(result_reason, str):
                        return (
                            f"Resume progress ResultReason for {obj_name} "
                            "is not a string"
                        )
                    try:
                        parsed_result = GLTTestResult(result_value)
                    except ValueError:
                        return (
                            f"Resume progress result for {obj_name} is invalid: "
                            f"{result_value}"
                        )
                    resume_results.append((attr, (
                        parsed_result,
                        result_reason,
                    )))
                resume_test_list.append(test_obj)

        summary = session_progress.summary
        if summary is None:
            return "Resume progress has no usable summary"

        self.progress = progress
        self.test_tags = summary.test_tags
        self.test_categories = summary.categories
        for attr, result in resume_results:
            setattr(attr, "result", result)

        # Update the test list only after finding resumable test entries.
        if resume_test_list:
            self.test_list = resume_test_list

        return ""

    def progress_record_time(self, item: str, status: str, save: bool = True) -> None:
        """
        Save time stamp for a test item.
        item: test case name in string.
        status: 'Start', 'Completed' or custom status for test case itself.
        """
        current_time = datetime.now().strftime(glt_logger.LOGGING_DATE_FORMAT)
        self.progress_update(item, status, current_time, save)

    def progress_save(self) -> None:
        """Save test progress to filesystem."""
        with open(
            f"{self.log_dir}/test_progress.json", "w", encoding="UTF-8"
        ) as f:
            json.dump(self.progress, f, indent=4)

    def sort(self) -> None:
        """Topological sort nodes and cover it as test_list."""
        tests_by_name = {
            self.get_test_case_name(node): node for node in self.test_list
        }
        for dependency_id, dependent_ids in self.dependencies.items():
            for dependent_id in dependent_ids:
                if dependency_id not in tests_by_name:
                    raise ValueError(
                        f"Dependency {dependency_id} required by "
                        f"{dependent_id} not found")
                if dependent_id not in tests_by_name:
                    raise ValueError(
                        f"Test {dependent_id} depends on {dependency_id} "
                        "but is not in the scheduler")

        # Register all tests first so initially ready tests keep their input order.
        sorter = TopologicalSorter()
        for node in self.test_list:
            sorter.add(node)
        for dependency_id, dependent_ids in self.dependencies.items():
            if not dependent_ids:
                continue
            prerequisite = tests_by_name[dependency_id]
            for dependent_id in dependent_ids:
                sorter.add(tests_by_name[dependent_id], prerequisite)

        try:
            ordered = list(sorter.static_order())
        except CycleError as error:
            raise ValueError("Cycle detected in dependencies") from error

        self.test_list = ordered
        for case in self.test_list:
            name = self.get_test_case_name(case)
            self.test_map[name] = case
