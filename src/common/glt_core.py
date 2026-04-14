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

from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from functools import wraps
import inspect
import json
import os
from typing import Callable
from common import glt_logger
from common import glt_utils


class PostHooks:
    """
    Optional hooks run by the framework after a test completes.
    The runner invokes the matching hook: PASS → postsuccess, SKIP → postskip,
    FAIL → postfail, AFAIL → postafail. Each slot is a callable(scheduler, attr, logger).
    """

    def __init__(self):
        self.postsuccess = None  # callable(scheduler, attr, logger)
        self.postskip = None
        self.postafail = None
        self.postfail = None

    @classmethod
    def create(
        cls,
        postsuccess=None,
        postskip=None,
        postafail=None,
        postfail=None,
    ):
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
        return hooks

    @staticmethod
    def for_conditional_reboot(trigger_key: str, condition_fn: Callable[[], bool]):
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


class GLTConditionalCheck:
    """The class to check the condition"""

    def __init__(self):
        self.all_true = []  # Functions must all return true
        self.all_false = []  # Functions must all return false
        self.any_true = []  # At least one must return True
        self.any_false = []  # At least one must return False

    def check(self) -> tuple[bool, str]:
        """Validate the conditions for all function groups."""
        allow = False
        reason = ""
        # Check all_true
        for func in self.all_true:
            if not func():
                return False, f"{func.__name__} in all_true returned False"
            allow = True
            reason += "(ALL TRUE)"

        # Check all_false
        for func in self.all_false:
            if func():
                return False, f"{func.__name__} in all_false returned True"
            allow = True
            reason += "(ALL FALSE)"

        # Check any_true
        if self.any_true:
            can_do = False
            for func in self.any_true:
                if func():
                    can_do = True
                    reason += f"(ANY TRUE: {func.desc})"
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
                    reason += f"(ANY FALSE: {func.desc})"
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


class GLTRequires:
    """To check reaching requirements before launch"""

    def __init__(self):
        self.all_true = []  # Functions must all return true
        self.all_false = []  # Functions must all return false
        self.any_true = []  # At least one must return True
        self.any_false = []  # At least one must return False

    def _dependency_names(self):
        return self.all_true + self.all_false + self.any_true + self.any_false

    def categories(self):
        """Return the set of category prefixes from "category:name" entries."""
        return {
            dep.split(":", 1)[0]
            for dep in self._dependency_names()
            if ":" in dep
        }

    @staticmethod
    def any_fail(category, name):
        """Decorator for any_fail when discover function"""

        def decorator(func):
            if not hasattr(func.__wrapped__, "requires"):
                func.__wrapped__.requires = GLTRequires()
            func.__wrapped__.requires.any_false.append(f"{category}:{name}")
            return func

        return decorator

    @staticmethod
    def any_pass(category, name):
        """Decorator for any_pass when discover function"""

        def decorator(func):
            if not hasattr(func.__wrapped__, "requires"):
                func.__wrapped__.requires = GLTRequires()
            func.__wrapped__.requires.any_true.append(f"{category}:{name}")
            return func

        return decorator

    def can_start(self, sd) -> tuple[bool, str]:
        """Validate the conditions for all function groups."""
        # Check all_true
        for func in self.all_true:
            attr = sd.get_test_obj_by_name(func).__wrapped__.attr
            if attr.result[0] != GLTTestResult.PASS:
                return False, f"{attr.name} in all_true returned False"

        # Check all_false
        for func in self.all_false:
            attr = sd.get_test_obj_by_name(func).__wrapped__.attr
            if attr.result[0] != GLTTestResult.FAIL:
                return False, f"{attr.name} in all_false returned True"

        # Check any_true
        if self.any_true:
            result = False
            for func in self.any_true:
                attr = sd.get_test_obj_by_name(func).__wrapped__.attr
                if attr.result[0] == GLTTestResult.PASS:
                    result = True
                    break
            if not result:
                return False, "No function in any_true returned True"

        # Check any_false
        if self.any_false:
            result = False
            for func in self.any_false:
                attr = sd.get_test_obj_by_name(func).__wrapped__.attr
                if attr.result[0] == GLTTestResult.FAIL:
                    result = True
                    break
            if not result:
                return False, "No function in any_false returned False"

        return True, ""

    @staticmethod
    def must_fail(category, name):
        """Decorator for must_fail when discover function"""

        def decorator(func):
            if not hasattr(func.__wrapped__, "requires"):
                func.__wrapped__.requires = GLTRequires()
            func.__wrapped__.requires.all_false.append(f"{category}:{name}")
            return func

        return decorator

    @staticmethod
    def must_pass(category, name):
        """Decorator for must_pass when discover function"""

        def decorator(func):
            if not hasattr(func.__wrapped__, "requires"):
                func.__wrapped__.requires = GLTRequires()
            func.__wrapped__.requires.all_true.append(f"{category}:{name}")
            return func

        return decorator


class GLTPlanType(Enum):
    """GLT supported test plans"""

    SANITY = ("sanity", "(default) Quick sanity check.", ["ext_executable"])
    STANDARD = ("standard", "Standard test cases, reboot will be performed.", ["ext_executable"])
    STRESS = ("stress", "Stress system, reboot will be performed.", ["ext_executable"])
    EXT_EXECUTABLE = ("ext_executable", "External test cases.", [])

    def __init__(self, value, description, nested):
        self._value_ = value
        self.description = description
        self.nested = nested

    @classmethod
    def get_all_nested(cls, plan_value):
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

    def __init__(self, plan_type):
        valid_values = [t.value for t in GLTPlanType]
        if plan_type not in valid_values:
            raise ValueError(
                f"Invalid plan type: {plan_type}. Must be one of {valid_values}."
            )
        self.plan_type = plan_type

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if not hasattr(wrapper, "plans"):
            setattr(wrapper, "plans", [])

        wrapper.plans.append(self.plan_type)
        return wrapper


class GLTTestAttr:  # pylint: disable=too-many-instance-attributes
    """GLT test case attributes"""

    def __init__(self, cat=None, name=None, func=None):
        self.plans = []
        self.category = cat
        self.name = name
        self.func = func
        self.allow_failure = GLTAllowFailure()
        self.conditional_skip = GLTConditionalSkip()
        self.result = None

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if not hasattr(func, "attr"):
            setattr(func, "attr", self)
        # The name bound in the test module is this wrapper, not the inner func.
        # Set wrapper.attr so the author can write test_xyz.attr.post_hooks = ...
        # without reaching through __wrapped__.
        wrapper.attr = self

        caller_frame = inspect.stack()[1]
        filepath = caller_frame.filename

        if not self.category:
            self.category = self._extract_cat(filepath)

        if not self.name:
            self.name = func.__name__

        if not self.func:
            self.func = func

        if not self.result:
            self.result = (GLTTestResult.NA, "")

        if hasattr(func, "plans"):
            self.plans = func.plans

        return wrapper

    def _extract_cat(self, filepath):
        # Extract 'cat' (e.g., 'arm64') from the file path
        parts = filepath.split(os.path.sep)
        if "modules" in parts:
            idx = parts.index("modules")
            return parts[idx + 1]
        raise ValueError(f"Not able to detect category from {filepath}")

    # pylint: disable=too-many-branches
    def dump_attrs(self, logger=None):
        """Dump test case attributes"""
        if logger is None:
            output = print
        else:
            output = logger.debug
        output(f"[{self.category}:{self.name}]")
        output(f" - function: {self.func}")
        if self.plans:
            output(" - plans:")
            for plan in self.plans:
                output(f"  - {plan}")
        if hasattr(self.func, "requires"):
            output(" - requires:")
            if self.func.requires.all_true:
                output("  - all_true:")
                for item in self.func.requires.all_true:
                    output(f"   - [{item}]")
            if self.func.requires.all_false:
                output("  - all_false:")
                for item in self.func.requires.all_false:
                    output(f"   - [{item}]")
            if self.func.requires.any_true:
                output("  - any_true:")
                for item in self.func.requires.any_true:
                    output(f"   - [{item}]")
            if self.func.requires.any_false:
                output("  - any_false:")
                for item in self.func.requires.any_false:
                    output(f"   - [{item}]")
        if hasattr(self.func.attr, "conditional_skip"):
            output(" - conditional_skip:")
            if self.func.attr.conditional_skip.all_true:
                output("  - all_true:")
                for f in self.func.attr.conditional_skip.all_true:
                    output(f"   - [{f.desc}]")
            if self.func.attr.conditional_skip.all_false:
                output("  - all_false:")
                for f in self.func.attr.conditional_skip.all_false:
                    output(f"   - [{f.desc}]")
            if self.func.attr.conditional_skip.any_true:
                output("  - any_true:")
                for f in self.func.attr.conditional_skip.any_true:
                    output(f"   - [{f.desc}]")
            if self.func.attr.conditional_skip.any_false:
                output("  - any_false:")
                for f in self.func.attr.conditional_skip.any_false:
                    output(f"   - [{f.desc}]")
        if hasattr(self.func.attr, "allow_failure"):
            output(" - allow_failure:")
            if self.func.attr.allow_failure.all_true:
                output("  - all_true:")
                for f in self.func.attr.allow_failure.all_true:
                    output(f"   - [{f.desc}]")
            if self.func.attr.allow_failure.all_false:
                output("  - all_false:")
                for f in self.func.attr.allow_failure.all_false:
                    output(f"   - [{f.desc}]")
            if self.func.attr.allow_failure.any_true:
                output("  - any_true:")
                for f in self.func.attr.allow_failure.any_true:
                    output(f"   - [{f.desc}]")
            if self.func.attr.allow_failure.any_false:
                output("  - any_false:")
                for f in self.func.attr.allow_failure.any_false:
                    output(f"   - [{f.desc}]")
        if self.result:
            output(f" - result: [{self.result[0].value}] {self.result[1]}")

    # pylint: enable=too-many-branches


class GLTTestSkip(Exception):
    """Exception class for skiped GLT test case"""


class GLTTestFail(Exception):
    """Exception class for failed GLT test case"""


class GLTTestResult(Enum):
    """GLT Test results"""

    SKIP = "SKIPPED"
    PASS = "PASSED"
    AFAIL = "Allowed FAILURE"
    FAIL = "FAILED"
    NA = "N/A"

    @classmethod
    def colors(cls):
        """Get the color map"""
        return {
            cls.FAIL: "\033[31m",  # Red
            cls.PASS: "\033[32m",  # Green
            cls.SKIP: "\033[33m",  # Yellow
            cls.AFAIL: "\033[34m",  # Blue
            cls.NA: "\033[35m",  # Magenta
        }

    @classmethod
    def colorize(cls, result):
        """Return the colored string"""
        colors = cls.colors()
        color = colors.get(result, "")
        return f"{color}{result.value}\033[0m"  # Reset color

    @staticmethod
    def skip(reason="", func=None):
        """Skip the test case and apply reason to function"""
        if not func:
            caller_frame = inspect.stack()[1]
            caller_func = caller_frame.function
            func = caller_frame.frame.f_globals.get(caller_func)
        if hasattr(func, "__wrapped__"):
            func.__wrapped__.attr.result = (GLTTestResult.SKIP, reason)
        elif hasattr(func, "attr"):
            func.attr.result = (GLTTestResult.SKIP, reason)

        raise GLTTestSkip(reason)

    @staticmethod
    def fail(reason="", func=None):
        """Fail the test case and apply reason to function"""
        if not func:
            caller_frame = inspect.stack()[1]
            caller_func = caller_frame.function
            func = caller_frame.frame.f_globals.get(caller_func)
        if hasattr(func, "__wrapped__"):
            func.__wrapped__.attr.result = (GLTTestResult.FAIL, reason)
        elif hasattr(func, "attr"):
            func.attr.result = (GLTTestResult.FAIL, reason)

        raise GLTTestFail(reason)


# pylint: disable=too-many-instance-attributes
class GLTScheduler:
    """
    GLT Scheduler to control the test cases be performed and order.
    Test case will be stored as f"{category}:{test_case}".
    """

    def __init__(self, test_objs, logdir=None):
        self.dependencies = defaultdict(list)
        self.test_categories = []
        self.test_tags = set()
        self.test_list = []
        self.test_map = {}
        self.progress = []
        self.log_dir = logdir
        for test_obj in test_objs:
            name = self.get_test_case_name(test_obj)
            self.test_list.append(test_obj)
            if hasattr(test_obj.__wrapped__, "requires"):
                reqs = test_obj.__wrapped__.requires
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

    def add_dependency(self, case, depends_on):
        """Add a dependency: the case depends on depends_on."""
        self.dependencies[depends_on].append(case)

    def get_test_case_name(self, test_obj):
        """Return the test case name for structuring"""
        return (
            f"{test_obj.__wrapped__.attr.category}:"
            + f"{test_obj.__wrapped__.attr.name}"
        )

    def get_test_obj_by_name(self, name):
        """Return the test object by test case name"""
        return None if not self.test_map else self.test_map.get(name)

    def progress_delete_time(self, item, status, save=True):
        """
        Delete time stamp for a test item.
        item: test case name in string.
        status: 'Start', 'Completed' or custom status for test case itself.
        """
        for e in self.progress:
            if item in e:
                e[item].pop(status, None)

        if save:
            self.progress_save()

    def progress_check(self, item, k) -> bool:
        """Check if key exists in specific test item"""
        for ti in self.progress:
            if item in ti and k in ti[item]:
                return True
        return False

    def progress_update(self, item, k, v, save=True):
        """
        Save test progress.
        item: test case name in string.
        k: key to save in dict.
        v: value to save in dict.
        """
        if any(item in e for e in self.progress):
            for e in self.progress:
                if item in e:
                    e[item].update({k: v})
        else:
            self.progress.append({item: {k: v}})

        if save:
            self.progress_save()

    def progress_read(self, path=None):
        """Load the test progress from file."""
        resume_test_list = []
        if path is None:
            with open(
                f"{self.log_dir}/test_progress.json", "r", encoding="UTF-8"
            ) as f:
                self.progress = json.load(f)
        else:
            with open(
                f"{path}/glt-test-suite/test_progress.json",
                "r",
                encoding="UTF-8",
            ) as f:
                self.progress = json.load(f)

        # Apply the previous results
        for item in self.progress:
            if item.get("Summary"):
                self.test_tags = set(item.get("Summary").get("TestTags"))
                self.test_categories = item.get("Summary").get("Categories")
                continue

            for case, status in item.items():
                if not case.startswith("test_"):
                    continue
                obj_name = f"{status['Category']}:{case}"
                test_obj = self.get_test_obj_by_name(obj_name)
                attr = test_obj.__wrapped__.attr
                if "Result" in status:
                    attr.result = (
                        GLTTestResult(status["Result"]),
                        status["ResultReason"],
                    )
                resume_test_list.append(test_obj)

        # Update the test list to only include previous session's tests.
        self.test_list = resume_test_list

    def progress_record_time(self, item, status, save=True):
        """
        Save time stamp for a test item.
        item: test case name in string.
        status: 'Start', 'Completed' or custom status for test case itself.
        """
        current_time = datetime.now().strftime(glt_logger.LOGGING_DATE_FORMAT)
        self.progress_update(item, status, current_time, save)

    def progress_save(self):
        """Save test progress to filesystem."""
        with open(
            f"{self.log_dir}/test_progress.json", "w", encoding="UTF-8"
        ) as f:
            json.dump(self.progress, f, indent=4)

    def sort(self):
        """Topological sort nodes and cover it as test_list."""
        # Count in-degrees (number of dependencies) for each node
        in_degree = {
            self.get_test_case_name(node): 0 for node in self.test_list
        }
        for case in self.dependencies:
            for depends_on in self.dependencies[case]:
                in_degree[depends_on] += 1

        # Collect nodes with no dependencies
        queue = deque(self.test_list)
        total_nodes = len(self.test_list)
        self.test_list = []

        while queue:
            node = queue.popleft()
            test_case_name = self.get_test_case_name(node)
            if in_degree[test_case_name] == 0:
                for depends_on in self.dependencies[test_case_name]:
                    in_degree[depends_on] -= 1
                self.test_list.append(node)
            else:
                queue.append(node)

        if len(self.test_list) != total_nodes:
            raise ValueError("Cycle detected in dependencies")

        for case in self.test_list:
            name = self.get_test_case_name(case)
            self.test_map[name] = case
