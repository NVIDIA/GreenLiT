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
The main module handles the CLI actions.
"""

import argparse
import glob
import importlib.util
import inspect
import json
import os
import platform
import re
import signal
import sys
from common import glt_conf as GLTConf
from common import glt_core as GLTCore
from common import glt_logger
from common import glt_utils
from common import glt_ext_launcher
from common import perf_utils


def _truncate_reason(text, max_len=120):
    """Return the first non-empty line of *text*, truncated to *max_len*.

    Keeps the summary report clean regardless of how verbose the
    original reason string is.  Full details are always in the log.
    """
    # Remove ANSI escape sequences before truncating to avoid malformed terminal outputs
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            if len(stripped) > max_len:
                return stripped[:max_len] + "..."
            return stripped
    return text.strip()[:max_len] if text.strip() else ""


def apply_glt_configs(scheduler, logger, target_categories):
    """Read the configs and apply to test_objs"""
    target = set(target_categories)
    yaml_data = GLTConf.load_glt_yaml_files(logger)
    for test_item, contents in yaml_data[0].items():
        category = test_item.split(":", 1)[0]
        if category not in target:
            logger.debug(f"Skipping {test_item} from config: category not in run targets.")
            continue
        obj = scheduler.get_test_obj_by_name(test_item)
        if not obj:
            raise ValueError(f"Not found {test_item} from test cases.")
        logger.debug(f"Detected {test_item} from config.")
        if "allow-failure" in contents:
            logger.debug(" contains allow-failure:")
            for cond_list in contents["allow-failure"]:
                logger.debug(f"  condition: {cond_list['condition']}")
                func = GLTConf.cond_to_func(logger, cond_list["condition"])
                setattr(func, "desc", cond_list["condition"]["name"])
                obj.__wrapped__.attr.allow_failure.any_true.append(func)
        if "conditional-skip" in contents:
            logger.debug(" contains conditional-skip:")
            for cond_list in contents["conditional-skip"]:
                logger.debug(f"  condition: {cond_list['condition']}")
                func = GLTConf.cond_to_func(logger, cond_list["condition"])
                setattr(func, "desc", cond_list["condition"]["name"])
                obj.__wrapped__.attr.conditional_skip.any_true.append(func)


def check_required_commands(logger):
    """Warn if some commands are missing in the system."""
    commands = ['awk', 'basename', 'cat', 'chmod', 'cp', 'cut', 'find',
                'bmc-info', 'gpiodetect', 'gpioinfo', 'grep', 'insmod', 'ipmitool',
                'journalctl', 'ln', 'ls', 'lspci', 'make', 'mkdir', 'modprobe', 'perf',
                'pkill', 'reboot', 'rm', 'rmmod', 'sed', 'systemctl', 'tar', 'touch',
                'udevadm', 'uname', 'wc', 'stress-ng']

    for cmd in commands:
        if not glt_utils.is_cli_available(cmd):
            logger.warning(f"Command '{cmd}' is missing in the system.")
        elif cmd == 'perf':
            perf_version = perf_utils.get_perf_version(logger)
            if perf_version:
                logger.info("perf userspace: %s", perf_version)
            warning = perf_utils.get_perf_kernel_version_warning(
                perf_version, platform.release())
            if warning:
                logger.warning(warning)


def check_required_files(logger):
    """Warn if required files are missing in the system."""
    systemd_files = [
        '/usr/local/lib/systemd/system/greenlit.service',
        '/usr/local/lib/systemd/system/greenlit.timer'
    ]

    for file_path in systemd_files:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} is missing.")

    linux_dir = '/usr/local/share/greenlit/linux'
    if not os.path.isdir(linux_dir):
        logger.warning(f"Linux source code is not in {linux_dir}.")


def check_root_privileges():
    """Check if the user is running with root privileges."""
    if os.geteuid() != 0:
        print("Error: GreenLiT must be run with root privileges.")
        print("Please run the command with sudo, e.g.:")
        print("  sudo glt-test-suite [options]")
        sys.exit(1)


def create_initial_parser():
    """Create the initial parser.

    add_help=False so that -h/--help falls through to the full parser
    (parse_arguments) which includes plan and category options.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--resume", nargs="?", const=-1, default=None, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ext-dir", default="/usr/share/greenlit/executing_dir")
    return parser


def get_cached_session_path(index):
    """Get sepecific session by index"""
    sessions = glt_utils.get_cache_session_list()
    total = len(sessions)
    if index < 0:
        index += total + 1

    if index == 0 or index > total:
        return None

    target = None
    for s in sessions:
        if index == 1:
            target = s
            break
        index -= 1

    return target


def get_target_categories(args, supported_categories):
    """Return the categories this run targets.

    Args:
        args: Parsed argparse namespace with category boolean flags.
        supported_categories: List of all known category names.

    Returns:
        List of target category names. If the user passed category
        flags, that subset is returned; otherwise falls back to all
        supported categories (default behavior).
    """
    selected = [cat for cat in supported_categories if getattr(args, cat)]
    return selected if selected else list(supported_categories)


def glt_signal_handler(logger, scheduler):
    """Wrapper of signal handler for GLT"""

    def signal_handler(signum, frame):
        """Singal handler for signal module"""
        del frame
        logger.info(f"Signal {signum} received.")

        scheduler.progress_save()
        # Print out the status
        data = scheduler.progress_read()
        logger.info(json.dumps(data, indent=4))
        os._exit(0)

    return signal_handler


def list_cached_session():
    """list cahced sessions for --list --resume"""
    sessions = glt_utils.get_cache_session_list()
    rcount = len(sessions) * -1
    count = 1
    for s in sessions:
        print(f"{count} ({rcount}): {s}")
        count += 1
        rcount += 1


def list_test_cases(scheduler):
    """Print filtered test cases to stdout for --list."""
    for test_obj in scheduler.test_list:
        attr = test_obj.__wrapped__.attr
        if attr.category not in scheduler.test_categories or \
                not set(attr.plans).intersection(scheduler.test_tags):
            continue
        print(f"{attr.category}:{attr.name}")


def parse_arguments(supported_categories):
    """Parsing CLI arguments"""
    parser = argparse.ArgumentParser(description="GreenLiT test suite")

    # General options
    general_group = parser.add_argument_group("General options")
    general_group.add_argument("--list", action="store_true", help="List available test cases")
    general_group.add_argument("--resume", nargs="?", const=-1, default=None, type=int,
                               help="Resume the specific session. (e.g. 1, 2 or -1 for the last session)")
    general_group.add_argument("--verbose", action="store_true", help="Verbose log (logging.debug)")
    general_group.add_argument(
        "--ext-dir",
        default="/usr/share/greenlit/executing_dir",
        help="Directory containing external executable tests (default: /usr/share/greenlit/executing_dir)")

    # Test plan options
    plan_group = parser.add_argument_group("Supported plans")
    test_plan = plan_group.add_mutually_exclusive_group(required=False)
    for plan in GLTCore.GLTPlanType:
        test_plan.add_argument(f"--{plan.value}", action="store_true", help=f"{plan.description}")

    # Category options
    category_group = parser.add_argument_group("Supported categories")
    for category in supported_categories:
        category_group.add_argument(f"--{category}", action="store_true", help=f"Perform {category} category cases.")

    return parser.parse_args()


# pylint: disable=too-many-branches
def perform_tests(logger, scheduler):
    """Launch test cases by categories"""
    if not scheduler.test_categories or not scheduler.test_tags:
        logger.error("categories: %s test tags: %s", scheduler.test_categories, scheduler.test_tags)
        return

    for test_obj in scheduler.test_list:
        attr = test_obj.__wrapped__.attr
        if attr.category not in scheduler.test_categories or not set(attr.plans).intersection(scheduler.test_tags):
            continue

        # Child logger carries the test identity in %(name)s
        test_logger = logger.getChild(f"{attr.category}:{attr.name}")

        if scheduler.progress_check(attr.name, "Completed"):
            test_logger.info("Test case already completed")
            continue

        # For resume cases, don't overwirte the tags
        if not scheduler.progress_check(attr.name, "Start"):
            scheduler.progress_record_time(f"{attr.name}", "Start")

        try:
            if hasattr(test_obj.__wrapped__, "requires"):
                res, reason = test_obj.__wrapped__.requires.can_start(scheduler)
                if not res:
                    attr.result = (GLTCore.GLTTestResult.SKIP, reason)
                    continue

            if hasattr(attr, "conditional_skip"):
                res, reason = attr.conditional_skip.can_skip()
                if res:
                    attr.result = (GLTCore.GLTTestResult.SKIP, reason)
                    continue

            attr.func(scheduler, test_logger)
            attr.result = (GLTCore.GLTTestResult.PASS, "")
        except GLTCore.GLTTestSkip as e:
            test_logger.info("[%s] %s", attr.result[0].value, e)
        except GLTCore.GLTTestFail as e:
            test_logger.info("[%s] %s", attr.result[0].value, e)
            if hasattr(attr.func, "allow_failure"):
                res, reason = attr.func.allow_failure.can_fail()
                if not res:
                    attr.result = (GLTCore.GLTTestResult.AFAIL, reason)
        except GLTCore.GLTTestWarn as e:
            test_logger.info("[%s] %s", attr.result[0].value, e)
        finally:
            scheduler.progress_update(attr.name, "Result", attr.result[0].value)
            scheduler.progress_update(attr.name, "ResultReason", attr.result[1])
            attr.dump_attrs(test_logger)
        scheduler.progress_record_time(f"{attr.name}", "Completed")

        post_hooks = getattr(attr, "post_hooks", None)
        if post_hooks is not None and isinstance(post_hooks, GLTCore.PostHooks):
            if attr.result[0] == GLTCore.GLTTestResult.PASS:
                post_hooks.postsuccess(scheduler, attr, test_logger)
            elif attr.result[0] == GLTCore.GLTTestResult.SKIP:
                post_hooks.postskip(scheduler, attr, test_logger)
            elif attr.result[0] == GLTCore.GLTTestResult.FAIL:
                post_hooks.postfail(scheduler, attr, test_logger)
            elif attr.result[0] == GLTCore.GLTTestResult.AFAIL:
                post_hooks.postafail(scheduler, attr, test_logger)
            elif attr.result[0] == GLTCore.GLTTestResult.WARN:
                post_hooks.postwarn(scheduler, attr, test_logger)
            # Hook may not return (e.g. reboot); no further tests this run.

    ret, reason = glt_utils.collect_logs(scheduler.log_dir)
    if not ret:
        logger.warning(f"Failed to collect logs: {reason}")

    show_report(logger, scheduler)
# pylint: enable=too-many-branches


def registry_test_cases(logger, scheduler):
    """Register all test cases."""
    if not scheduler.test_categories or not scheduler.test_tags:
        logger.error("categories: %s test tags: %s", scheduler.test_categories, scheduler.test_tags)
        return

    scheduler.progress_update('Summary', 'TestTags', list(scheduler.test_tags))

    scheduler.progress_update('Summary', 'Categories', scheduler.test_categories)

    for test_obj in scheduler.test_list:
        attr = test_obj.__wrapped__.attr
        if attr.category not in scheduler.test_categories or not set(attr.plans).intersection(scheduler.test_tags):
            continue

        scheduler.progress_record_time(f'{attr.name}', 'Registry')
        scheduler.progress_update(attr.name, 'Category', attr.category)


def resolve_plan_and_categories(args, scheduler, supported_categories):
    """Resolve test plan and categories from CLI arguments."""
    selected_plan = None
    for plan in GLTCore.GLTPlanType:
        if getattr(args, plan.value):
            selected_plan = plan.value
            break
    if selected_plan is None:
        selected_plan = GLTCore.GLTPlanType.SANITY.value

    scheduler.test_tags = GLTCore.GLTPlanType.get_all_nested(selected_plan)
    scheduler.test_categories = get_target_categories(args, supported_categories)


def resume_from_session(scheduler, session):
    """Resume status from previous session"""
    scheduler.progress_read(session)
    scheduler.progress_save()


def setup_session(initial_args):
    """Create or resume a session and configure the logger.

    Returns:
        tuple: (session path, logger)
    """
    if initial_args.resume is not None:
        session = get_cached_session_path(initial_args.resume)
    else:
        session = glt_utils.create_cache_session()

    logger = glt_logger.get_logger(session, "glt-test-suite")
    if initial_args.verbose:
        glt_logger.verbose(logger)

    return session, logger


def show_report(logger, scheduler):
    """Print the test report."""
    detail = ""
    p = f = af = s = w = 0
    for test_obj in scheduler.test_list:
        attr = test_obj.__wrapped__.attr
        reason = ""
        if attr.result[0].value == GLTCore.GLTTestResult.PASS.value:
            p += 1
        elif attr.result[0].value == GLTCore.GLTTestResult.FAIL.value:
            f += 1
            reason = f"- {_truncate_reason(attr.result[1])}"
        elif attr.result[0].value == GLTCore.GLTTestResult.AFAIL.value:
            af += 1
            reason = f"- {_truncate_reason(attr.result[1])}"
        elif attr.result[0].value == GLTCore.GLTTestResult.SKIP.value:
            s += 1
            reason = f"- {_truncate_reason(attr.result[1])}"
        elif attr.result[0].value == GLTCore.GLTTestResult.WARN.value:
            w += 1
            reason = f"- {_truncate_reason(attr.result[1])}"
        detail += f"\
\t[{attr.result[0].value}] {scheduler.get_test_case_name(test_obj)}\n"
        if reason:
            detail += f"\
\t\t{reason}\n"
    res = f"""
===========================================================

Summary:
\t{GLTCore.GLTTestResult.PASS.value:<18}: {p}
\t{GLTCore.GLTTestResult.FAIL.value:<18}: {f}
\t{GLTCore.GLTTestResult.AFAIL.value:<18}: {af}
\t{GLTCore.GLTTestResult.SKIP.value:<18}: {s}
\t{GLTCore.GLTTestResult.WARN.value:<18}: {w}

Details:
{detail}
==========================================================="""
    logger.info(res)


def test_case_discover(categories):
    """Discover test cases from selected category modules.

    Args:
        categories: List of category directory names to scan.

    Returns:
        List of discovered test function objects.
    """
    cur_dir = os.path.dirname(__file__)
    modules_dir = os.path.join(cur_dir, "..", "modules")
    test_objs = []

    for category in categories:
        if category == glt_ext_launcher.CATEGORY_NAME:
            continue

        category_path = os.path.join(modules_dir, category)
        for fullpath in glob.glob(os.path.join(category_path, "glt_test_*.py")):
            filepath = os.path.basename(fullpath)
            module_name = os.path.splitext(filepath)[0]

            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(module_name, fullpath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                test_objs.extend(test_case_init(module))

    return test_objs


def test_case_discover_with_prereqs(initial_categories, supported_categories):
    """Discover tests for @initial_categories, expanding the import set to
    cover any cross-category prerequisites referenced by their @requires
    decorators.
    """
    supported = set(supported_categories)
    loaded = set(initial_categories)
    pending = initial_categories
    test_objs = []
    while pending:
        new_objs = test_case_discover(pending)
        test_objs.extend(new_objs)
        referenced = set()
        for obj in new_objs:
            wrapped = getattr(obj, "__wrapped__", None)
            reqs = getattr(wrapped, "requires", None)
            if reqs is not None:
                referenced |= reqs.categories()
        pending = (referenced & supported) - loaded
        loaded |= pending
    return test_objs


def test_case_init(module):
    """Retrieve object of test function"""
    test_obj = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            test_obj.append(obj)
    return test_obj


def test_category_discover():
    """Discover test categories from module directory names.

    Scans subdirectories of src/modules/.  Each subdirectory name is a
    category.  No modules are imported.

    Returns:
        Sorted list of category name strings.
    """
    modules_dir = os.path.join(os.path.dirname(__file__), "..", "modules")
    categories = []
    for entry in sorted(os.listdir(modules_dir)):
        if entry.startswith("__") or entry.startswith("."):
            continue
        if os.path.isdir(os.path.join(modules_dir, entry)):
            categories.append(entry)
    return categories


def main():
    """Main function"""

    check_root_privileges()

    # Minimal pre-parse: we need --list, --resume, --verbose, and --ext-dir
    # before full argument parsing because categories are discovered dynamically.
    initial_parser = create_initial_parser()
    initial_args, _ = initial_parser.parse_known_args()

    # --list --resume: list cached sessions and exit (no discovery needed)
    if initial_args.resume is not None and initial_args.list:
        list_cached_session()
        return

    # --- Phase 1: Discover categories from filesystem without importing any modules ---

    supported_categories = test_category_discover()

    ext_dir = initial_args.ext_dir
    external_test_objs = []
    if os.path.isdir(ext_dir):
        external_test_objs = glt_ext_launcher.discover(ext_dir, ext_dir)
        if external_test_objs and glt_ext_launcher.CATEGORY_NAME not in supported_categories:
            supported_categories.append(glt_ext_launcher.CATEGORY_NAME)

    args = parse_arguments(supported_categories)

    # --- Phase 2: Resolve which categories user selected ---

    target_categories = get_target_categories(args, supported_categories)

    # --- Phase 3: Import only selected category modules ---

    test_objs = test_case_discover_with_prereqs(
        target_categories, supported_categories)

    if glt_ext_launcher.CATEGORY_NAME in target_categories:
        test_objs.extend(external_test_objs)

    # --- Phase 4: Build scheduler, resolve plan ---

    scheduler = GLTCore.GLTScheduler(test_objs)
    scheduler.sort()
    resolve_plan_and_categories(args, scheduler, supported_categories)

    # --- --list: clean output, no session, no side effects ---

    if initial_args.list:
        list_test_cases(scheduler)
        return

    # --- Phase 5: Create session and run tests ---

    session, logger = setup_session(initial_args)

    # Wire session into the scheduler (ext_test workdirs resolve lazily)
    scheduler.log_dir = f"{session}/glt-test-suite"

    apply_glt_configs(scheduler, logger, target_categories)

    signal.signal(signal.SIGINT, glt_signal_handler(logger, scheduler))
    signal.signal(signal.SIGTERM, glt_signal_handler(logger, scheduler))

    glt_utils.log_system_info(logger)

    if not glt_utils.is_systemd_running():
        logger.warning("PID1 is not systemd. Session resume will not work.")

    check_required_commands(logger)
    check_required_files(logger)

    logger.info("== Session %s %s ==", session,
                "attached" if initial_args.resume is not None else "started")

    if initial_args.resume is not None:
        resume_from_session(scheduler, session)
    else:
        registry_test_cases(logger, scheduler)
    perform_tests(logger, scheduler)

    logger.info("== Session %s completed ==", session)


if __name__ == "__main__":
    main()
