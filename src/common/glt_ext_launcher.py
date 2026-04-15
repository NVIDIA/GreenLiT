"""
External test launcher.

This module is used to discover and run external test cases.
"""
import os
import subprocess
from enum import IntEnum
from common import glt_core as GLTCore


CATEGORY_NAME = "external"


class GLTExtShellExitCode(IntEnum):
    """GLT external shell exit codes"""
    PASS = 0
    FAIL = 1
    XFAIL = 2   # Expected failure (test is expected to fail and did fail)
    XPASS = 3   # Unexpected pass (test is expected to fail but passed)
    SKIP = 4


def _generate_test_func(exe_path, relative_name):
    def test_func(sd, logger):
        workdir = os.path.join(sd.log_dir, CATEGORY_NAME, relative_name)
        env = os.environ.copy()
        env["GTL_WORKDIR"] = workdir
        try:
            logger.info("Running %s", exe_path)
            result = subprocess.run(
                [exe_path],
                env=env,
                cwd=os.path.dirname(exe_path),
                check=False,
                capture_output=True)
            logger.debug("Result: %s", result)
            if result.returncode == GLTExtShellExitCode.PASS:
                test_func.attr.result = (
                    GLTCore.GLTTestResult.PASS,
                    f"Finished {exe_path}. Exit code: {result.returncode}. PASS. " +
                    "Stdout: " + result.stdout.decode("utf-8", errors="replace") +
                    ",Stderr: " + result.stderr.decode("utf-8", errors="replace"))
                logger.info(f"Finished {exe_path}: {test_func.attr.result[0].value}: {test_func.attr.result[1]}")
            elif result.returncode == GLTExtShellExitCode.XFAIL:
                test_func.attr.result = (
                    GLTCore.GLTTestResult.AFAIL,
                    f"Finished {exe_path}. Exit code: {result.returncode}. Test is expected to fail and did fail. " +
                    "Stdout: " + result.stdout.decode("utf-8", errors="replace") +
                    ",Stderr: " + result.stderr.decode("utf-8", errors="replace"))
                logger.info(f"Finished {exe_path}: {test_func.attr.result[0].value}: {test_func.attr.result[1]}")
            elif result.returncode == GLTExtShellExitCode.XPASS:
                GLTCore.GLTTestResult.fail(
                    f"Finished {exe_path}. Exit code: {result.returncode}. Test is expected to fail but did pass. " +
                    "Stdout: " + result.stdout.decode("utf-8", errors="replace") +
                    ",Stderr: " + result.stderr.decode("utf-8", errors="replace"),
                    func=test_func)
            elif result.returncode == GLTExtShellExitCode.SKIP:
                GLTCore.GLTTestResult.skip(
                    f"Finished {exe_path}. Exit code: {result.returncode}. Test is expected to skip. " +
                    "Stdout: " + result.stdout.decode("utf-8", errors="replace") +
                    ",Stderr: " + result.stderr.decode("utf-8", errors="replace"),
                    func=test_func)
            else:
                GLTCore.GLTTestResult.fail(
                    f"Finished {exe_path}. Exit code: {result.returncode}. Test failed with unexpected exit code. " +
                    "Stdout: " + result.stdout.decode("utf-8", errors="replace") +
                    ",Stderr: " + result.stderr.decode("utf-8", errors="replace"),
                    func=test_func)
        except (OSError, ValueError) as e:
            GLTCore.GLTTestResult.fail(
                f"Finished {exe_path}. Test failed with unexpected error: {e}",
                func=test_func)
    return test_func


def discover(current_dir, root_dir) -> list[callable]:
    '''Recursively discover all executables under current_dir (DFS).

    Discovery is session-independent: the workdir for each external test
    is resolved lazily from scheduler.log_dir at execution time.
    '''
    tests = []
    dirs = []
    files = []
    entries = sorted(os.listdir(current_dir))
    for entry in entries:
        full_path = os.path.join(current_dir, entry)
        if os.path.isdir(full_path):
            dirs.append(entry)
        elif os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            files.append(entry)
    # Recurse into subdirectories first (DFS)
    for d in dirs:
        tests.extend(discover(os.path.join(current_dir, d), root_dir))
    # Then process files (a-z)
    for fname in files:
        fpath = os.path.join(current_dir, fname)
        relpath_to_name = os.path.relpath(fpath, root_dir).replace(os.sep, "_").replace(".", "_")
        func = _generate_test_func(fpath, relpath_to_name)
        attr = GLTCore.GLTTestAttr(cat=CATEGORY_NAME, name=relpath_to_name, func=func)
        attr.plans = [GLTCore.GLTPlanType.EXT_EXECUTABLE.value]
        func.attr = attr
        func.__name__ = relpath_to_name
        wrapped_func = attr(func)
        tests.append(wrapped_func)
    return tests
