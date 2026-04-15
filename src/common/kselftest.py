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
This is a python module to defines constants and functions for building
kselftest test cases.
"""

import logging
import os
import sys
import subprocess
import traceback


# Define all kselftest target test cases
TEST_TARGETS = ["size", "rtc/property", "kvm"]
ALL_KSELFTEST_TARGET_TESTS = f"TARGETS='{' '.join(TEST_TARGETS)}'"


class Kselftest:
    """This class provides functionalities for running kselftests."""

    def __init__(self, log_file):
        """Init process for Kselftest class.
            1. Check if linux root directory is valid or not.
            2. Create log file if needed.

        Args:
            log_file (str): The log file name.
        """

        # Get linux build root directory from linux_env module
        self.linux_root_directory = "/usr/local/share/greenlit/linux"

        # Setup logging file if provided
        if log_file:
            logging.basicConfig(filename=log_file, level=logging.INFO)

            # Open the log file in write mode and reset all logs
            with open(log_file, "w", encoding="latin-1") as log_file_handle:
                log_file_handle.write("Log file reset at script start.\n")

    def __del__(self):
        self.linux_root_directory = None

    @staticmethod
    def execute_command(command, cwd):
        """Executes a shell command and returns its output.

        Args:
            command (str): The shell command to execute.
            cwd (str): The directory to use as the working directory.

        Returns:
            True: if no errors.
            False: if detect errors during command execution.
        """

        if not os.path.isdir(cwd):
            print(f"Folder '{cwd}' does not exist.")
            return False

        try:
            cmd_output = subprocess.run(
                command,
                shell=True,
                check=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cmd_output_str = (cmd_output.stdout +
                              cmd_output.stderr).decode("latin-1")
            print(cmd_output_str)
            logging.info(cmd_output_str)
        except (UnicodeDecodeError, subprocess.CalledProcessError) as error:
            print(traceback.format_exc())
            logging.error("traceback: %s", traceback.format_exc())
            print(error.stdout)
            logging.error("Error error.stdout: %s", error.stdout)
            print(error.stderr)
            logging.error("Error error.stderr: %s", error.stderr)
            return False

        return True

    def build_targets_test(self, target):
        """Executes a shell command to build kselftest case and returns
           its output.

        Args:
            target (str): The kselftest target test case to build.

        Returns:
            True: if no build errors.
            False: if detect build errors.

        Example usage:
            target = "TARGETS='kvm'"
            build_targets_test(target)
        """

        if "TARGETS=" not in target:
            logging.error("Error getting kselftest build target: %s", {target})
            return False

        if target == "TARGETS=all":
            target = ALL_KSELFTEST_TARGET_TESTS

        # Before building individual kselftest test case, we need to
        # build headers
        command = "make headers"
        logging.info(command)
        if not self.execute_command(command, self.linux_root_directory):
            return False

        # Build individual kselftest test cases
        command = "make -C tools/testing/selftests " + target
        logging.info(command)
        if not self.execute_command(command, self.linux_root_directory):
            return False

        return True

    def run_targets_test(self, target, root_permission=False):
        """Executes a shell command to run kselftest case and returns
           True or False.

        Args:
            target (str): The kselftest target test case to build.
            root_permission (bool): run the test with root permission or not.

        Returns:
            True: if no build errors.
            False: if detect build errors.

        Example usage:
            target = "TARGETS='kvm'"
            run_targets_test(target, True)
        """

        if "TARGETS=" not in target:
            logging.error("Error getting kselftest build target: %s", {target})
            return False

        # Run individual kselftest test cases
        command = "make -C tools/testing/selftests run_tests " + target
        if root_permission:
            command = "sudo " + command

        logging.info(command)
        if not self.execute_command(command, self.linux_root_directory):
            return False

        return True

    @staticmethod
    def check_log(log_file):
        """Check kselftest test log whether all test case pass.

        Args:
            log_file (str): The kselftest test log file.
            It should be in greenlit root directory.

        Returns:
            True: if all test pass.
            False: if detect one or more test failure.
        """

        # Check the log file is valid or not
        if not os.path.exists(log_file):
            print("log_file not found ...")
            return False

        # Use failure_counts to check how many test cases are fail
        failure_counts = 0

        # Read run_rtc.log file
        with open(log_file, "r", encoding="latin-1") as log_file_handle:
            # Check each line if includes TAP13 failure pattern "not ok"
            for line in log_file_handle:
                if "not ok" in line:
                    failure_counts += 1

        # Print how many test cases are fail.
        if failure_counts:
            print("Test fail.\n")
            print("The failure case number is : " + str(failure_counts))
            logging.info("Test fail.\n")
            logging.info("The failure case number is : %s",
                         str(failure_counts))
            return False

        print("All test pass.")
        logging.info("All test pass.")
        return True


def main():
    """
    Main entry point of this python script.

    Returns:
        0: We return 0 if no error.
        255: We return 255 if encounter any error.
    """

    # Get Kselftest object
    log_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".log"
    try:
        kselftest = Kselftest(log_file_name)
    except ValueError as error:
        print("Allocate Kselftest object error:", error)
        return 255

    # Build all kselftest target test cases
    if not kselftest.build_targets_test("TARGETS=all"):
        print("Build all supported kselftest test cases fail.")
        logging.info("Build all supported kselftest test cases fail.")
        return 255

    print("All test cases build complete.")
    logging.info("All test cases build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
