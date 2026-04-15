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
This module operates the configurations GLT needed.
"""

import os
import platform
import re
import subprocess

import yaml

from common import glt_utils
from common.nv_device import NvDev


def cond_to_func(logger, cond):
    """Return a function by parsing the condition"""
    ret = None
    if "script" in cond:

        def new_func(script=cond["script"]):
            try:
                result = subprocess.run(script, shell=True, executable="/bin/bash",
                                        check=False, capture_output=True, text=True)
                logger.debug(f"STDOUT: {result.stdout.strip()}")
                logger.debug(f"STDERR: {result.stderr.strip()}")
                if result.returncode != 0:
                    logger.debug(f"Condition failed with {result.returncode}")
                return result.returncode == 0
            except FileNotFoundError:
                logger.debug(f"Command/File not found with {script}")
                return False
            except PermissionError:
                logger.debug("Permission denied: Check executable permissions.")
                return False
            except subprocess.TimeoutExpired:
                logger.debug("Command timed out with {script}")
                return False
            except OSError as e:
                logger.debug(f"OS error: {e}")
                return False

    elif "distro" in cond:
        rls = None
        kern = None
        if "release" in cond:
            rls = cond["release"]
        if "kernel-uname" in cond:
            kern = cond["kernel-uname"]

        def new_func(distro=cond["distro"], release=rls, kernel=kern):
            allowed = False

            if NvDev.get_release_info().get("NAME") == distro:
                # If no specific version requirements, allow if distro matches
                if release is None and kernel is None:
                    allowed = True
                else:
                    # Check release version if specified
                    if release is not None:
                        allowed = glt_utils.compare_version(NvDev.get_release_info().get("VERSION_ID"), release) == 0

                    # Check kernel version if specified
                    if allowed or release is None:
                        if kernel is not None:
                            kv = platform.release()
                            match = re.match(
                                r"^([<>=]+)\s*([0-9]+(?:\.[0-9a-z+_-]+)*(?:[+.-][a-z0-9+_-]+)*)$", kernel)

                            if not match:
                                raise ValueError(f"Invalid format: {kernel}")

                            operator = match.group(1)
                            tv = match.group(2)
                            res = glt_utils.compare_version(kv, tv)

                            if res is None:
                                raise ValueError("Invalid version: " + f"{kv} {kernel}")

                            # Evaluate version comparison based on operator
                            allowed = (
                                (operator == ">" and res > 0) or
                                (operator == ">=" and res >= 0) or
                                (operator == "<=" and res <= 0) or
                                (operator == "<" and res < 0) or
                                (operator == "==" and res == 0))

            return allowed

    ret = new_func
    return ret


def load_glt_yaml_files(logger, directory="/usr/share/greenlit/confs.d"):
    """Load yaml fils from GLT data directory"""
    yaml_data_list = []
    yaml_files = sorted(f for f in os.listdir(directory) if f.endswith((".yaml", ".yml")))

    for file_name in yaml_files:
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as yaml_file:
                data = yaml.safe_load(yaml_file)
                yaml_data_list.append(data)
                logger.debug(f"Loaded: {file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
        except PermissionError:
            logger.error(f"Permission denied for file: {file_path}")
        except OSError as e:
            logger.error(f"OS error while accessing {file_path}: {e}")

    logger.debug(yaml.dump(yaml_data_list, sort_keys=False, default_flow_style=False, width=80))

    return yaml_data_list
