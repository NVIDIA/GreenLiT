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

import logging
import os
import platform
import re
import subprocess
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import yaml

from common import glt_utils
from common.nv_device import NvDev


class _GLTConfModel(BaseModel):
    """Shared settings for GreenLiT YAML config models.

    Unknown keys are ignored so a site file can carry notes beside the schema.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        strict=True,
        validate_by_name=True,
        validate_by_alias=True,
    )


class GLTConfConditionFields(_GLTConfModel):
    """Fields inside one allow-failure or conditional-skip condition.

    A field is None when the YAML omits it. ``kernel_uname`` is the
    YAML key ``kernel-uname``.
    """

    name: str | None = None
    script: str | None = None
    distro: str | None = None
    release: str | None = None
    kernel_uname: str | None = Field(default=None, alias="kernel-uname")


class GLTConfConditionBlock(_GLTConfModel):
    """One allow-failure or conditional-skip condition block."""

    condition: GLTConfConditionFields


class GLTConfTestConditions(_GLTConfModel):
    """Allow-failure and conditional-skip condition blocks for one test.

    An empty list means that operation contributes no condition blocks.
    """

    allow_failure: list[GLTConfConditionBlock] = Field(
        default_factory=list,
        alias="allow-failure",
    )
    conditional_skip: list[GLTConfConditionBlock] = Field(
        default_factory=list,
        alias="conditional-skip",
    )


def _find_explicit_null_condition_field(value: object) -> str | None:
    """Return a known condition field set to null, or None when none is."""
    if not isinstance(value, dict):
        return None
    # model_fields is filled by the Pydantic metaclass, which pylint cannot see.
    for name, field in GLTConfConditionFields.model_fields.items():  # pylint: disable=no-member
        key = field.alias or name
        if key in value and value[key] is None:
            return key
    return None


def _format_validation_error(exc: ValidationError) -> str:
    """Return a short reason for a rejected config value."""
    parts: list[str] = []
    for err in exc.errors():
        location = err.get("loc", ())
        field = str(location[-1]) if location else "value"
        if err.get("type") == "string_type":
            parts.append(f"{field} must be a string")
        elif field == "condition":
            parts.append("condition is not a mapping")
        else:
            parts.append(f"{field}: {err.get('msg', 'invalid value')}")
    if not parts:
        return "invalid value"
    return "; ".join(parts)


# A non-string known field rejects that condition block. Dropping only the
# field would widen the match, so the rest of the block is omitted with it.
# Other condition blocks and other test entries in the same file are kept.
def _parse_condition_blocks(
    value: object, errors: list[str], where: str,
) -> list[GLTConfConditionBlock] | None:
    """Return valid condition blocks, or None when value is not a list.

    None means this operation is omitted. A malformed condition block is
    recorded in errors and omitted; sibling blocks are kept. An empty list
    means the value was a list with no valid condition blocks.
    """
    if not isinstance(value, list):
        errors.append(f"{where}: expected a list of condition blocks")
        return None
    blocks: list[GLTConfConditionBlock] = []
    for index, item in enumerate(value):
        block_where = f"{where} condition block {index}"
        # Extra fields beside condition are ignored, not a reason to drop the block.
        if not isinstance(item, dict) or "condition" not in item:
            errors.append(f"{block_where}: expected a mapping with a condition")
            continue
        null_field = _find_explicit_null_condition_field(item.get("condition"))
        if null_field is not None:
            errors.append(f"{block_where}: {null_field} must be a string")
            continue
        try:
            blocks.append(GLTConfConditionBlock.model_validate(item))
        except ValidationError as exc:
            errors.append(f"{block_where}: {_format_validation_error(exc)}")
    return blocks


def _parse_test_conditions(
    value: object, errors: list[str], where: str,
) -> GLTConfTestConditions | None:
    """Return one test's config, or None when value is not a mapping.

    None means this test entry is omitted. An operation that is not a list
    of condition blocks is omitted; the test's other operations are kept.
    """
    if not isinstance(value, dict):
        errors.append(f"{where}: expected a test configuration mapping")
        return None
    parsed: dict[str, list[GLTConfConditionBlock]] = {}
    for key, item in value.items():
        if key == "allow-failure":
            blocks = _parse_condition_blocks(item, errors, f"{where} allow-failure")
            if blocks is not None:
                parsed["allow_failure"] = blocks
        elif key == "conditional-skip":
            blocks = _parse_condition_blocks(item, errors, f"{where} conditional-skip")
            if blocks is not None:
                parsed["conditional_skip"] = blocks
    return GLTConfTestConditions(**parsed)


def _parse_test_conditions_map(
    value: object, errors: list[str],
) -> dict[str, GLTConfTestConditions] | None:
    """Return a config file mapping, or None when value is not a mapping.

    None means the file is skipped. A malformed condition block or test entry
    is recorded in errors and omitted; other entries in the file are kept.
    """
    if not isinstance(value, dict):
        return None
    document: dict[str, GLTConfTestConditions] = {}
    for test_id, body in value.items():
        if not isinstance(test_id, str):
            errors.append(f"{test_id!r}: test id is not a string")
            continue
        config = _parse_test_conditions(body, errors, test_id)
        if config is None:
            continue
        document[test_id] = config
    return document


def cond_to_func(
    logger: logging.Logger, cond: GLTConfConditionFields,
) -> Callable[[], bool]:
    """Return a function by parsing the condition"""
    if cond.script is not None:
        script = cond.script

        def new_func(script: str = script) -> bool:
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

    elif cond.distro is not None:
        distro = cond.distro
        release_value = cond.release
        kernel_value = cond.kernel_uname

        def new_func(
            distro: str = distro,
            release: str | None = release_value,
            kernel: str | None = kernel_value,
        ) -> bool:
            allowed = False

            if NvDev.get_release_info().get("NAME") == distro:
                # If no specific version requirements, allow if distro matches
                if release is None and kernel is None:
                    allowed = True
                else:
                    # The condition matches only when every specified check
                    # matches. The kernel check runs only when the release is
                    # omitted or matches.
                    release_matches = True
                    if release is not None:
                        release_matches = False
                        version_id = NvDev.get_release_info().get("VERSION_ID")
                        # A missing VERSION_ID does not match. Raising here would
                        # abort test selection in GLTConditionalCheck.check().
                        if isinstance(version_id, str):
                            try:
                                release_matches = (
                                    glt_utils.compare_version(version_id, release) == 0
                                )
                            except ValueError as error:
                                logger.warning(
                                    "Ignoring release condition %r: %s", release, error,
                                )

                    kernel_matches = True
                    if release_matches and kernel is not None:
                        kernel_matches = False
                        kv = platform.release()
                        match = re.match(
                            r"^([<>=]+)\s*([0-9]+(?:\.[0-9a-z+_-]+)*(?:[+.-][a-z0-9+_-]+)*)$",
                            kernel,
                        )

                        if not match:
                            logger.warning(
                                "Ignoring kernel-uname condition %r: invalid format", kernel,
                            )
                        else:
                            operator = match.group(1)
                            tv = match.group(2)
                            try:
                                res = glt_utils.compare_version(kv, tv)
                            except ValueError as error:
                                logger.warning(
                                    "Ignoring kernel-uname condition %r: %s", kernel, error,
                                )
                            else:
                                kernel_matches = (
                                    (operator == ">" and res > 0) or
                                    (operator == ">=" and res >= 0) or
                                    (operator == "<=" and res <= 0) or
                                    (operator == "<" and res < 0) or
                                    (operator == "==" and res == 0)
                                )

                    allowed = release_matches and kernel_matches

            return allowed

    else:
        logger.warning(f"Unsupported condition (ignored): {cond}")

        def new_func() -> bool:
            return False

    return new_func


def load_glt_yaml_files(
    logger: logging.Logger,
    directory: str | os.PathLike[str] = "/usr/share/greenlit/confs.d",
) -> list[dict[str, GLTConfTestConditions]]:
    """Load GreenLiT YAML config files from a directory.

    A file that is not a mapping is skipped. One malformed condition block is
    omitted and reported; valid condition blocks in that file are still returned.
    """
    yaml_data_list: list[dict[str, GLTConfTestConditions]] = []
    yaml_files = sorted(f for f in os.listdir(directory) if f.endswith((".yaml", ".yml")))

    for file_name in yaml_files:
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as yaml_file:
                errors: list[str] = []
                document = _parse_test_conditions_map(yaml.safe_load(yaml_file), errors)
                for detail in errors:
                    logger.error(f"Error parsing YAML file {file_path}: {detail}")
                if document is None:
                    logger.error(
                        f"Error parsing YAML file {file_path}: "
                        "expected a GreenLiT config mapping")
                    continue
                yaml_data_list.append(document)
                logger.debug(f"Loaded: {file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
        except PermissionError:
            logger.error(f"Permission denied for file: {file_path}")
        except OSError as e:
            logger.error(f"OS error while accessing {file_path}: {e}")

    logger.debug(yaml.dump(
        [
            {
                test_id: config.model_dump(by_alias=True, exclude_none=True)
                for test_id, config in document.items()
            }
            for document in yaml_data_list
        ],
        sort_keys=False,
        default_flow_style=False,
        width=80,
    ))

    return yaml_data_list
