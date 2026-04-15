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
Logger module for GreenLit.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from common import glt_core as GLTCore


LOGGING_FORMAT = "%(asctime)s - [%(levelname)s] %(name)s: %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_LEVEL = logging.INFO


def is_piped():
    """Check if stderr is being piped or redirected"""
    return not sys.stderr.isatty()


class GltFormatter(logging.Formatter):
    """Custom formatter to change the logger name format."""

    def format(self, record):
        original_name = record.name
        if ':' in original_name:
            prefix, test_name = original_name.split(':', 1)
            if '.' in prefix:
                module = prefix.split('.', 1)[-1]
                record.name = f"({module}) {test_name}"
            else:
                record.name = test_name

        result = super().format(record)

        record.name = original_name
        return result


class ColorFormatter(GltFormatter):
    """Color the console output for different level"""
    COLORS = {
        logging.DEBUG: "\033[34m",      # Blue
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[41m",   # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        base_message = super().format(record)
        # Color the level name
        level_name = record.levelname
        color = self.COLORS.get(record.levelno, self.RESET)
        colored_level_name = f"{color}{level_name}{self.RESET}"

        base_message = base_message.replace(level_name, colored_level_name, 1)

        # Color the result if found
        for result in (GLTCore.GLTTestResult.FAIL,
                       GLTCore.GLTTestResult.PASS,
                       GLTCore.GLTTestResult.SKIP,
                       GLTCore.GLTTestResult.AFAIL,
                       GLTCore.GLTTestResult.NA):
            if result.value in base_message:
                base_message = base_message.replace(result.value,
                                                    GLTCore.GLTTestResult.colorize(result))
        return base_message


def get_logger_dir(session, module_name) -> str:
    """Return the logger directory path"""
    log_path = os.path.join(session, module_name)
    return log_path


def get_logger(session, module_name) -> logging.Logger:
    """Return a logger for different purpose"""
    logger = logging.getLogger(module_name)

    if logger.handlers:
        return logger

    logger.setLevel(LOGGING_LEVEL)

    # File handler
    log_path = os.path.join(session, get_logger_dir(session, module_name),
                            module_name + '.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_handler = RotatingFileHandler(log_path, mode='a',
                                       maxBytes=10**6, backupCount=5)
    file_handler.setLevel(LOGGING_LEVEL)
    file_formatter = GltFormatter(
        LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)

    # Stream handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOGGING_LEVEL)
    if is_piped():
        console_formatter = GltFormatter(
            LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)
    else:
        console_formatter = ColorFormatter(
            LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # Add your custom handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def verbose(logger):
    """Set verbose logger"""
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        h.setLevel(logging.DEBUG)
