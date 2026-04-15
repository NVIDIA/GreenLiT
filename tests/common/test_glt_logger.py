"""Unit test script for common/glt_logger.py"""

import os
import logging
import tempfile
import shutil

from tests.test_base import GLTTestBase
from src.common.glt_logger import (
    ColorFormatter,
    get_logger_dir,
    get_logger,
    verbose,
)
from src.common import glt_core as GLTCore


class TestGLTLogger(GLTTestBase):
    """Test Class for GLT Logger"""

    def setUp(self):
        """Set up test environment"""
        super().setUp()
        # Create a temporary directory for log files
        self.temp_dir = tempfile.mkdtemp()
        self.module_name = "test_logger"

        # Reset the logging module to avoid interference between tests
        # First close all handlers to avoid ResourceWarning
        for logger in logging.Logger.manager.loggerDict.values():
            if hasattr(logger, "handlers"):
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

        # Then clear the logger dictionary
        logging.Logger.manager.loggerDict.clear()

        # Also close root logger handlers
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

    def tearDown(self):
        """Clean up after tests"""
        # Close all handlers to avoid ResourceWarning
        for logger in logging.Logger.manager.loggerDict.values():
            if hasattr(logger, "handlers"):
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

        # Also close root logger handlers
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        # Remove temporary directory and all its contents
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        super().tearDown()

    def test_color_formatter(self):
        """Test ColorFormatter class"""
        formatter = ColorFormatter("%(levelname)s: %(message)s")

        # Test different log levels
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Info message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        self.assertIn("\033[32mINFO\033[0m", formatted)  # Green for INFO

        record.levelno = logging.WARNING
        record.levelname = "WARNING"
        formatted = formatter.format(record)
        self.assertIn(
            "\033[33mWARNING\033[0m", formatted
        )  # Yellow for WARNING

        record.levelno = logging.ERROR
        record.levelname = "ERROR"
        formatted = formatter.format(record)
        self.assertIn("\033[31mERROR\033[0m", formatted)  # Red for ERROR

        # Test result colorization
        for result in (
            GLTCore.GLTTestResult.FAIL,
            GLTCore.GLTTestResult.PASS,
            GLTCore.GLTTestResult.SKIP,
            GLTCore.GLTTestResult.AFAIL,
            GLTCore.GLTTestResult.NA,
        ):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Test {result.value}",
                args=(),
                exc_info=None,
            )
            formatted = formatter.format(record)
            # Check that the result value is in the formatted string
            self.assertIn(result.value, formatted)
            # And that it has color codes
            self.assertIn("\033[", formatted)

    def test_get_logger_dir(self):
        """Test get_logger_dir function"""
        session = "/tmp/session"
        module = "test_module"

        log_dir = get_logger_dir(session, module)
        self.assertEqual(log_dir, os.path.join(session, "test_module"))

    def test_get_logger(self):
        """Test get_logger function"""
        # Create the directory structure needed for the log file
        log_dir = os.path.join(self.temp_dir, self.module_name)

        # Test logger creation
        logger = get_logger(self.temp_dir, self.module_name)

        # Verify logger properties
        self.assertEqual(logger.name, self.module_name)
        self.assertEqual(logger.level, logging.INFO)

        # Verify handlers
        self.assertEqual(len(logger.handlers), 2)

        # Check handler types
        handler_types = [type(h) for h in logger.handlers]
        self.assertIn(logging.handlers.RotatingFileHandler, handler_types)
        self.assertIn(logging.StreamHandler, handler_types)

        # Verify log file was created
        log_file = os.path.join(log_dir, f"{self.module_name}.log")
        self.assertTrue(os.path.exists(log_file))

        # Test logger reuse (should return same logger)
        logger2 = get_logger(self.temp_dir, self.module_name)
        self.assertIs(logger, logger2)
        self.assertEqual(len(logger2.handlers), 2)  # No new handlers added

    def test_verbose(self):
        """Test verbose function"""
        # Create a logger with default level (INFO)
        logger = get_logger(self.temp_dir, self.module_name)
        self.assertEqual(logger.level, logging.INFO)

        # Set to verbose
        verbose(logger)

        # Verify level changed to DEBUG
        self.assertEqual(logger.level, logging.DEBUG)

        # Verify all handlers are set to DEBUG
        for handler in logger.handlers:
            self.assertEqual(handler.level, logging.DEBUG)

        # Test that debug messages get through by checking the log file
        debug_message = "This is a debug message"
        logger.debug(debug_message)

        # Check the log file for the debug message
        log_file_path = os.path.join(
            self.temp_dir, self.module_name, f"{self.module_name}.log"
        )
        with open(log_file_path, "r", encoding="utf-8") as log_file:
            log_content = log_file.read()
            self.assertIn(debug_message, log_content)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
