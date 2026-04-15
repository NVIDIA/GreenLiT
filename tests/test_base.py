"""GLT Unit test base"""

import os
import unittest
from unittest import mock


class GLTTestBase(unittest.TestCase):
    """The base unittest class for GLT"""

    unittest = unittest

    def setUp(self):
        """Set up test fixtures"""
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.test_data_dir = os.path.join(self.project_root, "tests", "data")
        self.mock = mock
        self.mock_logger = mock.MagicMock()
        self.mock_sd = mock.MagicMock()

    def tearDown(self):
        """Clean up after tests"""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures"""

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level fixtures"""
