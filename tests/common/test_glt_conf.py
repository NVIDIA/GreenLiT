# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test script for common/glt_conf.py"""

import os
import tempfile
import textwrap

from pydantic import ValidationError

from tests.test_base import GLTTestBase
from src.common.glt_conf import GLTConfConditionFields, cond_to_func, load_glt_yaml_files


class TestGLTConf(GLTTestBase):
    """Test Class for GLT Conf"""

    def setUp(self):
        super().setUp()
        self.test_yaml_dir = os.path.join(self.test_data_dir, "yaml")

    def test_cond_to_func_script(self):
        """Ensure the filter function is working by using bash script"""
        # Test successful script execution
        condition = GLTConfConditionFields.model_validate({"script": "echo 'test'"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertTrue(func())

        # Test failed script execution
        condition = GLTConfConditionFields.model_validate({"script": "exit 1"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

        # Test non-existent command
        condition = GLTConfConditionFields.model_validate({"script": "nonexistentcommand"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

    @GLTTestBase.unittest.mock.patch(
        "platform.freedesktop_os_release", create=True
    )
    @GLTTestBase.unittest.mock.patch("builtins.open")
    def test_cond_to_func_distro(self, mock_open, mock_freedesktop):
        """Ensure the filter function is working by using distro"""
        release_info = {
            "NAME": "Ubuntu",
            "VERSION_ID": "22.04",
        }

        # Setup mock for platform.freedesktop_os_release
        mock_freedesktop.return_value = release_info

        # In case freedesktop_os_release is not used, also mock file reading
        mock_file = GLTTestBase.unittest.mock.MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.readlines.return_value = [
            'NAME="Ubuntu"\n',
            'VERSION_ID="22.04"\n',
        ]
        mock_open.return_value = mock_file

        # Run the tests
        condition = GLTConfConditionFields.model_validate({"distro": "Ubuntu"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertTrue(func())

        condition = GLTConfConditionFields.model_validate({"distro": "Ubuntu", "release": "22.04"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertTrue(func())

        condition = GLTConfConditionFields.model_validate({"distro": "Ubuntu", "release": "20.04"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

        # No VERSION_ID is a non-match, not an error that aborts selection.
        mock_freedesktop.return_value = {"NAME": "Ubuntu"}
        condition = GLTConfConditionFields.model_validate({"distro": "Ubuntu", "release": "22.04"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

        mock_freedesktop.return_value = {"NAME": "Ubuntu", "VERSION_ID": 2204}
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())

        with self.assertRaises(ValidationError):
            GLTConfConditionFields.model_validate({"distro": "Ubuntu", "release": 2204})

    def test_cond_to_func_distro_comparison_error(self):
        """An undecidable release rejects the whole condition without raising."""
        with self.mock.patch(
            "src.common.glt_conf.NvDev.get_release_info",
            return_value={"NAME": "Ubuntu", "VERSION_ID": "22.04"},
        ), self.mock.patch(
            "src.common.glt_conf.glt_utils.compare_version",
            side_effect=ValueError("Invalid version"),
        ):
            for kernel in (None, ">= 5.15.0"):
                with self.subTest(kernel=kernel):
                    self.mock_logger.reset_mock()
                    condition = GLTConfConditionFields(
                        distro="Ubuntu", release="bad version", kernel_uname=kernel,
                    )
                    self.assertFalse(cond_to_func(self.mock_logger, condition)())
                    # The kernel check is skipped once the release fails.
                    self.mock_logger.warning.assert_called_once()

            # An invalid condition must not prevent a later valid condition from matching.
            valid = GLTConfConditionFields(distro="Ubuntu")
            self.assertTrue(cond_to_func(self.mock_logger, valid)())

    @GLTTestBase.unittest.mock.patch("platform.release")
    @GLTTestBase.unittest.mock.patch(
        "platform.freedesktop_os_release", create=True
    )
    @GLTTestBase.unittest.mock.patch("builtins.open")
    def test_cond_to_func_kernel(
        self, mock_open, mock_freedesktop, mock_platform_release
    ):
        """Ensure the filter function is working by using kernel version"""
        # Setup mocks for OS release info
        release_info = {
            "NAME": "Ubuntu",
            "VERSION_ID": "22.04",
        }
        mock_freedesktop.return_value = release_info

        # In case freedesktop_os_release is not used, also mock file reading
        mock_file = GLTTestBase.unittest.mock.MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.readlines.return_value = [
            'NAME="Ubuntu"\n',
            'VERSION_ID="22.04"\n',
        ]
        mock_open.return_value = mock_file

        # Mock kernel version
        mock_platform_release.return_value = "5.15.0-56-generic"

        # Test kernel version comparison
        test_cases = [
            ({"distro": "Ubuntu", "kernel-uname": ">= 5.15.0"}, True),
            ({"distro": "Ubuntu", "kernel-uname": "> 5.16.0"}, False),
            ({"distro": "Ubuntu", "kernel-uname": "<= 5.15.0-57-generic"}, True),
            ({"distro": "Ubuntu", "kernel-uname": "< 5.14.0"}, False),
            ({"distro": "Ubuntu", "kernel-uname": "== 5.15.0-56-generic"}, True),
        ]

        for condition, expected in test_cases:
            func = cond_to_func(self.mock_logger, GLTConfConditionFields.model_validate(condition))
            self.assertEqual(
                func(), expected, f"Failed for condition: {condition}"
            )

    def test_cond_to_func_kernel_comparison_error(self):
        """An undecidable kernel rejects even a condition whose release matches."""
        with self.mock.patch(
            "src.common.glt_conf.NvDev.get_release_info",
            return_value={"NAME": "Ubuntu", "VERSION_ID": "22.04"},
        ), self.mock.patch(
            "src.common.glt_conf.platform.release", return_value="5.15.0-56-generic",
        ):
            for release in (None, "22.04"):
                with self.subTest(release=release), self.mock.patch(
                    "src.common.glt_conf.glt_utils.compare_version",
                    side_effect=(
                        [ValueError("Undecidable kernel")]
                        if release is None else [0, ValueError("Undecidable kernel")]
                    ),
                ):
                    self.mock_logger.reset_mock()
                    condition = GLTConfConditionFields(
                        distro="Ubuntu", release=release, kernel_uname=">= 5.15.0",
                    )
                    self.assertFalse(cond_to_func(self.mock_logger, condition)())
                    self.mock_logger.warning.assert_called_once()

    def test_cond_to_func_kernel_invalid_format(self):
        """Malformed kernel expressions are non-matches after a matching release."""
        with self.mock.patch(
            "src.common.glt_conf.NvDev.get_release_info",
            return_value={"NAME": "Ubuntu", "VERSION_ID": "22.04"},
        ), self.mock.patch(
            "src.common.glt_conf.glt_utils.compare_version", return_value=0,
        ):
            for kernel in ("", "not-a-version", ">= bad-version"):
                with self.subTest(kernel=kernel):
                    self.mock_logger.reset_mock()
                    condition = GLTConfConditionFields(
                        distro="Ubuntu", release="22.04", kernel_uname=kernel,
                    )
                    self.assertFalse(cond_to_func(self.mock_logger, condition)())
                    self.mock_logger.warning.assert_called_once()

    def test_cond_to_func_unsupported(self):
        """Unsupported conditions log a warning and return a False stub"""
        condition = GLTConfConditionFields.model_validate({"foo": "bar"})
        func = cond_to_func(self.mock_logger, condition)
        self.assertFalse(func())
        self.mock_logger.warning.assert_called_once()

    def test_load_glt_yaml_files(self):
        """Load the sample yaml file and check fields"""
        # Test successful yaml loading
        with self.mock.patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["00_test.yaml"]

            result = load_glt_yaml_files(self.mock_logger, self.test_yaml_dir)
            self.assertEqual(len(result), 1)
            for config in result[0].values():
                payload = config.model_dump(by_alias=True, exclude_none=True)
                for key, blocks in payload.items():
                    self.assertIn(
                        key,
                        ["allow-failure", "conditional-skip"],
                        f"Invalid key: {key}",
                    )
                    for block in blocks:
                        self.assertEqual(set(block), {"condition"})
                        for field in block["condition"]:
                            self.assertIn(
                                field,
                                [
                                    "name",
                                    "distro",
                                    "release",
                                    "kernel-uname",
                                    "script",
                                ],
                                f"Invalid key: {field}",
                            )

        # Test file not found
        with self.mock.patch("os.listdir") as mock_listdir, self.mock.patch(
            "builtins.open", side_effect=FileNotFoundError
        ):
            mock_listdir.return_value = ["test.yaml"]

            result = load_glt_yaml_files(self.mock_logger, "/test/dir")
            self.assertEqual(len(result), 0)

        # Test permission error
        with self.mock.patch("os.listdir") as mock_listdir, self.mock.patch(
            "builtins.open", side_effect=PermissionError
        ):
            mock_listdir.return_value = ["test.yaml"]

            result = load_glt_yaml_files(self.mock_logger, "/test/dir")
            self.assertEqual(len(result), 0)

    def test_load_glt_yaml_files_keeps_condition_blocks_with_extra_fields(self):
        """An extra field such as skip-reason does not drop the file."""
        contents = textwrap.dedent(
            """\
            arm64:test_one:
              notes: "ignored beside the operation"
              allow-failure:
                - condition:
                    name: "known"
                    distro: "Ubuntu"
                    skip-reason: "ignored inside the condition"
                  skip-reason:
                    - ignored beside the condition
            arm64:test_two:
              conditional-skip:
                - condition:
                    name: "other"
                    script: "true"
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "99_extra.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(contents)
            result = load_glt_yaml_files(self.mock_logger, directory)

        self.assertEqual(len(result), 1)
        document = result[0]
        self.assertEqual(
            document["arm64:test_one"].model_dump(
                by_alias=True, exclude_none=True, exclude_defaults=True,
            ),
            {"allow-failure": [{"condition": {"name": "known", "distro": "Ubuntu"}}]},
        )
        self.assertEqual(
            document["arm64:test_two"].model_dump(
                by_alias=True, exclude_none=True, exclude_defaults=True,
            ),
            {"conditional-skip": [{"condition": {"name": "other", "script": "true"}}]},
        )
        self.assertNotIn("notes", document["arm64:test_one"].model_dump())

    def test_load_glt_yaml_files_keeps_valid_blocks_when_one_is_malformed(self):
        """A bad condition block or test entry is omitted; other blocks still load."""
        contents = textwrap.dedent(
            """\
            arm64:test_kept:
              conditional-skip:
                - condition:
                    name: "valid skip"
                    distro: "Ubuntu"
                    release: "22.04"
            arm64:test_shape: 1
            arm64:test_mixed:
              allow-failure:
                - condition:
                    name: "unquoted release"
                    distro: "Ubuntu"
                    release: 22.04
                - condition:
                    name: "null release"
                    distro: "Ubuntu"
                    release:
                - condition:
                    name: "valid sibling"
                    script: "true"
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "99_mixed.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(contents)
            result = load_glt_yaml_files(self.mock_logger, directory)
            error_messages = [
                call.args[0] for call in self.mock_logger.error.call_args_list
            ]

        self.assertEqual(len(result), 1)
        document = result[0]
        self.assertEqual(
            document["arm64:test_kept"].model_dump(
                by_alias=True, exclude_none=True, exclude_defaults=True,
            ),
            {
                "conditional-skip": [{
                    "condition": {
                        "name": "valid skip",
                        "distro": "Ubuntu",
                        "release": "22.04",
                    },
                }],
            },
        )
        self.assertEqual(
            document["arm64:test_mixed"].model_dump(
                by_alias=True, exclude_none=True, exclude_defaults=True,
            ),
            {"allow-failure": [{"condition": {"name": "valid sibling", "script": "true"}}]},
        )
        self.assertNotIn("arm64:test_shape", document)
        self.assertNotIn("unquoted release", str(document))
        self.assertEqual(
            error_messages,
            [
                "Error parsing YAML file "
                f"{path}: arm64:test_shape: "
                "expected a test configuration mapping",
                "Error parsing YAML file "
                f"{path}: arm64:test_mixed allow-failure condition block 0: "
                "release must be a string",
                "Error parsing YAML file "
                f"{path}: arm64:test_mixed allow-failure condition block 1: "
                "release must be a string",
            ],
        )

    def test_load_glt_yaml_files_rejects_non_mapping_document(self):
        """A file whose top level is not a mapping is skipped entirely."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "99_list.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("- not a mapping\n")
            result = load_glt_yaml_files(self.mock_logger, directory)
            error_messages = [
                call.args[0] for call in self.mock_logger.error.call_args_list
            ]

        self.assertEqual(result, [])
        self.assertEqual(
            error_messages,
            [
                f"Error parsing YAML file {path}: "
                "expected a GreenLiT config mapping",
            ],
        )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
