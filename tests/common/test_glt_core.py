"""Unit test script for common/glt_core.py"""

import os

from tests.test_base import GLTTestBase
from src.common.glt_core import (
    GLTConditionalCheck,
    GLTAllowFailure,
    GLTConditionalSkip,
    GLTPlanType,
    GLTPlan,
    GLTTestAttr,
    GLTTestSkip,
    GLTTestFail,
    GLTTestWarn,
    GLTTestResult,
    GLTRequires,
    GLTScheduler,
)


class TestGLTCore(GLTTestBase):
    """Test Class for GLT Core"""

    def test_conditional_check(self):
        """Test the GLTConditionalCheck class"""

        # Define test functions
        def always_true():
            return True

        always_true.desc = "always returns true"

        def always_false():
            return False

        always_false.desc = "always returns false"

        def also_true():
            return True

        also_true.desc = "also returns true"

        def also_false():
            return False

        also_false.desc = "also returns false"

        # Test 1: Multiple functions in all categories that all pass
        check = GLTConditionalCheck()
        check.all_true.extend([always_true, also_true])
        check.all_false.extend([always_false, also_false])
        check.any_true.extend([always_true, also_true])
        check.any_false.extend([always_false, also_false])
        allow, reason = check.check()
        self.assertTrue(allow)
        self.assertIn("ALL TRUE", reason)
        self.assertIn("ALL FALSE", reason)
        self.assertIn("ANY TRUE", reason)
        self.assertIn("ANY FALSE", reason)

        # Test 2: all_true failure overrides other passing conditions
        check = GLTConditionalCheck()
        check.all_true.extend([always_true, always_false])  # This will fail
        check.all_false.extend([always_false, also_false])  # This would pass
        check.any_true.extend([always_true, also_true])  # This would pass
        check.any_false.extend([always_false, also_false])  # This would pass
        allow, reason = check.check()
        self.assertFalse(allow)
        self.assertIn("all_true returned False", reason)

        # Test 3: all_false failure overrides other passing conditions
        check = GLTConditionalCheck()
        check.all_true.extend([always_true, also_true])  # This would pass
        check.all_false.extend([always_false, always_true])  # This will fail
        check.any_true.extend([always_true, also_true])  # This would pass
        check.any_false.extend([always_false, also_false])  # This would pass
        allow, reason = check.check()
        self.assertFalse(allow)
        self.assertIn("all_false returned True", reason)

        # Test 4: any_true failure with other conditions passing
        check = GLTConditionalCheck()
        check.all_true.extend([always_true, also_true])  # This passes
        check.all_false.extend([always_false, also_false])  # This passes
        check.any_true.extend([always_false, also_false])  # This fails
        check.any_false.extend([always_false, also_false])  # This passes
        allow, reason = check.check()
        self.assertFalse(allow)
        self.assertIn("No function in any_true returned True", reason)

        # Test 5: any_false failure with other conditions passing
        check = GLTConditionalCheck()
        check.all_true.extend([always_true, also_true])  # This passes
        check.all_false.extend([always_false, also_false])  # This passes
        check.any_true.extend([always_true, also_true])  # This passes
        check.any_false.extend([always_true, also_true])  # This fails
        allow, reason = check.check()
        self.assertFalse(allow)
        self.assertIn("No function in any_false returned False", reason)

        # Test 6: Empty combinations with some conditions
        check = GLTConditionalCheck()
        check.all_true.append(always_true)  # This passes
        allow, reason = check.check()
        self.assertTrue(allow)
        self.assertIn("ALL TRUE", reason)

    def test_allow_failure(self):
        """Test the GLTAllowFailure class"""
        allow_failure = GLTAllowFailure()

        # Test with empty conditions
        can_fail, reason = allow_failure.can_fail()
        self.assertFalse(can_fail)
        self.assertEqual(reason, "")

        # Test with a condition that allows failure
        def always_true():
            return True

        always_true.desc = "always returns true"

        allow_failure.all_true.append(always_true)
        can_fail, reason = allow_failure.can_fail()
        self.assertTrue(can_fail)
        self.assertIn("ALL TRUE", reason)

    def test_conditional_skip(self):
        """Test the GLTConditionalSkip class"""
        conditional_skip = GLTConditionalSkip()

        # Test with empty conditions
        can_skip, reason = conditional_skip.can_skip()
        self.assertFalse(can_skip)
        self.assertEqual(reason, "")

        # Test with a condition that allows skipping
        def always_true():
            return True

        always_true.desc = "always returns true"

        conditional_skip.all_true.append(always_true)
        can_skip, reason = conditional_skip.can_skip()
        self.assertTrue(can_skip)
        self.assertIn("ALL TRUE", reason)

    def test_plan_type(self):
        """Test the GLTPlanType enum"""
        # Test enum values
        self.assertEqual(GLTPlanType.SANITY.value, "sanity")
        self.assertEqual(GLTPlanType.STANDARD.value, "standard")
        self.assertEqual(GLTPlanType.STRESS.value, "stress")

        # Test descriptions
        self.assertIn("Quick sanity check", GLTPlanType.SANITY.description)
        self.assertIn("Standard test cases", GLTPlanType.STANDARD.description)
        self.assertIn("Stress system", GLTPlanType.STRESS.description)
        self.assertIn(
            "may hang and require a power cycle",
            GLTPlanType.STANDARD.description,
        )
        self.assertIn(
            "may hang and require a power cycle",
            GLTPlanType.STRESS.description,
        )

    def test_plan(self):
        """Test the GLTPlan class"""
        # Test plan creation with valid type
        plan = GLTPlan("sanity")
        self.assertEqual(plan.plan_type, "sanity")

        # Test invalid plan type
        with self.assertRaises(ValueError):
            GLTPlan("invalid_plan")

        # Test the decorator functionality
        @GLTPlan("sanity")
        def test_func():
            return "test"

        # Verify the decorator adds plans attribute
        self.assertTrue(hasattr(test_func, "plans"))
        self.assertEqual(test_func.plans[0], "sanity")

        # Verify the function still works
        self.assertEqual(test_func(), "test")

        # Test multiple plans
        @GLTPlan("standard")
        @GLTPlan("stress")
        def multi_plan_func():
            return "multi"

        self.assertEqual(len(multi_plan_func.plans), 2)
        self.assertIn("standard", multi_plan_func.plans)
        self.assertIn("stress", multi_plan_func.plans)

    def test_test_attr(self):
        """Test the GLTTestAttr class"""

        # Test with explicit category and name
        @GLTTestAttr(cat="test_cat", name="test_name")
        def test_func():
            return "test"

        # Verify attributes are set on the function
        self.assertTrue(hasattr(test_func.__wrapped__, "attr"))
        self.assertEqual(test_func.__wrapped__.attr.category, "test_cat")
        self.assertEqual(test_func.__wrapped__.attr.name, "test_name")

        # Verify the function still works
        self.assertEqual(test_func(), "test")

        # Test auto-extraction of name
        @GLTTestAttr(cat="test_cat")
        def auto_name_func():
            return "auto_name"

        self.assertEqual(
            auto_name_func.__wrapped__.attr.name, "auto_name_func"
        )

        # Test with multiple decorators in real-world style
        @GLTTestAttr(cat="arm64")
        @GLTPlan("sanity")
        @GLTPlan("standard")
        def real_world_test():
            return "real_world_example"

        # Verify both plans are applied
        self.assertEqual(len(real_world_test.__wrapped__.attr.plans), 2)
        self.assertIn("sanity", real_world_test.__wrapped__.attr.plans)
        self.assertIn("standard", real_world_test.__wrapped__.attr.plans)
        self.assertEqual(real_world_test.__wrapped__.attr.category, "arm64")

        # Test default result
        self.assertEqual(
            real_world_test.__wrapped__.attr.result[0], GLTTestResult.NA
        )

    def test_extract_category(self):
        """Test the category extraction functionality"""

        # Create a mock frame/stack with controlled filepath
        mock_frame = self.mock.MagicMock()
        mock_valid_path = os.path.join(
            "src", "modules", "arm64", "glt_test_file.py"
        )
        mock_frame.filename = mock_valid_path

        # Mock inspect.stack() to return our controlled frame
        with self.mock.patch("inspect.stack") as mock_stack:
            mock_stack.return_value = [mock_frame, mock_frame]

            @GLTTestAttr()
            def test_func():
                pass

            # Test that category was extracted correctly
            self.assertEqual(test_func.__wrapped__.attr.category, "arm64")

        # Test with invalid path
        mock_frame.filename = "src/common/glt_test_file.py"
        with self.mock.patch("inspect.stack") as mock_stack:
            mock_stack.return_value = [mock_frame, mock_frame]

            # The decorator should raise ValueError with invalid path
            with self.assertRaises(ValueError):

                @GLTTestAttr()
                def test_func2():
                    pass

    def test_dump_attrs(self):
        """Test the dump_attrs method in GLTTestAttr"""

        # Define a condition function to use in decorators
        def test_condition():
            return True

        test_condition.desc = "test description"

        # Use decorator directly on test function
        @GLTTestAttr(cat="test_cat", name="test_name")
        def test_attrs_func():
            return "test_attrs"

        # Add conditions to the decorated function
        test_attrs_func.__wrapped__.attr.allow_failure.all_true.append(
            test_condition
        )
        test_attrs_func.__wrapped__.attr.conditional_skip.any_true.append(
            test_condition
        )

        # Set test result
        test_attrs_func.__wrapped__.attr.result = (
            GLTTestResult.PASS,
            "Passed successfully",
        )

        # Test with mock logger
        mock_logger = self.mock.MagicMock()
        test_attrs_func.__wrapped__.attr.dump_attrs(logger=mock_logger)
        mock_logger.debug.assert_called()

        # Test without logger
        output = []

        def mock_print(text):
            output.append(text)

        with self.mock.patch("builtins.print", mock_print):
            test_attrs_func.__wrapped__.attr.dump_attrs()

        # Verify output was generated
        self.assertTrue(len(output) > 0)

    def test_test_exceptions(self):
        """Test GLTTestSkip, GLTTestFail, and GLTTestWarn exceptions"""
        # Test GLTTestSkip
        with self.assertRaises(GLTTestSkip):
            raise GLTTestSkip("Skip reason")

        # Test GLTTestFail
        with self.assertRaises(GLTTestFail):
            raise GLTTestFail("Fail reason")

        # Test GLTTestWarn
        with self.assertRaises(GLTTestWarn):
            raise GLTTestWarn("Warn reason")

    def test_requires(self):
        """Test the GLTRequires class"""

        # Test initialization
        requires = GLTRequires()
        self.assertEqual(len(requires.all_true), 0)
        self.assertEqual(len(requires.all_false), 0)
        self.assertEqual(len(requires.any_true), 0)
        self.assertEqual(len(requires.any_false), 0)

        # Test decorators
        @GLTTestAttr(cat="test_cat", name="test_func")
        def test_func():
            return "test"

        # Create mock scheduler
        mock_sd = self.mock.MagicMock()
        mock_pass_attr = self.mock.MagicMock()
        mock_pass_attr.result = (GLTTestResult.PASS, "Passed")

        mock_fail_attr = self.mock.MagicMock()
        mock_fail_attr.result = (GLTTestResult.FAIL, "Failed")

        mock_pass_test = self.mock.MagicMock()
        mock_pass_test.__wrapped__ = self.mock.MagicMock()
        mock_pass_test.__wrapped__.attr = mock_pass_attr
        mock_pass_test.__wrapped__.attr.name = "pass_test"

        mock_fail_test = self.mock.MagicMock()
        mock_fail_test.__wrapped__ = self.mock.MagicMock()
        mock_fail_test.__wrapped__.attr = mock_fail_attr
        mock_fail_test.__wrapped__.attr.name = "fail_test"

        mock_sd.get_test_obj_by_name.side_effect = lambda x: (
            mock_pass_test if "pass" in x else mock_fail_test
        )

        # Test must_pass decorator
        decorated = GLTRequires.must_pass("category", "test_name")(test_func)
        self.assertTrue(hasattr(decorated.__wrapped__, "requires"))
        self.assertIn(
            "category:test_name",
            decorated.__wrapped__.requires.all_true
        )

        # Test must_fail decorator
        decorated = GLTRequires.must_fail("category", "test_name")(test_func)
        self.assertTrue(hasattr(decorated.__wrapped__, "requires"))
        self.assertIn(
            "category:test_name",
            decorated.__wrapped__.requires.all_false
        )

        # Test any_pass decorator
        decorated = GLTRequires.any_pass("category", "test_name")(test_func)
        self.assertTrue(hasattr(decorated.__wrapped__, "requires"))
        self.assertIn(
            "category:test_name",
            decorated.__wrapped__.requires.any_true
        )

        # Test any_fail decorator
        decorated = GLTRequires.any_fail("category", "test_name")(test_func)
        self.assertTrue(hasattr(decorated.__wrapped__, "requires"))
        self.assertIn(
            "category:test_name",
            decorated.__wrapped__.requires.any_false
        )

        # Test can_start logic with different conditions
        requires = GLTRequires()
        requires.all_true = ["pass_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertTrue(can_start)

        requires = GLTRequires()
        requires.all_true = ["fail_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("fail_test in all_true returned False", reason)

        requires = GLTRequires()
        requires.all_false = ["fail_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertTrue(can_start)

        requires = GLTRequires()
        requires.all_false = ["pass_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("pass_test in all_false returned True", reason)

        requires = GLTRequires()
        requires.any_true = ["pass_test", "fail_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertTrue(can_start)

        requires = GLTRequires()
        requires.any_true = ["fail_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertEqual(reason, "No function in any_true returned True")

        requires = GLTRequires()
        requires.any_false = ["fail_test", "pass_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertTrue(can_start)

        requires = GLTRequires()
        requires.any_false = ["pass_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertEqual(reason, "No function in any_false returned False")

    def test_requires_dependency_names(self):
        """GLTRequires exposes all referenced dependency test IDs."""
        requires = GLTRequires()
        requires.all_true = ["cat1:test_1"]
        requires.all_false = ["cat2:test_2"]
        requires.any_true = ["cat3:test_3"]
        requires.any_false = ["cat4:test_4"]

        self.assertEqual(
            requires.dependency_names(),
            ["cat1:test_1", "cat2:test_2", "cat3:test_3", "cat4:test_4"])

    def test_test_result(self):
        """Test the GLTTestResult enum"""
        # Test enum values
        self.assertEqual(GLTTestResult.SKIP.value, "SKIPPED")
        self.assertEqual(GLTTestResult.PASS.value, "PASSED")
        self.assertEqual(GLTTestResult.AFAIL.value, "Allowed FAILURE")
        self.assertEqual(GLTTestResult.FAIL.value, "FAILED")
        self.assertEqual(GLTTestResult.WARN.value, "WARNED")
        self.assertEqual(GLTTestResult.NA.value, "N/A")

        # Test colors method
        colors = GLTTestResult.colors()
        self.assertEqual(len(colors), 6)
        self.assertIn(GLTTestResult.SKIP, colors)
        self.assertIn(GLTTestResult.PASS, colors)
        self.assertIn(GLTTestResult.AFAIL, colors)
        self.assertIn(GLTTestResult.FAIL, colors)
        self.assertIn(GLTTestResult.WARN, colors)
        self.assertIn(GLTTestResult.NA, colors)

        # Test colorize method
        colored = GLTTestResult.colorize(GLTTestResult.FAIL)
        self.assertIn(GLTTestResult.FAIL.value, colored)
        self.assertIn("\033[31m", colored)  # Red color
        self.assertIn("\033[0m", colored)  # Reset color

        colored = GLTTestResult.colorize(GLTTestResult.PASS)
        self.assertIn(GLTTestResult.PASS.value, colored)
        self.assertIn("\033[32m", colored)  # Green color
        self.assertIn("\033[0m", colored)  # Reset color

        colored = GLTTestResult.colorize(GLTTestResult.SKIP)
        self.assertIn(GLTTestResult.SKIP.value, colored)
        self.assertIn("\033[33m", colored)  # Yellow color
        self.assertIn("\033[0m", colored)  # Reset color

        colored = GLTTestResult.colorize(GLTTestResult.AFAIL)
        self.assertIn(GLTTestResult.AFAIL.value, colored)
        self.assertIn("\033[34m", colored)  # Blue color
        self.assertIn("\033[0m", colored)  # Reset color

        colored = GLTTestResult.colorize(GLTTestResult.NA)
        self.assertIn(GLTTestResult.NA.value, colored)
        self.assertIn("\033[35m", colored)  # Magenta color
        self.assertIn("\033[0m", colored)  # Reset color

        colored = GLTTestResult.colorize(GLTTestResult.WARN)
        self.assertIn(GLTTestResult.WARN.value, colored)
        self.assertIn("\033[36m", colored)  # Cyan color
        self.assertIn("\033[0m", colored)  # Reset color

        # Test skip static method with custom function
        @GLTTestAttr(cat="test_cat")
        def skip_test_func():
            return "should be skipped"

        # We need to suppress the exception for testing
        try:
            GLTTestResult.skip("Skip reason", skip_test_func)
        except GLTTestSkip:
            pass

        # Verify the result was set
        self.assertEqual(
            skip_test_func.__wrapped__.attr.result[0], GLTTestResult.SKIP
        )
        self.assertEqual(
            skip_test_func.__wrapped__.attr.result[1], "Skip reason"
        )

        # Test fail static method with custom function
        @GLTTestAttr(cat="test_cat")
        def fail_test_func():
            return "should fail"

        # We need to suppress the exception for testing
        try:
            GLTTestResult.fail("Fail reason", fail_test_func)
        except GLTTestFail:
            pass

        # Verify the result was set
        self.assertEqual(
            fail_test_func.__wrapped__.attr.result[0], GLTTestResult.FAIL
        )
        self.assertEqual(
            fail_test_func.__wrapped__.attr.result[1], "Fail reason"
        )

        # Test warn static method with custom function
        @GLTTestAttr(cat="test_cat")
        def warn_test_func():
            return "should warn"

        try:
            GLTTestResult.warn("Warn reason", warn_test_func)
        except GLTTestWarn:
            pass

        # Verify the result was set
        self.assertEqual(
            warn_test_func.__wrapped__.attr.result[0], GLTTestResult.WARN
        )
        self.assertEqual(
            warn_test_func.__wrapped__.attr.result[1], "Warn reason"
        )

    def test_scheduler(self):
        """Test the GLTScheduler class"""

        # Create mock test objects with dependencies
        @GLTTestAttr(cat="test_cat", name="test1")
        def test1():
            return "test1"

        @GLTRequires.must_pass("test_cat", "test1")
        @GLTTestAttr(cat="test_cat", name="test2")
        def test2():
            return "test2"

        @GLTRequires.must_pass("test_cat", "test2")
        @GLTTestAttr(cat="test_cat", name="test3")
        def test3():
            return "test3"

        # Create a temporary directory for the scheduler
        with self.mock.patch("os.path.exists", return_value=True):
            with self.mock.patch("os.makedirs"):
                temp_dir = "/tmp/test_scheduler"

                # Initialize scheduler with test objects
                scheduler = GLTScheduler([test1, test2, test3], temp_dir)

                # Test initial state
                self.assertEqual(len(scheduler.test_list), 3)
                self.assertEqual(
                    len(scheduler.dependencies), 2
                )  # test1 and test2 are dependencies

                # Test get_test_case_name
                self.assertEqual(
                    scheduler.get_test_case_name(test1), "test_cat:test1"
                )

                # Test dependency tracking
                self.assertTrue("test_cat:test1" in scheduler.dependencies)
                self.assertTrue("test_cat:test2" in scheduler.dependencies)
                self.assertIn(
                    "test_cat:test2", scheduler.dependencies["test_cat:test1"]
                )
                self.assertIn(
                    "test_cat:test3", scheduler.dependencies["test_cat:test2"]
                )

                # Test topological sort
                with self.mock.patch.object(scheduler, "test_map", {}):
                    scheduler.sort()
                    # Verify correct order: test1 -> test2 -> test3
                    sorted_names = [
                        scheduler.get_test_case_name(obj)
                        for obj in scheduler.test_list
                    ]
                    self.assertEqual(
                        sorted_names,
                        ["test_cat:test1", "test_cat:test2", "test_cat:test3"],
                    )

                    # Verify test_map is populated
                    self.assertEqual(len(scheduler.test_map), 3)
                    self.assertEqual(
                        scheduler.get_test_obj_by_name("test_cat:test1"), test1
                    )

                # Test progress management
                with self.mock.patch("builtins.open", self.mock.mock_open()):
                    scheduler.progress_update("test1", "Status", "Running")
                    self.assertTrue(
                        scheduler.progress_check("test1", "Status")
                    )
                    scheduler.progress_record_time("test1", "Start")
                    self.assertTrue(scheduler.progress_check("test1", "Start"))
                    scheduler.progress_delete_time("test1", "Start")
                    self.assertFalse(
                        scheduler.progress_check("test1", "Start")
                    )

    def test_scheduler_sort_rejects_missing_dependency(self):
        """Scheduler sort reports dependencies that reference missing tests."""

        @GLTRequires.must_pass("test_cat", "test_missing")
        @GLTTestAttr(cat="test_cat", name="test_selected")
        def test_selected():
            return "selected"

        scheduler = GLTScheduler([test_selected])

        with self.assertRaises(ValueError) as ctx:
            scheduler.sort()

        self.assertIn("Dependency test_cat:test_missing", str(ctx.exception))
        self.assertIn("test_cat:test_selected", str(ctx.exception))

    def test_scheduler_sort_rejects_missing_dependent(self):
        """Scheduler sort reports dependency graph entries for missing tests."""

        @GLTTestAttr(cat="test_cat", name="test_dependency")
        def test_dependency():
            return "dependency"

        scheduler = GLTScheduler([test_dependency])
        scheduler.dependencies["test_cat:test_dependency"].append(
            "test_cat:test_missing_dependent")

        with self.assertRaises(ValueError) as ctx:
            scheduler.sort()

        self.assertIn(
            "Test test_cat:test_missing_dependent depends on "
            "test_cat:test_dependency",
            str(ctx.exception))
        self.assertIn("not in the scheduler", str(ctx.exception))

    def test_plan_nesting(self):
        """Test the GLTPlanType.get_all_nested method"""
        nested = GLTPlanType.get_all_nested(["sanity"])
        assert "sanity" in nested
        assert "ext_executable" in nested
        nested = GLTPlanType.get_all_nested(["standard"])
        assert "standard" in nested
        assert "ext_executable" in nested
        nested = GLTPlanType.get_all_nested(["ext_executable"])
        assert nested == {"ext_executable"}

    def test_requires_with_skipped_dependencies(self):
        """Test GLTRequires when dependency tests are skipped"""

        # Create mock scheduler and test objects
        mock_sd = self.mock.MagicMock()

        # Create mock attributes for different test results
        mock_pass_attr = self.mock.MagicMock()
        mock_pass_attr.result = (GLTTestResult.PASS, "Passed")
        mock_pass_attr.name = "pass_test"

        mock_fail_attr = self.mock.MagicMock()
        mock_fail_attr.result = (GLTTestResult.FAIL, "Failed")
        mock_fail_attr.name = "fail_test"

        mock_skip_attr = self.mock.MagicMock()
        mock_skip_attr.result = (GLTTestResult.SKIP, "Skipped")
        mock_skip_attr.name = "skip_test"

        mock_na_attr = self.mock.MagicMock()
        mock_na_attr.result = (GLTTestResult.NA, "Not Available")
        mock_na_attr.name = "na_test"

        mock_afail_attr = self.mock.MagicMock()
        mock_afail_attr.result = (GLTTestResult.AFAIL, "Allowed Failure")
        mock_afail_attr.name = "afail_test"

        # Create mock test objects
        def create_mock_test(attr):
            mock_test = self.mock.MagicMock()
            mock_test.__wrapped__ = self.mock.MagicMock()
            mock_test.__wrapped__.attr = attr
            return mock_test

        mock_pass_test = create_mock_test(mock_pass_attr)
        mock_fail_test = create_mock_test(mock_fail_attr)
        mock_skip_test = create_mock_test(mock_skip_attr)
        mock_na_test = create_mock_test(mock_na_attr)
        mock_afail_test = create_mock_test(mock_afail_attr)

        # Setup scheduler mock
        def get_test_by_name(name):
            if "pass" in name:
                return mock_pass_test
            if "afail" in name:  # Check this FIRST before "fail"
                return mock_afail_test
            if "fail" in name:
                return mock_fail_test
            if "skip" in name:
                return mock_skip_test
            if "na" in name:
                return mock_na_test
            return None

        mock_sd.get_test_obj_by_name.side_effect = get_test_by_name

        # Test 1: must_pass with skipped dependency - should prevent start
        requires = GLTRequires()
        requires.all_true = ["skip_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("skip_test in all_true returned False", reason)

        # Test 2: must_fail with skipped dependency - should prevent start
        requires = GLTRequires()
        requires.all_false = ["skip_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("skip_test in all_false returned True", reason)

        # Test 3: must_pass with NA dependency - should prevent start
        requires = GLTRequires()
        requires.all_true = ["na_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("na_test in all_true returned False", reason)

        # Test 4: must_fail with AFAIL dependency - should prevent start (AFAIL != FAIL)
        requires = GLTRequires()
        requires.all_false = ["afail_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("afail_test in all_false returned True", reason)

        # Test 5: any_pass with all skipped dependencies - should prevent start
        requires = GLTRequires()
        requires.any_true = ["skip_test", "na_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertEqual(reason, "No function in any_true returned True")

        # Test 6: any_fail with all skipped dependencies - should prevent start
        requires = GLTRequires()
        requires.any_false = ["skip_test", "na_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertEqual(reason, "No function in any_false returned False")

        # Test 7: any_pass with mixed results - should allow if at least one passes
        requires = GLTRequires()
        requires.any_true = ["skip_test", "pass_test", "fail_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertTrue(can_start)

        # Test 8: any_fail with mixed results - should allow if at least one fails
        requires = GLTRequires()
        requires.any_false = ["skip_test", "pass_test", "fail_test"]
        can_start, reason = requires.can_start(mock_sd)
        self.assertTrue(can_start)

        # Test 9: Complex scenario with multiple requirement types
        requires = GLTRequires()
        requires.all_true = ["pass_test"]  # This should pass
        requires.any_false = ["fail_test", "skip_test"]  # This should pass (fail_test fails)
        can_start, reason = requires.can_start(mock_sd)
        self.assertTrue(can_start)

        # Test 10: Complex scenario that should fail due to skipped all_true
        requires = GLTRequires()
        requires.all_true = ["pass_test", "skip_test"]  # This should fail due to skip_test
        requires.any_false = ["fail_test"]  # This would pass
        can_start, reason = requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("skip_test in all_true returned False", reason)

    def test_requires_edge_cases(self):
        """Test GLTRequires edge cases and error conditions"""

        # Test when get_test_obj_by_name returns None
        mock_sd = self.mock.MagicMock()
        mock_sd.get_test_obj_by_name.return_value = None

        requires = GLTRequires()
        requires.all_true = ["nonexistent_test"]

        # This should raise AttributeError when trying to access None.__wrapped__.attr
        with self.assertRaises(AttributeError):
            requires.can_start(mock_sd)

    def test_requires_decorators_comprehensive(self):
        """Comprehensive test of all GLTRequires decorators"""

        # Test 1: Single decorators work correctly - use fresh functions each time
        @GLTTestAttr(cat="test_cat", name="base_func1")
        def base_func1():
            return "base1"

        must_pass_func = GLTRequires.must_pass("dep_cat", "dep_test")(base_func1)
        self.assertIn("dep_cat:dep_test", must_pass_func.__wrapped__.requires.all_true)
        self.assertEqual(len(must_pass_func.__wrapped__.requires.all_false), 0)
        self.assertEqual(len(must_pass_func.__wrapped__.requires.any_true), 0)
        self.assertEqual(len(must_pass_func.__wrapped__.requires.any_false), 0)

        @GLTTestAttr(cat="test_cat", name="base_func2")
        def base_func2():
            return "base2"

        must_fail_func = GLTRequires.must_fail("dep_cat", "dep_test")(base_func2)
        self.assertIn("dep_cat:dep_test", must_fail_func.__wrapped__.requires.all_false)
        self.assertEqual(len(must_fail_func.__wrapped__.requires.all_true), 0)
        self.assertEqual(len(must_fail_func.__wrapped__.requires.any_true), 0)
        self.assertEqual(len(must_fail_func.__wrapped__.requires.any_false), 0)

        @GLTTestAttr(cat="test_cat", name="base_func3")
        def base_func3():
            return "base3"

        any_pass_func = GLTRequires.any_pass("dep_cat", "dep_test")(base_func3)
        self.assertIn("dep_cat:dep_test", any_pass_func.__wrapped__.requires.any_true)
        self.assertEqual(len(any_pass_func.__wrapped__.requires.all_true), 0)
        self.assertEqual(len(any_pass_func.__wrapped__.requires.all_false), 0)
        self.assertEqual(len(any_pass_func.__wrapped__.requires.any_false), 0)

        @GLTTestAttr(cat="test_cat", name="base_func4")
        def base_func4():
            return "base4"

        any_fail_func = GLTRequires.any_fail("dep_cat", "dep_test")(base_func4)
        self.assertIn("dep_cat:dep_test", any_fail_func.__wrapped__.requires.any_false)
        self.assertEqual(len(any_fail_func.__wrapped__.requires.all_true), 0)
        self.assertEqual(len(any_fail_func.__wrapped__.requires.all_false), 0)
        self.assertEqual(len(any_fail_func.__wrapped__.requires.any_true), 0)

        # Test 2: Multiple decorators of same type
        @GLTRequires.must_pass("cat1", "test1")
        @GLTRequires.must_pass("cat2", "test2")
        @GLTTestAttr(cat="multi_cat", name="multi_must_pass")
        def multi_must_pass():
            return "multi"

        self.assertEqual(len(multi_must_pass.__wrapped__.requires.all_true), 2)
        self.assertIn("cat1:test1", multi_must_pass.__wrapped__.requires.all_true)
        self.assertIn("cat2:test2", multi_must_pass.__wrapped__.requires.all_true)

        # Test 3: Mixed decorators
        @GLTRequires.must_pass("cat1", "test1")
        @GLTRequires.must_fail("cat2", "test2")
        @GLTRequires.any_pass("cat3", "test3")
        @GLTRequires.any_fail("cat4", "test4")
        @GLTTestAttr(cat="mixed_cat", name="mixed_requirements")
        def mixed_requirements():
            return "mixed"

        self.assertEqual(len(mixed_requirements.__wrapped__.requires.all_true), 1)
        self.assertEqual(len(mixed_requirements.__wrapped__.requires.all_false), 1)
        self.assertEqual(len(mixed_requirements.__wrapped__.requires.any_true), 1)
        self.assertEqual(len(mixed_requirements.__wrapped__.requires.any_false), 1)

        self.assertIn("cat1:test1", mixed_requirements.__wrapped__.requires.all_true)
        self.assertIn("cat2:test2", mixed_requirements.__wrapped__.requires.all_false)
        self.assertIn("cat3:test3", mixed_requirements.__wrapped__.requires.any_true)
        self.assertIn("cat4:test4", mixed_requirements.__wrapped__.requires.any_false)

        # Test 4: Decorator creates requires attribute if it doesn't exist
        def plain_func():
            return "plain"

        # Apply decorator to function without GLTTestAttr
        with self.assertRaises(AttributeError):
            # This should fail because plain_func doesn't have __wrapped__
            GLTRequires.must_pass("cat", "test")(plain_func)

    def test_requires_real_world_scenarios(self):
        """Test real-world scenarios for GLTRequires"""

        # Scenario 1: Hardware prerequisite test that gets skipped should block dependent tests
        @GLTTestAttr(cat="hardware", name="gpu_detection")
        def gpu_detection_test():
            return "gpu detected"

        @GLTRequires.must_pass("hardware", "gpu_detection")
        @GLTTestAttr(cat="performance", name="gpu_benchmark")
        def gpu_benchmark_test():
            return "benchmark complete"

        # Mock scheduler with skipped GPU detection
        mock_sd = self.mock.MagicMock()
        mock_gpu_attr = self.mock.MagicMock()
        mock_gpu_attr.result = (GLTTestResult.SKIP, "No GPU found")
        mock_gpu_attr.name = "gpu_detection"

        mock_gpu_test = self.mock.MagicMock()
        mock_gpu_test.__wrapped__ = self.mock.MagicMock()
        mock_gpu_test.__wrapped__.attr = mock_gpu_attr

        mock_sd.get_test_obj_by_name.return_value = mock_gpu_test

        # GPU benchmark should not start because GPU detection was skipped
        can_start, reason = gpu_benchmark_test.__wrapped__.requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("gpu_detection in all_true returned False", reason)

        # Scenario 2: Cleanup test that expects certain tests to fail
        @GLTTestAttr(cat="stress", name="memory_leak_test")
        def memory_leak_test():
            return "memory leak detected"

        @GLTRequires.must_fail("stress", "memory_leak_test")
        @GLTTestAttr(cat="cleanup", name="memory_cleanup")
        def memory_cleanup_test():
            return "memory cleaned"

        # If memory leak test was skipped (not failed), cleanup shouldn't run
        mock_leak_attr = self.mock.MagicMock()
        mock_leak_attr.result = (GLTTestResult.SKIP, "Memory test skipped due to low memory")
        mock_leak_attr.name = "memory_leak_test"

        mock_leak_test = self.mock.MagicMock()
        mock_leak_test.__wrapped__ = self.mock.MagicMock()
        mock_leak_test.__wrapped__.attr = mock_leak_attr

        mock_sd.get_test_obj_by_name.return_value = mock_leak_test

        can_start, reason = memory_cleanup_test.__wrapped__.requires.can_start(mock_sd)
        self.assertFalse(can_start)
        self.assertIn("memory_leak_test in all_false returned True", reason)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
