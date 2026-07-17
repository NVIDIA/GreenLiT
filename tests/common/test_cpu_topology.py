"""Unit tests for common/cpu_topology.py."""

from tests.test_base import GLTTestBase
from src.common.cpu_topology import parse_cpu_mask


class TestCpuTopology(GLTTestBase):
    """Test Linux CPU mask parsing helpers."""

    def test_parse_cpu_mask_empty_and_whitespace(self):
        """Empty or whitespace-only masks mean no CPUs, not a parse error."""
        self.assertEqual(parse_cpu_mask(""), set())
        self.assertEqual(parse_cpu_mask("   "), set())
        self.assertEqual(parse_cpu_mask("\t\n"), set())

    def test_parse_cpu_mask_handles_singles_ranges_and_lists(self):
        """Common sysfs masks must expand to the expected CPU set."""
        self.assertEqual(parse_cpu_mask("0"), {0})
        self.assertEqual(parse_cpu_mask("0-3"), {0, 1, 2, 3})
        self.assertEqual(parse_cpu_mask("0-2,8,10-11"), {0, 1, 2, 8, 10, 11})
        self.assertEqual(parse_cpu_mask(" 0-2 , 8 , 10-11 "), {0, 1, 2, 8, 10, 11})

    def test_parse_cpu_mask_rejects_reversed_range(self):
        """Reversed ranges are malformed and must fail loudly."""
        with self.assertRaises(ValueError):
            parse_cpu_mask("4-2")

    def test_parse_cpu_mask_rejects_empty_segments(self):
        """Empty segments in a non-empty mask are malformed, not silent skips."""
        for mask in ("0,,2", "0,", ",0", ","):
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    parse_cpu_mask(mask)
