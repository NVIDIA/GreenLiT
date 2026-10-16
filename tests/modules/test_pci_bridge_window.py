"""Unit tests for PCI bridge window type classification."""
# pylint: disable=protected-access,wrong-import-position

import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.glt_core import GLTTestFail, GLTTestSkip, GLTTestWarn
from src.modules.pci import glt_test_pci_bridge_window as window


class TestPciBridgeWindow(GLTTestBase):
    """Classify bridge-window flags reported by the probe."""

    def _run_scored_windows(self, windows):
        """Run the testcase with probe rows already parsed."""
        logger = self.mock.Mock()
        with self.mock.patch.object(
            window, "_bridge_window_tooling_skip_reason", return_value=None
        ), self.mock.patch.object(
            window, "_unload_bridge_window_probe"
        ), self.mock.patch.object(
            window, "_build_and_load_bridge_window_probe", return_value=None
        ), self.mock.patch.object(
            window, "parse_probe_report", return_value=windows
        ), self.mock.patch.object(
            window.glt_utils, "run_command"
        ), self.mock.patch(
            "builtins.open", self.mock.mock_open(read_data="")
        ):
            window.test_pci_bridge_window_type(None, logger)
        return logger

    def test_all_typed_windows_skip(self):
        """Typed windows alone do not prove the selector left a slot alone."""
        with self.assertRaises(GLTTestSkip) as caught:
            self._run_scored_windows([
                ("0006:05:00.0", "io", 0x100),
            ])
        self.assertIn("not observable", str(caught.exception))

    def test_disabled_startalign_or_unset_without_type_is_claimed(self):
        """Sizing a typeless slot sets DISABLED, STARTALIGN, or UNSET."""
        self.assertEqual(window.classify_untyped_window(0), "clear")
        self.assertEqual(
            window.classify_untyped_window(window.IORESOURCE_DISABLED),
            "claimed",
        )
        self.assertEqual(
            window.classify_untyped_window(window.IORESOURCE_STARTALIGN),
            "claimed",
        )
        self.assertEqual(
            window.classify_untyped_window(window.IORESOURCE_UNSET),
            "claimed",
        )
        self.assertEqual(
            window.classify_untyped_window(
                window.IORESOURCE_UNSET | window.IORESOURCE_STARTALIGN
            ),
            "claimed",
        )

    def test_load_skip_still_runs_make_clean(self):
        """A build warning still cleans the module directory."""
        logger = self.mock.Mock()
        with self.mock.patch.object(
            window, "_bridge_window_tooling_skip_reason", return_value=None
        ), self.mock.patch.object(
            window, "_unload_bridge_window_probe"
        ), self.mock.patch.object(
            window,
            "_build_and_load_bridge_window_probe",
            return_value="cannot build",
        ), self.mock.patch.object(
            window.glt_utils, "run_command"
        ) as run:
            with self.assertRaises(GLTTestWarn):
                window.test_pci_bridge_window_type(None, logger)
        run.assert_called_with(
            ["make", "clean"],
            "Failed to clean the bridge-window probe",
            timeout=window._MAKE_TIMEOUT_SECS,
        )

    def test_probe_report_parses_bdf_slot_and_flags(self):
        """The proc report is one window per line."""
        windows = window.parse_probe_report(
            "0006:05:00.0 io 0x20080000\n"
            "\n"
        )
        self.assertEqual(
            windows,
            [("0006:05:00.0", "io", 0x20080000)],
        )

    def test_probe_report_rejects_a_bad_line(self):
        """A line that is not bdf, slot, and flags is not a window."""
        with self.assertRaises(ValueError):
            window.parse_probe_report("0006:05:00.0 pref 0x102201\n")

    def test_typed_window_is_not_a_sizing_claim(self):
        """A window that already has a type bit is outside this oracle."""
        self.assertEqual(window.classify_untyped_window(0x100), "typed")
        self.assertEqual(window.classify_untyped_window(0x200), "typed")
        self.assertEqual(
            window.classify_untyped_window(0x100 | window.IORESOURCE_DISABLED),
            "typed",
        )
        self.assertEqual(
            window.classify_untyped_window(0x100 | window.IORESOURCE_UNSET),
            "typed",
        )

    def test_verdict_fails_a_claimed_window_and_passes_a_clear_one(self):
        """UNSET without a type fails. Flags 0 passes. No bridges skips."""
        with self.assertRaises(GLTTestFail) as caught:
            self._run_scored_windows([
                (
                    "0006:05:00.0",
                    "io",
                    window.IORESOURCE_UNSET | window.IORESOURCE_STARTALIGN,
                ),
            ])
        detail = str(caught.exception)
        self.assertIn("0006:05:00.0 io 0x20080000", detail)
        self.assertIn(window._UPSTREAM_SHA, detail)
        self.assertIn(window._NVIDIA_SHA, detail)

        logger = self._run_scored_windows([
            ("0006:05:00.0", "io", 0),
        ])
        logger.info.assert_called_once()
        self.assertIn("0006:05:00.0 io 0x0", logger.info.call_args[0][2])

        with self.assertRaises(GLTTestSkip):
            self._run_scored_windows([])
