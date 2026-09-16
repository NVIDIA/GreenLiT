"""Unit tests for Aquantia Atlantic runtime diagnostics."""
# pylint: disable=protected-access

import tempfile
from pathlib import Path

from common import glt_core as GLTCore
from src.modules.pci import glt_test_atlantic as atlantic
from tests.test_base import GLTTestBase


ATLANTIC_FAILURE_LOG = """
BUG: unable to handle page fault for address
RIP: 0010:skb_add_rx_frag_netmem+0x29/0xd0
Call Trace:
 <IRQ>
 aq_ring_rx_clean+0x175/0xe60 [atlantic]
 aq_vec_poll+0x81/0x1d0 [atlantic]
 net_rx_action+0x337/0x420
"""

ATLANTIC_FAILURE_LOG_LEGACY_HELPER = """
kernel: skb_add_rx_frag+0x21/0xc0
kernel: Call trace:
kernel: __aq_ring_rx_clean+0x180/0xe70 [atlantic]
"""

ATLANTIC_FAILURE_LOG_WITH_REGISTER_DUMP = "\n".join(
    [
        "BUG: unable to handle page fault for address",
        "RIP: 0010:skb_add_rx_frag_netmem+0x29/0xd0",
        *(f"register dump line {line_number}" for line_number in range(21)),
        "Call Trace:",
        " <IRQ>",
        " aq_ring_rx_clean+0x175/0xe60 [atlantic]",
        " aq_vec_poll+0x81/0x1d0 [atlantic]",
    ]
)

ATLANTIC_FAILURE_LOG_BUILTIN_DRIVER = """
BUG: unable to handle page fault for address
RIP: 0010:skb_add_rx_frag_netmem+0x29/0xd0
Call Trace:
 aq_ring_rx_clean+0x175/0xe60
 net_rx_action+0x337/0x420
"""

ATLANTIC_NON_FRAME_MENTION_LOG = """
BUG: unable to handle page fault for address
RIP: 0010:skb_add_rx_frag_netmem+0x29/0xd0
Call Trace:
 other_driver_rx_clean+0x175/0xe60 [other_driver]
 aq_ring_rx_clean completed without failure
"""

ATLANTIC_WRONG_MODULE_FRAME_LOG = """
BUG: unable to handle page fault for address
RIP: 0010:skb_add_rx_frag_netmem+0x29/0xd0
Call Trace:
 aq_ring_rx_clean+0x175/0xe60 [other_driver]
"""

UNRELATED_SKB_FAILURE_LOG = """
BUG: unable to handle page fault for address
RIP: 0010:skb_add_rx_frag_netmem+0x29/0xd0
Call Trace:
 other_driver_rx_clean+0x175/0xe60 [other_driver]
"""

UNRELATED_ATLANTIC_LOG = """
atlantic 0001:06:00.0 eno3: atlantic: link change old 10000 new 0
aq_ring_rx_clean completed without failure
"""

SEPARATE_TRACE_LOG = """
BUG: unable to handle page fault for address
RIP: 0010:skb_add_rx_frag_netmem+0x29/0xd0
Call Trace:
 other_driver_rx_clean+0x175/0xe60 [other_driver]
---[ end trace 0000000000000000 ]---
BUG: unrelated later Atlantic trace
Call Trace:
 aq_ring_rx_clean+0x175/0xe60 [atlantic]
"""

class TestAtlanticHelpers(GLTTestBase):
    """Test device discovery and runtime-log signature helpers."""

    @staticmethod
    def _add_pci_device(root, bdf, driver=None):
        """Create the sysfs fields used by Atlantic device discovery."""
        device_path = root / bdf
        device_path.mkdir()
        if driver is not None:
            driver_path = root / "drivers" / driver
            driver_path.mkdir(parents=True, exist_ok=True)
            (device_path / "driver").symlink_to(driver_path)

    def test_finds_all_devices_bound_to_atlantic(self):
        """Select Atlantic-bound PCI devices without product or device-ID gates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pci_root = Path(temp_dir)
            self._add_pci_device(pci_root, "0000:01:00.0", "atlantic")
            self._add_pci_device(pci_root, "0000:02:00.0", "vfio-pci")
            self._add_pci_device(pci_root, "0000:03:00.0", "atlantic")
            self._add_pci_device(pci_root, "0000:04:00.0")

            devices = atlantic._find_bound_atlantic_devices(pci_root)

        self.assertEqual(devices, ["0000:01:00.0", "0000:03:00.0"])

    def test_surfaces_pci_enumeration_failure(self):
        """Do not confuse an unavailable PCI sysfs directory with no NIC."""
        missing_root = Path("/definitely/missing/pci-devices")
        with self.assertRaisesRegex(RuntimeError, "Cannot enumerate PCI devices"):
            atlantic._find_bound_atlantic_devices(missing_root)

    def test_detects_known_atlantic_failure_signature(self):
        """Match the production failure shape only when frames are cohesive."""
        self.assertTrue(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                ATLANTIC_FAILURE_LOG
            )
        )

    def test_detects_legacy_helper_and_double_underscore_cleaner(self):
        """Support equivalent frame names seen across kernel versions."""
        self.assertTrue(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                ATLANTIC_FAILURE_LOG_LEGACY_HELPER
            )
        )

    def test_detects_fault_site_before_long_register_dump(self):
        """Retain the upstream RIP marker across the pre-trace register dump."""
        self.assertTrue(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                ATLANTIC_FAILURE_LOG_WITH_REGISTER_DUMP
            )
        )

    def test_detects_builtin_atlantic_cleaner_frame(self):
        """Accept a frame without module attribution for a built-in driver."""
        self.assertTrue(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                ATLANTIC_FAILURE_LOG_BUILTIN_DRIVER
            )
        )

    def test_rejects_non_frame_atlantic_cleaner_mention(self):
        """Require a stack-frame shape when module attribution is absent."""
        self.assertFalse(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                ATLANTIC_NON_FRAME_MENTION_LOG
            )
        )

    def test_rejects_atlantic_symbol_attributed_to_another_module(self):
        """Do not accept an Atlantic-named frame attributed elsewhere."""
        self.assertFalse(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                ATLANTIC_WRONG_MODULE_FRAME_LOG
            )
        )

    def test_rejects_non_atlantic_skb_failure(self):
        """Do not blame Atlantic for another driver's skb fragment crash."""
        self.assertFalse(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                UNRELATED_SKB_FAILURE_LOG
            )
        )

    def test_rejects_atlantic_log_without_skb_fragment_failure(self):
        """Do not treat ordinary Atlantic messages as the failure signature."""
        self.assertFalse(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                UNRELATED_ATLANTIC_LOG
            )
        )

    def test_rejects_frames_from_separate_call_traces(self):
        """Do not join an skb crash from one trace to an Atlantic frame later."""
        self.assertFalse(
            atlantic._has_atlantic_rx_fragment_overflow_failure(
                SEPARATE_TRACE_LOG
            )
        )


class TestAtlanticRxFragmentOverflowFailureRetainedJournals(GLTTestBase):
    """Test result mapping for the passive Atlantic runtime diagnostic."""

    @staticmethod
    def _journal_boots(count):
        """Return synthetic retained boots with stable IDs."""
        return [
            atlantic.glt_utils.JournalBoot(
                index=number,
                boot_id=f"{number:032x}",
            )
            for number in range(count)
        ]

    def setUp(self):
        """Reset the decorated result and provide applicable defaults."""
        super().setUp()
        test_func = atlantic.test_atlantic_rx_fragment_overflow_failure_retained_journals
        test_func.attr.result = (
            GLTCore.GLTTestResult.NA,
            "",
        )
        self.boots = self._journal_boots(
            atlantic.MIN_RETAINED_BOOT_COUNT_FOR_HISTORY
        )
        self.mock_devices = self.mock.patch.object(
            atlantic,
            "_find_bound_atlantic_devices",
            return_value=["0000:01:00.0"],
        ).start()
        self.mock_list_boots = self.mock.patch.object(
            atlantic.glt_utils,
            "list_journal_boots",
            return_value=self.boots,
        ).start()
        self.mock_get_kernel_log = self.mock.patch.object(
            atlantic.glt_utils,
            "get_kernel_log",
            return_value="ordinary current-boot kernel log",
        ).start()
        self.mock_get_kernel_release = self.mock.patch.object(
            atlantic,
            "_get_kernel_release_for_boot",
            return_value="6.17.0-test",
        ).start()
        self.addCleanup(self.mock.patch.stopall)

    def _run_test(self):
        """Run the decorated diagnostic."""
        atlantic.test_atlantic_rx_fragment_overflow_failure_retained_journals(
            None,
            self.mock_logger,
        )

    def test_is_in_all_native_plans(self):
        """Keep the passive diagnostic in standard and stress."""
        self.assertCountEqual(
            atlantic.test_atlantic_rx_fragment_overflow_failure_retained_journals.attr.plans,
            ["standard", "stress"],
        )

    def test_skips_when_boot_history_cannot_be_listed(self):
        """Skip when retained boot history is unavailable."""
        self.mock_list_boots.side_effect = RuntimeError("journal unavailable")

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_test()

        self.assertIn("Cannot enumerate retained journal boots", str(context.exception))
        self.mock_devices.assert_not_called()
        self.mock_get_kernel_log.assert_not_called()

    def test_skips_without_bound_atlantic_device(self):
        """Scan retained logs before skipping an inactive current driver."""
        self.mock_devices.return_value = []

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_test()

        self.assertIn(
            "No PCI device is currently bound to the atlantic driver",
            str(context.exception),
        )
        self.mock_list_boots.assert_called_once_with()
        self.assertEqual(self.mock_get_kernel_log.call_count, len(self.boots))

    def test_skips_when_pci_discovery_fails(self):
        """Treat discovery failure as an unavailable diagnostic prerequisite."""
        self.mock_devices.side_effect = RuntimeError("PCI discovery failed")

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_test()

        self.assertIn("Cannot determine whether any PCI device", str(context.exception))
        self.mock_list_boots.assert_called_once_with()
        self.assertEqual(self.mock_get_kernel_log.call_count, len(self.boots))

    def test_skips_when_no_retained_kernel_log_is_readable(self):
        """Skip instead of claiming no failure when all logs are unreadable."""
        self.mock_get_kernel_log.side_effect = RuntimeError("journal unavailable")

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            self._run_test()

        self.assertIn("No retained boot kernel logs are readable", str(context.exception))
        self.mock_devices.assert_not_called()

    def test_fails_when_atlantic_failure_signature_is_observed(self):
        """Fail on a cohesive Atlantic signature before current device gating."""
        target_boot = self.boots[3]

        def get_log(*, boot_id):
            if boot_id == target_boot.boot_id:
                return ATLANTIC_FAILURE_LOG
            return "ordinary retained kernel log"

        self.mock_get_kernel_log.side_effect = get_log

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_test()

        self.assertIn(
            "Known Atlantic RX fragment-overflow failure signature observed",
            str(context.exception),
        )
        self.assertIn(target_boot.boot_id, str(context.exception))
        self.assertIn("6.17.0-test", str(context.exception))
        self.assertIn("failure observation only", str(context.exception))
        self.assertIn("does not prove the current running kernel", str(context.exception))
        self.mock_get_kernel_release.assert_called_once_with(target_boot.boot_id)
        self.mock_devices.assert_not_called()

    def test_warns_but_passes_with_limited_boot_history(self):
        """A small retained history is still PASS when no failure is observed."""
        self.boots = self._journal_boots(1)
        self.mock_list_boots.return_value = self.boots

        self._run_test()

        self.mock_logger.warning.assert_called_once_with(
            "Only %d readable retained boot journal(s) were checked; this is "
            "a limited observation window, not proof of patch presence.",
            1,
        )

    def test_warns_but_passes_with_some_unreadable_boots(self):
        """Unreadable retained boots are reported and not counted as clean."""
        self.boots = self._journal_boots(
            atlantic.MIN_RETAINED_BOOT_COUNT_FOR_HISTORY + 1
        )
        unreadable_boot = self.boots[2]
        self.mock_list_boots.return_value = self.boots

        def get_log(*, boot_id):
            if boot_id == unreadable_boot.boot_id:
                raise RuntimeError("journal missing")
            return "ordinary retained kernel log"

        self.mock_get_kernel_log.side_effect = get_log

        self._run_test()

        self.mock_logger.warning.assert_called_once_with(
            "Unable to read kernel logs for %d of %d retained journal "
            "boot(s); those boot(s) were not counted as clean. First "
            "unreadable boot ID(s): %s.",
            1,
            len(self.boots),
            unreadable_boot.boot_id,
        )

    def test_passes_as_failure_not_observed_only(self):
        """Pass without implying patch presence or exercised reachability."""
        self._run_test()

        self.mock_logger.info.assert_called_once()
        message, readable_count, listed_count, devices = (
            self.mock_logger.info.call_args.args
        )
        self.assertIn("failure signature was not observed", message)
        self.assertIn("does not prove upstream 5ffcb7b890f6", message)
        self.assertIn("exceptional RX condition was exercised", message)
        self.assertEqual(readable_count, len(self.boots))
        self.assertEqual(listed_count, len(self.boots))
        self.assertEqual(devices, "0000:01:00.0")
        self.mock_get_kernel_log.assert_has_calls(
            [
                self.mock.call(boot_id=boot.boot_id)
                for boot in self.boots
            ]
        )
        self.mock_logger.warning.assert_not_called()
