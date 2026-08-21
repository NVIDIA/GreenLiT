"""Unit tests for Tegra410 PMU vendor metadata validation."""
# pylint: disable=duplicate-code,protected-access

from common import glt_core as GLTCore
from common import perf_utils
from modules.arm64 import glt_test_tegra410_pmu as tegra410_pmu
from tests.test_base import GLTTestBase


def _tegra410_perf_list_json():
    """Return representative patched Tegra410 perf-list JSON entries."""
    metric_groups = (
        (
            "L1D_Prefetcher_Effectiveness",
            (
                "l1d_cache_rw_miss_ratio",
                "l1d_demand_accesses",
                "l1d_demand_misses",
                "l1d_prf_accuracy",
                "l1d_prf_coverage",
                "l1d_refilled_prf",
                "l1d_useful_prf",
            ),
        ),
        (
            "L1I_Prefetcher_Effectiveness",
            (
                "l1i_cache_rd_miss_ratio",
                "l1i_demand_accesses",
                "l1i_demand_misses",
                "l1i_prf_accuracy",
                "l1i_prf_coverage",
                "l1i_refilled_prf",
                "l1i_useful_prf",
            ),
        ),
    )
    entries = [
        {"MetricName": metric_name, "MetricGroup": metric_group}
        for metric_group, metric_names in metric_groups
        for metric_name in metric_names
    ]
    entries.extend([
        {
            "EventName": "txreq_limit_below_half_cycles",
            "Encoding": "cpu/event=0x197/",
            "PublicDescription": (
                "Number of cycles in which the dynamic TXREQ limit is "
                "between 0 and 1/2 of the L2_TQ_SIZE."
            ),
        },
        {
            "EventName": "txreq_limit_count_cycles",
            "Encoding": "cpu/event=0x1cc/",
            "PublicDescription": (
                "Companion event of TXREQ_LIMIT_MAX_CYCLES and "
                "TXREQ_LIMIT_BELOW_HALF_CYCLES."
            ),
        },
    ])
    return entries


class TestTegra410Pmu(GLTTestBase):
    """Test Tegra410 PMU metadata helpers and decorated tests."""

    def _run_test_with_perf_list(self, test_case, perf_list_json):
        """Run one decorated metadata test with supplied perf-list JSON."""
        with self.mock.patch.object(
                tegra410_pmu, "NvDev") as nvdev_cls, \
                self.mock.patch.object(
                    perf_utils, "resolve_perf_command",
                    return_value=("/perf", "")), \
                self.mock.patch.object(
                    perf_utils, "get_perf_list_json",
                    return_value=perf_list_json):
            nvdev_cls.return_value.is_cpu_type.return_value = True
            test_case(None, self.mock_logger)

    def test_event_description_uses_public_description(self):
        """Event validation ignores unrelated brief descriptions."""
        self.assertEqual(
            tegra410_pmu._event_description({
                "BriefDescription": "brief text",
                "PublicDescription": "PUBLIC TEXT",
            }),
            "public text",
        )

    def test_l1_prefetcher_test_records_actionable_failure(self):
        """The decorated L1 test reports the missing commit on Vera."""
        perf_list_json = _tegra410_perf_list_json()
        perf_list_json[0]["MetricGroup"] = "L1I_Prefetcher_Effectiveness"

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "2d85e13c5526"):
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                perf_list_json,
            )

    def test_metadata_tests_skip_non_vera(self):
        """Both Tegra410 metadata tests skip before invoking perf elsewhere."""
        with self.mock.patch.object(
                tegra410_pmu, "NvDev") as nvdev_cls, \
                self.mock.patch.object(
                    perf_utils, "resolve_perf_command") as resolve_perf:
            nvdev_cls.return_value.is_cpu_type.return_value = False

            for test_case in (
                    tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                    tegra410_pmu.test_tegra410_txreq_limit_event):
                with self.subTest(test_case=test_case.__name__), \
                        self.assertRaises(GLTCore.GLTTestSkip):
                    test_case(None, self.mock_logger)

        resolve_perf.assert_not_called()

    def test_metadata_tests_fail_unavailable_tegra410_table(self):
        """Both metadata tests fail when perf exposes no Tegra410 table."""
        for test_case in (
                tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                tegra410_pmu.test_tegra410_txreq_limit_event):
            with self.subTest(test_case=test_case.__name__):
                with self.assertRaisesRegex(
                        GLTCore.GLTTestFail, "86ff690f45cc") as context:
                    self._run_test_with_perf_list(test_case, [])
                self.assertNotIn("2d85e13c5526", str(context.exception))
                self.assertNotIn("9453bc6a69ef", str(context.exception))

    def test_metadata_tests_use_supported_plans(self):
        """Tegra410 metadata tests run only in standard and stress plans."""
        for test_case in (
                tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                tegra410_pmu.test_tegra410_txreq_limit_event):
            with self.subTest(test_case=test_case.__name__):
                self.assertEqual(set(test_case.plans), {"standard", "stress"})

    def test_txreq_test_records_actionable_failure(self):
        """The decorated TXREQ test reports the missing commit."""
        perf_list_json = _tegra410_perf_list_json()
        perf_list_json[-2]["Encoding"] = "cpu/event=0x196/"

        with self.assertRaisesRegex(GLTCore.GLTTestFail, "9453bc6a69ef"):
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_txreq_limit_event,
                perf_list_json,
            )

    def test_l1_prefetcher_groups_accept_fixed_metadata(self):
        """Correct L1D and L1I group assignments pass the GLT test."""
        self._run_test_with_perf_list(
            tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
            _tegra410_perf_list_json(),
        )

    def test_l1_prefetcher_groups_fail_missing_metadata(self):
        """Missing metrics identify absent baseline Tegra410 metadata."""
        perf_list_json = _tegra410_perf_list_json()[1:]

        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "86ff690f45cc") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                perf_list_json,
            )

        reason = str(context.exception)
        self.assertIn("missing metrics", reason)
        self.assertNotIn("2d85e13c5526", reason)

    def test_l1_swapped_group_fails_with_missing_metadata(self):
        """One observed swapped group remains conclusive despite missing data."""
        perf_list_json = _tegra410_perf_list_json()[1:]
        perf_list_json[0]["MetricGroup"] = "L1I_Prefetcher_Effectiveness"

        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "2d85e13c5526") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                perf_list_json,
            )

        reason = str(context.exception)
        self.assertIn("incorrect groups", reason)
        self.assertNotIn("86ff690f45cc", reason)

    def test_l1_prefetcher_groups_reject_multiple_groups(self):
        """Each Tegra410 metric must name exactly one expected group."""
        perf_list_json = _tegra410_perf_list_json()
        perf_list_json[0]["MetricGroup"] = (
            "L1D_Prefetcher_Effectiveness;L1I_Prefetcher_Effectiveness"
        )

        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "2d85e13c5526") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                perf_list_json,
            )

        reason = str(context.exception)
        self.assertIn("incorrect groups", reason)
        self.assertIn(
            "L1D_Prefetcher_Effectiveness;L1I_Prefetcher_Effectiveness",
            reason,
        )

    def test_l1_prefetcher_groups_report_swapped_metadata(self):
        """The original swapped assignments identify the commit."""
        perf_list_json = _tegra410_perf_list_json()
        for entry in perf_list_json:
            metric_group = entry.get("MetricGroup")
            if metric_group == "L1D_Prefetcher_Effectiveness":
                entry["MetricGroup"] = "L1I_Prefetcher_Effectiveness"
            elif metric_group == "L1I_Prefetcher_Effectiveness":
                entry["MetricGroup"] = "L1D_Prefetcher_Effectiveness"

        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "2d85e13c5526") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_l1_prefetcher_metric_groups,
                perf_list_json,
            )

        reason = str(context.exception)
        self.assertIn("incorrect groups", reason)
        self.assertIn(
            "perf vendor events arm64: fix swapped MetricGroup",
            reason,
        )
        self.assertIn("lore.kernel.org/all/20260813170155", reason)

    def test_txreq_event_accepts_fixed_metadata(self):
        """The corrected event 0x0197 alias and companion text pass the test."""
        self._run_test_with_perf_list(
            tegra410_pmu.test_tegra410_txreq_limit_event,
            _tegra410_perf_list_json(),
        )

    def test_txreq_event_fails_missing_aliases(self):
        """No legacy or corrected alias identifies absent baseline metadata."""
        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "86ff690f45cc") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_txreq_limit_event,
                [],
            )

        reason = str(context.exception)
        self.assertIn("neither the legacy nor corrected", reason)
        self.assertNotIn("9453bc6a69ef", reason)

    def test_txreq_event_fails_missing_companion(self):
        """A missing companion event identifies incomplete baseline metadata."""
        perf_list_json = _tegra410_perf_list_json()[:-1]

        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "86ff690f45cc") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_txreq_limit_event,
                perf_list_json,
            )

        reason = str(context.exception)
        self.assertIn("companion event", reason)
        self.assertNotIn("9453bc6a69ef", reason)

    def test_txreq_legacy_companion_fails_without_alias(self):
        """An observed legacy companion reference is conclusive stale data."""
        perf_list_json = _tegra410_perf_list_json()[-1:]
        perf_list_json[0]["PublicDescription"] = (
            "Companion event of TXREQ_LIMIT_1QUARTER_CYCLES."
        )

        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "9453bc6a69ef") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_txreq_limit_event,
                perf_list_json,
            )

        reason = str(context.exception)
        self.assertIn("references the legacy alias", reason)
        self.assertNotIn("86ff690f45cc", reason)

    def test_txreq_event_reports_legacy_metadata(self):
        """The old event alias and companion text identify the commit."""
        perf_list_json = _tegra410_perf_list_json()
        perf_list_json[-2] = {
            "EventName": "txreq_limit_1quarter_cycles",
            "Encoding": "cpu/event=0x197/",
            "PublicDescription": (
                "Number of cycles in which the dynamic TXREQ limit is between "
                "1/4 of the L2_TQ_SIZE and 1/2 of the L2_TQ_SIZE."
            ),
        }
        perf_list_json[-1]["PublicDescription"] = (
            "Companion event of TXREQ_LIMIT_1QUARTER_CYCLES."
        )

        with self.assertRaisesRegex(
                GLTCore.GLTTestFail, "9453bc6a69ef") as context:
            self._run_test_with_perf_list(
                tegra410_pmu.test_tegra410_txreq_limit_event,
                perf_list_json,
            )

        reason = str(context.exception)
        self.assertIn("legacy alias", reason)
        self.assertIn(
            "perf vendor events arm64: Fix Tegra410 Olympus event 0x0197",
            reason,
        )
        self.assertIn("lore.kernel.org/all/20260813170155", reason)
