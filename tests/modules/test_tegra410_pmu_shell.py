"""End-to-end offline tests for the Tegra410 PMU external script."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "external_tests" / "pmu" / "test_tegra410_pmu.sh"
REQUIRED_PMUS = (
    "nvidia_ucf_pmu_0",
    "nvidia_pcie_pmu_0",
    "nvidia_pcie_tgt_pmu_0",
)

FAKE_PERF = r'''#!/bin/bash
if [[ "$1" == "--version" ]]; then
    echo "perf version 7.0"
    exit 0
fi
if [[ "$1" == "list" ]]; then
    if [[ "${FAKE_PERF_MODE}" == "list-failure" ]]; then
        echo "injected perf list failure" >&2
        exit 1
    fi
    if [[ "$2" != "metric" ]]; then
        if [[ "${FAKE_PERF_MODE}" == "list-unrelated" ]]; then
            echo "SOME_OTHER_EVENT"
        elif [[ "${FAKE_PERF_MODE}" == "list-near-match" ]]; then
            echo "CPU_CYCLES_NOT_AVAILABLE"
        elif [[ "${FAKE_PERF_MODE}" != "list-empty" ]]; then
            echo "  CPU_CYCLES  INST_RETIRED  L1D_CACHE_REFILL  BR_MIS_PRED"
        fi
    fi
    exit 0
fi
if [[ "$1" != "stat" ]]; then
    exit 2
fi

event=""
while (( $# > 0 )); do
    if [[ "$1" == "-e" ]]; then
        event=$2
        shift 2
    else
        shift
    fi
done

count=1000
if [[ "${FAKE_PERF_MODE}" == "zero" ]]; then
    count=0
elif [[ "${FAKE_PERF_MODE}" == "mixed-zero" && "$event" == nvidia_pcie_pmu_0/* ]]; then
    count=0
elif [[ -n "${FAKE_PERF_ZERO_PMU}" && "$event" == "${FAKE_PERF_ZERO_PMU}"/* ]]; then
    count=0
fi

if [[ "$event" == "cycles" ]]; then
    case "${FAKE_PERF_MODE}" in
        core-stat-failure)
            echo "injected core perf stat failure" >&2
            exit 1
            ;;
        core-empty)
            exit 0
            ;;
        core-unrelated)
            echo "1000,,instructions,100.00,100.00,," >&2
            exit 0
            ;;
        core-malformed)
            echo "not-a-number,,cycles,100.00,100.00,," >&2
            exit 0
            ;;
        core-not-counted)
            echo "<not counted>,,cycles,0,0.00,," >&2
            exit 0
            ;;
        core-not-supported)
            echo "<not supported>,,cycles,0,0.00,," >&2
            exit 0
            ;;
        core-zero)
            echo "0,,cycles,100.00,100.00,," >&2
            exit 0
            ;;
    esac
fi

case "${FAKE_PERF_MODE}" in
    stat-failure)
        echo "injected perf stat failure" >&2
        exit 1
        ;;
    empty)
        exit 0
        ;;
    unrelated)
        echo "1000,,other_pmu/cycles/,100.00,," >&2
        ;;
    malformed)
        echo "not-a-number,,$event,100.00,," >&2
        ;;
    not-counted)
        echo "<not counted>,,$event,0,0.00,," >&2
        ;;
    not-supported)
        echo "<not supported>,,$event,0,0.00,," >&2
        ;;
    *)
        echo "$count,,$event,100.00,100.00,," >&2
        ;;
esac
'''


class Tegra410PmuShellTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.event_root = self.root / "event_source"
        self.soc_root = self.root / "soc"
        self.soc_root.mkdir()
        (self.soc_root / "soc0").mkdir()
        self.soc_id = self.soc_root / "soc0" / "soc_id"
        self.soc_id.write_text("jep106:036b:0410\n", encoding="utf-8")
        self.cpuinfo = self.root / "cpuinfo"
        self.cpuinfo.write_text(
            "CPU implementer : 0x4e\nCPU part : 0x100\n", encoding="utf-8"
        )
        self.perf = self.root / "perf"
        self.perf.write_text(FAKE_PERF, encoding="utf-8")
        self.perf.chmod(0o755)
        for pmu_name in REQUIRED_PMUS:
            self.add_pmu(pmu_name)

    def add_pmu(self, name, cycles=True):
        pmu_path = self.event_root / name
        events_path = pmu_path / "events"
        events_path.mkdir(parents=True)
        (pmu_path / "type").write_text("100\n", encoding="utf-8")
        if cycles:
            (events_path / "cycles").write_text("event=0x0\n", encoding="utf-8")
        else:
            (events_path / "rd_bytes").write_text("event=0x1\n", encoding="utf-8")

    def run_suite(self, mode="positive", zero_pmu=""):
        env = os.environ.copy()
        env.update({
            "PERF_EVENT_DIR": str(self.event_root),
            "SOC_SYSFS_ROOT": str(self.soc_root),
            "CPUINFO_PATH": str(self.cpuinfo),
            "PERF": str(self.perf),
            "TEST_DURATION": "0",
            "FAKE_PERF_MODE": mode,
            "FAKE_PERF_ZERO_PMU": zero_pmu,
        })
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; main',
                "test-harness",
                str(SCRIPT),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

    def test_positive_required_pmus_pass_without_optional_devices(self):
        result = self.run_suite()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cycles counter activity observed (count=1000)", result.stdout)
        self.assertIn("CMEM Latency PMU", result.stdout)

    def test_each_required_pmu_is_required(self):
        for pmu_name in REQUIRED_PMUS:
            with self.subTest(pmu=pmu_name):
                shutil.rmtree(self.event_root / pmu_name)
                result = self.run_suite()
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("Required device missing", result.stdout)
                self.add_pmu(pmu_name)

    def test_all_zero_counts_are_inconclusive(self):
        result = self.run_suite(mode="zero")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("cycles count is zero", result.stdout)

    def test_one_zero_count_blocks_other_passes(self):
        result = self.run_suite(mode="mixed-zero")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("cycles counter activity observed", result.stdout)
        self.assertIn("cycles count is zero", result.stdout)

    def test_optional_pmu_zero_count_blocks_full_pass(self):
        self.add_pmu("nvidia_cmem_latency_pmu_0")
        result = self.run_suite(zero_pmu="nvidia_cmem_latency_pmu_0")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)

    def test_missing_cycles_does_not_select_arbitrary_event(self):
        (self.event_root / REQUIRED_PMUS[0] / "events" / "cycles").unlink()
        (self.event_root / REQUIRED_PMUS[0] / "events" / "rd_bytes").write_text(
            "event=0x1\n", encoding="utf-8"
        )
        result = self.run_suite()
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("No cycles event", result.stdout)

    def test_invalid_perf_stat_results_fail(self):
        modes = (
            "stat-failure",
            "empty",
            "unrelated",
            "malformed",
            "not-counted",
            "not-supported",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                result = self.run_suite(mode=mode)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("Test Summary", result.stdout)

    def test_invalid_core_perf_stat_results_fail(self):
        modes = (
            "core-stat-failure",
            "core-empty",
            "core-unrelated",
            "core-malformed",
            "core-not-counted",
            "core-not-supported",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                result = self.run_suite(mode=mode)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("Test Summary", result.stdout)

    def test_core_zero_count_is_inconclusive(self):
        result = self.run_suite(mode="core-zero")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("CPU vendor events - cycles count is zero", result.stdout)
        self.assertNotIn("Core PMU counter activity observed", result.stdout)

    def test_empty_unrelated_or_near_match_perf_list_fails(self):
        for mode in ("list-empty", "list-unrelated", "list-near-match"):
            with self.subTest(mode=mode):
                result = self.run_suite(mode=mode)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("No expected events found", result.stdout)

    def test_perf_list_failure_cannot_be_hidden_by_other_passes(self):
        result = self.run_suite(mode="list-failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("perf list failed", result.stdout)

    def test_t410_with_invalid_cpu_identity_fails(self):
        cpu_identities = (
            "",
            "CPU implementer : 0x41\nCPU part : 0x100\n",
            "CPU implementer : 0x4e\nCPU part : 0xd05\n",
        )
        for cpu_identity in cpu_identities:
            with self.subTest(cpu_identity=cpu_identity):
                self.cpuinfo.write_text(cpu_identity, encoding="utf-8")
                result = self.run_suite()
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("CPU vendor events", result.stdout)
                self.assertNotIn("All tests passed", result.stdout)

    def test_non_t410_platform_skips(self):
        self.soc_id.write_text("jep106:036b:0236\n", encoding="utf-8")
        result = self.run_suite()
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("Not T410 platform", result.stdout)


if __name__ == "__main__":
    unittest.main()
