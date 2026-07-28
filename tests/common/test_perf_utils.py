"""Unit tests for common/perf_utils.py."""
# pylint: disable=duplicate-code,too-many-public-methods,wrong-import-position

import subprocess
import sys
import tempfile
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.perf_utils import (
    PERF_TEST_TIMEOUT_SECONDS,
    PERF_VERSION_TIMEOUT_SECONDS,
    PerfTestStatus,
    get_perf_kernel_version_warning,
    get_perf_version,
    has_coresight_fallback_evidence,
    natural_sort_key,
    resolve_perf_command,
    run_perf_test_candidates,
)


class TestPerfUtils(GLTTestBase):  # pylint: disable=too-many-public-methods
    """Test perf helper behavior without requiring real perf hardware."""

    def _completed(self, stdout="", stderr="", returncode=0):
        return subprocess.CompletedProcess(
            args=["perf"],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def test_natural_sort_key_orders_perf_versions_numerically(self):
        """Perf candidate path sorting follows numeric version components."""
        paths = [
            Path("/usr/lib/linux-tools/6.9/perf"),
            Path("/usr/lib/linux-tools/6.18/perf"),
            Path("/usr/lib/linux-tools/6.17.13/perf"),
        ]

        self.assertEqual(
            [str(path) for path in sorted(paths, key=natural_sort_key)],
            [
                "/usr/lib/linux-tools/6.9/perf",
                "/usr/lib/linux-tools/6.17.13/perf",
                "/usr/lib/linux-tools/6.18/perf",
            ],
        )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_rejects_path_version_substring(self, mock_run):
        """A kernel version token must not match an adjacent numeric suffix."""
        mock_run.return_value = self._completed(stdout="perf version 6.10\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/lib/linux-tools/6.10/perf"],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.1-mpam+"),
        ):
            self.assertIsNone(resolve_perf_command()[0])

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_accepts_abbreviated_perf_version(self, mock_run):
        """A compatible abbreviated perf version can match the running kernel."""
        mock_run.return_value = self._completed(stdout="perf version 6.17\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/lib/linux-tools/6.17/perf"],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertEqual(
                resolve_perf_command()[0],
                "/usr/lib/linux-tools/6.17/perf",
            )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_prefers_exact_over_abbreviated_version(
        self, mock_run
    ):
        """An exact kernel-version candidate beats an earlier compatible one."""
        candidates = [
            "/usr/lib/linux-tools-6.17/perf",
            "/usr/lib/linux-tools/6.17.13/perf",
        ]
        results = {
            "/usr/lib/linux-tools-6.17/perf": self._completed(
                stdout="perf version 6.17\n"
            ),
            "/usr/lib/linux-tools/6.17.13/perf": self._completed(
                stdout="perf version 6.17.13\n"
            ),
        }

        def fake_run(cmd, **_):
            return results[cmd[0]]

        mock_run.side_effect = fake_run
        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=candidates
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertEqual(
                resolve_perf_command()[0],
                "/usr/lib/linux-tools/6.17.13/perf",
            )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_prefers_exact_for_local_build_release(
        self, mock_run
    ):
        """An exact candidate beats abbreviated compat for local build kernels."""
        candidates = [
            "/usr/lib/linux-tools-6.17/perf",
            "/usr/lib/linux-tools/6.17.13/perf",
        ]
        results = {
            "/usr/lib/linux-tools-6.17/perf": self._completed(
                stdout="perf version 6.17\n"
            ),
            "/usr/lib/linux-tools/6.17.13/perf": self._completed(
                stdout="perf version 6.17.13\n"
            ),
        }

        def fake_run(cmd, **_):
            return results[cmd[0]]

        mock_run.side_effect = fake_run
        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=candidates
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13+"),
        ):
            self.assertEqual(
                resolve_perf_command()[0],
                "/usr/lib/linux-tools/6.17.13/perf",
            )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_accepts_local_build_plus_suffix(self, mock_run):
        """A local build '+' suffix can match an otherwise identical perf version."""
        mock_run.return_value = self._completed(stdout="perf version 6.17.13\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/lib/linux-tools/6.17.13/perf"],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13+"),
        ):
            self.assertEqual(
                resolve_perf_command()[0],
                "/usr/lib/linux-tools/6.17.13/perf",
            )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_rejects_extended_version_suffix(self, mock_run):
        """A reported version with an adjacent numeric suffix is not compatible."""
        mock_run.return_value = self._completed(stdout="perf version 6.17.130\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/lib/linux-tools/6.17.130/perf"],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertIsNone(resolve_perf_command()[0])

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_skips_timed_out_candidate(self, mock_run):
        """A hanging perf wrapper does not block resolver fallback candidates."""
        candidates = ["perf", "/usr/lib/linux-tools/6.17/perf"]
        results = {
            "perf": subprocess.TimeoutExpired(
                ["perf", "--version"],
                timeout=PERF_VERSION_TIMEOUT_SECONDS,
            ),
            "/usr/lib/linux-tools/6.17/perf": self._completed(
                stdout="perf version 6.17.13\n"
            ),
        }

        def fake_run(cmd, **_):
            result = results[cmd[0]]
            if isinstance(result, Exception):
                raise result
            return result

        mock_run.side_effect = fake_run
        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=candidates
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertEqual(
                resolve_perf_command()[0],
                "/usr/lib/linux-tools/6.17/perf",
            )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_get_perf_version(self, mock_run):
        """perf userspace version is queried without a shell."""
        mock_run.return_value = self._completed(stdout="perf version 6.17.13\n")

        self.assertEqual(get_perf_version(self.mock_logger),
                         "perf version 6.17.13")
        mock_run.assert_called_once_with(
            ["perf", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_get_perf_version_handles_missing_perf(self, mock_run):
        """Missing perf reports no version string."""
        mock_run.side_effect = FileNotFoundError

        self.assertEqual(get_perf_version(self.mock_logger), "")
        self.mock_logger.debug.assert_called_once()

    def test_get_perf_kernel_version_warning_accepts_matching_versions(self):
        """Matching perf/kernel major.minor and linux-tools path is accepted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            linux_tools = Path(temp_dir)
            expected_dir = linux_tools / "6.17.0-1018-nvidia-64k"
            expected_dir.mkdir()
            (expected_dir / "perf").touch()

            self.assertEqual(
                get_perf_kernel_version_warning(
                    "perf version 6.17.13",
                    "6.17.0-1018-nvidia-64k",
                    linux_tools,
                ),
                "",
            )

    def test_get_perf_kernel_version_warning_reports_major_minor_mismatch(self):
        """Different perf/kernel major.minor versions are suspicious."""
        warning = get_perf_kernel_version_warning(
            "perf version 6.16.9",
            "6.17.0-1018-nvidia-64k",
            Path("/path/that/does/not/exist"),
        )

        self.assertIn("does not match running kernel major/minor", warning)
        self.assertIn("perf version 6.16.9", warning)
        self.assertIn("6.17.0-1018-nvidia-64k", warning)

    def test_get_perf_kernel_version_warning_reports_missing_linux_tools_perf(self):
        """Ubuntu-style linux-tools missing the running kernel perf is suspicious."""
        with tempfile.TemporaryDirectory() as temp_dir:
            warning = get_perf_kernel_version_warning(
                "perf version 6.17.13",
                "6.17.0-1018-nvidia-64k",
                Path(temp_dir),
            )

        self.assertIn("Matching kernel perf binary is missing", warning)
        self.assertIn("6.17.0-1018-nvidia-64k", warning)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_prefers_kernel_matching_binary(self, mock_run):
        """Perf resolver ignores broken wrappers and picks the matching version."""
        candidates = [
            "perf",
            "/usr/lib/linux-tools/6.18/perf",
            "/usr/lib/linux-tools/6.17/perf",
        ]
        results = {
            "perf": self._completed(
                stdout="WARNING: perf not found for kernel 6.17.13\n"
            ),
            "/usr/lib/linux-tools/6.18/perf": self._completed(
                stdout="perf version 6.18.0\n"
            ),
            "/usr/lib/linux-tools/6.17/perf": self._completed(
                stdout="perf version 6.17.13\n"
            ),
        }

        def fake_run(cmd, **_):
            return results[cmd[0]]

        mock_run.side_effect = fake_run
        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=candidates
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertEqual(
                resolve_perf_command()[0],
                "/usr/lib/linux-tools/6.17/perf",
            )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_returns_none_when_no_candidate_matches_kernel(
        self, mock_run
    ):
        """Perf resolver does not fall back to an unrelated valid perf binary."""
        mock_run.return_value = self._completed(stdout="perf version 6.18.0\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/lib/linux-tools/6.18/perf"],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertIsNone(resolve_perf_command()[0])

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_rejects_similar_version_path_prefix(self, mock_run):
        """Do not match linux-tools/6.17.130 when the running kernel is 6.17.13."""
        mock_run.return_value = self._completed(stdout="perf version 6.17.130\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/lib/linux-tools-6.17.130/perf"],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertIsNone(resolve_perf_command()[0])

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_accepts_package_revision_suffix(self, mock_run):
        """Perf package revision suffixes should not block base version matching."""
        mock_run.return_value = self._completed(stdout="perf version 6.17.13-1ubuntu1\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/bin/perf"],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            self.assertEqual(resolve_perf_command()[0], "/usr/bin/perf")

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_matches_linux_tools_release_directory(
        self, mock_run
    ):
        """Match perf under linux-tools-<release> when perf version differs slightly."""
        candidate = "/usr/lib/linux-tools-6.17.0-1018-nvidia-64k/perf"
        mock_run.return_value = self._completed(stdout="perf version 6.17.13\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=[candidate],
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.0-1022-nvidia-64k"),
        ):
            self.assertEqual(resolve_perf_command()[0], candidate)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_skips_candidate_when_version_probe_times_out(
        self, mock_run
    ):
        """A hung perf --version probe is skipped instead of blocking resolution."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["/usr/lib/linux-tools/6.17/perf", "--version"],
            timeout=PERF_VERSION_TIMEOUT_SECONDS,
        )

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates",
            return_value=["/usr/lib/linux-tools/6.17/perf"],
        ):
            self.assertIsNone(resolve_perf_command()[0])

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_returns_none_when_no_valid_perf_exists(
        self, mock_run
    ):
        """Perf resolver reports no command when every candidate is unusable."""
        mock_run.return_value = self._completed(
            stdout="WARNING: perf not found for kernel 6.17.13\n"
        )

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=["perf"]
        ):
            self.assertIsNone(resolve_perf_command()[0])

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_accepts_617_pmu_name(self, mock_run):
        """A perf 6.17 PMU selftest name can satisfy the candidate list."""
        mock_run.side_effect = [
            self._completed(stdout="  5.2: Parsing of all PMU events from sysfs\n"),
            self._completed(
                stderr="  5.2: Parsing of all PMU events from sysfs           : Ok\n"
            ),
        ]

        result = run_perf_test_candidates(
            ["perf all PMU test", "Parsing of all PMU events from sysfs"]
        )

        self.assertEqual(result.status, PerfTestStatus.PASS)
        self.assertEqual(result.test_name, "Parsing of all PMU events from sysfs")

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_uses_supplied_perf_command(self, mock_run):
        """Both selftest discovery and execution use the resolved perf command."""
        mock_run.side_effect = [
            self._completed(stdout="  5.2: perf all PMU test\n"),
            self._completed(stderr="  5.2: perf all PMU test           : Ok\n"),
        ]

        result = run_perf_test_candidates(
            ["perf all PMU test"],
            perf_cmd="/usr/lib/linux-tools/6.17/perf",
        )

        self.assertEqual(result.status, PerfTestStatus.PASS)
        self.assertEqual(
            mock_run.call_args_list[0].args[0],
            ["/usr/lib/linux-tools/6.17/perf", "test", "list"],
        )
        self.assertEqual(
            mock_run.call_args_list[0].kwargs.get("timeout"),
            PERF_TEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            mock_run.call_args_list[1].args[0],
            ["/usr/lib/linux-tools/6.17/perf", "test", "-vv", "perf all PMU test"],
        )
        self.assertEqual(
            mock_run.call_args_list[1].kwargs.get("timeout"),
            PERF_TEST_TIMEOUT_SECONDS,
        )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_list_timeout_returns_fail(self, mock_run):
        """A hung perf test list is reported as FAIL, not raised."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            ["perf", "test", "list"], PERF_TEST_TIMEOUT_SECONDS
        )

        result = run_perf_test_candidates(["perf all PMU test"])

        self.assertEqual(result.status, PerfTestStatus.FAIL)
        self.assertIn("timed out", result.reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_run_timeout_returns_fail(self, mock_run):
        """A hung perf test -vv is reported as FAIL, not raised."""
        mock_run.side_effect = [
            self._completed(stdout="  5.2: perf all PMU test\n"),
            subprocess.TimeoutExpired(
                ["perf", "test", "-vv", "perf all PMU test"],
                PERF_TEST_TIMEOUT_SECONDS,
            ),
        ]

        result = run_perf_test_candidates(["perf all PMU test"])

        self.assertEqual(result.status, PerfTestStatus.FAIL)
        self.assertEqual(result.test_name, "perf all PMU test")
        self.assertIn("timed out", result.reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_accepts_old_name_first(self, mock_run):
        """Existing perf selftest names remain preferred when listed."""
        mock_run.side_effect = [
            self._completed(
                stdout=(
                    "  5.2: perf all PMU test\n"
                    "  5.3: Parsing of all PMU events from sysfs\n"
                )
            ),
            self._completed(stderr="  5.2: perf all PMU test           : Ok\n"),
        ]

        result = run_perf_test_candidates(
            ["perf all PMU test", "Parsing of all PMU events from sysfs"]
        )

        self.assertEqual(result.status, PerfTestStatus.PASS)
        self.assertEqual(result.test_name, "perf all PMU test")

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_matches_substring_names(self, mock_run):
        """Perf selftest candidates match longer upstream test names."""
        mock_run.side_effect = [
            self._completed(
                stdout=(
                    "  71: Check Arm CoreSight trace data recording "
                    "and synthesized samples\n"
                )
            ),
            self._completed(
                stderr=(
                    "  71: Check Arm CoreSight trace data recording "
                    "and synthesized samples           : Ok\n"
                )
            ),
        ]

        result = run_perf_test_candidates(["arm coresight"])

        self.assertEqual(result.status, PerfTestStatus.PASS)
        self.assertEqual(result.test_name, "arm coresight")

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_reports_empty_output_as_no_match(self, mock_run):
        """Perf can return 0 with empty output for no matched test names."""
        mock_run.side_effect = [
            self._completed(stdout="  5.2: perf all PMU test\n"),
            self._completed(stdout="", stderr="", returncode=0),
        ]

        result = run_perf_test_candidates(["perf all PMU test"])

        self.assertEqual(result.status, PerfTestStatus.NO_MATCH)
        self.assertIn("empty output", result.reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_reports_nonzero_failure(self, mock_run):
        """A non-zero perf selftest result is reported as a failure."""
        mock_run.side_effect = [
            self._completed(
                stdout=(
                    "  71: Check Arm CoreSight trace data recording "
                    "and synthesized samples\n"
                )
            ),
            self._completed(stderr="broken\n", returncode=1),
        ]

        result = run_perf_test_candidates(["arm coresight"])

        self.assertEqual(result.status, PerfTestStatus.FAIL)
        self.assertIn("broken", result.reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_reports_unlisted_candidates(self, mock_run):
        """Missing candidate names are reported before running perf test."""
        mock_run.return_value = self._completed(stdout="  1: other test\n")

        result = run_perf_test_candidates(["arm coresight"])

        self.assertEqual(result.status, PerfTestStatus.NO_MATCH)
        self.assertIn("No matching perf test", result.reason)
        mock_run.assert_called_once()

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_run_perf_test_candidates_reports_generator_candidates(self, mock_run):
        """Generator candidates are preserved in no-match failure reasons."""
        mock_run.return_value = self._completed(stdout="  1: other test\n")

        result = run_perf_test_candidates(name for name in ["arm coresight"])

        self.assertEqual(result.status, PerfTestStatus.NO_MATCH)
        self.assertIn("arm coresight", result.reason)

    def test_has_coresight_fallback_evidence_requires_cs_etm_and_devices(self):
        """Coresight fallback requires both cs_etm and a Coresight device."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            event_sources = base / "event_source" / "devices"
            coresight = base / "coresight" / "devices"
            event_sources.mkdir(parents=True)
            coresight.mkdir(parents=True)

            self.assertFalse(
                has_coresight_fallback_evidence(event_sources, coresight)
            )

            (event_sources / "cs_etm").mkdir()
            self.assertFalse(
                has_coresight_fallback_evidence(event_sources, coresight)
            )

            (coresight / "tmc_etf0").mkdir()
            self.assertTrue(
                has_coresight_fallback_evidence(event_sources, coresight)
            )

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_reports_version_mismatch_reason(self, mock_run):
        """A version-mismatched perf is named in the reason, not called missing."""
        mock_run.return_value = self._completed(stdout="perf version 7.0.12\n")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=["perf"]
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="7.0.0-2015-nvidia-bos-64k"),
        ):
            command, reason = resolve_perf_command()

        self.assertIsNone(command)
        self.assertIn("7.0.12", reason)
        self.assertIn("7.0.0-2015-nvidia-bos-64k", reason)
        self.assertNotIn("Please install 'perf'", reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_reports_missing_binary_reason(self, mock_run):
        """A wrapper with no kernel binary says so instead of 'not installed'."""
        mock_run.return_value = self._completed(
            stdout="WARNING: perf not found for kernel 6.17.13\n"
        )

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=["perf"]
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            command, reason = resolve_perf_command()

        self.assertIsNone(command)
        self.assertIn("no perf binary", reason)
        self.assertIn("6.17.13-mpam+", reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_reports_not_installed_reason(self, mock_run):
        """An unexecutable perf is the only case reported as not installed."""
        mock_run.side_effect = FileNotFoundError(2, "No such file or directory")

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=["perf"]
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            command, reason = resolve_perf_command()

        self.assertIsNone(command)
        self.assertIn("could not be executed", reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_reports_timeout_reason(self, mock_run):
        """A hung --version probe is reported as a timeout, not a missing perf."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["perf", "--version"], timeout=PERF_VERSION_TIMEOUT_SECONDS
        )

        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=["perf"]
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            command, reason = resolve_perf_command()

        self.assertIsNone(command)
        self.assertIn("timed out", reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_reason_prefers_most_progressed_candidate(
        self, mock_run
    ):
        """A version mismatch outranks an unexecutable candidate in the reason."""
        candidates = ["/usr/lib/linux-tools/6.18/perf", "perf"]
        results = {
            "/usr/lib/linux-tools/6.18/perf": FileNotFoundError(
                2, "No such file or directory"
            ),
            "perf": self._completed(stdout="perf version 6.18.0\n"),
        }

        def fake_run(cmd, **_):
            result = results[cmd[0]]
            if isinstance(result, Exception):
                raise result
            return result

        mock_run.side_effect = fake_run
        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=candidates
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="6.17.13-mpam+"),
        ):
            command, reason = resolve_perf_command()

        self.assertIsNone(command)
        self.assertIn("6.18.0", reason)
        self.assertNotIn("could not be executed", reason)

    @GLTTestBase.unittest.mock.patch("common.perf_utils.subprocess.run")
    def test_resolve_perf_command_reason_prefers_missing_binary_over_stale_perf(
        self, mock_run
    ):
        """A wrapper with no binary for this kernel outranks a stale perf."""
        candidates = ["perf", "/usr/lib/linux-tools/6.8/perf"]
        results = {
            "perf": self._completed(
                stdout="WARNING: perf not found for kernel 7.0.0-64k\n"
            ),
            "/usr/lib/linux-tools/6.8/perf": self._completed(
                stdout="perf version 6.8.0\n"
            ),
        }

        def fake_run(cmd, **_):
            return results[cmd[0]]

        mock_run.side_effect = fake_run
        with self.mock.patch(
            "common.perf_utils._get_perf_candidates", return_value=candidates
        ), self.mock.patch(
            "common.perf_utils.platform.uname",
            return_value=self.mock.Mock(release="7.0.0-64k"),
        ):
            command, reason = resolve_perf_command()

        self.assertIsNone(command)
        self.assertIn("no perf binary", reason)
        self.assertIn("7.0.0-64k", reason)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
