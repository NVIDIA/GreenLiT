"""Unit tests for CoreSight perf compatibility."""
# pylint: disable=wrong-import-position,duplicate-code,protected-access

import sys
from pathlib import Path
from types import SimpleNamespace

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from common import perf_utils
from src.modules.hwtracing import glt_test_coresight


class TestPerfCoresightCompatibility(GLTTestBase):
    """Test CoreSight behavior around perf selftest compatibility."""

    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_perf_coresight_uses_fallback_when_selftest_is_missing(
        self, mock_nvdev, mock_perf_utils
    ):
        """Coresight test accepts fallback evidence when no selftest is listed."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_nvdev.get_release_info.return_value = {
            "NAME": "Ubuntu",
            "VERSION_ID": "24.04",
        }
        mock_perf_utils.resolve_perf_command.return_value = ("/perf", "")
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.NO_MATCH,
            "No matching perf test found",
        )
        mock_perf_utils.has_coresight_fallback_evidence.return_value = True

        glt_test_coresight.test_perf_coresight(None, self.mock_logger)

        mock_perf_utils.run_perf_test_candidates.assert_called_once_with(
            ["arm coresight"], perf_cmd="/perf"
        )
        mock_perf_utils.has_coresight_fallback_evidence.assert_called_once_with()
        self.mock_logger.info.assert_called_once_with(
            "No perf Coresight selftest found; using cs_etm/sysfs fallback."
        )

    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.perf_utils")
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_perf_coresight_fails_when_selftest_and_fallback_are_missing(
        self, mock_nvdev, mock_perf_utils
    ):
        """Coresight test fails when neither perf selftest nor fallback exists."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "grace"
        mock_nvdev.get_release_info.return_value = {
            "NAME": "Ubuntu",
            "VERSION_ID": "24.04",
        }
        mock_perf_utils.resolve_perf_command.return_value = ("/perf", "")
        mock_perf_utils.PerfTestStatus = perf_utils.PerfTestStatus
        mock_perf_utils.run_perf_test_candidates.return_value = perf_utils.PerfTestResult(
            perf_utils.PerfTestStatus.NO_MATCH,
            "No matching perf test found",
        )
        mock_perf_utils.has_coresight_fallback_evidence.return_value = False

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_coresight(None, self.mock_logger)
        self.assertIn("No matching perf test found", str(context.exception))


class TestPerfCsEtmMultisocketTraceIdMap(GLTTestBase):
    """Unit tests for multi-socket cs_etm trace ID map discrimination."""

    # Pylint 3 does not know the Pylint 4 too-many-positional-arguments message.
    # pylint: disable=unknown-option-value
    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def setUp(self):
        super().setUp()
        self.sd = SimpleNamespace(log_dir="/tmp/greenlit-test-logs")
        online_path = SimpleNamespace(
            read_text=lambda **_: "0-351\n"
        )
        online_path_patcher = self.mock.patch(
            "src.modules.hwtracing.glt_test_coresight._CPU_ONLINE_PATH",
            online_path,
        )
        online_path_patcher.start()
        self.addCleanup(online_path_patcher.stop)

    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_skips_on_non_grace_vera(self, mock_nvdev):
        """Skip when the platform is neither Grace nor Vera."""
        mock_nvdev.return_value.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("Grace or Vera", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_skips_without_cs_etm(
        self, mock_nvdev, mock_cs_etm_dir, _mock_ensure
    ):
        """Skip when the cs_etm event source is missing after module setup."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("cs_etm", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(1, 353)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_skips_when_selected_cpu_lacks_trbe_sink(
        self, mock_nvdev, mock_cs_etm_dir, _mock_ensure, _mock_trbe
    ):
        """Equal aggregate counts cannot hide a selected CPU's missing sink."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("TRBE sink", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(64)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CPU_ONLINE_PATH"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_skips_when_cpu_count_too_low(
        self, mock_nvdev, mock_online_path, mock_ensure
    ):
        """Skip small hosts before loading coresight_etm4x."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_online_path.read_text.return_value = "0-63\n"

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("enough online CPUs", str(context.exception))
        mock_ensure.assert_not_called()

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_passes_when_record_succeeds(
        self,
        mock_nvdev,
        mock_cs_etm_dir,
        _mock_ensure,
        _mock_trbe,
        mock_resolve,
        mock_run,
    ):
        """Pass when both the 111-CPU and 112-CPU records succeed."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True
        mock_run.side_effect = [
            SimpleNamespace(
                returncode=0, stdout="",
                stderr="[ perf record: Captured and wrote 0.6 MB ]",
            ),
            SimpleNamespace(
                returncode=0, stdout="",
                stderr="[ perf record: Captured and wrote 0.2 MB ]",
            ),
        ]

        glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
            self.sd, self.mock_logger
        )
        # Two records: the 111-CPU control, then the 112-CPU trigger.
        self.assertEqual(mock_run.call_count, 2)
        control_argv, trigger_argv = (c.args[0] for c in mock_run.call_args_list)
        self.assertEqual(control_argv[2:4], ["-e", "cs_etm//u"])
        self.assertEqual(trigger_argv[2:4], ["-e", "cs_etm//u"])
        # Both runs must use explicit -C, so both sit in perf's per-CPU mode and
        # the only uncontrolled variable left is the single extra CPU.
        control_cpus = control_argv[control_argv.index("-C") + 1].split(",")
        trigger_cpus = trigger_argv[trigger_argv.index("-C") + 1].split(",")
        self.assertEqual(len(control_cpus), glt_test_coresight._CONTROL_CPU_COUNT)
        self.assertEqual(
            len(trigger_cpus), glt_test_coresight._MIN_CPUS_FOR_TRACE_ID_PRESSURE
        )
        # The trigger list is the control list plus exactly one CPU.
        self.assertEqual(trigger_cpus[:-1], control_cpus)
        # Proof kernels rarely ship a version-matched perf; the production call
        # must opt into the unmatched fallback or those kernels regress to fail.
        self.assertTrue(mock_resolve.call_args.kwargs.get("allow_unmatched"))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_fails_on_mmap_enomem(
        self,
        mock_nvdev,
        mock_cs_etm_dir,
        _mock_ensure,
        _mock_trbe,
        _mock_resolve,
        mock_run,
    ):
        """Fail with causal message when unfixed kernels hit mmap ENOMEM."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True
        # Control at 111 CPUs succeeds, trigger at 112 does not: the exact
        # one-CPU split measured on the unfixed 352-CPU SUT.
        mock_run.side_effect = [
            SimpleNamespace(
                returncode=0, stdout="",
                stderr="[ perf record: Captured and wrote 0.6 MB ]",
            ),
            SimpleNamespace(
                returncode=255, stdout="",
                stderr="failed to mmap with 12 (Cannot allocate memory)",
            ),
        ]

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        message = str(context.exception)
        self.assertIn("ENOMEM", message)
        self.assertIn("5ad628a76176", message)
        self.assertIn(str(glt_test_coresight._CONTROL_CPU_COUNT), message)
        self.assertIn(
            str(glt_test_coresight._MIN_CPUS_FOR_TRACE_ID_PRESSURE), message
        )
        self.assertIn("consistent with a global", message)
        self.assertIn("not independently ruled out", message)

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_does_not_blame_trace_id_map_on_generic_enomem(
        self,
        mock_nvdev,
        mock_cs_etm_dir,
        _mock_ensure,
        _mock_trbe,
        _mock_resolve,
        mock_run,
    ):
        """Generic ENOMEM must not be diagnosed as the global trace ID limit."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True
        mock_run.side_effect = [
            SimpleNamespace(
                returncode=0, stdout="",
                stderr="[ perf record: Captured and wrote 0.6 MB ]",
            ),
            SimpleNamespace(
                returncode=255, stdout="", stderr="Cannot allocate memory",
            ),
        ]

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        message = str(context.exception)
        self.assertIn("failed unexpectedly", message)
        self.assertNotIn("5ad628a76176", message)

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=(None, "no usable perf"),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_fails_when_no_usable_perf(
        self, mock_nvdev, mock_cs_etm_dir, _mock_ensure,
        _mock_trbe, _mock_resolve, mock_run
    ):
        """No usable perf binary must fail before ever running record."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("perf", str(context.exception))
        mock_run.assert_not_called()

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run",
        side_effect=[
            SimpleNamespace(returncode=0, stdout="", stderr="wrote 0.6 MB"),
            glt_test_coresight.subprocess.TimeoutExpired(cmd=["perf"], timeout=300),
        ],
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_fails_when_record_times_out(
        self, mock_nvdev, mock_cs_etm_dir, _mock_ensure,
        _mock_trbe, _mock_resolve, _mock_run
    ):
        """A hung perf record surfaces a clean timeout failure."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("timed out", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run",
        side_effect=[
            SimpleNamespace(returncode=0, stdout="", stderr="wrote 0.6 MB"),
            OSError("perf not executable"),
        ],
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_fails_when_record_launch_errors(
        self, mock_nvdev, mock_cs_etm_dir, _mock_ensure,
        _mock_trbe, _mock_resolve, _mock_run
    ):
        """A perf record launch OSError surfaces a clean failure, not a traceback."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("failed to launch", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_fails_when_control_record_fails(
        self, mock_nvdev, mock_cs_etm_dir, _mock_ensure,
        _mock_trbe, _mock_resolve, mock_run
    ):
        """A failing 111-CPU control must fail, never blame the backport.

        Prerequisites are already established, so a 111-CPU miss is a recording
        defect. The trigger must not run at all.
        """
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True
        mock_run.side_effect = [
            SimpleNamespace(
                returncode=255, stdout="",
                stderr="failed to mmap with 12 (Cannot allocate memory)",
            ),
        ]

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        message = str(context.exception)
        self.assertIn(str(glt_test_coresight._CONTROL_CPU_COUNT), message)
        self.assertIn("did not succeed", message)
        self.assertNotIn("5ad628a76176", message)
        # The 112-CPU trigger must never have run.
        mock_run.assert_called_once()

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(),
    )
    def test_ensure_etm4x_loaded_returns_empty_set_when_modprobe_unavailable(
        self, _mock_cpu_ids, mock_run
    ):
        """modprobe soft-failures are unavailable, not crashes or hard fails.

        Covers nonzero exit, TimeoutExpired, and OSError (e.g. missing binary)
        so platforms without CoreSight still reach the cs_etm skip path.
        """
        cases = (
            SimpleNamespace(returncode=1, stdout="", stderr="Module not found"),
            glt_test_coresight.subprocess.TimeoutExpired(
                cmd=["modprobe", "coresight_etm4x"], timeout=30
            ),
            FileNotFoundError("modprobe"),
        )
        for case in cases:
            self.mock_logger.reset_mock()
            if isinstance(case, BaseException):
                mock_run.side_effect = case
                mock_run.return_value = None
            else:
                mock_run.side_effect = None
                mock_run.return_value = case
            with self.subTest(case=case):
                self.assertEqual(
                    glt_test_coresight._ensure_etm4x_loaded(self.mock_logger), set()
                )
                self.mock_logger.warning.assert_called()

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._coresight_cpu_ids",
        return_value=set(range(352)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._ensure_etm4x_loaded",
        return_value=set(range(1, 353)),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_skips_when_selected_cpu_lacks_ete_source(
        self, mock_nvdev, mock_cs_etm_dir, _mock_ensure,
        _mock_trbe, _mock_resolve
    ):
        """Equal aggregate counts cannot hide a selected CPU's missing source."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True

        with self.assertRaises(GLTCore.GLTTestSkip) as context:
            glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                self.sd, self.mock_logger
            )
        self.assertIn("ETE/ETM source", str(context.exception))

    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.subprocess.run"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight.perf_utils.resolve_perf_command",
        return_value=("/usr/bin/perf", ""),
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CORESIGHT_DEVICES_DIR"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CPU_ONLINE_PATH"
    )
    @GLTTestBase.unittest.mock.patch(
        "src.modules.hwtracing.glt_test_coresight._CS_ETM_EVENT_DIR"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_selects_real_cpu_ids_on_a_sparse_online_mask(
        self, mock_nvdev, mock_cs_etm_dir, mock_online_path,
        mock_coresight_devices_dir, _mock_resolve, mock_run
    ):
        """Both -C lists must name real online CPU IDs, not 0..N-1.

        A host with CPUs 56-111 offlined still has 224 online CPUs, enough to
        cross the boundary -- but naming 0-111 would try to open events on
        offline CPUs and fail for a reason that has nothing to do with trace
        ID maps.
        """
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        mock_cs_etm_dir.exists.return_value = True
        mock_online_path.read_text.return_value = "0-55,112-279\n"
        online_cpu_ids = list(range(56)) + list(range(112, 280))
        entries = [
            SimpleNamespace(name=f"{kind}{cpu}")
            for cpu in online_cpu_ids
            for kind in ("ete", "trbe")
        ]
        entries += [
            SimpleNamespace(name=name)
            for name in ("ete112-link", "etm_cti", "trbe112-link")
        ]
        mock_coresight_devices_dir.is_dir.return_value = True
        mock_coresight_devices_dir.iterdir.side_effect = lambda: iter(entries)
        mock_run.side_effect = [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
            self.sd, self.mock_logger
        )

        control_argv, trigger_argv = (c.args[0] for c in mock_run.call_args_list)
        control_cpus = [
            int(cpu)
            for cpu in control_argv[control_argv.index("-C") + 1].split(",")
        ]
        trigger_cpus = [
            int(cpu)
            for cpu in trigger_argv[trigger_argv.index("-C") + 1].split(",")
        ]
        self.assertEqual(len(control_cpus), 111)
        self.assertEqual(len(trigger_cpus), 112)
        # 56 low CPUs, then the mask jumps to 112 -- the offline 56..111 must
        # not appear in either list.
        self.assertEqual(control_cpus[:57], list(range(56)) + [112])
        self.assertFalse(any(56 <= cpu < 112 for cpu in trigger_cpus))
        self.assertEqual(trigger_cpus[:-1], control_cpus)

    @GLTTestBase.unittest.mock.patch("src.modules.hwtracing.glt_test_coresight.NvDev")
    def test_skips_when_online_cpu_mask_is_unusable(self, mock_nvdev):
        """Unreadable, malformed, and empty masks all fail closed as SKIP."""
        mock_nvdev.return_value.is_cpu_type.side_effect = lambda name: name == "vera"
        cases = (
            OSError("no such file"),
            "0-3,,8",
            "\n",
        )
        for case in cases:
            if isinstance(case, BaseException):
                read_text = self.mock.Mock(side_effect=case)
            else:
                read_text = self.mock.Mock(return_value=case)
            fake = SimpleNamespace(read_text=read_text)
            with self.subTest(case=case), self.mock.patch(
                "src.modules.hwtracing.glt_test_coresight._CPU_ONLINE_PATH", fake
            ), self.assertRaises(GLTCore.GLTTestSkip) as context:
                glt_test_coresight.test_perf_cs_etm_multisocket_trace_id_map(
                    self.sd, self.mock_logger
                )
            self.assertIn("online cpu mask", str(context.exception).lower())

    def test_coresight_cpu_ids_require_exact_device_names(self):
        """Extract CPU IDs only from exact ETE/ETM and TRBE device names."""
        entries = [
            SimpleNamespace(name=name)
            for name in (
                "ete0", "ete1", "etm12", "ete2-link", "etm_cti", "etr0",
                "trbe0", "trbe12", "trbe1-link",
            )
        ]
        fake_dir = SimpleNamespace(
            is_dir=lambda: True, iterdir=lambda: iter(entries)
        )
        with self.mock.patch(
            "src.modules.hwtracing.glt_test_coresight._CORESIGHT_DEVICES_DIR",
            fake_dir,
        ):
            self.assertEqual(
                glt_test_coresight._coresight_cpu_ids(
                    glt_test_coresight._ETE_SOURCE_RE
                ),
                {0, 1, 12},
            )
            self.assertEqual(
                glt_test_coresight._coresight_cpu_ids(
                    glt_test_coresight._TRBE_SINK_RE
                ),
                {0, 12},
            )

    # pylint: enable=too-many-arguments,too-many-positional-arguments
    # pylint: enable=unknown-option-value


if __name__ == "__main__":
    GLTTestBase.unittest.main()
