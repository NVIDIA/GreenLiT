"""Offline tests for Vera mTHP contpte consumer validation."""
# pylint: disable=protected-access,too-many-public-methods

import errno
import os
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from tests.test_base import GLTTestBase
from common import glt_core as GLTCore
from src.modules.mm import glt_test_mthp_contpte as mthp

MODULE_PATH = "src.modules.mm.glt_test_mthp_contpte"


def snapshot(mapped=16, cont=16, pfn=1, ptes=16, size=65536):
    return mthp.MthpContpteResult(
        addr=0x10000000, size=size, mapped_pages=mapped, cont_pages=cont,
        pfn_contiguous=pfn, page_size=4096, cont_pte_size=size, cont_ptes=ptes,
        large_folio=1, folio_order=ptes.bit_length() - 1,
    )


class TestMthpHelpers(unittest.TestCase):
    def test_candidates_require_enabled_size_and_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, enabled in (("hugepages-128kB", "always [madvise] never"), ("hugepages-64kB", "[always] madvise never"), ("hugepages-256kB", "always madvise [never]")):
                directory = root / name
                directory.mkdir()
                (directory / "enabled").write_text(enabled, encoding="utf-8")
                if "256kB" not in name:
                    stats = directory / "stats"
                    stats.mkdir()
                    (stats / "anon_fault_alloc").write_text("0\n", encoding="utf-8")
            candidates = mthp._mthp_candidates(root)
        self.assertEqual([size for size, _ in candidates], [65536, 131072])

    def test_unsupported_probe_mapping_skips(self):
        with unittest.mock.patch.object(
            mthp.fcntl, "ioctl", side_effect=OSError(errno.EOPNOTSUPP, "unsupported")
        ):
            with self.assertRaises(mthp.MthpSkip):
                mthp._probe_snapshot(3, 0x10000000, 65536)

    def test_snapshot_result_mapping(self):
        self.assertEqual(mthp._evaluate_snapshot(snapshot(), True)[0], "pass")
        self.assertEqual(mthp._evaluate_snapshot(snapshot(cont=15), True)[0], "fail")
        self.assertEqual(mthp._evaluate_snapshot(snapshot(pfn=0), True)[0], "fail")
        self.assertEqual(mthp._evaluate_snapshot(snapshot(mapped=15), True)[0], "skip")
        self.assertEqual(mthp._evaluate_snapshot(snapshot(ptes=1), True)[0], "skip")
        no_folio = snapshot()
        no_folio.large_folio = 0
        self.assertEqual(mthp._evaluate_snapshot(no_folio, True)[0], "fail")
        empty = snapshot(mapped=0, cont=0)
        empty.large_folio = 0
        self.assertEqual(mthp._evaluate_snapshot(empty, True)[0], "skip")
        self.assertEqual(mthp._evaluate_snapshot(snapshot(cont=0), False)[0], "pass")
        self.assertEqual(mthp._evaluate_snapshot(snapshot(cont=4), False)[0], "fail")
        self.assertEqual(mthp._evaluate_snapshot(snapshot(cont=16), False)[0], "fail")

    def test_real_userspace_mapping_data_and_mprotect_lifecycle(self):
        page_size = os.sysconf("SC_PAGE_SIZE")
        mapping = mthp.MthpMapping(page_size * 16, page_size)
        try:
            mapping.fill(1)
            mapping.verify(1)
            mapping.set_first_page_access(mthp.PROT_READ)
            mapping.verify(1)
            mapping.set_first_page_access(mthp.PROT_READ | mthp.PROT_WRITE)
            mapping.fill(2)
            mapping.verify(2)
        finally:
            mapping.close()
        self.assertIsNone(mapping.address)

    def test_candidate_exercise_checks_unfold_and_cleans_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_path = Path(temp_dir) / "stats"
            stats_path.write_text("0\n", encoding="utf-8")
            with unittest.mock.patch.object(mthp, "MthpMapping") as mapping_class, \
                    unittest.mock.patch.object(mthp, "_read_alloc_count", side_effect=[0, 1]), \
                    unittest.mock.patch.object(mthp, "_probe_snapshot", side_effect=[snapshot(), snapshot(cont=0)]):
                mapping = mapping_class.return_value
                size = mthp._exercise_candidate(3, (65536, stats_path), 4096)
        self.assertEqual(size, 65536)
        mapping.set_first_page_access.assert_any_call(mthp.PROT_READ)
        mapping.set_first_page_access.assert_any_call(mthp.PROT_READ | mthp.PROT_WRITE)
        mapping.verify.assert_any_call(1)
        mapping.verify.assert_any_call(2)
        mapping.close.assert_called_once()

    def test_candidate_without_allocation_skips_and_closes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_path = Path(temp_dir) / "stats"
            stats_path.write_text("0\n", encoding="utf-8")
            with unittest.mock.patch.object(mthp, "MthpMapping") as mapping_class, \
                    unittest.mock.patch.object(mthp, "_read_alloc_count", return_value=0):
                with self.assertRaises(mthp.MthpSkip):
                    mthp._exercise_candidate(3, (65536, stats_path), 4096)
        mapping_class.return_value.close.assert_called_once()


class TestMthpRuntime(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.module_dir = self.root / "module"
        self.module_dir.mkdir()
        self.headers_root = self.root / "headers"
        (self.headers_root / "release" / "build").mkdir(parents=True)
        self.module_state = self.root / "loaded"
        self.mock_config = self._start_patch(
            f"{MODULE_PATH}._read_kernel_config_text",
            return_value="CONFIG_ARM64_CONTPTE=y\nCONFIG_MODULES=y\nCONFIG_MODULE_UNLOAD=y\n",
        )
        self._start_patch(f"{MODULE_PATH}._mthp_candidates", return_value=[(65536, "stats")])
        self._start_patch(f"{MODULE_PATH}.glt_utils.is_cli_available", return_value=True)
        self._start_patch(f"{MODULE_PATH}.glt_utils.sign_kernel_module", return_value=True)
        self._start_patch(f"{MODULE_PATH}.os.open", return_value=99)
        self._start_patch(f"{MODULE_PATH}.os.close", return_value=0)
        self.mock_exercise = self._start_patch(f"{MODULE_PATH}._exercise_candidate", return_value=65536)
        self.mock_run = self._start_patch(f"{MODULE_PATH}.glt_utils.run_command")
        self.mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")

    def _start_patch(self, target, **kwargs):
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def run_runtime(self):
        return mthp._run_mthp_probe(
            self.mock_logger, module_dir=self.module_dir,
            module_state_path=self.module_state, headers_root=self.headers_root,
            kernel_release="release", device_path="/fake/device",
        )

    def test_positive_run_cleans_device_module_and_build(self):
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "pass")
        self.assertEqual(
            [call.args[0] for call in self.mock_run.call_args_list],
            [["make"], ["insmod", "mthp_contpte_probe.ko"], ["rmmod", "mthp_contpte_probe"], ["make", "clean"]],
        )

    def test_all_candidates_incomplete_skips(self):
        self.mock_exercise.side_effect = mthp.MthpSkip("not allocated")
        self.assertEqual(self.run_runtime()[0], "skip")

    def test_candidate_failure_fails(self):
        self.mock_exercise.side_effect = ValueError("bad data")
        self.assertEqual(self.run_runtime()[0], "fail")

    def test_non_unloadable_kernel_skips_before_build(self):
        self.mock_config.return_value = "CONFIG_ARM64_CONTPTE=y\nCONFIG_MODULES=y\n"
        self.assertEqual(self.run_runtime()[0], "skip")
        self.mock_run.assert_not_called()

    def test_unknown_module_after_insmod_failure_is_not_unloaded(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "insmod":
                self.module_state.mkdir()
                raise RuntimeError("load timeout")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        outcome, _ = self.run_runtime()
        self.assertEqual(outcome, "fail")
        self.assertNotIn(["rmmod", "mthp_contpte_probe"], [call.args[0] for call in self.mock_run.call_args_list])

    def test_cleanup_failure_blocks_pass(self):
        def fake_run(command, *_args, **_kwargs):
            if command[0] == "rmmod":
                raise RuntimeError("unload failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.mock_run.side_effect = fake_run
        self.assertEqual(self.run_runtime()[0], "fail")


class TestMthpNativeResult(GLTTestBase):
    def setUp(self):
        super().setUp()
        self.mock_nv_dev = self._start_patch(f"{MODULE_PATH}.NvDev")
        self.mock_nv_dev.return_value.is_cpu_type.return_value = True
        self._start_patch(f"{MODULE_PATH}.platform.machine", return_value="aarch64")
        self.mock_runner = self._start_patch(f"{MODULE_PATH}._run_mthp_probe", return_value=("pass", "passed"))

    def _start_patch(self, target, **kwargs):
        patcher = self.mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_non_vera_skips_before_probe(self):
        self.mock_nv_dev.return_value.is_cpu_type.return_value = False
        with self.assertRaises(GLTCore.GLTTestSkip):
            mthp.test_vera_mthp_contpte_consumer(None, self.mock_logger)
        self.mock_runner.assert_not_called()

    def test_result_mapping(self):
        mthp.test_vera_mthp_contpte_consumer(None, self.mock_logger)
        for outcome, exception in (("fail", GLTCore.GLTTestFail), ("skip", GLTCore.GLTTestSkip)):
            with self.subTest(outcome=outcome):
                self.mock_runner.return_value = (outcome, "reason")
                with self.assertRaises(exception):
                    mthp.test_vera_mthp_contpte_consumer(None, self.mock_logger)


if __name__ == "__main__":
    unittest.main()
