"""Unit tests for the Vera LS64 userspace ABI checks."""
# pylint: disable=wrong-import-position,protected-access

import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import glt_core as GLTCore
from src.modules.arm64 import glt_test_ls64_kvm as ls64_kvm


class TestVeraLs64UserspaceAbi(GLTTestBase):
    """Validate DGX-16333 helper and runtime decision logic."""

    def test_read_cpu_features_merges_feature_lines(self):
        """CPU feature parsing is case-insensitive and accumulates all CPUs."""
        mock_open = self.mock.mock_open(
            read_data=(
                "processor\t: 0\n"
                "Features\t: fp asimd LS64\n\n"
                "processor\t: 1\n"
                "Features\t: fp aes\n"
            )
        )
        with self.mock.patch("builtins.open", mock_open):
            features = ls64_kvm._read_cpu_features()

        self.assertIn("ls64", features)
        self.assertIn("aes", features)

    def test_vera_ls64_userspace_abi_has_no_broad_kselftest_dependency(self):
        """The LS64 gate is not blocked by unrelated broad KVM selftest setup."""
        test_func = ls64_kvm.test_vera_ls64_userspace_abi
        while hasattr(test_func, "__wrapped__"):
            self.assertFalse(hasattr(test_func, "requires"))
            test_func = test_func.__wrapped__
        self.assertFalse(hasattr(test_func, "requires"))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_skips_non_vera(self, mock_nvdev):
        """The runtime test only applies to Vera."""
        mock_nvdev.return_value.is_cpu_type.return_value = False

        with self.assertRaises(GLTCore.GLTTestSkip):
            ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_fails_missing_cpuinfo_ls64(self, mock_nvdev):
        """Vera must expose ls64 in /proc/cpuinfo."""
        mock_nvdev.return_value.is_cpu_type.return_value = True

        with self.mock.patch.object(ls64_kvm, "_read_cpu_features", return_value=set()):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)

        self.assertIn("does not expose", str(context.exception))
        self.assertIn(ls64_kvm.LS64_PATCH_SERIES, str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_fails_missing_hwcap(self, mock_nvdev):
        """Vera must expose the LS64 HWCAP bit."""
        mock_nvdev.return_value.is_cpu_type.return_value = True

        with self.mock.patch.object(
            ls64_kvm,
            "_read_cpu_features",
            return_value={ls64_kvm.LS64_CPUINFO_FEATURE},
        ), self.mock.patch.object(ls64_kvm, "_getauxval", return_value=0):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)

        self.assertIn("HWCAP3_LS64", str(context.exception))
        self.assertIn(ls64_kvm.LS64_PATCH_SERIES, str(context.exception))

    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm._getauxval")
    @GLTTestBase.unittest.mock.patch(
        "src.modules.arm64.glt_test_ls64_kvm._read_cpu_features"
    )
    @GLTTestBase.unittest.mock.patch("src.modules.arm64.glt_test_ls64_kvm.NvDev")
    def test_vera_ls64_userspace_abi_passes_when_capabilities_are_present(
        self, mock_nvdev, mock_features, mock_getauxval
    ):
        """The pass path is source-tree independent."""
        mock_nvdev.return_value.is_cpu_type.return_value = True
        mock_features.return_value = {ls64_kvm.LS64_CPUINFO_FEATURE}
        mock_getauxval.return_value = ls64_kvm.HWCAP3_LS64

        ls64_kvm.test_vera_ls64_userspace_abi(None, self.mock_logger)
        self.mock_logger.info.assert_called_once_with(
            "Vera LS64 userspace ABI checks passed."
        )


if __name__ == "__main__":
    GLTTestBase.unittest.main()
