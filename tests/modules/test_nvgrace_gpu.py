"""Hardware-independent checks for NVGrace CXL readiness patch detection."""
# pylint: disable=wrong-import-position
import sys
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import mock_open, patch

from tests.test_base import GLTTestBase
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import cxl_utils
from common.nv_device import NvDev
from src.modules.gpu import glt_test_nvgrace_gpu as gpu


class NvgraceGpuTestBase(GLTTestBase):
    """Exercise the decorated test with kernel interfaces mocked."""

    def setUp(self):
        super().setUp()
        self.cxl_dvsec = self.enterContext(patch.object(
            cxl_utils, "find_cxl_device_dvsec_base", return_value=0x100,
        ))
        self.gpu_bdfs = self.enterContext(patch.object(
            NvDev, "get_nvidia_3d_bdf", return_value="01:00.0\n",
        ))
        self.kconfig = self.enterContext(patch(
            "scripts.check_kernel_config.parse_kconfig",
            return_value={"CONFIG_NVGRACE_GPU_VFIO_PCI": "m"},
        ))
        self.run = self.enterContext(patch.object(gpu.glt_utils, "run_command"))
        self.run.side_effect = [
            CompletedProcess(["lsmod"], 0, stdout="Module Size Used by\n"),
            CompletedProcess(["modprobe"], 0),
        ]
        self.pci_devices = self.enterContext(patch.object(
            Path, "iterdir", return_value=[Path("/sys/bus/pci/devices/0000:01:00.0")],
        ))
        self.override = self.enterContext(patch.object(
            Path, "read_text", return_value="(null)\n",
        ))
        self.kallsyms = self.enterContext(patch(
            "builtins.open",
            mock_open(read_data=(
                "ffff800081234560 t nvgrace_gpu_wait_device_ready_cxl "
                "[nvgrace_gpu_vfio_pci]\n"
            )),
        ))



class TestNvgraceGpuLoading(NvgraceGpuTestBase):
    """Check module loading and PCI override guards."""

    def test_module_lookup_uses_requested_name(self):
        """Match the requested lsmod name exactly, not a hardcoded driver."""
        self.run.side_effect = None
        self.run.return_value = CompletedProcess(
            ["lsmod"], 0,
            stdout="Module Size Used by\nnvidia_uvm 1 0\n\n",
        )
        for name, expected in (
            ("nvidia_uvm", True),
            ("nvidia", False),
            ("nvgrace_gpu_vfio_pci", False),
        ):
            with self.subTest(module=name):
                # pylint: disable=protected-access
                self.assertEqual(gpu._module_is_loaded(name), (expected, None))

    def test_lsmod_failure_skips_before_loading(self):
        """Unknown module state must not trigger loading or override inspection."""
        self.run.side_effect = RuntimeError("Cannot inspect loaded modules")
        with self.assertRaisesRegex(gpu.GLTCore.GLTTestSkip, "loaded modules"):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
        self.run.assert_called_once_with(["lsmod"], "Cannot inspect loaded modules")
        self.pci_devices.assert_not_called()

    def test_loaded_module_does_not_load_again(self):
        """An already loaded driver needs no override scan or modprobe."""
        self.run.side_effect = [
            CompletedProcess(["lsmod"], 0, stdout="nvgrace_gpu_vfio_pci 1 0\n"),
        ]
        gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
        self.run.assert_called_once_with(["lsmod"], "Cannot inspect loaded modules")
        self.pci_devices.assert_not_called()

    def test_matching_override_skips_before_loading(self):
        """An override on any PCI device blocks loading, even outside GPU discovery."""
        self.pci_devices.return_value = [Path("/sys/bus/pci/devices/0000:02:00.0")]
        self.override.return_value = "nvgrace_gpu_vfio_pci\n"
        with self.assertRaisesRegex(gpu.GLTCore.GLTTestSkip, "driver_override"):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
        self.run.assert_called_once_with(["lsmod"], "Cannot inspect loaded modules")
        self.kallsyms.assert_not_called()

    def test_unreadable_override_skips_before_loading(self):
        """Unknown override state must not permit driver loading."""
        self.override.side_effect = PermissionError("denied")
        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
        self.run.assert_called_once_with(["lsmod"], "Cannot inspect loaded modules")

    def test_pci_enumeration_failure_skips_before_loading(self):
        """An incomplete PCI scan cannot establish safe loading."""
        self.pci_devices.side_effect = OSError("unavailable")
        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
        self.run.assert_called_once_with(["lsmod"], "Cannot inspect loaded modules")

    def test_nonmatching_overrides_allow_loading(self):
        """Empty and other-driver overrides do not select NVGrace VFIO."""
        for override in ("\n", "(null)\n", "vfio-pci\n"):
            with self.subTest(override=override):
                self.run.side_effect = [
                    CompletedProcess(["lsmod"], 0, stdout=""),
                    CompletedProcess(["modprobe"], 0),
                ]
                self.override.return_value = override
                self.run.reset_mock()
                gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
                self.assertEqual(
                    [call.args[0] for call in self.run.call_args_list],
                    [["lsmod"], ["modprobe", "nvgrace-gpu-vfio-pci"]],
                )

    def test_invalid_override_encoding_skips_before_loading(self):
        """An undecodable override cannot establish safe loading."""
        self.override.side_effect = UnicodeError("invalid encoding")
        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
        self.run.assert_called_once_with(["lsmod"], "Cannot inspect loaded modules")



class TestNvgraceGpuSymbols(NvgraceGpuTestBase):
    """Check applicability and kernel symbol matching."""

    def test_builtin_symbol_from_module_fails(self):
        """A module-owned symbol cannot satisfy a built-in configuration."""
        self.kconfig.return_value = {"CONFIG_NVGRACE_GPU_VFIO_PCI": "y"}
        self.kallsyms.return_value = mock_open(read_data=(
            "ffff800081234560 t nvgrace_gpu_wait_device_ready_cxl "
            "[other_module]\n"
        )).return_value

        with self.assertRaisesRegex(gpu.GLTCore.GLTTestFail, "682ecb14e838"):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_builtin_symbol_without_module_suffix_passes(self):
        """A built-in symbol has no kallsyms module suffix."""
        self.kconfig.return_value = {"CONFIG_NVGRACE_GPU_VFIO_PCI": "y"}
        self.kallsyms.return_value = mock_open(read_data=(
            "ffff800081234560 t nvgrace_gpu_wait_device_ready_cxl\n"
        )).return_value

        gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

        self.run.assert_not_called()

    def test_symbol_present_with_nonzero_address_passes(self):
        """The symbol match does not depend on the displayed address."""
        gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

        self.run.assert_any_call(
            ["modprobe", "nvgrace-gpu-vfio-pci"],
            "Cannot load NVGrace VFIO driver",
        )

    def test_compiler_isra_suffix_passes(self):
        """A compiler-assigned ISRA clone number still identifies the patch."""
        self.kallsyms.return_value = mock_open(read_data=(
            "ffffb8284c8f47b8 t nvgrace_gpu_wait_device_ready_cxl.isra.2 "
            "[nvgrace_gpu_vfio_pci]\n"
        )).return_value

        gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_symbol_name_must_match_exactly(self):
        """A synthetic look-alike cannot produce a false positive."""
        self.kallsyms.return_value = mock_open(read_data=(
            "ffff800081234560 t nvgrace_gpu_wait_device_ready_cxl_extra "
            "[nvgrace_gpu_vfio_pci]\n"
        )).return_value

        with self.assertRaisesRegex(gpu.GLTCore.GLTTestFail, "682ecb14e838"):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_missing_symbol_fails_with_patch_hint(self):
        """A loaded pre-patch module produces an actionable failure."""
        self.kallsyms.return_value = mock_open(
            read_data="ffff800081234560 t unrelated_symbol\n"
        ).return_value

        with self.assertRaisesRegex(gpu.GLTCore.GLTTestFail, "682ecb14e838"):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_disabled_driver_skips(self):
        """A disabled driver is outside this test's scope."""
        self.kconfig.return_value = {}

        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

        self.run.assert_not_called()

    def test_missing_kernel_config_skips(self):
        """A missing kernel config cannot establish applicability."""
        self.kconfig.side_effect = FileNotFoundError("missing")

        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

        self.run.assert_not_called()

    def test_module_load_failure_skips(self):
        """An unavailable driver is an unmet test prerequisite."""
        self.run.side_effect = [
            CompletedProcess(["lsmod"], 0, stdout=""),
            RuntimeError("Cannot load NVGrace VFIO driver"),
        ]

        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_no_nvidia_gpu_skips(self):
        """A system without an NVIDIA 3D GPU is outside the test scope."""
        self.gpu_bdfs.return_value = ""

        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_nvidia_gpu_without_cxl_dvsec_skips(self):
        """An NVIDIA GPU without a CXL Device DVSEC is not applicable."""
        self.cxl_dvsec.return_value = None

        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_symbol_from_other_module_fails(self):
        """A symbol from another module cannot prove this patch is present."""
        self.kallsyms.return_value = mock_open(read_data=(
            "ffff800081234560 t nvgrace_gpu_wait_device_ready_cxl "
            "[other_module]\n"
        )).return_value

        with self.assertRaisesRegex(gpu.GLTCore.GLTTestFail, "682ecb14e838"):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)

    def test_unreadable_kallsyms_skips(self):
        """An inaccessible symbol table cannot prove patch absence."""
        self.kallsyms.side_effect = PermissionError("denied")

        with self.assertRaises(gpu.GLTCore.GLTTestSkip):
            gpu.test_nvgrace_gpu_cxl_readiness(None, self.mock_logger)
