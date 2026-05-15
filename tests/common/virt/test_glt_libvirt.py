"""Unit tests for common.virt.glt_libvirt."""

# pylint: disable=consider-using-with

import os
import tempfile

from tests.test_base import GLTTestBase
from src.common.virt.glt_libvirt import GLTLibvirt


class TestGLTLibvirt(GLTTestBase):
    """Test cases for GLTLibvirt."""

    def setUp(self):
        super().setUp()
        self.run_patcher = self.mock.patch("subprocess.run")
        self.addCleanup(self.run_patcher.stop)
        self.mock_run = self.run_patcher.start()
        self.mock_run.return_value = self.mock.Mock(
            returncode=0, stdout="virsh version 9.0.0", stderr=""
        )
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.domain_xml_path = os.path.join(self.tempdir.name, "domain.xml")
        self.libvirt = GLTLibvirt(logger=self.mock_logger)

    def test_run_virsh_command_accepts_string(self):
        """String commands are split before invoking virsh."""
        self.mock_run.return_value = self.mock.Mock(
            returncode=0, stdout="running\n", stderr=""
        )

        success, output, return_code = self.libvirt.run_virsh_command("domstate vm1")

        self.assertTrue(success)
        self.assertEqual(output, "running")
        self.assertEqual(return_code, 0)
        self.mock_run.assert_called_with(
            ["virsh", "--connect", "qemu:///system", "domstate", "vm1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_run_virsh_command_returns_stderr_on_failure(self):
        """Failures return stderr and the command return code."""
        self.mock_run.return_value = self.mock.Mock(
            returncode=1, stdout="", stderr="domain not found"
        )

        success, output, return_code = self.libvirt.run_virsh_command(["domstate", "vm1"])

        self.assertFalse(success)
        self.assertEqual(output, "domain not found")
        self.assertEqual(return_code, 1)

    def test_define_and_start_defines_and_starts_domain(self):
        """define_and_start defines the domain XML and starts the named domain."""
        self.mock_run.reset_mock()
        self.mock_run.side_effect = [
            self.mock.Mock(returncode=0, stdout="defined", stderr=""),
            self.mock.Mock(returncode=0, stdout="started", stderr=""),
        ]

        with self.mock.patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
                self.mock.patch("os.unlink") as mock_unlink:
            mock_file = self.mock.MagicMock()
            mock_file.name = self.domain_xml_path
            mock_temp_file.return_value.__enter__.return_value = mock_file

            result = self.libvirt.define_and_start("<domain><name>vm1</name></domain>")

        self.assertTrue(result)
        mock_file.write.assert_called_once_with("<domain><name>vm1</name></domain>")
        mock_unlink.assert_called_once_with(self.domain_xml_path)
        self.assertEqual(self.mock_run.call_args_list[0].args[0][-2:], ["define", self.domain_xml_path])
        self.assertEqual(self.mock_run.call_args_list[1].args[0][-2:], ["start", "vm1"])

    def test_define_and_start_undefines_domain_when_start_fails(self):
        """A defined domain is cleaned up when start fails."""
        self.mock_run.reset_mock()
        self.mock_run.side_effect = [
            self.mock.Mock(returncode=0, stdout="defined", stderr=""),
            self.mock.Mock(returncode=1, stdout="", stderr="start failed"),
            self.mock.Mock(returncode=0, stdout="destroyed", stderr=""),
            self.mock.Mock(returncode=0, stdout="undefined", stderr=""),
        ]

        with self.mock.patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
                self.mock.patch("os.unlink"):
            mock_file = self.mock.MagicMock()
            mock_file.name = self.domain_xml_path
            mock_temp_file.return_value.__enter__.return_value = mock_file

            result = self.libvirt.define_and_start("<domain><name>vm1</name></domain>")

        self.assertFalse(result)
        self.assertEqual(
            self.mock_run.call_args_list[2].args[0][-2:],
            ["destroy", "vm1"],
        )
        self.assertEqual(
            self.mock_run.call_args_list[3].args[0][-3:],
            ["undefine", "vm1", "--nvram"],
        )

    def test_define_logs_domain_xml_for_debugging(self):
        """define logs XML at debug level for reproducibility."""
        self.mock_run.reset_mock()
        self.mock_run.return_value = self.mock.Mock(returncode=0, stdout="defined", stderr="")
        xml_config = "<domain><name>vm1</name></domain>"

        with self.mock.patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
                self.mock.patch("os.unlink"):
            mock_file = self.mock.MagicMock()
            mock_file.name = self.domain_xml_path
            mock_temp_file.return_value.__enter__.return_value = mock_file

            result = self.libvirt.define(xml_config)

        self.assertTrue(result)
        self.mock_logger.debug.assert_any_call(
            "Defining libvirt domain %s from XML:\n%s",
            "vm1",
            xml_config,
        )

    def test_generate_domain_xml(self):
        """Generated XML contains the direct boot configuration."""
        xml = self.libvirt.generate_domain_xml({
            "domain_name": "vm1",
            "memory_mb": 1024,
            "vcpus": 8,
            "arch": "aarch64",
            "kernel_path": "/boot/vmlinuz-test",
            "initrd_path": "/boot/initrd-test",
            "kernel_args": "console=ttyAMA0 rdinit=/bin/sh",
        })

        self.assertIn("<name>vm1</name>", xml)
        self.assertIn("<memory unit=\"MiB\">1024</memory>", xml)
        self.assertIn("<vcpu>8</vcpu>", xml)
        self.assertIn("arch=\"aarch64\"", xml)
        self.assertIn("machine=\"virt\"", xml)
        self.assertIn("<kernel>/boot/vmlinuz-test</kernel>", xml)
        self.assertIn("<initrd>/boot/initrd-test</initrd>", xml)
        self.assertIn("<cmdline>console=ttyAMA0 rdinit=/bin/sh</cmdline>", xml)

    def test_generate_domain_xml_requires_kernel_path(self):
        """Direct boot domains require a kernel path."""
        with self.assertRaises(ValueError):
            self.libvirt.generate_domain_xml({"domain_name": "vm1"})

    def test_generate_domain_xml_validates_domain_name_and_arch(self):
        """Domain XML generation rejects empty names and unsupported architectures."""
        with self.assertRaises(ValueError):
            self.libvirt.generate_domain_xml({
                "domain_name": "",
                "kernel_path": "/boot/vmlinuz-test",
            })

        with self.assertRaises(ValueError):
            self.libvirt.generate_domain_xml({
                "domain_name": "vm1",
                "arch": "riscv64",
                "kernel_path": "/boot/vmlinuz-test",
            })

    def test_generate_domain_xml_allows_machine_type_override(self):
        """Generated XML can override the architecture default machine type."""
        xml = self.libvirt.generate_domain_xml({
            "domain_name": "vm1",
            "arch": "x86_64",
            "machine_type": "pc",
            "kernel_path": "/boot/vmlinuz-test",
        })

        self.assertIn("arch=\"x86_64\"", xml)
        self.assertIn("machine=\"pc\"", xml)

    def test_undefine_retries_without_nvram(self):
        """Direct boot domains can be undefined when --nvram is unsupported."""
        self.mock_run.reset_mock()
        self.mock_run.side_effect = [
            self.mock.Mock(returncode=1, stdout="", stderr="no nvram"),
            self.mock.Mock(returncode=0, stdout="undefined", stderr=""),
        ]

        result = self.libvirt.undefine("vm1")

        self.assertTrue(result)
        self.assertEqual(self.mock_run.call_args_list[0].args[0][-3:], ["undefine", "vm1", "--nvram"])
        self.assertEqual(self.mock_run.call_args_list[1].args[0][-2:], ["undefine", "vm1"])


if __name__ == "__main__":
    GLTTestBase.unittest.main()
