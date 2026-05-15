# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Libvirt helper for GreenLiT tests."""

import os
import shlex
import subprocess
import tempfile
from typing import Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET


VirshCommand = Union[str, Sequence[str]]


class GLTLibvirt:
    """Small virsh wrapper for tests that need local libvirt domains."""

    def __init__(self, logger, uri: str = "qemu:///system"):
        if logger is None:
            raise ValueError("Logger is required")

        self.uri = uri
        self.logger = logger

        success, output, _ = self.run_virsh_command(["uri"])
        if not success:
            raise ConnectionError(f"Failed to connect to virsh: {output}")

    @staticmethod
    def _command_to_args(command: VirshCommand) -> list[str]:
        if isinstance(command, str):
            return shlex.split(command)
        return list(command)

    def run_virsh_command(
        self,
        command: VirshCommand,
        xml_file: Optional[str] = None,
        timeout: int = 60,
    ) -> Tuple[bool, str, int]:
        """Run a virsh command and return success, output, and return code."""
        full_command = ["virsh", "--connect", self.uri]
        full_command.extend(self._command_to_args(command))
        if xml_file:
            full_command.append(xml_file)

        self.logger.debug(
            "Running virsh command: %s",
            " ".join(shlex.quote(arg) for arg in full_command),
        )
        try:
            result = subprocess.run(
                full_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            return False, f"Command timed out after {timeout} seconds: {e}", -1
        except (OSError, subprocess.SubprocessError) as e:
            return False, str(e), -1

        output = result.stdout if result.returncode == 0 else result.stderr or result.stdout
        return result.returncode == 0, output.strip(), result.returncode

    def cleanup(self, domain_name: str) -> None:
        """Best-effort cleanup for a test domain."""
        self.destroy(domain_name)
        self.undefine(domain_name)

    def define(self, xml_config: str) -> bool:
        """Define a persistent libvirt domain from XML."""
        domain_name = self._get_domain_name(xml_config)
        self.logger.debug("Defining libvirt domain %s from XML:\n%s", domain_name, xml_config)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as xml_file:
            xml_file.write(xml_config)
            xml_file_path = xml_file.name

        try:
            success, output, _ = self.run_virsh_command(["define"], xml_file_path)
            if not success:
                self.logger.error("Failed to define domain %s: %s", domain_name, output)
                return False
            return True
        finally:
            os.unlink(xml_file_path)

    def define_and_start(self, xml_config: str) -> bool:
        """Define and start a persistent libvirt domain from XML."""
        domain_name = self._get_domain_name(xml_config)
        if not self.define(xml_config):
            return False

        if not self.start(domain_name):
            self.cleanup(domain_name)
            return False
        return True

    def destroy(self, domain_name: str) -> bool:
        """Force-stop a libvirt domain."""
        success, output, _ = self.run_virsh_command(["destroy", domain_name])
        if not success:
            self.logger.debug("Failed to destroy domain %s: %s", domain_name, output)
        return success

    @staticmethod
    def _get_domain_name(xml_config: str) -> str:
        """Extract the domain name from libvirt XML."""
        try:
            root = ET.fromstring(xml_config)
        except ET.ParseError as e:
            raise ValueError(f"Error parsing domain XML: {e}") from e

        name = root.findtext("name")
        if not name:
            raise ValueError("Domain XML does not contain a domain name")
        return name

    def start(self, domain_name: str) -> bool:
        """Start a defined libvirt domain."""
        success, output, _ = self.run_virsh_command(["start", domain_name])
        if not success:
            self.logger.error("Failed to start domain %s: %s", domain_name, output)
        return success

    def undefine(self, domain_name: str, nvram: bool = True) -> bool:
        """Undefine a libvirt domain."""
        command = ["undefine", domain_name]
        if nvram:
            command.append("--nvram")

        success, output, _ = self.run_virsh_command(command)
        if not success and nvram:
            self.logger.debug(
                "Retrying undefine for domain %s without --nvram after --nvram failure: %s",
                domain_name,
                output,
            )
            success, output, _ = self.run_virsh_command(["undefine", domain_name])
        if not success:
            self.logger.debug("Failed to undefine domain %s: %s", domain_name, output)
        return success

    @staticmethod
    def generate_domain_xml(config: dict) -> str:
        """Generate a small direct-boot domain XML document."""
        domain_name = config.get("domain_name")
        if not domain_name:
            raise ValueError("domain_name is required")
        memory_mb = int(config.get("memory_mb", 512))
        vcpus = int(config.get("vcpus", 1))
        arch = config.get("arch", "aarch64")
        if arch not in ("aarch64", "x86_64"):
            raise ValueError("arch must be one of: aarch64, x86_64")
        kernel_path = config.get("kernel_path")
        initrd_path = config.get("initrd_path")
        kernel_args = config.get("kernel_args", "")

        if memory_mb <= 0:
            raise ValueError("memory_mb must be greater than zero")
        if vcpus <= 0:
            raise ValueError("vcpus must be greater than zero")
        if not kernel_path:
            raise ValueError("kernel_path is required")

        defaults = {
            "aarch64": ("virt", "/usr/bin/qemu-system-aarch64"),
            "x86_64": ("pc-q35", "/usr/bin/qemu-system-x86_64"),
        }
        machine_type, default_emulator = defaults[arch]
        machine_type = config.get("machine_type", machine_type)
        emulator = config.get("emulator", default_emulator)

        domain = ET.Element("domain", {"type": "kvm"})
        ET.SubElement(domain, "name").text = domain_name
        ET.SubElement(domain, "memory", {"unit": "MiB"}).text = str(memory_mb)
        ET.SubElement(domain, "vcpu").text = str(vcpus)

        os_elem = ET.SubElement(domain, "os")
        ET.SubElement(os_elem, "type", {"arch": arch, "machine": machine_type}).text = "hvm"
        ET.SubElement(os_elem, "kernel").text = kernel_path
        if initrd_path:
            ET.SubElement(os_elem, "initrd").text = initrd_path
        if kernel_args:
            ET.SubElement(os_elem, "cmdline").text = kernel_args

        if config.get("enable_acpi", arch != "aarch64"):
            ET.SubElement(domain, "features").append(ET.Element("acpi"))
        ET.SubElement(domain, "cpu", {"mode": "host-passthrough"})
        ET.SubElement(domain, "on_poweroff").text = "destroy"
        ET.SubElement(domain, "on_reboot").text = "destroy"
        ET.SubElement(domain, "on_crash").text = "destroy"

        devices = ET.SubElement(domain, "devices")
        ET.SubElement(devices, "emulator").text = emulator
        serial = ET.SubElement(devices, "serial", {"type": "pty"})
        ET.SubElement(serial, "target", {"port": "0"})
        console = ET.SubElement(devices, "console", {"type": "pty"})
        ET.SubElement(console, "target", {"type": "serial", "port": "0"})

        return ET.tostring(domain, encoding="unicode")
