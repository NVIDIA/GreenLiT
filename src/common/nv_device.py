# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
This module handles the nvidia specific device information.
"""

import json
import os
import platform
import re
import subprocess
import shutil

# CPU type to SoC ID mapping
# Add new CPU types here as needed
CPU_SOC_IDS = {
    "grace": "036b:0241",
    "vera": "036b:0410",
}


class NvDev:
    """Nvidia device handler"""

    nca_available = False
    nca_dev_funcs = "/usr/local/sbin/nv_scripts/device_funcs.bash"

    def __init__(self):
        """Init function"""
        if os.path.exists(self.nca_dev_funcs):
            self.nca_available = True

    def __modalias_parser(self, parser) -> list[str]:
        # Use a dict to store mapping of parser and private function.
        parser_func_dict = {"dmi": self.__dmi_modalias_parser}
        if parser in parser_func_dict:
            modalias_list = parser_func_dict[parser]()
            return modalias_list
        return None

    def __dmi_modalias_parser(self) -> list[str]:
        dut_dmi_modalias = "/sys/class/dmi/id/modalias"
        modalias_list = []
        if os.path.isfile(dut_dmi_modalias):
            with open(dut_dmi_modalias, mode="r", encoding="utf-8") as f:
                modalias = f.read().replace("\n", "")
                modalias_list.append(modalias)
                return modalias_list
        return None

    @staticmethod
    def get_nvidia_3d_bdf() -> str:
        """Get device BDF of NVIDIA 3D class"""
        lspci_output = subprocess.run(r"""lspci -d 10de::302 | cut -d' ' -f1""",
                                      capture_output=True, shell=True, text=True, check=False)

        return lspci_output.stdout

    @staticmethod
    def get_release_info() -> dict:
        """Get the release information follows freedesktop.org standard."""
        if hasattr(platform, "freedesktop_os_release"):
            return platform.freedesktop_os_release()
        os_release = {}
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os_release[key] = value.strip('"')
        return os_release

    def plat_support_bmc(self) -> bool:
        """
        Check if platform supports BMC.
        Returns True if BMC is supported on this platform.
        Returns False if BMC is not supported.
        """
        if self.nca_available:
            command = f"source {self.nca_dev_funcs} && plat_support_bmc"
            p = subprocess.run(command, shell=True, executable="/bin/bash", check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0:
                return True
            if p.returncode == 1:
                return False

        if shutil.which("ipmitool"):
            p = subprocess.run(
                ["ipmitool", "mc", "info"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if p.returncode == 0:
                return True

        for dev in ("/dev/ipmi0", "/dev/ipmi/0", "/dev/ipmidev/0"):
            if os.path.exists(dev):
                return True

        return False

    def has_hopper_gpu(self) -> bool:
        """
        Check if system has hopper GPU.
        Returns True if system has hopper GPU.
        Returns False if system has hopper GPU.
        """
        if self.nca_available:
            command = f"source {self.nca_dev_funcs} && has_hopper_gpu"
            p = subprocess.run(command, shell=True, executable="/bin/bash", check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0:
                return True
            if p.returncode == 1:
                return False

        try:
            pci_devices = subprocess.check_output(["lspci", "-n"], stderr=subprocess.DEVNULL).decode("utf-8")
        except subprocess.CalledProcessError:
            return False

        hopper_list = ["2302", "2313", "2321", "2322", "2324", "2330", "2331", "2336",
                       "2337", "2339", "233a", "233d", "2342", "2343", "2345"]

        for pci_id in hopper_list:
            if f"10de:{pci_id}" in pci_devices:
                return True

        return False

    def is_cpu_type(self, cpu_name: str) -> bool:
        """
        Check if system is using a specific CPU type.
        
        Args:
            cpu_name: Name of the CPU type (e.g., 'grace', etc.)
        
        Returns:
            True if the system is using the specified CPU type, False otherwise.
        
        Raises:
            ValueError: If the cpu_name is not recognized.
        """
        if cpu_name not in CPU_SOC_IDS:
            valid_types = ", ".join(sorted(CPU_SOC_IDS.keys()))
            raise ValueError(f"Unknown CPU type '{cpu_name}'. Valid types: {valid_types}")

        soc_id = CPU_SOC_IDS[cpu_name]

        if self.nca_available:
            command = f"source {self.nca_dev_funcs} && is_{cpu_name}_cpu"

            p = subprocess.run(command, shell=True, executable="/bin/bash", check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0:
                return True
            if p.returncode == 1:
                return False

        command = f'grep -q "{soc_id}$" /sys/devices/soc*/soc_id'
        p = subprocess.run(command, shell=True, executable="/bin/bash", check=False,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            return True

        return False

    @staticmethod
    def get_cpu_socket_count() -> int:
        """Get the number of CPU sockets in the system."""
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True, check=False)
            match = re.search(r'Socket\(s\):\s+(\d+)', result.stdout)
            if match:
                return int(match.group(1))
            return -1
        except OSError:
            return -1

    def manifest_skip_func(self, category, specify_parser=None) -> bool:
        """
        Example:
        Add manifest_skip_func("tpm", "dmi") into your test case,
        it'll skip based on what modalias you want. For instance,
        the test case will skip the machine which matched dmi
        modalias in json file.

        Return True for matching modalias string
        Return False for no modalias string or no matching modalias
        in hardware-manifest.json
        """
        hw_manifest_path = "/usr/share/greenlit/hardware-manifest.json"
        modalias_list = []

        replacements = {
            "*": "[^:]*",
            ".": "\\.",
            ",": "\\,",
            "(": "\\(",
            ")": "\\)",
        }

        if os.path.isfile(hw_manifest_path):
            with open(hw_manifest_path, "r", encoding="utf-8") as f:
                hw_manifest = json.load(f)
        else:
            return False

        if specify_parser:
            modalias_list = self.__modalias_parser(specify_parser)
            if not modalias_list:
                return False

        for skip_modalias in hw_manifest[category]:
            if modalias_list:
                if not modalias_list[0].split(":")[0] == skip_modalias.split(":")[0]:
                    continue
            else:
                parser = skip_modalias.split(":")[0]
                modalias_list = self.__modalias_parser(parser)

            for old, new in replacements.items():
                skip_modalias = skip_modalias.replace(old, new)

            for modalias in modalias_list:
                match = re.match(f"{skip_modalias}$", modalias)
                if bool(match):
                    return True

        return False
