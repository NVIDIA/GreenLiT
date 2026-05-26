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

"""Kernel command-line management across bootloader implementations."""

from dataclasses import asdict, dataclass, fields
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from typing import Optional


# Fallback for direct helper usage. GreenLiT tests should pass a per-session
# transaction_root, e.g. "<scheduler.log_dir>/boot-config".
DEFAULT_TRANSACTION_ROOT = "/var/cache/greenlit-boot-config"
PROC_CMDLINE = "/proc/cmdline"
GRUB_DEFAULT_LOADER_BEGIN = "# BEGIN GreenLiT kernel cmdline drop-ins\n"
GRUB_DEFAULT_LOADER_END = "# END GreenLiT kernel cmdline drop-ins\n"
GRUB_REFRESH_TIMEOUT_SECONDS = 120
GRUBBY_TIMEOUT_SECONDS = 30


@dataclass
class KernelParameterResult:  # pylint: disable=too-many-instance-attributes
    """Result returned by kernel command-line operations.

    ``changed`` means GreenLiT changed persistent boot configuration.
    ``reboot_required`` means current ``/proc/cmdline`` does not reflect the
    persistent state yet.
    ``active`` is the current boot state.
    ``configured`` is whether a future boot config contains the parameter.
    ``added_by_glt`` is true only when the parameter is owned by a GreenLiT record/drop-in.
    """

    ok: bool
    reason: str = ""
    changed: bool = False
    reboot_required: bool = False
    backend: str = ""
    active: bool = False
    configured: bool = False
    added_by_glt: bool = False


@dataclass
class KernelCmdlineRecord:
    """Persisted record for kernel parameters appended by GreenLiT.

    Example ``metadata.json`` under ``<session>/glt-test-suite/boot-config``:

    ```json
    {
      "backend": "grub-default",
      "config_path": "/etc/default/grub.d/99_greenlit_test_gicv4_irq_scaling.cfg",
      "parameters": ["kvm-arm.vgic_v4_enable=1"],
      "target": ""
    }
    ```

    The GreenLiT scheduler separately records reboot progress in
    ``test_progress.json``. This metadata only maps a GLT tag to the boot
    configuration that GreenLiT owns and is allowed to remove.
    """

    backend: str
    parameters: list[str]
    config_path: str = ""
    target: str = ""


def validate_kernel_parameter(parameter: str) -> tuple[bool, str]:
    """Validate a single kernel command-line token."""
    invalid_chars = "\n\r\0\"'\\$`"
    if not parameter or not parameter.strip():
        return False, "kernel parameter is empty"
    if any(char in parameter for char in invalid_chars):
        return False, f"kernel parameter contains invalid characters ({invalid_chars})"
    if len(shlex.split(parameter)) != 1:
        return False, "kernel parameter must be a single command line token"
    return True, ""


def _safe_transaction_id(transaction_id) -> str:
    """Convert transaction_id to a safe filename, default to 'default' if empty."""
    if not transaction_id:
        return "default"
    value = str(transaction_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "default"
    if safe == value:
        return safe
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe}_{digest}"


def _cmdline_tokens(cmdline: str) -> list[str]:
    """Split kernel command line string into tokens."""
    try:
        return shlex.split(cmdline)
    except ValueError:
        return cmdline.split()


class KernelCmdlineBackend:
    """Base class for bootloader-specific kernel command-line backends."""

    name = "unsupported"

    def __init__(
        self,
        transaction_root: str = DEFAULT_TRANSACTION_ROOT,
        proc_cmdline: str = PROC_CMDLINE,
    ):
        self.transaction_root = transaction_root
        self.proc_cmdline = proc_cmdline

    @classmethod
    def is_available(cls) -> bool:
        """Return whether this backend can manage persistent boot config."""
        return False

    def _transaction_dir(self, transaction_id) -> str:
        """Return the per-GLT-tag metadata directory."""
        return os.path.join(self.transaction_root, _safe_transaction_id(transaction_id))

    def _metadata_path(self, transaction_id) -> str:
        """Return the metadata file path for a GLT tag."""
        return os.path.join(self._transaction_dir(transaction_id), "metadata.json")

    def _write_metadata(self, transaction_id, record: KernelCmdlineRecord) -> None:
        """Write the metadata record for GLT-owned appended parameters."""
        transaction_dir = self._transaction_dir(transaction_id)
        os.makedirs(transaction_dir, exist_ok=True)
        with open(self._metadata_path(transaction_id), "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, sort_keys=True)

    def _read_metadata(self, transaction_id) -> Optional[KernelCmdlineRecord]:
        """Read the metadata record for a GLT tag if it belongs to this backend."""
        try:
            with open(self._metadata_path(transaction_id), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        if not isinstance(data, dict) or data.get("backend") != self.name:
            return None
        record_fields = {field.name for field in fields(KernelCmdlineRecord)}
        record_data = {key: value for key, value in data.items() if key in record_fields}
        try:
            record = KernelCmdlineRecord(**record_data)
        except (TypeError, ValueError):
            return None
        if (
            not isinstance(record.parameters, list) or
            not all(isinstance(item, str) for item in record.parameters)
        ):
            return None
        if not isinstance(record.config_path, str) or not isinstance(record.target, str):
            return None
        return record

    def _remove_metadata(self, transaction_id) -> None:
        """Remove the metadata record for a GLT tag."""
        try:
            os.remove(self._metadata_path(transaction_id))
        except FileNotFoundError:
            return
        try:
            os.rmdir(self._transaction_dir(transaction_id))
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _get_glt_records(self, transaction_id=None) -> list[tuple[object, KernelCmdlineRecord]]:
        """Return GLT metadata records for this backend, optionally limited to one tag."""
        if transaction_id is not None:
            record = self._read_metadata(transaction_id)
            return [(transaction_id, record)] if record else []

        try:
            entries = os.listdir(self.transaction_root)
        except (FileNotFoundError, PermissionError, OSError):
            return []

        records = []
        for entry in entries:
            record = self._read_metadata(entry)
            if record:
                records.append((entry, record))
        return records

    def _glt_record_has_parameter(self, parameter: str, transaction_id=None) -> bool:
        """Return whether GreenLiT metadata owns this parameter for this backend."""
        return any(parameter in record.parameters for _, record in self._get_glt_records(transaction_id))

    def is_parameter_active(self, parameter: str) -> bool:
        """Return whether the parameter is present in the running kernel cmdline."""
        try:
            with open(self.proc_cmdline, "r", encoding="utf-8") as f:
                tokens = _cmdline_tokens(f.read())
        except (FileNotFoundError, PermissionError, OSError):
            return False
        return parameter in tokens

    def _config_has_parameter(self, parameter: str) -> bool:
        """Return whether persistent backend configuration contains parameter."""
        del parameter
        return False

    def append_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Append a parameter through this backend and tag it as GLT-owned."""
        valid, reason = validate_kernel_parameter(parameter)
        if not valid:
            return KernelParameterResult(False, reason, backend=self.name)
        return self._append_parameter(parameter, transaction_id)

    def _append_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Backend-specific implementation of append_parameter()."""
        del parameter, transaction_id
        return KernelParameterResult(False, "kernel command-line backend is unsupported", backend=self.name)

    def check_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Check current, persistent, and GLT-owned state for a parameter."""
        valid, reason = validate_kernel_parameter(parameter)
        active = self.is_parameter_active(parameter)
        if not valid:
            return KernelParameterResult(False, reason, backend=self.name, active=active)
        return self._check_parameter(parameter, transaction_id, active)

    def _check_parameter(
        self, parameter: str, transaction_id=None, active: bool = False
    ) -> KernelParameterResult:
        """Check active/configured/GLT-owned state for this backend."""
        configured = self._config_has_parameter(parameter)
        added_by_glt = self._glt_record_has_parameter(parameter, transaction_id)
        return KernelParameterResult(
            True,
            backend=self.name,
            active=active,
            configured=configured,
            added_by_glt=added_by_glt,
            reboot_required=active != configured,
        )

    def remove_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Remove a parameter only from GLT-owned persistent boot config."""
        valid, reason = validate_kernel_parameter(parameter)
        if not valid:
            return KernelParameterResult(False, reason, backend=self.name)
        return self._remove_parameter(parameter, transaction_id)

    def _remove_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Backend-specific implementation of remove_parameter()."""
        del parameter, transaction_id
        return KernelParameterResult(False, "kernel command-line backend is unsupported", backend=self.name)


class GrubDefaultBackend(KernelCmdlineBackend):
    """Manage kernel parameters through a GreenLiT-owned GRUB default drop-in."""

    name = "grub-default"

    def __init__(
        self,
        grub_default: str = "/etc/default/grub",
        grub_default_dir: str = "/etc/default/grub.d",
        transaction_root: str = DEFAULT_TRANSACTION_ROOT,
        proc_cmdline: str = PROC_CMDLINE,
    ):
        super().__init__(transaction_root=transaction_root, proc_cmdline=proc_cmdline)
        self.grub_default = grub_default
        self.grub_default_dir = grub_default_dir

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists("/etc/default/grub")

    @staticmethod
    def _get_grub_cmdline_tokens(value) -> list[str]:
        parsed = _cmdline_tokens(value)
        if not parsed:
            return []
        if len(parsed) == 1:
            return parsed[0].split()
        return parsed

    def _read_grub_lines(self) -> tuple[list[str], str]:
        """Read /etc/default/grub and /etc/default/grub.d snippets."""
        paths = [self.grub_default]
        if self._grub_supports_default_dropins():
            paths.extend(sorted(glob.glob(os.path.join(self.grub_default_dir, "*.cfg"))))
        elif self._is_default_dropin_loader_present():
            paths.extend(sorted(glob.glob(os.path.join(
                self.grub_default_dir, "99_greenlit_*.cfg"
            ))))
        lines = []
        for path in paths:
            file_lines, reason = self._read_grub_file(path)
            if reason:
                return [], reason
            lines.extend(file_lines)
        return lines, ""

    @staticmethod
    def _read_grub_file(path: str) -> tuple[list[str], str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.readlines(), ""
        except (FileNotFoundError, PermissionError, OSError) as e:
            return [], f"Unable to read {path}: {e}"

    def _write_dropin(self, dropin_path: str, parameters: list[str]) -> str:
        """Write the GLT-owned GRUB drop-in for one tag."""
        try:
            os.makedirs(self.grub_default_dir, exist_ok=True)
            with open(dropin_path, "w", encoding="utf-8") as f:
                f.write("# Written by GreenLiT. Remove through remove_kernel_parameter().\n")
                f.write(
                    f'GRUB_CMDLINE_LINUX="${{GRUB_CMDLINE_LINUX}} {" ".join(parameters)}"\n'
                )
        except (PermissionError, OSError) as e:
            return f"Unable to write {dropin_path}: {e}"
        return ""

    def _dropin_loader_block(self) -> str:
        """Return the managed /etc/default/grub block for non-drop-in GRUB."""
        pattern = os.path.join(self.grub_default_dir, "99_greenlit_*.cfg")
        return (
            GRUB_DEFAULT_LOADER_BEGIN +
            f"for greenlit_grub_cfg in {pattern}; do\n"
            '    [ -r "$greenlit_grub_cfg" ] && . "$greenlit_grub_cfg"\n'
            "done\n"
            "unset greenlit_grub_cfg\n" +
            GRUB_DEFAULT_LOADER_END
        )

    def _ensure_default_dropin_loader(self) -> str:
        """Add a GLT-owned source block if this GRUB ignores grub.d snippets."""
        if self._grub_supports_default_dropins():
            return ""
        try:
            with open(self.grub_default, "r", encoding="utf-8") as f:
                content = f.read()
            if GRUB_DEFAULT_LOADER_BEGIN in content:
                return ""
            with open(self.grub_default, "a", encoding="utf-8") as f:
                if content and not content.endswith("\n"):
                    f.write("\n")
                f.write("\n")
                f.write(self._dropin_loader_block())
        except (PermissionError, OSError) as e:
            return f"Unable to add GRUB drop-in loader: {e}"
        return ""

    def _is_default_dropin_loader_present(self) -> bool:
        """Return whether the GLT-owned source block is present."""
        try:
            with open(self.grub_default, "r", encoding="utf-8") as f:
                return GRUB_DEFAULT_LOADER_BEGIN in f.read()
        except (PermissionError, OSError):
            return False

    @staticmethod
    def _read_optional_file(path: str) -> str | None:
        """Return existing file content, or None if it cannot be read."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, PermissionError, OSError):
            return None

    @staticmethod
    def _restore_file_content(path: str, previous_content: str | None) -> None:
        """Best-effort restore for a file captured before a partial change."""
        try:
            if previous_content is None:
                os.remove(path)
            else:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(previous_content)
        except (FileNotFoundError, PermissionError, OSError):
            pass

    def _restore_metadata_content(
        self,
        transaction_id,
        previous_content: str | None,
    ) -> None:
        """Best-effort restore for a metadata file captured before a partial change."""
        metadata_path = self._metadata_path(transaction_id)
        self._restore_file_content(metadata_path, previous_content)
        if previous_content is None:
            try:
                os.rmdir(self._transaction_dir(transaction_id))
            except (FileNotFoundError, OSError):
                pass

    def _remove_default_dropin_loader_if_unused(self) -> str:
        """Remove the GLT-owned source block once no GLT records remain."""
        if self._get_glt_records():
            return ""
        try:
            with open(self.grub_default, "r", encoding="utf-8") as f:
                content = f.read()
            start = content.find(GRUB_DEFAULT_LOADER_BEGIN)
            end = content.find(GRUB_DEFAULT_LOADER_END)
            if start == -1 or end == -1:
                return ""
            end += len(GRUB_DEFAULT_LOADER_END)
            updated = content[:start].rstrip() + "\n" + content[end:].lstrip("\n")
            with open(self.grub_default, "w", encoding="utf-8") as f:
                f.write(updated)
        except (PermissionError, OSError) as e:
            return f"Unable to remove GRUB drop-in loader: {e}"
        return ""

    def _rollback_append_files(
        self,
        dropin_path: str,
        previous_dropin_content: str | None,
        loader_was_present: bool,
    ) -> None:
        """Best-effort cleanup for append failures before ownership is durable."""
        try:
            if previous_dropin_content is None:
                os.remove(dropin_path)
            else:
                with open(dropin_path, "w", encoding="utf-8") as f:
                    f.write(previous_dropin_content)
        except (FileNotFoundError, PermissionError, OSError):
            pass

        if not loader_was_present:
            self._remove_default_dropin_loader_if_unused()

    def _rollback_remove_files(
        self,
        previous_grub_default_content: str | None,
        snapshots: list[tuple[object, str, str | None, str | None]],
    ) -> None:
        """Best-effort rollback for remove failures before ownership is durable."""
        for record_id, config_path, dropin_content, metadata_content in reversed(snapshots):
            if config_path:
                self._restore_file_content(config_path, dropin_content)
            self._restore_metadata_content(record_id, metadata_content)
        if previous_grub_default_content is not None:
            self._restore_file_content(self.grub_default, previous_grub_default_content)

    def _config_has_parameter(self, parameter: str) -> bool:
        """Return whether any GRUB default config source contains parameter."""
        lines, reason = self._read_grub_lines()
        if reason:
            return False
        for line in lines:
            match = re.match(r"^(GRUB_CMDLINE_LINUX(?:_DEFAULT)?)=(.*)$", line.rstrip("\n"))
            if match and parameter in self._get_grub_cmdline_tokens(match.group(2)):
                return True
        return False

    @staticmethod
    def _grub_config_candidates() -> list[str]:
        """Return known generated GRUB config locations across distros."""
        return [
            "/boot/grub2/grub.cfg",
            "/boot/grub/grub.cfg",
            "/boot/efi/EFI/ubuntu/grub.cfg",
            "/boot/efi/EFI/redhat/grub.cfg",
            "/boot/efi/EFI/sles/grub.cfg",
        ]

    def _grub_supports_default_dropins(self) -> bool:
        """Return whether the local GRUB mkconfig script sources grub.d snippets."""
        for command in ("update-grub", "grub2-mkconfig", "grub-mkconfig"):
            command_path = shutil.which(command)
            if not command_path:
                continue
            if command == "update-grub":
                return True
            try:
                with open(command_path, "r", encoding="utf-8") as f:
                    if "default/grub.d" in f.read():
                        return True
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
        return False

    def _refresh_grub_config(self) -> tuple[bool, str]:
        """Regenerate the bootloader-specific GRUB config."""
        if shutil.which("update-grub"):
            command = ["update-grub"]
        elif shutil.which("grub2-mkconfig"):
            cfg_path = next(
                (path for path in self._grub_config_candidates() if os.path.exists(path)),
                self._grub_config_candidates()[0],
            )
            command = ["grub2-mkconfig", "-o", cfg_path]
        elif shutil.which("grub-mkconfig"):
            command = ["grub-mkconfig", "-o", "/boot/grub/grub.cfg"]
        else:
            return False, "No GRUB update command found"

        try:
            result = subprocess.run(
                command, shell=False, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=GRUB_REFRESH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as e:
            return False, f"GRUB update command timed out after {GRUB_REFRESH_TIMEOUT_SECONDS} seconds: {e}"
        except (OSError, subprocess.SubprocessError) as e:
            return False, str(e)
        if result.returncode != 0:
            return False, result.stderr or result.stdout
        return True, ""

    def _dropin_path(self, transaction_id) -> str:
        """Return the GLT-owned GRUB drop-in path for a tag."""
        return os.path.join(
            self.grub_default_dir,
            f"99_greenlit_{_safe_transaction_id(transaction_id)}.cfg",
        )

    def _record_config_path_matches(self, record: KernelCmdlineRecord, expected_path: str) -> bool:
        """Return whether metadata points at the expected GLT-owned drop-in."""
        return (
            not record.config_path or
            os.path.realpath(record.config_path) == os.path.realpath(expected_path)
        )

    # pylint: disable=too-many-return-statements
    def _append_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Append parameter through a GLT-owned GRUB drop-in."""
        status = self._check_parameter(
            parameter, transaction_id, active=self.is_parameter_active(parameter)
        )
        if status.added_by_glt or status.configured:
            return KernelParameterResult(
                True,
                backend=self.name,
                active=status.active,
                configured=status.configured,
                added_by_glt=status.added_by_glt,
                reboot_required=not status.active,
            )

        dropin_path = self._dropin_path(transaction_id)
        record = self._read_metadata(transaction_id) or KernelCmdlineRecord(
            backend=self.name,
            parameters=[],
            config_path=dropin_path,
        )
        if not self._record_config_path_matches(record, dropin_path):
            return KernelParameterResult(
                False,
                f"Transaction metadata references unexpected GRUB drop-in: {record.config_path}",
                backend=self.name,
            )
        if parameter not in record.parameters:
            record.parameters.append(parameter)
        record.config_path = dropin_path

        loader_was_present = self._is_default_dropin_loader_present()
        previous_grub_default_content = self._read_optional_file(self.grub_default)
        previous_dropin_content = self._read_optional_file(record.config_path)
        previous_metadata_content = self._read_optional_file(self._metadata_path(transaction_id))
        reason = self._ensure_default_dropin_loader()
        if reason:
            if not loader_was_present:
                self._restore_file_content(self.grub_default, previous_grub_default_content)
            return KernelParameterResult(False, reason, backend=self.name)

        reason = self._write_dropin(record.config_path, record.parameters)
        if reason:
            self._rollback_append_files(record.config_path, previous_dropin_content, loader_was_present)
            return KernelParameterResult(False, reason, backend=self.name)

        try:
            self._write_metadata(transaction_id, record)
        except (PermissionError, OSError) as e:
            self._rollback_append_files(record.config_path, previous_dropin_content, loader_was_present)
            return KernelParameterResult(False, f"Unable to create transaction metadata: {e}", backend=self.name)

        ok, reason = self._refresh_grub_config()
        if not ok:
            self._restore_metadata_content(transaction_id, previous_metadata_content)
            self._rollback_append_files(record.config_path, previous_dropin_content, loader_was_present)
            return KernelParameterResult(False, reason, backend=self.name)

        result = self._check_parameter(
            parameter, transaction_id, active=self.is_parameter_active(parameter)
        )
        result.changed = True
        return result

    # pylint: disable=too-many-return-statements,too-many-branches
    def _remove_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Remove parameter only from GLT-owned GRUB drop-ins."""
        changed = False
        previous_grub_default_content = self._read_optional_file(self.grub_default)
        snapshots = []
        for record_id, record in self._get_glt_records(transaction_id):
            if parameter not in record.parameters:
                continue

            expected_dropin_path = self._dropin_path(record_id)
            if not self._record_config_path_matches(record, expected_dropin_path):
                return KernelParameterResult(
                    False,
                    f"Transaction metadata references unexpected GRUB drop-in: {record.config_path}",
                    backend=self.name,
                )
            record.config_path = expected_dropin_path
            snapshots.append((
                record_id,
                record.config_path,
                self._read_optional_file(record.config_path) if record.config_path else None,
                self._read_optional_file(self._metadata_path(record_id)),
            ))
            record.parameters = [item for item in record.parameters if item != parameter]
            if record.parameters:
                reason = self._write_dropin(record.config_path, record.parameters)
                if reason:
                    self._rollback_remove_files(previous_grub_default_content, snapshots)
                    return KernelParameterResult(False, reason, backend=self.name)
                try:
                    self._write_metadata(record_id, record)
                except (PermissionError, OSError) as e:
                    self._rollback_remove_files(previous_grub_default_content, snapshots)
                    return KernelParameterResult(
                        False, f"Unable to update transaction metadata: {e}", backend=self.name
                    )
            else:
                try:
                    if record.config_path and os.path.exists(record.config_path):
                        os.remove(record.config_path)
                except (PermissionError, OSError) as e:
                    self._rollback_remove_files(previous_grub_default_content, snapshots)
                    return KernelParameterResult(
                        False, f"Unable to remove GRUB drop-in: {e}", backend=self.name
                    )
                try:
                    self._remove_metadata(record_id)
                except (PermissionError, OSError) as e:
                    self._rollback_remove_files(previous_grub_default_content, snapshots)
                    return KernelParameterResult(
                        False, f"Unable to remove transaction metadata: {e}", backend=self.name
                    )
                reason = self._remove_default_dropin_loader_if_unused()
                if reason:
                    self._rollback_remove_files(previous_grub_default_content, snapshots)
                    return KernelParameterResult(False, reason, backend=self.name)
            changed = True

        if not changed:
            return self._check_parameter(
                parameter, transaction_id, active=self.is_parameter_active(parameter)
            )

        ok, reason = self._refresh_grub_config()
        if not ok:
            self._rollback_remove_files(previous_grub_default_content, snapshots)
            return KernelParameterResult(False, reason, backend=self.name)
        result = self._check_parameter(
            parameter, transaction_id, active=self.is_parameter_active(parameter)
        )
        result.changed = True
        return result


class GrubbyBackend(KernelCmdlineBackend):
    """Manage kernel parameters through grubby on RHEL-family systems."""

    name = "grubby"

    def __init__(
        self,
        transaction_root: str = DEFAULT_TRANSACTION_ROOT,
        proc_cmdline: str = PROC_CMDLINE,
        target: str = "DEFAULT",
    ):
        super().__init__(transaction_root=transaction_root, proc_cmdline=proc_cmdline)
        self.target = target

    @classmethod
    def is_available(cls) -> bool:
        """Return whether grubby is installed."""
        return shutil.which("grubby") is not None

    @staticmethod
    def _run_grubby(args: list[str]) -> tuple[bool, str]:
        """Run grubby and return success plus output."""
        try:
            result = subprocess.run(
                ["grubby", *args], shell=False, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=GRUBBY_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as e:
            return False, f"grubby command timed out after {GRUBBY_TIMEOUT_SECONDS} seconds: {e}"
        except (OSError, subprocess.SubprocessError) as e:
            return False, str(e)
        output = result.stdout if result.returncode == 0 else result.stderr or result.stdout
        return result.returncode == 0, output.strip()

    def _config_has_parameter(self, parameter: str, target: str | None = None) -> bool:
        """Return whether grubby reports parameter in the target kernel args."""
        ok, output = self._run_grubby([f"--info={target or self.target}"])
        if not ok:
            return False
        for line in output.splitlines():
            if line.startswith("args="):
                parsed = _cmdline_tokens(line.split("=", 1)[1])
                tokens = parsed[0].split() if parsed else []
                return parameter in tokens
        return False

    def _append_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Append parameter with grubby and record it as GLT-owned."""
        active = self.is_parameter_active(parameter)
        if self._config_has_parameter(parameter):
            return KernelParameterResult(
                True,
                reboot_required=not active,
                backend=self.name,
                active=active,
                configured=True,
                added_by_glt=self._glt_record_has_parameter(parameter, transaction_id),
            )

        ok, reason = self._run_grubby([f"--update-kernel={self.target}", f"--args={parameter}"])
        if not ok:
            return KernelParameterResult(False, reason, backend=self.name)

        record = self._read_metadata(transaction_id) or KernelCmdlineRecord(
            backend=self.name,
            parameters=[],
            target=self.target,
        )
        if parameter not in record.parameters:
            record.parameters.append(parameter)
        record.target = record.target or self.target
        try:
            self._write_metadata(transaction_id, record)
        except (PermissionError, OSError) as e:
            ok, rollback_reason = self._run_grubby([
                f"--update-kernel={self.target}",
                f"--remove-args={parameter}",
            ])
            reason = f"Unable to create transaction metadata: {e}"
            if not ok:
                reason = f"{reason}; failed to roll back grubby argument: {rollback_reason}"
            return KernelParameterResult(False, reason, changed=not ok, backend=self.name)

        result = self._check_parameter(
            parameter, transaction_id, active=self.is_parameter_active(parameter)
        )
        result.changed = True
        return result

    def _remove_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Remove parameter through grubby only if metadata says GLT owns it."""
        record = self._read_metadata(transaction_id)
        if not record or parameter not in record.parameters:
            return self._check_parameter(
                parameter, transaction_id, active=self.is_parameter_active(parameter)
            )
        target = record.target or self.target

        ok, reason = self._run_grubby([f"--update-kernel={target}", f"--remove-args={parameter}"])
        if not ok:
            return KernelParameterResult(False, reason, backend=self.name)

        record.parameters = [item for item in record.parameters if item != parameter]
        try:
            if record.parameters:
                self._write_metadata(transaction_id, record)
            else:
                self._remove_metadata(transaction_id)
        except (PermissionError, OSError) as e:
            rollback_ok, rollback_reason = self._run_grubby([
                f"--update-kernel={target}",
                f"--args={parameter}",
            ])
            reason = f"Unable to update transaction metadata: {e}"
            if not rollback_ok:
                reason = f"{reason}; failed to roll back grubby argument: {rollback_reason}"
            return KernelParameterResult(
                False, reason, changed=not rollback_ok, backend=self.name
            )
        active = self.is_parameter_active(parameter)
        configured = self._config_has_parameter(parameter, target)
        result = KernelParameterResult(
            True,
            changed=True,
            reboot_required=active != configured,
            backend=self.name,
            active=active,
            configured=configured,
            added_by_glt=self._glt_record_has_parameter(parameter, transaction_id),
        )
        return result


class SystemdBootBackend(KernelCmdlineBackend):
    """Placeholder backend for systemd-boot loader entries."""

    name = "systemd-boot"

    @classmethod
    def is_available(cls) -> bool:
        """Return whether systemd-boot loader entry directories exist."""
        entry_dirs = [
            "/boot/loader/entries",
            "/efi/loader/entries",
            "/boot/efi/loader/entries",
        ]
        return any(os.path.isdir(path) for path in entry_dirs)

    def _append_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Report that systemd-boot append support is not implemented."""
        del parameter, transaction_id
        return KernelParameterResult(
            False,
            "systemd-boot kernel command-line management is not implemented yet",
            backend=self.name,
        )

    def _remove_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Report that systemd-boot remove support is not implemented."""
        del parameter, transaction_id
        return KernelParameterResult(
            False,
            "systemd-boot kernel command-line management is not implemented yet",
            backend=self.name,
        )


class UnsupportedBackend(KernelCmdlineBackend):
    """Backend used when no supported bootloader manager is detected."""

    name = "unsupported"

    def _append_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Report that no backend can append the parameter."""
        del parameter, transaction_id
        return KernelParameterResult(False, "No supported kernel command-line backend found", backend=self.name)

    def _remove_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Report that no backend can remove the parameter."""
        del parameter, transaction_id
        return KernelParameterResult(False, "No supported kernel command-line backend found", backend=self.name)


class KernelCmdlineManager:
    """Facade used by GreenLiT tests to manage kernel command-line parameters."""

    def __init__(self, backend: KernelCmdlineBackend):
        self.backend = backend

    @classmethod
    def detect(cls, transaction_root: str | None = None) -> "KernelCmdlineManager":
        """Detect the best available backend for this system."""
        transaction_root = transaction_root or DEFAULT_TRANSACTION_ROOT
        for backend_cls in (GrubbyBackend, GrubDefaultBackend, SystemdBootBackend):
            if backend_cls.is_available():
                return cls(backend_cls(transaction_root=transaction_root))
        return cls(UnsupportedBackend(transaction_root=transaction_root))

    def append_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Append parameter and tag it as GLT-owned."""
        return self.backend.append_parameter(parameter, transaction_id=transaction_id)

    def check_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Check active/configured/GLT-owned parameter state."""
        return self.backend.check_parameter(parameter, transaction_id=transaction_id)

    def is_parameter_active(self, parameter: str) -> bool:
        """Return whether parameter is present in the running kernel cmdline."""
        return self.backend.is_parameter_active(parameter)

    def remove_parameter(self, parameter: str, transaction_id=None) -> KernelParameterResult:
        """Remove parameter from GLT-owned boot config."""
        return self.backend.remove_parameter(parameter, transaction_id=transaction_id)
