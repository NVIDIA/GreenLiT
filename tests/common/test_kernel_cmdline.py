"""Unit tests for common/kernel_cmdline.py."""

# pylint: disable=consider-using-with

import json
import os
import tempfile

from tests.test_base import GLTTestBase
from src.common.kernel_cmdline import (
    GRUB_REFRESH_TIMEOUT_SECONDS,
    GRUBBY_TIMEOUT_SECONDS,
    GrubDefaultBackend,
    GrubbyBackend,
    KernelCmdlineManager,
    KernelCmdlineRecord,
    KernelParameterResult,
    SystemdBootBackend,
    UnsupportedBackend,
    _safe_transaction_id,  # pylint: disable=protected-access
    validate_kernel_parameter,
)


class TestKernelCmdline(GLTTestBase):  # pylint: disable=too-many-public-methods
    """Test kernel command-line management helpers."""

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.proc_cmdline = os.path.join(self.tempdir.name, "cmdline")
        self.grub_default = os.path.join(self.tempdir.name, "grub")
        self.grub_default_dir = os.path.join(self.tempdir.name, "grub.d")
        self.transaction_root = os.path.join(self.tempdir.name, "transactions")

    def write_proc_cmdline(self, value):
        """Write mocked /proc/cmdline content."""
        with open(self.proc_cmdline, "w", encoding="utf-8") as f:
            f.write(value)

    def write_grub_default(self, value):
        """Write mocked /etc/default/grub content."""
        with open(self.grub_default, "w", encoding="utf-8") as f:
            f.write(value)

    def read_metadata(self, transaction_id):
        """Read mocked transaction metadata."""
        with open(
            os.path.join(self.transaction_root, transaction_id, "metadata.json"),
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def make_grub_backend(self):
        """Create a GRUB backend using temporary files."""
        return GrubDefaultBackend(
            grub_default=self.grub_default,
            grub_default_dir=self.grub_default_dir,
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
        )

    def test_validate_kernel_parameter(self):
        """Kernel parameters must be a single safe token."""
        self.assertEqual(validate_kernel_parameter("foo=bar"), (True, ""))
        ok, reason = validate_kernel_parameter("bad\nparam")
        self.assertFalse(ok)
        self.assertIn("invalid characters", reason)
        ok, reason = validate_kernel_parameter("two tokens")
        self.assertFalse(ok)
        self.assertIn("single command line token", reason)

    def test_safe_transaction_id(self):
        """Transaction ids are normalized before being used in paths."""
        self.assertEqual(_safe_transaction_id("test_gicv4_irq_scaling"), "test_gicv4_irq_scaling")
        self.assertRegex(
            _safe_transaction_id("test case/../../bad"),
            r"^test_case_.._.._bad_[0-9a-f]{12}$",
        )
        self.assertRegex(_safe_transaction_id("###"), r"^default_[0-9a-f]{12}$")
        self.assertNotEqual(_safe_transaction_id("irqchip:test_a"), _safe_transaction_id("irqchip_test_a"))

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_append_and_remove(self, mock_which, mock_run):
        """GRUB backend writes a GLT-owned drop-in and removes it by tag."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.side_effect = lambda cmd: "/usr/sbin/update-grub" if cmd == "update-grub" else None
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")

        backend = self.make_grub_backend()
        result = backend.append_parameter("foo=bar", transaction_id="test case")

        self.assertTrue(result.ok)
        self.assertTrue(result.changed)
        self.assertTrue(result.reboot_required)
        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), 'GRUB_CMDLINE_LINUX="quiet"\n')
        transaction_dir = _safe_transaction_id("test case")
        dropin_path = os.path.join(
            self.grub_default_dir,
            f"99_greenlit_{transaction_dir}.cfg",
        )
        with open(dropin_path, "r", encoding="utf-8") as f:
            self.assertIn('GRUB_CMDLINE_LINUX="${GRUB_CMDLINE_LINUX} foo=bar"', f.read())
        self.assertTrue(os.path.exists(os.path.join(
            self.transaction_root, transaction_dir, "metadata.json"
        )))
        mock_run.assert_called_with(
            ["update-grub"], shell=False, check=False,
            stdout=self.mock.ANY, stderr=self.mock.ANY, text=True,
            timeout=GRUB_REFRESH_TIMEOUT_SECONDS,
        )

        result = backend.remove_parameter("foo=bar", transaction_id="test case")

        self.assertTrue(result.ok)
        self.assertTrue(result.changed)
        self.assertFalse(os.path.exists(dropin_path))
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, transaction_dir)))

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_preserves_unquoted_tokens(self, mock_which, mock_run):
        """GRUB parser keeps all unquoted command-line tokens."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default("GRUB_CMDLINE_LINUX=quiet splash\n")
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")

        result = self.make_grub_backend().append_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "GRUB_CMDLINE_LINUX=quiet splash\n")
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")
        with open(dropin_path, "r", encoding="utf-8") as f:
            self.assertIn("foo=bar", f.read())

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_creates_missing_cmdline_variable(self, mock_which, mock_run):
        """GRUB backend uses a drop-in even if the base setting is absent."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default("GRUB_TIMEOUT=5\n")
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")

        result = self.make_grub_backend().append_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "GRUB_TIMEOUT=5\n")
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")
        with open(dropin_path, "r", encoding="utf-8") as f:
            self.assertIn("foo=bar", f.read())

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_multiple_parameters_share_tag(self, mock_which, mock_run):
        """GRUB backend can remove one GLT-owned parameter while keeping another."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()

        self.assertTrue(backend.append_parameter("foo=bar", transaction_id="test").ok)
        self.assertTrue(backend.append_parameter("baz=qux", transaction_id="test").ok)
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")
        with open(dropin_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("foo=bar", content)
        self.assertIn("baz=qux", content)

        result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        with open(dropin_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("foo=bar", content)
        self.assertIn("baz=qux", content)
        self.assertEqual(self.read_metadata("test")["parameters"], ["baz=qux"])

        result = backend.remove_parameter("baz=qux", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertFalse(os.path.exists(dropin_path))
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_non_dropin_grub_uses_loader_block(self, mock_which, mock_run):
        """GRUB backend adds a managed loader when grub.d is not sourced."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.side_effect = lambda cmd: "/usr/sbin/grub2-mkconfig" if cmd == "grub2-mkconfig" else None
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()

        with self.mock.patch.object(backend, "_grub_supports_default_dropins", return_value=False):
            result = backend.append_parameter("foo=bar", transaction_id="test")
            self.assertTrue(result.ok)
            with open(self.grub_default, "r", encoding="utf-8") as f:
                self.assertIn("BEGIN GreenLiT kernel cmdline drop-ins", f.read())

            result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertNotIn("BEGIN GreenLiT kernel cmdline drop-ins", f.read())

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_non_dropin_loader_waits_for_last_record(self, mock_which, mock_run):
        """GRUB fallback loader is removed only after all GLT drop-ins are gone."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.side_effect = lambda cmd: "/usr/sbin/grub2-mkconfig" if cmd == "grub2-mkconfig" else None
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()

        with self.mock.patch.object(backend, "_grub_supports_default_dropins", return_value=False):
            self.assertTrue(backend.append_parameter("foo=bar", transaction_id="test-one").ok)
            self.assertTrue(backend.append_parameter("baz=qux", transaction_id="test-two").ok)

            self.assertTrue(backend.remove_parameter("foo=bar", transaction_id="test-one").ok)
            with open(self.grub_default, "r", encoding="utf-8") as f:
                self.assertIn("BEGIN GreenLiT kernel cmdline drop-ins", f.read())

            self.assertTrue(backend.remove_parameter("baz=qux", transaction_id="test-two").ok)

        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertNotIn("BEGIN GreenLiT kernel cmdline drop-ins", f.read())

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_rolls_back_dropin_failure(self, mock_which, mock_run):
        """GRUB fallback removes the loader if drop-in creation fails."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.side_effect = lambda cmd: "/usr/sbin/grub2-mkconfig" if cmd == "grub2-mkconfig" else None
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()

        with self.mock.patch.object(backend, "_grub_supports_default_dropins", return_value=False), \
                self.mock.patch.object(backend, "_write_dropin", return_value="drop-in failed"):
            result = backend.append_parameter("foo=bar", transaction_id="test")

        self.assertFalse(result.ok)
        self.assertIn("drop-in failed", result.reason)
        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertNotIn("BEGIN GreenLiT kernel cmdline drop-ins", f.read())
        self.assertFalse(os.path.exists(os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")))

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_rolls_back_metadata_failure(self, mock_which, mock_run):
        """GRUB fallback removes unowned config if metadata cannot be written."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.side_effect = lambda cmd: "/usr/sbin/grub2-mkconfig" if cmd == "grub2-mkconfig" else None
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")

        with self.mock.patch.object(backend, "_grub_supports_default_dropins", return_value=False), \
                self.mock.patch.object(backend, "_write_metadata", side_effect=OSError("metadata failed")):
            result = backend.append_parameter("foo=bar", transaction_id="test")

        self.assertFalse(result.ok)
        self.assertIn("metadata failed", result.reason)
        self.assertFalse(os.path.exists(dropin_path))
        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertNotIn("BEGIN GreenLiT kernel cmdline drop-ins", f.read())

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_rolls_back_refresh_failure_on_append(self, mock_which, mock_run):
        """GRUB append removes new source and metadata if config refresh fails."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.side_effect = lambda cmd: "/usr/sbin/grub2-mkconfig" if cmd == "grub2-mkconfig" else None
        mock_run.return_value = self.mock.Mock(returncode=1, stdout="", stderr="mkconfig failed")
        backend = self.make_grub_backend()
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")

        with self.mock.patch.object(backend, "_grub_supports_default_dropins", return_value=False):
            result = backend.append_parameter("foo=bar", transaction_id="test")

        self.assertFalse(result.ok)
        self.assertIn("mkconfig failed", result.reason)
        self.assertFalse(result.changed)
        self.assertFalse(os.path.exists(dropin_path))
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))
        with open(self.grub_default, "r", encoding="utf-8") as f:
            self.assertNotIn("BEGIN GreenLiT kernel cmdline drop-ins", f.read())

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_rolls_back_refresh_failure_on_multi_append(self, mock_which, mock_run):
        """GRUB append restores prior ownership data if config refresh fails."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()
        self.assertTrue(backend.append_parameter("foo=bar", transaction_id="test").ok)
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")

        mock_run.return_value = self.mock.Mock(returncode=1, stdout="", stderr="mkconfig failed")
        result = backend.append_parameter("baz=qux", transaction_id="test")

        self.assertFalse(result.ok)
        self.assertIn("mkconfig failed", result.reason)
        self.assertFalse(result.changed)
        with open(dropin_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("foo=bar", content)
        self.assertNotIn("baz=qux", content)
        self.assertEqual(self.read_metadata("test")["parameters"], ["foo=bar"])

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_rolls_back_refresh_failure_on_remove(self, mock_which, mock_run):
        """GRUB remove keeps ownership data retryable if config refresh fails."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()
        self.assertTrue(backend.append_parameter("foo=bar", transaction_id="test").ok)
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")

        mock_run.return_value = self.mock.Mock(returncode=1, stdout="", stderr="mkconfig failed")
        result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertFalse(result.ok)
        self.assertIn("mkconfig failed", result.reason)
        self.assertFalse(result.changed)
        with open(dropin_path, "r", encoding="utf-8") as f:
            self.assertIn("foo=bar", f.read())
        self.assertEqual(self.read_metadata("test")["parameters"], ["foo=bar"])

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_rolls_back_refresh_failure_on_multi_remove(self, mock_which, mock_run):
        """GRUB remove restores prior parameter lists when refresh fails."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = self.make_grub_backend()
        self.assertTrue(backend.append_parameter("foo=bar", transaction_id="test").ok)
        self.assertTrue(backend.append_parameter("baz=qux", transaction_id="test").ok)
        dropin_path = os.path.join(self.grub_default_dir, "99_greenlit_test.cfg")

        mock_run.return_value = self.mock.Mock(returncode=1, stdout="", stderr="mkconfig failed")
        result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertFalse(result.ok)
        with open(dropin_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("foo=bar", content)
        self.assertIn("baz=qux", content)
        self.assertEqual(self.read_metadata("test")["parameters"], ["foo=bar", "baz=qux"])

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_existing_config_still_requires_reboot(self, mock_which, mock_run):
        """If config already has the parameter, no restore metadata is written."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet foo=bar"\n')
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")

        result = self.make_grub_backend().append_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertFalse(result.changed)
        self.assertTrue(result.configured)
        self.assertFalse(result.added_by_glt)
        self.assertTrue(result.reboot_required)
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))
        mock_run.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_existing_default_config_is_detected(self, mock_which, mock_run):
        """GRUB check also detects parameters in GRUB_CMDLINE_LINUX_DEFAULT."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        self.write_grub_default('GRUB_CMDLINE_LINUX_DEFAULT="quiet foo=bar"\n')
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")

        result = self.make_grub_backend().append_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertFalse(result.changed)
        self.assertTrue(result.configured)
        self.assertFalse(result.added_by_glt)
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))
        mock_run.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grub_default_check_active_parameter(self, mock_run):
        """Check reports active parameters without modifying persistent config."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz foo=bar")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')

        result = self.make_grub_backend().check_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertTrue(result.active)
        self.assertFalse(result.configured)
        self.assertFalse(result.added_by_glt)
        self.assertTrue(result.reboot_required)
        mock_run.assert_not_called()

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_missing_dropin_remove_is_noop(self, mock_which, mock_run):
        """A missing GreenLiT drop-in is already removed and clears metadata."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        mock_which.return_value = "/usr/sbin/update-grub"
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")

        backend = self.make_grub_backend()
        backend._write_metadata(  # pylint: disable=protected-access
            "test",
            KernelCmdlineRecord(
                backend=backend.name,
                parameters=["foo=bar"],
                config_path=os.path.join(self.grub_default_dir, "99_greenlit_test.cfg"),
            ),
        )

        result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertTrue(result.changed)
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))

    def test_metadata_from_other_backend_is_ignored(self):
        """Shared metadata helpers ignore records owned by another backend."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        backend = self.make_grub_backend()
        backend._write_metadata(  # pylint: disable=protected-access
            "test",
            KernelCmdlineRecord(
                backend="grubby",
                parameters=["foo=bar"],
                target="DEFAULT",
            ),
        )

        result = backend.check_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertFalse(result.added_by_glt)

    def test_grub_default_rejects_unexpected_metadata_config_path(self):
        """GRUB metadata cannot redirect GLT operations to another file."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        backend = self.make_grub_backend()
        backend._write_metadata(  # pylint: disable=protected-access
            "test",
            KernelCmdlineRecord(
                backend=backend.name,
                parameters=["foo=bar"],
                config_path=os.path.join(self.tempdir.name, "not_greenlit.cfg"),
            ),
        )

        append_result = backend.append_parameter("baz=qux", transaction_id="test")
        remove_result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertFalse(append_result.ok)
        self.assertIn("unexpected GRUB drop-in", append_result.reason)
        self.assertFalse(remove_result.ok)
        self.assertIn("unexpected GRUB drop-in", remove_result.reason)

    def test_malformed_metadata_is_ignored(self):
        """Malformed but parseable metadata is treated as absent."""
        backend = self.make_grub_backend()
        metadata_dir = os.path.join(self.transaction_root, "test")
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "metadata.json")
        for content in (
            "[]",
            '{"backend": "grub-default"}',
            '{"backend": "grub-default", "parameters": "foo=bar"}',
            '{"backend": "grub-default", "parameters": ["foo=bar"], "config_path": []}',
            '{"backend": "grub-default", "parameters": ["foo=bar"], "target": []}',
        ):
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.assertIsNone(backend._read_metadata("test"))  # pylint: disable=protected-access

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.shutil.which")
    def test_grub_default_ignores_unsourced_dropins(self, mock_which):
        """Unsourced grub.d snippets are not treated as configured."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz")
        self.write_grub_default('GRUB_CMDLINE_LINUX="quiet"\n')
        os.makedirs(self.grub_default_dir, exist_ok=True)
        with open(
            os.path.join(self.grub_default_dir, "99_greenlit_test.cfg"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write('GRUB_CMDLINE_LINUX="${GRUB_CMDLINE_LINUX} foo=bar"\n')
        mock_which.return_value = None

        result = self.make_grub_backend().check_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertFalse(result.configured)

    def test_manager_wrapper_uses_backend(self):
        """Manager facade delegates operations to its backend."""
        backend = self.mock.MagicMock()
        backend.append_parameter.return_value = KernelParameterResult(True, changed=True)
        manager = KernelCmdlineManager(backend)

        result = manager.append_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertTrue(result.changed)
        backend.append_parameter.assert_called_once_with("foo=bar", transaction_id="test")

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grubby_append_and_remove(self, mock_run):
        """grubby backend uses --args and --remove-args for GLT-owned records."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        configured_args = set()

        def fake_run(command, **_kwargs):
            if "--info=DEFAULT" in command:
                args = " ".join(["quiet", *sorted(configured_args)])
                return self.mock.Mock(returncode=0, stdout=f'args="{args}"\n', stderr="")
            if "--update-kernel=DEFAULT" in command and "--args=foo=bar" in command:
                configured_args.add("foo=bar")
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            if "--update-kernel=DEFAULT" in command and "--remove-args=foo=bar" in command:
                configured_args.discard("foo=bar")
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            return self.mock.Mock(returncode=1, stdout="", stderr="unexpected")

        mock_run.side_effect = fake_run
        backend = GrubbyBackend(
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
        )

        result = backend.append_parameter("foo=bar", transaction_id="test")
        self.assertTrue(result.ok)
        self.assertTrue(result.changed)
        self.assertTrue(result.reboot_required)
        self.assertIn("--update-kernel=DEFAULT", mock_run.call_args_list[1].args[0])

        result = backend.remove_parameter("foo=bar", transaction_id="test")
        self.assertTrue(result.ok)
        self.assertTrue(result.changed)
        self.assertFalse(result.reboot_required)
        self.assertTrue(any(
            "--update-kernel=DEFAULT" in call.args[0] and
            "--remove-args=foo=bar" in call.args[0]
            for call in mock_run.call_args_list
        ))
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grubby_multiple_parameters_share_tag(self, mock_run):
        """grubby metadata can remove one GLT-owned parameter and keep another."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        configured_args = set()

        def fake_run(command, **_kwargs):
            if "--info=DEFAULT" in command:
                args = " ".join(["quiet", *sorted(configured_args)])
                return self.mock.Mock(returncode=0, stdout=f'args="{args}"\n', stderr="")
            if "--update-kernel=DEFAULT" in command and "--args=foo=bar" in command:
                configured_args.add("foo=bar")
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            if "--update-kernel=DEFAULT" in command and "--args=baz=qux" in command:
                configured_args.add("baz=qux")
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            if "--update-kernel=DEFAULT" in command and "--remove-args=foo=bar" in command:
                configured_args.discard("foo=bar")
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            if "--update-kernel=DEFAULT" in command and "--remove-args=baz=qux" in command:
                configured_args.discard("baz=qux")
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            return self.mock.Mock(returncode=1, stdout="", stderr="unexpected")

        mock_run.side_effect = fake_run
        backend = GrubbyBackend(
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
        )

        self.assertTrue(backend.append_parameter("foo=bar", transaction_id="test").ok)
        self.assertTrue(backend.append_parameter("baz=qux", transaction_id="test").ok)
        self.assertEqual(self.read_metadata("test")["parameters"], ["foo=bar", "baz=qux"])

        result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertEqual(self.read_metadata("test")["parameters"], ["baz=qux"])
        remove_command = next(
            call.args[0] for call in mock_run.call_args_list
            if "--remove-args=foo=bar" in call.args[0]
        )
        self.assertIn("--update-kernel=DEFAULT", remove_command)
        self.assertIn("--remove-args=foo=bar", remove_command)
        self.assertNotIn("--remove-args=baz=qux", remove_command)

        result = backend.remove_parameter("baz=qux", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))
        remove_command = next(
            call.args[0] for call in mock_run.call_args_list
            if "--remove-args=baz=qux" in call.args[0]
        )
        self.assertIn("--remove-args=baz=qux", remove_command)

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grubby_rolls_back_metadata_failure(self, mock_run):
        """grubby removes an appended argument if metadata cannot be written."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")

        def fake_run(command, **_kwargs):
            if "--info=DEFAULT" in command:
                return self.mock.Mock(returncode=0, stdout='args="quiet"\n', stderr="")
            if "--update-kernel=DEFAULT" in command:
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            return self.mock.Mock(returncode=1, stdout="", stderr="unexpected")

        mock_run.side_effect = fake_run
        backend = GrubbyBackend(
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
        )

        with self.mock.patch.object(
            backend, "_write_metadata", side_effect=OSError("metadata failed")
        ):
            result = backend.append_parameter("foo=bar", transaction_id="test")

        self.assertFalse(result.ok)
        self.assertIn("metadata failed", result.reason)
        self.assertFalse(result.changed)
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))
        self.assertIn("--args=foo=bar", mock_run.call_args_list[1].args[0])
        self.assertIn("--remove-args=foo=bar", mock_run.call_args_list[2].args[0])

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grubby_rolls_back_remove_metadata_failure(self, mock_run):
        """grubby re-adds a removed argument if metadata update fails."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")

        def fake_run(command, **_kwargs):
            if "--update-kernel=DEFAULT" in command:
                return self.mock.Mock(returncode=0, stdout="", stderr="")
            return self.mock.Mock(returncode=1, stdout="", stderr="unexpected")

        mock_run.side_effect = fake_run
        backend = GrubbyBackend(
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
        )
        backend._write_metadata(  # pylint: disable=protected-access
            "test",
            KernelCmdlineRecord(
                backend=backend.name,
                parameters=["foo=bar"],
                target="DEFAULT",
            ),
        )

        with self.mock.patch.object(
            backend, "_remove_metadata", side_effect=OSError("metadata failed")
        ):
            result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertFalse(result.ok)
        self.assertIn("metadata failed", result.reason)
        self.assertFalse(result.changed)
        self.assertIn("--remove-args=foo=bar", mock_run.call_args_list[0].args[0])
        self.assertIn("--args=foo=bar", mock_run.call_args_list[1].args[0])

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grubby_remove_uses_record_target_without_mutating_backend(self, mock_run):
        """grubby removal uses record target but keeps backend target unchanged."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        mock_run.return_value = self.mock.Mock(returncode=0, stdout="", stderr="")
        backend = GrubbyBackend(
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
            target="DEFAULT",
        )
        backend._write_metadata(  # pylint: disable=protected-access
            "test",
            KernelCmdlineRecord(
                backend=backend.name,
                parameters=["foo=bar"],
                target="ALL",
            ),
        )

        result = backend.remove_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertEqual(backend.target, "DEFAULT")
        self.assertTrue(any(
            call.args[0] == ["grubby", "--update-kernel=ALL", "--remove-args=foo=bar"]
            for call in mock_run.call_args_list
        ))
        self.assertTrue(any(
            call.args[0] == ["grubby", "--info=ALL"]
            for call in mock_run.call_args_list
        ))

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grubby_existing_config_still_requires_reboot(self, mock_run):
        """grubby does not remove preexisting parameters on restore."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        mock_run.return_value = self.mock.Mock(
            returncode=0, stdout='args="quiet foo=bar"\n', stderr=""
        )
        backend = GrubbyBackend(
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
        )

        result = backend.append_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        self.assertFalse(result.changed)
        self.assertTrue(result.reboot_required)
        self.assertFalse(result.added_by_glt)
        self.assertFalse(os.path.exists(os.path.join(self.transaction_root, "test")))

    @GLTTestBase.unittest.mock.patch("src.common.kernel_cmdline.subprocess.run")
    def test_grubby_check_uses_configured_target(self, mock_run):
        """grubby checks the configured kernel target instead of hardcoding DEFAULT."""
        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz quiet")
        mock_run.return_value = self.mock.Mock(returncode=0, stdout='args="quiet"\n', stderr="")
        backend = GrubbyBackend(
            transaction_root=self.transaction_root,
            proc_cmdline=self.proc_cmdline,
            target="ALL",
        )

        result = backend.check_parameter("foo=bar", transaction_id="test")

        self.assertTrue(result.ok)
        mock_run.assert_called_with(
            ["grubby", "--info=ALL"], shell=False, check=False,
            stdout=self.mock.ANY, stderr=self.mock.ANY, text=True,
            timeout=GRUBBY_TIMEOUT_SECONDS,
        )

    def test_detect_prefers_grubby_backend(self):
        """Detection prefers grubby before GRUB defaults and systemd-boot."""
        with self.mock.patch.object(
            GrubbyBackend, "is_available", classmethod(lambda cls: True)
        ), self.mock.patch.object(
            GrubDefaultBackend, "is_available", classmethod(lambda cls: True)
        ), self.mock.patch.object(
            SystemdBootBackend, "is_available", classmethod(lambda cls: True)
        ):
            manager = KernelCmdlineManager.detect()

        self.assertEqual(manager.backend.name, "grubby")

    def test_detect_uses_grub_default_before_systemd_boot(self):
        """Detection uses GRUB default config before the placeholder systemd backend."""
        with self.mock.patch.object(
            GrubbyBackend, "is_available", classmethod(lambda cls: False)
        ), self.mock.patch.object(
            GrubDefaultBackend, "is_available", classmethod(lambda cls: True)
        ), self.mock.patch.object(
            SystemdBootBackend, "is_available", classmethod(lambda cls: True)
        ):
            manager = KernelCmdlineManager.detect()

        self.assertEqual(manager.backend.name, "grub-default")

    def test_detect_unsupported_backend(self):
        """Detection returns a clear unsupported backend when none match."""
        with self.mock.patch.object(
            GrubbyBackend, "is_available", classmethod(lambda cls: False)
        ), self.mock.patch.object(
            GrubDefaultBackend, "is_available", classmethod(lambda cls: False)
        ), self.mock.patch.object(
            SystemdBootBackend, "is_available", classmethod(lambda cls: False)
        ):
            manager = KernelCmdlineManager.detect()
        result = manager.append_parameter("foo=bar", transaction_id="test")

        self.assertIsInstance(manager.backend, UnsupportedBackend)
        self.assertFalse(result.ok)
        self.assertIn("No supported", result.reason)

        remove_result = manager.remove_parameter("foo=bar", transaction_id="test")
        self.assertFalse(remove_result.ok)
        self.assertIn("No supported", remove_result.reason)

        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz foo=bar")
        manager.backend.proc_cmdline = self.proc_cmdline
        check_result = manager.check_parameter("foo=bar", transaction_id="test")
        self.assertTrue(check_result.ok)
        self.assertTrue(check_result.active)
        self.assertFalse(check_result.configured)
        self.assertFalse(check_result.added_by_glt)

    def test_systemd_boot_backend_reports_not_implemented(self):
        """systemd-boot is detectable but intentionally unsupported for mutation."""
        with self.mock.patch.object(
            GrubbyBackend, "is_available", classmethod(lambda cls: False)
        ), self.mock.patch.object(
            GrubDefaultBackend, "is_available", classmethod(lambda cls: False)
        ), self.mock.patch.object(
            SystemdBootBackend, "is_available", classmethod(lambda cls: True)
        ):
            manager = KernelCmdlineManager.detect()
        result = manager.append_parameter("foo=bar", transaction_id="test")

        self.assertIsInstance(manager.backend, SystemdBootBackend)
        self.assertFalse(result.ok)
        self.assertIn("not implemented", result.reason)

        remove_result = manager.remove_parameter("foo=bar", transaction_id="test")
        self.assertFalse(remove_result.ok)
        self.assertIn("not implemented", remove_result.reason)

        self.write_proc_cmdline("BOOT_IMAGE=/boot/vmlinuz foo=bar")
        manager.backend.proc_cmdline = self.proc_cmdline
        check_result = manager.check_parameter("foo=bar", transaction_id="test")
        self.assertTrue(check_result.ok)
        self.assertTrue(check_result.active)
        self.assertFalse(check_result.configured)
        self.assertFalse(check_result.added_by_glt)


if __name__ == "__main__":
    GLTTestBase.unittest.main()
