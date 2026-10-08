"""Unit tests for workqueue CACHE_SHARD validation."""
# pylint: disable=protected-access,too-many-public-methods,wrong-import-position

import errno
import sys
from pathlib import Path

from tests.test_base import GLTTestBase

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from common import glt_core as GLTCore
from src.modules.cpu import glt_test_workqueue_cache_shard as wq_cache_shard


class TestWorkqueueCacheShardHelpers(GLTTestBase):
    """Test deterministic workqueue CACHE_SHARD helpers."""

    def _run_scope_sysfs_round_trip_with_sysfs(self, scope_path, read_sysfs, write_sysfs):
        """Run the round-trip test with mocked sysfs accessors."""
        with self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                    side_effect=read_sysfs), \
                self.mock.patch.object(wq_cache_shard.glt_utils, "write_sysfs_value",
                                       side_effect=write_sysfs), \
                self.mock.patch.object(wq_cache_shard, "_find_affinity_scope_path",
                                       return_value=scope_path):
            return wq_cache_shard.test_workqueue_cache_shard_scope_sysfs_round_trip(
                None, self.mock_logger
            )

    def test_canonical_scope_extracts_default_display(self):
        """Workqueue sysfs reports default scopes as 'default (name)'."""
        self.assertEqual(
            wq_cache_shard._canonical_scope("default (cache_shard)\n"),
            "cache_shard",
        )
        self.assertEqual(wq_cache_shard._canonical_scope("cache\n"), "cache")

    def test_find_affinity_scope_path_skips_unreadable_entries(self):
        """Discovery should keep looking when a visible queue cannot be read."""
        root = "/sys/devices/virtual/workqueue"
        unreadable_path = f"{root}/writeback/affinity_scope"
        readable_path = f"{root}/events/affinity_scope"
        dirs = {root, f"{root}/writeback", f"{root}/events"}
        files = {unreadable_path, readable_path}

        def fake_read(path):
            if path == unreadable_path:
                raise OSError("permission denied")
            return "cache_shard"

        with self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read), \
                self.mock.patch.object(wq_cache_shard.os, "listdir",
                                       return_value=["writeback", "events"]), \
                self.mock.patch.object(wq_cache_shard.os.path, "isdir",
                                       side_effect=dirs.__contains__), \
                self.mock.patch.object(wq_cache_shard.os.path, "isfile",
                                       side_effect=files.__contains__):
            path = wq_cache_shard._find_affinity_scope_path(workqueue_roots=(root,))

        self.assertEqual(path, readable_path)

    def test_has_workqueue_cmdline_override_detects_parameter(self):
        """Boot-time overrides explain non-default observed sysfs values."""
        with self.mock.patch.object(
                wq_cache_shard.glt_utils,
                "read_sysfs_value",
                return_value="BOOT_IMAGE=/vmlinuz workqueue.default_affinity_scope=cache",
        ) as read_mock:
            self.assertTrue(
                wq_cache_shard._has_workqueue_cmdline_override(
                    "default_affinity_scope"
                )
            )
            read_mock.assert_called_once_with(wq_cache_shard.PROC_CMDLINE)

        with self.mock.patch.object(
                wq_cache_shard.glt_utils,
                "read_sysfs_value",
                return_value="BOOT_IMAGE=/vmlinuz quiet",
        ):
            self.assertFalse(
                wq_cache_shard._has_workqueue_cmdline_override("cache_shard_size")
            )

    def test_read_cache_shard_size_accepts_positive_integer(self):
        """The cache_shard_size sysfs parameter should parse as an integer."""
        with self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                    return_value="8"):
            shard_size, reason = wq_cache_shard._read_cache_shard_size()

        self.assertEqual(shard_size, 8)
        self.assertEqual(reason, "")

    def test_read_cache_shard_size_rejects_non_positive_value(self):
        """The kernel should clamp zero to a positive cache_shard_size."""
        with self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                    return_value="0"):
            shard_size, reason = wq_cache_shard._read_cache_shard_size()

        self.assertIsNone(shard_size)
        self.assertIn("not positive", reason)

    def test_scope_restore_token_preserves_default_scope(self):
        """A displayed default scope must be restored by writing 'default'."""
        self.assertEqual(
            wq_cache_shard._scope_restore_token("default (cache_shard)"),
            "default",
        )
        self.assertEqual(wq_cache_shard._scope_restore_token("cache"), "cache")

    def test_scope_sysfs_round_trip_passes_and_restores_default(self):
        """A healthy kernel accepts all scopes, rejects prefixes, and restores."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"
        files = {scope_path: "default (cache_shard)"}
        writes = []

        def fake_write(path, value):
            writes.append((path, value))
            if value in wq_cache_shard.WORKQUEUE_AFFINITY_SCOPES:
                files[path] = value
            elif value == wq_cache_shard.INVALID_CACHE_SHARD_PREFIX:
                raise OSError(errno.EINVAL, "invalid scope")
            elif value == "default":
                files[path] = "default (cache_shard)"
            else:
                files[path] = value

        self._run_scope_sysfs_round_trip_with_sysfs(
            scope_path, files.__getitem__, fake_write
        )

        self.assertEqual(
            writes,
            [(scope_path, scope) for scope in wq_cache_shard.WORKQUEUE_AFFINITY_SCOPES] +
            [
                (scope_path, wq_cache_shard.INVALID_CACHE_SHARD_PREFIX),
                (scope_path, "default"),
            ],
        )
        self.assertEqual(files[scope_path], "default (cache_shard)")

    def test_scope_sysfs_round_trip_restores_after_readback_error(self):
        """A write followed by readback failure must still restore scope."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"
        files = {scope_path: "default (cache_shard)"}
        writes = []

        def fake_read(path):
            if len(writes) == 1:
                raise OSError("readback failed")
            return files[path]

        def fake_write(path, value):
            writes.append((path, value))
            if value == "default":
                files[path] = "default (cache_shard)"
            else:
                files[path] = value

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_scope_sysfs_round_trip_with_sysfs(
                scope_path, fake_read, fake_write
            )

        self.assertIn("readback failed", str(context.exception))
        self.assertEqual(
            writes,
            [(scope_path, wq_cache_shard.WORKQUEUE_AFFINITY_SCOPES[0]),
             (scope_path, "default")],
        )
        self.assertEqual(files[scope_path], "default (cache_shard)")

    def test_scope_sysfs_round_trip_detects_cache_shard_prefix_bug(self):
        """The pre-fix parser maps cache_shard to cache by prefix matching."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"
        files = {scope_path: "default (cache)"}

        def fake_write(path, value):
            if value == wq_cache_shard.CACHE_SHARD_SCOPE:
                files[path] = "cache"
            elif value == "default":
                files[path] = "default (cache)"
            else:
                files[path] = value

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_scope_sysfs_round_trip_with_sysfs(
                scope_path, files.__getitem__, fake_write
            )

        self.assertIn("exact affinity-scope parsing", str(context.exception))
        self.assertEqual(files[scope_path], "default (cache)")

    def test_scope_sysfs_round_trip_fails_when_invalid_prefix_is_accepted(self):
        """Exact sysfs parsing must reject longer strings beginning with a valid scope."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"
        files = {scope_path: "cache"}

        def fake_write(path, value):
            if value == wq_cache_shard.INVALID_CACHE_SHARD_PREFIX:
                files[path] = "cache"
            else:
                files[path] = value

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_scope_sysfs_round_trip_with_sysfs(
                scope_path, files.__getitem__, fake_write
            )

        self.assertIn("Invalid affinity scope", str(context.exception))
        self.assertEqual(files[scope_path], "cache")

    def test_scope_sysfs_round_trip_fails_when_invalid_prefix_error_not_einval(self):
        """An invalid-scope write only proves exact parsing when it fails with EINVAL."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"
        files = {scope_path: "default (cache_shard)"}

        def fake_write(path, value):
            if value == wq_cache_shard.INVALID_CACHE_SHARD_PREFIX:
                raise PermissionError(errno.EACCES, "permission denied")
            if value == "default":
                files[path] = "default (cache_shard)"
            else:
                files[path] = value

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_scope_sysfs_round_trip_with_sysfs(
                scope_path, files.__getitem__, fake_write
            )

        self.assertIn("expected errno.EINVAL", str(context.exception))
        self.assertIn("permission denied", str(context.exception))
        self.assertEqual(files[scope_path], "default (cache_shard)")

    def test_scope_sysfs_round_trip_fails_without_write_permission(self):
        """GreenLiT root runs should fail unexpected sysfs write permission errors."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"

        def fake_write(_path, _value):
            raise PermissionError(errno.EACCES, "permission denied")

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_scope_sysfs_round_trip_with_sysfs(
                scope_path, lambda _path: "cache", fake_write
            )

        self.assertIn("permission denied", str(context.exception))

    def test_scope_sysfs_round_trip_reports_restore_failure(self):
        """A successful mutation must be restored or reported as a failure."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"
        files = {scope_path: "default (cache_shard)"}

        def fake_write(path, value):
            if value == wq_cache_shard.INVALID_CACHE_SHARD_PREFIX:
                raise OSError(errno.EINVAL, "invalid scope")
            if value == "default":
                raise OSError("restore failed")
            files[path] = value

        with self.assertRaises(GLTCore.GLTTestFail) as context:
            self._run_scope_sysfs_round_trip_with_sysfs(
                scope_path, files.__getitem__, fake_write
            )

        self.assertIn("restore", str(context.exception))


class TestWorkqueueCacheShardGreenlitTests(GLTTestBase):
    """Test GreenLiT-facing result behavior with mocks."""

    def test_default_scope_fails_when_default_is_cache(self):
        """The default-scope test points to the default affinity patch."""
        with self.mock.patch.object(wq_cache_shard, "_read_cache_shard_size",
                                    return_value=(8, "")), \
                self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                       return_value="cache"), \
                self.mock.patch.object(wq_cache_shard, "_has_workqueue_cmdline_override",
                                       return_value=False):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                wq_cache_shard.test_workqueue_cache_shard_default_scope(
                    None, self.mock_logger
                )

        self.assertIn("cache-shard default affinity", str(context.exception))
        self.assertIn("current effective", str(context.exception))
        self.assertIn("compiled-in default", str(context.exception))

    def test_default_scope_skips_when_default_scope_is_overridden(self):
        """A boot-time default_affinity_scope override is not a missing-patch signal."""
        with self.mock.patch.object(wq_cache_shard, "_read_cache_shard_size",
                                    return_value=(8, "")), \
                self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                       return_value="cache"), \
                self.mock.patch.object(wq_cache_shard, "_has_workqueue_cmdline_override",
                                       return_value=True):
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                wq_cache_shard.test_workqueue_cache_shard_default_scope(
                    None, self.mock_logger
                )

        self.assertIn("default_affinity_scope", str(context.exception))
        self.assertIn("current effective", str(context.exception))

    def test_default_scope_uses_workqueue_sysfs_default_display(self):
        """A queue display of 'default (cache_shard)' proves the default scope."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"

        def fake_read(path):
            if path == wq_cache_shard.DEFAULT_AFFINITY_SCOPE_PARAM:
                raise OSError("missing")
            return {scope_path: "default (cache_shard)"}[path]

        with self.mock.patch.object(wq_cache_shard, "_read_cache_shard_size",
                                    return_value=(8, "")), \
                self.mock.patch.object(wq_cache_shard, "_find_affinity_scope_path",
                                       return_value=scope_path), \
                self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                       side_effect=fake_read):
            wq_cache_shard.test_workqueue_cache_shard_default_scope(
                None, self.mock_logger
            )

    def test_default_scope_skips_when_fallback_scope_is_explicit(self):
        """A bare queue scope may be explicit, so it does not prove the default."""
        scope_path = "/sys/devices/virtual/workqueue/writeback/affinity_scope"

        def fake_read(path):
            if path == wq_cache_shard.DEFAULT_AFFINITY_SCOPE_PARAM:
                raise OSError("missing")
            return {scope_path: "cache_shard"}[path]

        with self.mock.patch.object(wq_cache_shard, "_read_cache_shard_size",
                                    return_value=(8, "")), \
                self.mock.patch.object(wq_cache_shard, "_find_affinity_scope_path",
                                       return_value=scope_path), \
                self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                       side_effect=fake_read):
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                wq_cache_shard.test_workqueue_cache_shard_default_scope(
                    None, self.mock_logger
                )

        self.assertIn("only a 'default (...)' queue display", str(context.exception))

    def test_default_scope_fails_when_feature_param_missing_on_workqueue_kernel(self):
        """A target kernel without cache_shard_size is missing CACHE_SHARD support."""
        with self.mock.patch.object(wq_cache_shard, "_read_cache_shard_size",
                                    return_value=(None, "missing")), \
                self.mock.patch.object(wq_cache_shard, "_module_loaded",
                                       return_value=True):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                wq_cache_shard.test_workqueue_cache_shard_default_scope(
                    None, self.mock_logger
                )

        self.assertIn("WQ_AFFN_CACHE_SHARD", str(context.exception))

    def test_default_scope_skips_when_workqueue_sysfs_unavailable(self):
        """A non-target kernel without workqueue sysfs is not a missing-backport signal."""
        with self.mock.patch.object(wq_cache_shard, "_read_cache_shard_size",
                                    return_value=(None, "missing")), \
                self.mock.patch.object(wq_cache_shard, "_module_loaded",
                                       return_value=False):
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                wq_cache_shard.test_workqueue_cache_shard_default_scope(
                    None, self.mock_logger
                )

        self.assertIn("/sys/module/workqueue", str(context.exception))

    def test_default_scope_tolerates_cache_shard_size_override(self):
        """A custom shard-size boot parameter should not fail the default-scope test."""
        with self.mock.patch.object(wq_cache_shard, "_read_cache_shard_size",
                                    return_value=(16, "")), \
                self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                       return_value="cache_shard"), \
                self.mock.patch.object(wq_cache_shard, "_has_workqueue_cmdline_override",
                                       return_value=True):
            wq_cache_shard.test_workqueue_cache_shard_default_scope(
                None, self.mock_logger
            )

    def test_scope_sysfs_round_trip_fails_when_scope_cannot_be_read(self):
        """The write test should fail when the selected affinity_scope is unreadable."""
        with self.mock.patch.object(wq_cache_shard, "_find_affinity_scope_path",
                                    return_value="/sys/wq/affinity_scope"), \
                self.mock.patch.object(wq_cache_shard.glt_utils, "read_sysfs_value",
                                       side_effect=OSError("read failed")):
            with self.assertRaises(GLTCore.GLTTestFail) as context:
                wq_cache_shard.test_workqueue_cache_shard_scope_sysfs_round_trip(
                    None, self.mock_logger
                )

        self.assertIn("read failed", str(context.exception))

    def test_scope_sysfs_round_trip_requires_default_scope_test(self):
        """The scheduler should run feature exposure checks before the write test."""
        requires = (
            wq_cache_shard.test_workqueue_cache_shard_scope_sysfs_round_trip
            .__wrapped__.requires
        )

        self.assertIn(
            "cpu:test_workqueue_cache_shard_default_scope",
            requires.all_true,
        )

    def test_scope_sysfs_round_trip_skips_without_wq_sysfs(self):
        """The write test should skip when CONFIG_WQ_SYSFS is unavailable."""
        with self.mock.patch.object(wq_cache_shard, "_find_affinity_scope_path",
                                    return_value=None):
            with self.assertRaises(GLTCore.GLTTestSkip) as context:
                wq_cache_shard.test_workqueue_cache_shard_scope_sysfs_round_trip(
                    None, self.mock_logger
                )

        self.assertIn("CONFIG_WQ_SYSFS", str(context.exception))
