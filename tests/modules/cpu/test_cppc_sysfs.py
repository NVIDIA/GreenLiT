"""Unit tests for CPPC cpufreq sysfs coverage."""
# pylint: disable=too-many-lines

import errno

from tests.test_base import GLTTestBase
from src.modules.cpu import glt_test_cpufreq_stress as cpufreq


class TestCppcBoostQos(GLTTestBase):  # pylint: disable=too-many-public-methods
    """Test CPPC boost-frequency QoS validation."""

    POLICY_DIR = "/sys/devices/system/cpu/cpu0/cpufreq"
    CPPC_CAPS_DIR = "/sys/devices/system/cpu/cpu0/acpi_cppc"
    POLICY_BOOST = "/sys/devices/system/cpu/cpufreq/policy0/boost"
    PEER_DIR = "/sys/devices/system/cpu/cpufreq/policy1"
    GLOBAL_BOOST = "/sys/devices/system/cpu/cpufreq/boost"
    OSPM_NOMINAL = f"{POLICY_DIR}/ospm_nominal_freq"

    def setUp(self):
        super().setUp()
        self.peer_user_requests = {}
        self.qos_elapsed = 0
        self.qos_updates = []
        self.qos_write_times = []

    def _invoke_boost_qos(self):
        """Run the decorated testcase and verify native result attribution."""
        testcase = cpufreq.test_cppc_boost_frequency_qos
        with self.mock.patch.object(
                testcase.attr, "result", (cpufreq.GLTCore.GLTTestResult.NA, "")):
            try:
                self.assertIsNone(testcase(None, self.mock_logger))
            except cpufreq.GLTCore.GLTTestSkip as error:
                verdict, reason = cpufreq.GLTCore.GLTTestResult.SKIP, str(error)
            except cpufreq.GLTCore.GLTTestFail as error:
                verdict, reason = cpufreq.GLTCore.GLTTestResult.FAIL, str(error)
            else:
                # A normal return is PASS; the runner records it afterwards.
                self.assertEqual(testcase.attr.result, (cpufreq.GLTCore.GLTTestResult.NA, ""))
                return cpufreq.GLTCore.GLTTestResult.PASS, None
            self.assertEqual(testcase.attr.result, (verdict, reason))
            return verdict, reason

    def _observe_result_exception(self, exception_type, events):
        """Record result timing without replacing native result attribution."""
        testcase = self

        class ObservedResult(exception_type):
            """Keep the real exception semantics while recording issued writes."""

            def __init__(self, reason):
                events.append((reason, len(testcase.qos_write_times)))
                super().__init__(reason)

        return ObservedResult

    def _run_boost_qos(  # pylint: disable=too-many-arguments
            self, *, overwrite_on_disable=False, overwrite_on_enable=False,
            boosted_max=1200000, fail_restore=False, original_boost=1,
            policy_boost=None, dynamic_ospm=False, ospm_nominal=1200000,
            ospm_write_errno=None, boost_enable_errno=None,
            peer_policy=False, scaling_min=600000, peer_restore_errno=None,
            partial_boost_error=False, ospm_restore_offset=0,
            peer_read_errno=None, nominal_restore_read_errno=None,
            policy_update_delay=0, peer_update_delay=(0, 0),
            peer_user_max=850000, peer_cleanup_read_errno=None,
            policy_intermediate=False, cap_override=None,
            restore_update_delays=None, peer_verify_errno=None,
            policy_boost_files=True, policy_dirs_available=True):
        """Model queued policy refreshes separately from writable QoS requests."""
        original_scaling_max = max(900000, scaling_min)
        if policy_boost is None:
            policy_boost = original_boost
        nominal_max = ospm_nominal if dynamic_ospm else 1000000
        files = {
            self.GLOBAL_BOOST: str(original_boost),
            f"{self.POLICY_DIR}/cpuinfo_max_freq": str(
                boosted_max if original_boost else nominal_max
            ),
            f"{self.POLICY_DIR}/scaling_min_freq": str(scaling_min),
            f"{self.POLICY_DIR}/scaling_max_freq": str(original_scaling_max),
            f"{self.CPPC_CAPS_DIR}/highest_perf": (
                "100" if dynamic_ospm else "120"
            ),
            f"{self.CPPC_CAPS_DIR}/nominal_perf": "100",
        }
        if dynamic_ospm:
            files[self.OSPM_NOMINAL] = str(nominal_max)
        policy_boost_states = {self.POLICY_BOOST: str(policy_boost)}
        peer_dirs = [
            f"/sys/devices/system/cpu/cpufreq/policy{index}"
            for index in range(1, int(peer_policy) + 1)
        ]
        self.peer_user_requests = {}
        for peer_dir in peer_dirs:
            policy_boost_states[f"{peer_dir}/boost"] = str(original_boost)
            peer_max_path = f"{peer_dir}/scaling_max_freq"
            self.peer_user_requests[peer_max_path] = peer_user_max
            files[peer_max_path] = str(min(
                peer_user_max, boosted_max if original_boost else 1000000
            ))
        if policy_boost_files:
            files.update(policy_boost_states)
        policy_root = "/sys/devices/system/cpu/cpufreq"
        glob_results = {
            f"{policy_root}/policy[0-9]*/boost": (
                list(policy_boost_states) if policy_boost_files else []
            ),
            f"{policy_root}/policy[0-9]*": (
                [f"{policy_root}/policy0", *peer_dirs]
                if policy_dirs_available else []
            ),
        }
        writes = []
        ospm_restored = False
        peer_cleanup_reads = 0
        pending_updates = {}
        self.qos_elapsed = 0
        self.qos_updates = []
        self.qos_write_times = []

        def apply_updates():
            for path, (deadline, value) in list(pending_updates.items()):
                if self.qos_elapsed >= deadline:
                    files[path] = value
                    self.qos_updates.append((path, int(value)))
                    del pending_updates[path]

        def queue_update(path, value, delay):
            # Coalesced work observes the latest aggregate at its scheduled time.
            deadline = self.qos_elapsed + delay
            if path in pending_updates:
                deadline = min(deadline, pending_updates[path][0])
            pending_updates[path] = (deadline, str(value))
            apply_updates()

        def resolve_path(path):
            return path.replace(
                "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq",
                f"{self.POLICY_DIR}/scaling_max_freq",
            )

        def fake_read(path):
            nonlocal peer_cleanup_reads
            apply_updates()
            if path == self.OSPM_NOMINAL:
                raise OSError(errno.EACCES, "OSPM nominal is write-only")
            if path == f"{self.PEER_DIR}/scaling_max_freq" and \
                    peer_read_errno is not None:
                raise OSError(peer_read_errno, "peer cap unreadable")
            if path == f"{self.PEER_DIR}/scaling_max_freq" and \
                    peer_cleanup_read_errno is not None and (
                        f"{self.POLICY_DIR}/scaling_max_freq",
                        original_scaling_max) in writes:
                peer_cleanup_reads += 1
                if peer_cleanup_reads > 1:
                    raise OSError(peer_cleanup_read_errno, "peer cleanup unreadable")
            if path == f"{self.PEER_DIR}/scaling_max_freq" and \
                    peer_verify_errno is not None and \
                    any(written_path == path for written_path, _ in writes):
                raise OSError(peer_verify_errno, "peer restore verification unreadable")
            if path == f"{self.POLICY_DIR}/cpuinfo_max_freq" and \
                    ospm_restored and files[self.GLOBAL_BOOST] == "0" and \
                    nominal_restore_read_errno is not None:
                raise OSError(nominal_restore_read_errno, "nominal unreadable")
            return files[resolve_path(path)]

        def fake_sleep(seconds):
            self.qos_elapsed = round(self.qos_elapsed + seconds, 10)
            apply_updates()

        def set_policy_boost(path, value):
            policy_boost_states[path] = str(value)
            if path in files:
                files[path] = str(value)

        def update_peer_boost(value):
            for peer_dir in peer_dirs:
                if policy_boost_states[f"{peer_dir}/boost"] == str(value):
                    continue
                set_policy_boost(f"{peer_dir}/boost", value)
                peer_max_path = f"{peer_dir}/scaling_max_freq"
                peer_boost_max = boosted_max if value else 1000000
                if (value and overwrite_on_enable) or \
                        (not value and overwrite_on_disable):
                    self.peer_user_requests[peer_max_path] = peer_boost_max
                queue_update(peer_max_path, min(
                    self.peer_user_requests[peer_max_path], peer_boost_max
                ), peer_update_delay[value])

        def fake_write(path, value):  # pylint: disable=too-many-branches
            nonlocal ospm_restored
            value = int(value)
            writes.append((path, value))
            self.qos_write_times.append((path, value, self.qos_elapsed))
            if path == self.GLOBAL_BOOST:
                if dynamic_ospm and value and boost_enable_errno is not None:
                    if partial_boost_error:
                        # The global loop rolls back its flag after an error,
                        # but policies updated before that error remain changed.
                        files[path] = "0"
                        set_policy_boost(self.POLICY_BOOST, value)
                    raise OSError(
                        boost_enable_errno, "global boost enable rejected"
                    )
                files[path] = str(value)
                set_policy_boost(self.POLICY_BOOST, value)
                update_peer_boost(value)
                files[f"{self.POLICY_DIR}/cpuinfo_max_freq"] = str(
                    boosted_max if value else (
                        int(files[self.OSPM_NOMINAL])
                        if dynamic_ospm else nominal_max
                    )
                )
                if (value and overwrite_on_enable) or \
                        (not value and overwrite_on_disable):
                    files[f"{self.POLICY_DIR}/scaling_max_freq"] = files[
                        f"{self.POLICY_DIR}/cpuinfo_max_freq"
                    ]
            elif path == self.OSPM_NOMINAL and ospm_write_errno is not None:
                raise OSError(ospm_write_errno, "OSPM nominal write rejected")
            elif fail_restore and value == original_scaling_max:
                raise OSError("restore rejected")
            elif path == f"{self.PEER_DIR}/scaling_max_freq" and \
                    peer_restore_errno is not None:
                raise OSError(peer_restore_errno, "peer restore rejected")
            else:
                if path.endswith("/scaling_max_freq"):
                    if path in self.peer_user_requests:
                        self.peer_user_requests[path] = value
                    resolved_path = resolve_path(path)
                    if policy_intermediate:
                        # An older queued update may become visible before
                        # the work that applies the latest userspace request.
                        files[resolved_path] = str(
                            (int(files[resolved_path]) + value) // 2
                        )
                        self.qos_updates.append((resolved_path, int(files[resolved_path])))
                    if cap_override and path == f"{self.POLICY_DIR}/scaling_max_freq" \
                            and value == cap_override[0]:
                        value = cap_override[1]
                    delay = policy_update_delay
                    if restore_update_delays and (
                            path in self.peer_user_requests or value == original_scaling_max):
                        delay = restore_update_delays.get(path, delay)
                    queue_update(resolved_path, value, delay)
                else:
                    files[resolve_path(path)] = str(value)
                if path == self.OSPM_NOMINAL and value == ospm_nominal:
                    ospm_restored = True
                    files[path] = str(value + ospm_restore_offset)

        with self.mock.patch.object(
                cpufreq, "_validate_cppc_prerequisites",
                return_value=(True, None)) as prerequisite_mock, \
             self.mock.patch.object(
                cpufreq, "_get_first_cppc_cpufreq_dir",
                return_value=(self.POLICY_DIR, None)), \
             self.mock.patch.object(
                 cpufreq.os.path, "isfile", side_effect=lambda path: path in files), \
             self.mock.patch.object(
                 cpufreq.os.path, "realpath", side_effect=resolve_path), \
             self.mock.patch.object(
                 cpufreq.glob, "glob", side_effect=lambda pattern: glob_results[pattern]), \
             self.mock.patch.object(
                 cpufreq.glt_utils, "read_sysfs_value", side_effect=fake_read), \
             self.mock.patch.object(
                 cpufreq.glt_utils, "write_sysfs_value", side_effect=fake_write), \
             self.mock.patch.object(
                 cpufreq.time, "monotonic", side_effect=lambda: self.qos_elapsed), \
             self.mock.patch.object(cpufreq.time, "sleep", side_effect=fake_sleep):
            verdict, reason = self._invoke_boost_qos()
        prerequisite_mock.assert_called_once_with()
        return verdict, reason, files, writes

    def test_boost_qos_accepts_non_vera_cppc_without_ospm(self):
        """Static CPPC boost support needs neither Vera detection nor OSPM."""
        with self.mock.patch.object(cpufreq, "NvDev") as nvdev_mock:
            nvdev_mock.return_value.is_cpu_type.return_value = False
            verdict, reason, files, writes = self._run_boost_qos()

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        nvdev_mock.assert_not_called()
        self.assertNotIn(self.OSPM_NOMINAL, files)
        self.assertFalse(any(path == self.OSPM_NOMINAL for path, _ in writes))
        cap_path = f"{self.POLICY_DIR}/scaling_max_freq"
        self.assertIn((cap_path, 733333), writes)
        self.assertIn((cap_path, 866666), writes)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[cap_path], "900000")

    def test_boost_qos_async_policy_writes_preserve_verdict(self):
        """Delayed user-cap writes keep the full-flow PASS/overwrite distinction."""
        for dynamic_ospm in (False, True):
            for original_boost in (0, 1):
                for overwrite in (None, "enable", "disable"):
                    with self.subTest(ospm=dynamic_ospm, boost=original_boost,
                                      overwrite=overwrite):
                        verdict, reason, files, _ = self._run_boost_qos(
                            dynamic_ospm=dynamic_ospm,
                            original_boost=original_boost,
                            policy_update_delay=0.25,
                            overwrite_on_enable=overwrite == "enable",
                            overwrite_on_disable=overwrite == "disable",
                        )
                        if overwrite is None:
                            self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
                            self.assertIsNone(reason)
                        else:
                            self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                            self.assertIn("overwrote the scaling-max QoS", reason)
                            self.assertNotIn("Failed to restore", reason)
                        self.assertEqual(files[self.GLOBAL_BOOST], str(original_boost))
                        self.assertEqual(
                            files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000"
                        )

    def test_boost_qos_cleanup_failure_overrides_skip_without_patch_hint(self):
        """A cleanup failure must replace SKIP and retain its original reason."""
        verdict, reason, files, writes = self._run_boost_qos(
            scaling_min=999999, fail_restore=True, peer_policy=True,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("no room for two distinct", reason)
        # The CLI summary keeps only the first line; show why SKIP became FAIL.
        self.assertTrue(reason.startswith("Failed to restore CPPC policy state"))
        self.assertIn("\nOriginal skip: ", reason)
        self.assertIn("restore rejected", reason)
        self.assertNotIn(cpufreq.CPPC_BOOST_QOS_PATCH, reason)
        self.assertIn((f"{self.POLICY_DIR}/scaling_max_freq", 999999), writes)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[f"{self.PEER_DIR}/boost"], "1")
        self.assertEqual(files[f"{self.PEER_DIR}/scaling_max_freq"], "850000")

    def test_boost_qos_cleanup_failures_keep_overwrite_reason_and_continue(self):
        """Keep both causes and restore later peers despite multiple errors."""
        for direction in ("enable", "disable"):
            with self.subTest(direction=direction):
                verdict, reason, files, writes = self._run_boost_qos(
                    overwrite_on_enable=direction == "enable",
                    overwrite_on_disable=direction == "disable",
                    fail_restore=True, peer_policy=2, peer_restore_errno=errno.EIO,
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                prefix = "Enabling" if direction == "enable" else "Disabling"
                self.assertTrue(reason.startswith(f"{prefix} CPPC boost overwrote"))
                self.assertIn("Failed to restore CPPC policy state", reason)
                self.assertIn("restore rejected", reason)
                self.assertIn("peer restore rejected", reason)
                self.assertEqual(reason.count("This symptom matches the fix in"), 1)
                self.assertIn("\n".join(cpufreq.CPPC_BOOST_QOS_PATCHES), reason)
                last_peer = "/sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq"
                self.assertIn((last_peer, 850000), writes)
                self.assertEqual(files[last_peer], "850000")
                self.assertEqual(files[self.GLOBAL_BOOST], "1")

    def test_boost_qos_intermediate_caps_preserve_verdict(self):
        """Intermediate readbacks must not become the boost-overwrite baseline."""
        for overwrite in (None, "enable", "disable"):
            with self.subTest(overwrite=overwrite):
                verdict, reason, files, _ = self._run_boost_qos(
                    policy_intermediate=True, policy_update_delay=0.25,
                    overwrite_on_enable=overwrite == "enable",
                    overwrite_on_disable=overwrite == "disable",
                )
                if overwrite is None:
                    self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
                    self.assertIsNone(reason)
                else:
                    self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                    self.assertIn("overwrote the scaling-max QoS", reason)
                    self.assertNotIn("Failed to restore", reason)
                cap_path = f"{self.POLICY_DIR}/scaling_max_freq"
                previous = 1000000 if overwrite == "disable" else 900000
                self.assertIn((cap_path, (previous + 733333) // 2), self.qos_updates)
                self.assertIn((cap_path, 733333), self.qos_updates)
                self.assertEqual(files[cap_path], "900000")

    def test_boost_qos_peer_cleanup_delayed_restores_share_timeout(self):
        """Queued restores share one verification budget regardless of fanout."""
        cap_path = f"{self.POLICY_DIR}/scaling_max_freq"
        timeout = cpufreq.CPPC_QOS_UPDATE_TIMEOUT_SECS
        for delay in (0.25, timeout):
            for peers in (1, 8, 351):
                with self.subTest(delay=delay, peers=peers):
                    peer_paths = [
                        f"/sys/devices/system/cpu/cpufreq/policy{index}/scaling_max_freq"
                        for index in range(1, peers + 1)
                    ]
                    verdict, reason, files, writes = self._run_boost_qos(
                        peer_policy=peers, overwrite_on_enable=True,
                        restore_update_delays={
                            path: delay for path in [cap_path, *peer_paths]
                        },
                    )
                    self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                    self.assertIn("Enabling CPPC boost overwrote", reason)
                    self.assertNotIn("Failed to restore", reason)
                    restore_times = [
                        stamp for path, value, stamp in self.qos_write_times
                        if path in peer_paths or (path == cap_path and value == 900000)
                    ]
                    # Five seconds for passive peer convergence, then at most
                    # one more five-second window for all restoration writes.
                    self.assertLessEqual(self.qos_elapsed - min(restore_times), 2 * timeout)
                    self.assertLessEqual(self.qos_elapsed - max(restore_times), timeout)
                    self.assertAlmostEqual(
                        self.qos_elapsed - min(restore_times),
                        timeout + (0.3 if delay == 0.25 else timeout),
                    )
                    self.assertEqual(files[cap_path], "900000")
                    for path in peer_paths:
                        self.assertEqual(files[path], "850000")
                        self.assertEqual(writes.count((path, 850000)), 1)

    def test_boost_qos_peer_cleanup_timeout_does_not_abandon_later_peers(self):
        """A stuck first restore cannot starve later restores or restart waits."""
        first_path = f"{self.PEER_DIR}/scaling_max_freq"
        second_path = "/sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq"
        verdict, reason, files, writes = self._run_boost_qos(
            peer_policy=2, overwrite_on_enable=True,
            restore_update_delays={first_path: float("inf"), second_path: 0.25},
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn(f"{first_path} restored to 1200000, expected 850000", reason)
        self.assertNotIn(f"{second_path} restored to", reason)
        self.assertEqual(files[second_path], "850000")
        self.assertEqual(writes.count((first_path, 850000)), 1)
        self.assertEqual(writes.count((second_path, 850000)), 1)
        self.assertAlmostEqual(
            self.qos_elapsed,
            3 * cpufreq.CPPC_QOS_UPDATE_SETTLE_SECS
            + 2 * cpufreq.CPPC_QOS_UPDATE_TIMEOUT_SECS,
        )

    def test_boost_qos_peer_cleanup_verification_handles_read_error(self):
        """A post-write read error does not stop verification of later peers."""
        verdict, reason, files, writes = self._run_boost_qos(
            peer_policy=2, overwrite_on_enable=True,
            policy_update_delay=0.25, peer_verify_errno=errno.EIO,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("peer restore verification unreadable", reason)
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        for peer in (1, 2):
            path = f"/sys/devices/system/cpu/cpufreq/policy{peer}/scaling_max_freq"
            self.assertEqual(files[path], "850000")
            self.assertEqual(writes.count((path, 850000)), 1)

    def test_boost_qos_peer_cleanup_wait_handles_read_error(self):
        """An unreadable peer fails cleanup without abandoning subsequent peers."""
        verdict, reason, files, _ = self._run_boost_qos(
            peer_policy=2, overwrite_on_enable=True,
            peer_cleanup_read_errno=errno.EIO,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("Enabling CPPC boost overwrote", reason)
        self.assertIn("peer cleanup unreadable", reason)
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(
            files["/sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq"],
            "850000",
        )

    def test_boost_qos_peer_cleanup_wait_has_shared_timeout(self):
        """Persistent mismatches consume one wait budget, then still restore."""
        self._run_boost_qos(overwrite_on_enable=True)
        baseline_elapsed = self.qos_elapsed
        for peer_count in (1, 8):
            with self.subTest(peers=peer_count):
                verdict, reason, files, writes = self._run_boost_qos(
                    peer_policy=peer_count, overwrite_on_enable=True,
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                self.assertIn("Enabling CPPC boost overwrote", reason)
                self.assertNotIn("Failed to restore", reason)
                self.assertAlmostEqual(
                    self.qos_elapsed - baseline_elapsed,
                    cpufreq.CPPC_QOS_UPDATE_TIMEOUT_SECS,
                )
                for path in self.peer_user_requests:
                    self.assertEqual(files[path], "850000")
                    self.assertEqual(writes.count((path, 850000)), 1)

    def test_boost_qos_unchanged_peers_do_not_delay_cleanup(self):
        """Already matching peer caps do not incur the convergence timeout."""
        self._run_boost_qos()
        baseline_elapsed = self.qos_elapsed
        verdict, reason, _, writes = self._run_boost_qos(peer_policy=8)
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        self.assertEqual(self.qos_elapsed, baseline_elapsed)
        self.assertFalse(any(path in self.peer_user_requests for path, _ in writes))

    def test_boost_qos_waits_for_default_peer_cap_before_rewriting(self):
        """Boost-bound peers settle without replacing their higher user request."""
        peer_max_path = f"{self.PEER_DIR}/scaling_max_freq"
        for original_boost in (0, 1):
            with self.subTest(boost=original_boost):
                delays = [0.25, 0.25]
                delays[original_boost] = 2.5
                verdict, reason, files, writes = self._run_boost_qos(
                    original_boost=original_boost, peer_policy=True,
                    peer_user_max=0x7fffffff, peer_update_delay=delays,
                    policy_update_delay=0.25,
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
                self.assertIsNone(reason)
                self.assertIn((peer_max_path, 1000000), self.qos_updates)
                self.assertIn((peer_max_path, 1200000), self.qos_updates)
                self.assertEqual(
                    files[peer_max_path], "1200000" if original_boost else "1000000"
                )
                self.assertEqual(self.peer_user_requests[peer_max_path], 0x7fffffff)
                self.assertFalse(any(path == peer_max_path for path, _ in writes))

    def test_boost_qos_waits_for_peer_cap_at_deadline(self):
        """A peer that converges on the final poll is not unnecessarily written."""
        peer_max_path = f"{self.PEER_DIR}/scaling_max_freq"
        verdict, reason, files, writes = self._run_boost_qos(
            peer_policy=True, peer_user_max=0x7fffffff,
            peer_update_delay=(0.25, cpufreq.CPPC_QOS_UPDATE_SETTLE_SECS
                               + cpufreq.CPPC_QOS_UPDATE_TIMEOUT_SECS),
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        self.assertEqual(files[peer_max_path], "1200000")
        self.assertEqual(self.peer_user_requests[peer_max_path], 0x7fffffff)
        self.assertFalse(any(path == peer_max_path for path, _ in writes))
        self.assertEqual(
            self.qos_elapsed,
            4 * cpufreq.CPPC_QOS_UPDATE_SETTLE_SECS + cpufreq.CPPC_QOS_UPDATE_TIMEOUT_SECS,
        )

    def test_boost_qos_restores_peer_cap_after_global_overwrite(self):
        """A global transition must not leave an unselected policy modified."""
        for dynamic_ospm in (False, True):
            for original_boost in (0, 1):
                with self.subTest(ospm=dynamic_ospm, boost=original_boost):
                    verdict, reason, files, _ = self._run_boost_qos(
                        peer_policy=True, dynamic_ospm=dynamic_ospm,
                        original_boost=original_boost, overwrite_on_enable=True,
                    )
                    self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                    self.assertIn("Enabling CPPC boost overwrote", reason)
                    self.assertEqual(
                        files[f"{self.PEER_DIR}/scaling_max_freq"], "850000"
                    )
                    self.assertEqual(
                        files[f"{self.PEER_DIR}/boost"], str(original_boost)
                    )

    def test_boost_qos_global_only_early_skip_restores_peer_caps(self):
        """An initial global disable may overwrite peers even before a skip."""
        verdict, reason, files, _ = self._run_boost_qos(
            policy_boost_files=False, peer_policy=True,
            overwrite_on_disable=True, scaling_min=999999,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertIn("no room for two distinct", reason)
        self.assertEqual(files[f"{self.PEER_DIR}/scaling_max_freq"], "850000")

    def test_boost_qos_global_only_preserves_caps_on_fixed_kernel(self):
        """Global-only boost remains usable without rewriting unchanged peers."""
        for original_boost in (0, 1):
            with self.subTest(boost=original_boost):
                verdict, reason, files, writes = self._run_boost_qos(
                    policy_boost_files=False, peer_policy=2,
                    original_boost=original_boost,
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
                self.assertIsNone(reason)
                self.assertEqual(files[self.GLOBAL_BOOST], str(original_boost))
                self.assertFalse(any(
                    path.endswith("/boost") and path != self.GLOBAL_BOOST for path in files
                ))
                for path, request in self.peer_user_requests.items():
                    self.assertEqual(files[path], "850000")
                    self.assertEqual(request, 850000)
                    self.assertFalse(any(written == path for written, _ in writes))

    def test_boost_qos_global_only_restore_failure_reports_and_continues(self):
        """A failed peer restore is reported without abandoning later peers."""
        verdict, reason, files, _ = self._run_boost_qos(
            policy_boost_files=False, peer_policy=2,
            overwrite_on_enable=True, peer_restore_errno=errno.EIO,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("Failed to restore CPPC policy state", reason)
        self.assertIn("peer restore rejected", reason)
        self.assertEqual(
            files["/sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq"], "850000"
        )

    def test_boost_qos_global_only_restores_peer_caps_after_overwrite(self):
        """Without per-policy boost files, both toggle directions restore peers."""
        for original_boost in (0, 1):
            for direction in ("enable", "disable"):
                with self.subTest(boost=original_boost, direction=direction):
                    verdict, reason, files, writes = self._run_boost_qos(
                        policy_boost_files=False, peer_policy=2,
                        original_boost=original_boost,
                        overwrite_on_enable=direction == "enable",
                        overwrite_on_disable=direction == "disable",
                        policy_update_delay=0.25,
                    )
                    self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                    self.assertIn("CPPC boost overwrote", reason)
                    self.assertNotIn("Failed to restore CPPC policy state", reason)
                    self.assertEqual(files[self.GLOBAL_BOOST], str(original_boost))
                    cap_path = f"{self.POLICY_DIR}/scaling_max_freq"
                    self.assertEqual(files[cap_path], "900000")
                    self.assertEqual(writes.count((cap_path, 900000)), 1)
                    for path, request in self.peer_user_requests.items():
                        self.assertEqual(files[path], "850000")
                        self.assertEqual(request, 850000)
                        self.assertIn((path, 850000), writes)

    def test_boost_qos_global_only_snapshot_failure_prevents_writes(self):
        """Missing or unreadable peer caps fail closed before a global write."""
        for read_errno in (errno.ENOENT, errno.EACCES, errno.EIO):
            for original_boost in (0, 1):
                with self.subTest(error=read_errno, boost=original_boost):
                    verdict, reason, _, writes = self._run_boost_qos(
                        policy_boost_files=False, peer_policy=True,
                        original_boost=original_boost, peer_read_errno=read_errno,
                    )
                    self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                    self.assertIn("peer cap unreadable", reason)
                    self.assertEqual(writes, [])

    def test_boost_qos_results_raise_before_cleanup(self):
        """Native result exceptions stop the test before finally restores state."""
        cases = (
            ("GLTTestSkip", {"scaling_min": 999999}, "SKIP"),
            ("GLTTestSkip", {"scaling_min": 999999, "fail_restore": True}, "FAIL"),
            ("GLTTestFail", {"overwrite_on_enable": True}, "FAIL"),
            ("GLTTestFail", {"overwrite_on_disable": True}, "FAIL"),
        )
        for exception_name, options, expected in cases:
            with self.subTest(exception=exception_name, options=options):
                events = []
                observed_exception = self._observe_result_exception(
                    getattr(cpufreq.GLTCore, exception_name), events,
                )
                with self.mock.patch.object(
                        cpufreq.GLTCore, exception_name, observed_exception):
                    verdict, reason, files, writes = self._run_boost_qos(
                        peer_policy=True, **options,
                    )
                self.assertIs(verdict, getattr(cpufreq.GLTCore.GLTTestResult, expected))
                self.assertEqual(len(events), 1, "Do not catch and replace the native result")
                original_reason, writes_before_exit = events[0]
                self.assertIn(original_reason, reason)
                if not options.get("fail_restore"):
                    self.assertEqual(reason, original_reason)
                for patch in cpufreq.CPPC_BOOST_QOS_PATCHES:
                    if exception_name == "GLTTestFail":
                        self.assertIn(patch, original_reason)
                    else:
                        self.assertNotIn(patch, reason)
                self.assertLess(writes_before_exit, len(writes))
                self.assertEqual(writes[writes_before_exit], (self.GLOBAL_BOOST, 1))
                self.assertEqual(files[self.GLOBAL_BOOST], "1")
                self.assertEqual(files[f"{self.PEER_DIR}/scaling_max_freq"], "850000")
                if not options.get("overwrite_on_disable"):
                    self.assertNotIn((f"{self.POLICY_DIR}/scaling_max_freq", 866666), writes)

    def test_boost_qos_restores_caps_when_initial_disable_then_skip(self):
        """An early skip still restores caps overwritten by fixture setup."""
        verdict, reason, files, _ = self._run_boost_qos(
            peer_policy=True, overwrite_on_disable=True, scaling_min=999999,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertIn("no room for two distinct", reason)
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "999999")
        self.assertEqual(files[f"{self.PEER_DIR}/scaling_max_freq"], "850000")

    def test_boost_qos_peer_cleanup_failure_is_reported(self):
        """A rejected peer restore is reported without abandoning later peers."""
        verdict, reason, files, _ = self._run_boost_qos(
            peer_policy=2, overwrite_on_enable=True,
            peer_restore_errno=errno.EIO, policy_update_delay=0.25,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("peer restore rejected", reason)
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(
            files["/sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq"], "850000"
        )

    def test_boost_qos_does_not_rewrite_unchanged_peer_cap(self):
        """Only the selected policy receives test requests on a fixed kernel."""
        verdict, reason, _, writes = self._run_boost_qos(peer_policy=True)
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        self.assertFalse(any(
            path == f"{self.PEER_DIR}/scaling_max_freq" for path, _ in writes
        ))

    def test_boost_qos_fails_before_writes_if_peer_cap_is_unreadable(self):
        """Every affected cap must be captured before any global mutation."""
        verdict, reason, _, writes = self._run_boost_qos(
            peer_policy=True, peer_read_errno=errno.EIO,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("peer cap unreadable", reason)
        self.assertEqual(writes, [])

    def test_boost_qos_skips_when_requested_cap_never_becomes_visible(self):
        """Stale or clamped caps cannot prove an independent userspace request."""
        for requested in (733333, 866666):
            for observed in (800000, 900000):
                with self.subTest(requested=requested, observed=observed):
                    verdict, reason, files, writes = self._run_boost_qos(
                        cap_override=(requested, observed),
                    )
                    self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
                    self.assertIn(f"requested {requested}, observed {observed}", reason)
                    self.assertNotIn("overwrote", reason)
                    self.assertNotIn("Failed to restore", reason)
                    self.assertEqual(
                        files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000"
                    )
                    self.assertEqual(
                        [value for path, value in writes if path == self.GLOBAL_BOOST],
                        [0, 1] if requested == 733333 else [0, 1, 1],
                    )

    def test_scaling_max_write_waits_for_async_policy_refresh(self):
        """Wait through stale/intermediate values and honor the final deadline."""
        scaling_max_path = f"{self.POLICY_DIR}/scaling_max_freq"
        cases = (
            (None, 0.2, 733333, 0.2),
            (0.1, 0.35, 733333, 0.4),
            (0.1, 0.45, 733333, 0.45),
            (0.1, 0.46, 800000, 0.45),
            (0.1, float("inf"), 800000, 0.45),
            (None, float("inf"), 900000, 0.45),
        )
        for intermediate_at, target_at, expected, elapsed in cases:
            with self.subTest(intermediate=intermediate_at, target=target_at):
                self.qos_elapsed = 0

                def read_value(_, intermediate_at=intermediate_at, target_at=target_at):
                    if self.qos_elapsed >= target_at:
                        return "733333"
                    if intermediate_at is not None and self.qos_elapsed >= intermediate_at:
                        return "800000"
                    return "900000"

                def advance(seconds):
                    self.qos_elapsed = round(self.qos_elapsed + seconds, 10)

                with self.mock.patch.object(
                        cpufreq.glt_utils, "write_sysfs_value") as write_mock, \
                     self.mock.patch.object(
                         cpufreq.glt_utils, "read_sysfs_value", side_effect=read_value), \
                     self.mock.patch.object(
                         cpufreq.time, "monotonic", side_effect=lambda: self.qos_elapsed), \
                     self.mock.patch.object(cpufreq.time, "sleep", side_effect=advance):
                    observed = getattr(cpufreq, "_write_cppc_scaling_max_and_wait")(
                        scaling_max_path, 733333, timeout_secs=0.45, poll_secs=0.2,
                    )

                self.assertEqual(observed, expected)
                self.assertEqual(self.qos_elapsed, elapsed)
                write_mock.assert_called_once_with(scaling_max_path, 733333)

    def test_boost_qos_fails_when_disable_overwrites_user_cap(self):
        """A boost-disable overwrite reproduces the pre-patch regression."""
        verdict, reason, files, _ = self._run_boost_qos(overwrite_on_disable=True)

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("Disabling CPPC boost overwrote", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[self.POLICY_BOOST], "1")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")

    def test_boost_qos_fails_when_enable_overwrites_user_cap(self):
        """A boost-enable overwrite reproduces the pre-patch regression."""
        verdict, reason, files, _ = self._run_boost_qos(overwrite_on_enable=True)

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("Enabling CPPC boost overwrote", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[self.POLICY_BOOST], "1")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")

    def test_boost_qos_fails_when_state_restore_fails(self):
        """Cleanup failure is reported and does not prevent peer restoration."""
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite):
                verdict, reason, files, _ = self._run_boost_qos(
                    fail_restore=True, peer_policy=True,
                    overwrite_on_enable=overwrite,
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                self.assertIn("Failed to restore CPPC policy state", reason)
                self.assertIn("restore rejected", reason)
                for patch in cpufreq.CPPC_BOOST_QOS_PATCHES:
                    if overwrite:
                        self.assertIn(patch, reason)
                    else:
                        self.assertNotIn(patch, reason)
                self.assertEqual(
                    files[f"{self.PEER_DIR}/scaling_max_freq"], "850000"
                )

    def test_boost_qos_passes_and_restores_original_state(self):
        """Independent QoS requests preserve both caps and restore policy state."""
        verdict, reason, files, writes = self._run_boost_qos()

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[self.POLICY_BOOST], "1")
        self.assertEqual(files[f"{self.POLICY_DIR}/cpuinfo_max_freq"], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(
            writes[-2:],
            [
                (self.GLOBAL_BOOST, 1),
                (f"{self.POLICY_DIR}/scaling_max_freq", 900000),
            ],
        )

    def test_boost_qos_passes_from_disabled_state_and_resynchronizes_policies(self):
        """Cleanup replays global state to keep every policy synchronized."""
        verdict, reason, files, writes = self._run_boost_qos(original_boost=0)

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "0")
        self.assertEqual(files[self.POLICY_BOOST], "0")
        self.assertEqual(files[f"{self.POLICY_DIR}/cpuinfo_max_freq"], "1000000")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(
            [value for path, value in writes if path == self.GLOBAL_BOOST],
            [1, 0, 0],
        )

    def test_boost_qos_skips_without_boost_headroom(self):
        """A policy without a higher boost frequency cannot prove the behavior."""
        verdict, reason, files, _ = self._run_boost_qos(boosted_max=1000000)

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertIn("does not expose a distinct boost frequency", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[self.POLICY_BOOST], "1")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")

    def test_boost_qos_skips_when_firmware_has_no_boost_headroom(self):
        """Equal performance without OSPM control is skipped before writes."""
        files = {
            self.GLOBAL_BOOST: "0",
            self.POLICY_BOOST: "0",
            f"{self.POLICY_DIR}/cpuinfo_max_freq": "1000000",
            f"{self.POLICY_DIR}/scaling_min_freq": "600000",
            f"{self.POLICY_DIR}/scaling_max_freq": "900000",
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq": "900000",
            f"{self.CPPC_CAPS_DIR}/highest_perf": "100",
            f"{self.CPPC_CAPS_DIR}/nominal_perf": "100",
        }

        with self.mock.patch.object(
                cpufreq, "_validate_cppc_prerequisites",
                return_value=(True, None)), \
             self.mock.patch.object(
                cpufreq, "_get_first_cppc_cpufreq_dir",
                return_value=(self.POLICY_DIR, None)), \
             self.mock.patch.object(
                 cpufreq.os.path, "isfile",
                 side_effect=lambda path: path != self.OSPM_NOMINAL), \
             self.mock.patch.object(
                 cpufreq.glob, "glob", side_effect=lambda pattern: (
                     [self.POLICY_BOOST] if pattern.endswith("/boost")
                     else ["/sys/devices/system/cpu/cpufreq/policy0"]
                 )), \
             self.mock.patch.object(
                 cpufreq.glt_utils, "read_sysfs_value",
                 side_effect=lambda path: files[path]), \
             self.mock.patch.object(
                 cpufreq.glt_utils, "write_sysfs_value") as write_mock:
            verdict, reason = self._invoke_boost_qos()

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertIn("no OSPM nominal-frequency control", reason)
        write_mock.assert_not_called()

    def test_boost_qos_skips_when_policy_dirs_unavailable(self):
        """An empty policy enumeration cannot establish a complete snapshot."""
        for policy_boost_files in (False, True):
            with self.subTest(boost_files=policy_boost_files):
                verdict, reason, _, writes = self._run_boost_qos(
                    peer_policy=True, policy_boost_files=policy_boost_files,
                    policy_dirs_available=False,
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
                self.assertIn("Cannot enumerate cpufreq policies", reason)
                self.assertEqual(writes, [])

    def test_boost_qos_passes_with_ospm_defined_boost_and_restores_state(self):
        """CPPC v4 can create a boost range by lowering OSPM nominal."""
        verdict, reason, files, writes = self._run_boost_qos(
            original_boost=0, dynamic_ospm=True
        )

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "0")
        self.assertEqual(files[self.POLICY_BOOST], "0")
        self.assertEqual(files[self.OSPM_NOMINAL], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/cpuinfo_max_freq"], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(
            [value for path, value in writes if path == self.GLOBAL_BOOST],
            [1, 0, 1, 0, 1, 0, 0],
        )

    def test_boost_qos_detects_overwrite_with_ospm_defined_boost(self):
        """The pre-patch overwrite is detected with an OSPM boost range."""
        verdict, reason, files, _ = self._run_boost_qos(
            original_boost=0, dynamic_ospm=True, overwrite_on_enable=True
        )

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("Enabling CPPC boost overwrote", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "0")
        self.assertEqual(files[self.POLICY_BOOST], "0")
        self.assertEqual(files[self.OSPM_NOMINAL], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/cpuinfo_max_freq"], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")

    def test_boost_qos_fails_and_restores_after_ospm_write_error(self):
        """An OSPM programming failure is reported and cleanup still runs."""
        verdict, reason, files, writes = self._run_boost_qos(
            original_boost=0, dynamic_ospm=True, ospm_write_errno=errno.EIO
        )

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertTrue(reason.startswith("CPPC boost-frequency QoS validation failed:"))
        self.assertIn("OSPM nominal write rejected", reason)
        self.assertIn("Additionally, Failed to restore CPPC policy state", reason)
        self.assertIn(f"cannot restore {self.OSPM_NOMINAL}", reason)
        for patch in cpufreq.CPPC_BOOST_QOS_PATCHES:
            self.assertNotIn(patch, reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "0")
        self.assertEqual(files[self.POLICY_BOOST], "0")
        self.assertEqual(files[self.OSPM_NOMINAL], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertIn((self.OSPM_NOMINAL, 1200000), writes)

    def test_boost_qos_preserves_write_only_ospm_state_while_boosted(self):
        """Disabled cpuinfo.max discovers and restores a write-only OSPM value."""
        verdict, reason, files, writes = self._run_boost_qos(
            original_boost=1,
            dynamic_ospm=True,
            ospm_nominal=1100000,
        )

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.PASS)
        self.assertIsNone(reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[self.POLICY_BOOST], "1")
        self.assertEqual(files[self.OSPM_NOMINAL], "1100000")
        self.assertEqual(files[f"{self.POLICY_DIR}/cpuinfo_max_freq"], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(
            [value for path, value in writes if path == self.GLOBAL_BOOST],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )

    def test_boost_qos_reports_nominal_restore_mismatch_while_boosted(self):
        """The boosted maximum must not hide an incorrect restored nominal."""
        verdict, reason, files, _ = self._run_boost_qos(
            dynamic_ospm=True, ospm_nominal=1100000, ospm_restore_offset=-1000,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("effective OSPM nominal restored to 1099000", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")

    def test_boost_qos_restores_boost_after_nominal_verification_error(self):
        """Failure to read nominal must not prevent restoring enabled boost."""
        verdict, reason, files, _ = self._run_boost_qos(
            dynamic_ospm=True, nominal_restore_read_errno=errno.EIO,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("cannot verify restored OSPM nominal", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")

    def test_boost_qos_skips_without_cleanup_after_unsupported_ospm_write(self):
        """A rejected optional write does not schedule restoration writes."""
        verdict, reason, files, writes = self._run_boost_qos(
            original_boost=0,
            dynamic_ospm=True,
            ospm_write_errno=errno.EOPNOTSUPP,
        )

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertIn("does not support programming", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "0")
        self.assertEqual(files[self.POLICY_BOOST], "0")
        self.assertEqual(files[self.OSPM_NOMINAL], "1200000")
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(writes, [(self.OSPM_NOMINAL, 1050000)])

    def test_boost_qos_skips_when_ospm_fixture_cannot_enable_boost(self):
        """An unsupported no-op skips without rewriting unchanged cap requests."""
        for error in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENODEV):
            with self.subTest(errno=error):
                verdict, reason, files, writes = self._run_boost_qos(
                    original_boost=0, dynamic_ospm=True, peer_policy=True,
                    boost_enable_errno=error, fail_restore=True,
                )

                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
                self.assertIn("did not provide global CPPC boost support", reason)
                self.assertEqual(files[self.GLOBAL_BOOST], "0")
                self.assertEqual(files[self.POLICY_BOOST], "0")
                self.assertEqual(files[self.OSPM_NOMINAL], "1200000")
                self.assertEqual(
                    files[f"{self.POLICY_DIR}/cpuinfo_max_freq"], "1200000"
                )
                self.assertEqual(
                    files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000"
                )
                self.assertEqual(
                    files[f"{self.PEER_DIR}/scaling_max_freq"], "850000"
                )
                self.assertEqual(
                    [value for path, value in writes if path == self.GLOBAL_BOOST],
                    [1],
                )
                self.assertFalse(any(
                    path.endswith("/scaling_max_freq") for path, _ in writes
                ))

    def test_boost_qos_restores_caps_after_disabled_fixture_rejects_boost(self):
        """A failed fixture must still restore caps changed by initial disable."""
        verdict, reason, files, writes = self._run_boost_qos(
            original_boost=1, dynamic_ospm=True, peer_policy=True,
            boost_enable_errno=errno.EINVAL, overwrite_on_disable=True,
        )
        # Enabling the original boost state is rejected too. Report that
        # cleanup failure, but do not abandon either affected cap restoration.
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("cannot restore", reason)
        self.assertEqual(files[f"{self.POLICY_DIR}/scaling_max_freq"], "900000")
        self.assertEqual(files[f"{self.PEER_DIR}/scaling_max_freq"], "850000")
        self.assertIn((f"{self.POLICY_DIR}/scaling_max_freq", 900000), writes)
        self.assertIn((f"{self.PEER_DIR}/scaling_max_freq", 850000), writes)
        self.assertEqual(
            [value for path, value in writes if path == self.GLOBAL_BOOST],
            [0, 1, 1],
        )

    def test_boost_qos_restores_after_partial_fixture_boost_error(self):
        """Even an unexpected error may follow a partially applied transition."""
        for error in (errno.EIO, errno.EINVAL):
            with self.subTest(errno=error):
                verdict, reason, files, writes = self._run_boost_qos(
                    original_boost=0, dynamic_ospm=True, peer_policy=True,
                    boost_enable_errno=error, partial_boost_error=True,
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                self.assertIn("global boost enable rejected", reason)
                self.assertNotIn("Failed to restore", reason)
                self.assertEqual(files[self.GLOBAL_BOOST], "0")
                self.assertEqual(files[self.POLICY_BOOST], "0")
                self.assertEqual(files[f"{self.PEER_DIR}/boost"], "0")
                self.assertEqual(files[self.OSPM_NOMINAL], "1200000")
                self.assertEqual(
                    [value for path, value in writes if path == self.GLOBAL_BOOST],
                    [1, 0],
                )

    def test_boost_qos_skips_without_global_boost_control(self):
        """Missing global boost control has an explicit skip reason."""
        with self.mock.patch.object(
                cpufreq, "_validate_cppc_prerequisites",
                return_value=(True, None)), \
             self.mock.patch.object(
                cpufreq, "_get_first_cppc_cpufreq_dir",
                return_value=(self.POLICY_DIR, None)), \
             self.mock.patch.object(cpufreq.os.path, "isfile", return_value=False), \
             self.mock.patch.object(
                 cpufreq.glt_utils, "read_sysfs_value") as read_mock, \
             self.mock.patch.object(
                 cpufreq.glt_utils, "write_sysfs_value") as write_mock:
            verdict, reason = self._invoke_boost_qos()

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertIn("does not expose global boost control", reason)
        read_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_boost_qos_skips_when_policy_state_is_not_synchronized(self):
        """A pre-existing mixed boost state is left untouched."""
        verdict, reason, files, writes = self._run_boost_qos(
            original_boost=1, policy_boost=0
        )

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertIn("boost states are not synchronized", reason)
        self.assertEqual(files[self.GLOBAL_BOOST], "1")
        self.assertEqual(files[self.POLICY_BOOST], "0")
        self.assertEqual(writes, [])

    def test_boost_qos_skips_before_writes_on_mixed_policy_drivers(self):
        """A global toggle is unsafe unless every policy uses cppc_cpufreq."""
        reason = "cppc_cpufreq not active for all cpufreq policies."
        with self.mock.patch.object(
                cpufreq, "_validate_cppc_prerequisites",
                return_value=(False, reason)), \
             self.mock.patch.object(
                 cpufreq, "_get_first_cppc_cpufreq_dir") as select_mock, \
             self.mock.patch.object(
                 cpufreq.glt_utils, "write_sysfs_value") as write_mock:
            verdict, observed_reason = self._invoke_boost_qos()

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
        self.assertEqual(observed_reason, reason)
        select_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_boost_qos_skips_non_cppc_and_missing_policies(self):
        """Capability gating rejects unsupported drivers without CPU filtering."""
        for drivers in ([], ["acpi-cpufreq"], ["cppc_cpufreq", "acpi-cpufreq"]):
            with self.subTest(drivers=drivers):
                driver_files = {
                    f"/sys/devices/system/cpu/cpu{index}/cpufreq/scaling_driver": driver
                    for index, driver in enumerate(drivers)
                }

                def open_driver(path, *_args, driver_files=driver_files, **_kwargs):
                    return self.mock.mock_open(read_data=driver_files[path])()

                with self.mock.patch.object(
                        cpufreq, "_cpu_cpufreq_glob", return_value=list(driver_files)), \
                     self.mock.patch("builtins.open", side_effect=open_driver), \
                     self.mock.patch.object(
                         cpufreq, "_get_first_cppc_cpufreq_dir") as select_mock, \
                     self.mock.patch.object(
                         cpufreq.glt_utils, "write_sysfs_value") as write_mock:
                    verdict, reason = self._invoke_boost_qos()

                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.SKIP)
                self.assertIn("cppc_cpufreq not active" if drivers else "cpufreq sysfs not available", reason)
                select_mock.assert_not_called()
                write_mock.assert_not_called()

    def test_boost_qos_fails_before_writes_on_unreadable_policy_driver(self):
        """Unreadable driver state is a validation failure, not a platform skip."""
        reason = "Cannot read policy0/scaling_driver: permission denied"
        with self.mock.patch.object(
                cpufreq, "_validate_cppc_prerequisites",
                return_value=(False, reason)), \
             self.mock.patch.object(
                 cpufreq, "_get_first_cppc_cpufreq_dir") as select_mock, \
             self.mock.patch.object(
                 cpufreq.glt_utils, "write_sysfs_value") as write_mock:
            verdict, observed_reason = self._invoke_boost_qos()

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertEqual(observed_reason, reason)
        select_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_boost_qos_fails_if_policy_selection_becomes_unreadable(self):
        """A second driver-read failure after preflight also fails closed."""
        reason = "Cannot read policy0/scaling_driver: input/output error"
        with self.mock.patch.object(
                cpufreq, "_validate_cppc_prerequisites",
                return_value=(True, None)), \
             self.mock.patch.object(
                 cpufreq, "_get_first_cppc_cpufreq_dir",
                 return_value=(None, reason)), \
             self.mock.patch.object(
                 cpufreq.glt_utils, "write_sysfs_value") as write_mock:
            verdict, observed_reason = self._invoke_boost_qos()

        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertEqual(observed_reason, reason)
        write_mock.assert_not_called()

    def test_boost_qos_failure_recommends_the_complete_series(self):
        """Identify the discriminator but include its lifetime fix in remediation."""
        for overwrite in ("enable", "disable"):
            with self.subTest(overwrite=overwrite):
                verdict, reason, _, _ = self._run_boost_qos(
                    overwrite_on_enable=overwrite == "enable",
                    overwrite_on_disable=overwrite == "disable",
                )
                self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
                self.assertIn("This symptom matches the fix in", reason)
                self.assertIn("6e39ba4e5a82", reason)
                self.assertIn("complete CPPC boost QoS series", reason)
                self.assertIn("\n".join(cpufreq.CPPC_BOOST_QOS_PATCHES), reason)

    def test_boost_qos_unrelated_failure_does_not_cite_patch(self):
        """An environmental failure reports its cause without a patch hint."""
        verdict, reason, _, _ = self._run_boost_qos(
            peer_policy=True, peer_read_errno=errno.EIO,
        )
        self.assertIs(verdict, cpufreq.GLTCore.GLTTestResult.FAIL)
        self.assertIn("peer cap unreadable", reason)
        for patch in cpufreq.CPPC_BOOST_QOS_PATCHES:
            self.assertNotIn(patch, reason)


class TestCppcSysfs(GLTTestBase):
    """Test CPPC sysfs helper validation."""

    def test_validate_cppc_sysfs_helpers_round_trips_limits_and_perf_limited(self):
        """A cppc_cpufreq policy with required sysfs files validates cleanly."""
        files = {
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/perf_limited": "0\n",
        }
        writes = []

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            return files[path].strip()

        def fake_write(path, value):
            writes.append((path, value))
            files[path] = f"{value}\n"

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read), \
             self.mock.patch.object(cpufreq.glt_utils, "write_sysfs_value",
                                    side_effect=fake_write):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertIn(
            ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq", "450000"),
            writes,
        )
        self.assertIn(
            ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "3447000"),
            writes,
        )
        self.assertIn(
            ("/sys/devices/system/cpu/cpu0/cpufreq/perf_limited", "0x3"),
            writes,
        )

    def test_validate_cppc_sysfs_helpers_fails_without_perf_limited(self):
        """Missing perf_limited reports the missing patch-facing ABI."""
        files = {
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq": "3447000\n",
        }

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            if path not in files:
                raise OSError("missing")
            return files[path].strip()

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read), \
             self.mock.patch.object(cpufreq.glt_utils, "write_sysfs_value"):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertFalse(ok)
        self.assertIn("perf_limited", reason)

    def test_validate_cppc_sysfs_helpers_restores_limits_after_failure(self):
        """Original scaling limits are restored when later sysfs validation fails."""
        files = {
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver": "cppc_cpufreq\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq": "450000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq": "3447000\n",
            "/sys/devices/system/cpu/cpu0/cpufreq/perf_limited": "0\n",
        }
        writes = []

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            return files[path].strip()

        def fake_write(path, value):
            writes.append((path, value))
            if path.endswith("/perf_limited"):
                raise OSError("write failed")
            files[path] = f"{value}\n"

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read), \
             self.mock.patch.object(cpufreq.glt_utils, "write_sysfs_value",
                                    side_effect=fake_write):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertFalse(ok)
        self.assertIn("write failed", reason)
        self.assertEqual(
            writes[-2:],
            [
                ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "3447000"),
                ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq", "450000"),
            ],
        )

    def test_validate_cppc_sysfs_helpers_fails_without_cppc_driver(self):
        """A non-CPPC policy is not a valid target for this coverage."""

        def fake_glob(filename):
            return [f"/sys/devices/system/cpu/cpu0/cpufreq/{filename}"]

        def fake_read(path):
            if path.endswith("/scaling_driver"):
                return "acpi-cpufreq"
            raise AssertionError(f"unexpected read: {path}")

        with self.mock.patch.object(cpufreq, "_cpu_cpufreq_glob", side_effect=fake_glob), \
             self.mock.patch.object(cpufreq.glt_utils, "read_sysfs_value",
                                    side_effect=fake_read):
            ok, reason = cpufreq._validate_cppc_sysfs_helpers(  # pylint: disable=protected-access
                self.mock_logger
            )

        self.assertFalse(ok)
        self.assertIn("cppc_cpufreq", reason)
