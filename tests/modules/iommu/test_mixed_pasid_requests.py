# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PASID 1 requests and bounded attributed traces prove the mixed-PASID discriminator."""

import errno
import json
import struct

from tests.modules.iommu.test_mixed_pasid import MixedPasidTestBase, native, trigger, worker


class TestMixedPasidRequests(MixedPasidTestBase):
    """Keep PASID 1 request scope separate from observed eligibility evidence."""

    def setUp(self):
        self.production_observation_timeout = worker.OBSERVATION_TIMEOUT
        super().setUp()
        self.allocation_calls = []

    def _capture_allocation(self, fd, request, payload, mutate=True):
        self.assertTrue(mutate)
        self.allocation_calls.append((fd, request, bytes(payload)))
        if request == trigger.VFIO_DEVICE_BIND_IOMMUFD:
            struct.pack_into("<I", payload, 12, 101)
        elif request == trigger.IOMMU_GET_HW_INFO:
            payload[28] = 20
        elif request == trigger.IOMMU_IOAS_ALLOC:
            struct.pack_into("<I", payload, 8, 202)
        elif request == trigger.IOMMU_HWPT_ALLOC:
            struct.pack_into("<I", payload, 16, 303)
        return 0

    def test_allocate_context_does_not_query_hw_info(self):
        """PASID 1 allocation needs only bind, IOAS, HWPT and base attachment."""
        resources = {"fds": []}
        with self.mock.patch("os.open", side_effect=[41, 42]), \
                self.mock.patch("fcntl.ioctl", side_effect=self._capture_allocation), \
                self.mock.patch.object(worker, "_find_vfio_cdev", return_value="/dev/vfio/devices/vfio9"):
            context = worker.allocate_context(dict(self.fixture, resources=resources))
        self.assertEqual(self.allocation_calls, [
            (42, trigger.VFIO_DEVICE_BIND_IOMMUFD, struct.pack("<2IiIQ", 24, 0, 41, 0, 0)),
            (41, trigger.IOMMU_IOAS_ALLOC, struct.pack("<3I", 12, 0, 0)),
            (41, trigger.IOMMU_HWPT_ALLOC, struct.pack("<8IQ2I", 48, 8, 101, 202, 0, 0, 0, 0, 0, 0, 0)),
            (42, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT, struct.pack("<4I", 16, 0, 303, 0)),
        ])
        self.assertNotIn("max_pasid_log2", context)

    def test_requests_issue_only_pasid_one(self):
        """A successful PASID 1 lifecycle must never issue an upper request."""
        text = self._text([self._event("entry"), self._event("set"),
                           self._event("return", ret=0), self._event("block")])
        upper_error = trigger.TriggerError("expected upper EINVAL")
        upper_error.__cause__ = OSError(errno.EINVAL, "upper attach")
        read_trace = self.mock.Mock(side_effect=[text, text, text])
        with self.mock.patch.object(worker, "_xioctl", side_effect=[None, None, upper_error]) as ioctl, \
                self.mock.patch.object(worker, "OBSERVATION_TIMEOUT", 2.0), \
                self.mock.patch.object(worker.time, "monotonic", side_effect=[0.0, 0.25, 1.0, 2.0]) as clock, \
                self.mock.patch.object(worker.time, "sleep") as sleep:
            report = worker.run_requests(dict(self.context, max_pasid_log2=20), self.fixture, read_trace)
        self.assertEqual(ioctl.call_args_list, [
            self.mock.call(42, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT, struct.pack("<4I", 16, 1, 303, 1)),
            self.mock.call(42, worker.VFIO_DEVICE_DETACH_IOMMUFD_PT, struct.pack("<3I", 12, 1, 1)),
        ])
        self.assertEqual(report["status"], "PASS", report)
        self.assertIs(report["valid"], True)
        self.assertEqual(report["detach_result"], 0)
        self.assertEqual(report["trace"], text)
        self.assertEqual(read_trace.call_count, 3)
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(sleep.call_args_list, [self.mock.call(0.02), self.mock.call(0.02)])

    def test_control_never_retries_another_pasid(self):
        """An attributed PASID 1 rejection stays the only attempted request."""
        text = self._text([self._event("entry"), self._event("return", ret=-errno.EINVAL)])
        attach_error = trigger.TriggerError("expected PASID 1 EINVAL")
        attach_error.__cause__ = OSError(errno.EINVAL, "PASID 1 attach")
        upper_error = trigger.TriggerError("expected upper EINVAL")
        upper_error.__cause__ = OSError(errno.EINVAL, "upper attach")
        read_trace = self.mock.Mock(side_effect=[text, text, text])
        with self.mock.patch.object(worker, "_xioctl", side_effect=[attach_error, upper_error]) as ioctl, \
                self.mock.patch.object(worker, "OBSERVATION_TIMEOUT", 2.0), \
                self.mock.patch.object(worker.time, "monotonic", side_effect=[0.0, 0.25, 1.0, 2.0]) as clock, \
                self.mock.patch.object(worker.time, "sleep") as sleep:
            report = worker.run_requests(dict(self.context, max_pasid_log2=20), self.fixture, read_trace)
        self.assertEqual(ioctl.call_args_list, [
            self.mock.call(42, trigger.VFIO_DEVICE_ATTACH_IOMMUFD_PT, struct.pack("<4I", 16, 1, 303, 1)),
        ])
        self.assertEqual(report["status"], "FAIL", report)
        self.assertIn("valid PASID reached core and returned EINVAL", report["reason"])
        self.assertEqual(read_trace.call_count, 3)
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(sleep.call_args_list, [self.mock.call(0.02), self.mock.call(0.02)])

    def test_trace_rejects_second_request(self):
        """A formerly accepted upper pair is extra evidence for PASID 1-only scope."""
        text = self._text([self._event("entry"), self._event("set"),
                           self._event("return", ret=0), self._event("block"),
                           self._event("entry", self.bound),
                           self._event("return", self.bound, ret=-errno.EINVAL)])
        report = worker.evaluate_trace(text, self.fixture, self.pid, self.tid, 0)
        self.assertEqual(report["status"], "FAIL", report)

    def test_native_accepts_single_pasid_proof(self):
        """The native boundary accepts PASID 1 proof with owned cleanup and no upper key."""
        process = self.mock.Mock(pid=self.pid, returncode=0)
        process.stdin.closed = False
        process.poll.return_value = 0
        report = {"type": "result", "status": "PASS", "reason": "", "valid": True,
                  "cleanup_ok": True, "residuals": [], "fixture": self.fixture, "modules_loaded": []}
        process.communicate.return_value = (json.dumps(report) + "\n", "")
        with self.mock.patch("subprocess.Popen", return_value=process), \
                self.mock.patch.object(native, "_mixed_pasid_first_message", return_value={
                    "type": "observer", "observer": self.observer}), \
                self.mock.patch.object(native, "_mixed_pasid_observer_identity", return_value=dict(self.observer)), \
                self.mock.patch.object(native, "_mixed_pasid_cleanup_readback") as cleanup, \
                self.mock.patch("os.kill"):
            self.assertIsNone(getattr(native, "test_mixed_pasid_attach")(self.mock_sd, self.mock_logger))
        process.stdin.write.assert_called_once_with("ACK\n")
        process.stdin.flush.assert_called_once_with()
        process.communicate.assert_called_once_with(timeout=180)
        cleanup.assert_called_once_with(report, self.observer)
        process.stdin.close.assert_called_once_with()
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()

    def test_late_duplicate_gpu_block_remains_failure(self):
        """A duplicate GPU block in the final snapshot invalidates an earlier lifecycle."""
        ready = {"event": "ready"}
        records = [self._event("entry"), self._event("set"),
                   self._event("return", ret=0), self._event("block")]
        read_trace = self.mock.Mock(side_effect=[
            self._text([ready]), self._text([ready] + records),
            self._text([ready] + records + [self._event("block")]),
        ])
        upper_error = trigger.TriggerError("expected upper EINVAL")
        upper_error.__cause__ = OSError(errno.EINVAL, "upper attach")
        with self.mock.patch.object(worker, "_xioctl", side_effect=[None, None, upper_error]), \
                self.mock.patch.object(worker, "OBSERVATION_TIMEOUT", 2.0), \
                self.mock.patch.object(worker.time, "monotonic", side_effect=[0.0, 0.25, 1.0, 2.0]) as clock, \
                self.mock.patch.object(worker.time, "sleep") as sleep:
            report = worker.run_requests(dict(self.context, max_pasid_log2=20), self.fixture, read_trace)
        self.assertEqual(report["status"], "FAIL", report)
        self.assertEqual(read_trace.call_count, 3)
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(sleep.call_args_list, [self.mock.call(0.02), self.mock.call(0.02)])

    def test_late_parent_block_remains_failure(self):
        """A parent callback in the final snapshot invalidates an earlier lifecycle."""
        ready = {"event": "ready"}
        records = [self._event("entry"), self._event("set"),
                   self._event("return", ret=0), self._event("block")]
        read_trace = self.mock.Mock(side_effect=[
            self._text([ready]), self._text([ready] + records),
            self._text([ready] + records + [self._event("block", member=self.fixture["parent"])]),
        ])
        upper_error = trigger.TriggerError("expected upper EINVAL")
        upper_error.__cause__ = OSError(errno.EINVAL, "upper attach")
        with self.mock.patch.object(worker, "_xioctl", side_effect=[None, None, upper_error]), \
                self.mock.patch.object(worker, "OBSERVATION_TIMEOUT", 2.0), \
                self.mock.patch.object(worker.time, "monotonic", side_effect=[0.0, 0.25, 1.0, 2.0]) as clock, \
                self.mock.patch.object(worker.time, "sleep") as sleep:
            report = worker.run_requests(dict(self.context, max_pasid_log2=20), self.fixture, read_trace)
        self.assertEqual(report["status"], "FAIL", report)
        self.assertEqual(read_trace.call_count, 3)
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(sleep.call_args_list, [self.mock.call(0.02), self.mock.call(0.02)])

    def test_production_observation_timeout_remains_two_seconds(self):
        """Unit fixture overrides cannot shorten the production observation window."""
        self.assertEqual(self.production_observation_timeout, 2.0)

    def _check_allocation_error(self, error):
        allocate, cleanup = worker.allocate_context, worker.cleanup_resources
        stack, mocked = self._worker_mocks()
        mocked["allocate_context"].side_effect = allocate
        mocked["cleanup_resources"].side_effect = cleanup
        with stack, self.mock.patch("os.open", side_effect=[41, 42]), \
                self.mock.patch("fcntl.ioctl", side_effect=error) as ioctl, \
                self.mock.patch("os.close") as close, \
                self.mock.patch.object(worker, "_cleanup_observer", return_value=[]), \
                self.mock.patch.object(worker, "_find_vfio_cdev", return_value="/dev/vfio/devices/vfio9"):
            report = worker.run_worker(self.mock.Mock(return_value=True))
        self.assertEqual(report["status"], "FAIL", report)
        self.assertIn("allocation failed", report["reason"])
        self.assertTrue(report["cleanup_ok"])
        self.assertEqual(close.call_args_list, [self.mock.call(42), self.mock.call(41)])
        mocked["run_requests"].assert_not_called()
        ioctl.assert_called_once_with(42, trigger.VFIO_DEVICE_BIND_IOMMUFD,
                                      bytearray(struct.pack("<2IiIQ", 24, 0, 41, 0, 0)), True)

    def test_allocation_errors_preserve_owned_fd_cleanup(self):
        """An allocation error closes both acquired descriptors through the owning child."""
        for error in (OSError(errno.EIO, "allocation failed"), trigger.TriggerError("allocation failed")):
            with self.subTest(error=type(error).__name__):
                self._check_allocation_error(error)

    def test_positive_ioctl_trace_disagreement_is_diagnosed(self):
        """Both disagreement directions override the kernel-control or detach verdict."""
        for trace_ret, actual_ret in ((0, -errno.EINVAL), (-errno.EINVAL, 0)):
            with self.subTest(trace_ret=trace_ret, actual_ret=actual_ret):
                failure = trigger.TriggerError("attributed EINVAL")
                failure.__cause__ = OSError(errno.EINVAL, "attach")
                with self.mock.patch.object(worker, "_xioctl", side_effect=failure if actual_ret else None):
                    report = worker.run_requests(self.context, self.fixture,
                                                 self.mock.Mock(return_value=self._text(self._records(trace_ret))))
                self.assertEqual(report["status"], "FAIL", report)
                self.assertIn("valid ioctl result disagrees", report["reason"])

    def test_malformed_trace_reason_is_preserved(self):
        """Malformed evidence retains its original diagnostic before ioctl agreement checks."""
        records = self._records()
        del records[1]
        for text, reason in (("{malformed", "Expecting property name"),
                             (self._text(records), "valid PASID needs one ordered")):
            with self.subTest(reason=reason), self.mock.patch.object(worker, "_xioctl"):
                report = worker.run_requests(self.context, self.fixture, self.mock.Mock(return_value=text))
            self.assertEqual(report["status"], "FAIL", report)
            self.assertIsNone(report["bound"])
            self.assertFalse(report["valid"])
            self.assertIn(reason, report["reason"])

    def test_evaluate_trace_requires_actual_successful_detach(self):
        """A good prefix cannot populate proof fields without actual detach success."""
        for detach in (None, -errno.EIO):
            with self.subTest(detach=detach):
                report = worker.evaluate_trace(self._text(self._records()), self.fixture,
                                               self.pid, self.tid, detach)
                self.assertEqual(report["status"], "FAIL")
                self.assertIn("explicit detach0", report["reason"])
                self.assertIsNone(report["bound"])
                self.assertFalse(report["valid"])

    def test_control_trace_does_not_require_actual_detach(self):
        """An attributed control rejection does not need an inapplicable detach."""
        report = worker.evaluate_trace(self._text(self._records(-errno.EINVAL)), self.fixture,
                                       self.pid, self.tid, None)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("valid PASID reached core and returned EINVAL", report["reason"])
        self.assertEqual(report["bound"], self.bound)
        self.assertFalse(report["valid"])

    def test_ready_and_blank_records_preserve_request_proof(self):
        """Accepted readiness and blank framing preserve lifecycle order at every boundary."""
        records = self._records()
        for position in (0, 2, len(records)):
            with self.subTest(position=position), self.mock.patch.object(worker, "_xioctl"):
                text = (self._text(records[:position]) + '\n{"event":"ready"}\n\n'
                        + self._text(records[position:]))
                report = worker.run_requests(self.context, self.fixture, self.mock.Mock(return_value=text))
            self.assertEqual(report["status"], "PASS", report)
            self.assertTrue(report["valid"])
            self.assertEqual(report["bound"], self.bound)
