// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define _GNU_SOURCE

#include "glt_arm64_kvm.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#if !defined(__aarch64__)
#error "glt_arm64_kvm must be built for AArch64"
#endif

static volatile sig_atomic_t run_timed_out;
static _Atomic(struct kvm_run *) running_vcpu;

_Static_assert(ATOMIC_POINTER_LOCK_FREE == 2,
	       "the SIGALRM handler requires lock-free pointer access");

static void alarm_handler(int signal_number)
{
	struct kvm_run *run = atomic_load(&running_vcpu);

	(void)signal_number;
	run_timed_out = 1;
	/* Also cover an alarm delivered just before entering KVM_RUN. */
	if (run != NULL)
		run->immediate_exit = 1;
}

static void set_error(char *error, size_t error_size, const char *format, ...)
{
	va_list arguments;

	if (error == NULL || error_size == 0)
		return;
	va_start(arguments, format);
	vsnprintf(error, error_size, format, arguments);
	va_end(arguments);
}

void glt_arm64_vm_destroy(struct glt_arm64_vm *vm)
{
	if (vm == NULL)
		return;
	if (vm->run != MAP_FAILED) {
		munmap(vm->run, vm->run_size);
		vm->run = MAP_FAILED;
		vm->run_size = 0;
	}
	if (vm->vcpu_fd >= 0) {
		close(vm->vcpu_fd);
		vm->vcpu_fd = -1;
	}
	if (vm->vm_fd >= 0) {
		close(vm->vm_fd);
		vm->vm_fd = -1;
	}
	if (vm->kvm_fd >= 0) {
		close(vm->kvm_fd);
		vm->kvm_fd = -1;
	}
}

int glt_arm64_vm_create(struct glt_arm64_vm *vm,
			const struct glt_kvm_memslot *memslots,
			size_t memslot_count,
			char *error, size_t error_size)
{
	struct kvm_vcpu_init vcpu_init = {0};
	size_t index;
	int run_size;

	if (vm == NULL || error == NULL || error_size == 0 ||
	    (memslot_count != 0 && memslots == NULL))
		return -1;
	*vm = (struct glt_arm64_vm) {
		.kvm_fd = -1,
		.vm_fd = -1,
		.vcpu_fd = -1,
		.run = MAP_FAILED,
	};
	error[0] = '\0';

	vm->kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
	if (vm->kvm_fd < 0) {
		set_error(error, error_size, "failed to open /dev/kvm: %s",
			  strerror(errno));
		goto failure;
	}
	if (ioctl(vm->kvm_fd, KVM_GET_API_VERSION, 0) != KVM_API_VERSION) {
		set_error(error, error_size, "unexpected KVM API version");
		goto failure;
	}

	vm->vm_fd = ioctl(vm->kvm_fd, KVM_CREATE_VM, 0);
	if (vm->vm_fd < 0) {
		set_error(error, error_size, "KVM_CREATE_VM failed: %s",
			  strerror(errno));
		goto failure;
	}
	for (index = 0; index < memslot_count; index++) {
		const struct glt_kvm_memslot *memslot = &memslots[index];
		struct kvm_userspace_memory_region region = {
			.slot = memslot->slot,
			.flags = memslot->flags,
			.guest_phys_addr = memslot->guest_phys_addr,
			.memory_size = memslot->memory_size,
			.userspace_addr =
				(uint64_t)(uintptr_t)memslot->userspace_addr,
		};

		if (memslot->userspace_addr == NULL || memslot->memory_size == 0) {
			set_error(error, error_size,
				  "invalid KVM memslot %u", memslot->slot);
			goto failure;
		}
		if (ioctl(vm->vm_fd, KVM_SET_USER_MEMORY_REGION, &region) != 0) {
			set_error(error, error_size,
				  "KVM_SET_USER_MEMORY_REGION slot %u failed: %s",
				  memslot->slot, strerror(errno));
			goto failure;
		}
	}

	vm->vcpu_fd = ioctl(vm->vm_fd, KVM_CREATE_VCPU, 0);
	if (vm->vcpu_fd < 0) {
		set_error(error, error_size, "KVM_CREATE_VCPU failed: %s",
			  strerror(errno));
		goto failure;
	}
	if (ioctl(vm->vm_fd, KVM_ARM_PREFERRED_TARGET, &vcpu_init) != 0 ||
	    ioctl(vm->vcpu_fd, KVM_ARM_VCPU_INIT, &vcpu_init) != 0) {
		set_error(error, error_size, "KVM_ARM_VCPU_INIT failed: %s",
			  strerror(errno));
		goto failure;
	}

	run_size = ioctl(vm->kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
	if (run_size < (int)sizeof(*vm->run)) {
		set_error(error, error_size,
			  "invalid KVM vCPU mmap size %d", run_size);
		goto failure;
	}
	vm->run_size = (size_t)run_size;
	vm->run = mmap(NULL, vm->run_size, PROT_READ | PROT_WRITE,
		       MAP_SHARED, vm->vcpu_fd, 0);
	if (vm->run == MAP_FAILED) {
		set_error(error, error_size, "failed to mmap kvm_run: %s",
			  strerror(errno));
		goto failure;
	}
	return 0;

failure:
	glt_arm64_vm_destroy(vm);
	return -1;
}

int glt_arm64_vm_set_reg(struct glt_arm64_vm *vm, uint64_t id,
			 uint64_t value, char *error, size_t error_size)
{
	struct kvm_one_reg register_value = {
		.id = id,
		.addr = (uint64_t)(uintptr_t)&value,
	};

	if (vm == NULL || vm->vcpu_fd < 0) {
		set_error(error, error_size, "KVM vCPU is not initialized");
		return -1;
	}
	if (ioctl(vm->vcpu_fd, KVM_SET_ONE_REG, &register_value) != 0) {
		set_error(error, error_size, "KVM_SET_ONE_REG failed: %s",
			  strerror(errno));
		return -1;
	}
	return 0;
}

int glt_arm64_vm_run(struct glt_arm64_vm *vm, unsigned int timeout_seconds,
		     char *error, size_t error_size)
{
	struct sigaction action = {0};
	struct sigaction previous_action;
	struct timespec started, finished;
	unsigned int previous_alarm;
	time_t elapsed_seconds;
	int run_result = -1;

	if (vm == NULL || vm->vcpu_fd < 0 || vm->run == MAP_FAILED) {
		set_error(error, error_size, "KVM vCPU is not initialized");
		return -1;
	}
	if (timeout_seconds == 0) {
		set_error(error, error_size, "KVM_RUN timeout must be non-zero");
		return -1;
	}
	previous_alarm = alarm(0);
	if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) {
		set_error(error, error_size, "clock_gettime failed: %s",
			  strerror(errno));
		if (previous_alarm != 0)
			alarm(previous_alarm);
		return -1;
	}

	action.sa_handler = alarm_handler;
	sigemptyset(&action.sa_mask);
	if (sigaction(SIGALRM, &action, &previous_action) != 0) {
		set_error(error, error_size, "sigaction(SIGALRM) failed: %s",
			  strerror(errno));
		if (previous_alarm != 0)
			alarm(previous_alarm);
		return -1;
	}
	run_timed_out = 0;
	vm->run->immediate_exit = 0;
	atomic_store(&running_vcpu, vm->run);
	/* One deadline covers all EINTR and KVM_EXIT_INTR retries. */
	alarm(timeout_seconds);

	for (;;) {
		if (run_timed_out)
			break;
		run_result = ioctl(vm->vcpu_fd, KVM_RUN, 0);
		if (run_timed_out)
			break;
		if (run_result != 0) {
			if (errno == EINTR)
				continue;
			set_error(error, error_size, "KVM_RUN failed: %s",
				  strerror(errno));
			run_result = -1;
			break;
		}
		if (vm->run->exit_reason == KVM_EXIT_INTR)
			continue;
		run_result = 0;
		break;
	}
	if (run_timed_out) {
		set_error(error, error_size,
			  "KVM_RUN timed out after %u seconds", timeout_seconds);
		run_result = -1;
	}

	alarm(0);
	atomic_store(&running_vcpu, NULL);
	vm->run->immediate_exit = 0;
	if (sigaction(SIGALRM, &previous_action, NULL) != 0) {
		if (run_result == 0)
			set_error(error, error_size,
				  "failed to restore SIGALRM handler: %s",
				  strerror(errno));
		return -1;
	}
	if (previous_alarm != 0) {
		if (clock_gettime(CLOCK_MONOTONIC, &finished) != 0) {
			alarm(previous_alarm);
			set_error(error, error_size, "clock_gettime failed: %s",
				  strerror(errno));
			return -1;
		}
		/* alarm() has whole-second precision; round remaining time up. */
		elapsed_seconds = finished.tv_sec - started.tv_sec -
			(finished.tv_nsec < started.tv_nsec);
		if ((uint64_t)elapsed_seconds < previous_alarm)
			alarm(previous_alarm - (unsigned int)elapsed_seconds);
		else
			raise(SIGALRM);
	}
	return run_result;
}
