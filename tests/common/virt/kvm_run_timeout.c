// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* Deterministic timeout tests: no real KVM, timers, signals, or sleeping. */
#include <assert.h>
#include <linux/kvm.h>

#ifndef __aarch64__
/* Only VM creation uses this Arm UAPI type; creation is never called here. */
struct kvm_vcpu_init {
	unsigned int target;
	unsigned int features[7];
};
#define __aarch64__ 1
#endif

#include "../../../src/common/virt/kvm_microvm/glt_arm64_kvm.c"

static struct sigaction current_action;
static struct kvm_run vcpu_run;
static const char *scenario;
static time_t now, alarm_due;
static unsigned int run_calls, caller_alarm_calls;
static bool immediate_exit_seen;

static void caller_alarm(int signal_number)
{
	assert(signal_number == SIGALRM);
	caller_alarm_calls++;
}

unsigned int __wrap_alarm(unsigned int seconds)
{
	unsigned int remaining = alarm_due > now ? alarm_due - now : 0;

	alarm_due = seconds != 0 ? now + seconds : 0;
	return remaining;
}

int __wrap_clock_gettime(clockid_t clock_id, struct timespec *value)
{
	assert(clock_id == CLOCK_MONOTONIC);
	if (strcmp(scenario, "clock_error") == 0) {
		errno = EINVAL;
		return -1;
	}
	*value = (struct timespec) {.tv_sec = now};
	if (strcmp(scenario, "caller_alarm_setup_delay") == 0 && now == 0)
		now += 2;
	return 0;
}

int __wrap_sigaction(int signal_number, const struct sigaction *action,
		     struct sigaction *previous)
{
	assert(signal_number == SIGALRM);
	if (previous != NULL)
		*previous = current_action;
	if (action != NULL)
		current_action = *action;
	return 0;
}

int __wrap_raise(int signal_number)
{
	current_action.sa_handler(signal_number);
	return 0;
}

int __wrap_ioctl(int fd, unsigned long request, ...)
{
	assert(fd == 4 && request == KVM_RUN);
	run_calls++;
	now += strcmp(scenario, "entry_alarm") == 0 ? 3 : 1;
	if (strcmp(scenario, "expired_alarm") == 0)
		now++;
	if (alarm_due != 0 && now >= alarm_due) {
		alarm_due = 0;
		current_action.sa_handler(SIGALRM);
	}
	if (strcmp(scenario, "entry_alarm") == 0) {
		/* SIGALRM arrives after the loop's check but before KVM entry. */
		immediate_exit_seen = vcpu_run.immediate_exit != 0;
		errno = EINTR;
		return -1;
	}
	/* Bound the old implementation, which restarts its timer every retry. */
	if (run_calls > 6 || strcmp(scenario, "ioctl_error") == 0) {
		errno = EIO;
		return -1;
	}
	if (strcmp(scenario, "eintr") == 0) {
		errno = EINTR;
		return -1;
	}
	vcpu_run.exit_reason = strcmp(scenario, "exit_intr") == 0 ?
		KVM_EXIT_INTR : KVM_EXIT_MMIO;
	return 0;
}

int main(int argc, char **argv)
{
	struct glt_arm64_vm vm = {.vcpu_fd = 4, .run = &vcpu_run};
	char error[256] = {0};
	int result;
	bool has_caller_alarm;

	assert(argc == 2);
	scenario = argv[1];
	current_action.sa_handler = caller_alarm;
	has_caller_alarm = strcmp(scenario, "caller_alarm") == 0 ||
		strcmp(scenario, "caller_alarm_setup_delay") == 0 ||
		strcmp(scenario, "clock_error") == 0;
	if (has_caller_alarm)
		alarm_due = 10;
	if (strcmp(scenario, "expired_alarm") == 0)
		alarm_due = 1;
	result = glt_arm64_vm_run(&vm,
		strcmp(scenario, "zero_timeout") == 0 ? 0 : 3,
		error, sizeof(error));

	if (strcmp(scenario, "eintr") == 0 ||
	    strcmp(scenario, "exit_intr") == 0 ||
	    strcmp(scenario, "entry_alarm") == 0) {
		assert(result == -1 && now == 3);
		assert(strstr(error, "timed out after 3 seconds") != NULL);
		if (strcmp(scenario, "entry_alarm") == 0)
			assert(immediate_exit_seen);
	} else if (strcmp(scenario, "zero_timeout") == 0) {
		assert(result == -1 && run_calls == 0);
		assert(strstr(error, "must be non-zero") != NULL);
	} else if (strcmp(scenario, "ioctl_error") == 0) {
		assert(result == -1 && run_calls == 1);
		assert(strstr(error, strerror(EIO)) != NULL);
	} else if (strcmp(scenario, "clock_error") == 0) {
		assert(result == -1 && run_calls == 0);
		assert(strstr(error, "clock_gettime failed") != NULL);
	} else {
		assert(result == 0 && run_calls == 1);
	}
	assert(current_action.sa_handler == caller_alarm);
	assert(vcpu_run.immediate_exit == 0);
	assert(atomic_load(&running_vcpu) == NULL);
	assert(alarm_due == (has_caller_alarm ? 10 : 0));
	assert(caller_alarm_calls ==
		(strcmp(scenario, "expired_alarm") == 0 ? 1U : 0U));
	return 0;
}
