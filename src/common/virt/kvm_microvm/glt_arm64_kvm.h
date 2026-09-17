// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef GLT_ARM64_KVM_H
#define GLT_ARM64_KVM_H

#include <linux/kvm.h>
#include <stddef.h>
#include <stdint.h>

#define GLT_ARM64_CORE_REG(name) \
	(KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | \
	 KVM_REG_ARM_CORE_REG(name))

struct glt_kvm_memslot {
	uint32_t slot;
	uint32_t flags;
	uint64_t guest_phys_addr;
	uint64_t memory_size;
	void *userspace_addr;
};

struct glt_arm64_vm {
	int kvm_fd;
	int vm_fd;
	int vcpu_fd;
	struct kvm_run *run;
	size_t run_size;
};

/*
 * Create one Arm64 VM with one vCPU and register all caller-owned memslots.
 * On failure, all resources acquired by this function are released.
 */
int glt_arm64_vm_create(struct glt_arm64_vm *vm,
			const struct glt_kvm_memslot *memslots,
			size_t memslot_count,
			char *error, size_t error_size);

int glt_arm64_vm_set_reg(struct glt_arm64_vm *vm, uint64_t id,
			 uint64_t value, char *error, size_t error_size);

/*
 * Run until KVM reports a non-KVM_EXIT_INTR userspace exit or timeout.
 * One timeout covers all retries. Any caller alarm is restored with the
 * remaining time (rounded up to whole seconds), or delivered if overdue.
 * This temporarily owns process-wide SIGALRM and is not a concurrent-vCPU API.
 */
int glt_arm64_vm_run(struct glt_arm64_vm *vm, unsigned int timeout_seconds,
		     char *error, size_t error_size);

/* Safe after partial creation and safe to call more than once. */
void glt_arm64_vm_destroy(struct glt_arm64_vm *vm);

#endif /* GLT_ARM64_KVM_H */
