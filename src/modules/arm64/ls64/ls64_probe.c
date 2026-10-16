// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Test program for ARM64 FEAT_LS64 instructions (LD64B/ST64B)
 *
 * This program tests the ld64b and st64b instructions which perform
 * 64-byte atomic loads and stores.
 *
 * Usage:
 *   ./ls64_probe normal-memory - Test LD64B/ST64B on normal memory
 *   ./ls64_probe misaligned    - Test a misaligned ST64B
 *   ./ls64_probe kvm-exit      - Test the KVM LS64 userspace exit
 *
 * Exit codes:
 *   0 - Test completed successfully (no signal)
 *   1 - Generic error (usage, allocation failure, etc.)
 *   2 - SIGBUS received (expected for normal memory without FEAT_LS64_WB)
 *   3 - SIGILL received (CPU doesn't support LS64)
 */

#define _GNU_SOURCE
#include "glt_arm64_kvm.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef KVM_EXIT_ARM_LDST64B
#define KVM_EXIT_ARM_LDST64B 42
#endif

#define GUEST_CODE_GPA 0x100000ULL
#define GUEST_MMIO_GPA 0x200000ULL
#define LD64B_X0_X8 0xf83fd100U
#define KVM_LS64_PATCHES "f174a9ffcd48 and 151b92c92a45"

/* Test data structure - must be 64-byte aligned for LD64B/ST64B */
struct ls64_data {
	uint64_t data[8];  /* 8 x 8 bytes = 64 bytes */
} __attribute__((aligned(64)));

static void signal_handler(int signal_number)
{
	_Exit(signal_number == SIGBUS ? 2 : 3);
}

/* Execute LD64B, which loads 64 bytes from [src] into registers x0-x7. */
static void do_ld64b(const void *src, struct ls64_data *dest)
{
	register const void *xn asm("x8") = src;
	register uint64_t x0 asm("x0");
	register uint64_t x1 asm("x1");
	register uint64_t x2 asm("x2");
	register uint64_t x3 asm("x3");
	register uint64_t x4 asm("x4");
	register uint64_t x5 asm("x5");
	register uint64_t x6 asm("x6");
	register uint64_t x7 asm("x7");

	/* LD64B x0, [x8] - loads 64 bytes into x0-x7 */
	asm volatile(".inst 0xf83fd100"
		     : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3),
		       "=r"(x4), "=r"(x5), "=r"(x6), "=r"(x7)
		     : "r"(xn)
		     : "memory");

	dest->data[0] = x0;
	dest->data[1] = x1;
	dest->data[2] = x2;
	dest->data[3] = x3;
	dest->data[4] = x4;
	dest->data[5] = x5;
	dest->data[6] = x6;
	dest->data[7] = x7;
}

/* Execute ST64B, which stores registers x0-x7 to 64 bytes at [dest]. */
static void do_st64b(void *dest, const struct ls64_data *src)
{
	register void *xn asm("x8") = dest;
	register uint64_t x0 asm("x0") = src->data[0];
	register uint64_t x1 asm("x1") = src->data[1];
	register uint64_t x2 asm("x2") = src->data[2];
	register uint64_t x3 asm("x3") = src->data[3];
	register uint64_t x4 asm("x4") = src->data[4];
	register uint64_t x5 asm("x5") = src->data[5];
	register uint64_t x6 asm("x6") = src->data[6];
	register uint64_t x7 asm("x7") = src->data[7];

	/* ST64B x0, [x8] - stores x0-x7 (64 bytes) to memory */
	asm volatile(".inst 0xf83f9100"
		     :
		     : "r"(xn), "r"(x0), "r"(x1), "r"(x2), "r"(x3),
		       "r"(x4), "r"(x5), "r"(x6), "r"(x7)
		     : "memory");
}

/*
 * Run one LD64B in an EL1 guest against an IPA outside all memslots.
 * A kernel with the KVM LS64 support must return KVM_EXIT_ARM_LDST64B.
 */
static int test_ls64_kvm_exit(void)
{
	struct glt_arm64_vm vm;
	struct glt_kvm_memslot memslot = {
		.slot = 0,
		.guest_phys_addr = GUEST_CODE_GPA,
	};
	char error[256] = "";
	long page_size;
	void *guest_mem;
	int result = EXIT_FAILURE;

	page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0) {
		perror("sysconf(_SC_PAGESIZE)");
		return EXIT_FAILURE;
	}

	guest_mem = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (guest_mem == MAP_FAILED) {
		perror("mmap(guest memory)");
		return EXIT_FAILURE;
	}

	*(uint32_t *)guest_mem = LD64B_X0_X8;
	__builtin___clear_cache(guest_mem, (char *)guest_mem + sizeof(uint32_t));
	memslot.memory_size = (uint64_t)page_size;
	memslot.userspace_addr = guest_mem;
	if (glt_arm64_vm_create(&vm, &memslot, 1,
				error, sizeof(error)) != 0 ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.pc),
				 GUEST_CODE_GPA, error, sizeof(error)) != 0 ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.regs[8]),
				 GUEST_MMIO_GPA, error, sizeof(error)) != 0 ||
	    glt_arm64_vm_run(&vm, 5, error, sizeof(error)) != 0) {
		fprintf(stderr, "%s\n", error);
		goto cleanup;
	}

	printf("KVM exit reason: %u\n", vm.run->exit_reason);
	if (vm.run->exit_reason != KVM_EXIT_ARM_LDST64B) {
		fprintf(stderr,
			"Expected KVM_EXIT_ARM_LDST64B (%d); check kernel patches "
			KVM_LS64_PATCHES "\n",
			KVM_EXIT_ARM_LDST64B);
		goto cleanup;
	}

	printf("KVM LS64 fault IPA: 0x%llx\n",
	       (unsigned long long)vm.run->arm_nisv.fault_ipa);
	if (vm.run->arm_nisv.fault_ipa != GUEST_MMIO_GPA) {
		fprintf(stderr, "Unexpected LS64 fault IPA\n");
		goto cleanup;
	}

	result = EXIT_SUCCESS;

cleanup:
	glt_arm64_vm_destroy(&vm);
	munmap(guest_mem, page_size);
	return result;
}

/*
 * Test LD64B/ST64B on normal memory
 * This should trigger SIGBUS unless FEAT_LS64_WB is supported
 */
static void test_ls64_on_normal_memory(void)
{
	struct ls64_data src __attribute__((aligned(64)));
	struct ls64_data dest __attribute__((aligned(64)));
	int i;

	/* Initialize source data */
	for (i = 0; i < 8; i++) {
		src.data[i] = 0x1122334455667700ULL + i;
	}
	memset(&dest, 0, sizeof(dest));

	/* Test ST64B - store to normal memory */
	do_st64b(&dest, &src);

	/* Test LD64B - load from normal memory */
	do_ld64b(&src, &dest);
}

/*
 * Test alignment requirements
 * LD64B/ST64B require 64-byte alignment
 */
static void test_ls64_misaligned(void)
{
	char *buffer = aligned_alloc(64, 256);
	struct ls64_data src __attribute__((aligned(64)));
	int i;

	if (!buffer) {
		fprintf(stderr, "Allocation failed\n");
		exit(1);
	}

	for (i = 0; i < 8; i++) {
		src.data[i] = 0xAABBCCDD00000000ULL + i;
	}

	do_st64b(buffer + 8, &src);

	free(buffer);
}

int main(int argc, char **argv)
{
	struct sigaction action = {
		.sa_handler = signal_handler,
	};

	if (argc < 2) {
		fprintf(stderr,
			"Usage: %s <normal-memory|misaligned|kvm-exit>\n",
			argv[0]);
		return EXIT_FAILURE;
	}

	sigemptyset(&action.sa_mask);
	if (sigaction(SIGBUS, &action, NULL) < 0 ||
	    sigaction(SIGILL, &action, NULL) < 0) {
		perror("sigaction(SIGBUS/SIGILL)");
		return EXIT_FAILURE;
	}

	/* Run the requested test */
	if (!strcmp(argv[1], "normal-memory")) {
		test_ls64_on_normal_memory();
	} else if (!strcmp(argv[1], "misaligned")) {
		test_ls64_misaligned();
	} else if (!strcmp(argv[1], "kvm-exit")) {
		return test_ls64_kvm_exit();
	} else {
		fprintf(stderr, "Unknown test: %s\n", argv[1]);
		fprintf(stderr,
			"Usage: %s <normal-memory|misaligned|kvm-exit>\n",
			argv[0]);
		return EXIT_FAILURE;
	}

	/* If we get here, test completed without signal */
	return 0;
}
