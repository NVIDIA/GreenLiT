// SPDX-License-Identifier: GPL-2.0
/*
 * Test program for ARM64 FEAT_LS64 instructions (LD64B/ST64B)
 * 
 * This program tests the ld64b and st64b instructions which perform
 * 64-byte atomic loads and stores.
 *
 * Usage:
 *   ./test_ls64 normal_memory  - Test LD64B/ST64B on normal memory
 *   ./test_ls64 alignment      - Test alignment requirements
 *
 * Exit codes:
 *   0 - Test completed successfully (no signal)
 *   1 - Generic error (usage, allocation failure, etc.)
 *   2 - SIGBUS received (expected for normal memory without FEAT_LS64_WB)
 *   3 - SIGILL received (CPU doesn't support LS64)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <sys/auxv.h>
#include <unistd.h>

/* HWCAP3 definitions */
#ifndef AT_HWCAP3
#define AT_HWCAP3 29
#endif

#ifndef HWCAP3_LS64
#define HWCAP3_LS64 (1UL << 3)
#endif

/* Test data structure - must be 64-byte aligned for LD64B/ST64B */
struct ls64_data {
	uint64_t data[8];  /* 8 x 8 bytes = 64 bytes */
} __attribute__((aligned(64)));

/*
 * Signal handler for SIGBUS
 * LD64B/ST64B on unsupported memory will trigger SIGBUS
 */
static void sigbus_handler(int sig __attribute__((unused)),
			   siginfo_t *info __attribute__((unused)),
			   void *context __attribute__((unused)))
{
	exit(2);
}

/*
 * Signal handler for SIGILL
 * If the CPU doesn't support LS64, we'll get SIGILL
 */
static void sigill_handler(int sig __attribute__((unused)),
			   siginfo_t *info __attribute__((unused)),
			   void *context __attribute__((unused)))
{
	exit(3);
}

/*
 * Install signal handlers
 */
static void setup_signal_handlers(void)
{
	struct sigaction sa;

	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = sigbus_handler;
	sa.sa_flags = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);

	if (sigaction(SIGBUS, &sa, NULL) < 0) {
		perror("sigaction(SIGBUS)");
		exit(1);
	}

	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = sigill_handler;
	sa.sa_flags = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);

	if (sigaction(SIGILL, &sa, NULL) < 0) {
		perror("sigaction(SIGILL)");
		exit(1);
	}
}

/*
 * Wrapper for LD64B instruction
 * LD64B loads 64 bytes from [src] into registers x0-x7
 */
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

/*
 * Wrapper for ST64B instruction
 * ST64B stores 64 bytes from registers x0-x7 to [dest]
 */
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
static void test_ls64_alignment(void)
{
	/* Allocate buffer with extra space for alignment testing */
	char *buffer = aligned_alloc(64, 256);
	struct ls64_data src __attribute__((aligned(64)));
	int i;

	if (!buffer) {
		fprintf(stderr, "Allocation failed\n");
		exit(1);
	}

	/* Initialize source */
	for (i = 0; i < 8; i++) {
		src.data[i] = 0xAABBCCDD00000000ULL + i;
	}

	/* Test with properly aligned address */
	do_st64b(buffer, &src);

	/* Test with misaligned address (should fail) */
	do_st64b(buffer + 8, &src);

	free(buffer);
}

/*
 * Check and report HWCAP3_LS64 status
 */
static void check_hwcap3_ls64(void)
{
	unsigned long hwcap3 = getauxval(AT_HWCAP3);

	printf("HWCAP3: 0x%lx\n", hwcap3);
	printf("HWCAP3_LS64: %s\n",
	       (hwcap3 & HWCAP3_LS64) ? "SET" : "NOT SET");
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <normal_memory|alignment>\n", argv[0]);
		exit(1);
	}

	/* Check HWCAP3_LS64 */
	check_hwcap3_ls64();

	/* Setup signal handlers to catch SIGBUS/SIGILL */
	setup_signal_handlers();

	/* Run the requested test */
	if (!strcmp(argv[1], "normal_memory")) {
		test_ls64_on_normal_memory();
	} else if (!strcmp(argv[1], "alignment")) {
		test_ls64_alignment();
	} else {
		fprintf(stderr, "Unknown test: %s\n", argv[1]);
		fprintf(stderr, "Usage: %s <normal_memory|alignment>\n", argv[0]);
		exit(1);
	}

	/* If we get here, test completed without signal */
	return 0;
}
