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
 *   ./test_ls64 kvm            - Test the KVM LS64 userspace exit
 *
 * Exit codes:
 *   0 - Test completed successfully (no signal)
 *   1 - Generic error (usage, allocation failure, etc.)
 *   2 - SIGBUS received (expected for normal memory without FEAT_LS64_WB)
 *   3 - SIGILL received (CPU doesn't support LS64)
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/kvm.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <sys/auxv.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

/* HWCAP3 definitions */
#ifndef AT_HWCAP3
#define AT_HWCAP3 29
#endif

#ifndef HWCAP3_LS64
#define HWCAP3_LS64 (1UL << 3)
#endif

#ifndef KVM_EXIT_ARM_LDST64B
#define KVM_EXIT_ARM_LDST64B 42
#endif

#define GUEST_CODE_GPA 0x100000ULL
#define GUEST_MMIO_GPA 0x200000ULL
#define LD64B_X0_X8 0xf83fd100U
#define KVM_LS64_PATCHES "f174a9ffcd48 and 151b92c92a45"

#define ARM64_CORE_REG(name) \
	(KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | \
	 KVM_REG_ARM_CORE_REG(name))

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

static void sigalrm_handler(int sig __attribute__((unused)))
{
	static const char message[] = "KVM_RUN timed out\n";
	ssize_t ignored;

	ignored = write(STDERR_FILENO, message, sizeof(message) - 1);
	(void)ignored;
	_exit(1);
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

static int set_kvm_reg(int vcpu_fd, uint64_t id, uint64_t value)
{
	struct kvm_one_reg reg = {
		.id = id,
		.addr = (uint64_t)&value,
	};

	return ioctl(vcpu_fd, KVM_SET_ONE_REG, &reg);
}

/*
 * Run one LD64B in an EL1 guest against an IPA outside all memslots.
 * A kernel with the KVM LS64 support must return KVM_EXIT_ARM_LDST64B.
 */
static int test_ls64_kvm_exit(void)
{
	struct kvm_userspace_memory_region region = {
		.slot = 0,
		.guest_phys_addr = GUEST_CODE_GPA,
	};
	struct kvm_vcpu_init init = {0};
	struct kvm_run *run;
	long page_size;
	int kvm_fd;
	int vm_fd;
	int vcpu_fd;
	int run_size;
	int run_result;
	void *guest_mem;
	int result = EXIT_FAILURE;

	page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0) {
		perror("sysconf(_SC_PAGESIZE)");
		return EXIT_FAILURE;
	}

	kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
	if (kvm_fd < 0) {
		perror("open(/dev/kvm)");
		return EXIT_FAILURE;
	}

	if (ioctl(kvm_fd, KVM_GET_API_VERSION, 0) != KVM_API_VERSION) {
		fprintf(stderr, "Unexpected KVM API version\n");
		goto close_kvm;
	}

	vm_fd = ioctl(kvm_fd, KVM_CREATE_VM, 0);
	if (vm_fd < 0) {
		perror("KVM_CREATE_VM");
		goto close_kvm;
	}

	guest_mem = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (guest_mem == MAP_FAILED) {
		perror("mmap(guest memory)");
		goto close_vm;
	}

	*(uint32_t *)guest_mem = LD64B_X0_X8;
	__builtin___clear_cache(guest_mem, (char *)guest_mem + sizeof(uint32_t));
	region.memory_size = page_size;
	region.userspace_addr = (uint64_t)guest_mem;
	if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
		perror("KVM_SET_USER_MEMORY_REGION");
		goto unmap_guest;
	}

	vcpu_fd = ioctl(vm_fd, KVM_CREATE_VCPU, 0);
	if (vcpu_fd < 0) {
		perror("KVM_CREATE_VCPU");
		goto unmap_guest;
	}

	if (ioctl(vm_fd, KVM_ARM_PREFERRED_TARGET, &init) < 0) {
		perror("KVM_ARM_PREFERRED_TARGET");
		goto close_vcpu;
	}
	if (ioctl(vcpu_fd, KVM_ARM_VCPU_INIT, &init) < 0) {
		perror("KVM_ARM_VCPU_INIT");
		goto close_vcpu;
	}

	run_size = ioctl(kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
	if (run_size < 0) {
		perror("KVM_GET_VCPU_MMAP_SIZE");
		goto close_vcpu;
	}
	if (run_size < (int)sizeof(*run)) {
		fprintf(stderr, "Invalid KVM vCPU mmap size: %d\n", run_size);
		goto close_vcpu;
	}
	run = mmap(NULL, run_size, PROT_READ | PROT_WRITE, MAP_SHARED,
		   vcpu_fd, 0);
	if (run == MAP_FAILED) {
		perror("mmap(kvm_run)");
		goto close_vcpu;
	}

	if (set_kvm_reg(vcpu_fd, ARM64_CORE_REG(regs.pc), GUEST_CODE_GPA) < 0) {
		perror("KVM_SET_ONE_REG(pc)");
		goto unmap_run;
	}
	if (set_kvm_reg(vcpu_fd, ARM64_CORE_REG(regs.regs[8]),
			GUEST_MMIO_GPA) < 0) {
		perror("KVM_SET_ONE_REG(x8)");
		goto unmap_run;
	}

	if (signal(SIGALRM, sigalrm_handler) == SIG_ERR) {
		perror("signal(SIGALRM)");
		goto unmap_run;
	}
	alarm(5);
	run_result = ioctl(vcpu_fd, KVM_RUN, 0);
	alarm(0);
	if (run_result < 0) {
		perror("KVM_RUN");
		goto unmap_run;
	}

	printf("KVM exit reason: %u\n", run->exit_reason);
	if (run->exit_reason != KVM_EXIT_ARM_LDST64B) {
		fprintf(stderr,
			"Expected KVM_EXIT_ARM_LDST64B (%d); check kernel patches "
			KVM_LS64_PATCHES "\n",
			KVM_EXIT_ARM_LDST64B);
		goto unmap_run;
	}

	printf("KVM LS64 fault IPA: 0x%llx\n",
	       (unsigned long long)run->arm_nisv.fault_ipa);
	if (run->arm_nisv.fault_ipa != GUEST_MMIO_GPA) {
		fprintf(stderr, "Unexpected LS64 fault IPA\n");
		goto unmap_run;
	}

	result = EXIT_SUCCESS;

unmap_run:
	munmap(run, run_size);
close_vcpu:
	close(vcpu_fd);
unmap_guest:
	munmap(guest_mem, page_size);
close_vm:
	close(vm_fd);
close_kvm:
	close(kvm_fd);
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
		fprintf(stderr, "Usage: %s <normal_memory|alignment|kvm>\n", argv[0]);
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
	} else if (!strcmp(argv[1], "kvm")) {
		return test_ls64_kvm_exit();
	} else {
		fprintf(stderr, "Unknown test: %s\n", argv[1]);
		fprintf(stderr, "Usage: %s <normal_memory|alignment|kvm>\n", argv[0]);
		exit(1);
	}

	/* If we get here, test completed without signal */
	return 0;
}
