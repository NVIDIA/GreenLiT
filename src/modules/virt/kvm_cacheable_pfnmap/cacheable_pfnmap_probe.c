// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* A real instruction fetch from nvgrace GPU USEMEM, not anonymous RAM.
 * Exit: 0 = verified; 1 = mapping/ABI failure; 2 = probe/cleanup error;
 * 3 = device prerequisite unavailable; 4 = host prerequisite unavailable.
 * Cleanup failure overrides all other outcomes.
 * The caller must reserve the entire GPU and its group for this test.
 */
#define _GNU_SOURCE
#include "glt_arm64_kvm.h"

#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/vfio.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CACHEABLE_PFNMAP_CAP 243
#define CODE_GPA UINT64_C(0x40000000)
#define VECTOR_GPA UINT64_C(0x80000000)
#define MMIO_GPA UINT64_C(0x10000000)
#define MARKER UINT64_C(0x43414348)
#define REGION_BUFFER_SIZE 65536
#define MAPPING_PATCH "https://git.kernel.org/torvalds/c/0c67288e0c8b"
#define ABI_PATCH "https://git.kernel.org/torvalds/c/f55ce5a6cd33"

/* VFIO resets the GPU on open, after the driver's probe-time readiness check.
 * Use the same documented register layout to check the hardware again before
 * touching coherent memory. All reads go through the owned VFIO device FD.
 */
static bool device_ready(int device)
{
	struct vfio_region_info bar = {
		.argsz = sizeof(bar), .index = VFIO_PCI_BAR0_REGION_INDEX,
	};
	struct timespec start, now, pause = { .tv_nsec = 50000000 };
	uint32_t c2c, hbm;

	if (ioctl(device, VFIO_DEVICE_GET_REGION_INFO, &bar) ||
	    !(bar.flags & VFIO_REGION_INFO_FLAG_READ) || bar.size < 0x200c0 ||
	    bar.offset > INT64_MAX - 0x200c0 ||
	    clock_gettime(CLOCK_MONOTONIC, &start))
		return false;
	do {
		if (pread(device, &c2c, sizeof(c2c), bar.offset + 0x1498) != sizeof(c2c) ||
		    pread(device, &hbm, sizeof(hbm), bar.offset + 0x200bc) != sizeof(hbm))
			return false;
		if (le32toh(c2c) == 0xff && le32toh(hbm) == 0xff) {
			puts("STIMULUS: C2C link and HBM training ready after VFIO open");
			return true;
		}
		nanosleep(&pause, NULL);
		if (clock_gettime(CLOCK_MONOTONIC, &now))
			return false;
	} while (now.tv_sec - start.tv_sec < 30);
	fprintf(stderr, "SETUP: GPU not ready: C2C=%#x HBM=%#x\n",
		le32toh(c2c), le32toh(hbm));
	return false;
}

/* Require the first page to belong to a real sparse mmap area, not BAR padding.
 * Bounds and increasing offsets also reject malformed/cyclic capability chains.
 */
static bool region_has_page(const struct vfio_region_info *info, size_t page)
{
	uint32_t offset = info->cap_offset;
	bool found = false;

	if (info->argsz > REGION_BUFFER_SIZE || info->argsz < sizeof(*info) ||
	    info->size < page || info->offset % page ||
	    (info->flags & (VFIO_REGION_INFO_FLAG_READ |
			   VFIO_REGION_INFO_FLAG_WRITE |
			   VFIO_REGION_INFO_FLAG_MMAP)) !=
	    (VFIO_REGION_INFO_FLAG_READ | VFIO_REGION_INFO_FLAG_WRITE |
	     VFIO_REGION_INFO_FLAG_MMAP))
		return false;
	if (!(info->flags & VFIO_REGION_INFO_FLAG_CAPS) || !offset)
		return false;
	while (offset) {
		const struct vfio_info_cap_header *header;
		const struct vfio_region_info_cap_sparse_mmap *sparse;
		size_t base = sizeof(*sparse);

		if (offset < sizeof(*info) || offset % 4 ||
		    offset > info->argsz - sizeof(*header))
			return false;
		header = (const void *)((const char *)info + offset);
		if (header->id == VFIO_REGION_INFO_CAP_SPARSE_MMAP) {
			if (header->version != 1 || info->argsz - offset < base)
				return false;
			sparse = (const void *)header;
			if (sparse->nr_areas > (info->argsz - offset - base) /
			    sizeof(sparse->areas[0]))
				return false;
			for (uint32_t i = 0; i < sparse->nr_areas; i++) {
				if (!sparse->areas[i].offset &&
				    sparse->areas[i].size >= page &&
				    sparse->areas[i].size <= info->size)
					found = true;
			}
		}
		if (header->next && header->next <= offset)
			return false;
		offset = header->next;
	}
	return found;
}

static bool mapping_is_pfnmap(void *mapping)
{
	FILE *stream = fopen("/proc/self/smaps", "re");
	char *line = NULL;
	size_t size = 0;
	bool selected = false, found = false;
	unsigned long start, end;

	if (!stream)
		return false;
	while (getline(&line, &size, stream) != -1) {
		if (sscanf(line, "%lx-%lx", &start, &end) == 2)
			selected = (uintptr_t)mapping >= start &&
				   (uintptr_t)mapping < end;
		if (selected && !strncmp(line, "VmFlags:", 8)) {
			char *save = NULL;
			for (char *word = strtok_r(line + 8, " \t\n", &save);
			     word; word = strtok_r(NULL, " \t\n", &save)) {
				if (!strcmp(word, "pf"))
					found = true;
			}
			break;
		}
	}
	free(line);
	fclose(stream);
	return found;
}

int main(int argc, char **argv)
{
	struct vfio_group_status status = { .argsz = sizeof(status) };
	struct vfio_device_info device_info = { .argsz = sizeof(device_info) };
	struct vfio_region_info *region = NULL;
	struct glt_arm64_vm vm = {
		.kvm_fd = -1, .vm_fd = -1, .vcpu_fd = -1, .run = MAP_FAILED,
	};
	struct glt_kvm_memslot vector_slot = {
		.slot = 1, .guest_phys_addr = VECTOR_GPA,
	};
	struct kvm_userspace_memory_region device_slot = {
		.slot = 0, .guest_phys_addr = CODE_GPA,
	};
	int container = -1, group = -1, device = -1, result = 2;
	int system_cap, system_errno, vm_cap, vm_errno;
	bool attached = false, saved = false;
	void *mapping = MAP_FAILED;
	void *vectors = MAP_FAILED;
	uint32_t original[2], code[2] = { htole32(0xf9000020), htole32(0x14000000) };
	uint64_t actual, esr = 0, elr = 0;
	char error[256] = "";
	long page = sysconf(_SC_PAGESIZE);

	if (argc != 3 || page < (long)sizeof(code)) {
		fprintf(stderr, "SETUP: expected PCI address and VFIO group node\n");
		return 2;
	}
	setbuf(stdout, NULL);
	/* Report injected guest aborts through a distinct MMIO address. Only the
	 * exception vectors use ordinary RAM; successful execution still has to
	 * start in and fetch instructions from the device-backed memslot.
	 */
	vectors = mmap(NULL, page, PROT_READ | PROT_WRITE,
		       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (vectors == MAP_FAILED) {
		perror("SETUP: exception vector allocation");
		goto cleanup;
	}
	for (size_t offset = 0; offset < 2048; offset += 128) {
		uint32_t *entry = (void *)((char *)vectors + offset);
		entry[0] = htole32(0xf9000420); /* str x0, [x1, #8] */
		entry[1] = htole32(0x14000000); /* b . */
	}
	__builtin___clear_cache(vectors, (char *)vectors + 2048);
	vector_slot.memory_size = page;
	vector_slot.userspace_addr = vectors;
	if (glt_arm64_vm_create(&vm, &vector_slot, 1, error, sizeof(error))) {
		fprintf(stderr, "SETUP: VM/vector memslot creation: %s\n", error);
		goto cleanup;
	}
	errno = 0;
	system_cap = ioctl(vm.kvm_fd, KVM_CHECK_EXTENSION, CACHEABLE_PFNMAP_CAP);
	system_errno = errno;
	errno = 0;
	vm_cap = ioctl(vm.vm_fd, KVM_CHECK_EXTENSION, CACHEABLE_PFNMAP_CAP);
	vm_errno = errno;
	printf("ABI: system=%d errno=%d vm=%d errno=%d\n",
	       system_cap, system_errno, vm_cap, vm_errno);
	if (system_cap == -1 && system_errno == EINVAL &&
	    vm_cap == 0 && vm_errno == 0) {
		fprintf(stderr, "UNAVAILABLE: VM capability reports no cacheable PFNMAP "
			"support; requires FEAT_S2FWB and CACHE_DIC\n");
		result = 4;
		goto cleanup;
	}
	container = open("/dev/vfio/vfio", O_RDWR | O_CLOEXEC);
	if (container < 0) {
		if (errno == ENOENT || errno == ENODEV)
			result = 4;
		perror("SETUP: open VFIO container");
		goto cleanup;
	}
	group = open(argv[2], O_RDWR | O_CLOEXEC);
	if (group < 0) {
		if (errno == EBUSY || errno == ENOENT || errno == ENODEV)
			result = 3;
		perror("SETUP: open nominated VFIO group");
		goto cleanup;
	}
	int api = ioctl(container, VFIO_GET_API_VERSION);
	if (api < 0 || ioctl(group, VFIO_GROUP_GET_STATUS, &status)) {
		perror("SETUP: query VFIO API/group");
		goto cleanup;
	}
	if (api != VFIO_API_VERSION) {
		fprintf(stderr, "UNAVAILABLE: VFIO API unsupported\n");
		result = 4;
		goto cleanup;
	}
	if (!(status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
		fprintf(stderr, "UNAVAILABLE: VFIO group not viable\n");
		result = 3;
		goto cleanup;
	}
	if (ioctl(group, VFIO_GROUP_SET_CONTAINER, &container)) {
		if (errno == EBUSY)
			result = 3;
		perror("SETUP: acquire VFIO group");
		goto cleanup;
	}
	attached = true;
	int iommu = strstr(argv[2], "/noiommu-") ?
		VFIO_NOIOMMU_IOMMU : VFIO_TYPE1v2_IOMMU;
	int extension = ioctl(container, VFIO_CHECK_EXTENSION, iommu);
	if (extension < 0) {
		perror("SETUP: query VFIO IOMMU backend");
		goto cleanup;
	}
	if (!extension) {
		fprintf(stderr, "UNAVAILABLE: required VFIO IOMMU backend\n");
		result = 4;
		goto cleanup;
	}
	if (ioctl(container, VFIO_SET_IOMMU, iommu)) {
		perror("SETUP: VFIO IOMMU backend");
		goto cleanup;
	}
	device = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, argv[1]);
	if (device < 0) {
		perror("SETUP: acquire nominated VFIO device");
		goto cleanup;
	}
	region = calloc(1, REGION_BUFFER_SIZE);
	if (!region || ioctl(device, VFIO_DEVICE_GET_INFO, &device_info)) {
		perror("SETUP: VFIO PCI device");
		goto cleanup;
	}
	if (!(device_info.flags & VFIO_DEVICE_FLAGS_PCI) ||
	    device_info.num_regions <= VFIO_PCI_BAR4_REGION_INDEX) {
		fprintf(stderr, "UNAVAILABLE: PCI device with USEMEM BAR4 required\n");
		result = 3;
		goto cleanup;
	}
	region->argsz = REGION_BUFFER_SIZE;
	region->index = VFIO_PCI_BAR4_REGION_INDEX;
	if (ioctl(device, VFIO_DEVICE_GET_REGION_INFO, region)) {
		perror("SETUP: query USEMEM BAR4");
		goto cleanup;
	}
	if (!region_has_page(region, page) || region->offset > INT64_MAX) {
		fprintf(stderr, "UNAVAILABLE: no valid cacheable USEMEM sparse mmap page\n");
		result = 3;
		goto cleanup;
	}
	if (!device_ready(device)) {
		fprintf(stderr, "SETUP: cannot establish post-reset GPU readiness\n");
		goto cleanup;
	}
	/* The host only writes instructions; execution occurs in the guest.
	 * VFIO's device file must not require executable host mappings.
	 */
	mapping = mmap(NULL, page, PROT_READ | PROT_WRITE,
		       MAP_SHARED, device, (off_t)region->offset);
	if (mapping == MAP_FAILED) {
		perror("SETUP: mmap USEMEM");
		goto cleanup;
	}
	if (!mapping_is_pfnmap(mapping)) {
		fprintf(stderr, "SETUP: USEMEM VMA lacks PFNMAP flag\n");
		goto cleanup;
	}
	printf("STIMULUS: device=%s BAR4 offset=%" PRIu64 " page=%ld PFNMAP=1\n",
	       argv[1], (uint64_t)region->offset, page);
	memcpy(original, mapping, sizeof(original));
	saved = true;
	puts("STIMULUS: original scratch bytes saved");
	memcpy(mapping, code, sizeof(code));
	puts("STIMULUS: guest instructions written");
	__builtin___clear_cache(mapping, (char *)mapping + sizeof(code));
	puts("STIMULUS: instruction cache synchronized");
	/* Keep device-slot rejection distinct from ordinary VM setup. The
	 * block-only partial backport rejects this valid PFNMAP slot with EINVAL.
	 */
	device_slot.memory_size = page;
	device_slot.userspace_addr = (uintptr_t)mapping;
	if (ioctl(vm.vm_fd, KVM_SET_USER_MEMORY_REGION, &device_slot)) {
		if (errno == EINVAL) {
			fprintf(stderr, "BEHAVIOR: cacheable PFNMAP memslot rejected; "
				"requires FEAT_S2FWB and CACHE_DIC; check "
				MAPPING_PATCH "\n");
			result = 1;
		} else {
			perror("SETUP: device memslot registration");
		}
		goto cleanup;
	}
	if (glt_arm64_vm_set_reg(&vm, ARM64_SYS_REG(3, 0, 12, 0, 0),
				 VECTOR_GPA, error, sizeof(error)) ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.pc),
				 CODE_GPA, error, sizeof(error)) ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.regs[0]),
				 MARKER, error, sizeof(error)) ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.regs[1]),
				 MMIO_GPA, error, sizeof(error))) {
		fprintf(stderr, "SETUP: vCPU registers: %s\n", error);
		goto cleanup;
	}
	if (glt_arm64_vm_run(&vm, 5, error, sizeof(error))) {
		fprintf(stderr, "PROBE: KVM_RUN non-conclusive: %s\n", error);
		goto cleanup;
	}
	printf("BEHAVIOR: exit=%u\n", vm.run->exit_reason);
	if (vm.run->exit_reason == KVM_EXIT_MMIO && vm.run->mmio.is_write &&
	    vm.run->mmio.phys_addr == MMIO_GPA + 8 &&
	    vm.run->mmio.len == sizeof(actual)) {
		struct kvm_one_reg esr_reg = {
			.id = ARM64_SYS_REG(3, 0, 5, 2, 0),
			.addr = (uintptr_t)&esr,
		};
		struct kvm_one_reg elr_reg = {
			.id = GLT_ARM64_CORE_REG(elr_el1),
			.addr = (uintptr_t)&elr,
		};
		memcpy(&actual, vm.run->mmio.data, sizeof(actual));
		if (ioctl(vm.vcpu_fd, KVM_GET_ONE_REG, &esr_reg) ||
		    ioctl(vm.vcpu_fd, KVM_GET_ONE_REG, &elr_reg)) {
			perror("SETUP: read guest exception state");
			goto cleanup;
		}
		if (le64toh(actual) == MARKER && esr == UINT64_C(0x86000010) &&
		    elr == CODE_GPA) {
			fprintf(stderr, "BEHAVIOR: device instruction abort verified "
				"ESR=%#" PRIx64 " ELR=%#" PRIx64 "; check "
				MAPPING_PATCH "\n", esr, elr);
			result = 1;
			goto cleanup;
		}
		fprintf(stderr, "PROBE: unexpected guest exception ESR=%#" PRIx64
			" ELR=%#" PRIx64 "\n", esr, elr);
		goto cleanup;
	}
	if (vm.run->exit_reason != KVM_EXIT_MMIO ||
	    !vm.run->mmio.is_write || vm.run->mmio.phys_addr != MMIO_GPA ||
	    vm.run->mmio.len != sizeof(actual)) {
		fprintf(stderr, "PROBE: non-conclusive exit, expected completion MMIO store\n");
		goto cleanup;
	}
	memcpy(&actual, vm.run->mmio.data, sizeof(actual));
	if (le64toh(actual) != MARKER) {
		fprintf(stderr, "PROBE: non-conclusive incorrect completion marker\n");
		goto cleanup;
	}
	printf("BEHAVIOR: cacheable PFNMAP instruction fetch and MMIO marker verified\n");
	if (system_cap != -1 || system_errno != EINVAL || vm_cap != 1) {
		fprintf(stderr, "ABI: expected system EINVAL and VM capability=1; check "
			ABI_PATCH "\n");
		result = 1;
		goto cleanup;
	}
	result = 0;

cleanup:
	glt_arm64_vm_destroy(&vm);
	if (saved) {
		memcpy(mapping, original, sizeof(original));
		__builtin___clear_cache(mapping, (char *)mapping + sizeof(original));
		if (memcmp(mapping, original, sizeof(original))) {
			fprintf(stderr, "CLEANUP: scratch-byte restoration failed\n");
			result = 2;
		}
	}
	if (mapping != MAP_FAILED && munmap(mapping, page)) {
		perror("CLEANUP: unmap USEMEM");
		result = 2;
	}
	if (vectors != MAP_FAILED && munmap(vectors, page)) {
		perror("CLEANUP: unmap exception vectors");
		result = 2;
	}
	free(region);
	if (device >= 0 && close(device)) {
		perror("CLEANUP: close VFIO device");
		result = 2;
	}
	if (attached && ioctl(group, VFIO_GROUP_UNSET_CONTAINER)) {
		perror("CLEANUP: detach VFIO container");
		result = 2;
	}
	if (group >= 0 && close(group)) {
		perror("CLEANUP: close VFIO group");
		result = 2;
	}
	if (container >= 0 && close(container)) {
		perror("CLEANUP: close VFIO container");
		result = 2;
	}
	return result;
}
