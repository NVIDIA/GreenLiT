// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// arm64 memory hot-remove per-page TLB-flush regression workload
//
// Registers every GPU with UVM (creating the coherent ZONE_DEVICE pagemap for
// the GPU memory window) and makes a LARGE amount of GPU memory device-
// resident via a GPU-side memset of managed memory. Process exit frees those
// managed allocations but leaves the pagemap in place; a later
// `rmmod nvidia_uvm` tears it down via memunmap_pages -> arch_remove_memory ->
// unmap_hotplug_range, which on a pre-fix arm64 kernel flushes the kernel TLB
// one page at a time and can soft-lockup. The Python test builds and runs this
// to prime registration, then times the rmmod after the process has exited
// (required so the module is not busy).
//
// The CUDA driver API is reached via dlopen so the workload has no build- or
// link-time dependency on the CUDA toolkit. arg1 (optional) = MB to make
// resident per GPU (0 or absent => ~90% of each GPU's free HBM).

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;

/* Resolve a driver symbol, trying a _v2 (or other) alias when the primary name
 * is absent. Returns NULL only if neither name is present. */
static void *sym(void *handle, const char *primary, const char *fallback)
{
    void *p = dlsym(handle, primary);
    if (!p && fallback)
        p = dlsym(handle, fallback);
    return p;
}

int main(int argc, char **argv)
{
    long forced_mb = argc > 1 ? atol(argv[1]) : 0;

    void *handle = dlopen("libcuda.so.1", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
        return 2;
    }

    CUresult (*cu_init)(unsigned) = sym(handle, "cuInit", NULL);
    CUresult (*cu_device_get_count)(int *) = sym(handle, "cuDeviceGetCount", NULL);
    CUresult (*cu_device_get)(CUdevice *, int) = sym(handle, "cuDeviceGet", NULL);
    CUresult (*cu_ctx_create)(CUcontext *, unsigned, CUdevice) =
        sym(handle, "cuCtxCreate_v2", "cuCtxCreate");
    CUresult (*cu_mem_get_info)(size_t *, size_t *) =
        sym(handle, "cuMemGetInfo_v2", "cuMemGetInfo");
    CUresult (*cu_mem_alloc_managed)(CUdeviceptr *, size_t, unsigned) =
        sym(handle, "cuMemAllocManaged", NULL);
    CUresult (*cu_memset_d8)(CUdeviceptr, unsigned char, size_t) =
        sym(handle, "cuMemsetD8_v2", "cuMemsetD8");
    CUresult (*cu_ctx_synchronize)(void) = sym(handle, "cuCtxSynchronize", NULL);

    if (!cu_init || !cu_device_get_count || !cu_device_get || !cu_ctx_create ||
        !cu_mem_get_info || !cu_mem_alloc_managed || !cu_memset_d8) {
        fprintf(stderr, "dlsym of a required CUDA driver symbol failed\n");
        return 2;
    }

    if (cu_init(0)) {
        fprintf(stderr, "cuInit failed\n");
        return 3;
    }

    int count = 0;
    CUresult count_rc = cu_device_get_count(&count);
    if (count_rc) {
        fprintf(stderr, "cuDeviceGetCount failed rc=%d\n", count_rc);
        return 4;
    }
    fprintf(stderr, "devices=%d (forced_mb=%ld, 0=auto ~90%% free)\n",
            count, forced_mb);

    /* Total MB we actually made device-resident (memset + synchronized) across
     * all GPUs. Only GPUs whose fill genuinely succeeded contribute. */
    size_t total_resident_mb = 0;

    for (int i = 0; i < count; i++) {
        CUdevice dev;
        if (cu_device_get(&dev, i)) {
            fprintf(stderr, "gpu %d: cuDeviceGet failed\n", i);
            continue;
        }

        CUcontext ctx;
        if (cu_ctx_create(&ctx, 0, dev)) {
            fprintf(stderr, "gpu %d: cuCtxCreate failed\n", i);
            continue;
        }

        size_t free_b = 0, total_b = 0;
        CUresult info_rc = cu_mem_get_info(&free_b, &total_b);
        if (info_rc) {
            /* Without a valid free size we would derive size 0 and "fill"
             * nothing; skip this GPU rather than pretend it was filled. */
            fprintf(stderr, "gpu %d: cuMemGetInfo failed rc=%d; skipping\n",
                    i, info_rc);
            continue;
        }

        size_t size = forced_mb > 0
                          ? (size_t)forced_mb * 1024UL * 1024UL
                          : (size_t)((double)free_b * 0.90);
        size &= ~((size_t)(2 * 1024 * 1024) - 1);   /* 2 MB aligned */
        if (!size) {
            fprintf(stderr, "gpu %d: size rounded to 0 (free=%zu MB); skipping\n",
                    i, free_b >> 20);
            continue;
        }

        CUdeviceptr ptr = 0;
        CUresult rc = cu_mem_alloc_managed(&ptr, size, 1u);
        if (rc) {
            fprintf(stderr, "gpu %d: cuMemAllocManaged %zu MB failed rc=%d (free=%zu MB)\n",
                    i, size >> 20, rc, free_b >> 20);
            continue;
        }

        /* cuMemsetD8 is the op that forces the pages device-resident; if it or
         * the synchronize fails the residency is not guaranteed, so this GPU
         * must NOT be counted toward the resident total. */
        CUresult set_rc = cu_memset_d8(ptr, 0xAB, size);   /* GPU-side write */
        if (set_rc) {
            fprintf(stderr, "gpu %d: cuMemsetD8 %zu MB failed rc=%d; not counted\n",
                    i, size >> 20, set_rc);
            continue;
        }

        if (!cu_ctx_synchronize) {
            fprintf(stderr, "gpu %d: cuCtxSynchronize unavailable; not counted\n", i);
            continue;
        }
        CUresult sync_rc = cu_ctx_synchronize();
        if (sync_rc) {
            fprintf(stderr, "gpu %d: cuCtxSynchronize failed rc=%d; not counted\n",
                    i, sync_rc);
            continue;
        }

        total_resident_mb += size >> 20;
        fprintf(stderr, "gpu %d: resident %zu MB (free was %zu / total %zu MB)\n",
                i, size >> 20, free_b >> 20, total_b >> 20);
    }

    /* Exit on purpose: CUDA must drop its nvidia_uvm refs so rmmod can run.
     * Process teardown frees this process's managed VA allocations, but the
     * coherent ZONE_DEVICE pagemap created at UVM parent-GPU init is only
     * memunmap'd in uvm_devmem_exit() (module unload) — that is the
     * arch_remove_memory / unmap_hotplug_range path under test. Keeping this
     * process alive across rmmod would hold the module busy and prevent the
     * timed unload. */
    fprintf(stderr, "done; coherent pagemap persists until rmmod nvidia_uvm\n");

    /* Machine-parseable summary of how much memory was truly made resident. */
    printf("TOTAL_RESIDENT_MB=%zu\n", total_resident_mb);
    fflush(stdout);

    /* If nothing was filled, the teardown cannot be exercised; fail loudly so
     * a broken workload on a buggy kernel is not misread as a fast PASS. */
    if (total_resident_mb == 0) {
        fprintf(stderr, "no GPU was filled; nothing to tear down\n");
        return 5;
    }
    return 0;
}
