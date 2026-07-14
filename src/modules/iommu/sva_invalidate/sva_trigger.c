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
// SVA invalidate-range size regression workload (kernel fix 95a5885).
//
// dlopen()s libcuda, creates a CUDA context, then requires a managed alloc +
// GPU memset as the SVA bind stimulus. Only after that succeeds does it loop
// mmap -> memset -> munmap. Each munmap should fire
// arm_smmu_mm_arch_invalidate_secondary_tlbs on an SVA-bound mm. The Python
// harness proves the path ran via PID-filtered kprobe hits; if bind/stimulus
// fails this binary exits non-zero with an explicit stderr reason.
// Ranges are page-aligned so the workload itself never feeds a degenerate
// length, but that does not make the path hang-proof: on an unfixed kernel
// with ARM_SMMU_FEAT_RANGE_INV, size = end - start + 1 can soft-lock
// __arm_smmu_tlb_inv_range during munmap. The Python harness FAILs on
// TimeoutExpired for that case.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
typedef int CUresult; typedef int CUdevice; typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
static void* S(void*h,const char*a,const char*b){void*p=dlsym(h,a);if(!p&&b)p=dlsym(h,b);return p;}
int main(int argc,char**argv){
  long ps=sysconf(_SC_PAGESIZE); int iters=argc>1?atoi(argv[1]):800;
  void*h=dlopen("libcuda.so.1",RTLD_NOW);
  if(!h){fprintf(stderr,"dlopen libcuda failed: %s\n",dlerror());return 2;}
  CUresult(*cuInit)(unsigned)=S(h,"cuInit",0);
  CUresult(*cuDeviceGet)(CUdevice*,int)=S(h,"cuDeviceGet",0);
  CUresult(*cuCtxCreate)(CUcontext*,unsigned,CUdevice)=S(h,"cuCtxCreate_v2","cuCtxCreate");
  CUresult(*cuMemAllocManaged)(CUdeviceptr*,size_t,unsigned)=S(h,"cuMemAllocManaged",0);
  CUresult(*cuMemsetD8)(CUdeviceptr,unsigned char,size_t)=S(h,"cuMemsetD8_v2","cuMemsetD8");
  CUresult(*cuCtxSync)(void)=S(h,"cuCtxSynchronize",0);
  if(!cuInit||!cuDeviceGet||!cuCtxCreate){fprintf(stderr,"dlsym core failed\n");return 2;}
  if(cuInit(0)){fprintf(stderr,"cuInit failed\n");return 3;}
  CUdevice dev; if(cuDeviceGet(&dev,0)){fprintf(stderr,"cuDeviceGet failed\n");return 3;}
  CUcontext ctx; if(cuCtxCreate(&ctx,0,dev)){fprintf(stderr,"cuCtxCreate failed\n");return 4;}
  if(!cuMemAllocManaged||!cuMemsetD8||!cuCtxSync){
    fprintf(stderr,"managed-alloc symbols unavailable; cannot establish SVA bind stimulus\n");
    return 5;
  }
  CUdeviceptr d=0;
  size_t managed_len=(size_t)64*(size_t)ps;
  if(cuMemAllocManaged(&d,managed_len,1u)){
    fprintf(stderr,"cuMemAllocManaged failed; SVA not bound\n");
    return 5;
  }
  if(cuMemsetD8(d,0xAB,managed_len)){
    fprintf(stderr,"cuMemsetD8 failed; SVA bind stimulus incomplete\n");
    return 5;
  }
  if(cuCtxSync()){
    fprintf(stderr,"cuCtxSynchronize failed; SVA bind stimulus incomplete\n");
    return 5;
  }
  for(int i=0;i<iters;i++){
    size_t len=(size_t)(1+(i%48))*(size_t)ps;
    void*p=mmap(NULL,len,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED)continue;
    memset(p,0xCD,len); munmap(p,len);
  }
  fprintf(stderr,"trigger done\n"); return 0;
}
