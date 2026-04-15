// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// Test Case: ARM SMMU v3 TLB Invalidation Range Latency
//
// Description:
//   CUDA test program that performs memory allocations and deallocations
//   to trigger TLB shootdown events. Used to measure TLB invalidation
//   performance on Grace-Hopper systems.
//
// Patch: https://git.kernel.org/torvalds/c/d5afb4b47e13

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>
#include <new>
#include <sys/mman.h>
#include <chrono>

#include <unistd.h>
long page_size = sysconf (_SC_PAGESIZE);

void benchmark(int iterations, size_t sz, bool use_mmap, bool touch_memory) {
  std::stringstream ss;
  ss << (use_mmap ? "mmap" : "new") << "_" << (touch_memory ? "touch" : "no_touch") << "_";
  std::string testcase_name = ss.str();

  double total_alloc_time_ms = 0;
  double total_release_time_ms = 0;

  for(int i = 0; i < iterations; i++) {
    char *p;

    auto alloc_start = std::chrono::steady_clock::now();
    if (use_mmap) {
      p = (char*) mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
      if (p == MAP_FAILED) std::abort();
    } else {
      p = new char[sz];
      if (p == nullptr) std::abort();
    }
    auto alloc_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> alloc_duration = alloc_end - alloc_start;
    total_alloc_time_ms += alloc_duration.count();

    if (touch_memory) {
      for (int j = 0; j < sz; j += page_size) {
        p[j] = 42;
      }
    }

    auto release_start = std::chrono::steady_clock::now();
    if (use_mmap) {
      if (munmap(p, sz) < 0) std::abort();
    } else {
      delete[] p;
    }

    // C timers have easier syntax...
    auto release_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> release_duration = release_end - release_start;
    total_release_time_ms += release_duration.count();
  }

  std::cout << "&&&& PERF " << testcase_name << "alloc " << total_alloc_time_ms / iterations << std::endl;
  std::cout << "&&&& PERF " << testcase_name << "release " << total_release_time_ms / iterations << std::endl;
}


int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Takes two argument for #TLB shootdowns and alloc size" << std::endl;
    std::abort();
  }

  int iterations = atoi(argv[1]);
  std::cout << "#TLB shootdowns: " << iterations << " with page size " << page_size << std::endl;
  assert(iterations > 0);
  size_t sz = (size_t)atof(argv[2]);
  assert(sz > 0);

  if (cudaSuccess != cudaFree(0)) {
    std::cerr << "cudaFree(0) failed\n";
    std::abort();
  }

  benchmark(iterations, sz, false, false);
  // benchmark(iterations, sz, false, true);
  benchmark(iterations, sz, true, false);
  // benchmark(iterations, sz, true, true);

  return 0;
}
