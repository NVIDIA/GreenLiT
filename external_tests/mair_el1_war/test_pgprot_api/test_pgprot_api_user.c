// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

static int expect_sigbus;

static void sigbus_handler(int sig)
{
    (void)sig;
    if (expect_sigbus)
        _Exit(0);
    _Exit(1);
}

static void test_aligned(const char *path)
{
    int fd;
    void *map;
    size_t page_size = (size_t)getpagesize();
    volatile uint32_t *w;

    fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    map = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(1);
    }

    w = (volatile uint32_t *)map;

    w[0] = 0x11223344;
    w[1] = 0x55667788;

    if (w[0] != 0x11223344 || w[1] != 0x55667788) {
        fprintf(stderr, "%s: aligned access verification failed\n", path);
        munmap(map, page_size);
        close(fd);
        exit(1);
    }

    munmap(map, page_size);
    close(fd);
}

static void test_unaligned(const char *path)
{
    int fd;
    void *map;
    size_t page_size = (size_t)getpagesize();
    volatile uint32_t *unaligned;
    struct sigaction sa_new, sa_old;

    fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    map = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(1);
    }

    sa_new.sa_handler = sigbus_handler;
    sigemptyset(&sa_new.sa_mask);
    sa_new.sa_flags = 0;

    if (sigaction(SIGBUS, &sa_new, &sa_old) != 0) {
        perror("sigaction");
        munmap(map, page_size);
        close(fd);
        exit(1);
    }

    unaligned = (volatile uint32_t *)((char *)map + 1);
    *unaligned = 0xa5a5a5a5;
    (void)*unaligned;

    sigaction(SIGBUS, &sa_old, NULL);

    /* Reached here: no SIGBUS. If we expected a fault, that's a failure. */
    if (expect_sigbus) {
        fprintf(stderr, "%s: expected unaligned fault but access succeeded\n", path);
        munmap(map, page_size);
        close(fd);
        exit(1);
    }

    munmap(map, page_size);
    close(fd);
}

int main(int argc, char **argv)
{
    /* argv[1] = device, argv[2] = expect SIGBUS (0|1), argv[3] = test: 1=aligned, 0=unaligned */
    if (argc != 4) {
        fprintf(stderr, "usage: %s <device> <expect_sigbus 0|1> <test 1=aligned 0=unaligned>\n", argv[0]);
        return 1;
    }

    const char *device = argv[1];
    expect_sigbus = atoi(argv[2]);
    int run_aligned = atoi(argv[3]);

    if ((expect_sigbus != 0 && expect_sigbus != 1) || (run_aligned != 0 && run_aligned != 1)) {
        fprintf(stderr, "usage: %s <device> <expect_sigbus 0|1> <test 1=aligned 0=unaligned>\n", argv[0]);
        return 1;
    }

    if (run_aligned) {
        test_aligned(device);
    } else {
        test_unaligned(device);
    }
    return 0;
}
