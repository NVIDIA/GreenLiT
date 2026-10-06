// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static uint64_t counter_after = 100;
static ssize_t counter_read_size = sizeof(counter_after);

static ssize_t mock_read(int fd, void *buffer, size_t size)
{
    assert(fd == 7 && size == sizeof(counter_after));
    memcpy(buffer, &counter_after, sizeof(counter_after));
    errno = EIO;
    return counter_read_size;
}

/* The Python test extracts these declarations from the production harness. */
#define read mock_read
#include "brbe_nvhe_oracle.inc"
#undef read

int main(int argc, char **argv)
{
    const struct evidence baseline = {
        .perf_samples = 11,
        .branch_entries = 321,
        .host_pre_hits = 1,
        .host_post_hits = 1,
        .context_switches = 6,
        .guest_start = 0x40000000,
        .guest_end = 0x40001000,
        .host_pre_start = 0x2000,
        .host_pre_end = 0x2100,
        .host_post_start = 0x3000,
        .host_post_end = 0x3100,
        .last_sample_ip = 0x40000004,
    };
    struct guest_perf_window window = {
        .samples = 10, .entries = 320, .instructions = 10,
    };
    struct evidence evidence;
    char error[256] = "";
    FILE *entries = tmpfile();
    struct {
        struct perf_event_header header;
        uint64_t ip;
        uint32_t pid;
        uint32_t tid;
        uint64_t time;
        uint64_t nr;
        struct perf_branch_entry branches[1];
    } sample = {
        .header = {.type = PERF_RECORD_SAMPLE},
        .ip = 0x40000004,
        .pid = 1,
        .tid = 1,
        .nr = 1,
        .branches = {{.from = 0x2004, .to = 0x2008}},
    };
    struct options options;
    char row[512];

    assert(argc == 2);
    options = (struct options){.result_json_path = argv[1]};
    assert(entries != NULL);
    /* Host controls cannot substitute for entries in this individual window. */
    evidence = baseline;
    evidence.branch_entries = window.entries;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);
    assert(evidence.guest_windows_verified == 0);

    /* An overflow taken in host code is not proof of sampling the guest. */
    evidence = baseline;
    evidence.last_sample_ip = 0x2004;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);
    evidence.last_sample_ip = evidence.guest_end;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);

    /* Retired instructions must span the remaining overflow period. */
    evidence = baseline;
    counter_after = 99;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);
    counter_after = 9;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);
    counter_after = 100;
    window.instructions = 100;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);
    window.instructions = 10;
    for (counter_read_size = -1; counter_read_size <= 7; counter_read_size += 8)
        assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                       sizeof(error)) != 0);
    counter_read_size = sizeof(counter_after);

    /* Exactly one new sample is required, not zero or multiple samples. */
    evidence.perf_samples = window.samples;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);
    evidence.perf_samples = window.samples + 2;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);

    /* Host-only branch entries are valid if the overflow itself was in guest. */
    evidence = baseline;
    evidence.perf_samples = window.samples;
    evidence.branch_entries = window.entries;
    sample.header.size = sizeof(sample);
    assert(parse_sample_record((const uint8_t *)&sample, sizeof(sample), entries,
                               &evidence, error, sizeof(error)) == 0);
    assert(evidence.guest_samples == 1 && evidence.guest_hits == 0);
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) == 0);
    assert(evidence.guest_windows_verified == 1);
    assert(evidence.min_guest_instruction_delta == 90);
    assert(write_result_json(&options, &evidence, 0, "fixed", "") == 0);
    rewind(entries);
    assert(fgets(row, sizeof(row), entries) != NULL);
    assert(strstr(row, "\t0x40000004\n") != NULL);

    /* Keep the minimum instruction delta across verified windows. */
    evidence = baseline;
    evidence.guest_windows_verified = 1;
    evidence.min_guest_instruction_delta = 95;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) == 0);
    assert(evidence.guest_windows_verified == 2);
    assert(evidence.min_guest_instruction_delta == 90);
    counter_after = 150;
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) == 0);
    assert(evidence.min_guest_instruction_delta == 90);

    /* A guest-IP sample with an empty stack still lacks observer evidence. */
    evidence = baseline;
    evidence.perf_samples = window.samples;
    evidence.branch_entries = window.entries;
    sample.nr = 0;
    sample.header.size = offsetof(typeof(sample), branches);
    assert(parse_sample_record((const uint8_t *)&sample, sample.header.size,
                               entries, &evidence, error, sizeof(error)) == 0);
    assert(evidence.guest_samples == 1);
    assert(check_guest_perf_window(7, 100, &window, &evidence, "guest", error,
                                   sizeof(error)) != 0);

    /* Contamination remains detectable even with a host-IP overflow. */
    evidence = baseline;
    sample.ip = 0x2004;
    sample.nr = 1;
    sample.header.size = sizeof(sample);
    sample.branches[0].to = evidence.guest_start;
    assert(parse_sample_record((const uint8_t *)&sample, sizeof(sample), entries,
                               &evidence, error, sizeof(error)) == 0);
    assert(evidence.guest_samples == 0 && evidence.guest_hits == 1);
    assert(fclose(entries) == 0);
    return 0;
}
