// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define _GNU_SOURCE

#include "glt_arm64_kvm.h"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#if !defined(__aarch64__)
#error "glt_brbe_nvhe_harness must be built for AArch64"
#endif

#define EXIT_FIXED_EVIDENCE 0
#define EXIT_CONTAMINATION 1
#define EXIT_INCONCLUSIVE 2

#define DEFAULT_ITERATIONS UINT64_C(128)
#define DEFAULT_GUEST_BRANCHES UINT64_C(20000000)
#define DEFAULT_SAMPLE_PERIOD UINT64_C(10000000)
#define PRE_GUEST_COUNT_LIMIT_DIVISOR UINT64_C(4)
#define HOST_CONTROL_SAMPLE_PERIOD UINT64_C(50000)
#define HOST_CONTROL_BRANCHES UINT64_C(100000)
#define PERF_DATA_PAGES 64U
#define KVM_RUN_TIMEOUT_SECONDS 5U

#define GUEST_CODE_GPA UINT64_C(0x40000000)
#define GUEST_DATA_GPA UINT64_C(0x40010000)
#define GUEST_MMIO_GPA UINT64_C(0x50000000)

#define GUEST_PSTATE_EL1H UINT64_C(0x5)
#define GUEST_PSTATE_DAIF UINT64_C(0x3c0)

extern unsigned char glt_guest_start[];
extern unsigned char glt_guest_entry[];
extern unsigned char glt_guest_end[];

extern void glt_host_pre_marker(uint64_t iterations);
extern unsigned char glt_host_pre_start[];
extern unsigned char glt_host_pre_end[];
extern void glt_host_post_marker(uint64_t iterations);
extern unsigned char glt_host_post_start[];
extern unsigned char glt_host_post_end[];

struct options {
	int cpu;
	uint64_t iterations;
	uint64_t guest_branches;
	uint64_t sample_period;
	const char *result_json_path;
	const char *entries_path;
};

struct guest_mailbox {
	uint64_t id_brbe;
	uint64_t brbcr_undef;
	uint64_t unexpected_exception;
	uint64_t undef_expected;
	uint64_t iterations;
};

_Static_assert(offsetof(struct guest_mailbox, id_brbe) == 0,
	       "guest mailbox ID offset changed");
_Static_assert(offsetof(struct guest_mailbox, brbcr_undef) == 8,
	       "guest mailbox UNDEF offset changed");
_Static_assert(offsetof(struct guest_mailbox, unexpected_exception) == 16,
	       "guest mailbox exception offset changed");
_Static_assert(offsetof(struct guest_mailbox, undef_expected) == 24,
	       "guest mailbox expectation offset changed");
_Static_assert(offsetof(struct guest_mailbox, iterations) == 32,
	       "guest mailbox iteration offset changed");

struct perf_ring {
	struct perf_event_mmap_page *metadata;
	uint8_t *data;
	size_t data_size;
	size_t mapping_size;
};

struct guest_perf_window {
	uint64_t samples;
	uint64_t entries;
	uint64_t instructions;
};

struct evidence {
	uint64_t iterations_completed;
	uint64_t perf_samples;
	uint64_t branch_entries;
	uint64_t guest_hits;
	uint64_t host_pre_hits;
	uint64_t host_post_hits;
	uint64_t lost_records;
	uint64_t throttle_records;
	uint64_t context_switches;
	uint64_t guest_start;
	uint64_t guest_end;
	uint64_t host_pre_start;
	uint64_t host_pre_end;
	uint64_t host_post_start;
	uint64_t host_post_end;
	uint64_t max_pre_guest_count;
	uint64_t guest_samples;
	uint64_t guest_windows_verified;
	uint64_t min_guest_instruction_delta;
	uint64_t last_sample_ip;
	uint64_t guest_id_brbe;
	uint64_t guest_brbcr_undef;
	uint64_t guest_unexpected_exception;
	bool nvhe_cmdline;
};

static bool address_in_range(uint64_t address, uint64_t start, uint64_t end)
{
	return address >= start && address < end;
}

static bool branch_hits_range(const struct perf_branch_entry *entry,
			      uint64_t start, uint64_t end)
{
	return address_in_range(entry->from, start, end) ||
	       address_in_range(entry->to, start, end);
}

static bool cmdline_has_token(const char *cmdline, const char *wanted)
{
	const char *cursor = cmdline;
	size_t wanted_len = strlen(wanted);

	while (*cursor != '\0') {
		const char *end;

		while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n')
			cursor++;
		if (*cursor == '\0')
			break;
		end = cursor;
		while (*end != '\0' && *end != ' ' && *end != '\t' &&
		       *end != '\n')
			end++;
		if ((size_t)(end - cursor) == wanted_len &&
		    memcmp(cursor, wanted, wanted_len) == 0)
			return true;
		cursor = end;
	}

	return false;
}

static bool check_nvhe_cmdline(char *error, size_t error_size)
{
	char cmdline[8192];
	size_t length;
	FILE *stream = fopen("/proc/cmdline", "r");
	bool has_mode;

	if (stream == NULL) {
		snprintf(error, error_size, "failed to open /proc/cmdline: %s",
			 strerror(errno));
		return false;
	}
	length = fread(cmdline, 1, sizeof(cmdline) - 1, stream);
	if (ferror(stream)) {
		snprintf(error, error_size, "failed to read /proc/cmdline: %s",
			 strerror(errno));
		fclose(stream);
		return false;
	}
	fclose(stream);
	cmdline[length] = '\0';

	has_mode = cmdline_has_token(cmdline, "kvm-arm.mode=nvhe") ||
		   cmdline_has_token(cmdline, "kvm_arm.mode=nvhe");
	if (!has_mode ||
	    !cmdline_has_token(cmdline, "arm64_sw.hvhe=0") ||
	    !cmdline_has_token(cmdline, "id_aa64mmfr1.vh=0")) {
		snprintf(error, error_size,
			 "nVHE cmdline precondition is missing; require "
			 "kvm-arm.mode=nvhe arm64_sw.hvhe=0 "
			 "id_aa64mmfr1.vh=0");
		return false;
	}

	return true;
}

static void json_write_string(FILE *stream, const char *value)
{
	const unsigned char *cursor = (const unsigned char *)value;

	fputc('"', stream);
	while (*cursor != '\0') {
		switch (*cursor) {
		case '"':
			fputs("\\\"", stream);
			break;
		case '\\':
			fputs("\\\\", stream);
			break;
		case '\b':
			fputs("\\b", stream);
			break;
		case '\f':
			fputs("\\f", stream);
			break;
		case '\n':
			fputs("\\n", stream);
			break;
		case '\r':
			fputs("\\r", stream);
			break;
		case '\t':
			fputs("\\t", stream);
			break;
		default:
			if (*cursor < 0x20)
				fprintf(stream, "\\u%04x", *cursor);
			else
				fputc(*cursor, stream);
			break;
		}
		cursor++;
	}
	fputc('"', stream);
}

static int write_result_json(const struct options *options,
			     const struct evidence *evidence,
			     int exit_code, const char *verdict,
			     const char *error)
{
	FILE *stream;

	if (options->result_json_path == NULL)
		return -1;
	stream = fopen(options->result_json_path, "w");
	if (stream == NULL) {
		fprintf(stderr, "failed to open result JSON %s: %s\n",
			options->result_json_path, strerror(errno));
		return -1;
	}

	fputs("{\n  \"schema_version\": 3,\n  \"verdict\": ", stream);
	json_write_string(stream, verdict);
	fprintf(stream,
		",\n  \"exit_code\": %d,\n"
		"  \"cpu\": %d,\n"
		"  \"iterations\": %" PRIu64 ",\n"
		"  \"iterations_completed\": %" PRIu64 ",\n"
		"  \"guest_branches\": %" PRIu64 ",\n"
		"  \"sample_period\": %" PRIu64 ",\n"
		"  \"max_pre_guest_count\": %" PRIu64 ",\n"
		"  \"guest_samples\": %" PRIu64 ",\n"
		"  \"guest_windows_verified\": %" PRIu64 ",\n"
		"  \"min_guest_instruction_delta\": %" PRIu64 ",\n"
		"  \"nvhe_cmdline\": %s,\n"
		"  \"samples\": %" PRIu64 ",\n"
		"  \"entries\": %" PRIu64 ",\n"
		"  \"guest_range_start\": \"0x%" PRIx64 "\",\n"
		"  \"guest_range_end\": \"0x%" PRIx64 "\",\n"
		"  \"guest_hits\": %" PRIu64 ",\n"
		"  \"host_pre_hits\": %" PRIu64 ",\n"
		"  \"host_post_hits\": %" PRIu64 ",\n"
		"  \"guest_brbe_id\": %" PRIu64 ",\n"
		"  \"guest_brbcr_undef\": %s,\n"
		"  \"guest_unexpected_exception\": %s,\n"
		"  \"lost_records\": %" PRIu64 ",\n"
		"  \"throttle_records\": %" PRIu64 ",\n"
		"  \"context_switches\": %" PRIu64 ",\n"
		"  \"host_pre_range\": \"0x%" PRIx64 "-0x%" PRIx64 "\",\n"
		"  \"host_post_range\": \"0x%" PRIx64 "-0x%" PRIx64 "\",\n"
		"  \"reason\": ",
		exit_code, options->cpu, options->iterations,
		evidence->iterations_completed, options->guest_branches,
		options->sample_period, evidence->max_pre_guest_count,
		evidence->guest_samples, evidence->guest_windows_verified,
		evidence->min_guest_instruction_delta,
		evidence->nvhe_cmdline ? "true" : "false",
		evidence->perf_samples, evidence->branch_entries,
		evidence->guest_start, evidence->guest_end,
		evidence->guest_hits, evidence->host_pre_hits,
		evidence->host_post_hits, evidence->guest_id_brbe,
		evidence->guest_brbcr_undef ? "true" : "false",
		evidence->guest_unexpected_exception ? "true" : "false",
		evidence->lost_records,
		evidence->throttle_records, evidence->context_switches,
		evidence->host_pre_start, evidence->host_pre_end,
		evidence->host_post_start, evidence->host_post_end);
	json_write_string(stream, error == NULL ? "" : error);
	fputs("\n}\n", stream);

	if (fclose(stream) != 0) {
		fprintf(stderr, "failed to close result JSON %s: %s\n",
			options->result_json_path, strerror(errno));
		return -1;
	}
	return 0;
}

static void set_error(char *error, size_t error_size, const char *format, ...)
{
	va_list arguments;

	va_start(arguments, format);
	vsnprintf(error, error_size, format, arguments);
	va_end(arguments);
}

static int parse_u64(const char *text, uint64_t minimum, uint64_t maximum,
		     uint64_t *value)
{
	char *end;
	unsigned long long parsed;

	errno = 0;
	parsed = strtoull(text, &end, 0);
	if (errno != 0 || end == text || *end != '\0' || parsed < minimum ||
	    parsed > maximum)
		return -1;
	*value = (uint64_t)parsed;
	return 0;
}

static void usage(FILE *stream, const char *program)
{
	fprintf(stream,
		"Usage: %s --cpu N [--iterations N] [--guest-branches N] "
		"[--sample-period N] --result-json PATH --entries PATH\n",
		program);
}

static int parse_options(int argc, char **argv, struct options *options)
{
	static const struct option long_options[] = {
		{ "cpu", required_argument, NULL, 'c' },
		{ "iterations", required_argument, NULL, 'i' },
		{ "guest-branches", required_argument, NULL, 'b' },
		{ "sample-period", required_argument, NULL, 'p' },
		{ "result-json", required_argument, NULL, 'j' },
		{ "entries", required_argument, NULL, 'e' },
		{ "help", no_argument, NULL, 'h' },
		{ NULL, 0, NULL, 0 },
	};
	uint64_t parsed;
	int option;
	bool cpu_seen = false;

	*options = (struct options) {
		.cpu = -1,
		.iterations = DEFAULT_ITERATIONS,
		.guest_branches = DEFAULT_GUEST_BRANCHES,
		.sample_period = DEFAULT_SAMPLE_PERIOD,
	};

	while ((option = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
		switch (option) {
		case 'c':
			if (parse_u64(optarg, 0, CPU_SETSIZE - 1, &parsed) != 0)
				return -1;
			options->cpu = (int)parsed;
			cpu_seen = true;
			break;
		case 'i':
			if (parse_u64(optarg, 1, UINT32_MAX, &options->iterations) != 0)
				return -1;
			break;
		case 'b':
			if (parse_u64(optarg, 1, UINT32_MAX,
				      &options->guest_branches) != 0)
				return -1;
			break;
		case 'p':
			if (parse_u64(optarg, PRE_GUEST_COUNT_LIMIT_DIVISOR,
				      UINT64_MAX,
				      &options->sample_period) != 0)
				return -1;
			break;
		case 'j':
			options->result_json_path = optarg;
			break;
		case 'e':
			options->entries_path = optarg;
			break;
		case 'h':
			usage(stdout, argv[0]);
			return 1;
		default:
			return -1;
		}
	}

	if (!cpu_seen || options->result_json_path == NULL ||
	    options->entries_path == NULL || optind != argc ||
	    strcmp(options->result_json_path, options->entries_path) == 0)
		return -1;
	return 0;
}

static int pin_to_cpu(int cpu, char *error, size_t error_size)
{
	cpu_set_t affinity;

	CPU_ZERO(&affinity);
	CPU_SET(cpu, &affinity);
	if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
		set_error(error, error_size, "failed to pin to CPU %d: %s", cpu,
			  strerror(errno));
		return -1;
	}
	if (sched_getcpu() != cpu) {
		set_error(error, error_size,
			  "thread did not migrate to requested CPU %d", cpu);
		return -1;
	}
	return 0;
}

static bool guest_range_overlaps_host(uint64_t start, uint64_t end,
				      char *error, size_t error_size)
{
	char line[1024];
	FILE *maps = fopen("/proc/self/maps", "r");

	if (maps == NULL) {
		set_error(error, error_size, "failed to open /proc/self/maps: %s",
			  strerror(errno));
		return true;
	}
	while (fgets(line, sizeof(line), maps) != NULL) {
		uint64_t map_start;
		uint64_t map_end;

		if (sscanf(line, "%" SCNx64 "-%" SCNx64,
			   &map_start, &map_end) != 2)
			continue;
		if (start < map_end && map_start < end) {
			set_error(error, error_size,
				  "fixed guest range 0x%" PRIx64 "-0x%" PRIx64
				  " overlaps host mapping 0x%" PRIx64 "-0x%" PRIx64,
				  start, end, map_start, map_end);
			fclose(maps);
			return true;
		}
	}
	if (ferror(maps)) {
		set_error(error, error_size, "failed to read /proc/self/maps: %s",
			  strerror(errno));
		fclose(maps);
		return true;
	}
	fclose(maps);
	return false;
}

static int setup_perf_ring(int perf_fd, long page_size, struct perf_ring *ring,
			   char *error, size_t error_size)
{
	void *mapping;

	ring->data_size = (size_t)page_size * PERF_DATA_PAGES;
	ring->mapping_size = (size_t)page_size + ring->data_size;
	mapping = mmap(NULL, ring->mapping_size, PROT_READ | PROT_WRITE,
		       MAP_SHARED, perf_fd, 0);
	if (mapping == MAP_FAILED) {
		set_error(error, error_size, "failed to mmap perf ring: %s",
			  strerror(errno));
		return -1;
	}
	ring->metadata = mapping;
	ring->data = (uint8_t *)mapping + page_size;
	return 0;
}

static void ring_copy(const struct perf_ring *ring, uint64_t offset,
		      void *destination, size_t length)
{
	size_t index = (size_t)(offset & (ring->data_size - 1));
	size_t first = ring->data_size - index;

	if (first > length)
		first = length;
	memcpy(destination, ring->data + index, first);
	if (first < length)
		memcpy((uint8_t *)destination + first, ring->data, length - first);
}

static int pull_bytes(const uint8_t **cursor, const uint8_t *end,
		      void *destination, size_t length)
{
	if ((size_t)(end - *cursor) < length)
		return -1;
	memcpy(destination, *cursor, length);
	*cursor += length;
	return 0;
}

static int parse_sample_record(const uint8_t *record, size_t record_size,
			       FILE *entries, struct evidence *evidence,
			       char *error, size_t error_size)
{
	const uint8_t *cursor = record + sizeof(struct perf_event_header);
	const uint8_t *end = record + record_size;
	uint64_t instruction_pointer;
	uint32_t pid;
	uint32_t tid;
	uint64_t time;
	uint64_t count;
	uint64_t sample_index = evidence->perf_samples;
	uint64_t entry_index;

	if (pull_bytes(&cursor, end, &instruction_pointer,
		       sizeof(instruction_pointer)) != 0 ||
	    pull_bytes(&cursor, end, &pid, sizeof(pid)) != 0 ||
	    pull_bytes(&cursor, end, &tid, sizeof(tid)) != 0 ||
	    pull_bytes(&cursor, end, &time, sizeof(time)) != 0 ||
	    pull_bytes(&cursor, end, &count, sizeof(count)) != 0) {
		set_error(error, error_size, "truncated PERF_RECORD_SAMPLE header");
		return -1;
	}
	if (count > (uint64_t)((end - cursor) /
				 sizeof(struct perf_branch_entry))) {
		set_error(error, error_size,
			  "invalid branch-stack count %" PRIu64, count);
		return -1;
	}

	evidence->last_sample_ip = instruction_pointer;
	if (address_in_range(instruction_pointer, evidence->guest_start,
			     evidence->guest_end))
		evidence->guest_samples++;
	evidence->perf_samples++;
	for (entry_index = 0; entry_index < count; entry_index++) {
		struct perf_branch_entry entry;
		bool guest_hit;
		bool pre_hit;
		bool post_hit;

		if (pull_bytes(&cursor, end, &entry, sizeof(entry)) != 0) {
			set_error(error, error_size,
				  "truncated branch entry %" PRIu64, entry_index);
			return -1;
		}
		guest_hit = branch_hits_range(&entry, evidence->guest_start,
					      evidence->guest_end);
		pre_hit = branch_hits_range(&entry, evidence->host_pre_start,
					    evidence->host_pre_end);
		post_hit = branch_hits_range(&entry, evidence->host_post_start,
					     evidence->host_post_end);
		evidence->branch_entries++;
		evidence->guest_hits += guest_hit;
		evidence->host_pre_hits += pre_hit;
		evidence->host_post_hits += post_hit;
		fprintf(entries,
			"%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%" PRIu64
			"\t0x%" PRIx64 "\t0x%" PRIx64 "\t%d\t%d\t%d\t0x%" PRIx64 "\n",
			sample_index, entry_index, pid, tid, time,
			(uint64_t)entry.from, (uint64_t)entry.to,
			guest_hit, pre_hit, post_hit, instruction_pointer);
	}

	return 0;
}

static int parse_perf_ring(struct perf_ring *ring, FILE *entries,
			   struct evidence *evidence,
			   char *error, size_t error_size)
{
	uint64_t head = __atomic_load_n(&ring->metadata->data_head,
					__ATOMIC_ACQUIRE);
	uint64_t tail = ring->metadata->data_tail;

	if (head - tail > ring->data_size) {
		set_error(error, error_size,
			  "perf ring overrun: head-tail=%" PRIu64 " size=%zu",
			  head - tail, ring->data_size);
		return -1;
	}
	while (tail < head) {
		struct perf_event_header header;
		uint8_t *record;

		if (head - tail < sizeof(header)) {
			set_error(error, error_size, "truncated perf record header");
			return -1;
		}
		ring_copy(ring, tail, &header, sizeof(header));
		if (header.size < sizeof(header) || header.size > ring->data_size ||
		    tail + header.size > head) {
			set_error(error, error_size,
				  "invalid perf record size %u", header.size);
			return -1;
		}
		record = malloc(header.size);
		if (record == NULL) {
			set_error(error, error_size, "failed to allocate perf record");
			return -1;
		}
		ring_copy(ring, tail, record, header.size);
		switch (header.type) {
		case PERF_RECORD_SAMPLE:
			if (parse_sample_record(record, header.size, entries,
						evidence, error, error_size) != 0) {
				free(record);
				return -1;
			}
			break;
		case PERF_RECORD_LOST:
			if (header.size >= sizeof(header) + 2 * sizeof(uint64_t)) {
				uint64_t lost;

				memcpy(&lost, record + sizeof(header) + sizeof(uint64_t),
				       sizeof(lost));
				evidence->lost_records += lost;
			} else {
				evidence->lost_records++;
			}
			break;
		case PERF_RECORD_THROTTLE:
		case PERF_RECORD_UNTHROTTLE:
			evidence->throttle_records++;
			break;
		default:
			break;
		}
		free(record);
		tail += header.size;
	}

	__atomic_store_n(&ring->metadata->data_tail, tail, __ATOMIC_RELEASE);
	return 0;
}

static int arm_perf_one_shot(int perf_fd, uint64_t sample_period,
			     const char *phase, char *error, size_t error_size)
{
	if (ioctl(perf_fd, PERF_EVENT_IOC_PERIOD, &sample_period) != 0) {
		set_error(error, error_size, "%s: PERF_EVENT_IOC_PERIOD failed: %s",
			  phase, strerror(errno));
		return -1;
	}
	if (ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0) != 0) {
		set_error(error, error_size, "%s: PERF_EVENT_IOC_RESET failed: %s",
			  phase, strerror(errno));
		return -1;
	}
	if (ioctl(perf_fd, PERF_EVENT_IOC_REFRESH, 1UL) != 0) {
		set_error(error, error_size,
			  "%s: PERF_EVENT_IOC_REFRESH(1) failed: %s",
			  phase, strerror(errno));
		return -1;
	}
	return 0;
}

static int stop_and_drain_perf_one_shot(int perf_fd, struct perf_ring *ring,
					FILE *entries,
					struct evidence *evidence,
					const char *phase,
					char *error, size_t error_size)
{
	char detail[512] = "";

	if (ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0) != 0) {
		set_error(error, error_size,
			  "%s: PERF_EVENT_IOC_DISABLE failed: %s",
			  phase, strerror(errno));
		return -1;
	}
	if (parse_perf_ring(ring, entries, evidence, detail, sizeof(detail)) != 0) {
		set_error(error, error_size, "%s: perf ring drain failed: %s",
			  phase, detail);
		return -1;
	}
	return 0;
}

static int require_one_new_sample(const struct evidence *evidence,
				  uint64_t samples_before, const char *phase,
				  char *error, size_t error_size)
{
	uint64_t new_samples;

	if (evidence->perf_samples < samples_before) {
		set_error(error, error_size, "%s: perf sample count regressed", phase);
		return -1;
	}
	new_samples = evidence->perf_samples - samples_before;
	if (new_samples != 1) {
		set_error(error, error_size,
			  "%s: one-shot expected exactly one new sample, observed %" PRIu64,
			  phase, new_samples);
		return -1;
	}
	return 0;
}

static int check_pre_guest_perf_state(int perf_fd, const struct perf_ring *ring,
				      uint64_t sample_period,
				      struct evidence *evidence,
				      uint64_t *instructions_before,
				      const char *phase,
				      char *error, size_t error_size)
{
	uint64_t head = __atomic_load_n(&ring->metadata->data_head,
					__ATOMIC_ACQUIRE);
	uint64_t tail = __atomic_load_n(&ring->metadata->data_tail,
					__ATOMIC_RELAXED);
	uint64_t count;
	uint64_t limit = sample_period / PRE_GUEST_COUNT_LIMIT_DIVISOR;
	ssize_t count_bytes;

	if (head != tail) {
		set_error(error, error_size,
			  "%s: perf ring is not empty before KVM_RUN: "
			  "head=%" PRIu64 " tail=%" PRIu64,
			  phase, head, tail);
		return -1;
	}

	count_bytes = read(perf_fd, &count, sizeof(count));
	if (count_bytes < 0) {
		set_error(error, error_size,
			  "%s: failed to read the pre-guest instruction count: %s",
			  phase, strerror(errno));
		return -1;
	}
	if ((size_t)count_bytes != sizeof(count)) {
		set_error(error, error_size,
			  "%s: pre-guest instruction count read returned %zd bytes",
			  phase, count_bytes);
		return -1;
	}
	if (count > evidence->max_pre_guest_count)
		evidence->max_pre_guest_count = count;
	if (count >= limit) {
		set_error(error, error_size,
			  "%s: pre-guest instruction count %" PRIu64
			  " is not below limit %" PRIu64,
			  phase, count, limit);
		return -1;
	}
	*instructions_before = count;
	return 0;
}

static int check_guest_perf_window(int perf_fd, uint64_t sample_period,
				   const struct guest_perf_window *window,
				   struct evidence *evidence, const char *phase,
				   char *error, size_t error_size)
{
	uint64_t count_after;
	uint64_t delta;
	ssize_t count_bytes;

    /* Fixed kernels retain host branches while BRBE generation is disabled in
     * guest. Require a guest-IP overflow and its instruction-counter advance.
     * A host-IP sample or an empty stack cannot prove guest sampling.
     */
    if (require_one_new_sample(evidence, window->samples, phase,
				   error, error_size) != 0)
		return -1;
	if (evidence->branch_entries <= window->entries) {
		set_error(error, error_size, "%s: sample has no new branch entries",
			  phase);
		return -1;
	}
	if (!address_in_range(evidence->last_sample_ip, evidence->guest_start,
			      evidence->guest_end)) {
		set_error(error, error_size,
			  "%s: sample IP 0x%" PRIx64 " is outside guest code",
			  phase, evidence->last_sample_ip);
		return -1;
	}
	count_bytes = read(perf_fd, &count_after, sizeof(count_after));
	if (count_bytes != (ssize_t)sizeof(count_after)) {
		if (count_bytes < 0)
			set_error(error, error_size, "%s: post-guest counter read failed: %s",
				  phase, strerror(errno));
		else
			set_error(error, error_size, "%s: post-guest counter read returned %zd bytes",
				  phase, count_bytes);
		return -1;
	}
	if (window->instructions >= sample_period ||
	    count_after < window->instructions) {
		set_error(error, error_size, "%s: inconsistent instruction counters",
			  phase);
		return -1;
	}
	delta = count_after - window->instructions;
	if (delta < sample_period - window->instructions) {
		set_error(error, error_size,
			  "%s: guest-window instruction delta %" PRIu64
			  " did not reach the remaining sample period %" PRIu64,
			  phase, delta, sample_period - window->instructions);
		return -1;
	}
	if (evidence->guest_windows_verified == 0 ||
	    delta < evidence->min_guest_instruction_delta)
		evidence->min_guest_instruction_delta = delta;
	evidence->guest_windows_verified++;
	return 0;
}

static int run_guest_iteration(struct glt_arm64_vm *vm,
			       uint64_t iteration, struct evidence *evidence,
			       char *error, size_t error_size)
{
	char run_error[256] = "";
	uint64_t mmio_value;

	if (glt_arm64_vm_run(vm, KVM_RUN_TIMEOUT_SECONDS,
			     run_error, sizeof(run_error)) != 0) {
		set_error(error, error_size,
			  "guest iteration %" PRIu64 ": %s",
			  iteration, run_error);
		return -1;
	}

	if (vm->run->exit_reason != KVM_EXIT_MMIO || !vm->run->mmio.is_write ||
	    vm->run->mmio.phys_addr != GUEST_MMIO_GPA ||
	    vm->run->mmio.len != sizeof(mmio_value)) {
		set_error(error, error_size,
			  "unexpected KVM exit at iteration %" PRIu64
			  ": reason=%u address=0x%" PRIx64
			  " write=%u len=%u", iteration, vm->run->exit_reason,
			  (uint64_t)vm->run->mmio.phys_addr,
			  vm->run->mmio.is_write, vm->run->mmio.len);
		return -1;
	}
	memcpy(&mmio_value, vm->run->mmio.data, sizeof(mmio_value));
	if (mmio_value != iteration + 1) {
		set_error(error, error_size,
			  "guest iteration mismatch: got %" PRIu64
			  " expected %" PRIu64, mmio_value, iteration + 1);
		return -1;
	}
	evidence->iterations_completed = iteration + 1;
	return 0;
}

int main(int argc, char **argv)
{
	struct options options;
	struct evidence evidence = {0};
	struct perf_event_attr perf_attributes = {0};
	struct perf_ring perf_ring = {0};
	struct glt_arm64_vm vm = {
		.kvm_fd = -1,
		.vm_fd = -1,
		.vcpu_fd = -1,
		.run = MAP_FAILED,
	};
	struct glt_kvm_memslot memslots[2];
	struct guest_mailbox *mailbox = MAP_FAILED;
	struct rusage usage_before;
	struct rusage usage_after;
	cpu_set_t original_affinity;
	FILE *entries = NULL;
	void *guest_code = MAP_FAILED;
	size_t guest_blob_size = 0;
	size_t guest_mapping_size = 0;
	long page_size = 0;
	uint64_t iteration;
	uint64_t samples_before;
	struct guest_perf_window guest_window;
	uint64_t control_hits_before;
	int perf_fd = -1;
	int exit_code = EXIT_INCONCLUSIVE;
	const char *verdict = "inconclusive";
	char phase[64];
	char error[512] = "";
	int parse_result;

	parse_result = parse_options(argc, argv, &options);
	if (parse_result != 0) {
		if (parse_result < 0)
			usage(stderr, argv[0]);
		return EXIT_INCONCLUSIVE;
	}

	evidence.host_pre_start = (uint64_t)(uintptr_t)glt_host_pre_start;
	evidence.host_pre_end = (uint64_t)(uintptr_t)glt_host_pre_end;
	evidence.host_post_start = (uint64_t)(uintptr_t)glt_host_post_start;
	evidence.host_post_end = (uint64_t)(uintptr_t)glt_host_post_end;
	guest_blob_size = (size_t)(glt_guest_end - glt_guest_start);
	evidence.guest_start = GUEST_CODE_GPA;
	evidence.guest_end = GUEST_CODE_GPA + (uint64_t)guest_blob_size;

	if (!check_nvhe_cmdline(error, sizeof(error)))
		goto finish;
	evidence.nvhe_cmdline = true;
	if (sched_getaffinity(0, sizeof(original_affinity), &original_affinity) != 0) {
		set_error(error, sizeof(error),
			  "failed to capture original CPU affinity: %s", strerror(errno));
		goto finish;
	}
	if (!CPU_ISSET(options.cpu, &original_affinity)) {
		set_error(error, sizeof(error),
			  "requested CPU %d is outside the original allowed affinity",
			  options.cpu);
		goto finish;
	}
	if (pin_to_cpu(options.cpu, error, sizeof(error)) != 0)
		goto finish;

	page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0 || ((uint64_t)page_size & ((uint64_t)page_size - 1)) != 0) {
		set_error(error, sizeof(error), "invalid host page size %ld", page_size);
		goto finish;
	}
	if ((uint64_t)page_size > GUEST_DATA_GPA - GUEST_CODE_GPA) {
		set_error(error, sizeof(error),
			  "host page size %ld overlaps fixed guest regions", page_size);
		goto finish;
	}
	guest_mapping_size = (size_t)page_size;
	if (guest_blob_size == 0 || guest_blob_size > (size_t)page_size) {
		set_error(error, sizeof(error),
			  "guest blob size %zu exceeds one host page", guest_blob_size);
		goto finish;
	}
	if (guest_range_overlaps_host(GUEST_CODE_GPA,
				      GUEST_CODE_GPA + (uint64_t)page_size,
				      error, sizeof(error)))
		goto finish;

	guest_code = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
			  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	mailbox = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
		      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (guest_code == MAP_FAILED || mailbox == MAP_FAILED) {
		set_error(error, sizeof(error), "failed to allocate guest memory: %s",
			  strerror(errno));
		goto finish;
	}
	memcpy(guest_code, glt_guest_start, guest_blob_size);
	__builtin___clear_cache(guest_code, (uint8_t *)guest_code + guest_blob_size);
	memslots[0] = (struct glt_kvm_memslot) {
		.slot = 0,
		.guest_phys_addr = GUEST_CODE_GPA,
		.memory_size = (uint64_t)page_size,
		.userspace_addr = guest_code,
	};
	memslots[1] = (struct glt_kvm_memslot) {
		.slot = 1,
		.guest_phys_addr = GUEST_DATA_GPA,
		.memory_size = (uint64_t)page_size,
		.userspace_addr = mailbox,
	};
	if (glt_arm64_vm_create(&vm, memslots, 2,
				error, sizeof(error)) != 0)
		goto finish;

	if (glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.pc),
			GUEST_CODE_GPA +
			(uint64_t)(glt_guest_entry - glt_guest_start),
			error, sizeof(error)) != 0 ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.pstate),
				 GUEST_PSTATE_EL1H | GUEST_PSTATE_DAIF,
				 error, sizeof(error)) != 0 ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(sp_el1),
				 GUEST_DATA_GPA + (uint64_t)page_size - 16,
				 error, sizeof(error)) != 0 ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.regs[0]),
				 GUEST_MMIO_GPA, error, sizeof(error)) != 0 ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.regs[1]),
				 options.guest_branches,
				 error, sizeof(error)) != 0 ||
	    glt_arm64_vm_set_reg(&vm, GLT_ARM64_CORE_REG(regs.regs[19]),
				 GUEST_DATA_GPA, error, sizeof(error)) != 0) {
		goto finish;
	}

	perf_attributes.size = sizeof(perf_attributes);
	perf_attributes.type = PERF_TYPE_HARDWARE;
	perf_attributes.config = PERF_COUNT_HW_INSTRUCTIONS;
	perf_attributes.sample_period = options.sample_period;
	perf_attributes.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID |
		PERF_SAMPLE_TIME | PERF_SAMPLE_BRANCH_STACK;
	perf_attributes.branch_sample_type = PERF_SAMPLE_BRANCH_ANY |
		PERF_SAMPLE_BRANCH_USER | PERF_SAMPLE_BRANCH_KERNEL |
		PERF_SAMPLE_BRANCH_TYPE_SAVE;
	perf_attributes.disabled = 1;
	perf_attributes.pinned = 1;
	perf_attributes.exclude_user = 0;
	perf_attributes.exclude_kernel = 0;
	perf_attributes.exclude_guest = 0;
	perf_attributes.exclude_host = 0;
	perf_attributes.exclude_hv = 1;
	perf_attributes.wakeup_events = 1;
	perf_fd = (int)syscall(__NR_perf_event_open, &perf_attributes, 0, -1, -1,
			       PERF_FLAG_FD_CLOEXEC);
	if (perf_fd < 0) {
		set_error(error, sizeof(error), "perf_event_open failed: %s",
			  strerror(errno));
		goto finish;
	}
	if (setup_perf_ring(perf_fd, page_size, &perf_ring,
			    error, sizeof(error)) != 0)
		goto finish;

	entries = fopen(options.entries_path, "w");
	if (entries == NULL) {
		set_error(error, sizeof(error), "failed to open entries file %s: %s",
			  options.entries_path, strerror(errno));
		goto finish;
	}
	fputs("sample\tentry\tpid\ttid\ttime\tfrom\tto\tguest_hit\t"
	      "host_pre_hit\thost_post_hit\tsample_ip\n", entries);

	if (getrusage(RUSAGE_THREAD, &usage_before) != 0) {
		set_error(error, sizeof(error), "getrusage before run failed: %s",
			  strerror(errno));
		goto finish;
	}

	samples_before = evidence.perf_samples;
	control_hits_before = evidence.host_pre_hits;
	if (arm_perf_one_shot(perf_fd, HOST_CONTROL_SAMPLE_PERIOD, "pre-control",
			      error, sizeof(error)) != 0)
		goto finish;
	glt_host_pre_marker(HOST_CONTROL_BRANCHES);
	if (stop_and_drain_perf_one_shot(perf_fd, &perf_ring, entries, &evidence,
					"pre-control", error, sizeof(error)) != 0)
		goto finish;
	if (require_one_new_sample(&evidence, samples_before, "pre-control",
				   error, sizeof(error)) != 0)
		goto finish;
	if (evidence.host_pre_hits == control_hits_before) {
		set_error(error, sizeof(error),
			  "pre-control: sampled BRBE stack has no pre-marker hit");
		goto finish;
	}
	if (sched_getcpu() != options.cpu) {
		set_error(error, sizeof(error),
			  "pre-control: thread migrated away from CPU %d", options.cpu);
		goto finish;
	}

	for (iteration = 0; iteration < options.iterations; iteration++) {
		snprintf(phase, sizeof(phase), "guest iteration %" PRIu64, iteration);
		guest_window.samples = evidence.perf_samples;
		guest_window.entries = evidence.branch_entries;
		if (arm_perf_one_shot(perf_fd, options.sample_period, phase,
				      error, sizeof(error)) != 0) {
			(void)ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
			goto finish;
		}
		if (check_pre_guest_perf_state(perf_fd, &perf_ring,
					       options.sample_period, &evidence,
					       &guest_window.instructions,
					       phase, error, sizeof(error)) != 0) {
			(void)ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
			goto finish;
		}
		if (run_guest_iteration(&vm, iteration, &evidence,
					 error, sizeof(error)) != 0) {
			(void)ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
			goto finish;
		}
		if (stop_and_drain_perf_one_shot(perf_fd, &perf_ring, entries,
						&evidence, phase,
						error, sizeof(error)) != 0)
			goto finish;
		if (evidence.guest_hits > 0) {
			exit_code = EXIT_CONTAMINATION;
			verdict = "contaminated";
			set_error(error, sizeof(error),
				  "%s: observed %" PRIu64
				  " BRBE entries in the guest code range",
				  phase, evidence.guest_hits);
			goto finish;
		}
		if (check_guest_perf_window(perf_fd, options.sample_period,
					    &guest_window, &evidence, phase,
					    error, sizeof(error)) != 0)
			goto finish;
		if (sched_getcpu() != options.cpu) {
			set_error(error, sizeof(error),
				  "%s: thread migrated away from CPU %d",
				  phase, options.cpu);
			goto finish;
		}
	}

	samples_before = evidence.perf_samples;
	control_hits_before = evidence.host_post_hits;
	if (arm_perf_one_shot(perf_fd, HOST_CONTROL_SAMPLE_PERIOD, "post-control",
			      error, sizeof(error)) != 0)
		goto finish;
	glt_host_post_marker(HOST_CONTROL_BRANCHES);
	if (stop_and_drain_perf_one_shot(perf_fd, &perf_ring, entries, &evidence,
					"post-control", error, sizeof(error)) != 0)
		goto finish;
	if (require_one_new_sample(&evidence, samples_before, "post-control",
				   error, sizeof(error)) != 0)
		goto finish;
	if (evidence.host_post_hits == control_hits_before) {
		set_error(error, sizeof(error),
			  "post-control: sampled BRBE stack has no post-marker hit");
		goto finish;
	}
	if (sched_getcpu() != options.cpu) {
		set_error(error, sizeof(error),
			  "post-control: thread migrated away from CPU %d", options.cpu);
		goto finish;
	}

	if (getrusage(RUSAGE_THREAD, &usage_after) != 0) {
		set_error(error, sizeof(error), "getrusage after run failed: %s",
			  strerror(errno));
		goto finish;
	}
	evidence.context_switches =
		(uint64_t)(usage_after.ru_nvcsw - usage_before.ru_nvcsw) +
		(uint64_t)(usage_after.ru_nivcsw - usage_before.ru_nivcsw);

	atomic_thread_fence(memory_order_acquire);
	if (evidence.iterations_completed != options.iterations ||
	    mailbox->iterations != options.iterations ||
	    mailbox->id_brbe != 0 ||
	    mailbox->brbcr_undef != 1 ||
	    mailbox->unexpected_exception != 0) {
		set_error(error, sizeof(error),
			  "guest isolation precondition failed: mailbox_iterations=%" PRIu64
			  " id_brbe=%" PRIu64 " brbcr_undef=%" PRIu64
			  " unexpected=%" PRIu64,
			  mailbox->iterations, mailbox->id_brbe,
			  mailbox->brbcr_undef,
			  mailbox->unexpected_exception);
		goto finish;
	}
	if (evidence.guest_hits > 0) {
		exit_code = EXIT_CONTAMINATION;
		verdict = "contaminated";
		set_error(error, sizeof(error),
			  "observed %" PRIu64
			  " BRBE entries in the guest code range",
			  evidence.guest_hits);
		goto finish;
	}
	if (evidence.lost_records != 0 || evidence.throttle_records != 0) {
		set_error(error, sizeof(error),
			  "incomplete perf evidence: lost=%" PRIu64
			  " throttle_records=%" PRIu64,
			  evidence.lost_records, evidence.throttle_records);
		goto finish;
	}
	if (evidence.perf_samples == 0 || evidence.branch_entries == 0 ||
	    evidence.host_pre_hits == 0 || evidence.host_post_hits == 0) {
		set_error(error, sizeof(error),
			  "insufficient fixed evidence: samples=%" PRIu64
			  " entries=%" PRIu64 " pre=%" PRIu64 " post=%" PRIu64,
			  evidence.perf_samples, evidence.branch_entries,
			  evidence.host_pre_hits, evidence.host_post_hits);
		goto finish;
	}

	if (evidence.guest_windows_verified != options.iterations ||
	    evidence.guest_samples != options.iterations) {
		set_error(error, sizeof(error),
			  "insufficient guest-window evidence: verified=%" PRIu64
			  " guest_samples=%" PRIu64 " expected=%" PRIu64,
			  evidence.guest_windows_verified, evidence.guest_samples,
			  options.iterations);
		goto finish;
	}
	exit_code = EXIT_FIXED_EVIDENCE;
	verdict = "fixed";
	error[0] = '\0';

finish:
	if (mailbox != MAP_FAILED) {
		atomic_thread_fence(memory_order_acquire);
		evidence.guest_id_brbe = mailbox->id_brbe;
		evidence.guest_brbcr_undef = mailbox->brbcr_undef;
		evidence.guest_unexpected_exception = mailbox->unexpected_exception;
	}
	if (entries != NULL) {
		int entries_errno = 0;
		const char *entries_operation = "flush";

		if (fflush(entries) != 0)
			entries_errno = errno;
		if (fclose(entries) != 0 && entries_errno == 0) {
			entries_errno = errno;
			entries_operation = "close";
		}
		entries = NULL;
		if (entries_errno != 0) {
			exit_code = EXIT_INCONCLUSIVE;
			verdict = "inconclusive";
			set_error(error, sizeof(error),
				  "failed to %s entries file: %s",
				  entries_operation, strerror(entries_errno));
		}
	}
	if (write_result_json(&options, &evidence, exit_code, verdict,
			      error) != 0) {
		exit_code = EXIT_INCONCLUSIVE;
		verdict = "inconclusive";
		if (error[0] == '\0')
			set_error(error, sizeof(error), "failed to write result JSON");
	}

	if (perf_ring.metadata != NULL)
		munmap(perf_ring.metadata, perf_ring.mapping_size);
	if (perf_fd >= 0)
		close(perf_fd);
	glt_arm64_vm_destroy(&vm);
	if (mailbox != MAP_FAILED)
		munmap(mailbox, guest_mapping_size);
	if (guest_code != MAP_FAILED)
		munmap(guest_code, guest_mapping_size);

	if (error[0] != '\0')
		fprintf(stderr, "%s: %s\n", verdict, error);
	else
		printf("fixed: no guest BRBE entries; host pre/post evidence present\n");
	return exit_code;
}
