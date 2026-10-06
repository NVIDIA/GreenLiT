/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
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

	.text
	.balign	64
	.global glt_host_pre_marker
	.global glt_host_pre_start
	.global glt_host_pre_end
	.type	glt_host_pre_marker, %function
glt_host_pre_marker:
	cbz	x0, 2f
glt_host_pre_start:
1:	subs	x0, x0, #1
	b.ne	1b
glt_host_pre_end:
2:	ret
	.size	glt_host_pre_marker, .-glt_host_pre_marker

	.balign	64
	.global glt_host_post_marker
	.global glt_host_post_start
	.global glt_host_post_end
	.type	glt_host_post_marker, %function
glt_host_post_marker:
	cbz	x0, 2f
glt_host_post_start:
1:	subs	x0, x0, #1
	b.ne	1b
glt_host_post_end:
2:	ret
	.size	glt_host_post_marker, .-glt_host_post_marker

	/*
	 * Everything in .guest is copied as one position-independent blob to
	 * GUEST_CODE_GPA.  The guest runs at EL1h with its MMU disabled, so its
	 * virtual branch addresses equal the fixed guest physical addresses.
	 */
	.section .guest,"ax",@progbits
	.balign	2048
	.global glt_guest_start
	.global glt_guest_vectors
glt_guest_start:
glt_guest_vectors:

	.macro	GUEST_VECTOR target
	b	\target
	.space	124
	.endm

	/* Current EL with SP0. */
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector

	/* Current EL with SPx.  KVM starts this guest in EL1h. */
	GUEST_VECTOR glt_guest_sync_handler
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector

	/* Lower EL using AArch64. */
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector

	/* Lower EL using AArch32. */
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector
	GUEST_VECTOR glt_guest_unexpected_vector

	/*
	 * Mailbox offsets, shared with struct guest_mailbox in the host source.
	 */
	.equ	MAILBOX_ID_BRBE, 0
	.equ	MAILBOX_BRBCR_UNDEF, 8
	.equ	MAILBOX_UNEXPECTED, 16
	.equ	MAILBOX_UNDEF_EXPECTED, 24
	.equ	MAILBOX_ITERATIONS, 32

	/* An expected trapped BRBCR_EL1 access arrives as EC=UNKNOWN (zero). */
glt_guest_sync_handler:
	mrs	x9, esr_el1
	lsr	x10, x9, #26
	cbnz	x10, glt_guest_sync_unexpected
	ldr	x11, [x19, #MAILBOX_UNDEF_EXPECTED]
	cbz	x11, glt_guest_sync_unexpected
	mov	x11, #1
	str	x11, [x19, #MAILBOX_BRBCR_UNDEF]
	str	xzr, [x19, #MAILBOX_UNDEF_EXPECTED]
	b	glt_guest_skip_fault

glt_guest_sync_unexpected:
	mov	x11, #1
	str	x11, [x19, #MAILBOX_UNEXPECTED]
	str	xzr, [x19, #MAILBOX_UNDEF_EXPECTED]

glt_guest_skip_fault:
	mrs	x12, elr_el1
	add	x12, x12, #4
	msr	elr_el1, x12
	eret

glt_guest_unexpected_vector:
	mov	x11, #1
	str	x11, [x19, #MAILBOX_UNEXPECTED]
	b	glt_guest_skip_fault

	.balign	64
	.global glt_guest_entry
	.type	glt_guest_entry, %function
glt_guest_entry:
	/* x0=MMIO exit GPA, x1=branches/iteration, x19=mailbox GPA. */
	mov	x20, x1
	mov	x21, xzr
	mov	x22, x0

	adr	x3, glt_guest_vectors
	msr	vbar_el1, x3
	isb

	/* ID_AA64DFR0_EL1.BRBE is bits [55:52] and must be hidden. */
	mrs	x3, id_aa64dfr0_el1
	ubfx	x3, x3, #52, #4
	str	x3, [x19, #MAILBOX_ID_BRBE]

	/*
	 * MRS x5, BRBCR_EL1.  Use the architectural encoding so that building
	 * the harness does not require a BRBE-aware assembler.
	 */
	mov	x3, #1
	str	x3, [x19, #MAILBOX_UNDEF_EXPECTED]
	.inst	0xd5319005
	/* Reaching here without the exception handler means BRBCR was exposed. */
	str	xzr, [x19, #MAILBOX_UNDEF_EXPECTED]

	.global glt_guest_branch_start
	.global glt_guest_branch_end
glt_guest_branch_start:
	/* The conditional branch depends on subs and executes x20 times. */
	mov	x4, x20
1:	subs	x4, x4, #1
	b.ne	1b
glt_guest_branch_end:
	add	x21, x21, #1
	add	x4, x19, #MAILBOX_ITERATIONS
	stlr	x21, [x4]
	str	x21, [x22]
	b	glt_guest_branch_start
	.size	glt_guest_entry, .-glt_guest_entry

	.global glt_guest_end
glt_guest_end:

	.section .note.GNU-stack,"",@progbits
