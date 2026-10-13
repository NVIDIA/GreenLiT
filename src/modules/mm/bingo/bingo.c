/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include <linux/module.h>
#include <linux/kernel.h>

/* TODO: Make an huge text section by padding NOP */
volatile char hugebss[512 * 1024 * 1024];       /* 512M .bss */
const char hugerodata[512 * 1024 * 1024];	/* 512M .roddata */

static int __init init_bingo(void)
{
	pr_info("Hello Bingo!\n");
	return 0;
}

static void __exit exit_bingo(void)
{
	pr_info("Goodbye Bingo!\n");
}

module_init(init_bingo);
module_exit(exit_bingo);

MODULE_AUTHOR("Matthew R. Ochs");
MODULE_DESCRIPTION("Bingo test module");
MODULE_LICENSE("GPL");
