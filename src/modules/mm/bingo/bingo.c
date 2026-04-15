/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
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
