// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only

#include <linux/module.h>
#include <asm/cpufeature.h>

static int __init bbml2_capability_probe_init(void)
{
#ifdef ARM64_HAS_BBML2_NOABORT
	pr_info("glt_bbml2_capability: compiled=1 enabled=%u\n",
		system_supports_bbml2_noabort() ? 1 : 0);
#else
	pr_info("glt_bbml2_capability: compiled=0 enabled=0\n");
#endif
	return 0;
}

static void __exit bbml2_capability_probe_exit(void)
{
}

module_init(bbml2_capability_probe_init);
module_exit(bbml2_capability_probe_exit);
MODULE_DESCRIPTION("Read the running kernel BBML2 noabort capability");
MODULE_LICENSE("GPL");
