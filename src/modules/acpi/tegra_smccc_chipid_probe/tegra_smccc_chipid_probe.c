// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only

#include <linux/atomic.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/sched.h>

static atomic_t hit_counter = ATOMIC_INIT(0);
static struct task_struct *caller;

static unsigned int smccc_hits;
module_param_named(smccc_hits, smccc_hits, uint, 0444);

static unsigned int chip_id;
module_param_named(chip_id, chip_id, uint, 0444);

static int count_smccc_hit(struct kprobe *probe, struct pt_regs *regs)
{
	if (current == READ_ONCE(caller))
		atomic_inc(&hit_counter);
	return 0;
}
NOKPROBE_SYMBOL(count_smccc_hit);

static struct kprobe version_probe = {
	.symbol_name = "arm_smccc_get_soc_id_version",
	.pre_handler = count_smccc_hit,
};

static int ignore_hit(struct kprobe *probe, struct pt_regs *regs)
{
	return 0;
}
NOKPROBE_SYMBOL(ignore_hit);

static int __init greenlit_tegra_smccc_init(void)
{
	struct kprobe lookup = {
		.symbol_name = "tegra_get_chip_id",
		.pre_handler = ignore_hit,
	};
	u8 (*get_chip_id)(void);
	int ret;

	/* tegra_get_chip_id is not exported. A kprobe resolves its address. */
	ret = register_kprobe(&lookup);
	if (ret)
		return ret;
	get_chip_id = (u8 (*)(void))lookup.addr;
	unregister_kprobe(&lookup);
	if (!get_chip_id)
		return -ENOENT;

	ret = register_kprobe(&version_probe);
	if (ret)
		return ret;

	atomic_set(&hit_counter, 0);
	WRITE_ONCE(caller, current);
	chip_id = get_chip_id();
	smccc_hits = atomic_read(&hit_counter);
	WRITE_ONCE(caller, NULL);

	return 0;
}

static void __exit greenlit_tegra_smccc_exit(void)
{
	unregister_kprobe(&version_probe);
}

module_init(greenlit_tegra_smccc_init);
module_exit(greenlit_tegra_smccc_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("NVIDIA Corporation");
MODULE_DESCRIPTION("GreenLiT tegra_get_chip_id SMCCC call probe");
