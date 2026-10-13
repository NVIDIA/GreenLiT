// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/i2c.h>
#include <linux/proc_fs.h>

static bool greenlit = true;

/* Copy args for usage on function return */
static int entry_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	memcpy(ri->data, regs, sizeof(struct pt_regs));
	return 0;
}
NOKPROBE_SYMBOL(entry_handler);

/*
 * Take advantage of the change from fixing commit:
 * msgs[i].len = msgs[i].buf[0]; // Wrong approach
 * msgs[i].len += msgs[i].buf[0]; // Correct approach
 * We can take advantage of the difference, i.e. the fix isn't in place when
 * msgs[i].len equals to msgs[i].buf[0].
 */
static int ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	struct pt_regs *args = (struct pt_regs *)ri->data;
	struct i2c_msg *msgs = (struct i2c_msg *)args->regs[1];
	int num = args->regs[2];
	int i;

	for (i = 0; i < num; i++) {
		if (msgs[i].flags & I2C_M_RECV_LEN &&
		    msgs[i].len == msgs[i].buf[0]) {
			greenlit = false;
			break;
		}
	}

	return 0;
}
NOKPROBE_SYMBOL(ret_handler);

static struct kretprobe glt_kretprobe = {
	.kp.symbol_name		= "tegra_i2c_xfer",
	.handler		= ret_handler,
	.entry_handler		= entry_handler,
	.data_size		= sizeof(struct pt_regs),
	.maxactive		= 20,
};

static int greenlit_show(struct seq_file *seq, void *v)
{
	seq_printf(seq, "%d\n", greenlit);
	return 0;
}

static int __init kretprobe_init(void)
{
	int ret;

	ret = register_kretprobe(&glt_kretprobe);
	if (ret < 0) {
		return ret;
	}

	if (!proc_create_single("greenlit", 0600, NULL, greenlit_show))
		unregister_kretprobe(&glt_kretprobe);

	return 0;
}

static void __exit kretprobe_exit(void)
{
	remove_proc_entry("greenlit", NULL);
	unregister_kretprobe(&glt_kretprobe);
}

module_init(kretprobe_init)
module_exit(kretprobe_exit)
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Kai-Heng Feng <kaihengf@nvidia.com>");
