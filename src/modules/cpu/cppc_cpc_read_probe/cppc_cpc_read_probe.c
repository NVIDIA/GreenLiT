// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <linux/cpu.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/sched/signal.h>
#include <linux/seq_file.h>
#include <linux/spinlock.h>
#include <linux/string.h>
#include <linux/uaccess.h>

#define GREENLIT_PROC_NAME "greenlit_cppc_cpc_read"

static DEFINE_SPINLOCK(probe_state_lock);
static bool armed;
static unsigned int injections;
static unsigned int missed_at_arm;
static pid_t target_tgid;

static int cpc_read_entry_handler(struct kretprobe_instance *ri,
				  struct pt_regs *regs)
{
	unsigned long flags;
	pid_t tgid = task_tgid_nr(current);
	int skip = 1;

	if (READ_ONCE(target_tgid) != tgid)
		return 1;

	spin_lock_irqsave(&probe_state_lock, flags);
	/* Only the first cpc_read() from the process that armed the probe. */
	if (target_tgid == tgid && armed) {
		armed = false;
		skip = 0;
	}
	spin_unlock_irqrestore(&probe_state_lock, flags);

	return skip;
}
NOKPROBE_SYMBOL(cpc_read_entry_handler);

static int cpc_read_ret_handler(struct kretprobe_instance *ri,
				struct pt_regs *regs)
{
	unsigned long flags;
	pid_t tgid = task_tgid_nr(current);

	spin_lock_irqsave(&probe_state_lock, flags);
	if (target_tgid != tgid) {
		spin_unlock_irqrestore(&probe_state_lock, flags);
		return 0;
	}

	/* Preserve a real failure and keep looking for the first successful read. */
	if ((int)regs_return_value(regs)) {
		armed = true;
		spin_unlock_irqrestore(&probe_state_lock, flags);
		return 0;
	}

	regs_set_return_value(regs, (unsigned long)-EIO);
	injections++;
	spin_unlock_irqrestore(&probe_state_lock, flags);
	return 0;
}
NOKPROBE_SYMBOL(cpc_read_ret_handler);

static struct kretprobe cpc_read_kretprobe = {
	.kp.symbol_name = "cpc_read",
	.entry_handler = cpc_read_entry_handler,
	.handler = cpc_read_ret_handler,
};

static int greenlit_cppc_show(struct seq_file *seq, void *unused)
{
	unsigned long flags;
	unsigned int observed_injections;
	unsigned int observed_misses;

	spin_lock_irqsave(&probe_state_lock, flags);
	observed_injections = injections;
	observed_misses = READ_ONCE(cpc_read_kretprobe.nmissed) - missed_at_arm;
	spin_unlock_irqrestore(&probe_state_lock, flags);
	seq_printf(seq, "%u %u\n", observed_injections, observed_misses);
	return 0;
}

static int greenlit_cppc_open(struct inode *inode, struct file *file)
{
	return single_open(file, greenlit_cppc_show, NULL);
}

static ssize_t greenlit_cppc_write(struct file *file,
				   const char __user *buffer,
				   size_t count, loff_t *position)
{
	char input[16];
	char *command;

	if (!count || count >= sizeof(input))
		return -EINVAL;
	if (copy_from_user(input, buffer, count))
		return -EFAULT;
	input[count] = '\0';
	command = strim(input);

	if (sysfs_streq(command, "arm")) {
		unsigned long flags;

		spin_lock_irqsave(&probe_state_lock, flags);
		injections = 0;
		missed_at_arm = READ_ONCE(cpc_read_kretprobe.nmissed);
		WRITE_ONCE(target_tgid, task_tgid_nr(current));
		armed = true;
		spin_unlock_irqrestore(&probe_state_lock, flags);
		return count;
	}

	if (sysfs_streq(command, "disarm")) {
		unsigned long flags;

		spin_lock_irqsave(&probe_state_lock, flags);
		armed = false;
		WRITE_ONCE(target_tgid, 0);
		spin_unlock_irqrestore(&probe_state_lock, flags);
		return count;
	}

	return -EINVAL;
}

static const struct proc_ops greenlit_cppc_ops = {
	.proc_open = greenlit_cppc_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
	.proc_write = greenlit_cppc_write,
};

static int __init greenlit_cppc_init(void)
{
	int ret;

	cpc_read_kretprobe.maxactive = max_t(int, 128, 2 * num_possible_cpus());
	ret = register_kretprobe(&cpc_read_kretprobe);
	if (ret)
		return ret;

	if (!proc_create(GREENLIT_PROC_NAME, 0600, NULL, &greenlit_cppc_ops)) {
		unregister_kretprobe(&cpc_read_kretprobe);
		return -ENOMEM;
	}

	return 0;
}

static void __exit greenlit_cppc_exit(void)
{
	unsigned long flags;

	spin_lock_irqsave(&probe_state_lock, flags);
	armed = false;
	WRITE_ONCE(target_tgid, 0);
	spin_unlock_irqrestore(&probe_state_lock, flags);
	remove_proc_entry(GREENLIT_PROC_NAME, NULL);
	unregister_kretprobe(&cpc_read_kretprobe);
}

module_init(greenlit_cppc_init);
module_exit(greenlit_cppc_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("NVIDIA Corporation");
MODULE_DESCRIPTION("GreenLiT process-scoped cpc_read error-injection probe");
