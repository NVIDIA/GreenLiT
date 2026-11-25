// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/kthread.h>
#include <linux/completion.h>
#include <linux/delay.h>
#include <linux/smp.h>
#include <linux/cpu.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>
#include <linux/jiffies.h>

#define GLT_NMI_IOCTL_TRIGGER _IOW('G', 0x70, struct glt_nmi_trigger)
#define GLT_NMI_MAX_IRQOFF_MS 25000
#define GLT_NMI_SLICE_MS 10

struct glt_nmi_trigger {
	__u32 cpu;
	__u32 duration_ms;
	__u32 armed;
	__u32 reserved;
};

static DEFINE_MUTEX(glt_nmi_trigger_lock);
static struct task_struct *glt_nmi_task;
static DECLARE_COMPLETION(glt_nmi_done);

static int glt_nmi_spin_thread(void *data)
{
	struct glt_nmi_trigger *trigger = data;
	unsigned long flags;
	u32 elapsed = 0;

	pr_info("glt_nmi_watchdog: start cpu=%u duration_ms=%u\n",
		trigger->cpu, trigger->duration_ms);
	local_irq_save(flags);
	while (elapsed < trigger->duration_ms && !kthread_should_stop()) {
		mdelay(GLT_NMI_SLICE_MS);
		elapsed += GLT_NMI_SLICE_MS;
	}
	local_irq_restore(flags);
	pr_info("glt_nmi_watchdog: end cpu=%u\n", trigger->cpu);
	complete(&glt_nmi_done);
	return 0;
}

static long glt_nmi_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct glt_nmi_trigger trigger;
	struct task_struct *task;
	long remaining;
	int ret = 0;

	if (cmd != GLT_NMI_IOCTL_TRIGGER)
		return -ENOTTY;
	if (copy_from_user(&trigger, (void __user *)arg, sizeof(trigger)))
		return -EFAULT;
	if (trigger.armed != 1 || !trigger.duration_ms ||
	    trigger.duration_ms > GLT_NMI_MAX_IRQOFF_MS)
		return -EINVAL;
	if (trigger.cpu >= nr_cpu_ids)
		return -EINVAL;

	mutex_lock(&glt_nmi_trigger_lock);
	if (glt_nmi_task) {
		mutex_unlock(&glt_nmi_trigger_lock);
		return -EBUSY;
	}
	cpu_hotplug_disable();
	if (!cpu_online(trigger.cpu)) {
		cpu_hotplug_enable();
		mutex_unlock(&glt_nmi_trigger_lock);
		return -EINVAL;
	}

	reinit_completion(&glt_nmi_done);
	task = kthread_create(glt_nmi_spin_thread, &trigger, "glt_nmi_spin_thread");
	if (IS_ERR(task)) {
		cpu_hotplug_enable();
		mutex_unlock(&glt_nmi_trigger_lock);
		return PTR_ERR(task);
	}
	kthread_bind(task, trigger.cpu);
	glt_nmi_task = task;
	wake_up_process(task);

	remaining = wait_for_completion_timeout(
		&glt_nmi_done,
		msecs_to_jiffies(trigger.duration_ms + 2000));
	if (!remaining)
		ret = -ETIMEDOUT;
	kthread_stop(task);
	glt_nmi_task = NULL;
	cpu_hotplug_enable();
	mutex_unlock(&glt_nmi_trigger_lock);
	return ret;
}

static const struct file_operations glt_nmi_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = glt_nmi_ioctl,
	.llseek = noop_llseek,
};

static struct miscdevice glt_nmi_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "glt_vera_nmi_watchdog",
	.fops = &glt_nmi_fops,
	.mode = 0600,
};

static int __init glt_nmi_init(void)
{
	return misc_register(&glt_nmi_misc);
}

static void __exit glt_nmi_exit(void)
{
	mutex_lock(&glt_nmi_trigger_lock);
	if (glt_nmi_task) {
		kthread_stop(glt_nmi_task);
		glt_nmi_task = NULL;
	}
	mutex_unlock(&glt_nmi_trigger_lock);
	misc_deregister(&glt_nmi_misc);
}

module_init(glt_nmi_init);
module_exit(glt_nmi_exit);
MODULE_DESCRIPTION("Bounded IRQ-off hardlockup probe for Vera NMI validation");
MODULE_LICENSE("GPL");
