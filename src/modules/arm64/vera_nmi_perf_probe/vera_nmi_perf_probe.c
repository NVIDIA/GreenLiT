// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/kthread.h>
#include <linux/completion.h>
#include <linux/smp.h>
#include <linux/cpu.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>
#include <linux/perf_event.h>
#include <linux/interrupt.h>
#include <linux/time64.h>
#include <asm/ptrace.h>
#include <asm/sysreg.h>

#define GLT_NMI_PERF_IOCTL_TRIGGER _IOWR('G', 0x71, struct glt_nmi_perf_trigger)
#define GLT_NMI_PERF_MAX_IRQOFF_MS 1000
#define GLT_NMI_PERF_SAMPLE_PERIOD 100000

struct glt_nmi_perf_trigger {
	__u32 cpu;
	__u32 duration_ms;
	__u32 armed;
	__u32 reserved;
	__u32 overflows;
	__u32 nmi_overflows;
	__u32 irqoff_overflows;
	__u32 irqoff_nmi_overflows;
	__u32 module_irqoff_nmi_overflows;
};

struct glt_nmi_perf_request {
	struct glt_nmi_perf_trigger *trigger;
	struct perf_event *event;
	int ret;
	atomic_t overflows;
	atomic_t nmi_overflows;
	atomic_t irqoff_overflows;
	atomic_t irqoff_nmi_overflows;
	atomic_t module_irqoff_nmi_overflows;
};

static DEFINE_MUTEX(glt_nmi_perf_trigger_lock);
static struct task_struct *glt_nmi_perf_task;
static DECLARE_COMPLETION(glt_nmi_perf_done);

static void glt_nmi_perf_overflow(struct perf_event *event,
					 struct perf_sample_data *data,
					 struct pt_regs *regs)
{
	struct glt_nmi_perf_request *request = event->overflow_handler_context;
	bool nmi = in_nmi();
	bool irqoff = !interrupts_enabled(regs);
	unsigned long pc = instruction_pointer(regs);

	atomic_inc(&request->overflows);
	if (nmi)
		atomic_inc(&request->nmi_overflows);
	if (!irqoff)
		return;

	atomic_inc(&request->irqoff_overflows);
	if (!nmi)
		return;

	atomic_inc(&request->irqoff_nmi_overflows);
	if (within_module(pc, THIS_MODULE))
		atomic_inc(&request->module_irqoff_nmi_overflows);
}

static int glt_nmi_perf_thread(void *data)
{
	struct glt_nmi_perf_request *request = data;
	struct glt_nmi_perf_trigger *trigger = request->trigger;
	struct perf_event_attr attr = {
		.type = PERF_TYPE_HARDWARE,
		.size = sizeof(attr),
		.config = PERF_COUNT_HW_INSTRUCTIONS,
		.sample_period = GLT_NMI_PERF_SAMPLE_PERIOD,
		.pinned = 1,
		.disabled = 1,
		.exclude_user = 1,
		.exclude_hv = 1,
	};
	unsigned long flags;
	u64 frequency;
	u64 start;
	u64 duration_ticks;
	int release_ret;

	if (smp_processor_id() != trigger->cpu) {
		request->ret = -EINVAL;
		goto out_complete;
	}

	request->event = perf_event_create_kernel_counter(
		&attr, trigger->cpu, NULL, glt_nmi_perf_overflow, request);
	if (IS_ERR(request->event)) {
		request->ret = PTR_ERR(request->event);
		request->event = NULL;
		goto out_complete;
	}

	frequency = read_sysreg(cntfrq_el0);
	if (!frequency) {
		request->ret = -EINVAL;
		goto out_release;
	}

	perf_event_enable(request->event);
	duration_ticks = frequency * trigger->duration_ms / MSEC_PER_SEC;
	start = read_sysreg(cntvct_el0);
	local_irq_save(flags);
	while ((read_sysreg(cntvct_el0) - start) < duration_ticks &&
	       !kthread_should_stop())
		cpu_relax();
	local_irq_restore(flags);
	perf_event_disable(request->event);

out_release:
	release_ret = perf_event_release_kernel(request->event);
	if (!request->ret)
		request->ret = release_ret;
	request->event = NULL;

out_complete:
	complete(&glt_nmi_perf_done);
	return 0;
}

static void glt_nmi_perf_store_results(struct glt_nmi_perf_request *request)
{
	struct glt_nmi_perf_trigger *trigger = request->trigger;

	trigger->overflows = atomic_read(&request->overflows);
	trigger->nmi_overflows = atomic_read(&request->nmi_overflows);
	trigger->irqoff_overflows = atomic_read(&request->irqoff_overflows);
	trigger->irqoff_nmi_overflows = atomic_read(&request->irqoff_nmi_overflows);
	trigger->module_irqoff_nmi_overflows =
		atomic_read(&request->module_irqoff_nmi_overflows);
}

static long glt_nmi_perf_ioctl(struct file *file, unsigned int cmd,
			       unsigned long arg)
{
	struct glt_nmi_perf_trigger trigger;
	struct glt_nmi_perf_request request;
	struct task_struct *task;
	long remaining;
	int ret = 0;

	if (cmd != GLT_NMI_PERF_IOCTL_TRIGGER)
		return -ENOTTY;
	if (copy_from_user(&trigger, (void __user *)arg, sizeof(trigger)))
		return -EFAULT;
	if (trigger.armed != 1 || !trigger.duration_ms || trigger.reserved ||
	    trigger.duration_ms > GLT_NMI_PERF_MAX_IRQOFF_MS ||
	    trigger.cpu >= nr_cpu_ids)
		return -EINVAL;

	memset(&request, 0, sizeof(request));
	request.trigger = &trigger;
	mutex_lock(&glt_nmi_perf_trigger_lock);
	if (glt_nmi_perf_task) {
		mutex_unlock(&glt_nmi_perf_trigger_lock);
		return -EBUSY;
	}
	cpu_hotplug_disable();
	if (!cpu_online(trigger.cpu)) {
		ret = -EINVAL;
		goto out_hotplug;
	}

	reinit_completion(&glt_nmi_perf_done);
	task = kthread_create(glt_nmi_perf_thread, &request,
			      "glt_nmi_perf_thread");
	if (IS_ERR(task)) {
		ret = PTR_ERR(task);
		goto out_hotplug;
	}
	kthread_bind(task, trigger.cpu);
	glt_nmi_perf_task = task;
	wake_up_process(task);

	remaining = wait_for_completion_timeout(
		&glt_nmi_perf_done,
		msecs_to_jiffies(trigger.duration_ms + 2000));
	if (!remaining)
		ret = -ETIMEDOUT;
	kthread_stop(task);
	glt_nmi_perf_task = NULL;
	if (!ret)
		ret = request.ret;

out_hotplug:
	cpu_hotplug_enable();
	mutex_unlock(&glt_nmi_perf_trigger_lock);
	if (ret)
		return ret;

	glt_nmi_perf_store_results(&request);
	if (copy_to_user((void __user *)arg, &trigger, sizeof(trigger)))
		return -EFAULT;
	return 0;
}

static const struct file_operations glt_nmi_perf_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = glt_nmi_perf_ioctl,
	.llseek = noop_llseek,
};

static struct miscdevice glt_nmi_perf_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "glt_vera_nmi_perf",
	.fops = &glt_nmi_perf_fops,
	.mode = 0600,
};

static int __init glt_nmi_perf_init(void)
{
	return misc_register(&glt_nmi_perf_misc);
}

static void __exit glt_nmi_perf_exit(void)
{
	mutex_lock(&glt_nmi_perf_trigger_lock);
	if (glt_nmi_perf_task) {
		kthread_stop(glt_nmi_perf_task);
		glt_nmi_perf_task = NULL;
	}
	mutex_unlock(&glt_nmi_perf_trigger_lock);
	misc_deregister(&glt_nmi_perf_misc);
}

module_init(glt_nmi_perf_init);
module_exit(glt_nmi_perf_exit);
MODULE_DESCRIPTION("Bounded IRQ-off pseudo-NMI PMU overflow probe for Vera");
MODULE_LICENSE("GPL");
