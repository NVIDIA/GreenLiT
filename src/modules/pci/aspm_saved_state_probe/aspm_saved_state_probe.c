// SPDX-License-Identifier: GPL-2.0-only

#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/pci_regs.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/spinlock.h>

#define GREENLIT_PROC_NAME "greenlit_pci_aspm_saved_state"
#define GREENLIT_LNKCTL_MASK (PCI_EXP_LNKCTL_ASPMC | PCI_EXP_LNKCTL_CLKREQ_EN)

/*
 * Mirrors the leading fields of drivers/pci/pcie/aspm.c:struct
 * pcie_link_state used by pcie_config_aspm_link(). Keep this in sync with
 * that private kernel struct.
 */
struct greenlit_pcie_link_state {
	struct pci_dev *pdev;
	struct pci_dev *downstream;
};

/*
 * Mirrors drivers/pci/pci.h:struct pci_cap_saved_state so the probe can read
 * the PCIe saved capability without relying on non-exported helpers.
 */
struct greenlit_pci_cap_saved_data {
	u16 cap_nr;
	bool cap_extended;
	unsigned int size;
	u32 data[];
};

struct greenlit_pci_cap_saved_state {
	struct hlist_node next;
	struct greenlit_pci_cap_saved_data cap;
};

struct greenlit_lnkctl_snapshot {
	bool valid;
	u16 value;
};

struct greenlit_aspm_retprobe_data {
	struct greenlit_pcie_link_state *link;
	struct greenlit_lnkctl_snapshot pdev_lnkctl;
	struct greenlit_lnkctl_snapshot downstream_lnkctl;
};

static unsigned long long config_hit_count;
static unsigned long long transition_hit_count;
static unsigned long long saved_state_hit_count;
static unsigned long long saved_state_inspection_count;
static unsigned long long saved_state_mismatch_count;
static DEFINE_RAW_SPINLOCK(greenlit_counter_lock);

static void *greenlit_first_arg(struct pt_regs *regs)
{
#if defined(CONFIG_ARM64)
	return (void *)regs->regs[0];
#elif defined(CONFIG_X86_64)
	return (void *)regs->di;
#else
	pr_warn_once("unsupported architecture for pcie_config_aspm_link probe "
		     "argument decoding\n");
	return NULL;
#endif
}

static int greenlit_saved_lnkctl(struct pci_dev *dev, u16 *lnkctl)
{
	struct greenlit_pci_cap_saved_state *tmp;
	u16 *cap;

	hlist_for_each_entry(tmp, &dev->saved_cap_space, next) {
		if (tmp->cap.cap_extended || tmp->cap.cap_nr != PCI_CAP_ID_EXP)
			continue;

		if (tmp->cap.size < 2 * sizeof(u16))
			continue;

		cap = (u16 *)&tmp->cap.data[0];
		*lnkctl = cap[1];
		return 0;
	}

	return -ENOENT;
}

static struct greenlit_lnkctl_snapshot greenlit_live_lnkctl(struct pci_dev *dev)
{
	struct greenlit_lnkctl_snapshot snapshot = {};

	/*
	 * PCI config reads are safe from this probe context; the PCI accessors
	 * serialize through raw spinlocks, as other atomic PCI paths do.
	 */
	if (dev && !pcie_capability_read_word(dev, PCI_EXP_LNKCTL,
					      &snapshot.value))
		snapshot.valid = true;

	return snapshot;
}

static void greenlit_counter_inc(unsigned long long *counter)
{
	unsigned long flags;

	raw_spin_lock_irqsave(&greenlit_counter_lock, flags);
	(*counter)++;
	raw_spin_unlock_irqrestore(&greenlit_counter_lock, flags);
}

static void greenlit_count_transition(
	struct greenlit_lnkctl_snapshot before,
	struct greenlit_lnkctl_snapshot after)
{
	if (!before.valid || !after.valid)
		return;

	if ((before.value & GREENLIT_LNKCTL_MASK) !=
	    (after.value & GREENLIT_LNKCTL_MASK))
		greenlit_counter_inc(&transition_hit_count);
}

static void greenlit_check_saved_lnkctl(struct pci_dev *dev, u16 current_lnkctl)
{
	u16 saved_lnkctl;

	if (!dev || greenlit_saved_lnkctl(dev, &saved_lnkctl))
		return;

	greenlit_counter_inc(&saved_state_inspection_count);
	if ((current_lnkctl & GREENLIT_LNKCTL_MASK) ==
	    (saved_lnkctl & GREENLIT_LNKCTL_MASK))
		greenlit_counter_inc(&saved_state_hit_count);
	else
		greenlit_counter_inc(&saved_state_mismatch_count);
}

static int aspm_config_entry_handler(struct kretprobe_instance *ri,
				     struct pt_regs *regs)
{
	struct greenlit_aspm_retprobe_data *data;

	data = (struct greenlit_aspm_retprobe_data *)ri->data;
	data->link = greenlit_first_arg(regs);
	if (!data->link)
		return 0;

	data->pdev_lnkctl = greenlit_live_lnkctl(data->link->pdev);
	data->downstream_lnkctl = greenlit_live_lnkctl(data->link->downstream);
	return 0;
}
NOKPROBE_SYMBOL(aspm_config_entry_handler);

static int aspm_config_ret_handler(struct kretprobe_instance *ri,
				   struct pt_regs *regs)
{
	struct greenlit_aspm_retprobe_data *data;
	struct greenlit_pcie_link_state *link;
	struct greenlit_lnkctl_snapshot pdev_lnkctl;
	struct greenlit_lnkctl_snapshot downstream_lnkctl;

	data = (struct greenlit_aspm_retprobe_data *)ri->data;
	link = data->link;
	if (!link)
		return 0;

	pdev_lnkctl = greenlit_live_lnkctl(link->pdev);
	downstream_lnkctl = greenlit_live_lnkctl(link->downstream);

	greenlit_counter_inc(&config_hit_count);
	greenlit_count_transition(data->pdev_lnkctl, pdev_lnkctl);
	greenlit_count_transition(data->downstream_lnkctl, downstream_lnkctl);
	if (pdev_lnkctl.valid)
		greenlit_check_saved_lnkctl(link->pdev, pdev_lnkctl.value);
	if (downstream_lnkctl.valid)
		greenlit_check_saved_lnkctl(link->downstream,
					    downstream_lnkctl.value);
	return 0;
}
NOKPROBE_SYMBOL(aspm_config_ret_handler);

static struct kretprobe aspm_config_kretprobe = {
	.kp.symbol_name = "pcie_config_aspm_link",
	.entry_handler = aspm_config_entry_handler,
	.handler = aspm_config_ret_handler,
	.data_size = sizeof(struct greenlit_aspm_retprobe_data),
	/* Missed instances are exported through procfs and cause conservative skips. */
	.maxactive = 64,
};

static int greenlit_aspm_saved_state_show(struct seq_file *seq, void *v)
{
	unsigned long flags;
	unsigned long long config_hits;
	unsigned long long transition_hits;
	unsigned long long saved_state_hits;
	unsigned long long saved_state_inspections;
	unsigned long long saved_state_mismatches;

	raw_spin_lock_irqsave(&greenlit_counter_lock, flags);
	config_hits = config_hit_count;
	transition_hits = transition_hit_count;
	saved_state_hits = saved_state_hit_count;
	saved_state_inspections = saved_state_inspection_count;
	saved_state_mismatches = saved_state_mismatch_count;
	raw_spin_unlock_irqrestore(&greenlit_counter_lock, flags);

	seq_printf(seq, "config_hits=%llu\n",
		   config_hits);
	seq_printf(seq, "transition_hits=%llu\n",
		   transition_hits);
	seq_printf(seq, "saved_state_hits=%llu\n",
		   saved_state_hits);
	seq_printf(seq, "saved_state_inspections=%llu\n",
		   saved_state_inspections);
	seq_printf(seq, "saved_state_mismatches=%llu\n",
		   saved_state_mismatches);
	seq_printf(seq, "probe_missed_instances=%lu\n",
		   aspm_config_kretprobe.nmissed);
	return 0;
}

static int __init aspm_saved_state_probe_init(void)
{
	int ret;

	ret = register_kretprobe(&aspm_config_kretprobe);
	if (ret < 0)
		return ret;

	if (!proc_create_single(GREENLIT_PROC_NAME, 0600, NULL,
				greenlit_aspm_saved_state_show)) {
		unregister_kretprobe(&aspm_config_kretprobe);
		return -ENOMEM;
	}

	return 0;
}

static void __exit aspm_saved_state_probe_exit(void)
{
	remove_proc_entry(GREENLIT_PROC_NAME, NULL);
	unregister_kretprobe(&aspm_config_kretprobe);
}

module_init(aspm_saved_state_probe_init);
module_exit(aspm_saved_state_probe_exit);

MODULE_AUTHOR("NVIDIA Corporation");
MODULE_DESCRIPTION("GreenLiT PCI ASPM saved-state update probe");
MODULE_LICENSE("GPL");
