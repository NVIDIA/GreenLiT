// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only
/*
 * Exercise paired L1SS software-cache saves without restoring PCI hardware.
 */

#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/pci_regs.h>
#include <linux/pm_runtime.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/suspend.h>
#include <linux/uaccess.h>

#define GREENLIT_PROC_NAME "greenlit_pci_aspm_l1ss_saved_state"

enum greenlit_partner { CHILD, PARENT, PARTNERS };
enum greenlit_register { CTL1, CTL2, CONTROLS };

/*
 * Mirrors drivers/pci/pci.h, as in the adjacent aspm_saved_state_probe helper.
 * L1SS uses exactly two u32s, stored in CTL2, CTL1 order. Reject other sizes
 * before touching a buffer. Keep this layout in sync with the kernel struct.
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

struct greenlit_cache {
	struct pci_saved_state *data;
	bool state_saved;
};

struct greenlit_pair {
	struct pci_dev *dev[PARTNERS];
	u32 *buffer[PARTNERS];
	bool device_locked[PARTNERS];
	bool cfg_locked[PARTNERS];
	bool pm_pinned[PARTNERS];
};

struct greenlit_result {
	char mode[sizeof("parent_saved_first")];
	char child[16];
	char parent[16];
	u32 before[PARTNERS][CONTROLS];
	u32 saved[PARTNERS][CONTROLS];
	u32 after[PARTNERS][CONTROLS];
	int capture_error;
	int restore_error;
};

static struct greenlit_result result;
static DEFINE_MUTEX(run_lock);

/**
 * greenlit_l1ss_buffer - Locate the existing two-dword L1SS software buffer.
 * @dev: Device protected by the caller's device and PCI rescan/remove locks.
 *
 * Return: Buffer in kernel CTL2/CTL1 order, or NULL for an absent/other layout.
 */
static u32 *greenlit_l1ss_buffer(struct pci_dev *dev)
{
	struct greenlit_pci_cap_saved_state *entry;

	hlist_for_each_entry(entry, &dev->saved_cap_space, next) {
		if (!entry->cap.cap_extended ||
		    entry->cap.cap_nr != PCI_EXT_CAP_ID_L1SS)
			continue;
		if (entry->cap.size != CONTROLS * sizeof(u32))
			return NULL;
		return entry->cap.data;
	}
	return NULL;
}

/**
 * greenlit_buffer_snapshot - Read a software buffer in named control order.
 * @buffer: Two-dword buffer in the kernel's CTL2/CTL1 order.
 * @values: Output array indexed by CTL1 and CTL2.
 */
static void greenlit_buffer_snapshot(u32 *buffer, u32 values[CONTROLS])
{
	values[CTL1] = READ_ONCE(buffer[1]);
	values[CTL2] = READ_ONCE(buffer[0]);
}

/**
 * greenlit_poison_buffer - Replace software cache bytes with complements.
 * @buffer: Existing two-dword software buffer; no hardware is accessed.
 * @before: Live control values read before the exercised saves.
 */
static void greenlit_poison_buffer(u32 *buffer, const u32 before[CONTROLS])
{
	WRITE_ONCE(buffer[0], ~before[CTL2]);
	WRITE_ONCE(buffer[1], ~before[CTL1]);
}

/**
 * greenlit_preserve - Copy original software save buffers and their flag.
 * @dev: Runtime-PM-pinned device under the caller's device and rescan locks.
 * @cache: Storage for the copied buffers and original state_saved flag.
 *
 * Return: Zero on success, or -ENOMEM if the snapshot cannot be allocated.
 */
static int greenlit_preserve(struct pci_dev *dev, struct greenlit_cache *cache)
{
	cache->state_saved = dev->state_saved;
	/* This exported API copies software caches only when state_saved is set. */
	dev->state_saved = true;
	cache->data = pci_store_saved_state(dev);
	dev->state_saved = cache->state_saved;
	return cache->data ? 0 : -ENOMEM;
}

/**
 * greenlit_restore_cache - Restore and free the original software snapshot.
 * @dev: Device whose software buffers and flag are restored.
 * @cache: Snapshot consumed by this call; an absent snapshot is harmless.
 *
 * Return: Zero on success, otherwise the pci_load_saved_state() error.
 */
static int greenlit_restore_cache(struct pci_dev *dev,
				  struct greenlit_cache *cache)
{
	int ret;

	if (!cache->data)
		return 0;
	/* Never call pci_restore_state(): this updates software buffers only. */
	ret = pci_load_saved_state(dev, cache->data);
	dev->state_saved = cache->state_saved;
	kfree(cache->data);
	cache->data = NULL;
	return ret;
}

/**
 * greenlit_port_driver - Recognize unbound devices and PCIe port drivers.
 * @dev: Device whose driver and PCIe type are stable under its device lock.
 *
 * Return: True for an unbound device or a PCIe port bound to pcieport.
 */
static bool greenlit_port_driver(struct pci_dev *dev)
{
	return !dev->driver ||
		((pci_pcie_type(dev) == PCI_EXP_TYPE_UPSTREAM ||
		  pci_pcie_type(dev) == PCI_EXP_TYPE_DOWNSTREAM ||
		  pci_pcie_type(dev) == PCI_EXP_TYPE_ROOT_PORT) &&
		 !strcmp(dev->driver->name, "pcieport"));
}

/**
 * greenlit_d0 - Check runtime, cached, and available hardware power state.
 * @dev: Runtime-PM-pinned device to inspect without changing its power state.
 *
 * Return: True for active D0 with a matching PMCSR when the capability exists.
 */
static bool greenlit_d0(struct pci_dev *dev)
{
	u16 pmcsr;

	if (!pm_runtime_active(&dev->dev) || dev->current_state != PCI_D0)
		return false;
	if (!dev->pm_cap)
		return true;
	return !pci_read_config_word(dev, dev->pm_cap + PCI_PM_CTRL, &pmcsr) &&
		(pmcsr & PCI_PM_CTRL_STATE_MASK) == PCI_D0;
}

/**
 * greenlit_controls - Read both L1SS controls from a present device.
 * @dev: Device with an L1SS capability to inspect.
 * @values: Output array indexed by CTL1 and CTL2.
 *
 * Return: Zero on success, a PCI error, or -ENODEV for inaccessible hardware.
 */
static int greenlit_controls(struct pci_dev *dev, u32 values[CONTROLS])
{
	int ret;

	if (!pci_device_is_present(dev))
		return -ENODEV;
	ret = pci_read_config_dword(dev, dev->l1ss + PCI_L1SS_CTL1,
				    &values[CTL1]);
	if (ret)
		return ret;
	ret = pci_read_config_dword(dev, dev->l1ss + PCI_L1SS_CTL2,
				    &values[CTL2]);
	if (!ret && (values[CTL1] == U32_MAX || values[CTL2] == U32_MAX))
		return -ENODEV;
	return ret;
}

/**
 * greenlit_release_pair - Drop a candidate's locks, PM pins and references.
 * @pair: Candidate whose acquired resources are tracked individually.
 */
static void greenlit_release_pair(struct greenlit_pair *pair)
{
	unsigned int partner;

	for (partner = 0; partner < PARTNERS; partner++) {
		if (pair->cfg_locked[partner])
			pci_cfg_access_unlock(pair->dev[partner]);
		if (pair->pm_pinned[partner])
			pm_runtime_put_noidle(&pair->dev[partner]->dev);
		if (pair->device_locked[partner])
			device_unlock(&pair->dev[partner]->dev);
		pci_dev_put(pair->dev[partner]);
	}
	memset(pair, 0, sizeof(*pair));
}

/**
 * greenlit_lock_pair - Check and lock an L1SS child/parent candidate.
 * @pair: Referenced devices to pin and lock, parent before child.
 * @bound_endpoint: Whether this pass accepts a driver-bound endpoint child.
 *
 * Locks serialize normal driver-core, PM and removal paths. An endpoint's
 * private worker is not universally serialized by device_lock(); these locks
 * cannot guarantee the absence of arbitrary out-of-contract driver activity.
 * Return: Zero with locks held, or a standard pre-run eligibility error.
 */
static int greenlit_lock_pair(struct greenlit_pair *pair, bool bound_endpoint)
{
	unsigned int partner;
	bool is_bound_endpoint;

	for (partner = PARTNERS; partner-- > 0;) {
		struct pci_dev *dev = pair->dev[partner];

		if (!device_trylock(&dev->dev))
			return -EBUSY;
		pair->device_locked[partner] = true;
		if (!greenlit_port_driver(dev) &&
		    (partner != CHILD || pci_pcie_type(dev) != PCI_EXP_TYPE_ENDPOINT))
			return -EOPNOTSUPP;
		pm_runtime_get_noresume(&dev->dev);
		pair->pm_pinned[partner] = true;
		if (!greenlit_d0(dev))
			return -EBUSY;
		if (!pci_cfg_access_trylock(dev))
			return -EBUSY;
		pair->cfg_locked[partner] = true;
		pair->buffer[partner] = greenlit_l1ss_buffer(dev);
		if (!pair->buffer[partner])
			return -EOPNOTSUPP;
	}
	is_bound_endpoint = pair->dev[CHILD]->driver &&
		pci_pcie_type(pair->dev[CHILD]) == PCI_EXP_TYPE_ENDPOINT;
	if (is_bound_endpoint != bound_endpoint)
		return -EAGAIN;
	return 0;
}

/**
 * greenlit_select_pair - Find a complete, lockable L1SS pair in the kernel.
 * @pair: Output referenced, locked and PM-pinned devices and software buffers.
 *
 * Prefer unbound children or pcieport switch ports; fall back to bound D0
 * endpoints. The caller holds system-sleep and PCI rescan/remove locks.
 * Return: Zero, or ENODEV/EBUSY/EOPNOTSUPP when no usable pair can be selected.
 */
static int greenlit_select_pair(struct greenlit_pair *pair)
{
	struct pci_dev *child = NULL;
	unsigned int pass;
	int unavailable = -ENODEV;

	for (pass = 0; pass < 2; pass++) {
		for_each_pci_dev(child) {
			struct pci_dev *parent = child->bus->self;
			int ret;

			if (!parent || !pci_is_pcie(child) || !pci_is_pcie(parent) ||
			    !child->l1ss || !parent->l1ss ||
			    (pci_pcie_type(child) != PCI_EXP_TYPE_UPSTREAM &&
			     pci_pcie_type(child) != PCI_EXP_TYPE_ENDPOINT) ||
			    (pci_pcie_type(parent) != PCI_EXP_TYPE_DOWNSTREAM &&
			     pci_pcie_type(parent) != PCI_EXP_TYPE_ROOT_PORT))
				continue;
			/* The iterator owns its own child reference until break/end. */
			pair->dev[CHILD] = pci_dev_get(child);
			pair->dev[PARENT] = pci_dev_get(parent);
			ret = greenlit_lock_pair(pair, pass == 1);
			if (!ret) {
				pci_dev_put(child);
				return 0;
			}
			greenlit_release_pair(pair);
			if (ret == -EBUSY || unavailable == -ENODEV)
				unavailable = ret == -EAGAIN ? -ENODEV : ret;
		}
	}
	return unavailable;
}

/**
 * greenlit_run - Execute a paired save and restore original software caches.
 *
 * Return: A standard pre-run errno, or zero with raw execution observations.
 */
static int greenlit_run(void)
{
	struct greenlit_pair pair = {};
	struct greenlit_cache cache[PARTNERS] = {};
	u32 buffer_before[PARTNERS][CONTROLS], restored[CONTROLS];
	bool state_saved_original[PARTNERS];
	bool parent_first = !strcmp(result.mode, "parent_saved_first");
	unsigned int sleep_flags, partner;
	int ret, write_error = 0;

	sleep_flags = lock_system_sleep();
	pci_lock_rescan_remove();
	ret = greenlit_select_pair(&pair);
	if (ret) {
		result.capture_error = ret;
		write_error = ret;
		goto unlock;
	}
	strscpy(result.child, pci_name(pair.dev[CHILD]), sizeof(result.child));
	strscpy(result.parent, pci_name(pair.dev[PARENT]), sizeof(result.parent));
	for (partner = 0; partner < PARTNERS; partner++) {
		state_saved_original[partner] = pair.dev[partner]->state_saved;
		greenlit_buffer_snapshot(pair.buffer[partner],
					 buffer_before[partner]);
	}
	for (partner = 0; partner < PARTNERS; partner++) {
		ret = greenlit_preserve(pair.dev[partner], &cache[partner]);
		if (ret) {
			result.capture_error = ret;
			write_error = ret;
			goto restore;
		}
	}
	for (partner = 0; partner < PARTNERS; partner++) {
		ret = greenlit_controls(pair.dev[partner], result.before[partner]);
		if (ret) {
			result.capture_error = ret;
			goto restore;
		}
	}
	pair.dev[PARENT]->state_saved = false;
	pair.dev[CHILD]->state_saved = false;
	if (parent_first) {
		result.capture_error = pci_save_state(pair.dev[PARENT]);
		if (result.capture_error)
			goto capture;
	}
	/* Poison after the optional parent save so it cannot satisfy the check. */
	for (partner = 0; partner < PARTNERS; partner++)
		greenlit_poison_buffer(pair.buffer[partner], result.before[partner]);
	result.capture_error = pci_save_state(pair.dev[CHILD]);

capture:
	for (partner = 0; partner < PARTNERS; partner++) {
		greenlit_buffer_snapshot(pair.buffer[partner], result.saved[partner]);
		ret = greenlit_controls(pair.dev[partner], result.after[partner]);
		if (!result.capture_error)
			result.capture_error = ret;
	}
restore:
	for (partner = 0; partner < PARTNERS; partner++) {
		ret = greenlit_restore_cache(pair.dev[partner], &cache[partner]);
		greenlit_buffer_snapshot(pair.buffer[partner], restored);
		if (!ret &&
		    (pair.dev[partner]->state_saved != state_saved_original[partner] ||
		     memcmp(restored, buffer_before[partner], sizeof(restored))))
			ret = -EIO;
		if (!result.restore_error)
			result.restore_error = ret;
	}
	/* An eligibility/resource error must not disguise failed restoration. */
	if (result.restore_error)
		write_error = 0;
	greenlit_release_pair(&pair);
unlock:
	pci_unlock_rescan_remove();
	unlock_system_sleep(sleep_flags);
	return write_error;
}

/**
 * greenlit_show - Serialize raw observations without triggering a save.
 * @seq: Proc reader's sequential output buffer.
 * @v: Single-open iterator argument, unused.
 *
 * Return: Always zero.
 */
static int greenlit_show(struct seq_file *seq, void *v)
{
	static const char * const partners[] = { "child", "parent" };
	unsigned int partner, control;

	mutex_lock(&run_lock);
	seq_printf(seq, "mode=%s\nchild=%s\nparent=%s\n",
		   result.mode, result.child, result.parent);
	seq_printf(seq, "capture_error=%d\nrestore_error=%d\n",
		   result.capture_error, result.restore_error);
	for (partner = 0; partner < PARTNERS; partner++) {
		const char *name = partners[partner];

		for (control = 0; control < CONTROLS; control++) {
			unsigned int number = control == CTL1 ? 1 : 2;

			seq_printf(seq, "%s_ctl%u_before=0x%08x\n",
				   name, number, result.before[partner][control]);
			seq_printf(seq, "%s_ctl%u_saved=0x%08x\n%s_ctl%u_after=0x%08x\n",
				   name, number, result.saved[partner][control],
				   name, number, result.after[partner][control]);
		}
	}
	mutex_unlock(&run_lock);
	return 0;
}

/**
 * greenlit_open - Attach the raw formatter to a proc reader.
 * @inode: Proc inode, unused.
 * @file: Reader whose sequential-file context is initialized.
 *
 * Return: Zero on success, or the error returned by single_open().
 */
static int greenlit_open(struct inode *inode, struct file *file)
{
	return single_open(file, greenlit_show, NULL);
}

/**
 * greenlit_write - Execute one explicitly requested scenario.
 * @file: Proc file, unused.
 * @buffer: Exactly "parent_unsaved\n" or "parent_saved_first\n".
 * @count: Command length in bytes.
 * @position: File offset, unused.
 *
 * Return: Bytes accepted, or a standard input/eligibility/resource errno.
 */
static ssize_t greenlit_write(struct file *file, const char __user *buffer,
			      size_t count, loff_t *position)
{
	char command[sizeof("parent_saved_first\n") - 1];
	const char *requested_mode;
	int ret;

	if (count != sizeof("parent_unsaved\n") - 1 && count != sizeof(command))
		return -EINVAL;
	if (copy_from_user(command, buffer, count))
		return -EFAULT;
	if (count == sizeof("parent_unsaved\n") - 1 &&
	    !memcmp(command, "parent_unsaved\n", count))
		requested_mode = "parent_unsaved";
	else if (count == sizeof(command) &&
		 !memcmp(command, "parent_saved_first\n", count))
		requested_mode = "parent_saved_first";
	else
		return -EINVAL;
	mutex_lock(&run_lock);
	memset(&result, 0, sizeof(result));
	strscpy(result.mode, requested_mode, sizeof(result.mode));
	ret = greenlit_run();
	mutex_unlock(&run_lock);
	if (ret)
		return ret;
	return count;
}

static const struct proc_ops greenlit_proc_ops = {
	.proc_open = greenlit_open, .proc_read = seq_read,
	.proc_lseek = seq_lseek, .proc_release = single_release,
	.proc_write = greenlit_write,
};

/**
 * greenlit_init - Create the raw-observation interface without running a save.
 *
 * Return: Zero on success, or -ENOMEM if the proc entry cannot be created.
 */
static int __init greenlit_init(void)
{
	if (!proc_create(GREENLIT_PROC_NAME, 0600, NULL, &greenlit_proc_ops))
		return -ENOMEM;
	return 0;
}

/**
 * greenlit_exit - Remove the proc interface after its users have drained.
 */
static void __exit greenlit_exit(void)
{
	remove_proc_entry(GREENLIT_PROC_NAME, NULL);
}

module_init(greenlit_init);
module_exit(greenlit_exit);
MODULE_AUTHOR("NVIDIA Corporation");
MODULE_DESCRIPTION("GreenLiT PCI paired L1SS software save-state test");
MODULE_LICENSE("GPL");
