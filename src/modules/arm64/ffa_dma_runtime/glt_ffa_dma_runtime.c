// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
/* Exercise DMA on an existing unbound device in the running FF-A bus.
 * No firmware calls, client binding, synthetic devices, or private driver
 * structure access. All temporary device DMA state is restored in init.
 */
#include <linux/arm_ffa.h>
#include <linux/compiler.h>
#include <linux/ctype.h>
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/minmax.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/string.h>
#include <linux/utsname.h>
#include <linux/uuid.h>

#define GLT_FFA_DMA_PROC "glt_ffa_dma_runtime"

static char *device_name;
static char *run_id;
module_param(device_name, charp, 0);
module_param(run_id, charp, 0);
MODULE_PARM_DESC(device_name, "Existing unbound arm-ffa-N device");
MODULE_PARM_DESC(run_id, "Unique 32-character lowercase hexadecimal run identity");

static const uuid_t tpm_uuid = UUID_INIT(0x17b862a4, 0x1806, 0x4faf,
		0x86, 0xb3, 0x08, 0x9a, 0x58, 0x35, 0x38, 0x61);
static struct proc_dir_entry *result_entry;
static const char *result_stage = "device_missing";
static int mask_present;
static int mask_set_rc;
static unsigned long allocation_bytes;
static int data_checked;
static int state_restored = 1;

static bool valid_parameters(void)
{
	size_t i;

	if (!device_name || strncmp(device_name, "arm-ffa-", 8) ||
	    !device_name[8] || !run_id || strlen(run_id) != 32)
		return false;
	for (i = 8; device_name[i]; i++)
		if (!isdigit(device_name[i]))
			return false;
	for (i = 0; i < 32; i++)
		if (!((run_id[i] >= '0' && run_id[i] <= '9') ||
		      (run_id[i] >= 'a' && run_id[i] <= 'f')))
			return false;
	return true;
}

static void exercise_existing_device(void)
{
	struct device *dev;
	struct ffa_device *ffa_dev;
	u64 *saved_mask_pointer;
	u64 saved_mask, saved_coherent, allocation_mask;
#ifdef CONFIG_DMA_NEED_SYNC
	bool saved_skip_sync;
#endif
	dma_addr_t dma_handle;
	u8 *buffer;
	size_t i;

	dev = bus_find_device_by_name(&ffa_bus_type, NULL, device_name);
	if (!dev)
		return;

	device_lock(dev);
	if (dev->driver) {
		result_stage = "device_bound";
		goto unlock_put;
	}
	ffa_dev = to_ffa_dev(dev);
	if (uuid_equal(&ffa_dev->uuid, &tpm_uuid)) {
		result_stage = "tpm_device";
		goto unlock_put;
	}

	/* Do not use dma_coerce_mask_and_coherent(): it would repair the bug. */
	result_stage = "dma_mask_missing";
	if (!dev->dma_mask)
		goto unlock_put;
	mask_present = 1;
	saved_mask_pointer = dev->dma_mask;
	saved_mask = *dev->dma_mask;
	saved_coherent = dev->coherent_dma_mask;
#ifdef CONFIG_DMA_NEED_SYNC
	saved_skip_sync = dev->dma_skip_sync;
#endif
	/* Preserve configured limits; fresh FF-A devices can have zero values. */
	allocation_mask = min_not_zero(saved_mask, saved_coherent);
	if (!allocation_mask)
		allocation_mask = DMA_BIT_MASK(64);
	state_restored = 0;
	result_stage = "dma_mask_set";
	mask_set_rc = dma_set_mask_and_coherent(dev, allocation_mask);
	if (mask_set_rc)
		goto restore;

	result_stage = "dma_allocation";
	buffer = dma_alloc_coherent(dev, PAGE_SIZE, &dma_handle, GFP_KERNEL);
	if (!buffer)
		goto restore;
	allocation_bytes = PAGE_SIZE;
	result_stage = "dma_buffer_check";
	memset(buffer, 0xa5, PAGE_SIZE);
	WRITE_ONCE(buffer[0], 0x36);
	WRITE_ONCE(buffer[PAGE_SIZE - 1], 0x79);
	data_checked = 1;
	for (i = 0; i < PAGE_SIZE; i++) {
		u8 expected = i == 0 ? 0x36 : i == PAGE_SIZE - 1 ? 0x79 : 0xa5;

		if (READ_ONCE(buffer[i]) != expected) {
			data_checked = 0;
			break;
		}
	}
	dma_free_coherent(dev, PAGE_SIZE, buffer, dma_handle);
	if (data_checked)
		result_stage = "complete";
restore:
	/* The public DMA APIs change values, not the mask pointer. */
	dev->dma_mask = saved_mask_pointer;
	*saved_mask_pointer = saved_mask;
	dev->coherent_dma_mask = saved_coherent;
#ifdef CONFIG_DMA_NEED_SYNC
	dev->dma_skip_sync = saved_skip_sync;
#endif
	state_restored = dev->dma_mask == saved_mask_pointer &&
			 *dev->dma_mask == saved_mask &&
			 dev->coherent_dma_mask == saved_coherent;
#ifdef CONFIG_DMA_NEED_SYNC
	state_restored = state_restored && dev->dma_skip_sync == saved_skip_sync;
#endif
	if (!state_restored)
		result_stage = "restore_failed";
unlock_put:
	device_unlock(dev);
	put_device(dev);
}

static int result_show(struct seq_file *seq, void *unused)
{
	seq_puts(seq, "schema_version=2\nevidence_kind=running_kernel_dma\n");
	seq_printf(seq, "run_id=%s\nkernel_release=%s\ndevice=%s\n",
		   run_id, init_utsname()->release, device_name);
	seq_printf(seq, "stage=%s\nmask_present=%d\nmask_set_rc=%d\n",
		   result_stage, mask_present, mask_set_rc);
	seq_printf(seq, "allocation_bytes=%lu\ndata_checked=%d\nstate_restored=%d\n",
		   allocation_bytes, data_checked, state_restored);
	return 0;
}

static int result_open(struct inode *inode, struct file *file)
{
	return single_open(file, result_show, NULL);
}

static const struct proc_ops result_ops = {
	.proc_open = result_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};

static int __init glt_ffa_dma_runtime_init(void)
{
	if (!valid_parameters())
		return -EINVAL;
	exercise_existing_device();
	result_entry = proc_create(GLT_FFA_DMA_PROC, 0400, NULL, &result_ops);
	return result_entry ? 0 : -ENOMEM;
}

static void __exit glt_ffa_dma_runtime_exit(void)
{
	proc_remove(result_entry);
}
module_init(glt_ffa_dma_runtime_init);
module_exit(glt_ffa_dma_runtime_exit);
MODULE_DESCRIPTION("GreenLiT running-kernel FF-A coherent DMA allocation check");
MODULE_LICENSE("GPL v2");
