// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
/*
 * Query the public StandaloneMm Buffers service on its existing FF-A device.
 * GET_MM_NS_BUFFER_FID only returns metadata: never map or access the buffer,
 * publish its physical address, bind a driver, or replace transport operations.
 *
 * Public protocol references at edk2-nvidia revision
 * e1d1a17337c6331c71ba12aeb4928dbddaf0add6:
 * https://github.com/NVIDIA/edk2-nvidia/blob/e1d1a17337c6331c71ba12aeb4928dbddaf0add6/Silicon/NVIDIA/Library/StMmDxeGetBuffersLib/StMmDxeGetBuffersLib.c#L125
 * https://github.com/NVIDIA/edk2-nvidia/blob/e1d1a17337c6331c71ba12aeb4928dbddaf0add6/Silicon/NVIDIA/Include/Library/StandaloneMmOpteeDeviceMem.h#L50
 * https://github.com/NVIDIA/edk2-nvidia/blob/e1d1a17337c6331c71ba12aeb4928dbddaf0add6/Silicon/NVIDIA/NVIDIA.dec#L250
 */
#include <linux/arm_ffa.h>
#include <linux/device.h>
#include <linux/errno.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/string.h>
#include <linux/utsname.h>
#include <linux/uuid.h>

#define GLT_FFA_STMM_PROC "glt_ffa_stmm_runtime"
#define GET_MM_NS_BUFFER_FID 1UL
/*
 * Packed AArch64 EFI_GUID(16) + UINTN(8) + Data[1], not Data's offset (24):
 * https://github.com/tianocore/edk2/blob/1e1fb123c84b5898bbb3cca127f97ab1c237bccd/MdePkg/Include/Protocol/MmCommunication.h#L14
 */
#define MM_COMMUNICATE_HEADER_BYTES 25UL

static char *device_name;
static unsigned short partition_id;
static char *run_id;
module_param(device_name, charp, 0);
module_param(partition_id, ushort, 0);
module_param(run_id, charp, 0);
MODULE_PARM_DESC(device_name, "Existing unbound Buffers arm-ffa-N device");
MODULE_PARM_DESC(partition_id, "Expected partition ID shared with the real TPM service");
MODULE_PARM_DESC(run_id, "Unique 32-character lowercase hexadecimal run identity");

static const uuid_t buffers_uuid = UUID_INIT(0xe257054d, 0x2092, 0x4e1e,
		0xb4, 0x0c, 0x2e, 0xf1, 0xf6, 0x8e, 0x4b, 0x0a);

struct query_result {
	uuid_t uuid;
	unsigned short partition_id;
	u32 properties;
	bool mode_32bit;
	bool driver_bound;
	bool req2_supported;
	bool api_available;
	bool query_attempted;
	int api_rc;
	bool response_received;
	unsigned long service_status;
	bool base_nonzero;
	unsigned long buffer_size;
	const char *stage;
};

/* Written only during init, before creating the read-only result entry. */
static struct query_result result = {
	.stage = "device_missing",
};

static struct proc_dir_entry *result_entry;

static bool valid_parameters(void)
{
	size_t i;

	if (!device_name || strncmp(device_name, "arm-ffa-", 8) ||
	    !device_name[8] || !partition_id || !run_id ||
	    strlen(run_id) != 32)
		return false;
	for (i = 8; device_name[i]; i++)
		if (device_name[i] < '0' || device_name[i] > '9')
			return false;
	for (i = 0; i < 32; i++)
		if (!((run_id[i] >= '0' && run_id[i] <= '9') ||
		      (run_id[i] >= 'a' && run_id[i] <= 'f')))
			return false;
	return true;
}

static void query_documented_buffers_service(void)
{
	struct ffa_send_direct_data2 message = {
		.data = { GET_MM_NS_BUFFER_FID },
	};
	struct ffa_device *ffa_dev;
	struct device *dev;

	uuid_copy(&result.uuid, &buffers_uuid);
	result.partition_id = partition_id;
	/* bus_find_device_by_name() returns a referenced real bus device. */
	dev = bus_find_device_by_name(&ffa_bus_type, NULL, device_name);
	if (!dev)
		goto clear_message;

	device_lock(dev);
	if (!device_is_registered(dev))
		goto unlock_put;
	ffa_dev = to_ffa_dev(dev);
	uuid_copy(&result.uuid, &ffa_dev->uuid);
	result.partition_id = ffa_dev->vm_id;
	result.properties = ffa_dev->properties;
	result.mode_32bit = ffa_dev->mode_32bit;
	result.driver_bound = !!dev->driver;
	result.req2_supported = ffa_partition_supports_direct_req2_recv(ffa_dev);
	result.api_available = ffa_dev->ops && ffa_dev->ops->msg_ops &&
			       ffa_dev->ops->msg_ops->sync_send_receive2;
	result.stage = "device_identity_changed";
	if (strcmp(dev_name(dev), device_name) ||
	    !uuid_equal(&ffa_dev->uuid, &buffers_uuid) ||
	    ffa_dev->vm_id != partition_id)
		goto unlock_put;
	result.stage = "device_bound";
	if (result.driver_bound)
		goto unlock_put;
	result.stage = "req2_unsupported";
	if (!result.req2_supported)
		goto unlock_put;

	result.stage = "api_missing";
	if (!result.api_available)
		goto unlock_put;

	/* Exactly one request; all other thirteen request words remain zero. */
	result.query_attempted = true;
	result.api_rc = ffa_dev->ops->msg_ops->sync_send_receive2(ffa_dev, &message);
	result.stage = "api_error";
	if (result.api_rc)
		goto unlock_put;
	result.response_received = true;
	result.service_status = message.data[0];
	result.stage = "service_error";
	if (result.service_status)
		goto unlock_put;

	/* Other response words are metadata only after API and service success. */
	result.base_nonzero = message.data[1] != 0;
	result.buffer_size = message.data[2];
	result.stage = "complete";
unlock_put:
	device_unlock(dev);
	put_device(dev);
clear_message:
	memzero_explicit(&message, sizeof(message));
}

static int result_show(struct seq_file *seq, void *unused)
{
	seq_puts(seq, "schema_version=2\nevidence_kind=running_kernel_ffa_multi_uuid\n");
	seq_printf(seq, "run_id=%s\nkernel_release=%s\ndevice=%s\n",
		   run_id, init_utsname()->release, device_name);
	seq_printf(seq, "uuid=%pUb\npartition_id=0x%04x\nproperties=0x%08x\n",
		   &result.uuid, result.partition_id, result.properties);
	seq_printf(seq, "mode_32bit=%u\ndriver_bound=%u\nreq2_supported=%u\n",
		   result.mode_32bit, result.driver_bound, result.req2_supported);
	seq_printf(seq, "api_available=%u\nquery_attempted=%u\napi_rc=%d\n",
		   result.api_available, result.query_attempted, result.api_rc);
	seq_printf(seq, "response_received=%u\nservice_status=0x%lx\n",
		   result.response_received, result.service_status);
	seq_printf(seq, "base_nonzero=%u\nbuffer_size=%lu\nminimum_size=%lu\n",
		   result.base_nonzero, result.buffer_size, MM_COMMUNICATE_HEADER_BYTES);
	seq_printf(seq, "stage=%s\n", result.stage);
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

static int __init glt_ffa_stmm_runtime_init(void)
{
	if (!valid_parameters())
		return -EINVAL;
	query_documented_buffers_service();
	result_entry = proc_create(GLT_FFA_STMM_PROC, 0400, NULL, &result_ops);
	return result_entry ? 0 : -ENOMEM;
}

static void __exit glt_ffa_stmm_runtime_exit(void)
{
	proc_remove(result_entry);
}
module_init(glt_ffa_stmm_runtime_init);
module_exit(glt_ffa_stmm_runtime_exit);
MODULE_DESCRIPTION("GreenLiT running-kernel public FF-A Buffers metadata query");
MODULE_LICENSE("GPL v2");
