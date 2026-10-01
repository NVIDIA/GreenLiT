// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

/*
 * Print raw I/O bridge-window flags for bridges sizing actually walks.
 *
 * drivers/pci/pci-sysfs.c resource_show() replaces a bridge window with an
 * empty resource when IORESOURCE_UNSET or IORESOURCE_DISABLED is set, so
 * sysfs cannot tell a window the sizer claimed from one it left alone.
 * The non-prefetch window is always typed before sizing, so only the I/O
 * slot can show the unfixed claim. A bridge with no subordinate bus is
 * never passed to pbus_size_io().
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

#define GREENLIT_PROC_NAME "greenlit_pci_bridge_window"

static void greenlit_emit_window(struct seq_file *seq, struct pci_dev *dev,
				 const char *name, int index)
{
	seq_printf(seq, "%s %s %#lx\n", pci_name(dev), name,
		   pci_resource_flags(dev, index));
}

static int greenlit_bridge_window_show(struct seq_file *seq, void *v)
{
	struct pci_dev *dev = NULL;

	for_each_pci_dev(dev) {
		/* Sizing walks every type-1 bridge with a subordinate bus.
		 * It does not require class 0x0604.
		 */
		if (dev->hdr_type != PCI_HEADER_TYPE_BRIDGE)
			continue;
		if (!dev->subordinate)
			continue;

		greenlit_emit_window(seq, dev, "io", PCI_BRIDGE_IO_WINDOW);
	}

	return 0;
}

static int __init bridge_window_probe_init(void)
{
	if (!proc_create_single(GREENLIT_PROC_NAME, 0400, NULL,
				greenlit_bridge_window_show))
		return -ENOMEM;

	return 0;
}

static void __exit bridge_window_probe_exit(void)
{
	remove_proc_entry(GREENLIT_PROC_NAME, NULL);
}

module_init(bridge_window_probe_init);
module_exit(bridge_window_probe_exit);

MODULE_AUTHOR("NVIDIA Corporation");
MODULE_DESCRIPTION("GreenLiT PCI bridge window flag probe");
MODULE_LICENSE("GPL");
