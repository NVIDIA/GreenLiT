// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/dma-mapping.h>
#include <linux/pci.h>

#define DRV_NAME           "test_dma_alloc"

/* How hard to stress the allocators */
#define DMA_STRESS_ITERS   10000
#define DMA_BUF_SIZE       (64 * 1024)   /* 64 KiB */

/* Find a non‑bridge PCI device with no driver bound for dma_alloc_*() tests.
 * Returns a pci_dev with ref held, or NULL if none found.
 */
static struct pci_dev *find_test_pci_dev(void)
{
	struct pci_dev *pdev = NULL;
	u16 cmd;

	while ((pdev = pci_get_device(PCI_ANY_ID, PCI_ANY_ID, pdev))) {
		if (pci_is_bridge(pdev))
			continue;
		if (pci_dev_driver(pdev))
			continue;

        /* check bus master bit */
        if (pci_read_config_word(pdev, PCI_COMMAND, &cmd)) {
            pr_warn(DRV_NAME ": failed to read PCI_COMMAND, skipping\n");
            continue;
        }

        if (!(cmd & PCI_COMMAND_MASTER)) {
			pr_warn(DRV_NAME ": bus master not enabled, skipping\n");
            continue;
        }


		pr_info(DRV_NAME ": using test device %04x:%04x at %02x:%02x.%d\n",
			pdev->vendor, pdev->device,
			pdev->bus->number, PCI_SLOT(pdev->devfn),
			PCI_FUNC(pdev->devfn));
		return pdev;  /* ref held from pci_get_device() */
	}

	pr_info(DRV_NAME ": SKIPPED (no suitable PCI device found)\n");
	return NULL;
}

static int run_dma_alloc_noncoherent_test(struct device *dev)
{
	int i;
	void *cpu_addr;
	dma_addr_t dma_handle;

	for (i = 0; i < DMA_STRESS_ITERS; i++) {
		cpu_addr = dma_alloc_noncoherent(dev, DMA_BUF_SIZE,
						&dma_handle, DMA_BIDIRECTIONAL,
						GFP_KERNEL);
		if (!cpu_addr) {
			pr_err(DRV_NAME ": dma_alloc_noncoherent(%u) failed at iter=%d\n",
				DMA_BUF_SIZE, i);
			return -ENOMEM;
		}

		dma_free_noncoherent(dev, DMA_BUF_SIZE, cpu_addr, dma_handle,
				DMA_BIDIRECTIONAL);
	}

	return 0;
}

static int run_dma_alloc_wc_test(struct device *dev)
{
	int i;
	void *cpu_addr;
	dma_addr_t dma_handle;

	for (i = 0; i < DMA_STRESS_ITERS; i++) {
		cpu_addr = dma_alloc_wc(dev, DMA_BUF_SIZE,
					&dma_handle, GFP_KERNEL);
		if (!cpu_addr) {
			pr_err(DRV_NAME ": dma_alloc_wc(%u) failed at iter=%d\n",
				DMA_BUF_SIZE, i);
			return -ENOMEM;
		}

		dma_free_wc(dev, DMA_BUF_SIZE, cpu_addr, dma_handle);
	}

	return 0;
}

static int __init dma_alloc_test_init(void)
{
	struct pci_dev *pdev = find_test_pci_dev();
	if (!pdev)
		return -ENODEV;

	int ret = run_dma_alloc_noncoherent_test(&pdev->dev);
	if (ret) {
		pr_err(DRV_NAME ": dma_alloc_noncoherent() test FAILED (ret=%d)\n", ret);
		pci_dev_put(pdev);
		return ret;
	}

	ret = run_dma_alloc_wc_test(&pdev->dev);
	if (ret) {
		pr_err(DRV_NAME ": dma_alloc_wc() stress FAILED (ret=%d)\n", ret);
		pci_dev_put(pdev);
		return ret;
	}

	pr_info(DRV_NAME ": PASSED\n");
	pci_dev_put(pdev);
	return 0;
}

static void __exit dma_alloc_test_exit(void)
{
	pr_info(DRV_NAME ": exit\n");
}

module_init(dma_alloc_test_init);
module_exit(dma_alloc_test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Richard Cheng");
MODULE_DESCRIPTION("MAIR_EL1 WAR dma_alloc_noncoherent()/dma_alloc_wc() test");
MODULE_VERSION("1.0");