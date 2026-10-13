/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/highmem.h>


#define DRV_NAME "pgprot_api_test"

static struct page *page_wc;
static struct page *page_dma;

static int pgprot_wc_mmap(struct file *filp, struct vm_area_struct *vma)
{
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long pfn;

    if (!page_wc)
        return -ENODEV;

    if (vma->vm_pgoff)
        return -EINVAL;

    if (size != PAGE_SIZE)
        return -EINVAL;

    pfn = page_to_pfn(page_wc);

    vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
    vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);

    return remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
}

static int pgprot_dma_mmap(struct file *filp, struct vm_area_struct *vma)
{
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long pfn;

    if (!page_dma)
        return -ENODEV;

    if (vma->vm_pgoff)
        return -EINVAL;

    if (size != PAGE_SIZE)
        return -EINVAL;

    pfn = page_to_pfn(page_dma);

    vma->vm_page_prot = pgprot_dmacoherent(vma->vm_page_prot);
    vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);

    return remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
}

static const struct file_operations pgprot_wc_fops = {
    .owner    = THIS_MODULE,
    .mmap    = pgprot_wc_mmap,
};

static const struct file_operations pgprot_dma_fops = {
    .owner    = THIS_MODULE,
    .mmap    = pgprot_dma_mmap,
};


static struct miscdevice pgprot_wc_dev = {
    .minor    = MISC_DYNAMIC_MINOR,
    .name    = "pgprot_wc",
    .fops    = &pgprot_wc_fops,
};

static struct miscdevice pgprot_dma_dev = {
    .minor    = MISC_DYNAMIC_MINOR,
    .name    = "pgprot_dma",
    .fops    = &pgprot_dma_fops,
};

static int __init test_pgprot_api_init(void)
{
    int ret;

    page_wc = alloc_page(GFP_KERNEL);
    if (!page_wc) {
        pr_err(DRV_NAME ": failed to allocate WC test page\n");
        return -ENOMEM;
    }

    page_dma = alloc_page(GFP_KERNEL);
    if (!page_dma) {
        pr_err(DRV_NAME ": failed to allocate DMA test page\n");
        __free_page(page_wc);
        page_wc = NULL;
        return -ENOMEM;
    }

    ret = misc_register(&pgprot_wc_dev);
    if (ret) {
        pr_err(DRV_NAME ": failed to register /dev/pgprot_wc (%d)\n", ret);
        goto err_wc;
    }

    ret = misc_register(&pgprot_dma_dev);
    if (ret) {
        pr_err(DRV_NAME ": failed to register /dev/pgprot_dma (%d)\n", ret);
        goto err_dma;
    }

    pr_info(DRV_NAME ": loaded. Devices:\n");
    pr_info("  /dev/pgprot_wc  (pgprot_writecombine)\n");
    pr_info("  /dev/pgprot_dma (pgprot_dmacoherent)\n");
    return 0;

err_dma:
    misc_deregister(&pgprot_wc_dev);
err_wc:
    if (page_dma) {
        __free_page(page_dma);
        page_dma = NULL;
    }
    if (page_wc) {
        __free_page(page_wc);
        page_wc = NULL;
    }
    return ret;
}


static void __exit test_pgprot_api_exit(void)
{
    misc_deregister(&pgprot_dma_dev);
    misc_deregister(&pgprot_wc_dev);

    if (page_dma)
        __free_page(page_dma);
    if (page_wc)
        __free_page(page_wc);

    pr_info(DRV_NAME ": unloaded\n");
}

module_init(test_pgprot_api_init);
module_exit(test_pgprot_api_exit);

MODULE_DESCRIPTION("Test module for pgprot_writecombine() and pgprot_dmacoherent()");
MODULE_AUTHOR("Richard Cheng");
MODULE_LICENSE("GPL");
