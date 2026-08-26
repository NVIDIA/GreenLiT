// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only
//
// Observational probe for arm64 contpte_ptep_set_access_flags() false no-op
// (mainline 97c5550b7631 / DGX-15884).
//
// Direct call, not an SMMU/ATS walk. Crafts a CONT block where a sibling is
// dirty and the target is writable-but-clean (PTE_WRITE + PTE_RDONLY).
// Gathering via ptep_get() applies pte_mkdirty() which clears PTE_RDONLY in
// the comparison view, so the unfixed no-op check matches a dirty/write
// entry and returns 0 while raw sub-PTEs still have PTE_RDONLY.

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <asm/pgtable.h>
#include <asm/tlbflush.h>

#define GLT_CONTPTE_IOCTL_PROBE _IOR('G', 0x51, struct glt_contpte_probe_result)

struct glt_contpte_probe_result {
	__s32 set_access_ret;
	__u32 target_rdonly_before;
	/* Count of CONT slots still carrying PTE_RDONLY after the call. */
	__u32 slots_rdonly_after;
	__u32 false_noop;
	__u32 cont_ptes;
	__u32 page_size;
	/* 1 if pte_mknoncont(ptep_get(target)) == dirtying entry before call. */
	__u32 gather_matches_entry;
};

struct probe_file {
	unsigned long uaddr;
	bool mapped;
};

struct probe_ctx {
	struct vm_area_struct *vma;
	struct glt_contpte_probe_result *out;
	int err;
	bool ran;
};

#ifndef CONFIG_ARM64_CONTPTE
#error "CONFIG_ARM64_CONTPTE is required"
#endif

/* contpte_align_down() is private to arch/arm64/mm/contpte.c; mirror it. */
static inline pte_t *glt_contpte_align_down(pte_t *ptep)
{
	return PTR_ALIGN_DOWN(ptep, sizeof(*ptep) * CONT_PTES);
}

static struct page *probe_pages;
static unsigned int probe_order;

static int glt_contpte_open(struct inode *inode, struct file *file)
{
	struct probe_file *pf;

	pf = kzalloc(sizeof(*pf), GFP_KERNEL);
	if (!pf)
		return -ENOMEM;
	file->private_data = pf;
	return 0;
}

static int glt_contpte_release(struct inode *inode, struct file *file)
{
	kfree(file->private_data);
	file->private_data = NULL;
	return 0;
}

static int glt_contpte_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct probe_file *pf = file->private_data;
	unsigned long pfn;
	unsigned long size = vma->vm_end - vma->vm_start;

	if (!probe_pages || !pf)
		return -ENODEV;
	if (pf->mapped)
		return -EBUSY;
	if (size != CONT_PTE_SIZE)
		return -EINVAL;
	if (vma->vm_start & (CONT_PTE_SIZE - 1))
		return -EINVAL;
	if (vma->vm_flags & VM_EXEC)
		return -EINVAL;

	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);
	pfn = page_to_pfn(probe_pages);
	if (remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot))
		return -EAGAIN;

	pf->uaddr = vma->vm_start;
	pf->mapped = true;
	return 0;
}

static int probe_on_cont_block(pte_t *ptep, unsigned long addr, struct probe_ctx *ctx)
{
	struct vm_area_struct *vma = ctx->vma;
	struct glt_contpte_probe_result *out = ctx->out;
	pte_t *cont_ptep = glt_contpte_align_down(ptep);
	unsigned long cont_addr = ALIGN_DOWN(addr, CONT_PTE_SIZE);
	pte_t *target_ptep = cont_ptep + 1;
	unsigned long target_addr = cont_addr + PAGE_SIZE;
	pte_t orig[CONT_PTES];
	pte_t sibling, target, entry;
	pteval_t pfn0;
	u32 slots_rdonly = 0;
	int i;
	int ret;
	int err = 0;

	if (CONT_PTES < 2)
		return -EINVAL;

	pfn0 = pte_pfn(__ptep_get(cont_ptep));
	for (i = 0; i < CONT_PTES; i++) {
		orig[i] = __ptep_get(cont_ptep + i);

		/* remap_pfn_range mappings are pte_special; that is expected. */
		if (!pte_present(orig[i]))
			return -EINVAL;
		if (pte_pfn(orig[i]) != pfn0 + i)
			return -EINVAL;
	}

	for (i = 0; i < CONT_PTES; i++)
		__set_pte(cont_ptep + i, pte_mkcont(orig[i]));

	sibling = __ptep_get(cont_ptep);
	sibling = pte_mkyoung(pte_mkdirty(pte_mkwrite_novma(pte_mknoncont(sibling))));
	sibling = pte_mkcont(sibling);
	__set_pte(cont_ptep, sibling);

	target = __ptep_get(target_ptep);
	target = pte_mknoncont(target);
	target = pte_mkwrite_novma(target);
	target = pte_mkclean(target);
	target = pte_mkyoung(target);
	target = pte_mkcont(target);
	__set_pte(target_ptep, target);

	/*
	 * flush_tlb_mm() pulls unexported mmu_notifier helpers into
	 * out-of-tree modules. local_flush_tlb_all() is a stable inline
	 * (no notifier) at 97c5550b and current mainline. Mapping is this
	 * process's private remap_pfn_range of module pages.
	 */
	local_flush_tlb_all();

	out->target_rdonly_before = !!(pte_val(__ptep_get(target_ptep)) & PTE_RDONLY);
	out->cont_ptes = CONT_PTES;
	out->page_size = PAGE_SIZE;
	if (!out->target_rdonly_before || !pte_write(__ptep_get(target_ptep))) {
		ctx->ran = true;
		err = 1;
		goto restore;
	}

	entry = pte_mknoncont(__ptep_get(target_ptep));
	entry = pte_mkyoung(pte_mkdirty(entry));

	/*
	 * Positive control (DISP-002): the unfixed no-op path is
	 *   if (pte_val(pte_mknoncont(ptep_get(ptep))) == pte_val(entry))
	 *       return 0;
	 * while a raw sub-PTE still has PTE_RDONLY. If gathering/crafting
	 * drifted such that this predicate is false, an unfixed kernel would
	 * take the update path and look fixed — refuse to judge (userspace
	 * SKIPs on gather_matches_entry == 0).
	 */
	{
		pte_t gathered = pte_mknoncont(ptep_get(target_ptep));

		out->gather_matches_entry =
			(pte_val(gathered) == pte_val(entry)) ? 1 : 0;
	}
	if (!out->gather_matches_entry) {
		ctx->ran = true;
		err = 1;
		goto restore;
	}

	ret = contpte_ptep_set_access_flags(vma, target_addr, target_ptep, entry, 1);
	out->set_access_ret = ret;
	for (i = 0; i < CONT_PTES; i++) {
		if (pte_val(__ptep_get(cont_ptep + i)) & PTE_RDONLY)
			slots_rdonly++;
	}
	out->slots_rdonly_after = slots_rdonly;
	out->false_noop = (ret == 0 && slots_rdonly) ? 1 : 0;

	ctx->ran = true;
	err = 1;

restore:
	for (i = 0; i < CONT_PTES; i++)
		__set_pte(cont_ptep + i, orig[i]);
	local_flush_tlb_all();
	return err;
}

static int probe_pte_fn(pte_t *ptep, unsigned long addr, void *data)
{
	struct probe_ctx *ctx = data;

	if (ctx->ran)
		return 0;
	if (addr & (CONT_PTE_SIZE - 1))
		return 0;

	ctx->err = probe_on_cont_block(ptep, addr, ctx);
	if (ctx->err != 0)
		return ctx->err;
	return 0;
}

static long glt_contpte_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct probe_file *pf = file->private_data;
	struct glt_contpte_probe_result result = {};
	struct probe_ctx ctx = {};
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma;
	unsigned long reserve;
	unsigned long aligned;
	unsigned long mapped;
	long ret = 0;
	int err;

	if (cmd != GLT_CONTPTE_IOCTL_PROBE)
		return -ENOTTY;
	if (!mm || !pf)
		return -EINVAL;
	if (pf->mapped)
		return -EBUSY;

	reserve = vm_mmap(NULL, 0, 2 * CONT_PTE_SIZE, PROT_READ | PROT_WRITE,
			  MAP_PRIVATE | MAP_ANONYMOUS, 0);
	if (IS_ERR_VALUE(reserve))
		return reserve;

	aligned = ALIGN(reserve, CONT_PTE_SIZE);
	mapped = vm_mmap(file, aligned, CONT_PTE_SIZE, PROT_READ | PROT_WRITE,
			 MAP_SHARED | MAP_FIXED, 0);
	if (IS_ERR_VALUE(mapped)) {
		vm_munmap(reserve, 2 * CONT_PTE_SIZE);
		return mapped;
	}

	mmap_write_lock(mm);
	vma = find_vma(mm, pf->uaddr);
	if (!vma || vma->vm_start != pf->uaddr ||
	    vma->vm_end < pf->uaddr + CONT_PTE_SIZE) {
		mmap_write_unlock(mm);
		ret = -EINVAL;
		goto unmap;
	}

	ctx.vma = vma;
	ctx.out = &result;
	err = apply_to_page_range(mm, pf->uaddr, CONT_PTE_SIZE, probe_pte_fn, &ctx);
	mmap_write_unlock(mm);

	if (!ctx.ran) {
		ret = err < 0 ? err : -ENOENT;
		goto unmap;
	}
	if (ctx.err < 0) {
		ret = ctx.err;
		goto unmap;
	}

	if (copy_to_user((void __user *)arg, &result, sizeof(result)))
		ret = -EFAULT;

unmap:
	vm_munmap(reserve, 2 * CONT_PTE_SIZE);
	pf->mapped = false;
	pf->uaddr = 0;
	return ret;
}

static const struct file_operations glt_contpte_fops = {
	.owner		= THIS_MODULE,
	.open		= glt_contpte_open,
	.release	= glt_contpte_release,
	.mmap		= glt_contpte_mmap,
	.unlocked_ioctl	= glt_contpte_ioctl,
	.llseek		= noop_llseek,
};

static struct miscdevice glt_contpte_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "glt_contpte_access_flags",
	.fops = &glt_contpte_fops,
	.mode = 0600,
};

static int __init glt_contpte_init(void)
{
	int err;

	probe_order = CONT_PTE_SHIFT - PAGE_SHIFT;
	probe_pages = alloc_pages(GFP_KERNEL | __GFP_ZERO, probe_order);
	if (!probe_pages)
		return -ENOMEM;

	err = misc_register(&glt_contpte_misc);
	if (err) {
		__free_pages(probe_pages, probe_order);
		probe_pages = NULL;
		return err;
	}
	return 0;
}

static void __exit glt_contpte_exit(void)
{
	misc_deregister(&glt_contpte_misc);
	if (probe_pages) {
		__free_pages(probe_pages, probe_order);
		probe_pages = NULL;
	}
}

module_init(glt_contpte_init);
module_exit(glt_contpte_exit);

MODULE_AUTHOR("NVIDIA");
MODULE_DESCRIPTION("GreenLiT contpte set_access_flags false no-op probe");
MODULE_LICENSE("GPL");
