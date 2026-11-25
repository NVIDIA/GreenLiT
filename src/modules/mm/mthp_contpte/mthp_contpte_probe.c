// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-2.0-only

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <asm/pgtable.h>

#define GLT_MTHP_CONTPTE_IOCTL _IOWR('G', 0x60, struct glt_mthp_contpte_result)

struct glt_mthp_contpte_result {
	__u64 addr;
	__u64 size;
	__u32 mapped_pages;
	__u32 cont_pages;
	__u32 pfn_contiguous;
	__u32 page_size;
	__u32 cont_pte_size;
	__u32 cont_ptes;
	__u32 large_folio;
	__u32 folio_order;
};

struct glt_mthp_walk {
	unsigned long start;
	unsigned long pfn0;
	struct folio *folio;
	struct glt_mthp_contpte_result *result;
};

static int glt_mthp_pte_fn(pte_t *ptep, unsigned long addr, void *data)
{
	struct glt_mthp_walk *walk = data;
	struct glt_mthp_contpte_result *result = walk->result;
	unsigned int index = (addr - walk->start) >> PAGE_SHIFT;
	pte_t pte = __ptep_get(ptep);
	struct folio *folio;
	unsigned long pfn;

	if (!pte_present(pte))
		return 0;
	if (pte_special(pte))
		return -EOPNOTSUPP;
	folio = page_folio(pte_page(pte));
	if (!folio_test_anon(folio) ||
	    folio_order(folio) != CONT_PTE_SHIFT - PAGE_SHIFT)
		return -EOPNOTSUPP;
	if (!walk->folio) {
		walk->folio = folio;
		result->large_folio = 1;
		result->folio_order = folio_order(folio);
	} else if (folio != walk->folio) {
		return -EOPNOTSUPP;
	}

	pfn = pte_pfn(pte);
	if (!result->mapped_pages) {
		walk->pfn0 = pfn - index;
		result->pfn_contiguous = 1;
	} else if (pfn != walk->pfn0 + index) {
		result->pfn_contiguous = 0;
	}
	result->mapped_pages++;
	if (pte_cont(pte))
		result->cont_pages++;
	return 0;
}

static int glt_walk_existing_ptes(struct mm_struct *mm, unsigned long addr,
				  struct glt_mthp_walk *walk)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud;
	pmd_t *pmd;
	pmd_t pmdval;
	pte_t *ptep;
	spinlock_t *ptl;
	int attempt;
	int index;
	int err = 0;

	pgd = pgd_offset(mm, addr);
	if (pgd_none(*pgd) || pgd_bad(*pgd))
		return -EOPNOTSUPP;
	p4d = p4d_offset(pgd, addr);
	if (p4d_none(*p4d) || p4d_bad(*p4d))
		return -EOPNOTSUPP;
	pud = pud_offset(p4d, addr);
	if (pud_none(*pud) || pud_bad(*pud) || pud_trans_huge(*pud))
		return -EOPNOTSUPP;
	pmd = pmd_offset(pud, addr);
	for (attempt = 0; attempt < 3; attempt++) {
		rcu_read_lock();
		pmdval = READ_ONCE(*pmd);
		if (pmd_none(pmdval) || !pmd_present(pmdval) || pmd_trans_huge(pmdval)) {
			rcu_read_unlock();
			return -EOPNOTSUPP;
		}
		ptl = pte_lockptr(mm, &pmdval);
		spin_lock(ptl);
		if (!pmd_same(pmdval, READ_ONCE(*pmd))) {
			spin_unlock(ptl);
			rcu_read_unlock();
			continue;
		}
		ptep = pte_offset_kernel(&pmdval, addr);
		for (index = 0; index < CONT_PTES; index++) {
			err = glt_mthp_pte_fn(ptep + index, addr + index * PAGE_SIZE, walk);
			if (err)
				break;
		}
		spin_unlock(ptl);
		rcu_read_unlock();
		return err;
	}
	return -EAGAIN;
}

static long glt_mthp_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct glt_mthp_contpte_result result = {};
	struct glt_mthp_walk walk = {};
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma;
	unsigned long next, end;
	int err;

	if (cmd != GLT_MTHP_CONTPTE_IOCTL)
		return -ENOTTY;
	if (copy_from_user(&result, (void __user *)arg, sizeof(result)))
		return -EFAULT;
	if (!mm)
		return -EINVAL;
	if (result.size != CONT_PTE_SIZE || CONT_PTES < 2)
		return -EOPNOTSUPP;
	if (!result.size || (result.addr & (result.size - 1)))
		return -EINVAL;
	end = result.addr + result.size;
	if (end < result.addr)
		return -EINVAL;

	result.mapped_pages = 0;
	result.cont_pages = 0;
	result.pfn_contiguous = 1;
	result.page_size = PAGE_SIZE;
	result.cont_pte_size = CONT_PTE_SIZE;
	result.cont_ptes = CONT_PTES;
	walk.start = result.addr;
	walk.result = &result;

	mmap_read_lock(mm);
	for (next = result.addr; next < end;) {
		vma = find_vma(mm, next);
		if (!vma || vma->vm_start > next ||
		    !vma_is_anonymous(vma) ||
		    (vma->vm_flags & (VM_HUGETLB | VM_PFNMAP | VM_MIXEDMAP | VM_IO))) {
			mmap_read_unlock(mm);
			return -EOPNOTSUPP;
		}
		next = vma->vm_end;
	}
	err = glt_walk_existing_ptes(mm, result.addr, &walk);
	mmap_read_unlock(mm);
	if (err)
		return err;

	if (copy_to_user((void __user *)arg, &result, sizeof(result)))
		return -EFAULT;
	return 0;
}

static const struct file_operations glt_mthp_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = glt_mthp_ioctl,
	.llseek = noop_llseek,
};

static struct miscdevice glt_mthp_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "glt_mthp_contpte",
	.fops = &glt_mthp_fops,
	.mode = 0600,
};

static int __init glt_mthp_init(void)
{
	return misc_register(&glt_mthp_misc);
}

static void __exit glt_mthp_exit(void)
{
	misc_deregister(&glt_mthp_misc);
}

module_init(glt_mthp_init);
module_exit(glt_mthp_exit);
MODULE_DESCRIPTION("Read-only user mTHP contpte page-table probe");
MODULE_LICENSE("GPL");
