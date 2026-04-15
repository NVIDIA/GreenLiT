// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/device.h>
#include <linux/pci.h>

static u64 phys_addr_param;
static unsigned long map_size = PAGE_SIZE;

/*
 * - rw_len: how many bytes to test (will be rounded down to multiple of 4)
 * - rw_pattern: 32-bit pattern to write
 * - enable_write: enable write operations (default: 0 = read-only, safer for devices)
 */
static unsigned long rw_len = 0x1000;
static u32 rw_pattern = 0xa5a5a5a5;
static int enable_write = 0;

static char *pci_bdf;
static int pci_bar = -1;

module_param(phys_addr_param, ullong, 0444);
MODULE_PARM_DESC(phys_addr_param,
    "Physical address to ioremap_wc (MMIO, e.g. ASPEED BAR0 0x29000000)");

module_param(map_size, ulong, 0444);
MODULE_PARM_DESC(map_size,
    "Size to ioremap_wc (bytes, rounded up to PAGE_SIZE)");

module_param(rw_len, ulong, 0644);
MODULE_PARM_DESC(rw_len,
    "Number of bytes to cover with the R/W test (multiple of 4, <= map_size)");

module_param(rw_pattern, uint, 0644);
MODULE_PARM_DESC(rw_pattern,
    "32-bit pattern written during the R/W test");

module_param(enable_write, int, 0644);
MODULE_PARM_DESC(enable_write,
    "Enable write operations (0=read-only [default, safer], 1=read+write)");

module_param(pci_bdf, charp, 0444);
MODULE_PARM_DESC(pci_bdf,
    "PCI BDF for pci_ioremap_wc_bar test (e.g. 0005:c1:00.0)");

module_param(pci_bar, int, 0444);
MODULE_PARM_DESC(pci_bar,
    "PCI BAR index for pci_ioremap_wc_bar test (e.g. 0)");

static struct device *test_dev;

/* Shared R/W test for any mapping */
static int run_rw_test(void __iomem *base, unsigned long size)
{
    unsigned long bytes, offset;
    unsigned long errors = 0;

    /* limit rw_len to mapped size and to 32-bit alignment */
    bytes = min(rw_len, size);
    bytes &= ~0x3UL;

    if (!bytes) {
        pr_err("test_ioremap_wc_api: FAIL: rw_len too small after alignment\n");
        return -EINVAL;
    }

    if (enable_write) {
        /* Write pattern to memory */
        pr_info("test_ioremap_wc_api: Writing pattern 0x%08x to 0x%lx bytes\n",
                rw_pattern, bytes);

        for (offset = 0; offset < bytes; offset += 4)
            writel_relaxed(rw_pattern, base + offset);

        wmb();

        /* Read back and verify pattern */
        for (offset = 0; offset < bytes; offset += 4) {
            u32 val = readl_relaxed(base + offset);

            if (val != rw_pattern)
                errors++;
        }

        if (errors) {
            pr_err("test_ioremap_wc_api: FAIL: %lu mismatches (pattern=0x%08x, bytes=0x%lx)\n",
                   errors, rw_pattern, bytes);
            return -EIO;
        }
    } else {
        for (offset = 0; offset < bytes; offset += 4) {
            u32 val = readl_relaxed(base + offset);
            pr_info("0x%0x\n", val);
        }
    }

    return 0;
}


static int test_ioremap_wc(phys_addr_t phys, unsigned long size)
{
    void __iomem *mmio;
    int ret;

    if (!IS_ALIGNED(phys, PAGE_SIZE)) {
        pr_err("test_ioremap_wc(): FAIL: phys_addr 0x%pa is not page-aligned\n",
               &phys);
        return -EINVAL;
    }

    mmio = ioremap_wc(phys, size);
    if (!mmio) {
        pr_err("test_ioremap_wc(): FAIL: ioremap_wc(0x%pa, 0x%lx) failed\n",
               &phys, size);
        return -ENOMEM;
    }

    ret = run_rw_test(mmio, size);

    iounmap(mmio);
    return ret;
}


static int test_devm_ioremap_wc(struct device *dev, phys_addr_t phys, unsigned long size)
{
    void __iomem *mmio;
    int ret;

    if (!IS_ALIGNED(phys, PAGE_SIZE)) {
        pr_err("test_devm_ioremap_wc(): FAIL: phys_addr 0x%pa is not page-aligned\n",
               &phys);
        return -EINVAL;
    }

    mmio = devm_ioremap_wc(dev, phys, size);
    if (!mmio) {
        pr_err("test_devm_ioremap_wc(): FAIL: devm_ioremap_wc(0x%pa, 0x%lx) failed\n",
               &phys, size);
        return -ENOMEM;
    }

    ret = run_rw_test(mmio, size);

    return ret;
}

static int test_pci_ioremap_wc_bar(void)
{
    unsigned int dom, bus, slot, func;
    struct pci_dev *pdev;
    void __iomem *mmio;
    resource_size_t len;
    phys_addr_t start;
    int ret;

    if (sscanf(pci_bdf, "%x:%x:%x.%x", &dom, &bus, &slot, &func) != 4) {
        pr_err("test_pci_ioremap_wc_bar(): FAIL: invalid pci_bdf format '%s', expected domain:bus:slot.func\n",
               pci_bdf);
        return -EINVAL;
    }

    pdev = pci_get_domain_bus_and_slot(dom, bus, PCI_DEVFN(slot, func));
    if (!pdev) {
        pr_err("test_pci_ioremap_wc_bar(): FAIL: pci_get_domain_bus_and_slot failed for %s\n",
               pci_bdf);
        return -ENODEV;
    }

    start = pci_resource_start(pdev, pci_bar);
    len = pci_resource_len(pdev, pci_bar);
    if (!start || !len) {
        pr_err("test_pci_ioremap_wc_bar(): FAIL: BAR%d on %s has no valid resource\n",
               pci_bar, pci_bdf);
        pci_dev_put(pdev);
        return -EINVAL;
    }

    mmio = pci_ioremap_wc_bar(pdev, pci_bar);
    if (!mmio) {
        pr_err("test_pci_ioremap_wc_bar(): FAIL: pci_ioremap_wc_bar(%s, BAR%d) failed\n",
               pci_bdf, pci_bar);
        pci_dev_put(pdev);
        return -ENOMEM;
    }

    ret = run_rw_test(mmio, len);

    iounmap(mmio);
    pci_dev_put(pdev);
    return ret;
}

static bool module_param_check(void)
{
    if (!phys_addr_param) {
        pr_err("test_ioremap_wc_api: FAIL: phys_addr_param must be set\n");
        return false;
    }
    
    if (!map_size) {
        pr_err("test_ioremap_wc_api: FAIL: map_size must be > 0\n");
        return false;
    }

    if (!pci_bdf) {
        pr_err("test_ioremap_wc_api: FAIL: pci_bdf must be set\n");
        return false;
    }

    if (pci_bar < 0 || pci_bar > 5) {
        pr_err("test_ioremap_wc_api: FAIL: pci_bar must be in range [0, 5]\n");
        return false;
    }

    return true;
}

static int __init test_ioremap_wc_api_init(void)
{
    unsigned long size_aligned;
    phys_addr_t phys;
    int ret;

    if (!module_param_check()) {
        return -EINVAL;
    }

    phys = (phys_addr_t)phys_addr_param;
    size_aligned = PAGE_ALIGN(map_size);

    /* 1) Test ioremap_wc() */
    ret = test_ioremap_wc(phys, size_aligned);
    if (ret)
        return ret;

    /* 2) Test devm_ioremap_wc() */
    test_dev = root_device_register("test_ioremap_wc_api");
    if (IS_ERR(test_dev)) {
        pr_err("test_ioremap_wc_api: FAIL: root_device_register failed\n");
        return PTR_ERR(test_dev);
    }

    ret = test_devm_ioremap_wc(test_dev, phys, size_aligned);
    if (ret) {
        root_device_unregister(test_dev);
        test_dev = NULL;
        return ret;
    }

    /* 3) Test pci_ioremap_wc_bar() */
    ret = test_pci_ioremap_wc_bar();
    if (ret) {
        root_device_unregister(test_dev);
        test_dev = NULL;
        return ret;
    }

    /* All tests passed */
    pr_info("test_ioremap_wc_api: PASSED (pattern=0x%08x)\n", rw_pattern);
    return 0;
}

static void __exit test_ioremap_wc_api_exit(void)
{
    if (test_dev) {
        root_device_unregister(test_dev);
        test_dev = NULL;
    }
}

module_init(test_ioremap_wc_api_init);
module_exit(test_ioremap_wc_api_exit);

MODULE_DESCRIPTION("ioremap_wc() + devm_ioremap_wc() + pci_ioremap_wc_bar() test with 32-bit R/W pattern check");
MODULE_AUTHOR("Richard Cheng");
MODULE_LICENSE("GPL");
