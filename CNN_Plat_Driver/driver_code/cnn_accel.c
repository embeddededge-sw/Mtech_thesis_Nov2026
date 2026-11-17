// SPDX-License-Identifier: GPL-2.0
/*
 * cnn_accel.c - Platform driver for LeNet-5 CNN FPGA accelerator
 *
 * It handles:
 *   - MMIO register access via devm_ioremap_resource
 *   - Interrupt-driven AP_DONE, with polling fallback
 *   - CPU-driven BRAM loads via writel() (no DMA engine)
 *   - ioctl interface for userspace inference requests 
 */

#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/io.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/dma-mapping.h>
#include <linux/completion.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/delay.h>

#include "cnn_accel.h"

/* Compile-time guard: catch regressions before they reach the build */
_Static_assert(sizeof(struct cnn_weight_data)  < (1 << 14),
               "cnn_weight_data too large for ioctl");
_Static_assert(sizeof(struct cnn_inference_req) < (1 << 14),
               "cnn_inference_req too large for ioctl");

/* Register offsets (word-addressed in HW, byte-addressed here) */
#define CNN_CTRL_BASE           0x00000
#define CNN_CTRL_R_BASE         0x10000
#define INMODEL_BRAM_BASE       0x20000
#define WEIGHTS_BRAM_BASE       0x28000

/* AP control register bits */
#define AP_CTRL                 0x00
#define AP_START                BIT(0)
#define AP_DONE                 BIT(1)

/* Output register */
#define OUTMODEL0               0x10

/* Pointer registers */
#define R_INMODEL_1             0x10
#define R_INMODEL_2             0x14
#define R_WEIGHTS_1             0x1C
#define R_WEIGHTS_2             0x20

#define INPUT_SIZE              784
#define MAX_WEIGHTS             5738

#define CNN_DONE_TIMEOUT_MS     5000

struct cnn_accel {
    struct device           *dev;
    void __iomem            *regs;
    resource_size_t         regs_phys;
    struct miscdevice       miscdev;
    struct completion       done;
    struct mutex            lock;
    int                     irq;

    /* Scratch buffer for weights copied from userspace */
    u32                     *weight_buf;
};

static inline void cnn_write(struct cnn_accel *cnn, u32 offset, u32 val)
{
    writel(val, cnn->regs + offset);
}

static inline u32 cnn_read(struct cnn_accel *cnn, u32 offset)
{
    return readl(cnn->regs + offset);
}

static irqreturn_t cnn_irq_handler(int irq, void *data)
{
    struct cnn_accel *cnn = data;
    u32 status;

    status = cnn_read(cnn, CNN_CTRL_BASE + AP_CTRL);
    if (!(status & AP_DONE))
        return IRQ_NONE;

    complete(&cnn->done);
    return IRQ_HANDLED;
}

static void cnn_load_bram(struct cnn_accel *cnn, u32 bram_offset,
                          const u32 *data, unsigned int count)
{
    unsigned int i;

    for (i = 0; i < count; i++)
        cnn_write(cnn, bram_offset + i * 4, data[i]);
}

static void cnn_setup_pointers(struct cnn_accel *cnn)
{
    cnn_write(cnn, CNN_CTRL_R_BASE + R_INMODEL_1,
              cnn->regs_phys + INMODEL_BRAM_BASE);
    cnn_write(cnn, CNN_CTRL_R_BASE + R_INMODEL_2, 0);
    cnn_write(cnn, CNN_CTRL_R_BASE + R_WEIGHTS_1,
              cnn->regs_phys + WEIGHTS_BRAM_BASE);
    cnn_write(cnn, CNN_CTRL_R_BASE + R_WEIGHTS_2, 0);
}

static int cnn_run(struct cnn_accel *cnn)
{
    unsigned long timeout;

    reinit_completion(&cnn->done);

    /* Flush any stale AP_DONE */
    cnn_read(cnn, CNN_CTRL_BASE + AP_CTRL);

    /* Start the CNN IP */
    cnn_write(cnn, CNN_CTRL_BASE + AP_CTRL, AP_START);

    if (cnn->irq > 0) {
        timeout = wait_for_completion_timeout(&cnn->done,
                msecs_to_jiffies(CNN_DONE_TIMEOUT_MS));
        if (!timeout) {
            dev_err(cnn->dev, "CNN inference timed out\n");
            return -ETIMEDOUT;
        }
    } else {
        timeout = jiffies + msecs_to_jiffies(CNN_DONE_TIMEOUT_MS);
        while (!(cnn_read(cnn, CNN_CTRL_BASE + AP_CTRL) & AP_DONE)) {
            if (time_after(jiffies, timeout)) {
                dev_err(cnn->dev, "CNN inference poll timed out\n");
                return -ETIMEDOUT;
            }
            usleep_range(10, 50);
        }
    }

    return 0;
}

static int cnn_read_output(struct cnn_accel *cnn)
{
    u32 raw;

    raw = cnn_read(cnn, CNN_CTRL_BASE + OUTMODEL0);
    return (int)(raw >> 16);
}

static long cnn_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct cnn_accel *cnn = container_of(file->private_data,
                                         struct cnn_accel, miscdev);
    int ret;

    switch (cmd) {
    case CNN_IOCTL_LOAD_WEIGHTS: {
        struct cnn_weight_data wd;

        if (copy_from_user(&wd, (void __user *)arg, sizeof(wd)))
            return -EFAULT;

        if (wd.count == 0 || wd.count > MAX_WEIGHTS)
            return -EINVAL;
        if (!wd.data)
            return -EINVAL;

        mutex_lock(&cnn->lock);

        /* Copy weights from userspace into the persistent scratch buffer */
        if (copy_from_user(cnn->weight_buf,
                           (const void __user *)wd.data,
                           wd.count * sizeof(u32))) {
            mutex_unlock(&cnn->lock);
            return -EFAULT;
        }

        cnn_load_bram(cnn, WEIGHTS_BRAM_BASE, cnn->weight_buf, wd.count);

        mutex_unlock(&cnn->lock);
        return 0;
    }
    case CNN_IOCTL_RUN_INFERENCE: {
        struct cnn_inference_req req;

        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        if (req.input_size != INPUT_SIZE)
            return -EINVAL;

        mutex_lock(&cnn->lock);
        cnn_setup_pointers(cnn);
        cnn_load_bram(cnn, INMODEL_BRAM_BASE, req.image, INPUT_SIZE);

        ret = cnn_run(cnn);
        if (ret) {
            mutex_unlock(&cnn->lock);
            return ret;
        }

        req.result = cnn_read_output(cnn);
        mutex_unlock(&cnn->lock);

        if (copy_to_user((void __user *)arg, &req, sizeof(req)))
            return -EFAULT;
        return 0;
    }

    case CNN_IOCTL_RUN_NONHLS: {
        struct cnn_inference_req req;
        u32 img_words[784]; // 1 pixel per 32-bit word (matches hardware)
        int i;
        u32 raw;

        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        if (req.input_size != INPUT_SIZE)
            return -EINVAL;

        // 1 pixel per 32-bit word- hardware reads RDATA[7:0] from each word
        for (i = 0; i < 784; i++) {
		img_words[i] = req.image[i] & 0xFF;
        }

        mutex_lock(&cnn->lock);
        
        // Setup pointers (tells the IP where the image is)
        cnn_setup_pointers(cnn);
        
        // Load 784 words into BRAM (1 pixel per word)
        cnn_load_bram(cnn, INMODEL_BRAM_BASE, img_words, 784);

        // Run the accelerator
        ret = cnn_run(cnn);
        if (ret) {
            mutex_unlock(&cnn->lock);
            return ret;
        }

        // Read output: Udemy outputs raw 0-9, NOT fixed point
        raw = cnn_read(cnn, CNN_CTRL_BASE + OUTMODEL0);
        req.result = (int)((raw >> 16) & 0xF); 

        mutex_unlock(&cnn->lock);

        if (copy_to_user((void __user *)arg, &req, sizeof(req)))
            return -EFAULT;
        return 0;
    }

    default:
        return -ENOTTY;
    }
}

static const struct file_operations cnn_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = cnn_ioctl,
};

static int cnn_accel_probe(struct platform_device *pdev)
{
    struct cnn_accel *cnn;
    struct resource *res;
    int ret;

    cnn = devm_kzalloc(&pdev->dev, sizeof(*cnn), GFP_KERNEL);
    if (!cnn)
        return -ENOMEM;

    cnn->dev = &pdev->dev;
    mutex_init(&cnn->lock);
    init_completion(&cnn->done);

    /* Allocate a persistent weights scratch buffer for userspace copies */
    cnn->weight_buf = devm_kzalloc(&pdev->dev,
                                   MAX_WEIGHTS * sizeof(u32),
                                   GFP_KERNEL);
    if (!cnn->weight_buf)
        return -ENOMEM;

    /* Map register space */
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    cnn->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(cnn->regs))
        return PTR_ERR(cnn->regs);

    cnn->regs_phys = res->start;

    /* Request interrupt (optional - falls back to polling) */
    cnn->irq = platform_get_irq_optional(pdev, 0);
    if (cnn->irq > 0) {
        ret = devm_request_irq(&pdev->dev, cnn->irq, cnn_irq_handler,
                               0, dev_name(&pdev->dev), cnn);
        if (ret) {
            dev_warn(&pdev->dev,
                     "IRQ request failed (%d), using polling\n",
                     ret);
            cnn->irq = 0;
        } else {
            dev_info(&pdev->dev, "using interrupt-driven mode\n");
        }
    } else {
        dev_info(&pdev->dev, "no IRQ, using polling mode\n");
        cnn->irq = 0;
    }

    /* Register misc device */
    cnn->miscdev.minor = MISC_DYNAMIC_MINOR;
    cnn->miscdev.name  = "cnn0";
    cnn->miscdev.fops  = &cnn_fops;
    cnn->miscdev.parent = &pdev->dev;

    ret = misc_register(&cnn->miscdev);
    if (ret) {
        dev_err(&pdev->dev, "misc_register failed: %d\n", ret);
        return ret;
    }

    platform_set_drvdata(pdev, cnn);
    dev_info(&pdev->dev, "CNN accelerator registered as /dev/cnn0\n");

    return 0;
}

static void cnn_accel_remove(struct platform_device *pdev)
{
    struct cnn_accel *cnn = platform_get_drvdata(pdev);

    misc_deregister(&cnn->miscdev);
    dev_info(&pdev->dev, "CNN accelerator removed\n");
}

static const struct of_device_id cnn_accel_of_match[] = {
    { .compatible = "custom,cnn-accel" },
    { /* sentinel */ }
};
MODULE_DEVICE_TABLE(of, cnn_accel_of_match);

static struct platform_driver cnn_accel_driver = {
    .probe  = cnn_accel_probe,
    .remove = cnn_accel_remove,
    .driver = {
        .name           = "cnn-accel",
        .of_match_table = cnn_accel_of_match,
    },
};
module_platform_driver(cnn_accel_driver);

MODULE_AUTHOR("Mir Faisal");
MODULE_DESCRIPTION("Platform driver for LeNet-5 CNN FPGA accelerator");
MODULE_LICENSE("GPL");
