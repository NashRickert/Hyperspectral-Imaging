/*
 * Dual-port SRAM device driver for an FPGA
 * Reads and writes as a block driver to a fixed address corresponding
 * to a device tree entry.
 * Thus in this way it acts as a block version of a typical platform driver.
 * Much code was adapted from drivers/block/brd.c in the linux source code
 * Which implements a standard block ram-disk. Many features from that are
 * Not implemented because they are incorrect or superfluous for our use case.
 * This code was also written for a custom petalinux version of a 6.6.10 kernel.
 * Most things will be the same for a more modern (6.6.15) kernel. 
 * What needs to change is documented in comments.
 */

#include "linux/blk_types.h"
#include "linux/err.h"
#include <linux/init.h>
#include <linux/initrd.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/major.h>
#include <linux/blkdev.h>
#include <linux/bio.h>
#include <linux/highmem.h>
#include <linux/mutex.h>
#include <linux/pagemap.h>
#include <linux/xarray.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/backing-dev.h>
#include <linux/debugfs.h>
#include <linux/vmalloc.h>
#include <linux/platform_device.h>

#include <linux/uaccess.h>
#include <linux/mod_devicetable.h>
#include <linux/types.h>
#include <linux/io.h>
#include <linux/kernel.h>
#include <linux/mutex.h>

// Valid major number for experimental use
#define MY_MAJOR 241
#define DEF_MEM_SIZE 4096
#define DRIVER_NAME "dpsram driver B"

// Note that miscdevice is for character drivers, thus not relevant here

struct dpsram_device {
	void __iomem *base;
	size_t size;
	struct platform_device *pdev;
	struct gendisk *gd;
};

// ------- BLOCK DEVICE STUFF --------

// This is the function that actually handles reading and writing
static void my_submit_bio(struct bio *bio) {
	pr_info("Inside of function %s\n", __func__);
	struct dpsram_device *dev = (struct dpsram_device *)bio->bi_bdev->bd_disk->private_data;
	sector_t sector = bio->bi_iter.bi_sector;
	u32 offset = sector << SECTOR_SHIFT;
	void __iomem *mem_base;


	// Definitely out of range
	// (Without even considering bvec length)
	// I think I need this so we don't increment mem_base? Not positive
	// Probably also viable to remove and just have the check inside for_each_segment
	if (offset > dev->size) {
		bio_io_error(bio);
		return;
	}

	mem_base = dev->base + offset;

	struct bio_vec bvec;
	struct bvec_iter iter;

	int count = 0;
	
	bio_for_each_segment(bvec, bio, iter) {
		pr_info("Inside of segment %d\n", count);
		count++;
		pr_info("dev->base: %p\n", dev->base);

		// Alignment warning gotten from 6.6.10 brd.c code
		// Possibly not important/idk what it does
		WARN_ON_ONCE((bvec.bv_offset & (SECTOR_SIZE - 1)) ||
				(bvec.bv_len & (SECTOR_SIZE - 1)));

		pr_info("BIO dir: %s, mem_base: 0x%p, offset: 0x%u, len: %u (decimal)\n", 
		    bio_data_dir(bio) == READ ? "READ" : "WRITE",
		    mem_base, offset, bvec.bv_len);

		void *iovec_mem = bvec_kmap_local(&bvec);

		pr_info("offset + bvec.bv_len is %u\n", offset + bvec.bv_len);
		pr_info("dev->size is %lu\n", dev->size);
		if (offset + bvec.bv_len > dev->size) {
			pr_info("We are out of space\n");
			bio_io_error(bio);
			kunmap_local(iovec_mem);
			return;
		}

		if (bio_data_dir(bio) == READ) {
			memcpy_fromio(iovec_mem, mem_base, bvec.bv_len);
		}
		else {
			memcpy_toio(mem_base, iovec_mem, bvec.bv_len);
		}
		kunmap_local(iovec_mem);

		pr_info("mem_base before modification: %p\n", mem_base);
		pr_info("bvec.bv.len is (in decimal) 0x%u", bvec.bv_len);

		// I think that normal pointer arithmetic would also work here
		// Note that this pointer arithmetic wildly changes the virtual address in weird ways
		// I think this is fine and expected for the iomem api because of the fact that
		// I could read and write from registers at pointer-arithmetic induced offsets
		// However because of the 64 bit line issue with our hardware, it technically never got properly
		// tested in the submit bio function. But it also didn't cause anything to crash
		mem_base = (void __iomem *)((u8 __iomem *)mem_base + bvec.bv_len);
		pr_info("mem_base after modification (currently): %p\n", mem_base);

		offset += bvec.bv_len;

	}
	bio_endio(bio);

	pr_info("Leaving submit_bio\n");
}


static const struct block_device_operations my_bops = {
	.owner = THIS_MODULE,
	.submit_bio = my_submit_bio,
};

static int my_alloc(struct dpsram_device *dev) {
	struct gendisk *disk;
	char buf[DISK_NAME_LEN];
	int err = -ENOMEM;


	// The main difference between this version of the driver and one
	// that would work for a more recent kernel (like 6.6.15) is that
	// we use the queue_limits struct and assign it to disk
	// instead of doing the the blk_queue_... calls later in this func.
	// One can check a modern version of brd.c to observe this
	/* struct queue_limits lim = { */
	/* 	.physical_block_size	= PAGE_SIZE, */
	/* 	.max_hw_discard_sectors	= UINT_MAX, */
	/* 	.max_discard_segments	= 1, */
	/* 	.discard_granularity	= PAGE_SIZE, */
	/* 	.features		= BLK_FEAT_SYNCHRONOUS | */
	/* 				  BLK_FEAT_NOWAIT, */
	/* };  */


	snprintf(buf, DISK_NAME_LEN, "dpsramB");

	// This would be used in 6.6.15 instead of the uncommented line
	/* disk = dev->gd = blk_alloc_disk(&lim, NUMA_NO_NODE); */
	disk = dev->gd = blk_alloc_disk(NUMA_NO_NODE);


	if (IS_ERR(disk)) {
		err = PTR_ERR(disk);
		goto out_free_dev;
	}

	disk->major = MY_MAJOR;
	disk->first_minor = 0;
	disk->minors = 1; 
	disk->fops = &my_bops;
	disk->private_data = dev;
	strscpy(disk->disk_name, buf, DISK_NAME_LEN);

	// We shift because it expects capacity in sector units
	set_capacity(disk, dev->size >> SECTOR_SHIFT);

	// NGL idk what this stuff does, but I got it from 6.6.10 brd.c
	// Can also use PAGE_SIZE instead of sector size here.
	blk_queue_physical_block_size(disk->queue, 1 << SECTOR_SHIFT);
	/* Tell the block layer that this is not a rotational device */
	blk_queue_flag_set(QUEUE_FLAG_NONROT, disk->queue);
	blk_queue_flag_set(QUEUE_FLAG_SYNCHRONOUS, disk->queue);
	blk_queue_flag_set(QUEUE_FLAG_NOWAIT, disk->queue);

	err = add_disk(disk);
	if (err)
		goto out_cleanup_disk;
	return 0;

out_cleanup_disk:
	put_disk(disk);
out_free_dev:
	return err;
}

static void my_cleanup(struct dpsram_device *dev) {
	if (dev->gd) {
		del_gendisk(dev->gd);
		put_disk(dev->gd);
	}
}


// ------- PROBE AND REMOVE -------


static int dpsram_probe(struct platform_device *pdev)
{
	pr_info("Inside of function %s\n", __func__);
	struct dpsram_device *dev;
	struct resource *res;
	int err;

	// Allocate driver data
	dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
	if (!dev)
	    return -ENOMEM;

	// Get memory resources
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!res) {
		dev_err(&pdev->dev, "No memory resource found\n");
		return -ENODEV;
	}
	pr_info("Found resource: start=0x%pa, size=0x%llu\n",
		&res->start, resource_size(res));

	// Map physical memory to virtual address
	dev->base = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(dev->base))
		return PTR_ERR(dev->base);
	pr_info("devm_ioremap_resource returned %p\n", dev->base);

	// This size comes from the device tree
	dev->size = resource_size(res);

	// This sets the data field of some driver struct to point to our representation
	platform_set_drvdata(pdev, dev);

	dev->pdev = pdev;

	// Do the block specific allocation
	err = my_alloc(dev);
	if (err)
		goto out_free;

	dev_info(&pdev->dev, "DPSRAM device registered at %p with size %zu\n", dev->base, dev->size);

	return 0;

out_free:
	my_cleanup(dev);
	pr_info("DPSRAM Module not loaded!\n");
	return err;
}

static int dpsram_remove(struct platform_device *pdev)
{
	struct dpsram_device *dev = platform_get_drvdata(pdev);

	my_cleanup(dev);
	
	dev_info(&pdev->dev, "FPGA SRAM block device removed\n");
	return 0;
}

static const struct of_device_id dpsram_match[] = {
	/* { .compatible = "xlnx,dpsram-port-a" }, */
	{ .compatible = "xlnx,dpsram-port-b" },
	{},
};

/*
 * One could implement autoloading of the module so that one boot it gets
 * loaded when the compatible device-tree entry is seen
 * Currently this makes it match the right entry on insmod call
 */
MODULE_DEVICE_TABLE(of, dpsram_match);

static struct platform_driver dpsram_driver = {
	.driver = {
		.name = DRIVER_NAME,
		.of_match_table = dpsram_match,
	},
	.probe = dpsram_probe,
	.remove= dpsram_remove,
};

module_platform_driver(dpsram_driver);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Nash Rickert");
MODULE_DESCRIPTION("A device driver for reading/writing from dual port SRAM memory at fixed addresses on an FPGA");
