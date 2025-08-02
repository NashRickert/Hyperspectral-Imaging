### How to compile your module

You will likely run into issues with a standard module compilation since the compilation the the custom xilinx kernel differs from that of the module.

This is not a very elegant solution, but what I did to resolve this is enter a devshell on my VM through

petalinux-build -c kernel -x devshell

Note that may want to run

petalinux-build -x mrproper

and

petalinux-build -c kernel -x do_compile

before doing this, although it is probably optional. It is what I did and it will take a while, so feel free to skip and see if it works.

The devshell command will open a new terminal inside of a devshell environment where everything is as it was to compile the kernel. Your driver development directory should exist entirely inside your VM, not on the NFS. The reason for this will be explained later. Navigate to that directory. Write a makefile which points to this directory:

KDIR := /opt/Xilinx/Projects/KV260_Base_PetaLinux/build/tmp/work/xilinx_k26_kv-xilinx-linux/linux-xlnx/*/linux-xilinx_k26_kv-standard-build

Note that if the kernel wasn't compiled at least once, this might not contain everything necessary to build kernel modules, so run
petalinux-build -c kernel
(should only need to happen once, or everytime after you clean with petalinux). Now running the Makefile should hopefully work for you. You can confirm if you're ready by running find <project>/build -name "Module.symvers" and seeing it come up in the aforementioned directory. If it does, then it should be ready to be used to compile the module.

Note it doesn't seem like it's possible to run a petalinux-build command while a devshell is open, so exit the devshell when you need to do this and re-enter later.

Note this likely means that the steps elsewhere in the other documentation I gave to acquire kernel headers on the Ubuntu side are redundant/unnecessary.

This is what my Makefile looked like:


    # To be used in a devshell from the VM

    obj-m += dpsram_driver.o

    KDIR := /opt/Xilinx/Projects/KV260_Base_PetaLinux/build/tmp/work/xilinx_k26_kv-xilinx-linux/linux-xlnx/*/linux-xilinx_k26_kv-standard-build

    all:
	    make -C $(KDIR) M=$(PWD) modules

    clean:
	    make -C $(KDIR) M=$(PWD) clean

#### Important Note on Devshell Aborting

I got some significant issues with devshell aborting either immediately after entering or after the first command I would run. This for me was due to path mismatches in my NSF server where the tracked files ended out of sync for some reason. The way I resolved this was by navigating to /opt/Xilinx/Projects/KV260_Base_PetaLinux/build/tmp/work/xilinx_k26_kv-xilinx-linux/linux-xlnx/6.6.10-xilinx-v2024.1+gitAUTOINC+3af4295e00-r0/pseudo and running rm *. This deleted the database caches that were causing issues. I also ran this while the board was not powered up so the NFS server wasn't running at the moment, but I don't think this helped anything since I still got the issue in this setting until I ran the rm command. But I figured I should mention it. There was a previous point where I would sometimes get aborted but not always (perhaps based on board/vm synchrocity), but at a certain point it was happening everytime until I solved it.

#### Important Note on 2 drivers:

Make sure they have different names, disk numbers, and whatnot. There is DEVICE_NAME, the name from doing snprintf into buffer, and the major number to consider. Also obviously make sure that their match table compatible strings are different so that they bind to different device tree entries

#### Problem: system.dtb got killed

I encountered this issue when booting once, after doing apparently nothing that would seem to cause it aside from testing my drivers (which shouldn't affect the device tree, especially on the VM side). This was in the process of booting, after the image was loaded:

		 #########################################################################
		 ############
		 12.8 MiB/s
	done
	Bytes transferred = 36524976 (22d53b068.2.10; our IP address is 192.168.2.20
	Filename 'system.dtb'.
	Load address: 0x40000000
	Loading: #
		 0 Bytes/s
	done
	## Loading kernel from FIT Image at 08000000 ...
	   Using 'conf   2024-04-27   5:22:24 UTC
	     Type:         Kernel Image
	  cture: AArch64
	     OS:           Linux
	     Load Address: 0x05a9c75cf4c824b9c6e69a804532dda1
	   Verifying Hash Integrity ...+ OK
	ERROR: Did not find a cmdline Flattened Device Tree
	Could not find a valid device tree

I investigated and found that /srv/tftp/system.dtb was empty (or at least the thing it symlinked to was empty). Again I have literally no idea why, but I resolved it by regenerating the device-tree by running petalinux-build -c device-tree

Note that this advice also applies to stuff like having a missing kernel image (in which case you run petalinux-build -c kernel), etc.
