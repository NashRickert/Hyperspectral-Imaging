# To be used in a devshell from the VM

obj-m += a-dpsram.o b-dpsram.o

# KDIR := /opt/Xilinx/Projects/KV260_Base_PetaLinux/build/tmp/work-shared/xilinx-k26-kv/kernel-source
KDIR := /opt/Xilinx/Projects/KV260_Base_PetaLinux/build/tmp/work/xilinx_k26_kv-xilinx-linux/linux-xlnx/*/linux-xilinx_k26_kv-standard-build

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean
