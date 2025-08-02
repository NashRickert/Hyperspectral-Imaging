#!/opt/Xilinx/PetaLinux/2024.1/tool/sysroots/x86_64-petalinux-linux/bin/bash
# Run this from your petalinux project directory (or modify the script to cd in there) and make sure you're in petalinux shell
rm /opt/Xilinx/Projects/KV260_Base_PetaLinux/build/tmp/work/xilinx_k26_kv-xilinx-linux/linux-xlnx/6.6.10-xilinx-v2024.1+gitAUTOINC+3af4295e00-r0/pseudo/*
petalinux-build -c kernel -x devshell
