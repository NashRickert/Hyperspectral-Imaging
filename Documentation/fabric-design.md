### Beginning the project
1. Navigate to $proj_home/<project>/hardware/xilinx-kv260-startkit-2024.1
2. Open a .xpr file for a project with vivado (vivado xilinx-kv260-starterkit-2024.1.xpr). Blank project. Note this was the project that already existed by following the Hello World documentation previously.
3. Ensure that the language is VHDL from tools->settings
4. From the add command in the diagram interface, add first an AXI BRAM controller
5. Using the same add command, add the a Block Memory Generator
----

### Modifying Settings
1. Double click the blk_mem_gen (the generator you added)
2. Under basic, change the mode to standalone.
3. Change the memory type to uram
4. Change the port type to true dual port RAM
5. Navigate to the Port A section
6. Change the width to 32 bits
7. Change the depth to 4096 bits (for our purposes, since our device tree entries are 16Kb)
8. Do the same for Port B (or it might match automatically)
---
1. Double click the MPSoC process system that already existed in your project
2. Go to the PS-PL section. Under interfaces->master, enable AXI HPM0 FPD
---
1. Double click axi_bram_ctrl_0 (the axi BRAM controller you added)
2. Change protocol to AXI4
3. Change data width to 32 (to match the memory controller) (note I think this matching doesn't actually matter)
4. For us memory depth was 8192 and unmodifable. This will be fine since generator width is 4096, but if you can change this to 4096 to match the memory controller, do it
5. Note that this will likely change automatically later. We will address this later in this documentation
6. Set number of BRAM interfaces to 1
7. Change enable ECC to No
----
1. Copy paste the now modified axi_bram_ctrl_0 to make another one (for the sake of having two seperate drivers)
----
1. In the same way as before, add an AXI smart interconnect. Double click it and modify it so that we have 1 slave interface and 2 master interfaces
----
1. Drag connect the MPSoC M_AXI_HPM0_FPD port to S00_AXI on the smart connect we added.
----
1. Double click the MPSoC again
2. Go to clock configuration -> output clocks -> low power domain clocks -> PL fabric clocks. Then you should enable PL0, PL1 and set the requested freq to the max (which is 100). Only the first one is strictly necessary, the second may be useful for debugging.
3. In PS-PL config, go to general. Turn fabric reset enable on. Set the number of fabric resets to 1.
----

### Establish Connections
1. Select and then right click the pl_clk0 device on the MPSoC (or whatever your exported clock is). Select the make connection option. Select s_axl_aclk from both bram_ctrl0 and bram_ctrl1 and also select aclk from smart_connect0. Also select max_hpm0_fpd_aclk from zynq_ultra_ps_e_0. For our instructions so far, this should correspond to all available options.
2. Also go to make connection for pl_resetn0. Connect to the same clocks as before for the smart connect and both BRAM controllers.
----
1. Connect M00_AXI on smart_connect0 to S_AXI on axi_bram_ctrl_0. Likewise connect M01_AXI to S_AXI on axi_bram_ctrl_1
2. On axi_bram_ctrl_0, connect BRAM_PORTA to BRAM_PORTA on blk_mem_gen0. On axi_bram_ctrl1, connect BRAM_PORTA to BRAM_PORTB on blk_mem_gen0.
----
1. Save the file (with file-> save block design)
----

### Create custom addresses
1. In the address editor (instead of the diagram section) right click both axi_bram_ctrl_0/S_AXI and axi_bram_ctrl_1/S_AXI and assign them. They will be automatically assigned. Edit the base addresses of each to be as desired (for us this is 0x0010_0000_0000 and 0x0030_0000_0000).
2. Also change the range of each of these to 16k which corresponds to size of our device tree entry. Getting this wrong will cause bugs
----

### Generate Results and Export
1. Confirm the project wrapper in the sources section is right. If it isn't, right click on the block design file, click create HDL wrapper, and delete the old one. Make sure the new one is at the top level of the hierarchy.
----
1. Save the file
----
Generate the block design under the ip integrator section. After that is done, generate the bitstream.

Now you need to re-follow some of the steps from Zackery's hello world documentation: https://github.com/night1rider/Xilinx-KR260-Intro/blob/Hello-World/Documentation/00_hello_world_kr260.md

In particular, navigate to section 6: Vivado Base Hardware Project and export the hardware as described (with a new name). In the following section (Petalinux Generation), follow step 1 to export the hw-configuration, but do not do anything else. Note that after checking the hardware you should check the date of <project>/images/linux/system.bit to make sure it's updated. If it's not, you should search the system for .bit files. You might have to manually copy it into the images/linux directory. This is a petalinux issue. Then jump to section 13: Setting up the KR260 to use the NFS server and TFTP server. Follow the instructions starting at roughly step 7 where you build petalinux and export the uboot to the board and set it as default.

If you're using an ubuntu root fs, you may have to rename it, boot from the petalinux rootfs, (following instructions in section 13), use the xmutil command, and then move back to using the ubuntu rootfs. This is because xmutil won't be available on ubuntu.

Now do this:
1. cd /opt/Xilinx/Vivado/2024.1/data/xicom/cable_drivers/lin64/install_script/install_drivers/
sudo ./install_drivers
2. Make sure passthrough to the VM is enabled for your USB, and unplug and replug to the USB.
3. Open vivado to the project.
4. Under open hardware manager, select open target and autoconnect. 
5. Under open hardware manager, select program device and the device. Do this before running any drivers (it configures the hardware properly). Make sure you have the proper .bit file (Can check the date of it if you want)
Now in theory, writing and reading from your existing device tree nodes will work properly.

Note that the xmutil commands will cause the board to automatically load the system.bit configuration on boot. If you only need to change stuff about the fabric design, not the connections to the processing system (eg changing the range of a bram controller), you don't need to reflash with xmutil, but you should note that you'll have to reprogram the device on every boot because it won't be automatically loaded

IMPORTANT: Check the depth of the two bram controllers near the end of your design process. It's related to the range that you assigned to the fixed addresses. There's a good chance that vivado messed with it after you assigned that range. Just try to make sure that the depth and bit count multiplies to 16Kb. Less will cause issues in your driver. (To be clear, everything needs to match your device-tree entry which for us was 16Kb). I made a mistake where the controllers were only 4Kb and this meant that outside of the first page, we would hit a kernel panic when testing. Keep all this in mind. You may need to save and reload vivado a couple times because that seems to be when it likes to apply its automatic adjustments without telling you >:(. Btw for some reason sometimes the depth seems to get fixed and the only way to adjust is with the register size, so you might end up with something ugly like 256 bits and 512 depth instead of the nicer 32 bit width and 4096 depth. This is fine.

To see what a properly setup vivado setup (in terms of connections, not setttings), view the picture in this directory.

