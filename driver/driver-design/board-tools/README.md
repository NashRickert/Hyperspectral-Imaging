#### Confirmation of the Hardware Design

I ran the script test_hardware.sh which writes a register to 0x1000000000 and confirms the same register gets read from 0x3000000000 and vice versa to ensure the two addresses are accessing the same region of memory


#### Testing Process:

Because it is known that the drivers interact well with each other from 0x100000000 to
0x3000000000, all the testing was done with only driver.

For a relatively thourough account of the testing process and commands run, refer to terminal.log, which has terminal output. Note that this is not exhaustive of the testing done, but it is exhaustive of the testing done when the driver worked properly. There was one additional test done seperately to make sure that driver b could read what driver a wrote and vice versa.

In that log file, note the following:
1. When we write "HELLO FPGA" and attempt to read it, only the first 4 characters (64 bits) were successfully written
2. We get a bus error when attempting to access addresses that are not multiples of 64 bits with devmem. Also, using devmem confirms that we did not write past the first 64 bits.
3. Messing with block size of dd has no effect on how much gets written.
4. Using seek with dd doesn't work with an excessively small block size because block io must be offset/aligned with sector size (512). Once I figure this out, we can see that we also read/write successfully offset 512 bytes into our region, as confirmed by devmem

This behavior is hypothesized (an in fact almost certainly) because in one of our pieces of hardware (I forget which one exactly -- Zackery mentioned it to me), memory is organized as 64 bit lines. Thus this would explain why we can't read/write beyond 64 bits. If this can be resolved on the hardware side, it should be checked that large reads/writes work. If this can't be resolved on the hardware side, it will be necessary to rewrite the driver as a character driver.
