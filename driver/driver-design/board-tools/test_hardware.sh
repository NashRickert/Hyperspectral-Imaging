#!/bin/bash

# Checks that we can write from 0x10_0000_0000 and read from 0x30_0000_0000
# And vice versa to confirm hardware is properly setup
sudo busybox devmem 0x1000000000 32 0x12345678
echo "Should see 0x12345678"
sudo busybox devmem 0x3000000000 32  # Should show 0x12345678

sudo busybox devmem 0x3000000000 32 0x87654321
echo "Should see 0x87654321"
sudo busybox devmem 0x1000000000 32  # Should show 0x87654321
