#!/bin/bash

# turns off kvm thing so vbox can start
sh toggle_kvm.sh vbox


# Enable direct connect to the board (with 192.168.2.20)
sudo ip addr add 192.168.2.1/24 dev enp44s0
sudo ip link set enp44s0 up

sleep 3

# Opens relevant board tty
sudo minicom -D /dev/ttyUSB1 -b 115200


