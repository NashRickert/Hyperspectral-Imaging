#!/bin/bash

# Has dmesg write continuously to this file in the background
dmesg --follow > dmesg.log &
