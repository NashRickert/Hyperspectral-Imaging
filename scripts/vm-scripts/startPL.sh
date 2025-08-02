#!/bin/bash
# Set the prompt style before sourcing
PETALINUX_PROMPT="[\[\e[35m\]PetaLinux\[\e[m\]]\[\e[35m\]\u@\h\[\e[m\]:\[\e[35m\]\w\[\e[m\]\$ "
export PETALINUX_PROMPT

# Source PetaLinux settings
source /opt/Xilinx/PetaLinux/2024.1/tool/settings.sh

# Set the custom prompt
export PS1=$PETALINUX_PROMPT

# Start new shell with the prompt
exec bash --rcfile <(echo "PS1='$PETALINUX_PROMPT'")
