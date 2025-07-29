#!/bin/bash

case "$1" in
  vbox)
    sudo modprobe -r kvm_intel kvm
    echo "KVM disabled. VirtualBox ready."
    ;;
  kvm)
    sudo modprobe kvm
    sudo modprobe kvm_intel
    echo "KVM enabled."
    ;;
  *)
    echo "Usage: $0 {vbox|kvm}"
    ;;
esac
