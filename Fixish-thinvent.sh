#!/bin/bash
echo "Fixing Errors "
sudo cp /etc/xdg/konsolerc /etc/xdg/konsolerc.bak
sudo rm /etc/xdg/konsolerc
sudo pacman -Syu --noconfirm --needed