#!/bin/bash

# Define the source and destination paths
sd_card="/dev/mmcblk0"
emmc="/dev/mmcblk1"

echo "Cloning OS from Manjaro ARM SD card to eMMC..."

# Unmount the partitions on the SD card
sudo umount ${sd_card}*

# Unmount the partitions on the eMMC
sudo umount ${emmc}*

# Clone the boot partition
echo "Cloning boot partition..."
sudo dd if=${sd_card}p1 of=${emmc}p1 bs=4M

# Clone the root partition
echo "Cloning root partition..."
sudo dd if=${sd_card}p2 of=${emmc}p2 bs=4M

echo "OS cloning completed."
