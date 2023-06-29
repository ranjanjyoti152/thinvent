#!/bin/bash

echo "Start script to create MBR and filesystem"

hasdrives=$(lsblk | grep -oE '(mmcblk[0-9])' | sort | uniq)
if [ "$hasdrives" = "" ]; then
	echo "UNABLE TO FIND ANY EMMC OR SD DRIVES ON THIS SYSTEM!!!"
	exit 1
fi

avail=$(lsblk | grep -oE '(mmcblk[0-9]|sda[0-9])' | sort | uniq)
if [ "$avail" = "" ]; then
	echo "UNABLE TO FIND ANY DRIVES ON THIS SYSTEM!!!"
	exit 1
fi

runfrom=$(lsblk | grep /$ | grep -oE '(mmcblk[0-9]|sda[0-9])')
if [ "$runfrom" = "" ]; then
	echo "UNABLE TO FIND ROOT OF THE RUNNING SYSTEM!!!"
	exit 1
fi

emmc=$(echo "$avail" | sed "s/$runfrom//" | sed "s/sd[a-z][0-9]//g" | sed "s/ //g")
if [ "$emmc" = "" ]; then
	echo "UNABLE TO FIND YOUR EMMC DRIVE OR YOU ALREADY RUN FROM EMMC!!!"
	exit 1
fi

if [ "$runfrom" = "$avail" ]; then
	echo "YOU ARE ALREADY RUNNING FROM EMMC!!!"
	exit 1
fi

if [ "$runfrom" = "$emmc" ]; then
	echo "YOU ARE ALREADY RUNNING FROM EMMC!!!"
	exit 1
fi

if [ "$(echo "$emmc" | grep mmcblk)" = "" ]; then
	echo "YOU DO NOT APPEAR TO HAVE AN EMMC DRIVE!!!"
	exit 1
fi

DEV_EMMC="/dev/$emmc"

echo "$DEV_EMMC"

echo "Start backup u-boot default"

dd if="$DEV_EMMC" of=/boot/u-boot-default-aml.img bs=1M count=4

dd if=/dev/zero of="$DEV_EMMC" bs=512 count=1

echo "Start create MBR and partition"

parted -s "$DEV_EMMC" mklabel msdos
parted -s "$DEV_EMMC" mkpart primary fat32 512M 956M
parted -s "$DEV_EMMC" mkpart primary ext4 957M 100%

echo "Start restore u-boot"

dd if=/boot/u-boot-default-aml.img of="$DEV_EMMC" conv=fsync bs=1 count=442
dd if=/boot/u-boot-default-aml.img of="$DEV_EMMC" conv=fsync bs=512 skip=1 seek=1

sync

echo "Done"

echo "Start copy system to eMMC."

DIR_INSTALL="/ddbr/install"

if [ -d "$DIR_INSTALL" ]; then
    rm -rf "$DIR_INSTALL"
fi
mkdir -p "$DIR_INSTALL"

PART_BOOT="${DEV_EMMC}p1"
PART_ROOT="${DEV_EMMC}p2"

if grep -q "$PART_BOOT" /proc/mounts; then
    echo "Unmounting BOOT partition."
    umount -f "$PART_BOOT"
fi

echo -n "Formatting BOOT partition..."
mkfs.vfat -n "BOOT_MNJRO" "$PART_BOOT"
echo "Done."

if grep -q "$PART_ROOT" /proc/mounts; then
    echo "Unmounting ROOT partition."
    umount -f "$PART_ROOT"
fi

echo "Formatting ROOT partition..."
mke2fs -F -q -t ext4 -L ROOT_MNJRO -m 0 "$PART_ROOT"
e2fsck -n "$PART_ROOT"
echo "Done."

echo "BOOT PARTITION: $PART_BOOT"
echo "ROOT PARTITION: $PART_ROOT"

ROOT_PART=$(lsblk -p -o NAME,PARTUUID | grep "$PART_ROOT" | awk '{print $2}')
BOOT_PART=$(lsblk -p -o NAME,PARTUUID | grep "$PART_BOOT" | awk '{print $2}')

echo "BOOT PARTUUID: $BOOT_PART"
echo "ROOT PARTUUID: $ROOT_PART"

mount -o rw "$PART_BOOT" "$DIR_INSTALL"

echo -n "Copying BOOT..."
cp -r /boot/* "$DIR_INSTALL" && sync
echo "Done."

sed -i "s/PARTUUID=.*02/PARTUUID=$ROOT_PART/g" "$DIR_INSTALL/uEnv.ini"

rm "$DIR_INSTALL/s9*"
rm "$DIR_INSTALL/aml*"
rm "$DIR_INSTALL/boot.ini"

if [ -f /boot/u-boot.ext ]; then
    mv -f "$DIR_INSTALL/u-boot.sd" "$DIR_INSTALL/u-boot.emmc"
    sync
fi

umount "$DIR_INSTALL"

echo "Copying ROOTFS."

mount -o rw "$PART_ROOT" "$DIR_INSTALL"

cd /
echo "Copy BIN"
tar -cf - bin | (cd "$DIR_INSTALL"; tar -xpf -)
#echo "Copy BOOT"
#mkdir -p "$DIR_INSTALL/boot"
#tar -cf - boot | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Create DEV"
mkdir -p "$DIR_INSTALL/dev"
#tar -cf - dev | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Copy ETC"
tar -cf - etc | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Copy HOME"
tar -cf - home | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Copy LIB"
tar -cf - lib | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Create MEDIA"
mkdir -p "$DIR_INSTALL/media"
#tar -cf - media | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Create MNT"
mkdir -p "$DIR_INSTALL/mnt"
#tar -cf - mnt | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Copy OPT"
tar -cf - opt | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Create PROC"
mkdir -p "$DIR_INSTALL/proc"
echo "Copy ROOT"
tar -cf - root | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Create RUN"
mkdir -p "$DIR_INSTALL/run"
echo "Copy SBIN"
tar -cf - sbin | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Copy SELINUX"
tar -cf - selinux | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Copy SRV"
tar -cf - srv | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Create SYS"
mkdir -p "$DIR_INSTALL/sys"
echo "Create TMP"
mkdir -p "$DIR_INSTALL/tmp"
echo "Copy USR"
tar -cf - usr | (cd "$DIR_INSTALL"; tar -xpf -)
echo "Copy VAR"
tar -cf - var | (cd "$DIR_INSTALL"; tar -xpf -)
sed -i "s/PARTUUID=.*01/PARTUUID=$BOOT_PART/g" "$DIR_INSTALL/etc/fstab"
sync

rm "$DIR_INSTALL/usr/bin/ddbr"

cd /
sync

umount "$DIR_INSTALL"

echo "OS clone from SD card to eMMC completed."
