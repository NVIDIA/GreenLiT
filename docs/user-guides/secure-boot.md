# Secure Boot and Module Signing

Some GreenLiT test cases build and load custom kernel modules at runtime (e.g., `bingo.ko`, `kretprobe.ko`). When **UEFI Secure Boot** is enabled, the kernel will reject unsigned modules, causing these tests to skip or fail.

This guide explains how to set up a Machine Owner Key (MOK) so that GreenLiT can automatically sign modules before loading them.

## Quick Check

To verify if Secure Boot is enabled on your system:

```bash
mokutil --sb-state
```

If the output says `SecureBoot enabled`, you need to follow this guide.

## Prerequisites

- `mokutil` - for enrolling the key
- `openssl` - for generating the key pair
- Kernel headers installed (provides `sign-file` utility)

On Ubuntu/Debian:
```bash
apt install mokutil openssl shim-signed linux-headers-$(uname -r)
```

On RHEL/CentOS:
```bash
dnf install mokutil openssl kernel-devel-$(uname -r)
```

## Step 1: Generate a MOK Key Pair

GreenLiT uses the Ubuntu default MOK key path at `/var/lib/shim-signed/mok/`, which is the same location used by `dkms` and `shim-signed`.

**If your system already has MOK keys** (e.g., from a DKMS driver installation), you can skip this step -- GreenLiT will use the existing keys automatically.

To create new keys:

```bash
sudo mkdir -p /var/lib/shim-signed/mok

sudo openssl req -new -x509 \
    -newkey rsa:2048 \
    -keyout /var/lib/shim-signed/mok/MOK.priv \
    -outform DER \
    -out /var/lib/shim-signed/mok/MOK.der \
    -nodes \
    -days 36500 \
    -subj "/CN=GreenLiT Module Signing Key/"

sudo chmod 600 /var/lib/shim-signed/mok/MOK.priv
```

Alternatively, on Ubuntu you can use the `update-secureboot-policy` tool:

```bash
sudo update-secureboot-policy --new-key
```

This creates:
- `MOK.priv` - the private key (keep this secure)
- `MOK.der` - the public certificate (enrolled into firmware)

## Step 2: Enroll the Key

```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
```

Or on Ubuntu:

```bash
sudo update-secureboot-policy --enroll-key
```

You will be prompted to create a one-time password. Remember this password -- you will need it on the next reboot.

## Step 3: Reboot and Complete Enrollment

Reboot the system:

```bash
sudo reboot
```

During boot, the **MOK Manager** (blue screen) will appear:
1. Select **Enroll MOK**
2. Select **Continue**
3. Enter the password you created in Step 2
4. Select **Reboot**

## Step 4: Verify Enrollment

After reboot, verify the key was enrolled:

```bash
mokutil --list-enrolled | grep "GreenLiT"
```

You should see the `GreenLiT Module Signing Key` in the output.

## How It Works

When GreenLiT runs a test that needs to load a custom kernel module:

1. It checks if Secure Boot is enabled
2. If enabled, it looks for MOK keys at `/var/lib/shim-signed/mok/` (Ubuntu default, shared with dkms)
3. It signs the `.ko` file using the kernel's `sign-file` utility
4. It then loads the signed module with `insmod`

If Secure Boot is disabled, no signing is performed and modules are loaded as usual.

## Troubleshooting

### "Cannot find sign-file utility"

**On Ubuntu/Debian:**
```bash
apt install linux-headers-$(uname -r)
```

**On RHEL/CentOS:**
```bash
dnf install kernel-devel-$(uname -r)
```

**On Buildroot or custom kernel builds:**

If you built your kernel from source (e.g., Buildroot, Yocto, or a manual build), the `sign-file` utility is not installed by distro packages. Copy it from your kernel build tree:

```bash
sudo cp /path/to/linux-build/scripts/sign-file /usr/local/bin/sign-file
sudo chmod +x /usr/local/bin/sign-file
```

GreenLiT checks `/usr/local/bin/sign-file` first, so placing it there will be picked up automatically.

### "MOK keys not found"

Ensure the key files exist at the expected location:

```bash
ls -la /var/lib/shim-signed/mok/
# Should show MOK.priv and MOK.der
```

### Module still rejected after signing

Verify the MOK key is properly enrolled:

```bash
mokutil --list-enrolled
```

If the key is not listed, repeat Steps 2 and 3.

### Using a custom MOK key location

If your organization uses MOK keys at a different location, you can create symbolic links:

```bash
sudo mkdir -p /var/lib/shim-signed/mok
sudo ln -s /path/to/your/key.priv /var/lib/shim-signed/mok/MOK.priv
sudo ln -s /path/to/your/cert.der /var/lib/shim-signed/mok/MOK.der
```
