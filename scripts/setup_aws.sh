#!/usr/bin/env bash
# One-time setup for a g5.xlarge (A10G 24 GB) running plain Amazon Linux 2023 (no Deep Learning AMI),
# with the root EBS volume enlarged to ~100 GB in the console.
#
#   sudo bash scripts/setup_aws.sh        # system part: disk, packages, NVIDIA driver
#   bash scripts/install_env.sh           # user part: venv + Python packages (run as ec2-user)
#
# Everything lives on the root EBS volume under ~/work, so stopping the instance loses nothing.
set -euxo pipefail

# --- grow root partition + filesystem to the enlarged EBS volume ------------
growpart /dev/nvme0n1 1 || true          # "NOCHANGE" if already grown
xfs_growfs -d / || true

# --- base tools -------------------------------------------------------------
dnf install -y git tmux htop python3.12 python3.12-devel python3.12-pip gcc gcc-c++ make

# --- NVIDIA driver (open kernel module via DKMS; A10G = Ampere, supported) ---
dnf install -y nvidia-release
dnf install -y "kernel6.18-devel-$(uname -r | sed 's/\.x86_64$//')" || dnf install -y "kernel-devel-$(uname -r)"
dnf install -y dkms
dnf install -y nvidia-open nvidia-driver-cuda
nvidia-smi || echo "nvidia-smi failed: reboot the instance, then re-check"
echo SETUP_DONE
