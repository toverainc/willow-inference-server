#!/usr/bin/env bash
set -e

 # Install Docker
curl https://get.docker.com | sh && sudo systemctl --now enable docker

 # Install NVIDIA Container Toolkit 
curl https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o /etc/apt/sources.list.d/nvidia-container-toolkit.list 

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# ...existing code...
echo "It is recommended to reboot the system to activate the NVIDIA drivers. Reboot now with: sudo reboot"