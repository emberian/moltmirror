#!/bin/bash
# Cloud-init script for AWS EC2 deployment

set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y docker.io docker-compose
systemctl enable docker
systemctl start docker

# Create app directory
mkdir -p /opt/moltmirror
cd /opt/moltmirror

# Clone repo (or copy files)
# git clone https://github.com/yourusername/moltmirror.git .

# For now, we'll expect files to be copied via SCP/RSYNC

# Create data directory
mkdir -p /data

# Set permissions
usermod -aG docker ubuntu

# Pull and run
docker-compose up -d

# Install CloudWatch agent for monitoring (optional)
# wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
# dpkg -i amazon-cloudwatch-agent.deb

echo "Moltbook Analysis API deployed!"
echo "Check status with: docker-compose logs -f"
