#!/usr/bin/env bash
# One-shot server setup for SourceMind on Ubuntu 24.04 (t3.micro, 1 GB RAM).
# Run as root on the instance after /tmp/sourcemind.tgz has been uploaded:
#   sudo DOMAIN=<ip-with-dashes>.sslip.io bash provision.sh
# Idempotent: safe to re-run for redeploys.
set -euo pipefail
: "${DOMAIN:?set DOMAIN=<ip-with-dashes>.sslip.io}"
APP=/opt/sourcemind
export DEBIAN_FRONTEND=noninteractive

# 1. 2 GB swap so torch + bge-small never OOM on 1 GB RAM.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
sysctl -q -w vm.swappiness=20 && echo 'vm.swappiness=20' > /etc/sysctl.d/99-swappiness.conf

# 2. Packages: Python 3.12 (Ubuntu 24.04 default; RapidOCR needs <=3.12), OpenCV libs for OCR, Caddy.
apt-get update -q
apt-get install -y -q python3-venv python3-dev build-essential libgl1 libglib2.0-0 \
  debian-keyring debian-archive-keyring apt-transport-https curl gnupg
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q && apt-get install -y -q caddy
fi

# 3. App user + code (tarball keeps .env and storage/ in place on redeploy).
id sourcemind >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/sourcemind --shell /usr/sbin/nologin sourcemind
mkdir -p "$APP"
tar -xzf /tmp/sourcemind.tgz -C "$APP"
mkdir -p "$APP/.cache" "$APP/storage"
chown -R sourcemind:sourcemind "$APP"
[ -f "$APP/.env" ] && chmod 600 "$APP/.env"

# 4. Python venv: CPU-only torch first (the default Linux wheel pulls ~3 GB of CUDA libs).
sudo -u sourcemind -H bash -c "
  set -e
  cd $APP/backend
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
  .venv/bin/pip install -q -r requirements.txt
"

# 5. systemd + Caddy.
install -m 644 "$APP/deploy/sourcemind.service" /etc/systemd/system/sourcemind.service
install -m 644 "$APP/deploy/Caddyfile" /etc/caddy/Caddyfile
echo "DOMAIN=$DOMAIN" > /etc/default/caddy
mkdir -p /etc/systemd/system/caddy.service.d
printf '[Service]\nEnvironmentFile=/etc/default/caddy\n' > /etc/systemd/system/caddy.service.d/domain.conf
systemctl daemon-reload
systemctl enable sourcemind caddy
systemctl restart sourcemind
systemctl restart caddy
echo "Provisioned. Watch: journalctl -u sourcemind -f"
