#!/usr/bin/env bash
# Installs the three systemd services. Run as root (sudo).
#   sudo bash deploy/install_systemd.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DST=/etc/systemd/system

install -m 644 "$HERE/systemd/celegans-worker.service"  "$DST/"
install -m 644 "$HERE/systemd/celegans-webhook.service" "$DST/"
install -m 644 "$HERE/systemd/celegans-poller.service"  "$DST/"

systemctl daemon-reload
systemctl enable --now celegans-worker.service celegans-webhook.service celegans-poller.service
systemctl status --no-pager celegans-worker.service celegans-webhook.service celegans-poller.service || true
echo "Services installed. Check with: systemctl status celegans-*"
