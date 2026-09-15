#!/usr/bin/env bash
# Make the live pipeline start by itself on boot, so the demo needs no laptop,
# no SSH and nobody typing. Run once, on the Pi:
#
#     bash ~/saresq/tools/setup_demo.sh
#
# After this the Pi powers on and serves the demo within about 40 seconds.
# Nothing to remember on the day, nothing to go wrong in front of judges.
set -euo pipefail

REPO="$HOME/saresq"
VENV="$HOME/saresq-venv/bin/python3"
SERVICE=/etc/systemd/system/saresq-demo.service

[ -x "$VENV" ]                     || { echo "no venv at $VENV"; exit 1; }
[ -f "$REPO/tools/live_pipeline.py" ] || { echo "no live_pipeline.py"; exit 1; }

echo "== installing autostart service =="
sudo tee "$SERVICE" >/dev/null <<EOF
[Unit]
Description=SaResQ live pipeline (thermal + RGB + detector)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO
# Restart matters more than usual here: a torn I2C read or a camera hiccup
# should never end the demo. systemd brings it straight back.
ExecStart=$VENV $REPO/tools/live_pipeline.py --port 8091
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable saresq-demo.service
sudo systemctl restart saresq-demo.service
sleep 6

echo
echo "== status =="
systemctl --no-pager --lines=8 status saresq-demo.service || true

echo
echo "== reachable at =="
for ip in $(hostname -I); do echo "   http://$ip:8091"; done
echo
echo "Stop it:    sudo systemctl stop saresq-demo"
echo "Disable:    sudo systemctl disable saresq-demo"
echo "Watch logs: journalctl -u saresq-demo -f"
