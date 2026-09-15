#!/usr/bin/env bash
# Push the demo code from the Mac to the Pi and restart the service.
#
#     bash tools/deploy_demo.sh
#
# Run this from the repo on the laptop, with the Pi powered on and on the same
# network. It copies only what the demo needs, so it finishes in a second or
# two over a phone hotspot.
set -euo pipefail

PI=${PI:-saresq}                      # ~/.ssh/config alias; override with PI=...
REMOTE=/home/afzalamanullah/saresq

echo "== reaching $PI =="
ssh -o ConnectTimeout=8 "$PI" true || {
  echo "cannot reach $PI."
  echo "  - is the Pi powered and the green LED settled?"
  echo "  - is exactly ONE network with the demo SSID switched on?"
  exit 1
}

echo "== copying =="
scp -q tools/live_pipeline.py "$PI:$REMOTE/tools/live_pipeline.py"
scp -q configs/pipeline.yaml  "$PI:$REMOTE/configs/pipeline.yaml"

echo "== restarting =="
# A syntax error here would leave the service crash-looping under
# Restart=always with nothing on screen, so the file is compiled before the
# running demo is touched.
ssh "$PI" "python3 -m py_compile $REMOTE/tools/live_pipeline.py" \
  || { echo "the new file does not compile; service left alone"; exit 1; }
ssh "$PI" "sudo systemctl restart saresq-demo && sleep 5 && \
           systemctl is-active saresq-demo"

echo
echo "== live at =="
ssh "$PI" "hostname -I" | tr ' ' '\n' | grep -E '^[0-9]' | sed 's|^|   http://|;s|$|:8091|'
echo "   http://saresq.local:8091"
