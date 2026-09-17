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

# A bare IP carries no user and no key, and mDNS is exactly what is broken on
# the day you need to override with one -- so fill both in rather than relying
# on the ssh alias resolving.
if [[ "$PI" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  SSH_EXTRA=(-o StrictHostKeyChecking=no
             -o UserKnownHostsFile="$HOME/.ssh/known_hosts_saresq"
             -i "$HOME/.ssh/saresq_pi")
  PI="afzalamanullah@$PI"
else
  SSH_EXTRA=()
fi
ssh()  { command ssh  "${SSH_EXTRA[@]}" "$@"; }
scp()  { command scp  "${SSH_EXTRA[@]}" "$@"; }

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

# The thermal->RGB affine. Small, but the payload draws no heat contours
# without it and every iou/d_c feature stays zero, so it ships on every deploy
# rather than being remembered as a manual step.
if [ -f saresq/calib/thermal_to_rgb.json ]; then
  ssh "$PI" "mkdir -p $REMOTE/saresq/calib"
  scp -q saresq/calib/thermal_to_rgb.json "$PI:$REMOTE/saresq/calib/thermal_to_rgb.json"
  echo "   calibration pushed ($(python3 -c "import json;print(json.load(open('saresq/calib/thermal_to_rgb.json'))['residual_thermal_px'])" 2>/dev/null | cut -c1-5) thermal px)"
else
  echo "   NOTE: no calibration on this machine; the payload will draw no contours"
fi

# hazard.py is imported by live_pipeline at runtime; keep it in step with the
# model rather than assuming the card's copy matches.
scp -q saresq/detect/hazard.py "$PI:$REMOTE/saresq/detect/hazard.py"

# The scene classifier is 2.7 MB and is NOT part of the fast path above, which
# is deliberately two small files so a deploy finishes in a second over a phone
# hotspot. Copy it only when the Pi does not already have it -- otherwise every
# deploy would push 2.7 MB to overwrite an identical file.
HAZ=models/mobilenetv2_aider_224_int8.tflite
if [ -f "$HAZ" ]; then
  if ssh "$PI" "test -s $REMOTE/$HAZ"; then
    echo "   hazard model already on the Pi"
  else
    echo "   pushing hazard model (2.7 MB, first time only)"
    ssh "$PI" "mkdir -p $REMOTE/models"
    scp -q "$HAZ" "$PI:$REMOTE/$HAZ"
  fi
else
  echo "   WARNING: $HAZ not found locally; the scene strip will read 'model not found'"
fi

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
