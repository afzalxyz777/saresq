#!/usr/bin/env bash
# Ship the trained payload to the Raspberry Pi.
#
#   bash tools/deploy_to_pi.sh              # models + inference code
#   bash tools/deploy_to_pi.sh --with-deps  # also install the Pi-side runtime
#
# What crosses the wire is deliberately tiny: roughly 14 MB of models plus the
# pure-Python inference package. The Pi gets NO torch, NO ultralytics and NO
# TensorFlow -- it runs the ~3 MB LiteRT runtime and nothing else. Training
# needs 4 GB and a GPU; inference needs 38 MB and a CPU, and those are different
# problems solved on different machines.
#
# Measured footprint of the 160 px detector on the build machine:
#   model file 3.46 MB | tensor arena 8.39 MB | whole Python process 38.2 MB
# against ~905 MB usable on the 1 GB Pi 4.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${PI_HOST:-saresq}"          # ~/.ssh/config alias; see docs/pi_setup.md
DEST="${PI_DEST:-~/saresq}"
WITH_DEPS=0
[ "${1:-}" = "--with-deps" ] && WITH_DEPS=1

command -v rsync >/dev/null || { echo "FATAL: rsync not found"; exit 1; }

echo "== checking $HOST is reachable"
ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" 'echo "  connected: $(hostname), $(uname -m)"' || {
  echo "FATAL: cannot reach '$HOST'."
  echo "  The Pi's IP is DHCP and mDNS proved unreliable. Re-find it by MAC:"
  echo "    for i in \$(seq 2 40); do ping -c1 -t1 192.168.1.\$i >/dev/null 2>&1 & done; wait"
  echo "    arp -a | grep -i '88:a2:9e'"
  echo "  then update HostName in ~/.ssh/config."
  exit 1
}

echo "== models to ship"
shopt -s nullglob
MODELS=(models/*.tflite models/*.json)
[ ${#MODELS[@]} -gt 0 ] || { echo "FATAL: models/ is empty. Run training/run_all.sh first."; exit 1; }
for m in "${MODELS[@]}"; do printf "   %-46s %6.2f MB\n" "$m" "$(echo "scale=2; $(stat -f%z "$m")/1000000" | bc)"; done

echo "== creating remote layout"
ssh "$HOST" "mkdir -p $DEST/models $DEST/configs $DEST/results"

# Inference code only. training/ never ships -- it is useless on the Pi and
# pulls imports the Pi does not have.
echo "== syncing inference package"
rsync -az --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'dashboard/static' \
  saresq/ "$HOST:$DEST/saresq/"

echo "== syncing models + config"
rsync -az "${MODELS[@]}" "$HOST:$DEST/models/"
rsync -az configs/pipeline.yaml "$HOST:$DEST/configs/"

if [ "$WITH_DEPS" = "1" ]; then
  echo "== installing the Pi-side runtime (LiteRT, not TensorFlow)"
  ssh "$HOST" "bash -lc '
    set -e
    test -d ~/saresq-venv || python3 -m venv ~/saresq-venv
    ~/saresq-venv/bin/pip install --quiet --upgrade pip
    ~/saresq-venv/bin/pip install --quiet ai-edge-litert numpy opencv-python-headless scipy pyyaml
    echo \"  installed:\"; ~/saresq-venv/bin/pip list 2>/dev/null | grep -iE \"litert|numpy|opencv\"
  '"
fi

echo "== verifying on the Pi"
ssh "$HOST" "bash -lc '
  cd $DEST
  ~/saresq-venv/bin/python - <<PYEOF
import glob, time, numpy as np, sys
sys.path.insert(0, \".\")
from saresq.detect.tflite_detector import CropDetector
m = sorted(glob.glob(\"models/*160_w8a32.tflite\"))
if not m:
    print(\"  no 160 px model found\"); raise SystemExit(1)
det = CropDetector(m[0], conf=0.25)
print(f\"  loaded {m[0]}  layout={det.model.layout}  boxes={det.box_units}\")
img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
det.detect(img)
ts = [ (lambda t0: (det.detect(img), (time.perf_counter()-t0)*1000)[1])(time.perf_counter()) for _ in range(20) ]
print(f\"  ON-PI LATENCY: median {np.median(ts):.1f} ms, p90 {np.percentile(ts,90):.1f} ms per 160px crop\")
budget = np.median(ts)
for n in (2, 6):
    print(f\"    {n} crops/frame -> {budget*n:.0f} ms  ({\"OK\" if budget*n < 250 else \"TIGHT\"} against the 4 Hz thermal rate)\")
PYEOF
'"

echo
echo "== done. The Pi now has the models and inference code."
echo "   It has NO torch/ultralytics/TensorFlow, by design."
