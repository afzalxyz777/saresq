#!/usr/bin/env bash
# YOLO11n vs YOLOv8n A/B -- replicates the SHIPPED v8n recipe exactly, swapping
# only the backbone checkpoint.
#
# The shipped detector was not one run. Read from each run's args.yaml:
#
#   stage 0  yolov8n.pt          -> HIT-UAV        30 ep, batch 16, patience 10
#   stage 1  stage0 best.pt      -> thermal_combined 20 ep, batch 12, patience 8
#   stage 2  stage1 best.pt      -> thermal_combined 25 ep, batch 12, patience 10
#
# Giving 11n a single cold 45-epoch run would NOT be the same experiment: v8n
# reached the combined set with 30 epochs of HIT-UAV already behind it. Every
# other hyperparameter (AdamW, lr0 1e-3, lrf 0.01, warmup 3, close_mosaic 10,
# seed 0, imgsz 640) is the train_detector.py default and therefore identical.
#
# --stem v11n is load-bearing on stages 1 and 2: those continue via --weights,
# and _stem_for(None) returns "v8n", which would name the run directories after
# the wrong architecture and overwrite the shipped model's results.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEV=mps
COMBINED=data/converted/thermal_combined/thermal_combined.yaml
LOG=results/v11n_ab.log

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_stage() {
  local name="$1"; shift
  local dir="results/detector/${name}"
  if [ -f "${dir}/weights/best.pt" ]; then
    say "SKIP ${name} -- best.pt already present"
    return 0
  fi
  say "START ${name}"
  "$PY" training/train_detector.py "$@" >>"$LOG" 2>&1 &
  local pid=$!
  # The watchdog needs the run directory to exist before it can read results.csv.
  ( sleep 120
    [ -d "$dir" ] && "$PY" training/watch_divergence.py "$dir" --pid "$pid" \
        >>results/divergence_watch.log 2>&1 ) &
  wait "$pid"
  say "DONE ${name}"
}

say "=== YOLO11n A/B begins; replicating the 3-stage v8n recipe ==="

run_stage v11n_p3_hituav_640 \
  --variant p3 --model yolo11n.pt \
  --data data/converted/hituav/hituav.yaml \
  --tag hituav --epochs 30 --batch 16 --patience 10 --device "$DEV"

run_stage v11n_p3_thermalmix_640 \
  --variant p3 --weights results/detector/v11n_p3_hituav_640/weights/best.pt \
  --stem v11n --data "$COMBINED" \
  --tag thermalmix --epochs 20 --batch 12 --patience 8 --device "$DEV"

run_stage v11n_p3_thermalmix2_640 \
  --variant p3 --weights results/detector/v11n_p3_thermalmix_640/weights/best.pt \
  --stem v11n --data "$COMBINED" \
  --tag thermalmix2 --epochs 25 --batch 12 --patience 10 --device "$DEV"

say "=== training complete; scoring per source ==="
"$PY" training/eval_per_source.py \
  results/detector/v11n_p3_thermalmix2_640/weights/best.pt \
  --device "$DEV" >>"$LOG" 2>&1

say "=== A/B RESULT ==="
"$PY" - <<'PY' | tee -a "$LOG"
import json, pathlib
v8 = json.load(open("results/detector/v8n_p3_thermalmix2_640/per_source.json"))
p = pathlib.Path("results/detector/v11n_p3_thermalmix2_640/per_source.json")
if not p.exists():
    raise SystemExit("per_source.json missing for v11n -- eval did not complete")
v11 = json.load(open(p))
print(f"{'':22}{'HIT-UAV':>10}{'RGBT':>10}{'mean':>10}")
for name, d in (("v8n  (shipped)", v8), ("v11n (candidate)", v11)):
    h = d["per_source"]["hituav"]["person_AP50"]
    r = d["per_source"]["rgbt_thermal"]["person_AP50"]
    print(f"{name:22}{h:10.4f}{r:10.4f}{d['person_AP50_mean']:10.4f}")
d = v11["person_AP50_mean"] - v8["person_AP50_mean"]
print(f"\nmean Person AP50 delta: {d:+.4f}")
print("ADOPT 11n" if d > 0 else "KEEP v8n -- 11n does not beat the shipped model")
PY
