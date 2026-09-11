#!/usr/bin/env bash
# Run the whole ML pipeline end to end, in dependency order.
#
#   bash training/run_all.sh              # resume: skip steps whose output exists
#   FORCE=1 bash training/run_all.sh      # redo everything
#   EPOCHS=15 bash training/run_all.sh    # short run, to prove the wiring
#
# Every step is idempotent and checks for its own output, because on this
# machine the full sequence takes most of a night and a laptop that sleeps,
# runs out of battery or gets closed must not cost the whole run. Re-running is
# always safe.
#
# ONE ENVIRONMENT: .venv-tf (Python 3.12). TensorFlow publishes no macOS-arm64
# wheel for Python 3.13, which is what .venv runs, and the TFLite export needs
# TensorFlow. Rather than hand artefacts between two interpreters mid-pipeline,
# the whole ML chain runs in the 3.12 environment. .venv stays as the
# application/test environment and is not used here.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./.venv-tf/bin/python
EPOCHS="${EPOCHS:-30}"
FT_EPOCHS="${FT_EPOCHS:-25}"
BATCH="${BATCH:-16}"
# 8 GB of unified memory is the binding constraint, not the GPU. Six dataloader
# workers thrashed it (free memory to 8%, 20 s/iter); four leaves headroom.
WORKERS="${WORKERS:-4}"
# Stop when the run has genuinely plateaued rather than burning hours on the
# tail. The two detector runs must use the SAME value or the P2-vs-P3
# comparison stops being like-for-like.
PATIENCE="${PATIENCE:-10}"
# The RGB fine-tune only needs domain adaptation from an already-strong
# detector, so it does not need all 4900 images or many epochs.
FT_FRACTION="${FT_FRACTION:-1.0}"
FORCE="${FORCE:-0}"

HITUAV=data/converted/hituav/hituav.yaml
RGBT=data/converted/rgbt/rgbt.yaml
DET=results/detector
mkdir -p "$DET" results/fusion results/hazard models

[ -x "$PY" ] || { echo "FATAL: $PY missing. Create it with:"; \
  echo "  /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv-tf"; exit 1; }

step() {  # step <output-to-check> <label> <command...>
  local out="$1" label="$2"; shift 2
  if [ "$FORCE" != "1" ] && [ -e "$out" ]; then
    echo "== SKIP  $label (found $out)"; return 0
  fi
  echo "== START $label  $(date '+%H:%M:%S')"
  local t0; t0=$(date +%s)
  "$@"
  echo "== DONE  $label in $(( ($(date +%s) - t0) / 60 )) min"
}

# 1-2. The two detector heads. Identical in everything but the head, which is
#      the whole point: the difference between them IS the measurement.
step "$DET/v8n_p3_hituav_640/weights/best.pt" "detector P3 baseline" \
  $PY training/train_detector.py --variant p3 --data $HITUAV \
    --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" --cache disk --patience "$PATIENCE"

# P2 needs its own batch and AMP setting, and neither is a preference:
#   * 34,000 anchors vs P3's 8,400 pushed GPU memory to 8.15 GB on an 8 GB
#     machine -> swap thrashing -> 466 s/iter. batch 6 holds it at 3.2 GB.
#   * fp16 overflowed the assigner: cls_loss hit -1224 by iteration 6. AMP off.
# The comparison with P3 stays fair because Ultralytics accumulates gradients
# to a nominal batch of 64: batch 16 -> effective 64, batch 6 -> effective 66.
P2_BATCH="${P2_BATCH:-6}"
step "$DET/v8n_p2_hituav_640/weights/best.pt" "detector P2 small-object head" \
  $PY training/train_detector.py --variant p2 --data $HITUAV \
    --epochs "$EPOCHS" --batch "$P2_BATCH" --workers "$WORKERS" --cache disk \
    --patience "$PATIENCE" --amp False

# 3. Recall vs. object size, for both heads. Feeds the sweep width in step 8.
step "$DET/pixels_on_target.csv" "pixels-on-target curve" \
  $PY training/eval_pixels_on_target.py \
    --weights-p3 "$DET/v8n_p3_hituav_640/weights/best.pt" \
    --weights-p2 "$DET/v8n_p2_hituav_640/weights/best.pt" --data $HITUAV

# 4. RGB-domain fine-tune. The HIT-UAV model is a THERMAL detector; Pass B of
#    the fusion vector scores visible-light crops. Running the thermal model on
#    RGB would produce a p_rgb feature that means nothing.
# Fine-tuned from P3, not P2: P3 won the comparison outright (mAP50 0.639 vs
# 0.588, Person AP50 0.821 vs 0.801) at HALF the GFLOPs, and beat P2 in every
# object-size bin. P2 is retained only as the measured negative result.
#
# FT_EPOCHS was 8 on the first run and produced mAP50 0.0054 -- a dead model.
# Going 4 classes -> 1 REINITIALISES the detection head, so the fine-tune is
# learning detection from scratch on 9x15 px people, not nudging a trained head.
# It needs real epochs on the full dataset.
step "$DET/v8n_p3_rgbt_640/weights/best.pt" "detector RGB fine-tune (RGBT visible)" \
  $PY training/train_detector.py --variant p3 --data $RGBT --tag rgbt \
    --weights "$DET/v8n_p3_hituav_640/weights/best.pt" \
    --epochs "$FT_EPOCHS" --batch "$BATCH" --workers "$WORKERS" --cache False \
    --patience "$PATIENCE" --fraction "$FT_FRACTION"
# NOTE: cache=False here on purpose. RGBTDronePerson is 6125 images, and an
# .npy disk cache of it costs ~6 GB. This machine does not have the headroom to
# spend that on an 8-epoch fine-tune; JPEG decode is the cheaper resource here.

# 5. INT8 TFLite at each crop size, verified against the deployment runtime.
step models/yolov8n_p3_160_w8a32.tflite "TFLite w8a32 export" \
  $PY training/export_tflite.py \
    --weights "$DET/v8n_p3_hituav_640/weights/best.pt" --data $HITUAV --sizes 160 224 640

# 6. Scene hazard classifier -> three of the eighteen fusion features.
step models/mobilenetv2_aider_224_int8.tflite "hazard classifier (AIDER)" \
  $PY training/train_hazard.py

# 7. The fusion dataset. Built from the RGB detector's VALIDATION split so the
#    p_rgb feature is not measured on images the detector memorised.
step results/fusion/fusion_dataset.npz "fusion dataset (RGBT val)" \
  $PY training/make_fusion_dataset.py \
    --weights "$DET/v8n_p3_rgbt_640/weights/best.pt" \
    --hazard models/mobilenetv2_aider_224_int8.tflite --split val

step models/fusion_logreg.json "fusion head" \
  $PY training/train_fusion.py

# 8. HELD BACK ON PURPOSE -- do not re-enable without reading this.
#
# W is the constant the entire search model rests on, and the only gate-recall
# figure available to feed it is measured on RGBTDronePerson, which is OUT OF
# DOMAIN for a contrast-relative gate. Measured 2026-09-10: the gate fires
# correctly on people (z = +2.7..+4.8 at every ground-truth location, well over
# the 2.5 threshold), but 17.5% of an RGBT frame passes the gate, so each
# person's pixels merge by 8-connectivity into large blobs spanning hot roofs
# and roads. Per-person recall lands at ~13%.
#
# That ~13% is NOT the gate's recall over rubble. RGBT frames are wide urban
# scenes (grey-level std 47.6); the MLX90640 sees a 20.8 m swath of comparatively
# uniform debris, which is the regime the gate was designed for. Four fixes were
# tried and none moved it: max_blob_px filtering, rank/top-N selection, and
# tiling the degraded frame to the sensor's true 32x24 footprint (which made it
# slightly worse, 11.6%).
#
# Feeding 13% into W would understate coverage by roughly 5x and put a badly
# wrong number on a slide. So this step is opt-in until gate recall is measured
# on representative imagery -- ideally the Pi's own MLX90640 captures over
# rubble, which is the real answer.
#
#   RUN_SWEEP_WIDTH=1 bash training/run_all.sh
if [ "${RUN_SWEEP_WIDTH:-0}" = "1" ]; then
  step "$DET/sweep_width.json" "sweep width W" \
    $PY training/eval_sweep_width.py --variant p3 \
      --gate-recall-json results/fusion/fusion_dataset.json
else
  echo "== HELD  sweep width W (gate recall not yet measured in-domain; see comment)"
fi

echo
echo "===================== ARTEFACTS ====================="
ls -1 models/*.tflite models/*.json 2>/dev/null || true
ls -1 "$DET"/*.json "$DET"/*.csv results/fusion/* results/hazard/* 2>/dev/null || true
