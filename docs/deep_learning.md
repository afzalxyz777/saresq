# The deep-learning pipeline

Everything from raw datasets to the four artefacts the payload actually loads.
One command runs the whole thing:

```bash
bash training/run_all.sh                 # resume; skips steps whose output exists
EPOCHS=15 bash training/run_all.sh       # short run, proves the wiring
FORCE=1 bash training/run_all.sh         # redo everything
```

## Two environments, and why

| venv | Python | Holds | Used for |
|---|---|---|---|
| `.venv` | 3.13 | numpy 2.5, flask, opencv, scipy | the app, the dashboard, `pytest` |
| `.venv-tf` | 3.12 | tensorflow 2.21, torch 2.13, ultralytics, LiteRT | **the entire ML pipeline** |

TensorFlow publishes no macOS-arm64 wheel for Python 3.13, and the INT8 TFLite
export needs TensorFlow. Rather than pin the whole project back a version for
one build step, the ML chain lives in its own 3.12 environment.

**The versions in `.venv-tf` are pinned as a set and should not be bumped
casually.** `torchvision` must match `torch` exactly or the Ultralytics
validator dies with `operator torchvision::nms does not exist`, and
`litert-torch` (the export backend) requires `torch < 2.14`. The only
combination that satisfies both is **torch 2.13.0 + torchvision 0.28.0**.
Installing anything into this venv can silently move `torch`; re-check after.

## What runs, in order

1. **`train_detector.py --variant p3`** — YOLOv8n baseline on HIT-UAV (thermal).
2. **`train_detector.py --variant p2`** — same, plus a stride-4 head.
3. **`eval_pixels_on_target.py`** — recall vs. object pixel height, both heads.
4. **`train_detector.py --data rgbt --weights <p2>`** — RGB-domain fine-tune.
5. **`export_tflite.py`** — INT8 TFLite at 160 / 224 / 640, each verified.
6. **`train_hazard.py`** — MobileNetV2 on AIDER, + its own INT8 export.
7. **`make_fusion_dataset.py`** → **`train_fusion.py`** — the 18-feature head.
8. **`eval_sweep_width.py`** — the Koopman sweep width *W*.

### Why step 4 exists

Steps 1–2 produce a **thermal** detector. Pass B of the fusion vector scores
**visible-light** crops. Running the thermal model on RGB would make `p_rgb`
meaningless, so the model is fine-tuned into the RGB domain on
RGBTDronePerson's visible images first.

### Why the P2 head is worth measuring

Pixel sizes this payload actually produces, at the 20 m survey altitude
(GSD = altitude / focal_px = 20 / 1359.3 = **14.7 mm/px**):

| path | prone adult | standing adult |
|---|---|---|
| thermal 32×24 (the gate) | 2.6 px | ~0.7 px |
| RGB 160 px crop | 115 × 31 px | 31 × 24 px |
| RGB 640 full-frame fallback | 45 × 12 px | 12 × 9 px |

So the small-object problem is **not** in the crop path — a 31 px object on a
stride-8 grid is four cells across. It is in the **full-frame fallback**, the
mode the pipeline enters when the gate has been starved for three consecutive
frames. There a standing person is 9–12 px, roughly one stride-8 cell, and P2's
stride-4 level is the only change that puts more than one cell on the target.
HIT-UAV is flown at 60–130 m and sits in the same regime, which is what makes
it the right dataset to measure the difference on.

An earlier version of this document claimed 12–18 px for the *crop* path. That
was wrong by 4–8×; the table above is derived from `configs/pipeline.yaml`.

## Facts measured from the real export, not assumed

Ultralytics' current LiteRT export path produces something different from what
its older `onnx2tf` path produced, and from what most guides describe. Measured
against the actual file:

- **Input layout is `NCHW` (1, 3, S, S)**, not NHWC.
- **Input/output dtype is `float32`.** INT8 is internal to the graph; the I/O
  boundary is float in [0, 1]. A 12.16 MiB model quantises to 3.30 MiB (3.7×).
- **Box regressions are normalised to [0, 1]**, not pixels.

None of these raise when assumed wrongly — a pixel/normalised mix-up scales
every box by the input size and still returns plausible geometry. So
`saresq/detect/tflite_detector.py` **probes** the box convention at load time
rather than trusting it, and `export_tflite.py` records the layout in
`results/detector/tflite_exports.json`.

The whole chain was validated end-to-end before training started, by exporting
stock COCO `yolov8n.pt` and running it through `CropDetector` on the Pi camera
captures in `results/camtest/`: it reports `person 0.50` on the exposed frames
and nothing on the black one, at 2.4 ms per 160 px inference.

## Honest limits of the fusion head

`make_fusion_dataset.py` builds its rows from **RGBTDronePerson's validation
split** — never the split the detector trained on, or `p_rgb` would be measured
on images the detector had memorised and the head would learn to over-trust it.

Three of the eighteen features cannot be measured from still images:
`hits` and `hit_ratio` are properties of a track across frames, `alt_band` of a
mission pass. `train_fusion.py` excludes them from the fit and writes explicit
zeros, rather than letting the optimiser fit noise on constant input. They get
calibrated from operator verdicts (`training/export_verdicts.py`) once real
flights produce them.

The head is fitted **without class balancing**, which is the opposite of the
reflex. Its output feeds `saresq/fuse/ledger.py`, which accumulates `logit(p)`
across mission passes, so a systematic bias does not average out — it compounds
once per pass. Calibration is the product; ranking alone is not enough. The
report includes a reliability table and expected calibration error.

`train_hazard.py` *does* use class weighting, because AIDER is 68 % `normal`
and its five outputs are consumed as *features* that the fusion head
recalibrates. The two scripts differ on purpose.

## The sweep width, and a result worth keeping

`W` is the most load-bearing constant in the search model: coverage `C = W·L/A`
and `POD = 1 − e^(−C)` both scale with it, and the value in use (14 m) was
inherited from the spec, never measured.

**W is not "how wide the camera sees."** It is the integral of the lateral
range curve. A sensor that sees 20 m wide but detects half of what is under it
has a sweep width of 10 m.

The instinct is that the curve should droop at the swath edges — longer slant
range, oblique view. **For a nadir pinhole camera over flat ground it does
not.** A ground point at lateral offset *x* images at `u = f·x/h`, so
`du/dx = f/h` is constant: ground sample distance is uniform across the entire
swath, and perspective foreshortening exactly cancels the increased range. The
curve is flat, and the only geometric falloff is edge truncation, where a
target is partly outside the frame. That is modelled; it is small but real.

Not modelled, and all of which push W down: lens vignetting, off-axis MTF loss,
thermal angular sensitivity, terrain relief, occlusion by rubble. Read the
output as an optimistic geometric bound, gated by measured recall.

## Timing on this hardware

The M1 8 GB runs YOLOv8n at 640 px in roughly **6–14 min/epoch** depending on
what else is running — call it 4–9 h for a 40-epoch variant, and the pipeline
trains three. A Colab T4 does the same work about 8–10× faster; the runbook for
that is still `docs/colab_training.md`, and `run_all.sh` is resumable precisely
so the local path survives a closed laptop.

Do not run anything else heavy on the machine while it trains. Two concurrent
jobs on 8 GB drove memory-free to 14 % and slowed the training from 2.1 s/iter
to 9 s/iter — the pipeline is memory-bound long before it is compute-bound.
