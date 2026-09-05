# Running detector training on Colab

You chose to run Colab training yourself. Here's the exact handoff.

## 1. Push this repo to GitHub (one-time, on your machine)

```bash
cd /Users/afzalamanullah/saresq
git remote add origin https://github.com/<your-username>/saresq.git   # create the repo on github.com first (private is fine)
git branch -M main
git push -u origin main
```

`data/raw/`, `data/converted/`, and `models/*.pt`/`*.tflite` are gitignored, so this push is small — code and config only.

## 2. Open a new Colab notebook, set Runtime -> T4 GPU, and run these cells in order

**Cell 1 — clone and install**
```python
!git clone https://github.com/<your-username>/saresq.git
%cd saresq
!pip install -e ".[training]" --quiet
```

**Cell 2 — mount Drive (so checkpoints survive a session timeout)**
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/saresq_runs
```

**Cell 3 — fetch HIT-UAV directly (fast on Colab's network; don't route through your laptop)**
```python
!mkdir -p data/raw
!curl -L -o data/raw/hituav.zip "https://zenodo.org/api/records/7633134/files/suojiashun/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip/content"
!cd data/raw && unzip -q hituav.zip -d hituav
!find data/raw/hituav -maxdepth 3 | head -50
```

**Cell 4 — inspect one annotation file before trusting any converter**

Run this and paste the output back to me (or just read it yourself) — the spec's own task 0.2 flags this as "where projects lose days," and I don't want to guess the label format wrong:
```python
import glob
xmls = glob.glob('data/raw/hituav/**/*.xml', recursive=True)
jsons = glob.glob('data/raw/hituav/**/*.json', recursive=True)
txts = glob.glob('data/raw/hituav/**/*.txt', recursive=True)
print("xml:", len(xmls), xmls[:2])
print("json:", len(jsons), jsons[:2])
print("txt:", len(txts), txts[:2])
if xmls:
    print(open(xmls[0]).read())
if txts:
    print(open(txts[0]).read())
```

**I'll write `training/convert_hituav.py` against whatever this actually shows** (I'm downloading the same zip locally in parallel to write and test it myself — this Colab step is the fallback/cross-check, not a blocker on you).

**Cell 5 — once `training/convert_hituav.py` exists, convert + sanity-check (task 0.2's acceptance criterion)**
```python
!python training/convert_hituav.py --raw data/raw/hituav --out data/converted/hituav
# renders 20 random images with boxes drawn -- look at them before trusting anything downstream
```

**Cell 6 — train both variants**
```python
!python training/train_detector.py --variant p3 --data data/hituav.yaml --epochs 100 \
    --project /content/drive/MyDrive/saresq_runs
!python training/train_detector.py --variant p2 --data data/hituav.yaml --epochs 100 \
    --project /content/drive/MyDrive/saresq_runs
```
Expect roughly 40–90 s/epoch at batch 16, 640px on a T4 — so each run is 1–2.5 hours. `save_period=1` means a session timeout only costs the last epoch.

**Cell 7 — export INT8 TFLite (once training finishes)**
```python
!python training/export_tflite.py \
    --weights /content/drive/MyDrive/saresq_runs/v8n_p2_hituav_640/weights/best.pt \
    --data data/hituav.yaml --sizes 160 224 640
```

## 3. Hand results back

Download (or share the Drive link to) each run's `results.csv`, `weights/best.pt`, and the exported `.tflite` files. Send me the metrics (or just the `results.csv`) and I'll fold them into `results/detector_baseline.md` / `results/p2_vs_p3.md` and continue with the pixels-on-target eval (task 0.5) and INT8 accuracy delta (task 0.8's acceptance criterion).

## Timing note

Full 100-epoch runs for both variants will likely not finish before Sept 20 if you start them late — if time is short, cut `--epochs` to something that finishes today's session (even 20-30 epochs is enough to show the pipeline works end-to-end and produce a real, if under-trained, mAP number for the deck; label it as "N-epoch checkpoint, not converged" rather than a final result).
