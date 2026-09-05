# Running detector training on Colab

You chose to run Colab training yourself. All three dataset converters are
now written and verified against the real downloads (not guessed) — this
is the exact, ready-to-paste handoff.

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

**Cell 3 — fetch and convert HIT-UAV (verified structure, ~2 min on Colab's network)**
```python
!mkdir -p data/raw
!curl -L -o data/raw/hituav.zip "https://zenodo.org/api/records/7633134/files/suojiashun/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip/content"
!cd data/raw && unzip -q hituav.zip -d hituav
!python training/convert_hituav.py \
    --root "$(find data/raw/hituav -maxdepth 1 -type d -name 'suojiashun-*')/normal_xml" \
    --out data/converted/hituav
# expect: {'train': 2029, 'val': 290, 'test': 579}
```
Look at a few of `data/converted/hituav/verify/*.jpg` before trusting anything downstream (drag them into the Colab file browser).

**Cell 4 — train both detector variants on HIT-UAV (thermal)**
```python
!python training/train_detector.py --variant p3 --data data/converted/hituav/hituav.yaml --epochs 100 \
    --project /content/drive/MyDrive/saresq_runs
!python training/train_detector.py --variant p2 --data data/converted/hituav/hituav.yaml --epochs 100 \
    --project /content/drive/MyDrive/saresq_runs
```
Expect roughly 40–90 s/epoch at batch 16, 640px on a T4 — so each run is 1–2.5 hours. `save_period=1` means a session timeout only costs the last epoch.

**Cell 5 — pixels-on-target curve (task 0.5)**
```python
!python training/eval_pixels_on_target.py \
    --weights-p3 /content/drive/MyDrive/saresq_runs/v8n_p3_hituav_640/weights/best.pt \
    --weights-p2 /content/drive/MyDrive/saresq_runs/v8n_p2_hituav_640/weights/best.pt \
    --data data/converted/hituav/hituav.yaml --split val
```

**Cell 6 — second training stage: fine-tune on RGBTDronePerson's RGB images (Section 7.6)**
```python
!pip install gdown --quiet
!mkdir -p data/raw/rgbtdroneperson
!gdown --folder "https://drive.google.com/drive/folders/1Mi3NXQ-YG1iiIWkPbe3GQoDK68dARMN6" -O data/raw/rgbtdroneperson
!python training/convert_rgbt.py \
    --zip data/raw/rgbtdroneperson/RGBTDronePerson/RGBTDronePerson.zip \
    --out data/converted/rgbt
# expect: {'train': 4900, 'val': 1225}

!python training/train_detector.py --variant p2 --data data/converted/rgbt/rgbt.yaml --epochs 50 \
    --project /content/drive/MyDrive/saresq_runs
```

**Cell 7 — export INT8 TFLite (task 0.8, once training finishes)**
```python
!python training/export_tflite.py \
    --weights /content/drive/MyDrive/saresq_runs/v8n_p2_hituav_640/weights/best.pt \
    --data data/converted/hituav/hituav.yaml --sizes 160 224 640
```

**Cell 8 — hazard classifier on AIDER (task 0.12, independent of the above — can run first if you want a quick win)**
```python
!mkdir -p data/raw/aider
!curl -L -o data/raw/aider.zip "https://zenodo.org/api/records/3888300/files/AIDER.zip/content"
!cd data/raw/aider && unzip -q ../aider.zip
!python training/train_hazard.py --data-dir data/raw/aider/AIDER
```

## 3. Hand results back

Download (or share the Drive link to) each run's `results.csv`, `weights/best.pt`, and the exported `.tflite` files, plus `results/pixels_on_target.csv` and `results/hazard_classifier_report.md`. Send me those and I'll fold them into `results/detector_baseline.md` / `results/p2_vs_p3.md` and write up the final comparison.

## Timing note

Full 100-epoch runs for both variants will likely not finish before Sept 20 if you start them late — if time is short, cut `--epochs` to something that finishes today's session (even 20-30 epochs is enough to show the pipeline works end-to-end and produce a real, if under-trained, mAP number for the deck; label it as "N-epoch checkpoint, not converged" rather than a final result). The hazard classifier (Cell 8) is much faster (MobileNetV2, small dataset) and is a good first thing to kick off.
