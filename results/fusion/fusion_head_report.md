# Fusion head (logistic regression, 18 features)

Source: `fusion_dataset.npz` | grouped split by source image | 8330 train / 2814 test candidates

- ROC AUC **0.670**, average precision **0.112**
- Brier **0.0243** vs. base-rate baseline 0.0247
- Expected calibration error **0.0058**
- Ledger prior pi_0 = **0.0334** (was a 0.20 placeholder in pipeline.yaml)

## Weights (raw feature scale)

| feature | weight |
|---|---|
| `p_collapse` | -2.1241 |
| `p_flood` | -1.9010 |
| `ecc` | -1.2577 |
| `lum` | -0.8132 |
| `p_fire` | +0.6331 |
| `z_peak` | -0.3509 |
| `sign` | +0.0831 |
| `T_bg` | +0.0113 |
| `dT` | +0.0003 |
| `area_t` | -0.0001 |
| `p_rgb` | +0.0000 |
| `has_rgb` | +0.0000 |
| `a_rgb` | +0.0000 |
| `iou` | +0.0000 |
| `d_c` | +0.0000 |
| `hits` | +0.0000 |
| `hit_ratio` | +0.0000 |
| `alt_band` | +0.0000 |

## Calibration

| bin | n | predicted | observed |
|---|---|---|---|
| 0.0-0.1 | 2660 | 0.025 | 0.021 |
| 0.1-0.2 | 154 | 0.133 | 0.104 |

Zeroed, not fitted: alt_band, hit_ratio, hits — Not measurable from a still-image dataset; calibrate from operator verdicts (training/export_verdicts.py) once flight data exists.
