"""Degrade a public high-resolution thermal image to look like the MLX90640
(Section 11.1), so RGBTDronePerson can pretrain the fusion head.

RGBTDronePerson's thermal camera covers a different field of view than
ours at an altitude we don't know per-image, so -- per the spec -- the
match is by ground sample distance (a person ends up 1-3 pixels wide),
not by a fixed output resolution. The noise level is an explicit estimate
pending real side-by-side MLX90640 vs. this dataset's sensor comparison;
it is NOT a measured NETD-to-contrast ratio.
"""
from __future__ import annotations

import cv2
import numpy as np

# NETD (0.1 K RMS, Section 2.2) as a fraction of a typical scene's
# temperature spread (~2-4 K skin-to-background at 20 m, Section 5.1).
# This ratio is an ESTIMATE, not measured -- flagged for calibration
# against real KOL-SAR frames per Section 11.1's own caution.
NETD_RELATIVE_STD_ESTIMATE = 0.04
TARGET_PERSON_PX = 2.0  # Section 11.1: "people of about one to three pixels"


def scale_for_target_person_size(box_w_px: float, box_h_px: float, target_px: float = TARGET_PERSON_PX) -> float:
    """Downsample factor so a person of this box size becomes ~target_px wide."""
    avg_box_px = (box_w_px + box_h_px) / 2
    if avg_box_px <= 0:
        raise ValueError("box size must be positive")
    return min(1.0, target_px / avg_box_px)


def degrade_thermal(
    thermal_gray: np.ndarray,
    scale: float,
    netd_relative_std: float = NETD_RELATIVE_STD_ESTIMATE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Area-average downsample + Gaussian noise scaled to the degraded scene's contrast."""
    rng = rng or np.random.default_rng()
    h, w = thermal_gray.shape[:2]
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(thermal_gray.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    noise_std = netd_relative_std * max(float(small.std()), 1e-6)
    return small + rng.normal(0.0, noise_std, size=small.shape)


def degrade_box(box_xyxy: tuple[float, float, float, float], scale: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    return x1 * scale, y1 * scale, x2 * scale, y2 * scale
