"""The 18-number fusion feature vector, assembled from Passes A-E (Section 10.3).

Order matches Appendix B's fusion-head JSON export exactly, so a model
trained on vectors from this function can be deployed by loading that JSON
and doing nothing more than a dot product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    "p_rgb", "has_rgb", "a_rgb",
    "z_peak", "dT", "sign", "area_t", "ecc",
    "iou", "d_c",
    "hits", "hit_ratio",
    "T_bg", "lum", "p_flood", "p_fire", "p_collapse", "alt_band",
]


@dataclass
class RgbEvidence:
    """Pass B. None of these fields exist if the detector returned no box."""
    score: float
    box_area_px: float
    crop_area_px: float


@dataclass
class ThermalEvidence:
    """Pass A, from a single saresq.gate.blobs.Blob."""
    z_peak: float
    dT_peak: float
    sign: int
    area_t: int
    ecc: float


@dataclass
class RegistrationEvidence:
    """Pass C: agreement between the reprojected thermal box and the RGB box."""
    iou: float
    centroid_dist_norm: float


@dataclass
class TrackEvidence:
    """Pass D."""
    hits: int
    age: int

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.age if self.age > 0 else 0.0


@dataclass
class ContextEvidence:
    """Pass E."""
    t_bg: float
    lum: float
    p_flood: float
    p_fire: float
    p_collapse: float
    alt_band: int


def assemble_features(
    thermal: ThermalEvidence,
    rgb: RgbEvidence | None,
    registration: RegistrationEvidence | None,
    track: TrackEvidence,
    context: ContextEvidence,
) -> np.ndarray:
    """Returns the 18-length vector in FEATURE_NAMES order.

    rgb=None and registration=None (no box, no overlap to measure) are
    encoded as zeros plus the explicit has_rgb=0 / iou=0 indicators, per
    Section 10.3: "the head can learn that thermal-present-RGB-absent is a
    distinct pattern rather than a weak version of agreement."
    """
    p_rgb = rgb.score if rgb is not None else 0.0
    has_rgb = 1.0 if rgb is not None else 0.0
    a_rgb = (rgb.box_area_px / rgb.crop_area_px) if rgb is not None else 0.0

    iou = registration.iou if registration is not None else 0.0
    d_c = registration.centroid_dist_norm if registration is not None else 0.0

    return np.array([
        p_rgb, has_rgb, a_rgb,
        thermal.z_peak, thermal.dT_peak, float(thermal.sign), float(thermal.area_t), thermal.ecc,
        iou, d_c,
        float(track.hits), track.hit_ratio,
        context.t_bg, context.lum, context.p_flood, context.p_fire, context.p_collapse, float(context.alt_band),
    ], dtype=float)
