"""Sensing model (Section 15.3): probability a pass produces thermal/RGB evidence.

Every numeric constant here is explicitly a placeholder "until measured" per
the spec text — replace FILL_FRACTION, PIXELS_ON_TARGET and R_CURVE with the
real pixels-on-target curve (Section 7.7) once it exists, and the whole
point of the sensitivity sweep in run.py is to check whether the policy
ranking survives being wrong about these numbers.
"""
from __future__ import annotations

import numpy as np

# Section 3.3, fill-fraction column: altitude (m) -> fraction of a thermal
# pixel a standing person's footprint covers.
FILL_FRACTION_ALTS = np.array([10.0, 20.0, 30.0])
FILL_FRACTION_VALS = np.array([1.0, 0.73, 0.32])

# Section 3.3, RGB-at-1640-width standing-width column: altitude (m) -> px.
PIXELS_ALTS = np.array([10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0])
PIXELS_VALS = np.array([68.0, 45.0, 34.0, 23.0, 17.0, 11.0, 7.0])

# Section 15.3 recall-vs-pixels-on-target placeholder curve.
R_CURVE_PIXELS = np.array([7.0, 13.0, 30.0])
R_CURVE_VALS = np.array([0.4, 0.7, 0.9])

CONCEALMENT_G_THERMAL = {"open": 1.0, "partial": 0.7, "heavy": 0.3}
CONCEALMENT_G_RGB = {"open": 1.0, "partial": 0.5, "heavy": 0.1}
AMBIENT_K = {"below30": 0.95, "above30": 0.6}

P_BLOB_DISTRACTOR = 0.8
P_RGB_DISTRACTOR = 0.05

FUSION_MEANS = {"both": 0.85, "thermal_only": 0.45, "rgb_only": 0.2}
FUSION_KAPPA = 20.0  # Beta concentration; higher = tighter around the mean


def fill_fraction(altitude_m: float, table: tuple = (FILL_FRACTION_ALTS, FILL_FRACTION_VALS)) -> float:
    alts, vals = table
    return float(np.interp(altitude_m, alts, vals))


def pixels_on_target(altitude_m: float) -> float:
    return float(np.interp(altitude_m, PIXELS_ALTS, PIXELS_VALS))


def recall_from_pixels(px: float, curve: tuple = (R_CURVE_PIXELS, R_CURVE_VALS)) -> float:
    pts, vals = curve
    return float(np.interp(px, pts, vals))


def p_blob(altitude_m: float, concealment: str, ambient_band: str, g_heavy_scale: float = 1.0) -> float:
    g = dict(CONCEALMENT_G_THERMAL)
    g["heavy"] *= g_heavy_scale
    return fill_fraction(altitude_m) * g[concealment] * AMBIENT_K[ambient_band]


def p_rgb(altitude_m: float, concealment: str) -> float:
    return recall_from_pixels(pixels_on_target(altitude_m)) * CONCEALMENT_G_RGB[concealment]


def draw_p_k(rng: np.random.Generator, pattern: str, kappa: float = FUSION_KAPPA) -> float | None:
    if pattern == "none":
        return None
    mean = FUSION_MEANS[pattern]
    alpha, beta = mean * kappa, (1 - mean) * kappa
    return float(rng.beta(alpha, beta))


def sense_pass(
    rng: np.random.Generator,
    altitude_m: float,
    concealment: str | None,
    ambient_band: str,
    is_distractor: bool = False,
    ambient_k_scale: float = 1.0,
    g_heavy_scale: float = 1.0,
    rgb_fp_scale: float = 1.0,
) -> tuple[str, float | None]:
    """One sensing pass over one target; returns (evidence_pattern, p_k)."""
    if is_distractor:
        pb = P_BLOB_DISTRACTOR
        pr = P_RGB_DISTRACTOR * rgb_fp_scale
    else:
        k = dict(AMBIENT_K)
        k[ambient_band] *= ambient_k_scale
        pb = fill_fraction(altitude_m) * dict(CONCEALMENT_G_THERMAL, heavy=CONCEALMENT_G_THERMAL["heavy"] * g_heavy_scale)[concealment] * k[ambient_band]
        pr = p_rgb(altitude_m, concealment)

    thermal = rng.random() < pb
    rgb = rng.random() < pr
    if thermal and rgb:
        pattern = "both"
    elif thermal:
        pattern = "thermal_only"
    elif rgb:
        pattern = "rgb_only"
    else:
        pattern = "none"
    return pattern, draw_p_k(rng, pattern)
