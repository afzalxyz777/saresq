"""Contrast-relative thermal gating (Master Spec v3.0, Section 5.2).

Pure functions on NumPy arrays. No I/O, no hardware. This is what makes the
gate scale-free enough to run identically on the MLX90640's radiometric
degrees Celsius and on 8-bit public thermal datasets (Section 11.1).
"""
from __future__ import annotations

import numpy as np


def scene_background(frame: np.ndarray) -> float:
    """Robust scene background estimate: the frame median.

    Correction C10: this is NOT the sensor's T_a (die temperature) register.
    """
    return float(np.median(frame))


def robust_sigma(frame: np.ndarray, t_bg: float, sigma_min_k: float = 0.15) -> float:
    """Robust spread via median absolute deviation, floored at sigma_min_k."""
    mad = np.median(np.abs(frame - t_bg))
    return max(1.4826 * float(mad), sigma_min_k)


def compute_contrast(frame: np.ndarray, sigma_min_k: float = 0.15):
    """Return (t_bg, sigma, z, dT) for a thermal frame in degrees Celsius.

    z is the two-sided, unit-free contrast (Section 5.2); dT is the signed
    contrast in kelvin.
    """
    t_bg = scene_background(frame)
    sigma = robust_sigma(frame, t_bg, sigma_min_k)
    dT = frame - t_bg
    z = dT / sigma
    return t_bg, sigma, z, dT


def gate_mask(
    z: np.ndarray,
    dT: np.ndarray,
    z_t: float = 2.5,
    dt_min_k: float = 0.5,
    two_sided: bool = True,
) -> np.ndarray:
    """Candidate pixel mask.

    Regime 3 (background above skin temperature) needs the two-sided test;
    a one-sided "warmer than" test misses a person who is cooler than a
    sun-heated surface. dt_min_k stops the gate firing on pure sensor noise
    when sigma collapses to its floor in a perfectly flat scene.
    """
    if two_sided:
        z_ok = np.abs(z) >= z_t
    else:
        z_ok = z >= z_t
    dt_ok = np.abs(dT) >= dt_min_k
    return z_ok & dt_ok
