"""Blob extraction and features from the thermal gate (Section 5.3).

Positive (warmer) and negative (cooler) candidate masks are labelled
separately so a warm person next to a cool puddle is not merged into one
blob.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_STRUCT8 = np.ones((3, 3), dtype=bool)  # 8-connectivity


@dataclass
class Blob:
    area_t: int
    z_peak: float
    z_mean: float
    dT_peak: float  # signed kelvin, at the pixel of peak |z| (Section 10.3 feature 5)
    sign: int  # +1 warmer, -1 cooler
    ecc: float
    bbox_t: tuple[int, int, int, int]  # (u_min, v_min, u_max, v_max), inclusive, expanded by 1px

    @property
    def rank_score(self) -> float:
        """Section 5.3: favours strong contrast, discounts single-pixel spikes."""
        return self.z_peak * min(self.area_t, 4)


def _eccentricity(rows: np.ndarray, cols: np.ndarray) -> float:
    """Eccentricity from second central moments (u=cols, v=rows)."""
    if rows.size < 2:
        return 0.0
    uc, vc = cols.mean(), rows.mean()
    du, dv = cols - uc, rows - vc
    mu20 = float(np.mean(du * du))
    mu02 = float(np.mean(dv * dv))
    mu11 = float(np.mean(du * dv))
    common = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    lam1 = (mu20 + mu02 + common) / 2
    lam2 = (mu20 + mu02 - common) / 2
    if lam1 <= 1e-12:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - lam2 / lam1)))


def _blobs_from_mask(
    mask: np.ndarray, z: np.ndarray, dT: np.ndarray, sign: int, frame_shape: tuple[int, int]
) -> list[Blob]:
    labels, n = ndimage.label(mask, structure=_STRUCT8)
    height, width = frame_shape
    out: list[Blob] = []
    for label_id in range(1, n + 1):
        rows, cols = np.nonzero(labels == label_id)
        abs_z = np.abs(z[rows, cols])
        peak_idx = int(np.argmax(abs_z))
        u_min, u_max = int(cols.min()) - 1, int(cols.max()) + 1
        v_min, v_max = int(rows.min()) - 1, int(rows.max()) + 1
        bbox = (
            max(u_min, 0), max(v_min, 0),
            min(u_max, width - 1), min(v_max, height - 1),
        )
        out.append(Blob(
            area_t=int(rows.size),
            z_peak=float(abs_z.max()),
            z_mean=float(abs_z.mean()),
            dT_peak=float(dT[rows[peak_idx], cols[peak_idx]]),
            sign=sign,
            ecc=_eccentricity(rows, cols),
            bbox_t=bbox,
        ))
    return out


def extract_blobs(z: np.ndarray, dT: np.ndarray, mask: np.ndarray) -> list[Blob]:
    """Connected components over the candidate mask, split by sign (Section 5.3)."""
    pos_mask = mask & (dT > 0)
    neg_mask = mask & (dT < 0)
    blobs = _blobs_from_mask(pos_mask, z, dT, +1, z.shape)
    blobs += _blobs_from_mask(neg_mask, z, dT, -1, z.shape)
    return blobs
