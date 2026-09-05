"""Cross-modal (thermal-to-RGB) affine registration (Section 6).

Six-parameter affine fit by RANSAC least squares, replacing the reference
work's ratio-scaling (Section 6, intro): at 32 x 24 thermal resolution a
rounding translation from naive scaling costs a proportionally larger
fraction of a pixel than it does at higher resolution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Nominal RGB pixels per thermal pixel (Section 3.2); used only to express
# a reprojection residual in thermal-pixel units for the acceptance check.
NOMINAL_SCALE_RGB_PER_THERMAL_PX = 45.0


@dataclass
class Registration:
    matrix: np.ndarray  # 2x3 affine, thermal_px -> rgb_px
    residual_rgb_px: float
    inlier_mask: np.ndarray
    n_points: int

    @property
    def residual_thermal_px(self) -> float:
        return self.residual_rgb_px / NOMINAL_SCALE_RGB_PER_THERMAL_PX

    def reproject(self, thermal_pts: np.ndarray) -> np.ndarray:
        return apply_affine(self.matrix, thermal_pts)

    def save(self, path: str | Path, plate_serial: str = "", date: str = "") -> None:
        payload = {
            "matrix": self.matrix.tolist(),
            "residual_rgb_px": self.residual_rgb_px,
            "residual_thermal_px": self.residual_thermal_px,
            "n_points": self.n_points,
            "plate_serial": plate_serial,
            "date": date,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Registration":
        payload = json.loads(Path(path).read_text())
        return cls(
            matrix=np.array(payload["matrix"], dtype=float),
            residual_rgb_px=payload["residual_rgb_px"],
            inlier_mask=np.array([]),
            n_points=payload["n_points"],
        )


def apply_affine(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N, 2) -> (N, 2) via the 2x3 affine matrix."""
    pts = np.asarray(pts, dtype=float)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (matrix @ homogeneous.T).T


def fit_affine(
    thermal_pts: np.ndarray,
    rgb_pts: np.ndarray,
    ransac_thresh_px: float = 3.0,
) -> Registration:
    """Fit the six-parameter affine thermal_px -> rgb_px (Section 6.2).

    Requires at least 3 correspondences; the procedure recommends 12+
    spread across the shared field of view, including corners.
    """
    thermal_pts = np.asarray(thermal_pts, dtype=np.float32)
    rgb_pts = np.asarray(rgb_pts, dtype=np.float32)
    if len(thermal_pts) < 3:
        raise ValueError("need at least 3 correspondences to fit an affine transform")

    matrix, inliers = cv2.estimateAffine2D(
        thermal_pts, rgb_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px
    )
    if matrix is None:
        raise RuntimeError("affine estimation failed")
    inlier_mask = inliers.ravel().astype(bool)

    reproj = apply_affine(matrix, thermal_pts[inlier_mask])
    residual = np.sqrt(np.mean(np.sum((reproj - rgb_pts[inlier_mask]) ** 2, axis=1)))

    return Registration(
        matrix=matrix,
        residual_rgb_px=float(residual),
        inlier_mask=inlier_mask,
        n_points=len(thermal_pts),
    )
