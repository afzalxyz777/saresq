"""Predict image motion from GNSS/IMU so the tracker sees residual (target-own)
motion only, not the aircraft's own motion (Section 9.2).

At 5 m/s and 20 m every pixel moves 34 px per 100 ms; at 14 m/s, 95 px
(Section 3.4). A constant-velocity tracker initialised from one detection
with zero velocity loses a stationary survivor on the very next frame
unless this compensation runs first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SimilarityTransform:
    """Maps a point in frame k-1 to where it should appear in frame k,
    assuming the ground is flat and the point is stationary."""
    theta: float  # rotation about the image centre, radians
    tx: float
    ty: float

    def apply(self, pts: np.ndarray, center: tuple[float, float]) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        cu, cv = center
        c, s = math.cos(self.theta), math.sin(self.theta)
        rot = np.array([[c, -s], [s, c]])
        centered = pts - np.array([cu, cv])
        rotated = centered @ rot.T
        shifted = rotated + np.array([cu, cv]) + np.array([self.tx, self.ty])
        return shifted


def predict_transform(
    v_mps: float,
    course_rad: float,
    yaw_rad: float,
    roll_rate: float,
    pitch_rate: float,
    yaw_rate: float,
    dt_s: float,
    altitude_m: float,
    focal_px: float,
) -> SimilarityTransform:
    """Section 9.2 similarity transform from sensors already carried onboard.

    v_mps, course_rad: GNSS ground speed and course over ground.
    yaw_rad: camera yaw relative to the aircraft (fixed by mounting).
    roll_rate, pitch_rate, yaw_rate: gyroscope rates, rad/s (body axes p, q, r).
    """
    forward_mag = focal_px * v_mps * dt_s / altitude_m
    heading_in_image = course_rad - yaw_rad
    tx = forward_mag * math.sin(heading_in_image) + focal_px * pitch_rate * dt_s
    ty = forward_mag * math.cos(heading_in_image) + focal_px * roll_rate * dt_s
    theta = yaw_rate * dt_s
    return SimilarityTransform(theta=theta, tx=tx, ty=ty)
