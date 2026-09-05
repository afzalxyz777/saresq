"""From a pixel to a latitude/longitude (Section 12.1).

Convention (not fully pinned down by the spec, so fixed here explicitly):
camera frame at zero attitude has +x (image u, right) aligned with East and
+y (image v, down) aligned with North, +z (boresight) pointing straight
down -- i.e. the identity rotation is a level nadir shot. Roll/pitch/yaw
compose as an extrinsic Rz(yaw) @ Ry(pitch) @ Rx(roll) applied to that ray.
GNSS dominates the error budget (Section 12.2) far more than this
convention choice does.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M_PER_DEG = 111_320.0


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def pixel_to_latlon(
    u: float, v: float,
    principal_point: tuple[float, float],
    focal_px: float,
    roll: float, pitch: float, yaw: float,
    altitude_m: float,
    lat0: float, lon0: float,
) -> tuple[float, float]:
    """Section 12.1. roll/pitch/yaw in radians; returns (lat, lon) in degrees."""
    cu, cv = principal_point
    d_c = np.array([(u - cu) / focal_px, (v - cv) / focal_px, 1.0])
    d_g = _rotation_matrix(roll, pitch, yaw) @ d_c
    if d_g[2] <= 0:
        raise ValueError("ray points away from the ground; check altitude/attitude inputs")
    s = altitude_m / d_g[2]
    east = s * d_g[0]
    north = s * d_g[1]
    lat = lat0 + north / EARTH_RADIUS_M_PER_DEG
    lon = lon0 + east / (EARTH_RADIUS_M_PER_DEG * math.cos(math.radians(lat0)))
    return lat, lon
