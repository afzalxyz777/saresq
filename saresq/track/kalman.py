"""Minimal constant-velocity Kalman filter for a track's residual (own) motion.

State is (u, v, du, dv) in image pixels. Ego-motion compensation (Section 9.2)
is applied to the position separately, before predict(); this filter only
has to model the target's own motion, which for a survivor is zero.
"""
from __future__ import annotations

import numpy as np


class ConstantVelocityKF:
    def __init__(self, u: float, v: float, process_var: float = 4.0, meas_var: float = 9.0):
        self.x = np.array([u, v, 0.0, 0.0])  # u, v, du, dv
        self.P = np.eye(4) * 100.0
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, u: float, v: float) -> None:
        z = np.array([u, v])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def nudge_position(self, du: float, dv: float) -> None:
        """Apply the ego-motion compensation offset directly to the state's position."""
        self.x[0] += du
        self.x[1] += dv

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])
