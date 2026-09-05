"""Attitude from the MPU-6050: a complementary filter (Section 12.3).

Adequate for roll/pitch to within a degree or two on a stable platform; not
for a manoeuvring aircraft, where the flight controller's own estimate is
used instead (December). Yaw is not observable from this sensor at all (no
magnetometer) -- heading comes from GNSS course-over-ground while moving.
"""
from __future__ import annotations

import math

import numpy as np


def accel_to_roll_pitch(ax: float, ay: float, az: float) -> tuple[float, float]:
    """Roll/pitch (radians) from the gravity direction in accelerometer readings."""
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay ** 2 + az ** 2))
    return roll, pitch


class ComplementaryFilter:
    """theta = alpha*(theta + omega*dt) + (1-alpha)*theta_accel (Section 12.3)."""

    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.gyro_bias = np.zeros(2)  # roll_rate, pitch_rate

    def calibrate_bias(self, stationary_gyro_samples: np.ndarray) -> None:
        """Average ~200 samples while stationary at start-up (Section 12.3)."""
        self.gyro_bias = np.asarray(stationary_gyro_samples).mean(axis=0)

    def update(self, roll_rate: float, pitch_rate: float, ax: float, ay: float, az: float, dt: float) -> tuple[float, float]:
        roll_rate -= self.gyro_bias[0]
        pitch_rate -= self.gyro_bias[1]
        accel_roll, accel_pitch = accel_to_roll_pitch(ax, ay, az)
        self.roll = self.alpha * (self.roll + roll_rate * dt) + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * (self.pitch + pitch_rate * dt) + (1 - self.alpha) * accel_pitch
        return self.roll, self.pitch
