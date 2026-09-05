"""Geo-tagging tests (Section 12): projection geometry and the attitude filter."""
import math

import numpy as np

from saresq.geo.attitude import ComplementaryFilter, accel_to_roll_pitch
from saresq.geo.project import pixel_to_latlon

FOCAL_PX = 1359.3
PRINCIPAL_POINT = (820.0, 616.0)
ALT_M = 20.0
LAT0, LON0 = 22.5726, 88.3639  # Kolkata


def test_image_centre_at_zero_attitude_projects_to_the_fix():
    lat, lon = pixel_to_latlon(
        *PRINCIPAL_POINT, PRINCIPAL_POINT, FOCAL_PX, 0.0, 0.0, 0.0, ALT_M, LAT0, LON0
    )
    assert math.isclose(lat, LAT0, abs_tol=1e-9)
    assert math.isclose(lon, LON0, abs_tol=1e-9)


def test_off_centre_pixel_at_zero_attitude_offsets_by_pinhole_geometry():
    u = PRINCIPAL_POINT[0] + FOCAL_PX  # one focal-length right of centre -> 45 degrees off-axis
    lat, lon = pixel_to_latlon(
        u, PRINCIPAL_POINT[1], PRINCIPAL_POINT, FOCAL_PX, 0.0, 0.0, 0.0, ALT_M, LAT0, LON0
    )
    expected_east_m = ALT_M  # tan(45 deg) * altitude
    expected_lon = LON0 + expected_east_m / (111_320.0 * math.cos(math.radians(LAT0)))
    assert math.isclose(lon, expected_lon, rel_tol=1e-6)
    assert math.isclose(lat, LAT0, abs_tol=1e-9)  # pure across-track offset, no north component


def test_yaw_rotates_the_ground_offset_without_changing_its_magnitude():
    u = PRINCIPAL_POINT[0] + 200.0
    lat_a, lon_a = pixel_to_latlon(u, PRINCIPAL_POINT[1], PRINCIPAL_POINT, FOCAL_PX, 0, 0, 0, ALT_M, LAT0, LON0)
    lat_b, lon_b = pixel_to_latlon(u, PRINCIPAL_POINT[1], PRINCIPAL_POINT, FOCAL_PX, 0, 0, math.pi / 2, ALT_M, LAT0, LON0)

    dist_a = math.hypot((lat_a - LAT0) * 111_320.0, (lon_a - LON0) * 111_320.0 * math.cos(math.radians(LAT0)))
    dist_b = math.hypot((lat_b - LAT0) * 111_320.0, (lon_b - LON0) * 111_320.0 * math.cos(math.radians(LAT0)))
    assert math.isclose(dist_a, dist_b, rel_tol=1e-6)
    assert not math.isclose(lat_a, lat_b, abs_tol=1e-9)  # but the direction did change


def test_accel_to_roll_pitch_level_platform():
    roll, pitch = accel_to_roll_pitch(0.0, 0.0, 9.81)
    assert math.isclose(roll, 0.0, abs_tol=1e-9)
    assert math.isclose(pitch, 0.0, abs_tol=1e-9)


def test_complementary_filter_converges_to_accel_angle_when_stationary():
    cf = ComplementaryFilter(alpha=0.8)
    tilted_roll = math.radians(5.0)
    ax, ay, az = 0.0, 9.81 * math.sin(tilted_roll), 9.81 * math.cos(tilted_roll)
    for _ in range(200):
        roll, pitch = cf.update(roll_rate=0.0, pitch_rate=0.0, ax=ax, ay=ay, az=az, dt=0.02)
    assert math.isclose(roll, tilted_roll, abs_tol=1e-3)


def test_bias_calibration_removes_constant_gyro_drift():
    cf = ComplementaryFilter(alpha=0.995)
    bias_samples = np.full((200, 2), 0.01)  # constant 0.01 rad/s bias on both axes
    cf.calibrate_bias(bias_samples)
    for _ in range(500):
        roll, pitch = cf.update(roll_rate=0.01, pitch_rate=0.01, ax=0.0, ay=0.0, az=9.81, dt=0.02)
    # With the bias removed, a truly stationary level platform should not drift.
    assert abs(roll) < 0.01
    assert abs(pitch) < 0.01
