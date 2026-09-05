"""Task 0.7 acceptance test: recover a known affine from synthetic correspondences."""
import numpy as np

from saresq.calib.register import apply_affine, fit_affine

# A plausible ground-truth transform: nominal scale (Section 3.2) + a small
# rotation and offset from mounting tolerance.
TRUE_MATRIX = np.array([
    [45.0, -0.8, 30.0],
    [0.6, 37.0, 20.0],
])


def _synthetic_correspondences(n_inliers=24, n_outliers=6, noise_px=0.2, seed=0):
    rng = np.random.default_rng(seed)
    thermal_pts = rng.uniform([0, 0], [31, 23], size=(n_inliers, 2))
    rgb_clean = apply_affine(TRUE_MATRIX, thermal_pts)
    rgb_pts = rgb_clean + rng.normal(0, noise_px, size=rgb_clean.shape)

    # Outliers: correct thermal point, garbage RGB point (mismatched click, etc.)
    outlier_thermal = rng.uniform([0, 0], [31, 23], size=(n_outliers, 2))
    outlier_rgb = rng.uniform([0, 0], [1640, 1232], size=(n_outliers, 2))

    thermal_all = np.vstack([thermal_pts, outlier_thermal])
    rgb_all = np.vstack([rgb_pts, outlier_rgb])
    return thermal_all, rgb_all, n_inliers


def test_ransac_recovers_known_affine_and_rejects_outliers():
    thermal_pts, rgb_pts, n_inliers = _synthetic_correspondences(n_inliers=24, n_outliers=6)
    reg = fit_affine(thermal_pts, rgb_pts, ransac_thresh_px=5.0)

    assert np.allclose(reg.matrix, TRUE_MATRIX, atol=0.5)
    assert reg.residual_thermal_px < 0.05

    # 6 of 30 points (20 percent) are outliers and should be rejected.
    n_rejected = int((~reg.inlier_mask).sum())
    assert n_rejected >= 5  # RANSAC should catch nearly all of the 6 outliers
