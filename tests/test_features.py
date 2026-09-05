"""Fusion feature vector assembly (Section 10.3)."""
import numpy as np

from saresq.fuse.features import (
    FEATURE_NAMES,
    ContextEvidence,
    RegistrationEvidence,
    RgbEvidence,
    ThermalEvidence,
    TrackEvidence,
    assemble_features,
)

THERMAL = ThermalEvidence(z_peak=5.8, dT_peak=-4.0, sign=-1, area_t=6, ecc=0.7)
TRACK = TrackEvidence(hits=4, age=5)
CONTEXT = ContextEvidence(t_bg=33.0, lum=0.4, p_flood=0.1, p_fire=0.05, p_collapse=0.02, alt_band=1)


def test_feature_order_matches_appendix_b():
    assert FEATURE_NAMES == [
        "p_rgb", "has_rgb", "a_rgb",
        "z_peak", "dT", "sign", "area_t", "ecc",
        "iou", "d_c",
        "hits", "hit_ratio",
        "T_bg", "lum", "p_flood", "p_fire", "p_collapse", "alt_band",
    ]


def test_full_evidence_vector():
    rgb = RgbEvidence(score=0.62, box_area_px=800.0, crop_area_px=160.0 * 160.0)
    reg = RegistrationEvidence(iou=0.40, centroid_dist_norm=0.1)
    vec = assemble_features(THERMAL, rgb, reg, TRACK, CONTEXT)
    assert vec.shape == (18,)
    named = dict(zip(FEATURE_NAMES, vec))
    assert named["p_rgb"] == 0.62
    assert named["has_rgb"] == 1.0
    assert named["a_rgb"] == 800.0 / (160.0 * 160.0)
    assert named["iou"] == 0.40
    assert named["hits"] == 4
    assert named["hit_ratio"] == 0.8
    assert named["sign"] == -1.0
    assert named["alt_band"] == 1.0


def test_missing_rgb_is_zeros_plus_indicator_not_dropped():
    """Section 10.3: thermal-present/RGB-absent must be a distinct pattern, not omitted."""
    vec = assemble_features(THERMAL, None, None, TRACK, CONTEXT)
    named = dict(zip(FEATURE_NAMES, vec))
    assert named["has_rgb"] == 0.0
    assert named["p_rgb"] == 0.0
    assert named["a_rgb"] == 0.0
    assert named["iou"] == 0.0
    assert named["d_c"] == 0.0
    # Thermal evidence must still be present even though RGB is not.
    assert named["z_peak"] == 5.8
    assert vec.shape == (18,)
    assert not np.isnan(vec).any()
