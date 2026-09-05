"""Three-class assignment from calibrated probability + evidence pattern (Section 10.5).

Classes are assigned from the pattern, not the probability alone: a HIGH
score with no RGB corroboration is downgraded to MEDIUM, and an RGB
detection with no thermal support is downgraded to LOW regardless of its
own score.
"""
from __future__ import annotations


def classify(
    p: float,
    has_rgb: bool,
    iou: float,
    hits: int,
    z_peak: float,
    z_t: float = 2.5,
    high_thresh: float = 0.80,
    medium_thresh: float = 0.35,
    iou_min: float = 0.10,
    hits_min: int = 2,
) -> str:
    if has_rgb and z_peak < z_t:
        return "LOW"
    if p < medium_thresh:
        return "LOW"
    if p >= high_thresh and has_rgb and iou >= iou_min and hits >= hits_min:
        return "HIGH"
    return "MEDIUM"
