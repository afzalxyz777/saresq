"""Ego-motion-compensated SORT (Section 9.1, 9.4).

Standard SORT assumes a roughly still camera. At survey speed the aircraft's
own motion dwarfs a survivor's (zero) velocity, so association must run on
ego-motion-compensated predictions, not raw constant-velocity ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from saresq.track.egomotion import SimilarityTransform
from saresq.track.kalman import ConstantVelocityKF


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """IoU of two center-format boxes (u, v, w, h)."""
    ua, va, wa, ha = box_a
    ub, vb, wb, hb = box_b
    ax1, ay1, ax2, ay2 = ua - wa / 2, va - ha / 2, ua + wa / 2, va + ha / 2
    bx1, by1, bx2, by2 = ub - wb / 2, vb - hb / 2, ub + wb / 2, vb + hb / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = wa * ha + wb * hb - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    kf: ConstantVelocityKF
    w: float
    h: float
    hits: int = 1
    age: int = 1
    misses: int = 0
    confirmed: bool = False
    recent_hits: list[bool] = field(default_factory=lambda: [True])

    def box(self) -> tuple[float, float, float, float]:
        u, v = self.kf.position
        return u, v, self.w, self.h


class EgoMotionTracker:
    def __init__(
        self,
        iou_match: float = 0.3,
        confirm_hits: int = 2,
        confirm_window: int = 3,
        max_misses: int = 5,
        compensate: bool = True,
    ):
        self.iou_match = iou_match
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.max_misses = max_misses
        self.compensate = compensate
        self.tracks: list[Track] = []
        self._next_id = 1
        self.n_tracks_created = 0

    def step(
        self,
        detections: list[tuple[float, float, float, float]],
        transform: SimilarityTransform | None,
        center: tuple[float, float],
        dt: float = 1.0,
    ) -> list[Track]:
        # 1. Predict: compensate for ego-motion, then let the KF predict residual motion.
        for tr in self.tracks:
            if self.compensate and transform is not None:
                u0, v0 = tr.kf.position
                (u1, v1), = transform.apply(np.array([[u0, v0]]), center)
                tr.kf.nudge_position(u1 - u0, v1 - v0)
            tr.kf.predict(dt)

        # 2. Associate by IoU (Hungarian on 1 - IoU cost).
        matched_track_idx: set[int] = set()
        matched_det_idx: set[int] = set()
        if self.tracks and detections:
            cost = np.ones((len(self.tracks), len(detections)))
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - box_iou(tr.box(), det)
            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                if 1.0 - cost[i, j] >= self.iou_match:
                    matched_track_idx.add(i)
                    matched_det_idx.add(j)
                    tr = self.tracks[i]
                    u, v, w, h = detections[j]
                    tr.kf.update(u, v)
                    tr.w, tr.h = w, h
                    tr.hits += 1
                    tr.age += 1
                    tr.misses = 0
                    tr.recent_hits.append(True)
                    tr.recent_hits = tr.recent_hits[-self.confirm_window:]
                    if not tr.confirmed and sum(tr.recent_hits) >= self.confirm_hits:
                        tr.confirmed = True

        # 3. Unmatched tracks: age, miss, maybe delete.
        for i, tr in enumerate(self.tracks):
            if i not in matched_track_idx:
                tr.age += 1
                tr.misses += 1
                tr.recent_hits.append(False)
                tr.recent_hits = tr.recent_hits[-self.confirm_window:]
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]

        # 4. Unmatched detections spawn new tentative tracks.
        for j, det in enumerate(detections):
            if j not in matched_det_idx:
                u, v, w, h = det
                self.tracks.append(Track(track_id=self._next_id, kf=ConstantVelocityKF(u, v), w=w, h=h))
                self._next_id += 1
                self.n_tracks_created += 1

        return self.tracks
