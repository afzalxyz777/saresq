"""Thermal blob persistence across frames (Section 5.3's `persist` feature).

"Number of consecutive thermal frames in which a blob was found within 1.5
predicted thermal pixels of this one" -- ego-motion-compensated, using the
same similarity transform as the RGB tracker (Section 9.2), scaled down by
the RGB-per-thermal-pixel ratios (Section 3.2) since this operates in
32x24 thermal-pixel space, not 1640x1232 RGB space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from saresq.gate.blobs import Blob
from saresq.track.egomotion import SimilarityTransform

# Section 3.2 nominal scale: RGB px per thermal px, across-track / along-track.
SCALE_ACROSS = 45.3
SCALE_ALONG = 36.8


def _centroid(blob: Blob) -> tuple[float, float]:
    u_min, v_min, u_max, v_max = blob.bbox_t
    return (u_min + u_max) / 2, (v_min + v_max) / 2


def scale_transform_to_thermal(transform: SimilarityTransform) -> SimilarityTransform:
    """The RGB-space similarity transform, rescaled into thermal-pixel units."""
    return SimilarityTransform(
        theta=transform.theta,
        tx=transform.tx / SCALE_ACROSS,
        ty=transform.ty / SCALE_ALONG,
    )


@dataclass
class _TrackedBlob:
    position: tuple[float, float]
    persist: int


class BlobPersistenceTracker:
    def __init__(self, radius_thermal_px: float = 1.5, thermal_center: tuple[float, float] = (16.0, 12.0)):
        self.radius = radius_thermal_px
        self.center = thermal_center
        self._prev: list[_TrackedBlob] = []

    def update(self, blobs: list[Blob], transform: SimilarityTransform | None) -> list[int]:
        """Returns a persist count per input blob (same order), and advances state."""
        predicted: list[tuple[float, float]] = []
        for tb in self._prev:
            if transform is not None:
                thermal_transform = scale_transform_to_thermal(transform)
                (pu, pv), = thermal_transform.apply(np.array([tb.position]), self.center)
                predicted.append((pu, pv))
            else:
                predicted.append(tb.position)

        centroids = [_centroid(b) for b in blobs]
        used_prev: set[int] = set()
        counts: list[int] = []
        new_state: list[_TrackedBlob] = []

        for c in centroids:
            best_j, best_d = None, self.radius
            for j, p in enumerate(predicted):
                if j in used_prev:
                    continue
                d = ((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2) ** 0.5
                if d <= best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                used_prev.add(best_j)
                persist = self._prev[best_j].persist + 1
            else:
                persist = 1
            counts.append(persist)
            new_state.append(_TrackedBlob(position=c, persist=persist))

        self._prev = new_state
        return counts

    def reset(self) -> None:
        self._prev = []
