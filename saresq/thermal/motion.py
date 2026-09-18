"""Thermal motion detection: the evidence that a heat source is ALIVE.

WHY THIS IS NOT A NICETY
A 34 C blob tells you something is warm. It does not tell you whether it is a
casualty, a corpse, a car bonnet or a sun-warmed slab. Movement does. A heat
source that changes shape between frames is breathing, shifting, or reaching --
and for a rescue team deciding where to dig first, "this one is moving" is
worth more than another decimal place of detector confidence.

It also covers the case the visible branch cannot: in an unlit room or at night
over rubble the camera is blind (see RGB_BLIND_LUM in the payload), so thermal
is the only witness. Motion is the one extra thing thermal alone can still say.

WHAT COUNTS AS MOTION, AND WHAT MUST NOT
Three things masquerade as movement and each is handled explicitly:

  sensor noise    NETD is 0.1 K, so every pixel flickers. The threshold is
                  derived from the difference image's OWN robust spread rather
                  than being a fixed number, so it adapts to the sensor's mood.
  payload motion  On an aircraft the whole scene shifts. The previous frame is
                  aligned onto the current one first -- by the GNSS/IMU
                  transform in flight, or by phase correlation on a bench --
                  so only RESIDUAL change survives.
  thermal drift   Rooms and sensors warm slowly. Differences are taken over
                  about a second, far faster than drift, and the median is
                  removed from the difference so a uniform shift cancels.

A single hot pixel flickering is not a person, so a region must span at least
`min_px` connected pixels to count. At 20 m a prone adult is 2.6 thermal
pixels; at bench range it is far more, so 2 is deliberately permissive -- in
search and rescue a false alarm costs seconds of attention and a miss costs a
life.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:                        # pragma: no cover
    cv2 = None


#: Below this, a phase-correlation shift is indistinguishable from the
#: estimator's own noise on a 32x24 thermal frame and is treated as zero.
#: Measured, not chosen: over 3,000 pairs of a STILL low-texture scene the
#: spurious estimate has median 0.12 px (which is the phase-correlation floor
#: reported in Section VI-D), 99th percentile 0.84 px and maximum 1.13 px.
_ALIGN_FLOOR_PX = 1.0


@dataclass(frozen=True)
class MotionResult:
    moved: bool
    area_px: int          #: pixels that changed beyond the noise floor
    peak_dK: float        #: largest temperature change, kelvin
    n_regions: int        #: connected regions of change
    lag_s: float          #: how far back the comparison frame was
    sigma_dK: float       #: robust spread of the difference image

    @property
    def summary(self) -> str:
        if not self.moved:
            return "still"
        return f"{self.area_px} px moved, peak {self.peak_dK:.2f} K"


class ThermalMotion:
    """Frame-to-frame change detector over the raw 32x24 array.

    Reads raw frames, exactly like the gate, and for the same reason: the
    robust threshold needs independent samples, and a resampled or smoothed
    frame no longer has them.
    """

    def __init__(self, lag_frames: int = 4, k_sigma: float = 4.0,
                 dt_min_k: float = 0.45, min_px: int = 4,
                 peak_min_k: float = 0.60, border: int = 2,
                 compensate: bool = True):
        self.lag_frames = max(1, int(lag_frames))
        self.k_sigma = float(k_sigma)
        self.dt_min_k = float(dt_min_k)
        self.min_px = int(min_px)
        #: A real limb shifting changes temperature by more than a kelvin.
        #: Requiring a peak as well as an area stops a broad, shallow drift --
        #: an aircon plume crossing the frame, the sensor settling -- from
        #: being reported as a living person.
        self.peak_min_k = float(peak_min_k)
        #: Sub-pixel alignment leaves artefacts at the frame edge where there
        #: is nothing to warp from. Ignoring a two-pixel margin costs 30% of a
        #: 32x24 frame's border and removes the largest source of false motion
        #: when the payload is panning.
        self.border = int(border)
        self.compensate = bool(compensate)
        self._hist: deque = deque(maxlen=self.lag_frames + 1)
        self._t: deque = deque(maxlen=self.lag_frames + 1)

    def reset(self) -> None:
        self._hist.clear()
        self._t.clear()

    def add(self, frame: np.ndarray, t_s: float | None = None,
            shift: tuple[float, float] | None = None) -> MotionResult:
        """Add a frame and report the change since `lag_frames` ago.

        `shift` is the known (dx, dy) of THIS frame relative to the older one,
        in thermal pixels, when flight telemetry can supply it. Without it, and
        with compensate on, the offset is recovered from the frames themselves.
        """
        frame = np.asarray(frame, dtype=np.float64)
        self._hist.append(frame.copy())
        self._t.append(t_s if t_s is not None else 0.0)

        if len(self._hist) <= self.lag_frames:
            return MotionResult(False, 0, 0.0, 0, 0.0, 0.0)

        old = self._hist[0]
        lag_s = float(self._t[-1] - self._t[0]) if any(self._t) else 0.0

        aligned = self._align(old, frame, shift)
        d = frame - aligned
        # Remove any uniform component: the whole scene warming a little is
        # drift, not movement, and would otherwise light up every pixel.
        d = d - float(np.median(d))

        sigma = 1.4826 * float(np.median(np.abs(d)))
        thr = max(self.k_sigma * sigma, self.dt_min_k)
        mask = np.abs(d) >= thr
        if self.border:
            b = self.border
            edge = np.ones_like(mask)
            edge[b:-b, b:-b] = False
            mask &= ~edge

        area = int(mask.sum())
        n_regions = 0
        if area and cv2 is not None:
            n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            keep = np.zeros_like(mask)
            for i in range(1, n):
                region = lab == i
                if region.sum() >= self.min_px:
                    keep |= region
                    n_regions += 1
            mask = keep
            area = int(mask.sum())
        elif area:
            n_regions = 1 if area >= self.min_px else 0
            if n_regions == 0:
                area = 0

        peak = float(np.abs(d)[mask].max()) if area else 0.0
        if peak < self.peak_min_k:
            area, n_regions = 0, 0
        return MotionResult(moved=bool(area and n_regions),
                            area_px=area, peak_dK=peak, n_regions=n_regions,
                            lag_s=lag_s, sigma_dK=sigma)

    def _align(self, old: np.ndarray, new: np.ndarray,
               shift: tuple[float, float] | None) -> np.ndarray:
        """Bring the older frame onto the current one's grid.

        Without this every frame from a moving aircraft is 100% "motion". The
        shift is whole-scene; what remains after removing it is something in
        the scene that moved on its own.
        """
        if cv2 is None:
            return old
        if shift is not None:
            dx, dy = shift
        elif self.compensate:
            dx, dy = self._estimate(old, new)
        else:
            # `compensate` gates SELF-ESTIMATION only. A shift measured by
            # GNSS/IMU is a fact about the airframe and is always honoured; a
            # shift inferred from the frames is a guess, and the guess is only
            # safe when the payload is actually translating.
            return old
        if abs(dx) < 0.02 and abs(dy) < 0.02:
            return old
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(old.astype(np.float32), M,
                              (old.shape[1], old.shape[0]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE).astype(np.float64)

    @staticmethod
    def _estimate(old: np.ndarray, new: np.ndarray) -> tuple[float, float]:
        a = np.ascontiguousarray(new, dtype=np.float32)
        b = np.ascontiguousarray(old, dtype=np.float32)
        a = a - float(a.mean())
        b = b - float(b.mean())
        try:
            win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(b, a, win)
            # A whole-scene jump larger than this is not the payload drifting,
            # it is a bad correlation on a nearly featureless thermal frame.
            # Trusting it would warp the comparison and invent motion.
            if abs(dx) > 6 or abs(dy) > 6:
                return 0.0, 0.0
            # ...and so is anything BELOW the estimator's own noise floor.
            #
            # This one cost 23 false "movements" in 80 still frames before it
            # was found. On a low-texture thermal frame -- a warm body on a
            # flat background, which is the normal case -- phase correlation
            # scatters by up to ~0.5 px between two frames of a scene that has
            # not moved at all. Warping the older frame by that imaginary
            # offset shears every temperature gradient in it, and the
            # difference image then carries a crescent of real signal at the
            # edge of the warm blob. The detector is right to flag it; the
            # motion was manufactured one step earlier.
            #
            # phaseCorrelate's own response does NOT separate these cases --
            # measured at 1.11 mean for a true zero shift against 1.13 for a
            # true 1.5 px shift -- so confidence cannot be the gate and the
            # magnitude has to be.
            #
            # The cost of the deadband is nil. Ego-motion worth correcting is
            # LARGE: at 14 m/s and 20 m AGL a stationary point crosses 95 px
            # in 100 ms (Section IV-G). Shifts this rejects are ones the
            # payload cannot measure and does not need to.
            if float(np.hypot(dx, dy)) < _ALIGN_FLOOR_PX:
                return 0.0, 0.0
            return float(dx), float(dy)
        except Exception:
            return 0.0, 0.0
