"""Multi-frame super-resolution for the 32x24 thermal array (drizzle).

WHY THIS IS NOT INTERPOLATION
Interpolating one frame invents nothing: a prone adult that covers 2.61 thermal
pixels at 20 m still carries 2.61 pixels of evidence after being upscaled, and
the smooth result is a picture of the interpolation kernel, not of the person.

Drizzle is different in kind, and the difference is the aircraft's own motion.
The MLX90640 is badly undersampled -- its detector pitch is coarser than the
scene detail the optics deliver -- so each frame aliases. Photograph the same
ground from positions offset by a FRACTION of a pixel and each frame samples a
different part of that aliased signal. Combining them on a finer grid recovers
detail that no single frame contains. This is the variable-pixel linear
reconstruction of Fruchter & Hook (PASP 114, 144, 2002), written for Hubble's
undersampled WFPC2 and used ever since.

The requirement is therefore SUB-PIXEL DIVERSITY, and it is a physical
precondition, not a tuning knob. Frames separated by whole pixels, or by
nothing at all, add signal-to-noise and no resolution. `DrizzleResult.diversity`
measures it and the caller is expected to refuse the result when it is low --
claiming super-resolution from a stationary payload would be a lie that looks
convincing.

WHAT IT DELIVERS HERE, MEASURED -- AND WHAT IT DOES NOT
Against a synthetic scene at the real NETD of 0.1 K, 8 dithered frames at
scale 2 (tests/test_drizzle.py holds the regression):

    warm-target centroid error   0.621 -> 0.269 thermal px   (2.30x better)
    background noise             0.097 -> 0.044 K            (2.23x, sqrt8=2.83)

The centroid number is the one that matters operationally: it is the estimator
saresq.gate.contrast feeds to the affine fit, whose current residual is 0.506
thermal pixels, and it is what centres the crop the detector sees.

It does NOT sharpen detail much beyond the single-frame limit, and the reason
is worth stating plainly rather than discovering later. Drizzle recovers
SAMPLING, not the detector's own blur. The MLX90640's thermopile pixels
integrate over essentially their whole area -- a ~100% fill factor -- so the
box MTF is already near its first zero at the sampling frequency, and there is
little true resolution sitting above Nyquist to recover. Hubble's WFPC2, which
drizzle was written for, had a much smaller effective fill factor, which is
exactly why it gained more. So: better positions, less noise, no aliasing, and
an honest picture -- not a magically sharper one.

HOW IT FITS THE REST OF THE SYSTEM
Deliberately an ENRICHMENT stage, never a replacement for the gate:

    thermal frame ---> robust-z gate          (unchanged, raw 32x24, every frame)
                 \\
                  --> drizzle accumulator ---> finer grid, for
                                               display, blob shape, centroids

The gate keeps running on raw frames because its statistics require
independent samples -- resampled pixels are correlated, the MAD collapses, and
z inflates everywhere (the same reason interpolation must not feed it). The
drizzled product is a second, richer view for the things that benefit from
resolution: what an operator sees, the sub-pixel centroid that
saresq.calib.register needs, and the blob eccentricity and area that enter the
fusion vector.

SHIFTS COME FROM WHICHEVER SOURCE EXISTS
In flight, saresq.track.egomotion.predict_transform already yields the
inter-frame motion from GNSS and gyroscope, and
saresq.gate.persistence.scale_transform_to_thermal converts it into thermal
pixels. On a bench there is no GPS fix, so `estimate_shift` recovers the offset
from the frames themselves by phase correlation. The same accumulator serves
both, which is what makes the feature demonstrable indoors by hand.

MOVING TARGETS ARE OUT OF SCOPE, AND THAT IS HONEST
Drizzle assumes the scene is static and only the sensor moves. A walking person
smears across the accumulation. This matches the assumption the tracker already
makes -- a survivor's own velocity is zero -- and the accumulation window is
short (a second or two at 4 Hz) to bound the damage when it does not hold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                                     # cv2 is present on the payload and here
    import cv2
except Exception:                        # pragma: no cover - keeps import cheap
    cv2 = None


@dataclass(frozen=True)
class DrizzleResult:
    """The reconstructed field plus everything needed to judge it."""

    field: np.ndarray        #: (H*scale, W*scale) temperatures, same units as input
    weight: np.ndarray       #: accumulated footprint weight per output pixel
    n_frames: int
    scale: int
    pixfrac: float
    diversity: float         #: 0 = no sub-pixel spread, 1 = ideally dithered
    coverage: float          #: fraction of output pixels with any weight

    @property
    def trustworthy(self) -> bool:
        """Whether this may be presented AS super-resolution.

        Both conditions are physical. Below ~0.25 diversity the frames sampled
        nearly the same phase and the extra pixels carry no new information;
        below 0.98 coverage there are holes, and a hole filled by nothing is
        not a measurement.
        """
        return self.n_frames >= 3 and self.diversity >= 0.25 and self.coverage >= 0.98


def estimate_shift(ref: np.ndarray, frame: np.ndarray,
                   upsample: int = 2) -> tuple[float, float]:
    """Sub-pixel (dx, dy) that maps `frame` onto `ref`, by phase correlation.

    Returns the offset in INPUT pixels: a feature at column x in `frame` sits at
    x + dx in `ref`. Hanning-windowed, because a 32x24 frame has severe edge
    discontinuities and without the window the correlation locks onto them
    rather than onto the scene.

    ACCURACY, MEASURED on the synthetic scene in tests/test_drizzle.py at the
    real NETD, over 60 random shifts in +/-1.5 px:

        upsample 1   mean 0.147 px, p90 0.233
        upsample 2   mean 0.118 px, p90 0.184     <- default
        upsample 4   mean 0.388 px, p90 0.584

    Two-times upsampling puts the correlation peak on a finer grid and helps.
    Four times makes it markedly WORSE, because at that ratio the cubic kernel's
    own ringing dominates a frame this small -- more interpolation is not more
    information, the same lesson as everywhere else in this system.

    Roughly an eighth of a pixel of residual error is the floor here, and it is
    comparable to the dither being exploited. That is why `Drizzle` is useful
    for position and noise rather than for detail, and why nothing downstream
    should treat a drizzled field as if the shifts were exact.
    """
    if cv2 is None:
        raise RuntimeError("phase correlation needs OpenCV; pass shifts explicitly")
    a = np.ascontiguousarray(ref, dtype=np.float32)
    b = np.ascontiguousarray(frame, dtype=np.float32)
    k = max(1, int(upsample))
    if k > 1:
        a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    # Remove the DC term. Thermal frames sit on a large common offset, which
    # would otherwise dominate the correlation peak.
    a = a - float(a.mean())
    b = b - float(b.mean())
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(b, a, win)
    return float(dx) / k, float(dy) / k


def _diversity(shifts: np.ndarray) -> float:
    """Circular spread of the fractional parts of the shifts, per axis, worst case.

    Fractional phase is what matters and it wraps, so a circular statistic is
    the correct one: offsets of 0.02 and 0.98 px are nearly the SAME phase, and
    a linear variance would call them far apart. 1 - |mean(exp(2*pi*i*frac))| is
    0 when every frame lands on one phase and approaches 1 when they are spread
    evenly around the cycle.
    """
    if len(shifts) < 2:
        return 0.0
    out = []
    for axis in range(2):
        frac = np.mod(shifts[:, axis], 1.0)
        r = np.abs(np.mean(np.exp(2j * np.pi * frac)))
        out.append(float(1.0 - r))
    return min(out)          # both axes must be dithered, so take the worse


class Drizzle:
    """Accumulates shifted frames onto a finer grid.

    Cost is bounded and small: each input pixel writes into a fixed window of
    at most (ceil(scale*pixfrac)+1)^2 output pixels, so a whole 32x24 frame is
    a handful of vectorised operations on 768-element arrays. At scale 2 that
    is well under a millisecond, which is what lets it sit in the payload's
    4 Hz thermal loop without touching the frame budget.
    """

    def __init__(self, shape: tuple[int, int] = (24, 32), scale: int = 2,
                 pixfrac: float = 0.65, max_frames: int = 8):
        if not (0.0 < pixfrac <= 1.0):
            raise ValueError("pixfrac must be in (0, 1]")
        if scale < 1:
            raise ValueError("scale must be >= 1")
        self.in_h, self.in_w = shape
        self.scale = int(scale)
        self.pixfrac = float(pixfrac)
        self.max_frames = int(max_frames)
        self.out_h, self.out_w = self.in_h * self.scale, self.in_w * self.scale
        self.reset()

        # Input pixel centres, in input-pixel units.
        jj, ii = np.mgrid[0:self.in_h, 0:self.in_w]
        self._cx0 = (ii + 0.5).ravel()
        self._cy0 = (jj + 0.5).ravel()
        # Footprint half-width in OUTPUT pixels. pixfrac shrinks the input
        # pixel before it is dropped, which is the mechanism that keeps the
        # combined PSF narrow instead of smearing it by a whole input pixel.
        self._half = self.scale * self.pixfrac / 2.0
        self._win = int(np.ceil(2 * self._half)) + 1

    def reset(self) -> None:
        self._sci = np.zeros((self.out_h, self.out_w), dtype=np.float64)
        self._wht = np.zeros((self.out_h, self.out_w), dtype=np.float64)
        self._shifts: list[tuple[float, float]] = []

    @property
    def n_frames(self) -> int:
        return len(self._shifts)

    def add(self, frame: np.ndarray, shift: tuple[float, float] = (0.0, 0.0)) -> None:
        """Drop one frame onto the output grid.

        `shift` is (dx, dy) in INPUT pixels, the offset of this frame relative
        to the reference frame -- exactly what `estimate_shift` returns.
        """
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != (self.in_h, self.in_w):
            raise ValueError(f"expected {(self.in_h, self.in_w)}, got {frame.shape}")

        dx, dy = float(shift[0]), float(shift[1])
        # Map this frame's pixel centres onto the REFERENCE grid, then scale
        # into output coordinates. The sign follows the documented convention:
        # a feature at column x in `frame` sits at x + dx in `ref`, so the
        # frame's own coordinates are advanced by the shift, not reduced by it.
        # Getting this backwards does not fail loudly -- it quietly doubles the
        # dither and reconstructs a blurrier image than plain interpolation,
        # which is exactly how it was caught (RMS 0.56 K against 0.11 K).
        cx = (self._cx0 + dx) * self.scale
        cy = (self._cy0 + dy) * self.scale
        vals = frame.ravel()

        x0 = np.floor(cx - self._half).astype(np.int64)
        y0 = np.floor(cy - self._half).astype(np.int64)

        for a in range(self._win):
            nx = x0 + a
            # Overlap between the footprint [cx-half, cx+half] and output bin
            # [nx, nx+1]. This area weighting is the whole of drizzle: a frame
            # contributes to an output pixel in proportion to how much of its
            # shrunken footprint actually lands there.
            ox = np.clip(np.minimum(cx + self._half, nx + 1.0)
                         - np.maximum(cx - self._half, nx.astype(np.float64)), 0.0, None)
            if not ox.any():
                continue
            for b in range(self._win):
                ny = y0 + b
                oy = np.clip(np.minimum(cy + self._half, ny + 1.0)
                             - np.maximum(cy - self._half, ny.astype(np.float64)), 0.0, None)
                w = ox * oy
                ok = (w > 0) & (nx >= 0) & (nx < self.out_w) & (ny >= 0) & (ny < self.out_h)
                if not ok.any():
                    continue
                flat = (ny[ok] * self.out_w + nx[ok])
                np.add.at(self._sci.reshape(-1), flat, vals[ok] * w[ok])
                np.add.at(self._wht.reshape(-1), flat, w[ok])

        self._shifts.append((dx, dy))
        if len(self._shifts) > self.max_frames:
            # The window is bounded rather than growing without limit: the
            # static-scene assumption decays with time, so old frames are not
            # merely less useful, they are actively wrong once anything moves.
            self._decay()

    def _decay(self) -> None:
        """Drop the oldest contribution by exponential forgetting.

        Exactly removing one frame would mean storing every frame's footprint.
        Scaling both accumulators instead costs nothing, keeps the ratio
        (and therefore the temperatures) unchanged, and lets new frames
        dominate within a few additions.
        """
        k = 1.0 - 1.0 / self.max_frames
        self._sci *= k
        self._wht *= k
        self._shifts.pop(0)

    def result(self, fill: str = "nearest") -> DrizzleResult:
        """Reconstruct. Output temperatures are a weighted mean, so units are
        preserved -- drizzle's normalisation by accumulated weight is what makes
        it correct for an intensive quantity like temperature, not only for the
        photon counts it was invented for."""
        wht = self._wht
        with np.errstate(invalid="ignore", divide="ignore"):
            field = np.where(wht > 0, self._sci / np.maximum(wht, 1e-12), np.nan)

        coverage = float((wht > 0).mean())
        if fill == "nearest" and not np.isfinite(field).all():
            field = self._fill_holes(field)

        shifts = np.array(self._shifts, dtype=float) if self._shifts else np.zeros((0, 2))
        return DrizzleResult(
            field=field, weight=wht.copy(), n_frames=len(self._shifts),
            scale=self.scale, pixfrac=self.pixfrac,
            diversity=_diversity(shifts), coverage=coverage,
        )

    @staticmethod
    def _fill_holes(field: np.ndarray) -> np.ndarray:
        """Fill uncovered output pixels from their nearest covered neighbour.

        Holes appear when pixfrac is small and the dither did not reach
        everywhere. Filling them is a DISPLAY convenience; `coverage` still
        reports the truth, and `trustworthy` still refuses a sparse result.
        """
        out = field.copy()
        bad = ~np.isfinite(out)
        if not bad.any():
            return out
        if cv2 is not None:
            src = np.nan_to_num(out, nan=0.0).astype(np.float32)
            filled = cv2.inpaint(
                _to_u8(src), bad.astype(np.uint8), 2, cv2.INPAINT_NS)
            lo, hi = float(np.nanmin(out)), float(np.nanmax(out))
            out[bad] = (filled.astype(np.float64)[bad] / 255.0) * (hi - lo) + lo
            return out
        out[bad] = float(np.nanmean(out))
        return out


def _to_u8(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (((a - lo) / max(hi - lo, 1e-6)) * 255).astype(np.uint8)
