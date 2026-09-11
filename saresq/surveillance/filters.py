"""Alpha-beta smoothing and the association gate.

Why alpha-beta and not a full Kalman filter: with a single position sensor and
a constant-velocity motion model, the Kalman gain converges to a constant after
a handful of updates. The alpha-beta filter simply *starts* at that steady
state. It is what real terminal radar trackers ran for decades, it has closed-
form error statistics we can quote, and it costs two multiplies per axis --
which matters, because the same code has to run on the Pi.

Gains come from Kalata's steady-state solution rather than being tuned by hand,
so there is exactly one physical knob: the tracking index

    L = sigma_a * T^2 / sigma_z

the ratio of how far process noise can move the target in one scan to how well
we can measure it. Everything else follows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Chi-square 99.73% point with 2 degrees of freedom. A plot is accepted into a
# track only if its normalised squared distance from the prediction is below
# this -- the "validation gate" of the tracking literature. Two dof because the
# gate is over (east, north); 99.73% because in search and rescue the cost of
# splitting one survivor into two tracks is much lower than the cost of
# silently discarding a real detection.
GATE_CHI2 = 11.83


def kalata_gains(tracking_index: float) -> tuple[float, float]:
    """Steady-state (alpha, beta) for a given tracking index.

    Kalata's result for the constant-velocity alpha-beta filter:

        r     = (4 + L - sqrt(8L + L^2)) / 4
        alpha = 1 - r^2
        beta  = 2 (1 - r)^2

    L -> 0 gives a heavily smoothed, sluggish track; large L gives one that
    follows every measurement. Returned gains are always in the stable region.
    """
    L = max(1e-9, float(tracking_index))
    r = (4.0 + L - math.sqrt(8.0 * L + L * L)) / 4.0
    r = min(max(r, 0.0), 1.0)
    alpha = 1.0 - r * r
    beta = 2.0 * (1.0 - r) ** 2
    return alpha, beta


def variance_reduction(alpha: float, beta: float, T: float) -> tuple[float, float]:
    """Benedict-Bordner steady-state variance reduction factors.

    Returns (position VRF, velocity VRF): multiply by the measurement variance
    sigma_z^2 to get the smoothed position and velocity variances. This is how
    the displayed uncertainty ellipse gets a real number rather than a guess.
    """
    denom = alpha * (4.0 - 2.0 * alpha - beta)
    if denom <= 1e-12:
        return 1.0, 1.0 / max(T * T, 1e-9)
    vrf_p = (2.0 * alpha * alpha + beta * (2.0 - 3.0 * alpha)) / denom
    vrf_v = (2.0 * beta * beta) / (T * T * denom)
    return max(vrf_p, 1e-6), max(vrf_v, 1e-9)


def gate_radius_m(sigma_pred_m: float, sigma_meas_m: float) -> float:
    """Radius of the validation gate in metres.

    The gate has to cover both how badly we predicted and how badly we measure,
    so the variances add before the chi-square scaling.
    """
    s2 = sigma_pred_m * sigma_pred_m + sigma_meas_m * sigma_meas_m
    return math.sqrt(GATE_CHI2 * max(s2, 1e-9))


@dataclass
class AlphaBeta:
    """Two-axis constant-velocity alpha-beta filter in local metres.

    State is (east, north) position and velocity relative to a local tangent
    plane origin -- not lat/lon. Doing the filtering in degrees would make the
    east and north gains differ by a factor of cos(latitude), which is exactly
    the kind of silent bug that shows up as a track that drifts sideways.
    """

    scan_period_s: float
    sigma_meas_m: float = 2.1          # NEO-6M, ~2.5 m CEP -> ~2.1 m per axis
    sigma_accel_ms2: float = 1.2       # a quadrotor's plausible per-scan manoeuvre
    #: A stationary object -- a survivor lying in rubble, not an aircraft.
    #: Velocity is never estimated and the position uncertainty never grows,
    #: because a fix on something that is not moving does not become less
    #: certain with time; it becomes *older*, and age is shown separately.
    #: Without this the constant-velocity dead-reckoning term (0.5 a t^2) puts
    #: a 300 m uncertainty circle around a casualty who has not moved a metre.
    static: bool = False
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    var_p: float = 0.0
    var_v: float = 0.0
    alpha: float = field(init=False, default=0.0)
    beta: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        T = self.scan_period_s
        self._z2 = self.sigma_meas_m ** 2
        if self.static:
            self.sigma_accel_ms2 = 0.0
            self.alpha, self.beta = 1.0, 0.0
            self.var_p = self._z2
            self.var_v = 0.0
            return
        L = self.sigma_accel_ms2 * T * T / max(self.sigma_meas_m, 1e-6)
        self.alpha, self.beta = kalata_gains(L)
        vrf_p, vrf_v = variance_reduction(self.alpha, self.beta, T)
        self.var_p = vrf_p * self._z2
        self.var_v = vrf_v * self._z2

    # -- prediction -------------------------------------------------------
    def predict(self, dt: float) -> tuple[float, float]:
        return self.x + self.vx * dt, self.y + self.vy * dt

    def sigma_pred_m(self, dt: float) -> float:  # noqa: D401
        """One-sigma dead-reckoning error dt seconds after the last update.

        Three terms, all real: the position we already had, the velocity error
        integrated over dt, and the distance an unmodelled acceleration could
        have carried the target in that time. This is the number that makes the
        coast ellipse grow at an honest rate instead of a decorative one.
        """
        if self.static:
            return math.sqrt(self.var_p)
        drift = 0.5 * self.sigma_accel_ms2 * dt * dt
        return math.sqrt(self.var_p + self.var_v * dt * dt + drift * drift)

    # -- update -----------------------------------------------------------
    def update(self, zx: float, zy: float, dt: float) -> float:
        """Fold in a measurement. Returns the residual magnitude in metres."""
        px, py = self.predict(dt)
        rx, ry = zx - px, zy - py
        self.x = px + self.alpha * rx
        self.y = py + self.alpha * ry
        if self.static:
            # alpha == 1: snap to the fix, never estimate a velocity, and keep
            # the uncertainty at the measurement's own accuracy.
            self.var_p = self._z2
            return math.hypot(rx, ry)
        if dt > 1e-6:
            self.vx += (self.beta / dt) * rx
            self.vy += (self.beta / dt) * ry
        vrf_p, vrf_v = variance_reduction(self.alpha, self.beta, max(dt, 1e-6))
        self.var_p = vrf_p * self._z2
        self.var_v = vrf_v * self._z2
        return math.hypot(rx, ry)

    def coast(self, dt: float) -> None:
        """No plot this scan: move the state along its own velocity.

        Deliberately does NOT decay the velocity towards zero. A coasted track
        that quietly slows down looks plausible and is wrong -- the whole point
        is to show where the target would be if it kept doing what it was doing.
        """
        if self.static:
            return
        self.x, self.y = self.predict(dt)
        s = self.sigma_pred_m(dt)
        self.var_p = s * s
        self.var_v += (self.sigma_accel_ms2 * dt) ** 2

    @property
    def speed_ms(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def course_deg(self) -> float:
        """Course over ground, degrees true (0 = north, clockwise)."""
        return math.degrees(math.atan2(self.vx, self.vy)) % 360.0
