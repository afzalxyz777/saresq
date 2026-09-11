"""The aircraft's survey pattern, its sensor footprint, and the radio link.

Nothing here is decoration. The dashboard needs a live platform track before
the airframe exists, and a track that jitters along a scripted line would teach
the operator nothing. So the aircraft flies the boustrophedon the mission
planner would actually fly, at the altitude and leg spacing the payload
geometry dictates, and the radio drops out where the geometry says it must --
behind a building, on the far leg, every lap.

That last point is the reason this file exists at all. A link that fails at a
random time produces a demo. A link that fails at a *place* produces a finding:
"we lose the aircraft over the north-east block on every eastbound leg, and
here is the eleven seconds of coasted track that results."
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from saresq.surveillance.geo import LocalFrame, haversine_m

# --- payload and airframe constants (Master Spec section 4) ----------------
FOV_ACROSS_DEG = 55.0
FOV_ALONG_DEG = 35.0
SURVEY_ALT_M = 20.0
CRUISE_MS = 6.0
#: Fraction of the thermal swath re-covered by the next leg. 20% is the usual
#: mapping overlap and leaves room for crosswind drift.
SIDELAP = 0.20

# --- radio (SiK 433 MHz telemetry, the link the airframe already carries) --
TX_DBM = 20.0
ANT_GAIN_DBI = 2.0
FREQ_MHZ = 433.0
LAMBDA_M = 299.792458 / FREQ_MHZ
#: Above this the link carries alerts and imagery. Between the two it carries
#: alerts only -- 28-byte packets survive a channel that stalls a JPEG.
RSSI_UP_DBM = -95.0
RSSI_DOWN_DBM = -105.0
#: Attenuation through built structure at 433 MHz. Reinforced concrete and
#: glass measure roughly 4-7 dB per metre of material traversed; 5.0 is a
#: mid-range figure and the model caps the total so it stays plausible.
STRUCTURE_DB_PER_M = 5.0
STRUCTURE_DB_MAX = 48.0


@dataclass(frozen=True)
class Box:
    """A lat/lon rectangle. Used for the survey segment and for obstructions."""

    lat0: float
    lon0: float
    lat1: float
    lon1: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat0 <= lat <= self.lat1 and self.lon0 <= lon <= self.lon1

    @property
    def centre(self) -> tuple[float, float]:
        return (self.lat0 + self.lat1) / 2, (self.lon0 + self.lon1) / 2


@dataclass(frozen=True)
class Obstruction:
    """A building block, as a footprint plus a roof height above the datum."""

    name: str
    box: Box
    top_m: float


# The area actually flown: "Segment A", inside the SRTM analysis window that
# static/basemap.js carries. 240 m by 180 m -- one battery, not one afternoon.
SEGMENT_A = Box(22.57239, 88.36383, 22.57401, 88.36617)
GROUND_STATION = (22.5726, 88.3639)
GCS_ANTENNA_M = 2.0
#: Terrain in this AOI runs 2-20 m on SRTM with a mean near 11 m. The link
#: model needs one datum, not a surface, so it uses the mean and treats the
#: obstruction heights as absolute -- the DEM itself lives in the browser.
TERRAIN_DATUM_M = 11.0

OBSTRUCTIONS = (
    Obstruction("NE residential block", Box(22.57300, 88.36470, 22.57352, 88.36560), 28.0),
    Obstruction("Market shed row", Box(22.57262, 88.36420, 22.57284, 88.36470), 22.0),
)


def swath_m(alt_m: float = SURVEY_ALT_M) -> float:
    """Ground width the thermal array sees across-track."""
    return 2.0 * alt_m * math.tan(math.radians(FOV_ACROSS_DEG) / 2.0)


def leg_spacing_m(alt_m: float = SURVEY_ALT_M) -> float:
    return swath_m(alt_m) * (1.0 - SIDELAP)


# ---------------------------------------------------------------------------
# survey pattern
# ---------------------------------------------------------------------------
@dataclass
class SurveyPlan:
    """A boustrophedon ("lawnmower") over `box`, flown east-west.

    East-west legs rather than north-south because the segment is wider than it
    is tall, and the fewer turns you fly the less battery you spend not
    searching.
    """

    box: Box = SEGMENT_A
    alt_m: float = SURVEY_ALT_M
    speed_ms: float = CRUISE_MS
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    legs: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.waypoints:
            return
        frame = LocalFrame(*self.box.centre)
        e0, n0 = frame.to_en(self.box.lat0, self.box.lon0)
        e1, n1 = frame.to_en(self.box.lat1, self.box.lon1)
        spacing = leg_spacing_m(self.alt_m)
        n_legs = max(2, int(round((n1 - n0) / spacing)) + 1)
        step = (n1 - n0) / (n_legs - 1)
        pts: list[tuple[float, float]] = []
        for i in range(n_legs):
            n = n0 + i * step
            a, b = (e0, e1) if i % 2 == 0 else (e1, e0)
            pts.append(frame.to_ll(a, n))
            pts.append(frame.to_ll(b, n))
        self.waypoints = pts
        self.legs = []
        acc = 0.0
        for i in range(len(pts) - 1):
            d = haversine_m(*pts[i], *pts[i + 1])
            self.legs.append((acc, d))
            acc += d
        self.length_m = acc

    @property
    def duration_s(self) -> float:
        return self.length_m / self.speed_ms

    def at(self, s_m: float) -> tuple[float, float, float]:
        """Position and course at arc length `s_m` along the pattern.

        Wraps, so the aircraft re-flies the segment. That is not laziness: a
        real search flies repeated sweeps, and it is the only way the operator
        ever sees cumulative probability of detection climb.
        """
        total = self.length_m
        s = s_m % total if total > 0 else 0.0
        for i, (start, d) in enumerate(self.legs):
            if s <= start + d or i == len(self.legs) - 1:
                f = 0.0 if d <= 0 else (s - start) / d
                a, b = self.waypoints[i], self.waypoints[i + 1]
                lat = a[0] + (b[0] - a[0]) * f
                lon = a[1] + (b[1] - a[1]) * f
                frame = LocalFrame(*a)
                de, dn = frame.to_en(*b)
                return lat, lon, math.degrees(math.atan2(de, dn)) % 360.0
        return self.waypoints[0][0], self.waypoints[0][1], 0.0

    def coverage_c(self, sweeps: float, sweep_width_m: float) -> float:
        """Koopman coverage C = W L / A for `sweeps` passes of the segment."""
        frame = LocalFrame(*self.box.centre)
        e0, n0 = frame.to_en(self.box.lat0, self.box.lon0)
        e1, n1 = frame.to_en(self.box.lat1, self.box.lon1)
        area = abs((e1 - e0) * (n1 - n0))
        return sweep_width_m * self.length_m * sweeps / max(area, 1.0)


# ---------------------------------------------------------------------------
# radio link
# ---------------------------------------------------------------------------
def _knife_edge_db(v: float) -> float:
    """ITU-R P.526 single knife-edge diffraction loss for Fresnel parameter v."""
    if v < -0.7:
        return 0.0
    return 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)


def _fspl_db(d_m: float) -> float:
    d_km = max(d_m, 1.0) / 1000.0
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(FREQ_MHZ) + 32.44


@dataclass
class LinkBudget:
    """What the radio actually does at a given aircraft position.

    Free-space loss alone would never break a 433 MHz link at 400 m -- and
    saying so is more useful than inventing a dropout. What breaks it is the
    built environment: the model walks the line of sight, charges 5 dB for
    every metre of it that passes through a building, and adds knife-edge
    diffraction where the path merely grazes a roof.
    """

    obstructions: tuple[Obstruction, ...] = OBSTRUCTIONS
    gcs: tuple[float, float] = GROUND_STATION
    samples: int = 48

    def rssi_dbm(self, lat: float, lon: float, alt_agl_m: float) -> tuple[float, dict]:
        d = haversine_m(self.gcs[0], self.gcs[1], lat, lon)
        h_tx = TERRAIN_DATUM_M + GCS_ANTENNA_M
        h_rx = TERRAIN_DATUM_M + alt_agl_m
        through_m = 0.0
        worst_v = -9.0
        blocker = None
        step = d / max(1, self.samples)
        for i in range(1, self.samples):
            f = i / self.samples
            plat = self.gcs[0] + (lat - self.gcs[0]) * f
            plon = self.gcs[1] + (lon - self.gcs[1]) * f
            h = h_tx + (h_rx - h_tx) * f
            for ob in self.obstructions:
                if not ob.box.contains(plat, plon):
                    continue
                if h < ob.top_m:
                    through_m += step
                    blocker = ob.name
                else:
                    d1, d2 = max(f * d, 1.0), max((1 - f) * d, 1.0)
                    clearance = h - ob.top_m
                    v = -clearance * math.sqrt(2.0 * (d1 + d2) / (LAMBDA_M * d1 * d2))
                    if v > worst_v:
                        worst_v, blocker = v, ob.name
        loss = _fspl_db(d)
        struct_db = min(STRUCTURE_DB_MAX, through_m * STRUCTURE_DB_PER_M)
        diff_db = _knife_edge_db(worst_v) if through_m == 0.0 else 0.0
        rssi = TX_DBM + 2 * ANT_GAIN_DBI - loss - struct_db - diff_db
        return rssi, {
            "range_m": round(d, 1),
            "fspl_db": round(loss, 1),
            "structure_db": round(struct_db, 1),
            "diffraction_db": round(diff_db, 1),
            "through_m": round(through_m, 1),
            "blocker": blocker,
        }

    @staticmethod
    def state(rssi: float) -> str:
        if rssi >= RSSI_UP_DBM:
            return "UP"
        if rssi >= RSSI_DOWN_DBM:
            return "DEGRADED"
        return "DOWN"


# ---------------------------------------------------------------------------
# sensor
# ---------------------------------------------------------------------------
def lateral_sigma_m(sweep_width_m: float, p_max: float) -> float:
    """Sigma of the lateral range curve consistent with a given sweep width.

    The sweep width W is by definition the integral of the lateral range curve.
    For a Gaussian curve P(y) = p_max exp(-(y/s)^2) that integral is
    p_max * s * sqrt(pi), so fixing W and p_max fixes s. This is what keeps the
    simulated detections consistent with the W the analytics page quotes
    instead of being a second, unrelated invention.
    """
    return sweep_width_m / max(p_max * math.sqrt(math.pi), 1e-6)


def detection_probability(lateral_m: float, sweep_width_m: float, p_max: float = 0.60) -> float:
    s = lateral_sigma_m(sweep_width_m, p_max)
    return p_max * math.exp(-((lateral_m / s) ** 2))


def lateral_offset_m(ac_lat: float, ac_lon: float, course_deg: float,
                     lat: float, lon: float) -> tuple[float, float]:
    """(across-track, along-track) offset of a point from the aircraft, metres."""
    frame = LocalFrame(ac_lat, ac_lon)
    e, n = frame.to_en(lat, lon)
    c = math.radians(course_deg)
    along = e * math.sin(c) + n * math.cos(c)
    across = e * math.cos(c) - n * math.sin(c)
    return across, along
