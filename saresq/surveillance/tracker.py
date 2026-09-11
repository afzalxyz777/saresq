"""Track lifecycle: initiation, association, coasting, termination.

The state machine is the one terminal radar has used since ARTS, and the status
vocabulary is ASTERIX Cat 062 I062/080 so the words mean what a surveillance
engineer expects:

    TENTATIVE   CNF=1   initiating; held back from the display
    FIRM        CNF=0   confirmed by M-of-N; drawn normally
    COAST       CST=1   no plot this scan, position is dead-reckoned
    HELD                coasted past the drop threshold but kept anyway
    DROPPED     TSE=1   terminated, last message sent

One deliberate departure from air traffic control. A controller's tracker drops
a track after a few coasts, because an aircraft that stops replying has almost
certainly left the coverage volume. A survivor has not left anything. So the
tracker runs two profiles: the aircraft drops (PLATFORM), and survivor contacts
go to HELD and stay on the display until a human clears them (CONTACT). Losing
the radio must never delete a person from the map.
"""
from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from saresq.surveillance.filters import AlphaBeta, gate_radius_m
from saresq.surveillance.geo import LocalFrame


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    FIRM = "FIRM"
    COAST = "COAST"
    HELD = "HELD"
    DROPPED = "DROPPED"


#: A track that has never been updated is not displayed to the operator, which
#: is the whole reason tentative tracks exist -- see radartutorial's note that
#: tentative tracks are withheld to keep false tracks off the screen. We show
#: them, but greyed and explicitly marked, because in SAR the operator would
#: rather see a maybe than be protected from it.
DISPLAY_STATES = (TrackState.TENTATIVE, TrackState.FIRM, TrackState.COAST, TrackState.HELD)


@dataclass
class Plot:
    """One detection offered to the tracker this scan."""

    lat: float
    lon: float
    alt_m: float | None = None
    sigma_m: float = 2.1
    payload: dict[str, Any] = field(default_factory=dict)
    #: Set when the source already knows which track this belongs to (a target
    #: id from the store, an ICAO address on a real radar). Skips association.
    ident: str | None = None


@dataclass
class TrackerConfig:
    scan_period_s: float = 2.0
    #: M-of-N initiation. 3 of 4 is the textbook default and matches the
    #: example given in the track-initiation literature.
    init_m: int = 3
    init_n: int = 4
    #: Consecutive coasts before the track leaves COAST. What happens next is
    #: the profile's business: DROPPED for the aircraft, HELD for a contact.
    coasts_to_terminate: int = 5
    sigma_meas_m: float = 2.1
    sigma_accel_ms2: float = 1.2
    #: Profile for objects that do not move -- see AlphaBeta.static.
    static: bool = False
    #: How many past positions to keep for the trail. 60 scans at 2 s is two
    #: minutes of history, comfortably more than one survey leg.
    history_len: int = 60
    #: DROP terminates the track; HOLD keeps it forever in HELD.
    terminate_action: str = "DROP"
    label_prefix: str = "TRK"


class Track:
    """One tracked object and everything the display needs to draw it."""

    __slots__ = (
        "number", "label", "state", "filt", "frame", "created_t", "last_plot_t",
        "last_update_t", "hits", "misses", "coasts", "window", "history",
        "payload", "alt_m", "alt_rate_ms", "_prev_course", "_prev_speed",
        "trans", "longi", "vert", "ident", "simulated", "first_reported",
    )

    def __init__(self, number: int, label: str, frame: LocalFrame, plot: Plot,
                 t: float, cfg: TrackerConfig, simulated: bool = False):
        self.number = number
        self.label = label
        self.frame = frame
        self.state = TrackState.TENTATIVE
        e, n = frame.to_en(plot.lat, plot.lon)
        self.filt = AlphaBeta(cfg.scan_period_s, cfg.sigma_meas_m, cfg.sigma_accel_ms2,
                              static=cfg.static)
        self.filt.x, self.filt.y = e, n
        self.created_t = t
        self.last_plot_t = t
        self.last_update_t = t
        self.hits = 1
        self.misses = 0
        self.coasts = 0
        self.window: deque[bool] = deque([True], maxlen=cfg.init_n)
        self.history: deque[tuple[float, float, float, bool]] = deque(maxlen=cfg.history_len)
        self.history.append((t, plot.lat, plot.lon, False))
        self.payload = dict(plot.payload)
        self.alt_m = plot.alt_m
        self.alt_rate_ms = 0.0
        self._prev_course: float | None = None
        self._prev_speed: float | None = None
        self.trans = "UNDETERMINED"
        self.longi = "UNDETERMINED"
        self.vert = "UNDETERMINED"
        self.ident = plot.ident
        self.simulated = simulated
        self.first_reported = False
        # 1-of-1 initiation means the first plot confirms. Without this a
        # single-look source (a survivor seen on one pass) would sit tentative
        # forever and be deleted by the first coast.
        self._promote(cfg)

    # -- geometry ---------------------------------------------------------
    @property
    def lat(self) -> float:
        return self.frame.to_ll(self.filt.x, self.filt.y)[0]

    @property
    def lon(self) -> float:
        return self.frame.to_ll(self.filt.x, self.filt.y)[1]

    def shift(self, dt: float) -> None:
        """Move every timestamp on this track by `dt`.

        Used when the service re-bases its clock after a long idle gap. Shifting
        t0 without shifting the tracks leaves the platform (which gets re-plotted
        immediately) and the contacts (which do not) on two different clocks, and
        the contacts read hours stale on a picture that is minutes old.
        """
        self.created_t += dt
        self.last_plot_t += dt
        self.last_update_t += dt
        self.history = deque(((t + dt, la, lo, c) for (t, la, lo, c) in self.history),
                             maxlen=self.history.maxlen)

    def age_s(self, now: float) -> float:
        """Seconds since a real plot last touched this track. The single most
        useful number on the display, and the one a plain database view of
        `targets` can never give you."""
        return max(0.0, now - self.last_plot_t)

    def sigma_m(self, now: float) -> float:
        return self.filt.sigma_pred_m(self.age_s(now))

    # -- lifecycle --------------------------------------------------------
    def _promote(self, cfg: TrackerConfig) -> None:
        if self.state is TrackState.TENTATIVE and sum(self.window) >= cfg.init_m:
            self.state = TrackState.FIRM

    def apply_plot(self, plot: Plot, t: float, cfg: TrackerConfig) -> None:
        dt = max(1e-3, t - self.last_update_t)
        self.filt.update(*self.frame.to_en(plot.lat, plot.lon), dt)
        if plot.alt_m is not None:
            prev = self.alt_m
            # Altitude is smoothed separately and much harder than position:
            # it comes from the barometer/GPS, not from the position filter,
            # exactly as Mode C altitude is separate on a real radar.
            self.alt_m = plot.alt_m if prev is None else prev + 0.6 * (plot.alt_m - prev)
            if prev is not None:
                self.alt_rate_ms += 0.5 * ((self.alt_m - prev) / dt - self.alt_rate_ms)
        self.payload.update(plot.payload)
        self.hits += 1
        self.misses = 0
        self.coasts = 0
        self.window.append(True)
        self.last_plot_t = t
        self.last_update_t = t
        self.history.append((t, plot.lat, plot.lon, False))
        self._mode_of_movement(dt)
        if self.state in (TrackState.COAST, TrackState.HELD):
            self.state = TrackState.FIRM
        self._promote(cfg)

    def coast(self, t: float, cfg: TrackerConfig) -> None:
        dt = max(1e-3, t - self.last_update_t)
        self.filt.coast(dt)
        self.misses += 1
        self.coasts += 1
        self.window.append(False)
        self.last_update_t = t
        self.history.append((t, self.lat, self.lon, True))
        if self.state is TrackState.TENTATIVE:
            # An initiating track that misses is almost always noise. Kill it
            # immediately rather than letting clutter mature into a firm track.
            self.state = TrackState.DROPPED
            return
        if self.coasts >= cfg.coasts_to_terminate:
            self.state = TrackState.HELD if cfg.terminate_action == "HOLD" else TrackState.DROPPED
        elif self.state is not TrackState.HELD:
            self.state = TrackState.COAST

    def _mode_of_movement(self, dt: float) -> None:
        """ASTERIX I062/200. Cheap, and it is what turns a moving dot into a
        readable picture: 'left turn, decelerating, descending' is a sentence."""
        sp, co = self.filt.speed_ms, self.filt.course_deg
        if self._prev_course is not None and dt > 1e-3:
            d = (co - self._prev_course + 540.0) % 360.0 - 180.0
            rate = d / dt
            self.trans = "CONSTANT" if abs(rate) < 1.5 else ("RIGHT" if rate > 0 else "LEFT")
        if self._prev_speed is not None and dt > 1e-3:
            a = (sp - self._prev_speed) / dt
            self.longi = "CONSTANT" if abs(a) < 0.25 else ("INCREASING" if a > 0 else "DECREASING")
        self.vert = ("LEVEL" if abs(self.alt_rate_ms) < 0.3
                     else ("CLIMB" if self.alt_rate_ms > 0 else "DESCENT"))
        self._prev_course, self._prev_speed = co, sp

    # -- display ----------------------------------------------------------
    def status_bits(self, now: float) -> dict[str, int]:
        """The subset of ASTERIX Cat 062 I062/080 that we can honestly fill in."""
        return {
            "CNF": 1 if self.state is TrackState.TENTATIVE else 0,
            "CST": 1 if self.state in (TrackState.COAST, TrackState.HELD) else 0,
            "TSB": 1 if not self.first_reported else 0,
            "TSE": 1 if self.state is TrackState.DROPPED else 0,
            "SIM": 1 if self.simulated else 0,
            "MON": 1,  # monosensor: one GPS, one thermal camera, no fusion of radars
        }

    def to_dict(self, now: float) -> dict[str, Any]:
        age = self.age_s(now)
        hist = [
            {"t": round(t, 2), "lat": round(la, 7), "lon": round(lo, 7), "coast": c}
            for (t, la, lo, c) in self.history
        ]
        return {
            "number": self.number,
            "label": self.label,
            "state": self.state.value,
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "alt_m": None if self.alt_m is None else round(self.alt_m, 1),
            "speed_ms": round(self.filt.speed_ms, 2),
            "course_deg": round(self.filt.course_deg, 1),
            "vx": round(self.filt.vx, 3),
            "vy": round(self.filt.vy, 3),
            "alt_rate_ms": round(self.alt_rate_ms, 2),
            "age_s": round(age, 2),
            "sigma_m": round(self.sigma_m(now), 2),
            "hits": self.hits,
            "coasts": self.coasts,
            "quality": round(sum(self.window) / max(1, len(self.window)), 3),
            "trans": self.trans,
            "long": self.longi,
            "vert": self.vert,
            "status": self.status_bits(now),
            "history": hist,
            "payload": self.payload,
        }


class Tracker:
    """Greedy nearest-neighbour tracker over a validation gate.

    Greedy rather than globally optimal (JPDA, Munkres) on purpose: with a
    single aircraft and a handful of static contacts a kilometre apart, the
    assignment is never ambiguous, and greedy is auditable at a glance. If this
    ever tracks a swarm, that is the line to change.
    """

    def __init__(self, frame: LocalFrame, cfg: TrackerConfig | None = None,
                 simulated: bool = False):
        self.frame = frame
        self.cfg = cfg or TrackerConfig()
        self.simulated = simulated
        self.tracks: dict[int, Track] = {}
        self._numbers = itertools.count(1)
        self.scan = 0

    # -- main entry point -------------------------------------------------
    def update(self, t: float, plots: Iterable[Plot]) -> list[Track]:
        """Advance one scan. Returns the tracks worth displaying."""
        self.scan += 1
        plots = list(plots)
        unassigned = set(range(len(plots)))
        taken: set[int] = set()

        # 1. Anything carrying an identity binds directly -- a store target id
        #    is a hard association, not a guess, and must never be gated away.
        by_ident = {tr.ident: tr for tr in self.tracks.values() if tr.ident}
        for i, p in enumerate(plots):
            tr = by_ident.get(p.ident) if p.ident else None
            if tr is not None and tr.number not in taken:
                tr.apply_plot(p, t, self.cfg)
                taken.add(tr.number)
                unassigned.discard(i)

        # 2. Everything else associates by normalised distance inside the gate.
        cands: list[tuple[float, int, int]] = []
        for i in sorted(unassigned):
            p = plots[i]
            for tr in self.tracks.values():
                if tr.number in taken or tr.state is TrackState.DROPPED:
                    continue
                if tr.ident and p.ident and tr.ident != p.ident:
                    continue
                dt = max(1e-3, t - tr.last_update_t)
                px, py = tr.filt.predict(dt)
                zx, zy = self.frame.to_en(p.lat, p.lon)
                d = math.hypot(zx - px, zy - py)
                g = gate_radius_m(tr.filt.sigma_pred_m(dt), p.sigma_m)
                if d <= g:
                    cands.append((d / g, i, tr.number))
        cands.sort()
        for _norm, i, num in cands:
            if i not in unassigned or num in taken:
                continue
            self.tracks[num].apply_plot(plots[i], t, self.cfg)
            taken.add(num)
            unassigned.discard(i)

        # 3. Leftover plots start tentative tracks.
        for i in sorted(unassigned):
            num = next(self._numbers)
            label = plots[i].payload.get("label") or f"{self.cfg.label_prefix}{num:03d}"
            self.tracks[num] = Track(num, label, self.frame, plots[i], t, self.cfg,
                                     simulated=self.simulated)
            taken.add(num)

        # 4. Every track that got nothing this scan coasts.
        for tr in list(self.tracks.values()):
            if tr.number not in taken and tr.state is not TrackState.DROPPED:
                tr.coast(t, self.cfg)

        for num, tr in list(self.tracks.items()):
            if tr.state is TrackState.DROPPED:
                del self.tracks[num]

        out = [tr for tr in self.tracks.values() if tr.state in DISPLAY_STATES]
        for tr in out:
            tr.first_reported = True
        return out

    def shift(self, dt: float) -> None:
        for tr in self.tracks.values():
            tr.shift(dt)

    def snapshot(self, now: float) -> list[dict[str, Any]]:
        return [tr.to_dict(now) for tr in self.tracks.values()
                if tr.state in DISPLAY_STATES]
