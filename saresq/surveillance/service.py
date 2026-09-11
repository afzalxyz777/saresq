"""The live surveillance picture the /radar page consumes.

One object owns the whole picture: where the aircraft is, what the radio is
doing, which contacts have been reported, what is still queued on the aircraft,
and which parts of the segment have actually been searched.

    svc = RadarService(db_path="saresq.db")
    svc.snapshot()     # advances to now, returns the display state

Source of truth for contact positions is the store: if `targets` has rows, the
radar rediscovers those. Only when the store is empty does it fall back to
synthetic contacts, and every track it produces then carries the ASTERIX SIM
bit -- the same bit a real system uses to mark a simulated track, so a screen
grab can never be mistaken for a real sortie.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from saresq.sync.alerts import ALERT_BYTES, Alert, pack_alert, unpack_alert
from saresq.surveillance.flight import (
    GROUND_STATION,
    SEGMENT_A,
    SURVEY_ALT_M,
    Box,
    LinkBudget,
    SurveyPlan,
    detection_probability,
    lateral_offset_m,
    leg_spacing_m,
    swath_m,
)
from saresq.surveillance.geo import LocalFrame, haversine_m
from saresq.surveillance.tracker import Plot, Tracker, TrackerConfig

SCAN_PERIOD_S = 2.0
#: Effective sweep width. Assumed until eval_pixels_on_target.py measures it
#: from real footage -- the one number in this file waiting on the Colab run.
SWEEP_WIDTH_M = 14.0
COVERAGE_NX, COVERAGE_NY = 60, 45
#: Cap on how far the service will fast-forward in one call. Without it, a tab
#: left open overnight would try to replay 40,000 scans on the next poll.
MAX_CATCHUP_SCANS = 400
EVENT_LOG_LEN = 90

_CLASS_BANDS = ((0.75, "HIGH"), (0.40, "MEDIUM"), (0.0, "LOW"))

#: ICD 203 likelihood bands. The operator never reads a bare probability
#: without the words the intelligence community standardised for it.
_ICD203 = (
    (0.05, "almost no chance"), (0.20, "very unlikely"), (0.45, "unlikely"),
    (0.55, "roughly even chance"), (0.80, "likely"), (0.95, "very likely"),
    (1.01, "almost certainly"),
)


def icd203(p: float) -> str:
    for hi, word in _ICD203:
        if p < hi:
            return word
    return "almost certainly"


def classify(p: float) -> str:
    for lo, name in _CLASS_BANDS:
        if p >= lo:
            return name
    return "LOW"


def _det_roll(key: str, scan: int) -> float:
    """Reproducible uniform in [0,1).

    A seeded RNG would drift with call order; hashing (contact, scan) means the
    same scan always produces the same outcome no matter how the page is
    refreshed. Reloading the dashboard must not re-roll whether a survivor was
    found.
    """
    h = hashlib.sha256(f"{key}:{scan}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2 ** 32


@dataclass
class Contact:
    """A survivor position the aircraft is trying to find."""

    key: str
    lat: float
    lon: float
    target_id: int | None = None
    from_store: bool = False
    detections: list[float] = field(default_factory=list)
    first_seen_t: float | None = None
    last_seen_t: float | None = None
    reported: bool = False

    @property
    def p_cum(self) -> float:
        """Cumulative probability this contact is a real survivor.

        1 - prod(1 - p_i) over independent looks: the same combination rule the
        Koopman POD curve on the analytics page uses, applied per contact
        instead of per segment.
        """
        q = 1.0
        for p in self.detections:
            q *= (1.0 - p)
        return 1.0 - q


class RadarService:
    """Stateful, single-process. One ground station, one picture.

    Deliberately not shared across worker processes: this is the laptop in the
    tent, not a web farm. Running it under multiple workers would give each one
    its own aircraft, which is worse than obvious -- it is subtly wrong.
    """

    def __init__(self, db_path: str = "saresq.db", *, plan: SurveyPlan | None = None,
                 scan_period_s: float = SCAN_PERIOD_S, sweep_width_m: float = SWEEP_WIDTH_M,
                 clock=time.time):
        self.plan = plan or SurveyPlan()
        self.scan_period_s = scan_period_s
        self.sweep_width_m = sweep_width_m
        self.db_path = db_path
        self._clock = clock
        self._lock = threading.RLock()
        self.frame = LocalFrame(*self.plan.box.centre)
        self.link = LinkBudget()

        self.platform = Tracker(self.frame, TrackerConfig(
            scan_period_s=scan_period_s, sigma_meas_m=2.1, sigma_accel_ms2=1.2,
            coasts_to_terminate=20, terminate_action="DROP", label_prefix="SQ",
            history_len=140,
        ), simulated=True)
        self.contacts_tracker = Tracker(self.frame, TrackerConfig(
            scan_period_s=scan_period_s, sigma_meas_m=3.5, sigma_accel_ms2=0.05,
            init_m=1, init_n=1, coasts_to_terminate=8, terminate_action="HOLD",
            label_prefix="C", history_len=20, static=True,
        ), simulated=True)

        self.t0 = self._clock()
        self.scan = 0
        self.contacts: list[Contact] = []
        self._contacts_loaded = False
        self.queue: list[dict[str, Any]] = []      # alerts held on the aircraft
        self.delivered: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.coverage = bytearray(COVERAGE_NX * COVERAGE_NY)
        self.link_state = "UP"
        self.link_since = self.t0
        self.last_rssi = 0.0
        self.last_detail: dict[str, Any] = {}
        self.bytes_delivered = 0
        #: Fault injection. The propagation model above is honest, and what it
        #: honestly says is that a 433 MHz link at 400 m has roughly 60 dB of
        #: margin: it degrades behind the block, it does not die. Rather than
        #: tune the constants until a dropout appears -- which would make every
        #: other number in the model a lie -- outages are injected explicitly
        #: and labelled INJECTED wherever they are shown.
        self.faults: dict[str, float | None] = {"link": None, "gps": None}

    # -- setup ------------------------------------------------------------
    def _load_contacts(self) -> None:
        rows: list[Any] = []
        try:
            from saresq.store.db import Store
            with Store(self.db_path) as store:
                rows = store.all_targets()
        except Exception:
            rows = []
        for r in rows:
            lat, lon = r.get("lat"), r.get("lon")
            if lat is None or lon is None:
                continue
            self.contacts.append(Contact(key=f"T{r['target_id']}", lat=float(lat),
                                         lon=float(lon), target_id=r["target_id"],
                                         from_store=True))
        if not self.contacts:
            # Nothing in the store yet. Place synthetic casualties on a fixed
            # lattice inside the segment so the picture is reproducible between
            # runs and between machines -- a demo that moves is a demo you
            # cannot talk over.
            b = self.plan.box
            for i, (fu, fv) in enumerate(((0.18, 0.22), (0.44, 0.71), (0.63, 0.35),
                                          (0.79, 0.83), (0.31, 0.52), (0.88, 0.14))):
                self.contacts.append(Contact(
                    key=f"S{i+1}",
                    lat=b.lat0 + (b.lat1 - b.lat0) * fv,
                    lon=b.lon0 + (b.lon1 - b.lon0) * fu,
                ))
        self._contacts_loaded = True

    # -- fault injection --------------------------------------------------
    def _fault_active(self, name: str, t: float) -> bool:
        until = self.faults.get(name)
        if until is None:
            return False
        if t >= until:
            self.faults[name] = None
            self._event(t, "fault", f"{name.upper()} fault cleared", "good")
            return False
        return True

    def inject(self, name: str, seconds: float = 30.0) -> dict[str, Any]:
        """Force a failure for `seconds`, or clear it if already active.

        Two genuinely different failures, which the display must not conflate:
        `link` loses the radio, so the aircraft keeps flying and keeps finding
        people but nothing reaches the ground -- the track coasts and alerts
        queue. `gps` keeps the radio, so telemetry still arrives, but there is
        no position in it -- the track coasts while the link reads UP.
        """
        with self._lock:
            now = self._clock()
            self.step_to(now)
            if name not in self.faults:
                raise KeyError(name)
            if self.faults[name] is not None:
                self.faults[name] = None
                self._event(now, "fault", f"{name.upper()} fault cleared by operator", "good")
            else:
                self.faults[name] = now + max(1.0, seconds)
                self._event(now, "fault",
                            f"INJECTED {name.upper()} FAULT for {seconds:.0f} s", "bad")
            return {"fault": name, "until": self.faults[name], "now": now}

    def _event(self, t: float, kind: str, text: str, sev: str = "info") -> None:
        # Kept sorted rather than appended blind: operator actions are stamped
        # with wall-clock time while scans run on their own clock, so an
        # injected fault can otherwise appear below the scan that reacted to it.
        rec = {"t": round(t - self.t0, 1), "kind": kind, "text": text, "sev": sev}
        i = len(self.events)
        while i > 0 and self.events[i - 1]["t"] > rec["t"]:
            i -= 1
        self.events.insert(i, rec)
        del self.events[:-EVENT_LOG_LEN]

    # -- coverage ---------------------------------------------------------
    def _mark_coverage(self, lat: float, lon: float, course: float) -> None:
        b = self.plan.box
        half = swath_m() / 2.0
        e_lat, e_lon = (b.lat1 - b.lat0) / COVERAGE_NY, (b.lon1 - b.lon0) / COVERAGE_NX
        for j in range(COVERAGE_NY):
            clat = b.lat0 + (j + 0.5) * e_lat
            if abs(clat - lat) > half / 110000.0 * 3:
                continue
            for i in range(COVERAGE_NX):
                clon = b.lon0 + (i + 0.5) * e_lon
                across, along = lateral_offset_m(lat, lon, course, clat, clon)
                if abs(along) > self.plan.speed_ms * self.scan_period_s or abs(across) > half:
                    continue
                k = j * COVERAGE_NX + i
                if self.coverage[k] < 9:
                    self.coverage[k] += 1

    # -- the scan ---------------------------------------------------------
    def _scan_once(self, t: float) -> None:
        self.scan += 1
        s = (t - self.t0) * self.plan.speed_ms
        lat, lon, course = self.plan.at(s)
        rssi, detail = self.link.rssi_dbm(lat, lon, SURVEY_ALT_M)
        state = self.link.state(rssi)
        injected = self._fault_active("link", t)
        if injected:
            state, rssi = "DOWN", -120.0
            detail = dict(detail, blocker="INJECTED FAULT", injected=True)
        gps_denied = self._fault_active("gps", t)
        detail["gps_denied"] = gps_denied
        self.last_rssi, self.last_detail = rssi, detail

        if state != self.link_state:
            was, self.link_state, self.link_since = self.link_state, state, t
            if state == "DOWN":
                self._event(t, "link", f"LINK LOST — {detail.get('blocker') or 'obstruction'}"
                                       f" at {detail['range_m']:.0f} m, {rssi:.0f} dBm",
                            "bad")
            elif was == "DOWN":
                self._event(t, "link", f"LINK RESTORED — {rssi:.0f} dBm, "
                                       f"{len(self.queue)} alert(s) to flush", "good")
            else:
                self._event(t, "link", f"link {state.lower()} — {rssi:.0f} dBm",
                            "warn" if state == "DEGRADED" else "good")

        self._mark_coverage(lat, lon, course)

        # -- the aircraft's own track ------------------------------------
        # No downlink, no plot. This is the whole point: the track does not
        # vanish, it coasts on its last velocity and says so.
        plots: list[Plot] = []
        if state != "DOWN" and not gps_denied:
            j = (_det_roll("gps", self.scan) - 0.5) * 2.0
            k = (_det_roll("gps2", self.scan) - 0.5) * 2.0
            plots.append(Plot(
                lat=lat + j * 2.1 / 110574.0, lon=lon + k * 2.1 / 102796.0,
                alt_m=SURVEY_ALT_M + j * 0.4, sigma_m=2.1, ident="SQ-01",
                payload={"label": "SQ-01", "role": "platform"},
            ))
        # Snapshot coasts as well as state: apply_plot resets the counter, so
        # reading it afterwards always reports a zero-second coast.
        before = {tr.number: (tr.state.value, tr.coasts) for tr in self.platform.tracks.values()}
        self.platform.update(t, plots)
        for tr in self.platform.tracks.values():
            was, was_coasts = before.get(tr.number, (None, 0))
            if was == tr.state.value:
                continue
            if tr.state.value == "COAST":
                why = "no position in telemetry" if gps_denied else "no downlink"
                self._event(t, "track", f"{tr.label} COASTING — {why}, dead reckoning "
                                        f"from {tr.filt.speed_ms:.1f} m/s", "warn")
            elif was == "COAST":
                self._event(t, "track", f"{tr.label} reacquired after "
                                        f"{was_coasts * self.scan_period_s:.0f} s coast, "
                                        f"{tr.filt.sigma_pred_m(0.0):.1f} m residual", "good")
            elif was is None:
                self._event(t, "track", f"{tr.label} track initiated (tentative)", "info")

        # -- looking for people -------------------------------------------
        new_plots: list[Plot] = []
        for c in self.contacts:
            across, along = lateral_offset_m(lat, lon, course, c.lat, c.lon)
            if abs(along) > self.plan.speed_ms * self.scan_period_s / 2.0:
                continue
            p = detection_probability(abs(across), self.sweep_width_m)
            if _det_roll(c.key, self.scan) >= p:
                continue
            c.detections.append(p)
            c.last_seen_t = t
            if c.first_seen_t is None:
                c.first_seen_t = t
            rec = self._encode(c, t)
            # Tier-1 alerts are 28 bytes and survive a degraded channel that
            # would stall imagery, so DEGRADED still delivers them.
            if state == "DOWN":
                self.queue.append(rec)
                self._event(t, "alert", f"{c.key} detected, p={c.p_cum:.2f} — "
                                        f"HELD ON AIRCRAFT, no link", "warn")
            else:
                new_plots.append(self._plot_for(rec))
                self.delivered.append(rec)
                self.bytes_delivered += ALERT_BYTES
                self._event(t, "alert",
                            f"{c.key} {classify(c.p_cum)} p={c.p_cum:.2f} "
                            f"({icd203(c.p_cum)}) — {across:+.0f} m abeam", "good")

        # -- flush anything the link was holding ---------------------------
        if state != "DOWN" and self.queue:
            flushed, self.queue = self.queue, []
            for rec in flushed:
                new_plots.append(self._plot_for(rec))
                self.delivered.append(rec)
                self.bytes_delivered += ALERT_BYTES
            self._event(t, "sync", f"flushed {len(flushed)} queued alert(s), "
                                   f"{len(flushed) * ALERT_BYTES} B", "good")

        known = {tr.ident for tr in self.contacts_tracker.tracks.values()}
        self.contacts_tracker.update(t, new_plots)
        for tr in self.contacts_tracker.tracks.values():
            if tr.ident not in known:
                self._event(t, "track", f"contact track {tr.label} initiated at "
                                        f"{tr.lat:.5f}N {tr.lon:.5f}E", "info")

    def _encode(self, c: Contact, t: float) -> dict[str, Any]:
        """Build the real Tier-1 packet, not a description of one.

        The bytes that go in the queue here are the same bytes `saresq.sync`
        would put on the radio, so anything the display says about backlog size
        or flush time is measured rather than asserted.
        """
        p = c.p_cum
        alert = Alert(
            target_id=c.target_id if c.target_id is not None else abs(hash(c.key)) % 100000,
            t_s=int(t), lat=c.lat, lon=c.lon, p_final=p, class_=classify(p),
            decision="CONFIRM" if p >= 0.75 else "REOBSERVE_LOWER",
            n_passes=len(c.detections), alt_m=SURVEY_ALT_M,
        )
        return {"key": c.key, "t": t, "wire": pack_alert(alert), "target_id": c.target_id}

    def _plot_for(self, rec: dict[str, Any]) -> Plot:
        # Decode rather than carry the values alongside: if the wire format
        # ever loses a digit of precision, the map is where it shows up.
        a = unpack_alert(rec["wire"])
        return Plot(
            lat=a.lat, lon=a.lon, sigma_m=3.5, ident=rec["key"],
            payload={
                "label": rec["key"], "role": "contact", "p": round(a.p_final, 3),
                "class": a.class_, "likelihood": icd203(a.p_final),
                "decision": a.decision, "target_id": rec["target_id"],
                "n_looks": a.n_passes, "wire_bytes": len(rec["wire"]),
            },
        )

    # -- public -----------------------------------------------------------
    def step_to(self, now: float | None = None) -> None:
        with self._lock:
            if not self._contacts_loaded:
                self._load_contacts()
            now = self._clock() if now is None else now
            due = int((now - self.t0) / self.scan_period_s) - self.scan + 1
            if due > MAX_CATCHUP_SCANS:
                # Too far behind to replay honestly (a tab left open overnight).
                # Treat the gap as the sim having been paused: move t0 forward
                # AND move every existing track and contact by the same amount,
                # so the whole picture stays on one clock. Shifting t0 alone
                # leaves the platform -- which is re-plotted on the next scan --
                # and the contacts -- which are not -- on different timelines,
                # and the contacts then read hours stale on a picture that is
                # minutes old.
                delta = (now - self.scan * self.scan_period_s) - self.t0
                self.t0 += delta
                self.platform.shift(delta)
                self.contacts_tracker.shift(delta)
                for c in self.contacts:
                    if c.first_seen_t is not None:
                        c.first_seen_t += delta
                    if c.last_seen_t is not None:
                        c.last_seen_t += delta
                self.link_since += delta
                for name, until in self.faults.items():
                    if until is not None:
                        self.faults[name] = until + delta
                self._event(now, "sys",
                            f"clock re-based, {delta / 60:.0f} min gap skipped", "warn")
                due = 1
            for _ in range(max(0, due)):
                self._scan_once(self.t0 + self.scan * self.scan_period_s + self.scan_period_s)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        self.step_to(now)
        with self._lock:
            now = self._clock() if now is None else now
            plat = self.platform.snapshot(now)
            cons = self.contacts_tracker.snapshot(now)
            covered = sum(1 for v in self.coverage if v)
            sweeps = self.plan.speed_ms * (now - self.t0) / max(self.plan.length_m, 1.0)
            C = self.plan.coverage_c(sweeps, self.sweep_width_m)
            b = self.plan.box
            return {
                "t": round(now - self.t0, 2),
                "wall": now,
                "scan": self.scan,
                "scan_period_s": self.scan_period_s,
                "simulated": True,
                "source": "store" if any(c.from_store for c in self.contacts) else "synthetic",
                "link": {
                    "state": self.link_state,
                    "rssi_dbm": round(self.last_rssi, 1),
                    "since_s": round(now - self.link_since, 1),
                    "queued": len(self.queue),
                    "queued_bytes": len(self.queue) * ALERT_BYTES,
                    "delivered": len(self.delivered),
                    "delivered_bytes": self.bytes_delivered,
                    "up_dbm": -95.0, "down_dbm": -105.0,
                    **self.last_detail,
                },
                "faults": {k: (None if v is None else round(max(0.0, v - now), 1))
                           for k, v in self.faults.items()},
                "platform": plat[0] if plat else None,
                "contacts": cons,
                "events": list(reversed(self.events[-40:])),
                "search": {
                    "sweeps": round(sweeps, 2),
                    "coverage_c": round(C, 3),
                    "pod": round(1.0 - math.exp(-C), 4),
                    "cells_seen": covered,
                    "cells_total": len(self.coverage),
                    "fraction": round(covered / len(self.coverage), 4),
                    "sweep_width_m": self.sweep_width_m,
                    "swath_m": round(swath_m(), 1),
                    "leg_spacing_m": round(leg_spacing_m(), 1),
                    "track_length_m": round(self.plan.length_m, 0),
                    "grid": "".join(str(v) for v in self.coverage),
                    "nx": COVERAGE_NX, "ny": COVERAGE_NY,
                },
                "aoi": {"lat0": b.lat0, "lon0": b.lon0, "lat1": b.lat1, "lon1": b.lon1},
                "gcs": {"lat": GROUND_STATION[0], "lon": GROUND_STATION[1]},
                "obstructions": [
                    {"name": o.name, "lat0": o.box.lat0, "lon0": o.box.lon0,
                     "lat1": o.box.lat1, "lon1": o.box.lon1, "top_m": o.top_m}
                    for o in self.link.obstructions
                ],
                "waypoints": [[round(a, 6), round(b_, 6)] for a, b_ in self.plan.waypoints],
            }
