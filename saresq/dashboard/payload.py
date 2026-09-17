"""Bridge the live payload (tools/live_pipeline.py, port 8091) to the dashboard.

The payload and the ground station were built as two halves that had never been
joined: the payload serves its own page on 8091 and keeps captures in a RAM ring
buffer, the dashboard reads saresq.db, and nothing wrote from one to the other.
This is the join, and it lives on the GROUND side on purpose -- it polls the
payload's existing HTTP endpoints rather than adding a writer to the aircraft:

  * the Pi needs no redeploy, and nothing new can crash the demo mid-flight
  * it works with the payload on another machine, which is the real topology
  * a dropped link degrades to "last seen 8 s ago" instead of losing captures

Two rates, because they answer different questions. /stats at ~1 Hz is "where is
the aircraft now" and drives the live marker. /events every few seconds is "what
did it find", and those are durable, so they go into the store.

POSITION WITHOUT A FIX
A NEO-6M indoors never gets a fix, so bench captures carry lat=None -- and a
find with no position is useless to a rescuer. Rather than invent coordinates,
an operator can set a manual datum (set_origin) and unfixed events are placed
there with pos_source="manual", which the UI must show differently from "gps".
Inventing a position and labelling it like a measurement is the one thing this
must never do.
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request

#: Two captures closer than this are the same physical target seen twice, not
#: two finds. 12 m is a little over the geolocation CEP the deck quotes, so it
#: merges genuine re-sightings without swallowing two people in one room.
MERGE_RADIUS_M = 12.0


def _haversine_m(a_lat, a_lon, b_lat, b_lon) -> float:
    R = 6371000.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _klass(conf: float) -> str:
    """Confidence -> the HIGH/MEDIUM/LOW band the rest of the stack speaks."""
    if conf >= 0.60:
        return "HIGH"
    if conf >= 0.35:
        return "MEDIUM"
    return "LOW"


def _decision(conf: float, has_fix: bool) -> str:
    if conf >= 0.60 and has_fix:
        return "CONFIRM"
    if conf >= 0.35:
        return "REOBSERVE_LOWER"
    return "LOG_AND_RESUME"


class PayloadLink(threading.Thread):
    """Polls one payload and mirrors what it finds into the store."""

    daemon = True

    def __init__(self, host: str, store_factory, media_store=None,
                 stats_hz: float = 1.0, events_s: float = 3.0):
        super().__init__(name="payload-link")
        self.host = host if ":" in host else f"{host}:8091"
        self._store_factory = store_factory
        self._media = media_store
        self._stats_period = 1.0 / max(stats_hz, 0.1)
        self._events_period = max(events_s, 1.0)
        self._lock = threading.Lock()
        self._state: dict = {"connected": False, "why": "not started"}
        self._seen: set[tuple[str, int]] = set()   # (session, event id) already stored
        self._origin: tuple[float, float] | None = None
        self._stop = threading.Event()
        self._ingested = 0
        self._last_ok = 0.0

    # ---- operator-set datum -------------------------------------------------
    def set_origin(self, lat: float, lon: float) -> None:
        with self._lock:
            self._origin = (float(lat), float(lon))

    @property
    def origin(self):
        with self._lock:
            return self._origin

    def stop(self) -> None:
        self._stop.set()

    # ---- what the UI reads --------------------------------------------------
    def live(self) -> dict:
        with self._lock:
            s = dict(self._state)
        s["host"] = self.host
        s["ingested"] = self._ingested
        s["age_s"] = (time.time() - self._last_ok) if self._last_ok else None
        o = self.origin
        s["origin"] = {"lat": o[0], "lon": o[1]} if o else None
        return s

    # ---- polling ------------------------------------------------------------
    def _get(self, path: str, timeout: float = 4.0):
        url = f"http://{self.host}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def run(self) -> None:
        next_events = 0.0
        while not self._stop.is_set():
            t0 = time.time()
            try:
                st = self._get("/stats")
                self._last_ok = time.time()
                with self._lock:
                    self._state = self._shape(st)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
                with self._lock:
                    # Keep the last good reading visible and mark it stale rather
                    # than blanking the panel: "last seen 8 s ago" is information,
                    # an empty panel is not.
                    self._state["connected"] = False
                    self._state["why"] = f"{type(e).__name__}"
            if time.time() >= next_events:
                next_events = time.time() + self._events_period
                try:
                    self._ingest(self._get("/events"))
                except (urllib.error.URLError, OSError, ValueError, TimeoutError):
                    pass
            self._stop.wait(max(0.05, self._stats_period - (time.time() - t0)))

    @staticmethod
    def _shape(st: dict) -> dict:
        g = st.get("gate", {}) or {}
        d = st.get("detect", {}) or {}
        gps = st.get("gps", {}) or {}
        hz = st.get("hazard", {}) or {}
        th = st.get("thermal", {}) or {}
        n = int(d.get("n", 0) or 0)
        return {
            "connected": True,
            "verdict": ("PERSON" if n else ("HEAT" if g.get("fired") else "CLEAR")),
            "n": n,
            "det_ms": d.get("ms"),
            "model": d.get("model"),
            "z_max": g.get("z_max"), "z_t": g.get("z_t"),
            "blobs": g.get("n_blobs", 0), "fired": bool(g.get("fired")),
            "t_min": th.get("min"), "t_max": th.get("max"),
            "t_spread": th.get("spread"), "t_fps": th.get("fps"),
            "scene": hz.get("top") if hz.get("ok") else None,
            "scene_p": hz.get("p") if hz.get("ok") else None,
            "lat": gps.get("lat"), "lon": gps.get("lon"),
            "sats": gps.get("sats"), "in_view": gps.get("in_view"),
            "fix": bool(gps.get("quality")), "gps_state": gps.get("state"),
            "hdop": gps.get("hdop"), "alt": gps.get("alt"),
            "temp": st.get("temp"), "up": st.get("up"),
        }

    # ---- durable half -------------------------------------------------------
    def _ingest(self, events) -> None:
        if not isinstance(events, list):
            return
        store = self._store_factory()
        try:
            targets = store.all_targets()
            for ev in sorted(events, key=lambda e: e.get("id", 0)):
                key = (str(ev.get("s", "")), int(ev.get("id", 0)))
                if key in self._seen:
                    continue
                self._seen.add(key)
                self._store_event(store, ev, targets)
        finally:
            try:
                store.close()
            except Exception:                              # noqa: BLE001
                pass

    def _store_event(self, store, ev: dict, targets: list) -> None:
        lat, lon = ev.get("lat"), ev.get("lon")
        src = "gps"
        if lat is None or lon is None:
            o = self.origin
            if o is None:
                # No fix and no operator datum: it is still a real find, so it
                # must reach the review queue -- but it gets no coordinates and
                # therefore no map pin. Better absent than fabricated.
                lat = lon = None
                src = "none"
            else:
                lat, lon, src = o[0], o[1], "manual"

        conf = float(ev.get("conf", 0.0) or 0.0)
        t_ns = int(time.time() * 1e9)
        has_fix = src == "gps"

        tid = None
        if lat is not None:
            for t in targets:
                if t.get("lat") is None:
                    continue
                if _haversine_m(lat, lon, t["lat"], t["lon"]) <= MERGE_RADIUS_M:
                    tid = t["target_id"]
                    break

        if tid is None:
            tid = store.insert_target(
                first_seen_ns=t_ns, last_seen_ns=t_ns, lat=lat, lon=lon,
                pos_err_m=(2.5 if has_fix else 25.0),
                p_final=conf, **{"class": _klass(conf)},
                n_passes=1, decision=_decision(conf, has_fix), thumb_path=None)
            targets.append({"target_id": tid, "lat": lat, "lon": lon})
        else:
            prev = store.get_target(tid) or {}
            n = int(prev.get("n_passes") or 0) + 1
            p = max(float(prev.get("p_final") or 0.0), conf)
            store.update_target(tid, last_seen_ns=t_ns, n_passes=n, p_final=p,
                                **{"class": _klass(p)},
                                decision=_decision(p, has_fix))

        store.insert_pass(
            target_id=tid, t_start_ns=t_ns, t_end_ns=t_ns, alt_band=0,
            alt_m=None, speed_mps=None, p_pass=conf, lr=None, weight=1.0,
            p_rgb_max=conf, z_peak_max=float(ev.get("z", 0.0) or 0.0),
            iou_max=None, hits=int(ev.get("n", 0) or 0),
            t_bg=None, t_ambient=float(ev.get("tmax", 0.0) or 0.0), lum=None,
            p_flood=None, p_fire=None, p_collapse=None)
        self._ingested += 1
