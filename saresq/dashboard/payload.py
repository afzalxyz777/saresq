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
import pathlib
import threading
import time
import urllib.error
import urllib.request

from saresq.dashboard.contacts import project_contacts
from saresq.fuse.verdict import phrase as _verdict_phrase
from saresq.fuse.verdict import verdict as _verdict

#: Two captures closer than this are the same physical target seen twice, not
#: two finds. 12 m is a little over the geolocation CEP the deck quotes, so it
#: merges genuine re-sightings without swallowing two people in one room.
MERGE_RADIUS_M = 12.0

#: How long a gap ends a run of captures that carry no position. Long enough to
#: span the payload's own 6 s capture rate several times over, short enough that
#: two separate sightings minutes apart do not become one target.
UNLOCATED_GAP_NS = 90 * 1_000_000_000

#: How long the payload must be unreachable before the mission is cleared.
#: Not a blip tolerance -- a shutdown detector. The payload polls at 1 Hz, so
#: this is ~45 consecutive failures: far beyond a wifi roam or a pass behind a
#: building, and reached within seconds of somebody switching the payload off.
PAYLOAD_GONE_S = 45.0

#: How stale the ground station's own look at the live frame may be before it
#: stops counting toward the verdict. Two seconds is a few payload frames: long
#: enough to bridge one slow inference, short enough that a frozen second
#: opinion cannot keep asserting a person who has walked out of shot.
GROUND_LOOK_TTL_S = 2.5

#: Two thresholds, because they answer two different questions and the second
#: is the harder claim. PRESENCE decides "is anyone there at all", and feeds
#: only the verdict's has_visual -- a weak box there costs an operator a
#: second look, which in search and rescue is the cheap error. COUNT decides
#: the number actually SHOWN to them, and a number is a claim about how many
#: people need rescuing, so it must be right rather than eager.
#:
#: Calibrated against six live frames with one person standing in shot:
#: every threshold from 0.10 to 0.40 reported two people on one of the six,
#: the phantom box scoring 0.433 against the real detection's 0.60-0.92.
#: 0.50 and above were exact on all six. The real person clears PRESENCE by a
#: wide margin in every frame, so the split costs no sensitivity.
GROUND_PRESENCE_CONF = 0.25
GROUND_COUNT_CONF = 0.50


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

    def __init__(self, host: str, store_factory, media_root=None,
                 stats_hz: float = 1.0, events_s: float = 3.0):
        super().__init__(name="payload-link")
        self.host = host if ":" in host else f"{host}:8091"
        self._store_factory = store_factory
        # A ROOT PATH, not a MediaStore. A MediaStore holds a Store, a Store
        # holds an sqlite3 connection, and an sqlite3 connection may only be
        # used on the thread that opened it -- so one built by the caller on
        # the main thread throws the moment this thread touches it. The store
        # is therefore constructed per batch, beside the one _ingest already
        # opens, and the two share a connection.
        self._media_root = media_root
        self._stats_period = 1.0 / max(stats_hz, 0.1)
        self._events_period = max(events_s, 1.0)
        self._lock = threading.Lock()
        self._state: dict = {"connected": False, "why": "not started"}
        self._seen: set[tuple[str, int]] = set()   # (session, event id) already stored
        #: The payload stamps every event with a token fixed at ITS process
        #: start, so a power cycle changes it. That token -- not the dashboard's
        #: lifetime and not the link dropping -- is what bounds a mission.
        self._session: str | None = None
        self._session_started: float | None = None
        self._origin: tuple[float, float] | None = None
        self._origin_src = "manual"
        self._origin_acc: float | None = None
        #: Height above ground, in metres, for placing live contacts. None
        #: means nobody has said, and the projection falls back to the survey
        #: altitude and declares the assumption. Not derived from GNSS
        #: altitude: that is height above the ellipsoid, and the difference
        #: from the ground the casualty is lying on needs a terrain model this
        #: payload does not carry.
        self._agl_m: float | None = None
        self._stop = threading.Event()
        self._ingested = 0
        self._last_ok = 0.0
        #: The ground station's own look at the SAME live frame, by a model
        #: far larger than the aircraft can carry (yolov8m, 25.9 M parameters,
        #: against the payload's 3.0 M). The payload is compute-bound and
        #: cannot be asked to run this; the laptop is not. Both opinions are
        #: about the same photograph, so the verdict may take the better of
        #: the two rather than being limited by what fits on the aircraft.
        #: None until a look succeeds -- never a default of "saw nobody",
        #: which would be a missing measurement reported as a negative one.
        self._ground_look: dict | None = None
        self._rescorer = None

    # ---- fallback datum -----------------------------------------------------
    def set_origin(self, lat: float, lon: float, source: str = "manual",
                   accuracy_m: float | None = None) -> None:
        """Where to place captures that carry no GPS fix.

        `source` is carried through to the UI because the two available
        fallbacks are not equivalent and must not look it:

          manual  - an operator pointed at the map. As good as their knowledge.
          browser - the laptop's own Wi-Fi/IP geolocation. This is the GROUND
                    STATION's position, not the aircraft's. On a bench where
                    both sit in one room that is a fair stand-in; the moment
                    the drone flies it is not, and the label has to keep saying
                    so rather than quietly becoming wrong.
        """
        with self._lock:
            self._origin = (float(lat), float(lon))
            self._origin_src = source
            self._origin_acc = accuracy_m

    def set_agl(self, agl_m: float | None) -> None:
        """Operator-stated height above ground for the contact projection.

        A payload on a table in a college room is one metre up, not the twenty
        the survey profile assumes, and pinning contacts twenty metres out from
        an aircraft that is sitting still would be a worse lie than admitting
        the number was a default. So it is settable, and what was used is
        always reported back.
        """
        with self._lock:
            self._agl_m = None if agl_m is None else max(0.5, float(agl_m))

    @property
    def agl_m(self) -> float | None:
        with self._lock:
            return self._agl_m

    def attach_rescorer(self, engine) -> None:
        """Give the link the ground station's detector, once it has loaded.

        Loading takes seconds and must not delay the link coming up, so this
        arrives late rather than being a constructor argument.
        """
        self._rescorer = engine

    def _ground_look_now(self):
        """Run the ground station's model over the payload's current frame."""
        engine = self._rescorer
        if engine is None or not getattr(engine, "available", False):
            return
        try:
            import cv2
            import numpy as np
            raw = self._blob("/rgb.jpg")
            if not raw:
                return
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
            res = engine.score([img])[0]
        except Exception:
            return                     # a lost second opinion is never fatal
        confs = res.confs or ()
        with self._lock:
            self._ground_look = {
                # How many to SAY: the strict bar.
                "n": sum(1 for c in confs if c >= GROUND_COUNT_CONF),
                # Whether anyone is there at all: the permissive one.
                "present": bool(res.p >= GROUND_PRESENCE_CONF),
                "p": float(res.p), "ms": float(res.ms), "t": time.time()}

    @property
    def origin(self):
        with self._lock:
            return self._origin

    def stop(self) -> None:
        self._stop.set()

    def _wipe_media(self) -> None:
        """Delete the blobs behind the rows purge_mission just removed.

        Rows without blobs would be broken thumbnails; blobs without rows would
        be an invisible leak that fills the card over a day of flying. Only the
        media root is touched, and only its contents.
        """
        root = self._media_root
        if not root:
            return
        import shutil
        try:
            base = pathlib.Path(root)
            for child in base.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        except OSError as e:                          # noqa: BLE001
            print(f"payload-link: could not clear media root: {e}", flush=True)

    # ---- what the UI reads --------------------------------------------------
    def live(self) -> dict:
        with self._lock:
            s = dict(self._state)
        s["host"] = self.host
        s["ingested"] = self._ingested
        s["age_s"] = (time.time() - self._last_ok) if self._last_ok else None
        o = self.origin
        with self._lock:
            src, acc = self._origin_src, self._origin_acc
        s["origin"] = ({"lat": o[0], "lon": o[1], "source": src, "accuracy_m": acc}
                       if o else None)
        s["session"] = self._session
        s["session_age_s"] = (time.time() - self._session_started
                              if self._session_started else None)

        # ---- live contacts ---------------------------------------------
        # Where the gate's current blobs are ON THE GROUND. Done here rather
        # than in _shape() because it needs the fallback datum, which is the
        # dashboard's to know, not the payload's.
        #
        # A contact is NOT a target: no ledger entry, no fused probability, no
        # pass count. It exists while the blob does. The map draws the two
        # differently and this is why.
        with self._lock:
            agl = self._agl_m
        plat, plon = s.get("lat"), s.get("lon")
        psrc = "gps"
        if plat is None or plon is None:
            if o:
                plat, plon, psrc = o[0], o[1], (src or "manual")
        s["agl_m"] = agl
        s["agl_assumed"] = agl is None
        s["contacts"] = project_contacts(
            s.get("gate_blobs") or [],
            lat=plat, lon=plon, agl_m=agl,
            # No magnetometer, and the NEO-6M reports no course standing
            # still. Left as None so every contact carries the "heading"
            # assumption rather than a fabricated north-up bearing that looks
            # surveyed.
            yaw_deg=None,
            hdop=s.get("hdop"), pos_source=psrc,
        ) if s.get("connected") else []
        return s

    # ---- polling ------------------------------------------------------------
    def _get(self, path: str, timeout: float = 4.0):
        url = f"http://{self.host}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def _blob(self, path: str, timeout: float = 6.0) -> bytes | None:
        try:
            with urllib.request.urlopen(f"http://{self.host}{path}", timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            return None

    def run(self) -> None:
        next_events = 0.0
        while not self._stop.is_set():
            t0 = time.time()
            try:
                st = self._get("/stats")
                self._last_ok = time.time()
                # Look at the same frame the payload just described, before
                # shaping -- so the verdict below sees this poll's opinion and
                # not the previous one's.
                self._ground_look_now()
                with self._lock:
                    self._state = self._shape(st)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
                with self._lock:
                    # Keep the last good reading visible and mark it stale rather
                    # than blanking the panel: "last seen 8 s ago" is information,
                    # an empty panel is not.
                    self._state["connected"] = False
                    self._state["why"] = f"{type(e).__name__}"
                self._purge_if_payload_gone()
            if time.time() >= next_events:
                next_events = time.time() + self._events_period
                try:
                    self._ingest(self._get("/events"))
                except (urllib.error.URLError, OSError, ValueError, TimeoutError):
                    pass
            self._stop.wait(max(0.05, self._stats_period - (time.time() - t0)))

    def _purge_if_payload_gone(self) -> None:
        """Clear the mission once the payload has been off long enough to mean it.

        A mission was previously bounded only by the payload's session token
        changing, which is right when it power-cycles and comes BACK: the new
        token says "different sortie". It said nothing about a payload that is
        switched off and stays off, so yesterday's evidence and review queue
        sat there looking live against a dead link.

        The delay is the whole design. A radio blip, a wifi roam, a few
        dropped polls -- none of those are a new mission, and purging on the
        first failed request would throw away a real mission's evidence every
        time the aircraft flew behind something. PAYLOAD_GONE_S is long enough
        that only a deliberate shutdown reaches it.
        """
        if self._last_ok == 0.0 or self._session is None:
            return                              # nothing to clear yet
        if time.time() - self._last_ok < PAYLOAD_GONE_S:
            return                              # a gap, not a shutdown
        store = self._store_factory()
        try:
            n = store.purge_mission()
            self._wipe_media()
        finally:
            store.close()
        with self._lock:
            self._seen.clear()
            self._session = None
            self._session_started = None
        print(f"payload-link: payload gone for {PAYLOAD_GONE_S:.0f}s -- "
              f"cleared {n} row(s); evidence and review start empty",
              flush=True)

    def _shape(self, st: dict) -> dict:
        g = st.get("gate", {}) or {}
        d = st.get("detect", {}) or {}
        gps = st.get("gps", {}) or {}
        hz = st.get("hazard", {}) or {}
        mv = st.get("motion", {}) or {}
        th = st.get("thermal", {}) or {}
        n_air = int(d.get("n", 0) or 0)
        # BOTH machines looked at this frame. The aircraft ran a 3.0 M-parameter
        # detector because that is what fits inside a 250 ms budget on a Pi; the
        # ground station ran 25.9 M on the same picture because it has a GPU and
        # no such budget. Neither is "the" answer -- the payload's silence at
        # 3.0 M is a statement about its compute, not about the scene.
        #
        # So take the better-informed look, the same MAX rule the review queue
        # already applies to stored crops. A second opinion may promote, never
        # bury: if the ground model sees nobody, the aircraft's own count still
        # stands. This is why /api/rescore reports separately -- the provenance
        # is auditable even though the operator reads one number.
        look = self._ground_look
        n_gnd, gnd_present = 0, False
        if look and (time.time() - look["t"]) <= GROUND_LOOK_TTL_S:
            n_gnd = int(look.get("n", 0) or 0)
            gnd_present = bool(look.get("present"))
        n = max(n_air, n_gnd)
        # Presence can be true while the count is zero: a single box at 0.30 is
        # enough to say somebody is there and not enough to say how many. The
        # verdict takes presence; the number shown takes the count.
        has_visual = bool(n) or gnd_present
        out = {
            "connected": True,
            # Three outcomes, not two. A thermal blob at body temperature with
            # a BLIND camera is not the same as one the detector examined and
            # rejected: the first is the normal night case this payload exists
            # for, the second is a real negative. Collapsing them made the
            # console report "no person confirmed" over two sleeping people in
            # an unlit room, where the crops handed to the detector were black
            # squares at 10-21 of 255 luminance.
            # The ladder lives in saresq/fuse/verdict.py with the reasoning
            # for its ordering. Briefly: a body-temperature source that MOVED
            # outranks a bare visual detection, because a stock COCO detector
            # will box a mannequin, a poster or a corpse, while motion plus
            # body heat is two independent physical measurements agreeing on
            # the thing this mission actually searches for -- a LIVING human.
            # The LADDER asks only "did the camera see a person", so it gets
            # presence. The WORDING below gets `n`, the strict count, so a
            # confirmed sighting never names a number it cannot stand behind.
            "verdict": _verdict(n_visual=(n or int(has_visual)),
                                gate_fired=bool(g.get("fired")),
                                moved=bool(mv.get("moved")),
                                rgb_blind=bool(d.get("rgb_blind"))),
            "rgb_blind": bool(d.get("rgb_blind")),
            "crop_lum": d.get("lum"),
            "motion": bool(mv.get("moved")),
            "verdict_why": None,   # filled below
            "motion_ago_s": mv.get("ago_s"),
            "motion_area": mv.get("area"),
            "motion_peak": mv.get("peak"),
            "n": n,
            # Auditable provenance: one number is displayed, but who saw what
            # is always answerable. n_air is the aircraft alone.
            "n_air": n_air,
            "n_ground": n_gnd,
            "det_ms": d.get("ms"),
            "model": d.get("model"),
            "z_max": g.get("z_max"), "z_t": g.get("z_t"),
            "blobs": g.get("n_blobs", 0), "fired": bool(g.get("fired")),
            # The blobs themselves, not just how many. live() projects these
            # onto the map; capped at the same six the payload sends so a
            # runaway frame cannot flood the poll.
            "gate_blobs": list(g.get("blobs") or [])[:6],
            "t_min": th.get("min"), "t_max": th.get("max"),
            "t_spread": th.get("spread"), "t_fps": th.get("fps"),
            "scene": hz.get("top") if hz.get("ok") else None,
            "scene_p": hz.get("p") if hz.get("ok") else None,
            # The full five-way distribution, not just the winner. The Hazards
            # page needs it to show that fire/flood/collapse were evaluated and
            # came back near zero -- "normal at 0.99" is a measurement, whereas
            # a page showing nothing is indistinguishable from a page that is
            # broken. The payload has always sent this; only the winner was
            # being kept.
            "scene_probs": hz.get("probs") if hz.get("ok") else None,
            "scene_ms": hz.get("ms") if hz.get("ok") else None,
            "scene_why": None if hz.get("ok") else hz.get("why"),
            "lat": gps.get("lat"), "lon": gps.get("lon"),
            "sats": gps.get("sats"), "in_view": gps.get("in_view"),
            "fix": bool(gps.get("quality")), "gps_state": gps.get("state"),
            "hdop": gps.get("hdop"), "alt": gps.get("alt"),
            # Per-crop z and temperature, so the feed page's captions come from
            # the same poll as the count and cannot disagree with the pictures.
            "crops": [{"i": c.get("i", i), "z": c.get("z"), "T": c.get("T")}
                      for i, c in enumerate(st.get("crops") or [])],
            "temp": st.get("temp"), "up": st.get("up"),
        }
        # The reason travels with the verdict. A one-word state an operator
        # cannot expand is a state they will eventually guess at, and guessing
        # is what "no person confirmed" over two sleeping people came from.
        out["verdict_why"] = _verdict_phrase(
            out["verdict"], rgb_blind=out["rgb_blind"], n_visual=n)
        return out

    # ---- durable half -------------------------------------------------------
    def _ingest(self, events) -> None:
        if not isinstance(events, list):
            return
        store = self._store_factory()
        try:
            # A new token means a different sortie. Clear before ingesting, so
            # the first event of the new flight lands in an empty store rather
            # than merging into a target from the last one.
            #
            # Keyed on the TOKEN CHANGING, never on the link dropping: a radio
            # blip is not a new mission, and wiping the map every time the
            # aircraft passes behind a building would destroy the record of the
            # flight in progress.
            seen_tokens = {str(e.get("s", "")) for e in events if e.get("s")}
            if seen_tokens:
                token = sorted(seen_tokens)[-1]
                if self._session is not None and token != self._session:
                    n = store.purge_mission()
                    self._wipe_media()
                    self._seen.clear()
                    self._ingested = 0
                    print(f"payload-link: new payload session {token} "
                          f"(was {self._session}) -- cleared "
                          f"{n.get('targets', 0)} target(s), {n.get('media', 0)} artefact(s)",
                          flush=True)
                if token != self._session:
                    self._session = token
                    self._session_started = time.time()

            # Read AFTER any purge. Reading first left a stale in-memory list
            # pointing at deleted rows, and the next event tried to hang a pass
            # off a target that no longer existed -- a foreign-key failure that
            # silently dropped every capture of the new mission.
            targets = store.all_targets()

            media = None
            if self._media_root:
                from saresq.store.media import MediaStore
                media = MediaStore(self._media_root, store)
            for ev in sorted(events, key=lambda e: e.get("id", 0)):
                key = (str(ev.get("s", "")), int(ev.get("id", 0)))
                if key in self._seen:
                    continue
                self._seen.add(key)
                self._store_event(store, ev, targets, media)
        finally:
            try:
                store.close()
            except Exception:                              # noqa: BLE001
                pass

    def _store_event(self, store, ev: dict, targets: list, media=None) -> None:
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
                with self._lock:
                    src = self._origin_src
                lat, lon = o[0], o[1]

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
        else:
            # No position at all. These used to skip the merge entirely and so
            # produced one target per capture -- a stationary person indoors
            # became twenty-five "finds" in a couple of minutes, which is worse
            # than useless because the count is what an operator triages on.
            #
            # Without coordinates the only evidence two captures are the same
            # subject is that they are CONTINUOUS, so unlocated captures fold
            # into the most recent unlocated target while sightings keep
            # arriving. A real gap starts a new one, because after a quiet
            # minute there is no longer any reason to think it is the same
            # person.
            recent = None
            for t in targets:
                if t.get("lat") is not None:
                    continue
                if recent is None or (t.get("last_seen_ns") or 0) > (recent.get("last_seen_ns") or 0):
                    recent = t
            if recent is not None and (t_ns - (recent.get("last_seen_ns") or 0)) <= UNLOCATED_GAP_NS:
                tid = recent["target_id"]

        if tid is None:
            tid = store.insert_target(
                first_seen_ns=t_ns, last_seen_ns=t_ns, lat=lat, lon=lon,
                pos_err_m=(2.5 if has_fix else 25.0),
                p_final=conf, **{"class": _klass(conf)},
                n_passes=1, decision=_decision(conf, has_fix), thumb_path=None)
            targets.append({"target_id": tid, "lat": lat, "lon": lon,
                            "last_seen_ns": t_ns})
        else:
            prev = store.get_target(tid) or {}
            n = int(prev.get("n_passes") or 0) + 1
            p = max(float(prev.get("p_final") or 0.0), conf)
            store.update_target(tid, last_seen_ns=t_ns, n_passes=n, p_final=p,
                                **{"class": _klass(p)},
                                decision=_decision(p, has_fix))
            for t in targets:
                if t["target_id"] == tid:
                    t["last_seen_ns"] = t_ns
                    break

        # Scene class, stored as the three hazard probabilities the fusion
        # vector expects. The classifier is five-way and only ever reports its
        # top class over the wire, so the winner takes its own confidence and
        # the other two are zero -- which is what "the scene was classified as
        # flooding at 0.94" actually means for those features.
        scene = ev.get("scene")
        sp = float(ev.get("scene_p") or 0.0)
        p_flood = sp if scene == "flooded_areas" else 0.0
        p_fire = sp if scene == "fire" else 0.0
        p_collapse = sp if scene == "collapsed_building" else 0.0

        pass_id = store.insert_pass(
            target_id=tid, t_start_ns=t_ns, t_end_ns=t_ns, alt_band=0,
            alt_m=None, speed_mps=None, p_pass=conf, lr=None, weight=1.0,
            p_rgb_max=conf, z_peak_max=float(ev.get("z", 0.0) or 0.0),
            iou_max=None, hits=int(ev.get("n", 0) or 0),
            t_bg=None, t_ambient=float(ev.get("tmax", 0.0) or 0.0), lum=None,
            p_flood=p_flood, p_fire=p_fire, p_collapse=p_collapse)

        # A hazard row only when the classifier actually saw one. "normal" is a
        # real answer and belongs in the pass features above, but it is not a
        # hazard and must not put a pin on the map.
        if scene and scene != "normal" and sp >= 0.5 and lat is not None:
            store.insert_hazard(t_ns=t_ns, lat=lat, lon=lon,
                                **{"class": scene}, p=sp)

        self._fetch_media(media, ev, tid, pass_id, t_ns, conf)
        self._ingested += 1

    def _fetch_media(self, media, ev: dict, tid: int, pass_id: int,
                     t_ns: int, conf: float) -> None:
        """Pull the frozen artefacts for one event into the media store.

        Done here rather than on the aircraft because the payload keeps events
        in a RAM ring of 24: they are already encoded and already frozen, and
        the only thing missing was somebody fetching them before the ring wraps.

        The crops are the evidence -- the 160 px regions the detector actually
        judged. The detector frame is the context that makes a crop readable.
        The thermal goes in as RAW centi-kelvin rather than the colour-mapped
        picture, so a reviewer a week later has temperatures rather than a
        screenshot of a palette.
        """
        if media is None:
            return
        eid = ev.get("id")
        session = str(ev.get("s", ""))
        try:
            from saresq.store.media import (KIND_RGB_CROP, KIND_THERMAL_PATCH,
                                            KIND_THUMB)

            for i in range(int(ev.get("ncrops", 0) or 0)):
                blob = self._blob(f"/event/{eid}/crop{i}")
                if blob:
                    media.put_bytes(blob, KIND_RGB_CROP, target_id=tid,
                                          pass_id=pass_id, t_ns=t_ns, priority=conf,
                                      synced=True)
            det = self._blob(f"/event/{eid}/detect")
            if det:
                media.put_bytes(det, KIND_THUMB, target_id=tid,
                                      pass_id=pass_id, t_ns=t_ns, priority=conf,
                                      synced=True)
            raw = self._blob(f"/event/{eid}/raw")
            # 32x24 uint16 = 1536 bytes exactly; anything else is not a frame.
            if raw and len(raw) == 32 * 24 * 2:
                media.put_bytes(raw, KIND_THERMAL_PATCH, target_id=tid,
                                      pass_id=pass_id, t_ns=t_ns, priority=conf,
                                      synced=True,
                                      width=32, height=24)
        except Exception as e:                      # noqa: BLE001
            # Evidence is valuable but never worth losing the detection over:
            # the target and its pass are already committed at this point.
            print(f"payload-link: media for event {eid}/{session} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
