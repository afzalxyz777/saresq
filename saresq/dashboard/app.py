"""Flask dashboard (Section 13.2) plus the operator review queue.

Vendored Leaflet (static/leaflet/) means this works with zero internet -- the
point of the offline-resilience demo (Section 13.3). Nothing in the pipeline
waits for this; it's a read-only view onto saresq.db, except for POST /api/verdict,
which appends to `verdicts` and never mutates the machine's belief.

    python -m saresq.dashboard.app --db saresq.db --media results/media
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

from saresq.dashboard.payload import PayloadLink
from saresq.dashboard.readiness import readiness
from saresq.store.db import Store
from saresq.store.media import MediaStore

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["DB_PATH"] = "saresq.db"
app.config["SIMULATE"] = False
app.config["THUMB_DIR"] = "results/thumbs"
app.config["MEDIA_DIR"] = "results/media"
app.config["RESULTS_DIR"] = "results"
app.config["CERT_DIR"] = ".certs"
# Jinja caches templates when debug is off, so an edited page would keep
# serving the old markup until the process restarted. Cheap to reload here,
# and it removes a genuinely confusing failure mode.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

VALID_VERDICTS = {"SURVIVOR", "NOT_SURVIVOR", "UNSURE", "DISPATCHED"}
#: Set by main() when --payload is given. None means "no aircraft configured",
#: which the UI shows as a state of its own rather than as a link failure --
#: the dashboard is fully usable against a recorded database with no payload.
LINK: PayloadLink | None = None

#: The ground station's second-opinion detector. None until --rescore-model
#: loads, and None for good on a machine without torch -- every read of it is
#: guarded, because re-scoring is an enhancement and must never be able to stop
#: the console from running a mission.
RESCORE = None
_MIME = {"jpg": "image/jpeg", "mp4": "video/mp4", "bin": "application/octet-stream"}


def get_store() -> Store:
    return Store(app.config["DB_PATH"])


def _root(key: str) -> pathlib.Path:
    """Absolute path for a configured directory.

    Flask resolves a relative send_file/send_from_directory path against the
    app's root_path (saresq/dashboard/), NOT the working directory -- so
    `--media-dir media` would silently look in the package directory. Always
    resolve against the CWD before handing a path to Flask.
    """
    return pathlib.Path(app.config[key]).expanduser().resolve()


@app.route("/")
def page_overview():
    return render_template("overview.html", page="/")


@app.route("/map")
def page_map():
    return render_template("map.html", page="/map")


@app.route("/radar")
def page_radar():
    return render_template("radar.html", page="/radar")


@app.route("/feed")
def page_feed():
    return render_template("feed.html", active="feed")


#: Frames the payload serves as one complete JPEG per request. The .mjpg
#: variants exist too but are not used: a browser repaints a
#: multipart/x-mixed-replace stream while a frame is still arriving, so the top
#: of the incoming image appears over grey. Fetching discrete JPEGs and drawing
#: only after decode is what fixed that on the payload's own page.
FEED_KINDS = {"thermal": "/thermal.jpg", "rgb": "/rgb.jpg", "detect": "/detect.jpg"}


@app.route("/api/feed/<kind>")
def api_feed(kind: str):
    """Proxy one live frame from the payload.

    Proxied rather than pointed at directly so the ground station stays the
    single origin an operator needs: the browser may be on a network that
    reaches the laptop but not the aircraft, and going direct would also put
    the payload's address into every page.
    """
    import urllib.error
    import urllib.request

    if LINK is None:
        return ("no payload configured", 409)
    path = FEED_KINDS.get(kind)
    if path is None:
        if kind.startswith("crop"):
            path = "/crop/" + kind[4:]
        else:
            abort(404)
    try:
        # Short: three panels poll this, and an unreachable aircraft must fail
        # fast enough that the page keeps ticking rather than stacking up
        # requests that all time out together.
        with urllib.request.urlopen(f"http://{LINK.host}{path}", timeout=2.0) as r:
            blob = r.read()
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        # 503, not 500: the aircraft being out of range is an expected state of
        # the world, and the page shows it as "link down" rather than an error.
        return (f"payload unreachable: {type(e).__name__}", 503)
    resp = app.response_class(blob, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/evidence")
def page_evidence():
    return render_template("evidence.html", page="/evidence")


@app.route("/review")
def page_review():
    return render_template("review.html", page="/review")


@app.route("/analytics")
def page_analytics():
    return render_template("analytics.html", page="/analytics")


@app.route("/hazards")
def page_hazards():
    return render_template("hazards.html", page="/hazards")


@app.route("/api/rescore")
def api_rescore():
    """Second-opinion status, plus what it has been worth so far.

    `delta_mean` is the average of (ground score - payload score) over every
    crop scored by both. It is reported whatever its sign: if the larger model
    is not helping, that is the number that says so.
    """
    with get_store() as store:
        stats = store.rescore_stats()
    if RESCORE is None:
        stats.update({"available": False, "why": "not enabled (--rescore-model)",
                      "running": False})
        return jsonify(stats)
    stats.update(RESCORE.state())
    return jsonify(stats)


# ---------------------------------------------------------------------------
# radar
# ---------------------------------------------------------------------------
_radar_lock = threading.Lock()
_radar: "RadarService | None" = None


def get_radar():
    """One surveillance picture per process, created on first use.

    Deliberately a module-level singleton rather than per-request: a tracker is
    stateful by definition -- rebuilding it on every poll would reset every
    track's history and it would never coast, which is the one behaviour the
    page exists to show.
    """
    global _radar
    with _radar_lock:
        if _radar is None:
            from saresq.surveillance.service import RadarService
            _radar = RadarService(db_path=app.config["DB_PATH"],
                                  allow_synthetic=bool(app.config.get("SIMULATE")))
        return _radar


@app.route("/api/radar")
def api_radar():
    """The surveillance picture, with the real aircraft in it when one is flying.

    The fix is pushed here rather than in the link's own thread so the radar
    stays a pure consumer: it never reaches out to the payload, and the whole
    service still runs standalone against a recorded store.
    """
    r = get_radar()
    if LINK is not None:
        st = LINK.live()
        if st.get("connected") and st.get("lat") is not None and st.get("fix"):
            r.set_external_fix(st["lat"], st["lon"])
    return jsonify(r.snapshot())


@app.route("/api/radar/fault", methods=["POST"])
def api_radar_fault():
    """Inject or clear a link/GPS failure so the coast behaviour is
    demonstrable on demand rather than waited for."""
    body = request.get_json(silent=True) or {}
    name = body.get("fault")
    try:
        seconds = float(body.get("seconds", 30))
    except (TypeError, ValueError):
        abort(400, "seconds must be a number")
    try:
        return jsonify(get_radar().inject(str(name), seconds))
    except KeyError:
        abort(400, "fault must be one of ['gps', 'link']")


@app.route("/api/live")
def api_live():
    """Live payload state: position, verdict, scene, link health.

    Always 200 with a `configured` flag rather than 404 when there is no
    payload. The map polls this every second; a 404 would fill the console with
    errors for the entirely normal case of reviewing a recorded mission.
    """
    if LINK is None:
        return jsonify({"configured": False, "connected": False})
    d = LINK.live()
    d["configured"] = True
    return jsonify(d)


@app.route("/api/origin", methods=["POST"])
def api_origin():
    """Operator-set datum for captures that have no GPS fix.

    Indoors a NEO-6M never fixes, so bench captures carry no position. This
    lets the operator say where the payload actually is. Everything placed this
    way is tagged pos_source="manual" and must be drawn differently from a
    measured fix -- the point is to be useful without ever dressing an
    assumption up as a measurement.
    """
    if LINK is None:
        return jsonify({"error": "no payload configured"}), 409
    body = request.get_json(silent=True) or {}
    try:
        lat, lon = float(body["lat"]), float(body["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "lat and lon required"}), 400
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "out of range"}), 400
    src = str(body.get("source") or "manual")
    if src not in ("manual", "browser"):
        src = "manual"
    acc = body.get("accuracy_m")
    LINK.set_origin(lat, lon, source=src,
                    accuracy_m=float(acc) if isinstance(acc, (int, float)) else None)
    return jsonify({"ok": True, "origin": {"lat": lat, "lon": lon, "source": src}})


@app.route("/api/agl", methods=["POST"])
def api_agl():
    """Height above ground used to place live contacts.

    The gate's blobs are projected to the ground with one angle and one
    height. The angle is measured; the height is not -- GNSS altitude is
    height above the ellipsoid, and turning that into height above the rubble
    someone is lying on needs a terrain model this payload does not carry. So
    the operator states it, and the map says whose number it is.

    POST {"agl_m": 2.0}   set it
    POST {"agl_m": null}  clear it, back to the survey altitude as an
                          explicitly-flagged assumption
    """
    if LINK is None:
        return jsonify({"error": "no payload configured"}), 409
    body = request.get_json(silent=True) or {}
    v = body.get("agl_m")
    if v is None:
        LINK.set_agl(None)
        return jsonify({"ok": True, "agl_m": None, "assumed": True})
    try:
        agl = float(v)
    except (TypeError, ValueError):
        return jsonify({"error": "agl_m must be a number or null"}), 400
    if not (0.5 <= agl <= 500.0):
        return jsonify({"error": "agl_m out of range (0.5-500 m)"}), 400
    LINK.set_agl(agl)
    return jsonify({"ok": True, "agl_m": LINK.agl_m, "assumed": False})


@app.route("/api/readiness")
def api_readiness():
    return jsonify(readiness(app.config["DB_PATH"],
                             app.config["RESULTS_DIR"], app.config["MEDIA_DIR"]))


@app.route("/api/analytics")
def api_analytics():
    """Training metrics, if a Colab run has dropped them into results/.

    Ultralytics writes a results.csv whose last row is the final epoch; we read
    that rather than asking the user to transcribe numbers into a slide.
    """
    import csv as _csv

    res = _root("RESULTS_DIR")
    out = {"detector": None, "hazard": None, "sim": None}

    # WHICH run to report is not a free choice. Globbing and taking the first
    # match sorts alphabetically, which silently promoted a REJECTED candidate
    # (v11n_p3_hituav) over the model we actually fly (v8n_p3_thermalmix2) the
    # moment that directory existed -- the console would have shown a judge
    # 0.6305 for a model that is not on the aircraft.
    #
    # So derive it from a real artefact instead of a name: tflite_exports.json
    # records the weights each deployed .tflite was built from, and the model
    # on the Pi is by definition the one that was exported. If that file is
    # missing we fall back to the old scan, but the response always says which
    # run it read so the number can never be anonymous.
    preferred: list[pathlib.Path] = []
    try:
        exports = json.loads((res / "detector" / "tflite_exports.json").read_text())
        for e in exports.get("exports", []):
            src = e.get("source_weights")
            if not src:
                continue
            run = pathlib.Path(src).parent.parent / "results.csv"
            if run.exists() and run not in preferred:
                preferred.append(run)
    except Exception:
        pass

    for cand in preferred + sorted(res.glob("detector/**/results.csv")) + sorted(res.glob("**/results.csv")):
        try:
            with open(cand, newline="") as fh:
                rows = [r for r in _csv.DictReader(fh)]
            if not rows:
                continue
            last = {k.strip(): v for k, v in rows[-1].items() if k}
            out["detector"] = {"file": cand.name, "run": cand.parent.name,
                               "shipped": bool(preferred and cand == preferred[0]),
                               "epochs": len(rows), "final": last}
            break
        except Exception:
            continue
    for cand in sorted(res.glob("**/hazard_classifier_report.md")):
        try:
            out["hazard"] = {"file": cand.name, "text": cand.read_text()[:4000]}
        except Exception:
            pass
        break
    for cand in sorted(res.glob("sim/**/*.csv")) + sorted(res.glob("**/policy_comparison.csv")):
        try:
            with open(cand, newline="") as fh:
                out["sim"] = {"file": cand.name, "rows": [r for r in _csv.DictReader(fh)][:20]}
        except Exception:
            pass
        break
    return jsonify(out)


#: Hard freshness window for the operator-facing pages. Evidence and Review
#: show what the payload is seeing NOW: a crop older than this is from a
#: moment that has passed, and on a live search that is worse than showing
#: nothing, because it puts a survivor on screen who was found somewhere else
#: 10 minutes ago. Everything remains in the database and on disk -- this
#: filters the VIEW, it does not delete evidence.
FRESH_WINDOW_S = 40.0


def _fresh_cutoff_ns() -> int:
    return int((time.time() - FRESH_WINDOW_S) * 1e9)


@app.route("/api/media")
def api_media():
    """All evidence rows, newest first, optionally filtered by kind or target."""
    kind = request.args.get("kind")
    target = request.args.get("target_id", type=int)
    sql = "SELECT * FROM media WHERE t_ns >= ?"
    args = [_fresh_cutoff_ns()]
    if kind:
        sql += " AND kind = ?"; args.append(kind)
    if target is not None:
        sql += " AND target_id = ?"; args.append(target)
    sql += " ORDER BY t_ns DESC, media_id DESC LIMIT 500"
    with get_store() as store:
        return jsonify([dict(r) for r in store.conn.execute(sql, tuple(args)).fetchall()])


@app.route("/api/targets")
def api_targets():
    with get_store() as store:
        out = []
        for t in store.all_targets():
            t = dict(t)
            # The ground station's best second opinion on this target, attached
            # alongside the aircraft's own number rather than replacing it. The
            # UI shows both; a reviewer should always be able to see where a
            # confidence came from.
            best = store.best_rescore(t["target_id"])
            t["rescore"] = ({"p": best["p"], "n": best["n"], "model": best["model"],
                             "imgsz": best["imgsz"], "p_payload": best["p_payload"]}
                            if best else None)
            out.append(t)
        return jsonify(out)


@app.route("/api/targets/<int:target_id>/passes")
def api_target_passes(target_id: int):
    with get_store() as store:
        return jsonify(store.get_passes_for_target(target_id))


@app.route("/api/hazards")
def api_hazards():
    with get_store() as store:
        return jsonify(store.all_hazards())


# ---------------------------------------------------------------------------
# review queue
# ---------------------------------------------------------------------------
@app.route("/api/review/queue")
def api_review_queue():
    """Unjudged targets, most-confident first, each with its evidence."""
    cutoff = _fresh_cutoff_ns()
    with get_store() as store:
        out = []
        for t in store.review_queue():
            t = dict(t)
            # Same 40 s window as Evidence. A target last seen before it is no
            # longer what the payload is looking at, and a queue that keeps
            # them accumulates a backlog of places the aircraft has already
            # flown past. The row stays in the database for the mission record.
            if (t.get("last_seen_ns") or 0) < cutoff:
                continue
            t["media"] = [m for m in store.media_for_target(t["target_id"])
                          if (dict(m).get("t_ns") or 0) >= cutoff]
            t["passes"] = store.get_passes_for_target(t["target_id"])
            best = store.best_rescore(t["target_id"])
            t["rescore"] = ({"p": best["p"], "n": best["n"], "model": best["model"],
                             "imgsz": best["imgsz"], "p_payload": best["p_payload"]}
                            if best else None)
            out.append(t)
        return jsonify(out)


@app.route("/api/review/stats")
def api_review_stats():
    """How the machine and the operator agree -- the safety audit, and the
    number worth putting on a slide."""
    with get_store() as store:
        verdicts = store.all_verdicts()
        pending = len(store.review_queue())
        agree = disagree = 0
        for v in verdicts:
            if v["verdict"] == "UNSURE":
                continue
            machine_says_yes = (v["decision_at_verdict"] == "CONFIRM")
            human_says_yes = v["verdict"] in ("SURVIVOR", "DISPATCHED")
            if machine_says_yes == human_says_yes:
                agree += 1
            else:
                disagree += 1
        return jsonify({
            "pending": pending, "judged": len(verdicts),
            "agree": agree, "disagree": disagree,
            "agreement": (agree / (agree + disagree)) if (agree + disagree) else None,
        })


@app.route("/api/verdict", methods=["POST"])
def api_verdict():
    body = request.get_json(silent=True) or {}
    target_id = body.get("target_id")
    verdict = body.get("verdict")
    if not isinstance(target_id, int):
        abort(400, "target_id must be an integer")
    if verdict not in VALID_VERDICTS:
        abort(400, f"verdict must be one of {sorted(VALID_VERDICTS)}")

    with get_store() as store:
        target = store.get_target(target_id)
        if target is None:
            abort(404, f"no target {target_id}")
        feats = store.latest_features_for_target(target_id)
        vid = store.insert_verdict(
            target_id=target_id,
            verdict=verdict,
            operator=str(body.get("operator") or "unknown"),
            t_ns=time.time_ns(),
            # Snapshot the machine's belief so the two can always be compared.
            p_final_at_verdict=target["p_final"],
            decision_at_verdict=target["decision"],
            note=body.get("note"),
            features_json=json.dumps(feats) if feats is not None else None,
        )
        return jsonify({"verdict_id": vid, "target_id": target_id, "verdict": verdict})


@app.route("/api/verdicts")
def api_verdicts():
    with get_store() as store:
        return jsonify(store.all_verdicts())


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------
#: media_id is `INTEGER PRIMARY KEY` with no AUTOINCREMENT, so SQLite restarts
#: it at 1 after purge_mission() empties the table. /media/7 is therefore a
#: DIFFERENT crop every mission, and a browser that cached the last one shows
#: an operator a survivor from a flight that ended hours ago. Nothing about the
#: data was stale; the URL was reused. Blobs are LAN-local and small, so
#: refetching them costs nothing next to that.
def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.direct_passthrough = False
    resp.headers.pop("ETag", None)
    resp.headers.pop("Last-Modified", None)
    return resp


@app.route("/media/<int:media_id>")
def media_blob(media_id: int):
    with get_store() as store:
        row = store.get_media(media_id)
        if row is None:
            abort(404)
        path = _root("MEDIA_DIR") / row["rel_path"]
        if not path.exists():
            abort(404)
        return _no_store(send_file(
            path,
            mimetype=_MIME.get(path.suffix.lstrip("."), "application/octet-stream")))


@app.route("/media/<int:media_id>/render")
def media_render(media_id: int):
    """Colour-map a raw thermal patch to PNG so an operator can actually see it.

    Rendered on demand rather than stored: the raw uint16 kelvin is the record,
    and any palette choice is a display decision we may want to change later.
    """
    import io

    import cv2
    import numpy as np

    with get_store() as store:
        row = store.get_media(media_id)
        if row is None or row["kind"] != "thermal_patch":
            abort(404)
        media = MediaStore(_root("MEDIA_DIR"), store)
        k = media.read_thermal_patch(media_id)

    lo, hi = float(np.min(k)), float(np.max(k))
    span = max(hi - lo, 1e-3)
    norm = ((k - lo) / span * 255).astype(np.uint8)
    big = cv2.resize(norm, (row["width"] * 8, row["height"] * 8), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", cv2.applyColorMap(big, cv2.COLORMAP_INFERNO))
    if not ok:
        abort(500)
    return _no_store(send_file(io.BytesIO(buf.tobytes()), mimetype="image/png"))


@app.route("/thumbs/<path:filename>")
def thumbs(filename: str):
    return _no_store(send_from_directory(_root("THUMB_DIR"), filename))


# ---------------------------------------------------------------------------
# installable app
# ---------------------------------------------------------------------------
@app.route("/sw.js")
def service_worker():
    """Served from the root, not /static/.

    A service worker can only control URLs at or below its own path, so one
    living at /static/sw.js could never cache the pages themselves.

    A hand-maintained VERSION constant is a bug waiting to happen: the cache is
    cache-first for static assets, so forgetting to bump it after editing a
    renderer serves the old JavaScript forever while the HTML updates normally.
    That failure is invisible from the server and looks to the user like the
    new feature simply does not work. So the version is STAMPED HERE from the
    content of the shell files, and editing any of them invalidates the cache
    by itself.
    """
    import hashlib

    static = pathlib.Path(app.root_path) / "static"
    src = (static / "sw.js").read_text()

    h = hashlib.sha256()
    for name in sorted(p.name for p in static.glob("*.js")):
        h.update(name.encode())
        h.update((static / name).read_bytes())
    for name in ("console.css",):
        f = static / name
        if f.exists():
            h.update(f.read_bytes())
    src = src.replace('const VERSION = "saresq-v3";',
                      f'const VERSION = "saresq-{h.hexdigest()[:12]}";')

    resp = app.response_class(src, mimetype="application/javascript")
    # The worker script itself must not be cached, or a stale one keeps
    # serving a stale shell forever.
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


@app.route("/cert")
def certificate():
    """Hand the TLS certificate to a phone so it can trust this ground station.

    iOS will happily "Add to Home Screen" a site with an untrusted certificate,
    but it will NOT register a service worker on one -- so the app installs and
    then has no offline cache, which is the one capability that matters in a
    field tent. Downloading this on the phone and trusting it in
    Settings -> General -> About -> Certificate Trust Settings fixes that.

    The MIME type is what makes Safari offer to install it as a profile rather
    than showing it as a text file.
    """
    crt = _root("CERT_DIR") / "saresq.crt"
    if not crt.exists():
        abort(404, "no certificate -- start the server with --https")
    return send_file(crt, mimetype="application/x-x509-ca-cert",
                     as_attachment=True, download_name="saresq-ground-station.crt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None,
                    help="mission store. OMIT for an ephemeral one that is deleted "
                         "when the dashboard stops -- nothing from a demo run is "
                         "left behind to be mistaken for a later mission's data.")
    ap.add_argument("--simulate", action="store_true",
                    help="let the radar draw a rehearsal picture when the store is "
                         "empty. Off by default: an empty scope is the correct "
                         "picture before a flight. Simulated tracks carry the "
                         "ASTERIX SIM bit.")
    ap.add_argument("--thumb-dir", default="results/thumbs")
    ap.add_argument("--media-dir", default="results/media")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--https", action="store_true",
                    help="serve over TLS with a self-signed certificate. Needed to install "
                         "the app or use it offline from a phone: a service worker requires "
                         "a secure context, and http://<lan-ip> is not one.")
    ap.add_argument("--cert-dir", default=".certs")
    ap.add_argument("--rescore-model", default="yolov8m.pt",
                    help="second-opinion detector run on uploaded crops "
                         "(default yolov8m.pt; skipped silently if absent)")
    ap.add_argument("--rescore-imgsz", type=int, default=640,
                    help="input size for the second opinion; the crop is 160 px, "
                         "and upscaling moves the target up the recall curve")
    ap.add_argument("--rescore-batch", type=int, default=8)
    ap.add_argument("--no-rescore", action="store_true",
                    help="disable ground-station re-scoring entirely")
    ap.add_argument("--payload", default=None,
                    help="live payload as host or host:port, e.g. 192.168.1.2 "
                         "(port defaults to 8091). Omit to review a recorded database.")
    ap.add_argument("--origin", default=None,
                    help="lat,lon datum for captures with no GPS fix (indoor demos)")
    args = ap.parse_args()

    # Ephemeral by default. A dashboard left running through a demo accumulates
    # real detections; if those are still in saresq.db next week they are
    # indistinguishable from that day's mission. Persisting is a deliberate
    # choice the operator makes with --db, never the default.
    _tmpdb = None
    if args.db:
        app.config["DB_PATH"] = args.db
    else:
        import atexit
        import tempfile
        _tmpdb = pathlib.Path(tempfile.mkdtemp(prefix="saresq-mission-")) / "mission.db"
        app.config["DB_PATH"] = str(_tmpdb)
        atexit.register(lambda: __import__("shutil").rmtree(_tmpdb.parent, ignore_errors=True))
        print(f"  ephemeral store {_tmpdb} (deleted on exit; pass --db to keep a mission)")
    app.config["SIMULATE"] = bool(args.simulate)
    app.config["THUMB_DIR"] = str(pathlib.Path(args.thumb_dir).expanduser().resolve())
    app.config["MEDIA_DIR"] = str(pathlib.Path(args.media_dir).expanduser().resolve())
    app.config["RESULTS_DIR"] = str(pathlib.Path(args.results_dir).expanduser().resolve())
    pathlib.Path(app.config["THUMB_DIR"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(app.config["MEDIA_DIR"]).mkdir(parents=True, exist_ok=True)

    app.config["CERT_DIR"] = str(pathlib.Path(args.cert_dir).expanduser().resolve())

    global LINK
    if args.payload:
        # The media store was never passed, so every crop and thermal patch the
        # payload froze was fetched by nobody and Evidence stayed empty.
        LINK = PayloadLink(args.payload, store_factory=get_store,
                           media_root=app.config["MEDIA_DIR"])
        if args.origin:
            try:
                la, lo = (float(v) for v in args.origin.split(","))
                LINK.set_origin(la, lo)
                print(f"  manual datum set to {la:.5f}, {lo:.5f} (used only without a GPS fix)")
            except ValueError:
                print(f"  ignoring --origin {args.origin!r}: expected 'lat,lon'")
        LINK.start()
        print(f"  payload link -> http://{LINK.host}")

    # Re-scoring starts in its own thread and loads the model there, so a cold
    # MPS warmup (several seconds) never delays the console coming up. If the
    # weights or torch are missing it reports why on /api/rescore and the rest
    # of the station is unaffected.
    global RESCORE
    if not args.no_rescore:
        def _spin_up():
            global RESCORE
            from saresq.rescore import Rescorer, RescoreWorker
            from saresq.store.media import MediaStore
            engine = Rescorer(args.rescore_model, imgsz=args.rescore_imgsz)
            if not engine.available:
                print(f"  re-scoring off: {engine.why}")
                RESCORE = RescoreWorker(get_store,
                                        lambda s: MediaStore(app.config["MEDIA_DIR"], s),
                                        engine, batch=args.rescore_batch)
                return
            worker = RescoreWorker(
                get_store,
                lambda s: MediaStore(app.config["MEDIA_DIR"], s),
                engine, batch=args.rescore_batch)
            worker.start()
            RESCORE = worker
            # The same engine also looks at the LIVE frame for the verdict, not
            # only at stored crops for the review queue. One load, two uses.
            if LINK is not None:
                LINK.attach_rescorer(engine)
            print(f"  re-scoring crops with {args.rescore_model} "
                  f"({engine.params/1e6:.1f} M params) at {args.rescore_imgsz} px on {engine.device}")
        threading.Thread(target=_spin_up, name="rescore-init", daemon=True).start()

    ssl_ctx = None
    if args.https:
        ssl_ctx = _self_signed(pathlib.Path(app.config["CERT_DIR"]))
        ip = _lan_ip()
        print("\n  On this machine   https://localhost:%d/radar" % args.port)
        print("  On a phone        https://%s:%d/radar" % (ip, args.port))
        print("  Trust it first    https://%s:%d/cert   (needed for offline use)\n" % (ip, args.port))
    app.run(host=args.host, port=args.port, debug=False, ssl_context=ssl_ctx, threaded=True)


def _lan_ip() -> str:
    """Best guess at the address a phone on the same Wi-Fi should open."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # never actually sends a packet
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _self_signed(cert_dir: pathlib.Path) -> tuple[str, str]:
    """Create (once) and return a self-signed cert covering this machine's LAN IP.

    Uses the openssl binary rather than Flask's ssl_context="adhoc", which
    needs the `cryptography` package -- one fewer dependency for a Pi image,
    and a cert that persists so the phone only has to trust it once.
    """
    import subprocess

    cert_dir.mkdir(parents=True, exist_ok=True)
    crt, key = cert_dir / "saresq.crt", cert_dir / "saresq.key"
    ip = _lan_ip()

    # Regenerate whenever the machine's address has changed, otherwise the
    # certificate no longer covers the URL the phone is asked to open and every
    # browser rejects it outright rather than merely warning.
    stale = True
    if crt.exists() and key.exists():
        try:
            txt = subprocess.run(["openssl", "x509", "-in", str(crt), "-noout", "-text"],
                                 check=True, capture_output=True, text=True).stdout
            stale = f"IP Address:{ip}" not in txt or "TLS Web Server Authentication" not in txt
        except (OSError, subprocess.CalledProcessError):
            stale = True

    if stale:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             # 825 days is Apple's hard ceiling for a server certificate; a
             # longer one is rejected on iOS no matter how it was installed.
             "-days", "820",
             "-subj", "/CN=SaResQ Ground Station/O=SaResQ",
             "-addext", f"subjectAltName=DNS:localhost,IP:127.0.0.1,IP:{ip}",
             # Required since iOS 13: a server certificate without an explicit
             # serverAuth EKU is not trusted even after the user installs it.
             "-addext", "extendedKeyUsage=serverAuth",
             "-addext", "basicConstraints=critical,CA:true",
             "-addext", "keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign",
             "-keyout", str(key), "-out", str(crt)],
            check=True, capture_output=True,
        )
        print(f"  wrote self-signed certificate for {ip} to {cert_dir}/")
    return str(crt), str(key)


if __name__ == "__main__":
    main()
