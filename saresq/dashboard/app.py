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

from saresq.dashboard.readiness import readiness
from saresq.store.db import Store
from saresq.store.media import MediaStore

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["DB_PATH"] = "saresq.db"
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


@app.route("/evidence")
def page_evidence():
    return render_template("evidence.html", page="/evidence")


@app.route("/review")
def page_review():
    return render_template("review.html", page="/review")


@app.route("/analytics")
def page_analytics():
    return render_template("analytics.html", page="/analytics")


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
            _radar = RadarService(db_path=app.config["DB_PATH"])
        return _radar


@app.route("/api/radar")
def api_radar():
    return jsonify(get_radar().snapshot())


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
    for cand in sorted(res.glob("detector/**/results.csv")) + sorted(res.glob("**/results.csv")):
        try:
            with open(cand, newline="") as fh:
                rows = [r for r in _csv.DictReader(fh)]
            if not rows:
                continue
            last = {k.strip(): v for k, v in rows[-1].items() if k}
            out["detector"] = {"file": cand.name, "epochs": len(rows), "final": last}
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


@app.route("/api/media")
def api_media():
    """All evidence rows, newest first, optionally filtered by kind or target."""
    kind = request.args.get("kind")
    target = request.args.get("target_id", type=int)
    sql = "SELECT * FROM media WHERE 1=1"
    args = []
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
        return jsonify(store.all_targets())


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
    with get_store() as store:
        out = []
        for t in store.review_queue():
            t = dict(t)
            t["media"] = store.media_for_target(t["target_id"])
            t["passes"] = store.get_passes_for_target(t["target_id"])
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
@app.route("/media/<int:media_id>")
def media_blob(media_id: int):
    with get_store() as store:
        row = store.get_media(media_id)
        if row is None:
            abort(404)
        path = _root("MEDIA_DIR") / row["rel_path"]
        if not path.exists():
            abort(404)
        return send_file(path, mimetype=_MIME.get(path.suffix.lstrip("."), "application/octet-stream"))


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
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


@app.route("/thumbs/<path:filename>")
def thumbs(filename: str):
    return send_from_directory(_root("THUMB_DIR"), filename)


# ---------------------------------------------------------------------------
# installable app
# ---------------------------------------------------------------------------
@app.route("/sw.js")
def service_worker():
    """Served from the root, not /static/.

    A service worker can only control URLs at or below its own path, so one
    living at /static/sw.js could never cache the pages themselves.
    """
    resp = send_from_directory(pathlib.Path(app.root_path) / "static", "sw.js",
                               mimetype="application/javascript")
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
    ap.add_argument("--db", default="saresq.db")
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
    args = ap.parse_args()

    app.config["DB_PATH"] = args.db
    app.config["THUMB_DIR"] = str(pathlib.Path(args.thumb_dir).expanduser().resolve())
    app.config["MEDIA_DIR"] = str(pathlib.Path(args.media_dir).expanduser().resolve())
    app.config["RESULTS_DIR"] = str(pathlib.Path(args.results_dir).expanduser().resolve())
    pathlib.Path(app.config["THUMB_DIR"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(app.config["MEDIA_DIR"]).mkdir(parents=True, exist_ok=True)

    app.config["CERT_DIR"] = str(pathlib.Path(args.cert_dir).expanduser().resolve())

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
