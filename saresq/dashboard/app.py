"""Flask dashboard (Section 13.2): a Leaflet map reading the SQLite store.

Vendored Leaflet (static/leaflet/) means this works with zero internet --
the point of the offline-resilience demo (Section 13.3). Nothing in the
pipeline waits for this; it's a read-only view onto saresq.db.

    python -m saresq.dashboard.app --db saresq.db --port 5050
"""
from __future__ import annotations

import argparse
import pathlib

from flask import Flask, jsonify, send_from_directory

from saresq.store.db import Store

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["DB_PATH"] = "saresq.db"
app.config["THUMB_DIR"] = "results/thumbs"


def get_store() -> Store:
    return Store(app.config["DB_PATH"])


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


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


@app.route("/thumbs/<path:filename>")
def thumbs(filename: str):
    return send_from_directory(app.config["THUMB_DIR"], filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="saresq.db")
    ap.add_argument("--thumb-dir", default="results/thumbs")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()

    app.config["DB_PATH"] = args.db
    app.config["THUMB_DIR"] = args.thumb_dir
    pathlib.Path(args.thumb_dir).mkdir(parents=True, exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
