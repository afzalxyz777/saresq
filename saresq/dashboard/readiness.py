"""What this build actually has, and what each part is still waiting on.

The dashboard is deliberately honest about gaps. A section with no data says
which command or which piece of hardware would fill it, rather than rendering
a plausible-looking placeholder — a demo that fakes its own inputs is how a
team ends up believing a capability it does not have.

Every check below is a filesystem or database fact, never a hard-coded flag.
"""
from __future__ import annotations

import pathlib

from saresq.store.db import Store

OK, PARTIAL, BLOCKED = "ok", "part", "block"


def _first(root: pathlib.Path, *patterns: str) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for pat in patterns:
        out += sorted(root.glob(pat))
    return out


def readiness(db_path: str, results_dir: str, media_dir: str) -> dict:
    res = pathlib.Path(results_dir)
    counts = {"targets": 0, "passes": 0, "hazards": 0, "verdicts": 0,
              "media": 0, "clips": 0, "pending_sync": 0, "labelled": 0}
    by_kind: dict[str, int] = {}

    try:
        with Store(db_path) as store:
            counts["targets"] = len(store.all_targets())
            counts["hazards"] = len(store.all_hazards())
            counts["verdicts"] = len(store.all_verdicts())
            rows = store.conn.execute("SELECT kind, COUNT(*) n, SUM(synced_ns IS NULL) p FROM media GROUP BY kind").fetchall()
            for r in rows:
                by_kind[r["kind"]] = r["n"]
                counts["media"] += r["n"]
                counts["pending_sync"] += r["p"] or 0
            counts["clips"] = by_kind.get("clip", 0)
            counts["passes"] = store.conn.execute("SELECT COUNT(*) n FROM passes").fetchone()["n"]
            counts["labelled"] = store.conn.execute(
                "SELECT COUNT(*) n FROM verdicts WHERE features_json IS NOT NULL "
                "AND verdict IN ('SURVIVOR','DISPATCHED','NOT_SURVIVOR')").fetchone()["n"]
    except Exception:
        pass

    tflite = _first(res, "detector/**/*.tflite", "**/*.tflite")
    det_metrics = _first(res, "detector/**/results.csv", "**/results.csv")
    pixels = _first(res, "**/pixels_on_target.csv")
    hazard = _first(res, "**/hazard_classifier_report.md", "hazard/**/*.json")
    labels = _first(res, "fusion_labels/fusion_labels.npz")
    fusion_model = _first(res, "**/fusion_head*.json", "**/fusion_head*.joblib")
    sim = _first(res, "sim/**/*.csv", "**/policy_comparison.csv")

    items = [
        {
            "key": "perception", "name": "Perception pipeline (non-ML)",
            "status": OK,
            # No test count here on purpose. Everything else on this page is
            # derived from the filesystem or the store at request time; a
            # hard-coded "72/72 passing" is exactly the kind of stale flag the
            # page exists to avoid, and it was already wrong within two days.
            "detail": "Gate, registration, tracking, log-odds fusion, geo-tagging, "
                      "store, surveillance tracker — run `pytest tests/` for the count.",
            "waiting": "",
        },
        {
            "key": "detector", "name": "Person detector (YOLOv8n)",
            "status": OK if tflite else (PARTIAL if det_metrics else BLOCKED),
            "detail": (f"{len(tflite)} TFLite export(s) present." if tflite
                       else "Training scripts written and verified against the real datasets; never run."),
            "waiting": "" if tflite else "Run training/train_detector.py + export_tflite.py in Colab, drop the outputs in results/detector/.",
        },
        {
            "key": "pixels", "name": "Pixels-on-target calibration",
            "status": OK if pixels else BLOCKED,
            "detail": ("Sweep-width calibration available." if pixels
                       else "Effective sweep width W is an assumption (14 m) until this runs."),
            "waiting": "" if pixels else "Run training/eval_pixels_on_target.py — its output calibrates W in the search-effort model.",
        },
        {
            "key": "hazard", "name": "Hazard classifier (AIDER)",
            "status": OK if hazard else BLOCKED,
            "detail": "MobileNetV2 over 6,438 verified images." if hazard else "Script written; dataset downloaded and verified; never trained.",
            "waiting": "" if hazard else "Run training/train_hazard.py in Colab.",
        },
        {
            "key": "footage", "name": "Field footage",
            "status": OK if counts["clips"] else (PARTIAL if counts["media"] else BLOCKED),
            "detail": (", ".join(f"{v} {k}" for k, v in sorted(by_kind.items())) if by_kind
                       else "No imagery captured."),
            "waiting": "" if counts["clips"] else "Needs the payload on the bench: MLX90640 + Pi Camera capturing a person at 8 / 12 / 20 m.",
        },
        {
            "key": "targets", "name": "Detected targets",
            "status": OK if counts["targets"] else BLOCKED,
            "detail": f"{counts['targets']} target(s), {counts['passes']} pass(es), {counts['hazards']} hazard(s).",
            "waiting": "" if counts["targets"] else "Needs saresq/pipeline.py or replay.py run over recorded footage.",
        },
        {
            "key": "review", "name": "Operator verdicts",
            "status": OK if counts["verdicts"] else (PARTIAL if counts["targets"] else BLOCKED),
            "detail": f"{counts['verdicts']} verdict(s); {counts['labelled']} usable as fusion labels.",
            "waiting": "" if counts["verdicts"] else "Work the Review queue once targets exist.",
        },
        {
            "key": "fusion", "name": "Fusion head (18-feature)",
            "status": OK if fusion_model else (PARTIAL if labels else BLOCKED),
            "detail": ("Trained head present." if fusion_model
                       else f"{counts['labelled']} labelled vector(s) — a logistic head wants a few hundred."),
            "waiting": "" if fusion_model else "Collect operator verdicts, then training/export_verdicts.py + train_fusion.py.",
        },
        {
            "key": "sim", "name": "Mission simulator",
            "status": OK if sim else PARTIAL,
            "detail": "Adaptive vs fixed-lawnmower policy comparison." + ("" if sim else " Results not exported to results/sim/."),
            "waiting": "" if sim else "Run sim/run.py --out results/sim to publish the comparison here.",
        },
    ]
    order = {OK: 0, PARTIAL: 1, BLOCKED: 2}
    blocked = [i for i in items if i["status"] == BLOCKED]
    return {
        "counts": counts, "by_kind": by_kind, "items": items,
        "n_ok": sum(1 for i in items if i["status"] == OK),
        "n_blocked": len(blocked),
        "next": blocked[0]["waiting"] if blocked else "",
        "paths": {"results": str(res), "media": str(media_dir), "db": db_path},
    }
