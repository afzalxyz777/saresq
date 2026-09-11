"""Turn operator verdicts into a labelled fusion-head training set.

This closes a loop that is otherwise open in this project. `train_fusion.py`
needs (18-feature vector -> is-this-really-a-person) pairs, and nothing in the
pipeline produces the label. An operator working the review queue produces
exactly that label, for free, as a side effect of the safety gate we wanted
anyway.

Label mapping:
    SURVIVOR, DISPATCHED  -> 1
    NOT_SURVIVOR          -> 0
    UNSURE                -> dropped (an operator who cannot tell is not a label)

Usage:
    python training/export_verdicts.py --db saresq.db --out results/fusion_labels
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from saresq.fuse.features import FEATURE_NAMES
from saresq.store.db import Store

POSITIVE = {"SURVIVOR", "DISPATCHED"}
NEGATIVE = {"NOT_SURVIVOR"}


def export(db_path: str, out_dir: str) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, dropped_unsure, dropped_nofeat = [], 0, 0
    with Store(db_path) as store:
        for v in store.all_verdicts():
            if v["verdict"] not in POSITIVE and v["verdict"] not in NEGATIVE:
                dropped_unsure += 1
                continue
            if not v["features_json"]:
                # No frozen feature vector: the track was judged but the
                # pipeline never wrote pass_features. Not recoverable after
                # the fact -- the background stats are gone.
                dropped_nofeat += 1
                continue
            feats = json.loads(v["features_json"])
            if feats is None or len(feats) != len(FEATURE_NAMES):
                dropped_nofeat += 1
                continue
            rows.append({
                "target_id": v["target_id"],
                "y": 1 if v["verdict"] in POSITIVE else 0,
                "verdict": v["verdict"],
                "operator": v["operator"],
                "p_machine": v["p_final_at_verdict"],
                "decision_machine": v["decision_at_verdict"],
                "x": feats,
            })

    X = np.array([r["x"] for r in rows], dtype=np.float32) if rows else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int8)

    np.savez(out / "fusion_labels.npz", X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    with open(out / "fusion_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target_id", "y", "verdict", "operator", "p_machine", "decision_machine", *FEATURE_NAMES])
        for r in rows:
            w.writerow([r["target_id"], r["y"], r["verdict"], r["operator"],
                        r["p_machine"], r["decision_machine"], *r["x"]])

    # Where the machine and the operator disagreed. These are the rows that
    # actually carry information -- the ones worth reading before retraining.
    disagreements = [
        r for r in rows
        if (r["decision_machine"] == "CONFIRM") != bool(r["y"])
    ]

    summary = {
        "n": len(rows),
        "positive": int(y.sum()) if len(y) else 0,
        "negative": int((y == 0).sum()) if len(y) else 0,
        "dropped_unsure": dropped_unsure,
        "dropped_no_features": dropped_nofeat,
        "disagreements": len(disagreements),
        "n_features": len(FEATURE_NAMES),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="saresq.db")
    ap.add_argument("--out", default="results/fusion_labels")
    args = ap.parse_args()

    s = export(args.db, args.out)
    print(f"exported {s['n']} labelled vectors ({s['positive']} positive / {s['negative']} negative)")
    print(f"  dropped: {s['dropped_unsure']} UNSURE, {s['dropped_no_features']} without frozen features")
    print(f"  machine/operator disagreements: {s['disagreements']}")
    if s["n"] < 50:
        print("\n  NOTE: a logistic fusion head over 18 features wants a few hundred")
        print("  examples before it beats the spec's hand-set weights. Keep flying.")
    print(f"\nwrote {args.out}/fusion_labels.npz, .csv, summary.json")


if __name__ == "__main__":
    main()
