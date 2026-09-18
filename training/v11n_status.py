#!/usr/bin/env python
"""Status and verdict for the YOLO11n vs YOLOv8n A/B.

Run it any time: mid-training it shows the matched-epoch race and an ETA;
once `per_source.json` exists it applies the adoption rule and prints a verdict.

    .venv/bin/python training/v11n_status.py

THE ADOPTION RULE IS PRE-REGISTERED, AND THAT IS THE POINT
It is written here before the result is known, so the decision cannot be
rationalised afterwards. thermal_combined.yaml already states the shape of it:
a combined model counts as an improvement only if it HOLDS HIT-UAV Person AP
while RAISING RGBT. Two numbers are added to that here.

  1. A margin, not a sign. Run-to-run scatter on these runs is worth one to two
     points of AP -- v8n's own HIT-UAV curve swung 0.30 -> 0.24 -> 0.36 across
     three consecutive epochs. Adopting on +0.001 would be adopting noise.

  2. Ties go to the incumbent, and a tie is not a neutral outcome -- it is a
     loss, because switching is not free:
       - 11n trains ~2.6x slower per epoch on MPS (measured, not quoted)
       - adopting costs a TFLite re-export, a re-verified INT8 gap, a fresh
         on-Pi latency benchmark, and every quoted number in the deck redone
       - its one clear advantage, 14% fewer parameters, buys nothing here:
         peak resident memory is 95.2 MB of 905 MB available, so capacity is
         not the binding constraint. Pixels on target is.
     A model that merely matches v8n must therefore be rejected.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DET = REPO / "results" / "detector"

# --- the pre-registered rule -------------------------------------------------
MARGIN = 0.02          # mean Person AP50 must beat v8n by at least this
HITUAV_FLOOR = 0.851   # and must not give up HIT-UAV (v8n's shipped figure)

STAGES = [
    ("hituav",      "v8n_p3_hituav_640",      "v11n_p3_hituav_640",      30),
    ("thermalmix",  "v8n_p3_thermalmix_640",  "v11n_p3_thermalmix_640",  20),
    ("thermalmix2", "v8n_p3_thermalmix2_640", "v11n_p3_thermalmix2_640", 25),
]


def rows(run: str) -> list[dict]:
    p = DET / run / "results.csv"
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def fmt_h(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


def race() -> None:
    for tag, v8_run, v11_run, total in STAGES:
        a, b = rows(v8_run), rows(v11_run)
        if not b:
            print(f"\nstage {tag:12} not started")
            continue
        done = len(b)
        print(f"\nstage {tag:12} epoch {done}/{total}")
        print(f"  {'ep':>3} {'v8n':>9} {'v11n':>9} {'delta':>9}")
        # Show the last five matched epochs; the early ones are all noise.
        for i in range(max(0, done - 5), done):
            bv = float(b[i]["metrics/mAP50(B)"])
            if i < len(a):
                av = float(a[i]["metrics/mAP50(B)"])
                print(f"  {i+1:>3} {av:9.4f} {bv:9.4f} {bv - av:+9.4f}")
            else:
                print(f"  {i+1:>3} {'--':>9} {bv:9.4f} {'--':>9}")
        if done >= 2:
            per_ep = (float(b[-1]["time"]) - float(b[-2]["time"]))
            left = (total - done) * per_ep
            print(f"  pace {per_ep:.0f} s/epoch; {total - done} left ~ {fmt_h(left)}")
        if a:
            print(f"  v8n finished this stage at mAP50 {a[-1]['metrics/mAP50(B)']}")


def verdict() -> int:
    v8p = DET / "v8n_p3_thermalmix2_640" / "per_source.json"
    v11p = DET / "v11n_p3_thermalmix2_640" / "per_source.json"
    if not v11p.exists():
        print("\nno v11n per_source.json yet -- training or evaluation still running")
        return 0

    v8 = json.loads(v8p.read_text())
    v11 = json.loads(v11p.read_text())

    def unpack(d):
        return (d["per_source"]["hituav"]["person_AP50"],
                d["per_source"]["rgbt_thermal"]["person_AP50"],
                d["person_AP50_mean"])

    h8, r8, m8 = unpack(v8)
    h11, r11, m11 = unpack(v11)

    print("\n=== PERSON AP50, PER SOURCE ===")
    print(f"{'':20}{'HIT-UAV':>10}{'RGBT':>10}{'mean':>10}")
    print(f"{'v8n (shipped)':20}{h8:10.4f}{r8:10.4f}{m8:10.4f}")
    print(f"{'v11n (candidate)':20}{h11:10.4f}{r11:10.4f}{m11:10.4f}")
    print(f"{'delta':20}{h11-h8:+10.4f}{r11-r8:+10.4f}{m11-m8:+10.4f}")

    beats = m11 >= m8 + MARGIN
    holds = h11 >= HITUAV_FLOOR
    print(f"\nrule: mean >= {m8 + MARGIN:.4f} (v8n + {MARGIN})  -> {'PASS' if beats else 'FAIL'}"
          f"  (actual {m11:.4f})")
    print(f"      HIT-UAV >= {HITUAV_FLOOR:.4f}                -> {'PASS' if holds else 'FAIL'}"
          f"  (actual {h11:.4f})")

    if beats and holds:
        print("\nADOPT YOLO11n -- clears the margin and holds HIT-UAV.")
        return 0
    print("\nKEEP YOLOv8n. 11n does not earn the switch; a tie or a small win is")
    print("a loss once re-export, re-benchmarking and the slower training are priced in.")
    print("Honest line for judges: 'We trained and evaluated YOLO11n end to end on")
    print("our own data under the identical three-stage recipe. It did not beat")
    print("v8n by a margin that justified re-validating the deployed stack, because")
    print("our accuracy is bounded by pixels on target, not model capacity.'")
    return 0


if __name__ == "__main__":
    race()
    sys.exit(verdict())
