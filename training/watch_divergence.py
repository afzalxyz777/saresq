"""Watch a live Ultralytics run and kill it if the learning curve diverges.

    nohup .venv/bin/python training/watch_divergence.py \
        results/detector/v8n_p3_thermalmix3_640 --pid 44855 \
        > results/divergence_watch.log 2>&1 &

WHY KILLING IS THE RIGHT RESPONSE
best.pt is already on disk and is never overwritten by a worse epoch, so there
is nothing left to protect once the curve has turned. An unattended run that has
diverged spends the rest of the night making the checkpoint no better while
holding the GPU, so stopping it early costs nothing and saves the window.

WHAT COUNTS AS DIVERGENCE HERE
Not "the number went down" -- a single bad epoch is normal, and mosaic
augmentation makes val mAP jump around by 0.05 on its own. These are the
failures where continuing is provably pointless:

  * a non-finite loss (NaN/inf) -- gradients are gone, nothing recovers
  * a NEGATIVE classification loss. This is not hypothetical: on 2026-09-10
    the P2 head under fp16 on MPS ran cls_loss 4.97 -> 46 -> -1224 by
    iteration 6. It does not raise, it just trains to garbage, and it is the
    reason train_detector.py forces amp=False for P2.
  * loss exploding past 3x its best -- the optimiser has left the basin
  * mAP50 collapsing under half its best for 3 straight epochs

OVERFITTING IS WARNED ABOUT, NOT KILLED
Rising val loss against falling train loss is overfitting, not divergence.
best.pt already holds the pre-overfit weights and `patience` stops the run on
its own, so killing would only throw away epochs that might still recover.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import signal
import sys
import time


def read_csv(path: pathlib.Path):
    try:
        lines = path.read_text().strip().splitlines()
    except (OSError, FileNotFoundError):
        return [], []
    if len(lines) < 2:
        return [], []
    head = [h.strip() for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) != len(head):
            continue          # a row being written right now: skip, do not guess
        row = {}
        for k, v in zip(head, parts):
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = math.nan
        rows.append(row)
    return head, rows


def col(row: dict, *names):
    for n in names:
        if n in row:
            return row[n]
    return math.nan


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(pid: int, why: str) -> None:
    print(f"\n{'!'*66}\nDIVERGENCE: {why}\nstopping pid {pid}; best.pt on disk is kept\n{'!'*66}",
          flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not alive(pid):
                break
            time.sleep(1)
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
    except OSError as e:
        print(f"could not signal {pid}: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--explode", type=float, default=3.0, help="loss multiple of best")
    ap.add_argument("--collapse", type=float, default=0.5, help="mAP fraction of best")
    ap.add_argument("--collapse-epochs", type=int, default=3)
    ap.add_argument("--overfit-epochs", type=int, default=5)
    ap.add_argument("--no-kill", action="store_true", help="report only")
    args = ap.parse_args()

    csv = pathlib.Path(args.run_dir) / "results.csv"
    print(f"watching {csv}\npid {args.pid}, polling every {args.every:.0f}s", flush=True)

    seen = 0
    best_map = 0.0
    best_loss = math.inf
    collapse_run = 0
    overfit_run = 0
    prev_vcls = math.inf
    prev_tcls = math.inf

    while True:
        if not alive(args.pid):
            print(f"\ntraining process {args.pid} has exited. "
                  f"{seen} epoch(s) were checked, no divergence detected.", flush=True)
            return

        _, rows = read_csv(csv)
        for row in rows[seen:]:
            seen += 1
            ep = col(row, "epoch")
            tb = col(row, "train/box_loss")
            tc = col(row, "train/cls_loss")
            td = col(row, "train/dfl_loss")
            vb = col(row, "val/box_loss")
            vc = col(row, "val/cls_loss")
            m50 = col(row, "metrics/mAP50(B)")

            losses = {"train/box": tb, "train/cls": tc, "train/dfl": td,
                      "val/box": vb, "val/cls": vc}
            print(f"ep{int(ep):<3} mAP50={m50:.4f} "
                  f"train(box={tb:.3f} cls={tc:.3f} dfl={td:.3f}) "
                  f"val(box={vb:.3f} cls={vc:.3f})", flush=True)

            # --- hard failures -------------------------------------------------
            bad = [k for k, v in losses.items() if not math.isfinite(v)]
            if bad:
                if not args.no_kill:
                    stop(args.pid, f"non-finite loss at epoch {int(ep)}: {', '.join(bad)}")
                return
            neg = [k for k, v in losses.items() if v < 0]
            if neg:
                if not args.no_kill:
                    stop(args.pid, f"NEGATIVE loss at epoch {int(ep)}: "
                                   f"{', '.join(f'{k}={losses[k]:.2f}' for k in neg)}. "
                                   f"A loss cannot be negative -- this is the fp16/MPS "
                                   f"overflow mode, see train_detector.py.")
                return

            tot = tb + tc + td
            best_loss = min(best_loss, tot)
            if tot > best_loss * args.explode:
                if not args.no_kill:
                    stop(args.pid, f"train loss exploded at epoch {int(ep)}: "
                                   f"{tot:.2f} vs best {best_loss:.2f} "
                                   f"({tot / best_loss:.1f}x)")
                return

            best_map = max(best_map, m50)
            if best_map > 0 and m50 < best_map * args.collapse:
                collapse_run += 1
                print(f"     ^ mAP50 {m50:.4f} is under {args.collapse:.0%} of best "
                      f"{best_map:.4f}  [{collapse_run}/{args.collapse_epochs}]", flush=True)
                if collapse_run >= args.collapse_epochs and not args.no_kill:
                    stop(args.pid, f"mAP50 collapsed for {collapse_run} straight epochs "
                                   f"({m50:.4f} vs best {best_map:.4f})")
                    return
            else:
                collapse_run = 0

            # --- soft warning: overfitting, which is NOT a reason to stop ------
            if math.isfinite(prev_vcls):
                if vc > prev_vcls and tc < prev_tcls:
                    overfit_run += 1
                    if overfit_run >= args.overfit_epochs:
                        print(f"     ^ WARNING: val/cls has risen while train/cls fell for "
                              f"{overfit_run} epochs -- overfitting. Not stopping: best.pt "
                              f"holds the pre-overfit weights and patience will end it.",
                              flush=True)
                else:
                    overfit_run = 0
            prev_vcls, prev_tcls = vc, tc

        time.sleep(args.every)


if __name__ == "__main__":
    main()
