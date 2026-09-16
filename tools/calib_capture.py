"""Drive the calibration capture over HTTP: you move the board, this presses Save.

    .venv/bin/python tools/calib_capture.py --host 192.168.1.2

Nothing is installed on the Pi. The demo already exposes GET /save (the button
in the app) and GET /stats, so this is the same two endpoints the phone uses.

It waits for the targets to actually appear before capturing anything. An empty
room spans ~3 K; a tray of ice against it spans ~30 K. Watching that number is
the difference between capturing a calibration and capturing eight pictures of a
wall -- which is the failure mode that only shows up later, when the fit has no
targets to find and the whole bench session has to be repeated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def get(host: str, path: str, timeout: float = 8.0):
    with urllib.request.urlopen(f"http://{host}:8091{path}", timeout=timeout) as r:
        return r.read()


def stats(host: str) -> dict:
    return json.loads(get(host, "/stats"))


def spread_of(s: dict) -> float:
    return float(s.get("thermal", {}).get("spread", 0.0))


def bar(v: float, lo: float, hi: float, w: int = 30) -> str:
    n = max(0, min(w, int(w * (v - lo) / max(hi - lo, 1e-6))))
    return "[" + "#" * n + "-" * (w - n) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.2")
    ap.add_argument("--n", type=int, default=8, help="how many frame sets")
    ap.add_argument("--gap", type=float, default=6.0, help="seconds to move the board")
    ap.add_argument("--min-spread", type=float, default=10.0,
                    help="K between coldest and hottest pixel before capturing")
    ap.add_argument("--wait", type=float, default=180.0, help="seconds to wait for targets")
    ap.add_argument("--clear", action="store_true", help="delete old frame sets first")
    args = ap.parse_args()

    try:
        s = stats(args.host)
    except Exception as e:
        sys.exit(f"cannot reach the payload at {args.host}:8091 -- {e}")

    print(f"payload reachable. thermal now: spread {spread_of(s):.1f} K "
          f"({s['thermal']['min']:.1f}-{s['thermal']['max']:.1f} C)\n")

    if args.clear:
        print("NOTE: clear old frames yourself over ssh; this tool only captures.\n")

    # --- wait for the targets to come into view -------------------------------
    print(f"Put the board in the payload's view. Waiting for spread >= "
          f"{args.min_spread:.0f} K ...")
    t0 = time.time()
    while True:
        sp = spread_of(stats(args.host))
        ok = sp >= args.min_spread
        print(f"\r  spread {sp:5.1f} K {bar(sp, 0, max(args.min_spread * 1.5, 30))} "
              f"{'TARGETS IN VIEW' if ok else 'waiting...':<16}", end="", flush=True)
        if ok:
            print("\n")
            break
        if time.time() - t0 > args.wait:
            print("\n")
            sys.exit(f"gave up after {args.wait:.0f}s. The board is not in the THERMAL "
                     f"field of view -- check the thermal tab in the app, not your eye. "
                     f"The two cameras do not see the same rectangle.")
        time.sleep(1.0)

    # --- capture --------------------------------------------------------------
    print(f"Capturing {args.n} frame sets, {args.gap:.0f}s apart.")
    print("Move the board to a NEW position each time you are told, then hold it still.\n")
    got = 0
    for k in range(1, args.n + 1):
        for t in range(int(args.gap), 0, -1):
            print(f"\r  [{k}/{args.n}] move the board... hold still in {t}s   ",
                  end="", flush=True)
            time.sleep(1.0)
        s = stats(args.host)
        sp = spread_of(s)
        if sp < args.min_spread:
            print(f"\r  [{k}/{args.n}] SKIPPED -- spread fell to {sp:.1f} K "
                  f"(board out of thermal view)      ")
            continue
        get(args.host, "/save")
        got += 1
        time.sleep(0.6)   # let the capture thread pick up save_next and write
        print(f"\r  [{k}/{args.n}] captured    spread {sp:5.1f} K  "
              f"max {s['thermal']['max']:.1f} C  blobs {s['gate']['n_blobs']}        ")

    print(f"\n{got} frame set(s) captured.\nnext:  PI={args.host} bash tools/calib_pull.sh")


if __name__ == "__main__":
    main()
