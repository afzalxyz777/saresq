"""Native desktop window onto the Pi's thermal stream.

    .venv/bin/python tools/thermal_view.py              # default host
    .venv/bin/python tools/thermal_view.py --host 192.168.1.16

Run this on the laptop, not the Pi. It opens a real OS window rather than a
browser tab, which is what you want when filming or demonstrating: no address
bar, no tab strip, and it can go fullscreen.

It reads the MJPEG stream that tools/thermal_live.py publishes. The stats are
fetched separately from /stats rather than recomputed here, so the numbers on
screen are the same ones the Pi's own gate would see -- there is no second
implementation to drift.

Keys:  q quit   f fullscreen   s save PNG   c cycle colour (on the Pi)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
import urllib.request

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
SAVE_DIR = REPO / "results" / "thermal"
WIN = "SaResQ thermal"


class Stats(threading.Thread):
    """Poll /stats so the overlay never blocks the video."""

    daemon = True

    def __init__(self, base: str):
        super().__init__()
        self.base = base
        self.data: dict = {}
        self.stop_flag = False

    def run(self) -> None:
        while not self.stop_flag:
            try:
                with urllib.request.urlopen(self.base + "/stats", timeout=3) as r:
                    self.data = json.load(r)
            except Exception:
                self.data = {}
            time.sleep(0.5)


def frames(url: str):
    """Yield JPEGs out of a multipart/x-mixed-replace stream."""
    with urllib.request.urlopen(url, timeout=10) as r:
        buf = b""
        while True:
            chunk = r.read(4096)
            if not chunk:
                return
            buf += chunk
            # JPEG start-of-image .. end-of-image, the only framing we need
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", start + 2)
            if start != -1 and end != -1:
                jpg = buf[start:end + 2]
                buf = buf[end + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img


def overlay(img: np.ndarray, st: dict) -> np.ndarray:
    if not st:
        return img
    h = img.shape[0]
    bar = img.copy()
    cv2.rectangle(bar, (0, h - 34), (img.shape[1], h), (12, 14, 18), -1)
    img = cv2.addWeighted(bar, 0.75, img, 0.25, 0)
    z = st.get("z", 0.0)
    fires = z >= 2.5
    txt = (f"{st.get('min',0):.1f}-{st.get('max',0):.1f}C   "
           f"spread {st.get('spread',0):.1f}C   "
           f"peak +{z:.1f}sigma   {st.get('fps',0):.1f}fps")
    cv2.putText(img, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    if fires:
        cv2.putText(img, "GATE", (img.shape[1] - 66, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 120), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.1.16")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    stats = Stats(base)
    stats.start()
    full = False
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 704, 560)
    print(f"streaming from {base}   (q quit, f fullscreen, s save)")

    try:
        for img in frames(base + "/stream"):
            cv2.imshow(WIN, overlay(img, stats.data))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break
            if k == ord("f"):
                full = not full
                cv2.setWindowProperty(
                    WIN, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            if k == ord("s"):
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                p = SAVE_DIR / f"view_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(p), img)
                print("saved", p)
            if k == ord("c"):
                try:
                    urllib.request.urlopen(base + "/key?k=c", timeout=2).read()
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass
    except urllib.error.URLError as exc:
        print(f"cannot reach {base}: {exc.reason}")
        print("is tools/thermal_live.py running on the Pi?")
        return 1
    finally:
        stats.stop_flag = True
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
