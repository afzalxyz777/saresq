"""End-to-end demo on real hardware: thermal gate + RGB frame + the detector.

    ~/saresq-venv/bin/python3 tools/demo_detect.py

Runs on the Pi. Captures one RGB frame from the camera module and one thermal
frame from the MLX90640, runs the shipped gate over the thermal and the shipped
TFLite detector over the RGB, and writes an annotated side-by-side to
results/demo/.

Capture goes through `rpicam-still` rather than picamera2 because picamera2 is
a system package and this venv was built without system site packages. Shelling
out costs about a second per frame and avoids rebuilding the venv, which is not
a thing to do the night before a demonstration.

Nothing here re-implements detection or gating: both come from saresq/, so what
you see on screen is the code that would fly.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "demo"


def capture_rgb(path: pathlib.Path, width: int, height: int,
                timeout_ms: int = 2500) -> np.ndarray | None:
    cmd = ["rpicam-still", "-n", "-o", str(path), "--width", str(width),
           "--height", str(height), "-t", str(timeout_ms), "--immediate"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=45)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"  camera capture failed: {exc}")
        return None
    if r.returncode != 0:
        print(f"  rpicam-still exited {r.returncode}: "
              f"{r.stderr.decode(errors='replace')[-300:]}")
        return None
    return cv2.imread(str(path))


def capture_thermal(n: int = 6) -> np.ndarray | None:
    try:
        import adafruit_mlx90640
        import board
        import busio
    except ImportError:
        print("  MLX driver not installed")
        return None
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    buf = [0.0] * 768
    stack = []
    for _ in range(n + 5):
        try:
            mlx.getFrame(buf)
        except (ValueError, RuntimeError, OSError):
            time.sleep(0.15)
            continue
        a = np.array(buf, dtype=np.float32).reshape(24, 32)
        if np.isfinite(a).all() and 0 < a.mean() < 90:
            stack.append(a)
        if len(stack) >= n:
            break
    if not stack:
        return None
    return np.fliplr(np.median(np.stack(stack), axis=0))


def run_gate(thermal: np.ndarray, cfg: dict) -> dict:
    """The shipped gate statistic: how far the hottest pixels sit above scene."""
    g = cfg.get("gate", {})
    z_t = float(g.get("z_t", 2.5))
    sigma = max(float(thermal.std()), float(g.get("sigma_min_k", 0.15)))
    z = (thermal - thermal.mean()) / sigma
    mask = z >= z_t
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > int(g.get("max_blob_px", 30)):
            continue
        cy, cx = float(cents[i][1]), float(cents[i][0])
        blobs.append({"row": round(cy, 1), "col": round(cx, 1), "px": area,
                      "z": round(float(z[labels == i].max()), 2),
                      "T": round(float(thermal[labels == i].max()), 1)})
    blobs.sort(key=lambda b: -b["z"])
    return {"z_max": round(float(z.max()), 2), "z_t": z_t,
            "fired": bool(blobs), "blobs": blobs,
            "frac_above": round(float(mask.mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=640, choices=[224, 640])
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--width", type=int, default=1640)
    ap.add_argument("--height", type=int, default=1232)
    ap.add_argument("--no-thermal", action="store_true")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "pipeline.yaml").read_text())
    conf = args.conf if args.conf is not None else float(
        cfg.get("detector", {}).get("conf", 0.15))

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")

    print("1. RGB capture ...")
    rgb = capture_rgb(OUT / f"rgb_{stamp}.jpg", args.width, args.height)
    if rgb is None:
        print("   no RGB frame -- is the ribbon seated and the Pi rebooted?")
        return 1
    print(f"   {rgb.shape[1]}x{rgb.shape[0]}")

    thermal, gate = None, None
    if not args.no_thermal:
        print("2. thermal capture ...")
        thermal = capture_thermal()
        if thermal is not None:
            gate = run_gate(thermal, cfg)
            print(f"   {thermal.min():.1f}-{thermal.max():.1f}C  "
                  f"z_max {gate['z_max']}  gate "
                  f"{'FIRED' if gate['fired'] else 'quiet'}  "
                  f"{len(gate['blobs'])} blob(s)")

    print("3. detector ...")
    from saresq.detect.tflite_detector import CropDetector
    model = REPO / "models" / f"yolov8n_p3_{args.imgsz}_w8a32.tflite"
    if not model.exists():
        print(f"   missing {model}")
        return 1
    det = CropDetector(str(model), conf=conf,
                       iou=float(cfg.get("detector", {}).get("nms_iou", 0.5)),
                       num_threads=int(cfg.get("detector", {}).get("threads", 4)))
    t0 = time.time()
    dets = det.detect(rgb)
    ms = (time.time() - t0) * 1000
    print(f"   {len(dets)} detection(s) in {ms:.0f} ms  "
          f"(box units: {det.box_units}, conf>={conf})")

    vis = rgb.copy()
    for d in dets:
        # CropDetector yields Detection dataclasses in SOURCE-frame pixels,
        # not tuples -- indexing them raises TypeError.
        x0, y0, x1, y1 = (int(v) for v in d.xyxy)
        score, cls = float(d.score), int(d.cls)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (60, 230, 120), 3)
        cv2.putText(vis, f"{cls}:{score:.2f}", (x0, max(y0 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 230, 120), 2, cv2.LINE_AA)
        print(f"     cls {cls}  conf {score:.3f}  "
              f"box {x1-x0}x{y1-y0} px at ({x0},{y0})")

    cv2.imwrite(str(OUT / f"detect_{stamp}.jpg"), vis)

    if thermal is not None:
        lo, hi = float(thermal.min()), float(thermal.max())
        norm = np.clip((thermal - lo) / max(hi - lo, 1e-6), 0, 1)
        tim = cv2.applyColorMap(
            cv2.resize((norm * 255).astype(np.uint8),
                       (vis.shape[1] // 3, vis.shape[0] // 3),
                       interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_INFERNO)
        h, w = tim.shape[:2]
        vis[0:h, vis.shape[1] - w:] = tim
        cv2.rectangle(vis, (vis.shape[1] - w, 0), (vis.shape[1] - 1, h),
                      (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"combined_{stamp}.jpg"), vis)
        np.save(OUT / f"thermal_{stamp}.npy", thermal)

    (OUT / f"result_{stamp}.json").write_text(json.dumps(
        {"detections": len(dets), "ms": round(ms, 1), "imgsz": args.imgsz,
         "conf": conf, "gate": gate}, indent=2))
    print(f"\nwrote {OUT}/detect_{stamp}.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
