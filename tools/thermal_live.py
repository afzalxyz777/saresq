"""Live MLX90640 view in a browser. Run on the Pi, watch from anything.

    ~/saresq-venv/bin/python3 tools/thermal_live.py
    then open  http://<pi-ip>:8090

Why a browser and not a window: the Pi is headless on a drone, so there is no
display to open a matplotlib figure on, and X-forwarding a 24x32 array over
wifi costs more than the frames are worth. MJPEG over HTTP works from a phone,
a laptop, or a judge's screen with nothing installed at either end.

The sensor is read in its own thread at a fixed cadence and the newest frame
handed to whoever asks. That matters: MLX90640 reads are blocking and take
~40 ms, so doing them inside the request would couple frame rate to the number
of viewers and tear frames when two people watched at once.

Keys on the page: C cycles the colour map, F freezes the scale to the current
range (auto-scaling makes a still room look dramatic and a real target look
flat, so freeze before judging contrast), S saves a PNG + .npy pair.
"""
from __future__ import annotations

import argparse
import http.server
import json
import pathlib
import socketserver
import threading
import time

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
SAVE_DIR = REPO / "results" / "thermal"

MAPS = [("inferno", cv2.COLORMAP_INFERNO), ("jet", cv2.COLORMAP_JET),
        ("hot", cv2.COLORMAP_HOT), ("bone", cv2.COLORMAP_BONE),
        ("turbo", cv2.COLORMAP_TURBO)]

PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>SaResQ thermal</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0c0e12;color:#e8e6e2;
      font:14px/1.5 ui-monospace,Menlo,monospace;
      display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
 h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
 img{width:min(92vw,720px);image-rendering:auto;border:1px solid #262b33;border-radius:4px}
 #stats{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;color:#98a1ac}
 #stats b{color:#e8e6e2}
 .hot{color:#ff7a45}
 kbd{background:#1a1f26;border:1px solid #303741;border-radius:3px;padding:1px 5px}
 #keys{color:#727b86;font-size:12.5px}
</style>
<h1>MLX90640 &mdash; live</h1>
<img src="/stream" alt="thermal stream">
<div id=stats></div>
<div id=keys><kbd>C</kbd> colour <kbd>F</kbd> freeze scale <kbd>S</kbd> save frame</div>
<script>
async function poll(){
  try{
    const r = await fetch('/stats'); const s = await r.json();
    document.getElementById('stats').innerHTML =
      `<span>min <b>${s.min.toFixed(1)}&deg;C</b></span>`+
      `<span>max <b class=hot>${s.max.toFixed(1)}&deg;C</b></span>`+
      `<span>spread <b>${s.spread.toFixed(1)}&deg;C</b></span>`+
      `<span>peak <b>${s.z>=2.5?'&#9679; ':''}+${s.z.toFixed(1)}&sigma;</b></span>`+
      `<span>${s.map}${s.frozen?' &middot; frozen':''}</span>`+
      `<span><b>${s.fps.toFixed(1)}</b> fps</span>`;
  }catch(e){}
  setTimeout(poll, 500);
}
poll();
addEventListener('keydown', e=>{
  const k = e.key.toLowerCase();
  if('cfs'.includes(k)) fetch('/key?k='+k);
});
</script>
"""


class Sensor(threading.Thread):
    """Reads the array continuously; hands out the newest frame."""

    daemon = True

    def __init__(self, hz: int = 8):
        super().__init__()
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.fps = 0.0
        self.hz = hz
        self.stop_flag = False

    def run(self) -> None:
        import adafruit_mlx90640
        import board
        import busio

        i2c = busio.I2C(board.SCL, board.SDA)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        rate = {2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ}
        mlx.refresh_rate = rate.get(self.hz, rate[8])
        print("serial:", [hex(x) for x in mlx.serial_number], flush=True)

        buf = [0.0] * 768
        last = time.time()
        while not self.stop_flag:
            try:
                mlx.getFrame(buf)
            except (ValueError, RuntimeError, OSError):
                # a torn subpage read; the next one is usually fine
                time.sleep(0.05)
                continue
            a = np.array(buf, dtype=np.float32).reshape(24, 32)
            if not np.isfinite(a).all():
                continue
            now = time.time()
            dt = now - last
            last = now
            with self.lock:
                self.frame = np.fliplr(a)     # the array reads mirrored
                self.fps = 0.8 * self.fps + 0.2 * (1.0 / dt) if dt > 0 else self.fps

    def read(self) -> tuple[np.ndarray | None, float]:
        with self.lock:
            return (None, 0.0) if self.frame is None else (self.frame.copy(), self.fps)


class State:
    def __init__(self) -> None:
        self.map_i = 0
        self.frozen: tuple[float, float] | None = None
        self.save_next = False
        self.last = {"min": 0.0, "max": 0.0, "spread": 0.0, "z": 0.0}


def render(img: np.ndarray, st: State, scale: int = 22) -> np.ndarray:
    lo, hi = st.frozen if st.frozen else (float(img.min()), float(img.max()))
    norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    big = cv2.resize((norm * 255).astype(np.uint8), (32 * scale, 24 * scale),
                     interpolation=cv2.INTER_CUBIC)
    out = cv2.applyColorMap(big, MAPS[st.map_i][1])

    # mark the hottest pixel -- this is what the gate keys on
    hy, hx = np.unravel_index(int(img.argmax()), img.shape)
    cx, cy = int((hx + 0.5) * scale), int((hy + 0.5) * scale)
    cv2.circle(out, (cx, cy), 13, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{img.max():.1f}C", (cx + 17, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{lo:.1f}C", (8, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def make_handler(sensor: Sensor, st: State):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):       # quiet; the console shows frames
            pass

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                    # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif path == "/stats":
                self._send(200, "application/json",
                           json.dumps({**st.last, "map": MAPS[st.map_i][0],
                                       "frozen": st.frozen is not None,
                                       "fps": sensor.read()[1]}).encode())
            elif path == "/key":
                k = self.path.partition("k=")[2][:1]
                img, _ = sensor.read()
                if k == "c":
                    st.map_i = (st.map_i + 1) % len(MAPS)
                elif k == "f":
                    st.frozen = None if st.frozen else (
                        (float(img.min()), float(img.max())) if img is not None else None)
                elif k == "s":
                    st.save_next = True
                self._send(200, "text/plain", b"ok")
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=f")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.stream()
            else:
                self._send(404, "text/plain", b"no")

        def stream(self):
            try:
                while True:
                    img, fps = sensor.read()
                    if img is None:
                        time.sleep(0.1)
                        continue
                    st.last = {"min": float(img.min()), "max": float(img.max()),
                               "spread": float(img.max() - img.min()),
                               "z": float((img.max() - img.mean())
                                          / max(img.std(), 1e-6))}
                    if st.save_next:
                        st.save_next = False
                        SAVE_DIR.mkdir(parents=True, exist_ok=True)
                        stamp = time.strftime("%H%M%S")
                        cv2.imwrite(str(SAVE_DIR / f"live_{stamp}.png"),
                                    render(img, st))
                        np.save(SAVE_DIR / f"live_{stamp}.npy", img)
                        print(f"saved live_{stamp}", flush=True)
                    ok, jpg = cv2.imencode(".jpg", render(img, st),
                                           [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if not ok:
                        continue
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode()
                                     + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    time.sleep(1.0 / 12)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--hz", type=int, default=8, choices=[2, 4, 8])
    args = ap.parse_args()

    sensor = Sensor(hz=args.hz)
    sensor.start()
    st = State()
    with Server(("0.0.0.0", args.port), make_handler(sensor, st)) as srv:
        print(f"open http://<pi-ip>:{args.port}   (ctrl-C to stop)", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
