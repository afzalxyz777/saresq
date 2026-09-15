"""Clean MLX90640 still: slow refresh, median stack, honest scale.

Refresh is dropped to 2 Hz. At 8 Hz a subpage arrives every 62 ms while the
I2C read of 834 words takes ~37 ms, so any hiccup tears the frame across a
subpage boundary -- which is what the residual chessboard actually is. At
2 Hz there is half a second of slack and the tear cannot happen.
"""
import sys, time
import numpy as np, board, busio, adafruit_mlx90640, cv2

DELAY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
if DELAY:
    print(f"aim the lens -- capturing in {DELAY:.0f}s", flush=True)
    time.sleep(DELAY)

i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

frame = [0.0] * 768
stack = []
for _ in range(N + 6):
    try:
        mlx.getFrame(frame)
    except (ValueError, RuntimeError, OSError):
        time.sleep(0.2); continue
    a = np.array(frame, dtype=np.float32).reshape(24, 32)
    if np.isfinite(a).all() and 0 < a.mean() < 90:
        stack.append(a)
    if len(stack) >= N:
        break

img = np.median(np.stack(stack), axis=0)
img = np.fliplr(img)                       # sensor reads mirrored
spread = float(img.max() - img.min())
print(f"{len(stack)} frames | {img.min():.1f}-{img.max():.1f}C | "
      f"spread {spread:.1f}C | mean {img.mean():.1f}C")

lo, hi = float(img.min()), float(img.max())
norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
big = cv2.resize((norm*255).astype(np.uint8), (32*22, 24*22),
                 interpolation=cv2.INTER_CUBIC)
out = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)
cv2.putText(out, f"{lo:.1f}C", (8, 512), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255,255,255), 1, cv2.LINE_AA)
cv2.putText(out, f"{hi:.1f}C", (600, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255,255,255), 1, cv2.LINE_AA)
cv2.imwrite("/tmp/thermal.png", out)
np.save("/tmp/thermal.npy", img)

# the gate's own statistic: how far the hottest pixel sits above the scene
z = (img.max() - img.mean()) / max(img.std(), 1e-6)
print(f"hottest pixel z = +{z:.1f} sigma  (gate fires at z_t = 2.5)")
