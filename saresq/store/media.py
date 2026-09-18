"""Evidence media: the crops, thermal patches and clips a human reviews.

Design notes that matter:

* **Storage is cheap, transport is not.** A 20-minute mission over a 3 ha
  segment with ~40 candidate tracks costs roughly 2.6 MB of evidence -- a 32 GB
  card holds over a thousand missions. So nothing here optimises for disk. The
  fields that exist (priority, sent_bytes, sha256) exist to serve
  saresq.sync, which is where the real constraint lives.

* **Content-addressed.** A blob lands at blobs/<sha[:2]>/<sha>.<ext>. Writing
  the same bytes twice is a no-op, which matters because the same crop is
  routinely both a thumb source and a full crop.

* **Write locally first, always.** The pipeline calls put_*() and returns. It
  never waits for a link. Losing the radio costs you latency, never evidence.
"""
from __future__ import annotations

import hashlib
import pathlib
import time

import numpy as np

from saresq.store.db import Store

# Kinds, in the order a sync agent should ship them.
KIND_THUMB = "thumb"
KIND_RGB_CROP = "rgb_crop"
KIND_THERMAL_PATCH = "thermal_patch"
KIND_CLIP = "clip"

_EXT = {KIND_THUMB: "jpg", KIND_RGB_CROP: "jpg", KIND_THERMAL_PATCH: "bin", KIND_CLIP: "mp4"}

# Thermal patches are stored as raw uint16 centi-kelvin, little-endian: exact,
# trivially parseable, and a 32x24 MLX90640 frame is 1536 bytes on the nose.
# uint16 centi-kelvin tops out at 655.35 K, well clear of any fire we'd survey.
THERMAL_SCALE = 100.0
THERMAL_DTYPE = "<u2"


def _now_ns() -> int:
    return time.time_ns()


class MediaStore:
    """Writes evidence blobs and registers them in the `media` table."""

    def __init__(self, root: str | pathlib.Path, store: Store):
        self.root = pathlib.Path(root)
        self.store = store
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _write_blob(self, data: bytes, ext: str) -> tuple[str, str]:
        """Returns (sha256, path relative to root). Idempotent by content."""
        sha = hashlib.sha256(data).hexdigest()
        rel = f"blobs/{sha[:2]}/{sha}.{ext}"
        dest = self.root / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a power loss can never leave a torn blob
            # that still hashes to a name something else will trust.
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
        return sha, rel

    def put_bytes(
        self,
        data: bytes,
        kind: str,
        *,
        target_id: int | None = None,
        pass_id: int | None = None,
        t_ns: int | None = None,
        priority: float = 0.0,
        width: int | None = None,
        height: int | None = None,
        synced: bool = False,
    ) -> int:
        """`synced` marks the artefact as already downlinked.

        The sync fields were written for the AIRCRAFT's store, where NULL means
        "still queued to send". On the GROUND station the same row means the
        opposite: the ground station only ever holds an artefact because it
        successfully fetched it. Leaving it NULL made Evidence label 247
        pictures it was displaying on screen as "queued on aircraft", which is
        a contradiction an operator can see -- and exactly the kind of untrue
        status line the rest of this console exists to avoid.
        """
        if kind not in _EXT:
            raise ValueError(f"unknown media kind {kind!r}; expected one of {sorted(_EXT)}")
        sha, rel = self._write_blob(data, _EXT[kind])
        return self.store.insert_media(
            target_id=target_id, pass_id=pass_id, t_ns=t_ns if t_ns is not None else _now_ns(),
            kind=kind, sha256=sha, rel_path=rel, bytes=len(data),
            width=width, height=height, priority=priority,
            synced_ns=_now_ns() if synced else None,
            sent_bytes=len(data) if synced else 0,
        )

    # ------------------------------------------------------------------
    def put_rgb_crop(self, bgr: np.ndarray, *, quality: int = 80, kind: str = KIND_RGB_CROP, **kw) -> int:
        """Encode an OpenCV BGR crop as JPEG and store it."""
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed on the RGB crop")
        h, w = bgr.shape[:2]
        return self.put_bytes(buf.tobytes(), kind, width=w, height=h, **kw)

    def put_thumb(self, bgr: np.ndarray, *, size: int = 96, quality: int = 60, **kw) -> int:
        """A ~3 KB thumbnail -- the first thing over the link, and often all an
        operator needs to reject a sun-heated sheet of corrugated iron."""
        import cv2

        h, w = bgr.shape[:2]
        scale = size / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
        return self.put_rgb_crop(bgr, quality=quality, kind=KIND_THUMB, **kw)

    def put_thermal_patch(self, kelvin: np.ndarray, **kw) -> int:
        """Store a thermal patch losslessly to 0.01 K."""
        a = np.asarray(kelvin, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise ValueError("thermal patch contains non-finite values")
        counts = np.rint(a * THERMAL_SCALE)
        if counts.min() < 0 or counts.max() > np.iinfo(np.uint16).max:
            raise ValueError(
                f"thermal patch {a.min():.1f}-{a.max():.1f} K is outside the "
                f"0-{np.iinfo(np.uint16).max / THERMAL_SCALE:.1f} K storable range"
            )
        data = counts.astype(THERMAL_DTYPE).tobytes()
        h, w = a.shape[:2]
        return self.put_bytes(data, KIND_THERMAL_PATCH, width=w, height=h, **kw)

    def put_clip(self, encoded: bytes, **kw) -> int:
        """Store an already-encoded clip. Encoding belongs to the camera stack
        (hardware H.264 on the Pi), not here -- re-encoding on the CPU would
        steal budget from the gate."""
        return self.put_bytes(encoded, KIND_CLIP, **kw)

    # ------------------------------------------------------------------
    def path_for(self, media_id: int) -> pathlib.Path:
        row = self.store.get_media(media_id)
        if row is None:
            raise KeyError(f"no media row {media_id}")
        return self.root / row["rel_path"]

    def read(self, media_id: int) -> bytes:
        return self.path_for(media_id).read_bytes()

    def read_thermal_patch(self, media_id: int) -> np.ndarray:
        row = self.store.get_media(media_id)
        if row is None or row["kind"] != KIND_THERMAL_PATCH:
            raise KeyError(f"media {media_id} is not a thermal patch")
        raw = np.frombuffer(self.read(media_id), dtype=THERMAL_DTYPE)
        return (raw.astype(np.float64) / THERMAL_SCALE).reshape(row["height"], row["width"])
