"""Background re-scoring worker.

Runs on the ground station beside the payload link. It is deliberately a
*pull* loop over the database rather than something the link pushes into:

* the link's job is to not lose captures, and it should never be blocked
  behind a 300 ms inference;
* a pull loop picks up crops that arrived while the ground station was
  restarted, or that were replayed from a snapshot, with no extra code;
* and because `rescores.media_id` is UNIQUE, "what still needs doing" is a
  question the database answers exactly, so the worker holds no state that
  could drift out of sync with reality.

Threading: SQLite connections belong to the thread that opened them, so the
worker builds its own Store per batch from the factory it was handed -- the
same pattern PayloadLink._ingest uses, and for the same reason.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np

from saresq.rescore.engine import Rescorer


class RescoreWorker(threading.Thread):
    daemon = True

    def __init__(self, store_factory: Callable[[], object],
                 media_factory: Callable[[object], object],
                 rescorer: Rescorer,
                 batch: int = 8, idle_s: float = 2.0):
        super().__init__(name="rescore")
        self._store_factory = store_factory
        self._media_factory = media_factory
        self.rescorer = rescorer
        self.batch = int(batch)
        self.idle_s = float(idle_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._scored = 0
        self._last_ms: float | None = None
        self._last_err: str | None = None

    def stop(self) -> None:
        self._stop.set()

    # -- status for the UI ----------------------------------------------------
    def state(self) -> dict:
        with self._lock:
            d = {"scored_session": self._scored, "last_ms": self._last_ms,
                 "error": self._last_err, "running": self.is_alive()}
        d.update(self.rescorer.describe())
        return d

    # -- the loop -------------------------------------------------------------
    def run(self) -> None:
        if not self.rescorer.available:
            return                       # nothing to do; /api/rescore says why
        while not self._stop.is_set():
            try:
                did = self._tick()
            except Exception as exc:
                with self._lock:
                    self._last_err = str(exc)
                did = 0
            # Only sleep when the queue is empty. While a mission is producing
            # crops the loop runs flat out, which is what keeps the operator's
            # second opinion within a couple of seconds of the capture.
            if did == 0:
                self._stop.wait(self.idle_s)

    def _tick(self) -> int:
        store = self._store_factory()
        try:
            rows = store.unscored_crops(limit=self.batch)
            if not rows:
                return 0
            media = self._media_factory(store)

            crops, keep = [], []
            for r in rows:
                img = self._load(media, r)
                if img is not None:
                    crops.append(img)
                    keep.append(r)
            if not crops:
                # Unreadable blobs would otherwise be retried forever. Record a
                # zero so the queue drains and the operator sees a gap rather
                # than a worker that appears wedged.
                for r in rows:
                    store.insert_rescore(
                        media_id=r["media_id"], target_id=r["target_id"],
                        t_ns=int(time.time() * 1e9), p=0.0, n=0,
                        p_payload=r.get("priority"),
                        model=self.rescorer.weights, imgsz=self.rescorer.imgsz, ms=0.0)
                return len(rows)

            results = self.rescorer.score(crops)
            now = int(time.time() * 1e9)
            for r, res in zip(keep, results):
                store.insert_rescore(
                    media_id=r["media_id"], target_id=r["target_id"], t_ns=now,
                    p=res.p, n=res.n,
                    # media.priority is p_final at capture -- the aircraft's own
                    # belief, frozen. Copying it here means the comparison
                    # survives even if the target is later re-judged.
                    p_payload=r.get("priority"),
                    model=self.rescorer.weights, imgsz=self.rescorer.imgsz, ms=res.ms)
            with self._lock:
                self._scored += len(results)
                self._last_ms = results[0].ms if results else None
                self._last_err = None
            return len(rows)
        finally:
            try:
                store.close()
            except Exception:
                pass

    @staticmethod
    def _load(media, row) -> np.ndarray | None:
        try:
            raw = media.read(row["media_id"])
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            return img if img is not None and img.size else None
        except Exception:
            return None
