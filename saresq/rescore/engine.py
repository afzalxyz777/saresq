"""The second-opinion detector.

Design notes, because most of the speed here is in what the code does NOT do:

* **The model is loaded once and kept warm.** Ultralytics will happily reload
  per call; at ~1.4 s a load that would dominate everything else.
* **The first inference is thrown away.** On MPS the first forward pass pays
  for graph compilation and is 10-30x slower than the steady state. Warming up
  at construction means the first real crop an operator is waiting on gets the
  fast path, not the slow one.
* **Crops are scored in batches.** A batch of 16 is not 16x the cost of one
  crop; on MPS it is closer to 3x, because the per-call overhead (Python,
  tensor setup, kernel dispatch) is paid once instead of sixteen times.
* **Upscaling is the point, not a side effect.** The crop is 160 px, and the
  measured recall curve for this detector family (results/detector/
  rgb_confirm.json) is a steep function of apparent target size: 0.075 below
  16 px, 0.553 at 24-32 px, 0.815 above 48 px. Feeding a 160 px crop to a
  640 px input turns a 40 px person into a 160 px one and moves it up that
  curve. No information is added -- the gain is that the detector's receptive
  fields are sized for objects it can now actually resolve.
* **Only `person` is asked for.** Restricting classes at inference removes the
  other 79 from NMS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

#: COCO class 0. Both the payload's detector and every model used here put
#: `person` at index 0, which is also why the payload can run a stock COCO
#: checkpoint on its visible branch without a remap.
PERSON_CLASS = 0


@dataclass(frozen=True)
class RescoreResult:
    p: float          # best person confidence in this crop, 0.0 if none
    n: int            # person boxes at or above the threshold
    ms: float         # wall-clock inference cost attributed to this crop
    #: Every person confidence in the frame, highest first. The engine runs at
    #: one permissive threshold so a caller can apply its own without a second
    #: forward pass -- and the right threshold differs by question. "Is anyone
    #: there" tolerates a weak box; "there are three people" must not.
    confs: tuple[float, ...] = ()


class Rescorer:
    """A larger detector, loaded once, run in batches.

    `available` is False rather than raising when the weights or torch are
    missing: the ground station must start and run normally on a machine that
    cannot do this, because the mission matters more than the enhancement.
    """

    def __init__(self, weights: str = "yolov8m.pt", imgsz: int = 640,
                 conf: float = 0.10, device: str | None = None,
                 warmup: bool = True):
        self.weights = weights
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.available = False
        self.why: str | None = None
        self._model = None
        self.device = device or self._pick_device()

        try:
            from ultralytics import YOLO
        except Exception as exc:                       # torch/ultralytics absent
            self.why = f"ultralytics unavailable: {exc}"
            return
        try:
            self._model = YOLO(self.weights)
            self.params = sum(p.numel() for p in self._model.model.parameters())
            self.available = True
        except Exception as exc:                       # weights missing, no network
            self.why = f"could not load {self.weights}: {exc}"
            return

        if warmup:
            self._warmup()

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
        except Exception:
            return "cpu"
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _warmup(self) -> None:
        """Pay the graph-compilation cost now, on a fake frame, not on the
        operator's first real crop."""
        try:
            blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            self.score([blank])
            self.warmup_ms = (time.perf_counter() - t0) * 1e3
        except Exception as exc:
            self.available = False
            self.why = f"warmup failed: {exc}"

    def score(self, crops: list[np.ndarray]) -> list[RescoreResult]:
        """Score a batch of BGR crops. Order of results matches the input.

        A crop that fails to decode does not sink the batch -- it comes back as
        p=0.0, because a lost second opinion must never remove a candidate the
        payload already raised.
        """
        if not self.available or not crops:
            return [RescoreResult(0.0, 0, 0.0) for _ in crops]

        t0 = time.perf_counter()
        try:
            preds = self._model.predict(
                crops, imgsz=self.imgsz, conf=self.conf, device=self.device,
                classes=[PERSON_CLASS], verbose=False,
            )
        except Exception:
            return [RescoreResult(0.0, 0, 0.0) for _ in crops]
        per_crop_ms = (time.perf_counter() - t0) * 1e3 / max(len(crops), 1)

        out: list[RescoreResult] = []
        for r in preds:
            try:
                confs = r.boxes.conf.tolist() if r.boxes is not None else []
            except Exception:
                confs = []
            out.append(RescoreResult(
                p=float(max(confs)) if confs else 0.0,
                n=len(confs),
                ms=per_crop_ms,
                confs=tuple(sorted((float(c) for c in confs), reverse=True)),
            ))
        return out

    def describe(self) -> dict:
        return {
            "available": self.available,
            "why": self.why,
            "model": self.weights,
            "imgsz": self.imgsz,
            "device": self.device,
            "conf": self.conf,
            "params": getattr(self, "params", None),
        }
