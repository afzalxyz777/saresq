"""The cascaded crop detector as it runs on the payload computer (Section 7.9).

The architectural bet of this whole system is here. A full 640x640 forward pass
is 409,600 pixels of work; a 160x160 crop is 25,600. The thermal gate nominates
two to six crops per frame, so the detector does roughly 1/16th to 1/3rd of the
work a whole-frame detector would -- which is the difference between a pipeline
that keeps up on a 1 GB Pi 4 and one that does not. Everything in this module
exists to keep that arithmetic honest:

* no per-call allocation (see ``saresq.detect.runtime``),
* no resize when the crop already matches the model's input size, which is the
  common case by construction,
* decode and NMS in NumPy on the survivors only (``saresq.detect.decode``).

**On box coordinates.** Ultralytics' TFLite export emits box regressions
normalised to the input square, while its PyTorch path emits pixels. Getting
this wrong does not raise -- it produces boxes that are plausibly shaped and
entirely wrong, scaled by 160x. So the convention is not assumed here: it is
measured once at load time by ``_probe_box_units`` against the real file. The
project has been bitten before by parsing a format from its documentation
rather than from the artefact; this is that lesson applied to a model export.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from .decode import decode_and_nms
from .runtime import TFLiteModel


@dataclass(frozen=True)
class Detection:
    """One box in the coordinates of the *source frame*, not the crop."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio, pad to ``size``x``size``.

    Returns the padded image plus the (gain, pad_x, pad_y) needed to map boxes
    back. Padding is 114 grey rather than black because a black border reads to
    the network as a large cold region, which for a *thermal* model is not
    neutral padding -- it is a strong feature.
    """
    h, w = image.shape[:2]
    if h == size and w == size:
        return image, 1.0, 0, 0
    gain = min(size / h, size / w)
    nh, nw = int(round(h * gain)), int(round(w * gain))
    import cv2

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=image.dtype)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, gain, pad_x, pad_y


class CropDetector:
    """One TFLite detector at one input size."""

    def __init__(
        self,
        model_path: str | pathlib.Path,
        conf: float = 0.15,
        iou: float = 0.5,
        num_threads: int = 2,
        box_units: str | None = None,
    ):
        self.model = TFLiteModel(model_path, num_threads=num_threads)
        self.conf = conf
        self.iou = iou
        self.box_units = box_units or self._probe_box_units()

    @property
    def imgsz(self) -> int:
        return self.model.imgsz

    @property
    def last_ms(self) -> float:
        return self.model.last_ms

    def _probe_box_units(self) -> str:
        """Decide 'normalized' vs 'pixels' by measuring, once, on real output.

        The box rows of a YOLOv8 head carry a regression for *every* anchor
        regardless of confidence, so even a featureless grey input produces
        the full box tensor. If those values live near 0-1 the export
        normalised them; if they span the input size it did not. The two
        hypotheses differ by the input size (160x or more), far outside any
        ambiguity, so a midpoint threshold is safe.
        """
        size = self.model.imgsz
        grey = np.full((1, size, size, 3), 114, dtype=np.uint8)
        raw = self.model.infer_one(grey)
        boxes = np.squeeze(raw)[:4]
        peak = float(np.max(np.abs(boxes)))
        return "normalized" if peak <= max(2.0, size * 0.05) else "pixels"

    def _decode(self, raw: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.squeeze(raw)
        if y.shape[0] > y.shape[1]:  # (anchors, 4+nc) -> (4+nc, anchors)
            y = y.T
        if self.box_units == "normalized":
            y = y.copy()
            y[:4] *= size
        return decode_and_nms(y, conf=self.conf, iou=self.iou)

    def detect(self, crop: np.ndarray, origin: tuple[float, float] = (0.0, 0.0)) -> list[Detection]:
        """Detect in one crop; return boxes offset by ``origin`` into frame coords."""
        size = self.model.imgsz
        padded, gain, pad_x, pad_y = letterbox(crop, size)
        raw = self.model.infer_one(padded[None, ...])
        boxes, scores, classes = self._decode(raw, size)
        ox, oy = origin
        out: list[Detection] = []
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            # Undo the letterbox before applying the crop's origin, or the pad
            # offset gets scaled by the gain and every box drifts.
            out.append(Detection(
                x1=float((x1 - pad_x) / gain + ox), y1=float((y1 - pad_y) / gain + oy),
                x2=float((x2 - pad_x) / gain + ox), y2=float((y2 - pad_y) / gain + oy),
                score=float(score), cls=int(cls),
            ))
        return out

    def detect_crops(self, crops: list[np.ndarray], origins: list[tuple[float, float]]) -> list[list[Detection]]:
        """Run the gate's whole crop budget. Sequential by design (see runtime)."""
        return [self.detect(c, o) for c, o in zip(crops, origins)]

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"<CropDetector {self.model.path.name} imgsz={self.imgsz} boxes={self.box_units}>"


class CascadedDetector:
    """Picks the crop-size model per blob, per ``configs/pipeline.yaml``.

    Small blobs get the 160 model, blobs at or above ``large_blob_px`` get the
    224 one. Two models are held resident rather than one resized: TFLite fixes
    its input shape at export time, and upscaling a 160 crop to 224 adds cost
    without adding information, while downscaling a 224 crop throws away the
    resolution that made it worth the larger model.
    """

    def __init__(
        self,
        model_small: str | pathlib.Path,
        model_large: str | pathlib.Path,
        large_blob_px: int = 9,
        conf: float = 0.15,
        iou: float = 0.5,
        num_threads: int = 2,
    ):
        self.small = CropDetector(model_small, conf=conf, iou=iou, num_threads=num_threads)
        self.large = CropDetector(model_large, conf=conf, iou=iou, num_threads=num_threads)
        self.large_blob_px = large_blob_px

    def for_blob(self, area_t: int) -> CropDetector:
        return self.large if area_t >= self.large_blob_px else self.small

    def detect(self, crop: np.ndarray, area_t: int, origin: tuple[float, float] = (0.0, 0.0)) -> list[Detection]:
        return self.for_blob(area_t).detect(crop, origin)
