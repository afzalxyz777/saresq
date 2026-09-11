"""Scene hazard classifier (Section 8.2): MobileNetV2 trained on AIDER.

This runs on the **whole frame**, once per second, not on detection crops.
AIDER's images are entire aerial scenes -- a flooded street, a collapsed
block -- so the classifier's learned features are scene-level texture and
layout. Feeding it a 160 px crop centred on one person would ask it a question
it was never trained to answer, and it would still return five confident-looking
numbers. The rate (1 Hz against the detector's per-frame cadence) is set in
``configs/pipeline.yaml`` for the same reason: scene context does not change
between consecutive frames of a survey pass.

Its output is not a decision. It lands in the fusion vector as three of the
eighteen features (``p_flood``, ``p_fire``, ``p_collapse``), where the head
decides what a flooded scene should do to a candidate's probability. That
indirection is the point: "warm blob in a flood scene" and "warm blob in a
fire scene" mean very different things for survival, and neither belongs in a
hand-written rule.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .runtime import TFLiteModel

#: Order is fixed by training/train_hazard.py's ``class_names=CLASSES``, which
#: overrides Keras's default alphabetical ordering. If one changes the other
#: must, and ``HazardClassifier`` will not detect the mismatch -- a TFLite file
#: carries no class names. Change both or neither.
CLASSES = ["collapsed_building", "fire", "flooded_areas", "traffic_incident", "normal"]


class HazardClassifier:
    """Five-way scene classifier over a full frame."""

    def __init__(self, model_path: str | pathlib.Path, num_threads: int = 2, classes: list[str] | None = None):
        self.model = TFLiteModel(model_path, num_threads=num_threads)
        self.classes = classes or CLASSES
        n_out = int(np.prod(self.model.infer_one(
            np.zeros((1, self.model.imgsz, self.model.imgsz, 3), dtype=np.uint8)
        ).shape))
        if n_out != len(self.classes):
            raise ValueError(
                f"{self.model.path.name} outputs {n_out} classes but {len(self.classes)} names "
                f"were given; training/train_hazard.py and saresq.detect.hazard.CLASSES disagree"
            )

    @property
    def last_ms(self) -> float:
        return self.model.last_ms

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        """Class probabilities for one BGR frame of any size."""
        import cv2

        size = self.model.imgsz
        if frame.shape[0] != size or frame.shape[1] != size:
            # INTER_AREA for downscaling: a survey frame is far larger than
            # 224 px, and bilinear decimation aliases the fine texture the
            # scene classifier keys on.
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        raw = self.model.infer_one(frame[None, ...].astype(np.uint8))
        probs = np.squeeze(raw).astype(np.float64)
        # An INT8 softmax comes back quantised and no longer sums to exactly 1.
        # Renormalising matters because these feed a log-odds ledger, where a
        # 3% sum error is a systematic bias, not rounding.
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return {name: float(p) for name, p in zip(self.classes, probs)}

    def hazard_features(self, frame: np.ndarray) -> tuple[float, float, float]:
        """(p_flood, p_fire, p_collapse) in fusion-vector order."""
        p = self.predict(frame)
        return p["flooded_areas"], p["fire"], p["collapsed_building"]
