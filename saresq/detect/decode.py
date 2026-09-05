"""Decode a raw YOLOv8 TFLite output tensor and run NMS in NumPy (Section 7.9).

The exported model does NOT include non-maximum suppression: the output
tensor has shape (1, 4 + n_classes, n_anchors) with boxes already in
(cx, cy, w, h) pixel coordinates (Ultralytics bakes the DFL decode into the
export graph). Everything from here is ordinary NumPy, run on the Pi.
"""
from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def box_iou_batch(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (N, 4) array of xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.clip(union, 1e-9, None)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy per-class NMS. Returns indices to keep, highest score first."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        same_class = classes[rest] == classes[i]
        ious = box_iou_batch(boxes_xyxy[i], boxes_xyxy[rest])
        suppressed = same_class & (ious > iou_thresh)
        order = rest[~suppressed]
    return np.array(keep, dtype=int)


def decode_and_nms(
    y: np.ndarray, conf: float = 0.15, iou: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y: (4 + n_classes, n_anchors) raw model output (batch dim already squeezed).

    Returns (boxes_xyxy [N,4], scores [N], classes [N] int).
    """
    y = y.T  # (n_anchors, 4 + n_classes)
    boxes_xywh = y[:, :4]
    class_scores = y[:, 4:]
    classes = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(classes)), classes]

    mask = scores >= conf
    boxes_xywh, scores, classes = boxes_xywh[mask], scores[mask], classes[mask]
    if len(scores) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, classes, iou)
    return boxes_xyxy[keep], scores[keep], classes[keep]
