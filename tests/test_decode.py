"""Task-adjacent test for saresq.detect.decode (Section 7.9): decode + NMS in NumPy."""
import numpy as np

from saresq.detect.decode import decode_and_nms, xywh_to_xyxy


def _make_raw_output(entries: list[tuple[float, float, float, float, int, float, int]], n_classes: int = 2):
    """entries: list of (cx, cy, w, h, class_id, score, _unused) -> raw (4+n_classes, N) tensor."""
    n = len(entries)
    y = np.zeros((4 + n_classes, n))
    for i, (cx, cy, w, h, cls, score, _) in enumerate(entries):
        y[0, i], y[1, i], y[2, i], y[3, i] = cx, cy, w, h
        y[4 + cls, i] = score
    return y


def test_xywh_to_xyxy():
    boxes = np.array([[50.0, 50.0, 20.0, 10.0]])
    xyxy = xywh_to_xyxy(boxes)
    assert np.allclose(xyxy, [[40.0, 45.0, 60.0, 55.0]])


def test_low_confidence_detections_are_dropped():
    y = _make_raw_output([(50, 50, 20, 20, 0, 0.05, 0)])
    boxes, scores, classes = decode_and_nms(y, conf=0.15)
    assert len(boxes) == 0


def test_nms_suppresses_overlapping_same_class_detections():
    y = _make_raw_output([
        (50, 50, 20, 20, 0, 0.9, 0),
        (52, 51, 20, 20, 0, 0.7, 0),  # heavily overlapping, same class, lower score
        (150, 150, 20, 20, 0, 0.6, 0),  # far away, should survive
    ])
    boxes, scores, classes = decode_and_nms(y, conf=0.15, iou=0.5)
    assert len(boxes) == 2
    assert scores[0] == 0.9  # highest score kept first
    assert 150 in boxes[:, 0] or 140 in boxes[:, 0]  # the far box survives (xyxy left edge ~140)


def test_nms_does_not_suppress_across_classes():
    """Per-class NMS: two different classes overlapping heavily should both survive."""
    y = _make_raw_output([
        (50, 50, 20, 20, 0, 0.9, 0),
        (50, 50, 20, 20, 1, 0.8, 0),
    ])
    boxes, scores, classes = decode_and_nms(y, conf=0.15, iou=0.5)
    assert len(boxes) == 2
    assert set(classes.tolist()) == {0, 1}
