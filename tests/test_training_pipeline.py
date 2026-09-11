"""Tests for the pure logic inside the training scripts.

None of these need a GPU, a trained model or a dataset. They cover the parts
of the ML pipeline where a wrong answer looks completely normal: a crop window
that silently slides off frame, a degradation scale that puts people at the
wrong size, a sweep width that overstates coverage. Those errors do not raise
-- they just produce a number that goes on a slide.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.eval_pixels_on_target import read_yolo_labels, shrink_scene
from training.eval_sweep_width import edge_truncation_factor, recall_at_size, swath_width_m
from training.make_fusion_dataset import crop_window, degradation_scale, iou_matrix
from training.thermal_degradation import TARGET_PERSON_PX


class TestCropWindow:
    def test_a_centred_window_is_centred(self):
        assert crop_window(320, 240, 160, 640, 480) == (240, 160, 400, 320)

    @pytest.mark.parametrize("cx,cy", [(0, 0), (639, 479), (5, 470), (635, 3)])
    def test_a_window_near_any_edge_stays_fully_in_frame(self, cx, cy):
        x1, y1, x2, y2 = crop_window(cx, cy, 160, 640, 480)
        assert 0 <= x1 and 0 <= y1
        assert x2 <= 640 and y2 <= 480
        assert (x2 - x1, y2 - y1) == (160, 160)

    def test_the_window_slides_rather_than_clipping(self):
        """A clipped window would have to be padded, putting a synthetic border
        next to a target that sits near the frame edge. Sliding keeps every
        pixel real, at the cost of the target being off-centre."""
        x1, y1, x2, y2 = crop_window(10, 10, 160, 640, 480)
        assert (x1, y1) == (0, 0)
        assert (x2 - x1) == 160

    def test_a_frame_smaller_than_the_window_is_handled(self):
        x1, y1, x2, y2 = crop_window(50, 50, 160, 100, 80)
        assert (x1, y1, x2, y2) == (0, 0, 100, 80)


class TestDegradationScale:
    def test_a_typical_person_lands_at_the_target_pixel_size(self):
        """The whole MLX90640 simulation rests on this: RGBTDronePerson is
        matched to our sensor by ground sample distance, not resolution."""
        gt = np.array([[0, 0, 9, 15]], dtype=np.float32)  # the dataset's median box
        scale = degradation_scale(gt, fallback=0.1)
        sizes = np.array([9.0, 15.0])
        assert np.median(sizes) * scale == pytest.approx(TARGET_PERSON_PX)

    def test_the_median_not_the_mean_sets_the_scale(self):
        """One mislabelled or occluded box must not set the scale for a frame."""
        gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 900, 900]], dtype=np.float32)
        scale = degradation_scale(gt, fallback=0.1)
        assert np.median([10, 10, 900, 10, 10, 900]) == 10.0
        assert scale == pytest.approx(TARGET_PERSON_PX / 10.0)

    def test_an_image_with_no_people_uses_the_fallback(self):
        assert degradation_scale(np.zeros((0, 4)), fallback=0.17) == 0.17

    def test_the_scale_never_upsamples(self):
        """A person already smaller than the target must not be blown up --
        that would invent detail the sensor never had."""
        gt = np.array([[0, 0, 1, 1]], dtype=np.float32)
        assert degradation_scale(gt, fallback=0.1) == 1.0


class TestIoU:
    def test_identical_boxes_score_one(self):
        box = np.array([[10, 10, 20, 20]], dtype=np.float32)
        assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[50, 50, 60, 60]], dtype=np.float32)
        assert iou_matrix(a, b)[0, 0] == 0.0

    def test_a_known_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)   # area 100
        b = np.array([[5, 0, 15, 10]], dtype=np.float32)   # area 100, inter 50
        assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)

    def test_empty_inputs_give_an_empty_matrix(self):
        assert iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)


class TestShrinkScene:
    def test_padding_uses_the_scene_median_not_black(self):
        """On a thermal image black is not neutral filler -- it reads as a
        large very cold region, exactly the contrast the model hunts for."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        out = shrink_scene(img, 0.5)
        assert out.shape == img.shape
        assert out[-1, -1, 0] == 200        # padded area carries the median
        assert not (out == 0).any()

    def test_scale_one_is_a_no_op(self):
        img = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
        assert shrink_scene(img, 1.0) is img

    def test_the_scaled_content_occupies_the_expected_corner(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = 10
        img[0:50, 0:50] = 250
        out = shrink_scene(img, 0.5)
        assert out[0:25, 0:25].mean() > 200   # the bright quarter, halved
        assert out.shape == (100, 100, 3)


class TestYoloLabels:
    def test_only_the_person_class_is_read(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("0 0.5 0.5 0.1 0.2\n1 0.2 0.2 0.4 0.4\n2 0.9 0.9 0.1 0.1\n")
        boxes = read_yolo_labels(path, 100, 100)
        assert boxes.shape == (1, 4)
        np.testing.assert_allclose(boxes[0], [45, 40, 55, 60])

    def test_a_missing_label_file_is_an_empty_array_not_an_error(self, tmp_path):
        assert read_yolo_labels(tmp_path / "nope.txt", 100, 100).shape == (0, 4)

    def test_blank_lines_are_tolerated(self, tmp_path):
        path = tmp_path / "b.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n\n   \n")
        assert read_yolo_labels(path, 50, 50).shape == (1, 4)


class TestSweepWidthGeometry:
    def test_swath_matches_the_thermal_sensor_at_survey_altitude(self):
        """55 deg across-track at 20 m AGL. If this changes, every coverage
        number in the deck changes with it."""
        assert swath_width_m(55.0, 20.0) == pytest.approx(20.83, abs=0.01)

    def test_the_rgb_swath_is_wider_so_thermal_is_the_binding_constraint(self):
        assert swath_width_m(62.2, 20.0) > swath_width_m(55.0, 20.0)

    def test_edge_truncation_removes_one_target_width_in_total(self):
        assert edge_truncation_factor(32, 2.6) == pytest.approx(1 - 2.6 / 32)

    def test_a_target_wider_than_the_swath_gives_zero_not_a_negative(self):
        assert edge_truncation_factor(4, 10.0) == 0.0

    def test_recall_lookup_picks_the_bin_containing_the_size(self):
        rows = [
            {"variant": "p2", "scale": "1.0", "bin_px": "8-16", "recall": "0.61"},
            {"variant": "p2", "scale": "1.0", "bin_px": "32-inf", "recall": "0.93"},
            {"variant": "p3", "scale": "1.0", "bin_px": "32-inf", "recall": "0.88"},
        ]
        assert recall_at_size(rows, "p2", 115.5) == (0.93, "32-inf")
        assert recall_at_size(rows, "p2", 12.0) == (0.61, "8-16")
        assert recall_at_size(rows, "p3", 40.0) == (0.88, "32-inf")

    def test_a_downsampled_row_is_never_used_for_the_native_lookup(self):
        """scale < 1.0 rows describe a simulated higher altitude, not the
        survey altitude, and must not silently satisfy the lookup."""
        rows = [{"variant": "p2", "scale": "0.5", "bin_px": "32-inf", "recall": "0.4"}]
        with pytest.raises(SystemExit, match="eval_pixels_on_target"):
            recall_at_size(rows, "p2", 115.5)
