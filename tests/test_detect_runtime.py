"""Tests for the on-device inference layer.

These deliberately do not need a .tflite file. The quantisation arithmetic,
the letterbox transform and the fusion head are all pure functions, and they
are exactly the places where a bug is silent: a wrong scale or a mismatched
pad offset produces plausible boxes and plausible probabilities, never an
exception. Pinning them here means the only thing a real model export can
break is the export itself.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from saresq.detect.runtime import _Quant
from saresq.detect.tflite_detector import letterbox
from saresq.fuse.features import FEATURE_NAMES
from saresq.fuse.head import FORMAT, FusionHead

ULTRALYTICS_IMAGE_QUANT = _Quant(scale=1.0 / 255.0, zero_point=-128, dtype=np.dtype(np.int8))


class TestQuantisation:
    def test_the_int8_image_fast_path_is_exactly_the_slow_path(self):
        """The optimisation must be an identity, not an approximation.

        ``quantize_image_u8`` skips the divide-round-clip and does a single
        integer subtract. If that ever diverges from the general affine map --
        even by one level -- every crop the Pi sees is subtly different from
        what the model was calibrated on, and nothing anywhere would raise.
        """
        pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        fast = ULTRALYTICS_IMAGE_QUANT.quantize_image_u8(pixels)
        slow = ULTRALYTICS_IMAGE_QUANT.quantize(pixels.astype(np.float32) / 255.0)
        assert fast.dtype == np.int8
        np.testing.assert_array_equal(fast, slow)

    def test_the_fast_path_covers_the_full_int8_range(self):
        pixels = np.array([[0, 128, 255]], dtype=np.uint8)
        q = ULTRALYTICS_IMAGE_QUANT.quantize_image_u8(pixels)
        np.testing.assert_array_equal(q, np.array([[-128, 0, 127]], dtype=np.int8))

    def test_a_non_standard_scale_falls_back_instead_of_corrupting(self):
        """A model exported with a different input scale must not take the
        subtract path -- that is the failure this guard exists to prevent."""
        odd = _Quant(scale=0.5 / 255.0, zero_point=-100, dtype=np.dtype(np.int8))
        assert not odd.is_unit_byte_scale
        pixels = np.array([[0, 100, 200]], dtype=np.uint8)
        np.testing.assert_array_equal(
            odd.quantize_image_u8(pixels), odd.quantize(pixels.astype(np.float32) / 255.0)
        )

    def test_a_raw_pixel_graph_gets_raw_pixels(self):
        """The hazard classifier keeps ``preprocess_input`` inside its graph, so
        its input tensor is uint8 with scale 1.0 / zero-point 0 -- real units
        are 0-255, not 0-1.

        This is a regression test for a silent, total failure. The old code
        assumed a uint8 image tensor always meant a 0-1 graph and computed
        ``rint(pixels / 255)``, which is 0 below 128 and 1 above: the model saw
        a two-level near-black image and answered "normal" at full confidence
        for every input, its own training images included. Nothing raised, the
        INT8 file was correct, and three of the eighteen fusion features were
        identically zero because of it.
        """
        hazard = _Quant(scale=1.0, zero_point=0, dtype=np.dtype(np.uint8))
        assert hazard.image_real_max == 255.0
        pixels = np.array([[0, 1, 128, 200, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(hazard.quantize_image_u8(pixels), pixels)

    def test_a_unit_range_graph_still_normalises(self):
        """The other convention must keep working: Ultralytics' 0-1 input."""
        assert ULTRALYTICS_IMAGE_QUANT.image_real_max == pytest.approx(1.0)
        pixels = np.array([[0, 128, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(
            ULTRALYTICS_IMAGE_QUANT.quantize_image_u8(pixels),
            np.array([[-128, 0, 127]], dtype=np.int8),
        )

    def test_the_two_image_conventions_are_not_confusable(self):
        """The threshold sits between ranges that differ by 255x, so it cannot
        be tripped by an ordinary calibration wobble."""
        raw = _Quant(scale=1.0, zero_point=0, dtype=np.dtype(np.uint8))
        unit = ULTRALYTICS_IMAGE_QUANT
        assert unit.image_real_max < 2.0 < raw.image_real_max
        assert raw.image_real_max / unit.image_real_max > 100

    def test_quantise_dequantise_round_trips_within_one_level(self):
        q = _Quant(scale=0.02, zero_point=-7, dtype=np.dtype(np.int8))
        values = np.linspace(-2.0, 2.0, 101)
        back = q.dequantize(q.quantize(values))
        assert np.max(np.abs(back - values)) <= q.scale

    def test_a_float_tensor_is_passed_through_untouched(self):
        """scale == 0 is TFLite's marker for a float tensor, not a real scale."""
        q = _Quant(scale=0.0, zero_point=0, dtype=np.dtype(np.float32))
        assert not q.is_quantized
        values = np.array([0.25, -1.5, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(q.quantize(values), values)
        np.testing.assert_array_equal(q.dequantize(values), values)

    def test_quantisation_saturates_rather_than_wrapping(self):
        q = _Quant(scale=0.01, zero_point=0, dtype=np.dtype(np.int8))
        out = q.quantize(np.array([100.0, -100.0]))
        np.testing.assert_array_equal(out, np.array([127, -128], dtype=np.int8))


class TestLetterbox:
    def test_a_crop_already_at_model_size_is_not_touched(self):
        """The common case by construction: a 160 px crop into a 160 px model.
        It must be a no-op, or the cascade's efficiency argument is false."""
        crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        out, gain, pad_x, pad_y = letterbox(crop, 160)
        assert (gain, pad_x, pad_y) == (1.0, 0, 0)
        assert out is crop

    def test_a_wide_crop_is_padded_symmetrically_and_maps_back(self):
        crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        out, gain, pad_x, pad_y = letterbox(crop, 160)
        assert out.shape == (160, 160, 3)
        assert gain == pytest.approx(0.8)
        # A box on the source maps forward through (gain, pad) and back exactly.
        for x, y in [(0.0, 0.0), (200.0, 100.0), (37.5, 12.25)]:
            fx, fy = x * gain + pad_x, y * gain + pad_y
            assert (fx - pad_x) / gain == pytest.approx(x)
            assert (fy - pad_y) / gain == pytest.approx(y)

    def test_padding_is_grey_not_black(self):
        """Black padding reads as a large cold region to a thermal model --
        a strong feature, not neutral filler."""
        crop = np.full((80, 160, 3), 200, dtype=np.uint8)
        out, _, _, pad_y = letterbox(crop, 160)
        assert pad_y > 0
        assert np.all(out[0, :, :] == 114)


def _write_head(tmp_path, **overrides):
    blob = {
        "format": FORMAT,
        "feature_names": FEATURE_NAMES,
        "weights": [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
        "pi_0": 0.2,
    }
    blob.update(overrides)
    path = tmp_path / "head.json"
    path.write_text(json.dumps(blob))
    return path


class TestFusionHead:
    def test_a_zero_head_returns_one_half(self, tmp_path):
        head = FusionHead.load(_write_head(tmp_path))
        assert head.predict(np.zeros(len(FEATURE_NAMES))) == pytest.approx(0.5)

    def test_reordered_feature_names_are_rejected(self, tmp_path):
        """The bug this catches is otherwise invisible: every weight lands on
        the wrong feature and the head still returns confident probabilities."""
        scrambled = list(reversed(FEATURE_NAMES))
        with pytest.raises(ValueError, match="feature order"):
            FusionHead.load(_write_head(tmp_path, feature_names=scrambled))

    def test_a_short_weight_vector_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="18 weights"):
            FusionHead.load(_write_head(tmp_path, weights=[0.1] * 17))

    def test_an_unknown_format_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="format"):
            FusionHead.load(_write_head(tmp_path, format="something/2"))

    def test_extreme_logits_do_not_overflow_to_nan(self, tmp_path):
        """exp(+800) is inf, and inf/inf is nan -- which would turn the most
        confident detection in the mission into a silently dropped one."""
        weights = [100.0] + [0.0] * (len(FEATURE_NAMES) - 1)
        head = FusionHead.load(_write_head(tmp_path, weights=weights))
        big = np.zeros(len(FEATURE_NAMES)); big[0] = 10.0
        small = np.zeros(len(FEATURE_NAMES)); small[0] = -10.0
        assert head.predict(big) == pytest.approx(1.0)
        assert head.predict(small) == pytest.approx(0.0)
        assert np.isfinite([head.predict(big), head.predict(small)]).all()

    def test_explain_ranks_by_absolute_contribution_and_keeps_sign(self, tmp_path):
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("p_rgb")] = 2.0     # +2.0 * 1.0
        weights[FEATURE_NAMES.index("z_peak")] = -1.0   # -1.0 * 3.0 -> larger
        head = FusionHead.load(_write_head(tmp_path, weights=weights))
        x = np.zeros(len(FEATURE_NAMES))
        x[FEATURE_NAMES.index("p_rgb")] = 1.0
        x[FEATURE_NAMES.index("z_peak")] = 3.0
        top = head.explain(x, top=2)
        assert top[0] == ("z_peak", -3.0)
        assert top[1] == ("p_rgb", 2.0)

    def test_wrong_feature_count_at_predict_is_rejected(self, tmp_path):
        head = FusionHead.load(_write_head(tmp_path))
        with pytest.raises(ValueError, match="18 features"):
            head.predict(np.zeros(5))


class TestScalerFold:
    def test_folding_a_scaler_into_the_weights_is_exact(self):
        """train_fusion.py exports weights that act on RAW features, so the Pi
        needs no scaler. The rewrite must be exact for every input, not close."""
        from training.train_fusion import fold_scaler

        rng = np.random.default_rng(0)
        n = len(FEATURE_NAMES)
        w = rng.normal(size=n)
        b = 0.37
        mean = rng.normal(size=n) * 5
        scale = rng.uniform(0.5, 3.0, size=n)

        w_raw, b_raw = fold_scaler(w, b, mean, scale)
        for _ in range(50):
            x = rng.normal(size=n) * 4
            standardised = w @ ((x - mean) / scale) + b
            raw = w_raw @ x + b_raw
            assert raw == pytest.approx(standardised, rel=1e-12, abs=1e-12)
