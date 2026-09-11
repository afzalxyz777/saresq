"""A thin, allocation-free TFLite wrapper for the payload computer.

Everything here exists because of one constraint: the airframe carries a
**1 GB Raspberry Pi 4** with no accelerator, and the perception loop must keep
up with a survey flight. Three decisions follow from that, and each is worth
stating because a future session will otherwise "simplify" one of them away:

1. **Allocate once.** ``Interpreter.allocate_tensors()`` and the input/output
   buffers are set up in ``__init__`` and never again. Re-allocating per frame
   is the single easiest way to turn a 30 ms inference into a 90 ms one, and
   it is what almost every TFLite tutorial does.

2. **Write through ``set_tensor`` with a pre-quantised buffer.** We do the
   float->int8 conversion ourselves in NumPy rather than letting the runtime
   do it, because the conversion for a YOLOv8 export is exactly
   ``uint8_pixel - 128`` (scale 1/255, zero-point -128) -- a subtraction over
   the crop, not a multiply-add over it. ``_Quant.quantize`` detects that case
   and takes the cheap path.

3. **No batching.** TFLite's batch dimension is fixed at export time and the
   crop budget is 2-6 crops (``configs/pipeline.yaml``), so a batched export
   would have to be padded to the worst case and would waste work on the
   common case. We loop instead. The loop is in Python but the body is a
   single ``invoke()``, so the interpreter dominates.

The runtime import is deliberately forgiving: ``ai-edge-litert`` is what
Google ships for the Pi today, ``tflite_runtime`` is what older Pi guides
install, and ``tensorflow.lite`` is what a development laptop already has.
All three expose the same ``Interpreter`` surface we use.
"""
from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any

import numpy as np


def _load_interpreter_class():
    """Return a TFLite ``Interpreter`` class from whichever runtime is installed.

    Ordered cheapest-import first: on the Pi we want the 3 MB LiteRT wheel to
    win, never a 600 MB TensorFlow that happens to also be present.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        return Interpreter
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "No TFLite runtime found. On the Pi: pip install ai-edge-litert. "
            "On a laptop: pip install tensorflow."
        ) from exc


@dataclasses.dataclass(frozen=True)
class _Quant:
    """Affine quantisation parameters for one tensor.

    TFLite stores an int8/uint8 tensor as ``real = scale * (q - zero_point)``.
    ``scale == 0`` is TFLite's marker for "this tensor is float", not a
    degenerate scale, so it must be checked before dividing by it.
    """

    scale: float
    zero_point: int
    dtype: np.dtype

    @property
    def is_quantized(self) -> bool:
        return self.scale != 0.0 and self.dtype in (np.int8, np.uint8)

    @classmethod
    def from_detail(cls, detail: dict[str, Any]) -> "_Quant":
        scale, zero_point = detail["quantization"]
        return cls(float(scale), int(zero_point), np.dtype(detail["dtype"]))

    @property
    def is_unit_byte_scale(self) -> bool:
        """True when real values are exactly ``uint8_pixel / 255``.

        This is what Ultralytics emits for an INT8 image input (scale 1/255,
        zero-point -128), and it is the case ``quantize_image_u8`` exploits.
        """
        return (
            self.dtype == np.int8
            and abs(self.scale - 1.0 / 255.0) < 1e-9
            and self.zero_point == -128
        )

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Real values (in the model's own units) -> the model's dtype."""
        if not self.is_quantized:
            return x.astype(self.dtype, copy=False)
        q = np.rint(x / self.scale) + self.zero_point
        lo, hi = np.iinfo(self.dtype).min, np.iinfo(self.dtype).max
        return np.clip(q, lo, hi).astype(self.dtype)

    @property
    def image_real_max(self) -> float:
        """Largest real value this tensor can represent.

        This is how we tell what units the graph wants for an image, and it is
        not cosmetic -- see ``quantize_image_u8``. For a tensor calibrated on
        images the representable range is exactly the image range, so it comes
        out near 1.0 for a graph that wants 0-1, and near 255 for one that
        wants raw pixels.
        """
        hi = np.iinfo(self.dtype).max if self.is_quantized else 1
        return float(self.scale * (hi - self.zero_point)) if self.is_quantized else 1.0

    def quantize_image_u8(self, pixels: np.ndarray) -> np.ndarray:
        """Quantise raw ``uint8`` image pixels, skipping the float round-trip.

        The caller always hands us pixels in 0-255. What the GRAPH wants is not
        always 0-1, and assuming it was is a bug this cost real time:

        * Ultralytics' detector exports normalise inside the caller's units --
          real range 0-1 -- so the input must be ``pixels / 255``.
        * The hazard classifier keeps ``mobilenet_v2.preprocess_input`` INSIDE
          the graph, so its input tensor is raw 0-255 (uint8, scale 1.0,
          zero-point 0).

        Feeding the second one ``pixels / 255`` quantises to ``rint(p / 255)``,
        which is 0 for every pixel under 128 and 1 above it: the model receives
        a near-black two-level image and answers "normal" with full confidence,
        for every input, including the ones it was trained on. Nothing raises.
        The INT8 file was fine the whole time; three of the eighteen fusion
        features were dead because of this line.

        So the units are read off the tensor rather than assumed. The threshold
        is safe by a wide margin: the two conventions differ by 255x.
        """
        if self.is_quantized and self.image_real_max > 2.0:
            # Graph works in raw pixel units -- pass them through its own affine.
            return self.quantize(pixels.astype(np.float32))
        if self.is_unit_byte_scale:
            # 0-1 graph at the unit-byte scale: quantize(pixels / 255) collapses
            # to exactly ``pixels - 128`` -- no division, no rounding, no
            # clipping, and no intermediate float array the size of the crop. On
            # the Pi that is 76 KB touched per 160x160 crop instead of 307 KB.
            return (pixels.astype(np.int16, copy=False) - 128).astype(np.int8)
        return self.quantize(pixels.astype(np.float32) / 255.0)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        if not self.is_quantized:
            return q.astype(np.float32, copy=False)
        return (q.astype(np.float32) - self.zero_point) * self.scale


class TFLiteModel:
    """One loaded TFLite graph with its tensors allocated exactly once.

    ``infer`` takes and returns real (dequantised) values, so callers never
    see int8. ``last_ms`` carries the wall time of the most recent
    ``invoke()`` -- the latency numbers in the deck come from this field on
    the real Pi, not from a laptop estimate.
    """

    def __init__(self, model_path: str | pathlib.Path, num_threads: int = 2):
        self.path = pathlib.Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"TFLite model not found: {self.path}. Export it with "
                f"training/export_tflite.py; the Pi never builds these itself."
            )
        interpreter_cls = _load_interpreter_class()
        self._interpreter = interpreter_cls(model_path=str(self.path), num_threads=num_threads)
        self._interpreter.allocate_tensors()

        in_detail = self._interpreter.get_input_details()[0]
        out_details = self._interpreter.get_output_details()
        self._in_index = in_detail["index"]
        self._in_quant = _Quant.from_detail(in_detail)
        self.input_shape = tuple(int(v) for v in in_detail["shape"])
        self._out_indices = [d["index"] for d in out_details]
        self._out_quants = [_Quant.from_detail(d) for d in out_details]

        # Measured from the file, never assumed. Ultralytics' LiteRT export
        # path emits NCHW float32 inputs with int8 kept *internal* to the
        # graph, while its older onnx2tf path emitted NHWC int8. Both are
        # "an INT8 TFLite export of YOLOv8"; they differ in layout, dtype and
        # normalisation, and feeding one the other's tensor raises a shape
        # error at best and silently transposes an image at worst.
        if len(self.input_shape) != 4:
            raise ValueError(f"expected a 4-D image input, got {self.input_shape}")
        self.layout = "NCHW" if self.input_shape[1] == 3 else "NHWC"

        self.last_ms: float = 0.0

    @property
    def imgsz(self) -> int:
        """Square input side, read from whichever axis the layout puts it on."""
        return int(self.input_shape[2] if self.layout == "NCHW" else self.input_shape[1])

    def infer(self, x: np.ndarray) -> list[np.ndarray]:
        """Run one forward pass.

        ``x`` is always given NHWC batch-1, whatever the model wants -- images
        are NHWC everywhere else in this codebase and in OpenCV, so the
        transpose belongs here, once, rather than at every call site. ``uint8``
        is taken as raw image pixels; anything else as real values already in
        the model's units.
        """
        if x.dtype == np.uint8:
            q = self._in_quant.quantize_image_u8(x)
        else:
            q = self._in_quant.quantize(x)
        if self.layout == "NCHW":
            q = np.ascontiguousarray(q.transpose(0, 3, 1, 2))
        self._interpreter.set_tensor(self._in_index, q)
        t0 = time.perf_counter()
        self._interpreter.invoke()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return [
            quant.dequantize(self._interpreter.get_tensor(index))
            for index, quant in zip(self._out_indices, self._out_quants)
        ]

    def infer_one(self, x: np.ndarray) -> np.ndarray:
        """``infer`` for the single-output graphs (both of ours)."""
        return self.infer(x)[0]

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        q = "int8" if self._in_quant.is_quantized else "float"
        return f"<TFLiteModel {self.path.name} in={self.input_shape} {q}>"
