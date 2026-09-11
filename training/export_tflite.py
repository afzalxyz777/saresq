"""Export a trained detector to INT8 TFLite at each crop size (Section 7.8).

    # MUST run in the TensorFlow venv -- see the note on Python versions below
    ./.venv-tf/bin/python training/export_tflite.py \
        --weights results/detector/v8n_p2_hituav_640/weights/best.pt \
        --data data/converted/hituav/hituav.yaml --sizes 160 224 640

Never run on the Pi. The route is .pt -> ONNX -> TensorFlow -> TFLite, which
needs torch, onnx2tf and a full TensorFlow install; the payload carries only
the ~3 MB LiteRT runtime.

**Python version.** TensorFlow publishes no macOS-arm64 wheel for Python 3.13,
which is what ``.venv`` runs. Rather than pin the whole project back a version
for one build step, the export lives in ``.venv-tf`` on Python 3.12. That split
is the reason this file imports nothing from ``saresq``: it must run under an
interpreter the rest of the project does not use.

**Every export is verified before it is trusted.** ``verify_export`` loads the
file back with the same runtime the Pi will use, runs a real inference, and
reports size, latency and -- critically -- whether the box regressions came out
normalised or in pixels. That last one cannot be inferred from documentation:
it has changed between Ultralytics versions, it does not raise when wrong, and
getting it wrong scales every box by the input size. Measure it, print it, pin
it in the report.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np


def verify_export(path: pathlib.Path, runs: int = 20) -> dict:
    """Load an exported model with the deployment runtime and measure it."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # pragma: no cover - falls back inside the TF venv
        from tensorflow.lite import Interpreter  # type: ignore

    interpreter = Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    # Read the layout off the file. The LiteRT export path emits NCHW float32
    # (int8 kept internal to the graph); the older onnx2tf path emitted NHWC
    # int8. Both are valid "INT8 TFLite YOLOv8" exports.
    shape = tuple(int(v) for v in in_detail["shape"])
    layout = "NCHW" if shape[1] == 3 else "NHWC"
    size = shape[2] if layout == "NCHW" else shape[1]

    dtype = np.dtype(in_detail["dtype"])
    if dtype in (np.int8, np.uint8):
        fill = np.iinfo(dtype).min + 114 if dtype == np.int8 else 114
    else:
        fill = 114 / 255.0
    sample = np.full(shape, fill, dtype=dtype)

    interpreter.set_tensor(in_detail["index"], sample)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_detail["index"])
    scale, zero_point = out_detail["quantization"]
    real = (raw.astype(np.float32) - zero_point) * scale if scale else raw.astype(np.float32)

    y = np.squeeze(real)
    if y.shape[0] > y.shape[1]:
        y = y.T
    box_peak = float(np.max(np.abs(y[:4])))
    box_units = "normalized" if box_peak <= max(2.0, size * 0.05) else "pixels"

    # Latency on the build machine is NOT a Pi number and is labelled as such;
    # it is here to catch an export that is pathologically slow (a graph that
    # fell back to float, say), not to make a performance claim.
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        timings.append((time.perf_counter() - t0) * 1000.0)

    return {
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 3),
        "imgsz": size,
        "layout": layout,
        "input_shape": list(shape),
        "input_dtype": str(dtype),
        "input_quant": [float(in_detail["quantization"][0]), int(in_detail["quantization"][1])],
        "output_shape": [int(v) for v in out_detail["shape"]],
        "box_units": box_units,
        "box_peak_on_grey": round(box_peak, 3),
        "host_latency_ms_median": round(float(np.median(timings)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset YAML; source of INT8 calibration images")
    ap.add_argument("--sizes", type=int, nargs="+", default=[160, 224, 640])
    ap.add_argument("--fraction", type=float, default=0.25, help="share of images used for calibration")
    ap.add_argument("--models-dir", default=str(repo / "models"))
    ap.add_argument("--report", default=str(repo / "results" / "detector" / "tflite_exports.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    models_dir = pathlib.Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    variant = "p2" if "_p2_" in args.weights else "p3"

    records = []
    for size in args.sizes:
        model = YOLO(args.weights)  # reloaded per size: export mutates the model in place
        # quantize="w8a32" (int8 weights, float32 activations), NOT full int8.
        #
        # Measured on 150 HIT-UAV val images against the PyTorch model, same
        # letterbox and same decode for every row:
        #
        #   variant              size      Person AP50   % of ref   ms/frame
        #   PyTorch float32         --          0.7252     100.0%        --
        #   fp32 TFLite          12.26 MB       0.7252     100.0%      89.0
        #   w8a32 (this)          3.32 MB       0.7193      99.2%      40.4
        #   w8a16                 3.38 MB       0.7208      99.4%    1064.8
        #   full int8             3.33 MB       0.2315      31.9%      25.1
        #
        # Full int8 quantises the ACTIVATIONS too, and that is what breaks it:
        # the box head's internal 1/imgsz normalisation is itself quantised, so
        # the effective divisor came out ~636 instead of 640. Correcting the
        # scale recovers 0.2315 -> 0.5312, so roughly half the loss is that
        # constant and half is residual activation noise -- and at 9-45 px
        # people, a 1-2 px error is most of an IoU-0.5 match. None of it raises;
        # the file loads, runs fast, and returns plausible boxes.
        #
        # w8a32 keeps the int8 WEIGHTS, so the file is the same 3.3 MB, but
        # leaves activations in float32 and loses 0.8% instead of 68%. w8a16 is
        # equally accurate and 26x slower (no optimised int16 kernels), and fp32
        # is 3.7x the size for 0.8% more accuracy. w8a32 is the pick on
        # evidence, not preference.
        exported = model.export(
            format="litert", quantize="w8a32", imgsz=size, data=args.data,
            fraction=args.fraction,
            nms=False,  # NMS runs in NumPy on the Pi (saresq/detect/decode.py)
        )
        src = pathlib.Path(exported)
        dst = models_dir / f"yolov8n_{variant}_{size}_w8a32.tflite"
        dst.write_bytes(src.read_bytes())
        record = verify_export(dst)
        record["variant"] = variant
        record["source_weights"] = args.weights
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)

    units = {r["box_units"] for r in records}
    if len(units) > 1:
        raise SystemExit(f"exports disagree on box units {units}; the Pi cannot be configured for both")

    report = pathlib.Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"exports": records, "box_units": units.pop()}, indent=2))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
