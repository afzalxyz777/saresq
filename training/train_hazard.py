"""Train the MobileNetV2 scene hazard classifier on AIDER (Section 8.2).

    # runs in the TensorFlow venv (Python 3.12), not the training venv
    ./.venv-tf/bin/python training/train_hazard.py --data-dir data/raw/aider/AIDER

Two stages: train the new head with the backbone frozen, then unfreeze the
last third and fine-tune at 10x smaller LR. Verified against the real download:
6438 images across collapsed_building (511), fire (521), flooded_areas (526),
traffic_incident (485), normal (4390). The real folder is ``flooded_areas``,
not ``flood`` as Section 8.1's prose shorthand suggests -- caught by listing
the actual archive rather than trusting the spec's wording.

**Augmentation lives in the data pipeline, not in the model.** Keras
preprocessing layers can be placed inside the graph, and the original version
of this script did that. They are inactive at inference, but they still export:
the converter carries their random-number ops into the TFLite file, which then
ships dead weight to a 1 GB Pi and makes the quantised graph harder to read
when something goes wrong. ``preprocess_input`` *does* stay in the graph, on
purpose -- that one is not augmentation but the model's input contract, and
keeping it inside means the Pi feeds raw 0-255 pixels and cannot get the
normalisation subtly wrong.

**Class weighting is used here, unlike the fusion head.** AIDER is 68% normal,
and an unweighted model scores 68% by answering "normal" every time. These
five numbers are consumed as *features* by the fusion head, which recalibrates
them, so ranking matters more than absolute calibration -- the opposite of the
trade-off in ``train_fusion.py``, and the reason the two scripts differ.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

#: Order is the contract with saresq/detect/hazard.py. Passed explicitly to
#: Keras as class_names to override its default alphabetical ordering, which
#: would silently put "collapsed_building, fire, flooded_areas, normal,
#: traffic_incident" -- swapping the last two relative to the deployed code.
CLASSES = ["collapsed_building", "fire", "flooded_areas", "traffic_incident", "normal"]
IMG_SIZE = 224


def build_model(n_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # 0-255 -> [-1, 1]
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def augmenter() -> tf.keras.Sequential:
    """Nadir aerial augmentation, applied to the dataset, never to the model.

    RandomRotation is full-circle because a downward-looking camera has no
    canonical up -- a rotated flooded street is still a flooded street.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(1.0, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.2),
    ], name="aerial_augment")


def export_int8_tflite(model: tf.keras.Model, rep_ds, out_path: pathlib.Path) -> pathlib.Path:
    """Full-integer quantisation with a real calibration set.

    The representative dataset must be *real images*, not noise: the converter
    records the actual activation range at every layer, and calibrating on
    random input would set ranges that never occur in flight, wasting most of
    the int8 codebook and quietly costing accuracy.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative():
        for batch in rep_ds.take(100):
            images = batch[0] if isinstance(batch, tuple) else batch
            for i in range(tf.shape(images)[0]):
                yield [tf.expand_dims(tf.cast(images[i], tf.float32), 0)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(converter.convert())
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--data-dir", default=str(repo / "data" / "raw" / "aider" / "AIDER"))
    ap.add_argument("--head-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keras-out", default=str(repo / "models" / "mobilenetv2_aider_224.keras"))
    ap.add_argument("--tflite-out", default=str(repo / "models" / "mobilenetv2_aider_224_int8.tflite"))
    ap.add_argument("--report", default=str(repo / "results" / "hazard" / "hazard_classifier_report.md"))
    ap.add_argument("--report-only", action="store_true",
                    help="skip training and quantisation; load --keras-out and "
                         "regenerate the report and metrics from it. For when "
                         "training succeeded and only the reporting failed -- "
                         "which has now happened twice, and each retrain costs "
                         "~35 min for numbers the saved model already contains. "
                         "The split is rebuilt by this same code path, so the "
                         "validation set is identical (it is seeded).")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    common = dict(validation_split=0.2, seed=args.seed, image_size=(IMG_SIZE, IMG_SIZE),
                  batch_size=args.batch, class_names=CLASSES, label_mode="int")
    train_raw = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    # shuffle MUST stay on (the default) for both subsets.
    #
    # An earlier version passed shuffle=False here to make the report's row
    # order reproducible. That silently destroyed the experiment: in Keras,
    # shuffle=False also stops the *file list* being shuffled before the split,
    # and image_dataset_from_directory takes the LAST 20% of a class-ordered
    # list. Since `normal` is the last class and holds 4390 of 6433 images, the
    # entire 1286-image validation set came out as `normal` -- four empty rows
    # in the confusion matrix and a meaningless 97.8% accuracy.
    #
    # Determinism comes from `seed`, which is already in `common`, not from
    # disabling the shuffle.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset="validation", **common)

    train_labels = np.concatenate([y.numpy() for _, y in train_raw])
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_labels)
    class_weight = dict(enumerate(weights))
    # Captured HERE, next to the array it describes, rather than recomputed in
    # the report 70 lines below. That recomputation read a variable also called
    # `labels` which by then had been rebound to sklearn's list of class ids, so
    # `labels == CLASSES.index("normal")` compared a list to an int, produced a
    # plain False, and died on `.mean()` -- after training AND quantisation had
    # finished. Same failure shape as the six runs the `labels=` fix was for:
    # the expensive work completes and the reporting throws it away.
    normal_frac = float((train_labels == CLASSES.index("normal")).mean())

    aug = augmenter()
    train_ds = (train_raw
                .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    keras_out = pathlib.Path(args.keras_out)
    if args.report_only:
        if not keras_out.exists():
            raise SystemExit(f"--report-only needs an existing model at {keras_out}")
        print(f"report-only: loading {keras_out}, skipping training")
        model = tf.keras.models.load_model(keras_out)
        return finish(args, model, val_ds, normal_frac, skip_export=True)

    model, base = build_model(len(CLASSES))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, class_weight=class_weight)

    base.trainable = True
    for layer in base.layers[: len(base.layers) * 2 // 3]:
        layer.trainable = False
    # BatchNorm must stay in inference mode when fine-tuning a small dataset:
    # updating its running statistics from 32-image batches is the classic way
    # to make validation accuracy collapse while training accuracy climbs.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
              class_weight=class_weight,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_accuracy", patience=4, restore_best_weights=True)])

    keras_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(keras_out)
    return finish(args, model, val_ds, normal_frac, train_raw=train_raw)


def finish(args, model, val_ds, normal_frac, train_raw=None, skip_export=False):
    """Evaluate, quantise and write the report.

    Split out of main() so --report-only reaches it by the same route. The
    reporting is where this script has failed twice, always AFTER the expensive
    work; keeping one copy of it means a fix is a fix everywhere.
    """
    # Labels and predictions are gathered in ONE pass over val_ds. Iterating it
    # twice would be a correctness bug now that shuffling is back on: a shuffled
    # tf.data pipeline can re-order between iterations, which would pair each
    # prediction with some other image's label and produce a confusion matrix
    # that looks plausible and is pure noise.
    y_true_parts, y_pred_parts = [], []
    for x_batch, y_batch in val_ds:
        y_true_parts.append(y_batch.numpy())
        y_pred_parts.append(np.argmax(model.predict(x_batch, verbose=0), axis=1))
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    # labels= pinned explicitly. Without it sklearn infers the class set from
    # the data and raises "Number of classes, 4, does not match size of
    # target_names, 5" whenever a class happens to be absent -- which killed
    # six consecutive runs AFTER they had finished training, discarding ~2
    # hours of work each time. Being explicit also keeps the confusion matrix
    # 5x5 and aligned with CLASSES no matter what the split contains.
    # Named label_ids, not `labels`: the training-set label ARRAY is also in
    # scope and shadowing it is exactly what broke the report last time.
    label_ids = list(range(len(CLASSES)))
    present = sorted(set(y_true.tolist()))
    if len(present) < len(CLASSES):
        print(f"WARNING: validation split contains only classes {present} "
              f"of {label_ids} -- the split is not representative.")
    report = classification_report(y_true, y_pred, labels=label_ids,
                                   target_names=CLASSES, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    tflite_path = pathlib.Path(args.tflite_out)
    if skip_export:
        if not tflite_path.exists():
            raise SystemExit(f"--report-only found no export at {tflite_path}; "
                             f"re-run without --report-only to build it")
        print(f"report-only: keeping existing export {tflite_path}")
    else:
        tflite_path = export_int8_tflite(model, train_raw, tflite_path)
    size_mb = tflite_path.stat().st_size / 1e6

    out = pathlib.Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Hazard classifier (MobileNetV2 / AIDER)\n\n"
        f"Validation split: {len(y_true)} images. Class-weighted training "
        f"(AIDER is {100 * normal_frac:.0f}% `normal`).\n\n"
        f"```\n{report}\n```\n\nConfusion matrix (rows=true, cols=pred), order "
        f"{CLASSES}:\n\n```\n{cm}\n```\n\n"
        f"INT8 TFLite export: `{tflite_path}` ({size_mb:.2f} MB).\n"
    )
    (out.parent / "hazard_metrics.json").write_text(json.dumps({
        "val_images": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "classes": CLASSES,
        "confusion_matrix": cm.tolist(),
        "tflite_mb": size_mb,
    }, indent=2))

    print(report)
    print(f"exported {tflite_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
