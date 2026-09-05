"""Train the MobileNetV2 hazard classifier on AIDER (Section 8.2).

Two stages: (1) train the new head only, backbone frozen; (2) unfreeze the
last third of the backbone and fine-tune at 10x smaller LR. Class-balanced
via class_weight, since AIDER is ~2/3 "normal" and an unweighted model just
learns to always predict that.

    python training/train_hazard.py --data-dir data/raw/AIDER

Expects the AIDER layout: <data-dir>/<class_name>/*.jpg for each of
collapsed_building, fire, flood, traffic_incident, normal (verify this
against the actual download before trusting it -- see Section 19.1 task
0.12's own caution about verifying conversions before training on them).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

CLASSES = ["collapsed_building", "fire", "flood", "traffic_incident", "normal"]
IMG_SIZE = 224


def build_model(n_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet",
    )
    base.trainable = False

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(1.0),          # any angle: nadir has no gravity direction
        tf.keras.layers.RandomZoom(0.15),               # approximates 85-100% crops
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="models/mobilenetv2_aider_224.keras")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=0,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=args.batch, class_names=CLASSES,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=0,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=args.batch, class_names=CLASSES,
    )

    # Class weights from the training split's file counts (AIDER's public
    # release is roughly 2/3 "normal" -- Section 8.1).
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=labels)
    class_weight = dict(enumerate(weights))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model, base = build_model(len(CLASSES))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, class_weight=class_weight)

    # Stage 2: unfreeze the last third of the backbone, fine-tune at 10x smaller LR.
    base.trainable = True
    n_layers = len(base.layers)
    for layer in base.layers[: n_layers * 2 // 3]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, class_weight=class_weight)

    pathlib.Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    model.save(args.out)

    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    cm = confusion_matrix(y_true, y_pred)
    print(report)
    print("confusion matrix (rows=true, cols=pred):\n", cm)

    pathlib.Path("results/hazard_classifier_report.md").write_text(
        f"# Hazard classifier (MobileNetV2 / AIDER)\n\n```\n{report}\n```\n\nConfusion matrix:\n```\n{cm}\n```\n"
    )


if __name__ == "__main__":
    main()
