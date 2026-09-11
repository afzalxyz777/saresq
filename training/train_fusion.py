"""Fit the fusion head (Section 10.4) and export it as a plain dot product.

    python training/train_fusion.py --dataset results/fusion/fusion_dataset.npz

Writes ``models/fusion_logreg.json``: eighteen weights, a bias, and the prior
the ledger starts from. Deploying it requires no scikit-learn, no pickle and
no model format -- ``saresq/fuse/head.py`` loads the JSON and does one dot
product, which is the whole reason the head is a linear model rather than the
small MLP that would score a point or two higher offline.

Two decisions here are deliberate and both cut against the reflex:

**No class balancing.** The candidate population is heavily negative, so the
instinct is ``class_weight="balanced"``. That would be a mistake. The ledger
consumes ``logit(p_k)`` and accumulates it across passes (``saresq/fuse/
ledger.py``), so a systematic bias in p_k does not average out -- it compounds,
once per pass, in the same direction. Balancing deliberately distorts the
predicted base rate to buy recall at a 0.5 threshold, and we never threshold
at 0.5: the three-class rule and the ledger both read the probability itself.
Calibration is the product here; ranking is not enough.

**Grouped splitting.** Candidates from one frame share a background, an
ambient temperature and often a person, so a random row split would put
near-duplicates on both sides and report a fantasy score. Splitting by source
image is the honest version.

Standardisation is fitted for the optimiser's benefit and then folded back
into the exported weights, so the JSON acts directly on raw feature values
and the Pi never needs to know a scaler existed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from saresq.fuse.features import FEATURE_NAMES  # noqa: E402


def fold_scaler(weights: np.ndarray, bias: float, mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, float]:
    """Rewrite w.((x-mu)/s) + b as w'.x + b'.

    Exact, not an approximation: the two forms produce identical logits for
    every x. It exists so the deployed head is eighteen multiply-adds with no
    preprocessing step to keep in sync between training and the payload.
    """
    w_raw = weights / scale
    b_raw = float(bias - np.sum(weights * mean / scale))
    return w_raw, b_raw


def reliability_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Observed frequency vs. predicted probability, per bin.

    This is the table that says whether "0.8" means 0.8. For a head whose
    output is accumulated in log-odds it matters more than AUC, and it is the
    one number a judge can check against the deck's confidence language.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not in_bin.any():
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(in_bin.sum()),
            "predicted": float(p[in_bin].mean()),
            "observed": float(y_true[in_bin].mean()),
        })
    return rows


def expected_calibration_error(rows: list[dict], total: int) -> float:
    return sum(r["n"] / total * abs(r["predicted"] - r["observed"]) for r in rows) if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--dataset", default=str(repo / "results" / "fusion" / "fusion_dataset.npz"))
    ap.add_argument("--out", default=str(repo / "models" / "fusion_logreg.json"))
    ap.add_argument("--report", default=str(repo / "results" / "fusion" / "fusion_head_report.md"))
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    blob = np.load(args.dataset, allow_pickle=True)
    X, y, groups = blob["X"], blob["y"], blob["groups"]
    names = [str(n) for n in blob["feature_names"]]
    unavailable = {str(n) for n in blob["unavailable"]}
    if names != FEATURE_NAMES:
        raise SystemExit("dataset feature order does not match saresq.fuse.features.FEATURE_NAMES")
    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit(f"dataset has a single class ({y.sum()}/{len(y)} positive); nothing to fit")

    # Drop the structurally-unmeasurable columns before fitting rather than
    # after. Fitting them on constant input would produce a weight of exactly
    # zero *by luck of the regulariser* and a nonsense confidence interval;
    # excluding them says plainly that they are not calibrated yet.
    keep_idx = [i for i, n in enumerate(names) if n not in unavailable]
    X_fit = X[:, keep_idx]

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X_fit, y, groups))
    X_tr, X_te, y_tr, y_te = X_fit[train_idx], X_fit[test_idx], y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(
        C=args.C, max_iter=5000, solver="lbfgs",
        class_weight=None,  # deliberate -- see the module docstring
    ).fit(scaler.transform(X_tr), y_tr)

    p_te = model.predict_proba(scaler.transform(X_te))[:, 1]
    rows = reliability_table(y_te, p_te)
    metrics = {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "base_rate_train": float(y_tr.mean()), "base_rate_test": float(y_te.mean()),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "average_precision": float(average_precision_score(y_te, p_te)),
        "brier": float(brier_score_loss(y_te, p_te)),
        "brier_baseline": float(brier_score_loss(y_te, np.full_like(p_te, y_tr.mean()))),
        "ece": float(expected_calibration_error(rows, len(y_te))),
    }

    w_fit, b_raw = fold_scaler(model.coef_[0], float(model.intercept_[0]), scaler.mean_, scaler.scale_)

    # Re-expand to the full 18 in FEATURE_NAMES order, with explicit zeros
    # where the dataset could not measure the feature.
    weights = np.zeros(len(FEATURE_NAMES))
    for slot, i in enumerate(keep_idx):
        weights[i] = w_fit[slot]

    head = {
        "format": "saresq.fusion.logreg/1",
        "feature_names": FEATURE_NAMES,
        "weights": [float(v) for v in weights],
        "bias": b_raw,
        # The ledger's starting prior IS this population's base rate. Deriving
        # it here rather than leaving the 0.20 placeholder in pipeline.yaml is
        # the point of the exercise: every log-odds update is measured against
        # logit(pi_0), so a wrong prior biases every target on every pass.
        "pi_0": float(y_tr.mean()),
        "zeroed_features": sorted(unavailable),
        "zeroed_reason": (
            "Not measurable from a still-image dataset; calibrate from operator "
            "verdicts (training/export_verdicts.py) once flight data exists."
        ),
        "metrics": metrics,
        "source_dataset": str(pathlib.Path(args.dataset).name),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(head, indent=2))

    # Verify the fold: the exported dot product must reproduce sklearn exactly.
    # Note X[test_idx], not X_te: the exported vector is full-width (18), while
    # X_te carries only the 15 columns that were fitted. The three zeroed
    # weights contribute nothing, so the two agree exactly -- which is the
    # property being checked.
    z = X[test_idx] @ weights + b_raw
    p_check = 1.0 / (1.0 + np.exp(-z))
    max_dev = float(np.max(np.abs(p_check - p_te)))
    if max_dev > 1e-6:
        raise SystemExit(f"scaler fold is wrong: exported head deviates by {max_dev:.2e}")

    ranked = sorted(zip(FEATURE_NAMES, weights), key=lambda kv: -abs(kv[1]))
    lines = [
        "# Fusion head (logistic regression, 18 features)", "",
        f"Source: `{head['source_dataset']}` | grouped split by source image | "
        f"{metrics['n_train']} train / {metrics['n_test']} test candidates", "",
        f"- ROC AUC **{metrics['roc_auc']:.3f}**, average precision **{metrics['average_precision']:.3f}**",
        f"- Brier **{metrics['brier']:.4f}** vs. base-rate baseline {metrics['brier_baseline']:.4f}",
        f"- Expected calibration error **{metrics['ece']:.4f}**",
        f"- Ledger prior pi_0 = **{head['pi_0']:.4f}** (was a 0.20 placeholder in pipeline.yaml)", "",
        "## Weights (raw feature scale)", "", "| feature | weight |", "|---|---|",
    ]
    lines += [f"| `{n}` | {v:+.4f} |" for n, v in ranked]
    lines += ["", "## Calibration", "", "| bin | n | predicted | observed |", "|---|---|---|---|"]
    lines += [f"| {r['bin']} | {r['n']} | {r['predicted']:.3f} | {r['observed']:.3f} |" for r in rows]
    lines += ["", f"Zeroed, not fitted: {', '.join(head['zeroed_features'])} — {head['zeroed_reason']}", ""]
    report = pathlib.Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))

    print(json.dumps(metrics, indent=2))
    print(f"\nexported {out} (fold verified to {max_dev:.1e})\nreport {report}")


if __name__ == "__main__":
    main()
