
"""Improve AIES TF-IDF + Logistic Regression without touching original model.

Run from project root:
    python Scripts/11_improve_baseline.py --data-dir DATA/processed

Selects configuration and threshold on BigVul validation ONLY. Evaluates
BigVul test and CVEfixes strict as untouched holdouts. Never trains on them.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TOKEN_PATTERN = (
    r"[A-Za-z_][A-Za-z0-9_]*|\d+|->|==|!=|<=|>=|"
    r"\+\+|--|&&|\|\||<<|>>|[^\sA-Za-z0-9_]"
)


def load(path, sample=0, seed=42):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    frame = pd.read_csv(path, usecols=["code", "label"])
    frame["code"] = frame["code"].fillna("").astype(str)
    frame["label"] = pd.to_numeric(
        frame["label"], errors="raise"
    ).astype(int)

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Labels must be 0/1 in {path}")

    if sample and len(frame) > sample:
        frame = frame.groupby(
            "label", group_keys=False
        ).sample(
            frac=sample / len(frame),
            random_state=seed,
        )

    return frame.reset_index(drop=True)


def scores_to_metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "fpr": float(fp / (fp + tn))
        if fp + tn else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pr_auc": float(
            average_precision_score(y, score)
        ),
        "roc_auc": float(
            roc_auc_score(y, score)
        ) if len(np.unique(y)) == 2 else None,
    }

    return result, pred


def threshold_search(y, score, max_fpr):
    """
    Select threshold using validation data only.

    Objective:
        Maximize F1 while keeping validation FPR
        at or below max_fpr.
    """

    thresholds = np.unique(
        np.r_[
            0.0,
            score,
            1.0,
            np.nextafter(1.0, 2.0),
        ]
    )

    negatives = (y == 0).sum()
    positives = (y == 1).sum()

    order = np.argsort(-score, kind="stable")

    ordered_score = score[order]
    ordered_y = y[order]

    cum_tp = np.cumsum(ordered_y == 1)
    cum_fp = np.cumsum(ordered_y == 0)

    indices = (
        np.searchsorted(
            -ordered_score,
            -thresholds,
            side="right",
        ) - 1
    )

    tp = np.where(
        indices >= 0,
        cum_tp[np.maximum(indices, 0)],
        0,
    )

    fp = np.where(
        indices >= 0,
        cum_fp[np.maximum(indices, 0)],
        0,
    )

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros(len(tp), dtype=float),
        where=(tp + fp) > 0,
    )

    recall = (
        tp / positives
        if positives
        else np.zeros(len(tp))
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp), dtype=float),
        where=(precision + recall) > 0,
    )

    fpr = (
        fp / negatives
        if negatives
        else np.zeros(len(tp))
    )

    eligible = np.where(
        fpr <= max_fpr + 1e-12
    )[0]

    if not len(eligible):
        raise ValueError(
            "No eligible threshold found"
        )

    # Highest F1; lower FPR breaks ties.
    chosen = eligible[
        np.lexsort(
            (fpr[eligible], -f1[eligible])
        )[0]
    ]

    return float(thresholds[chosen])


def make_vectorizer(kind, max_features):
    if kind == "word":
        return TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=3,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )

    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def save_cm(y, pred, title, path):
    cm = confusion_matrix(
        y, pred, labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(5.8, 4.5)
    )

    ax.imshow(cm, cmap="Blues")

    for (i, j), value in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=14,
            color=(
                "white"
                if value > cm.max() / 2
                else "black"
            ),
        )

    ax.set_xticks(
        [0, 1],
        ["Pred safe", "Pred vulnerable"],
    )

    ax.set_yticks(
        [0, 1],
        ["Actually safe", "Actually vulnerable"],
    )

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        default="DATA/processed",
    )

    ap.add_argument(
        "--output-dir",
        default="models/baseline_improved",
    )

    ap.add_argument(
        "--report-dir",
        default="reports/baseline_improved",
    )

    ap.add_argument(
        "--max-chars",
        type=int,
        default=20000,
    )

    ap.add_argument(
        "--max-features",
        type=int,
        default=200000,
    )

    ap.add_argument(
        "--max-val-fpr",
        type=float,
        default=0.05,
        help="Maximum validation false-positive rate",
    )

    ap.add_argument(
        "--sample",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    if not 0 <= args.max_val_fpr <= 1:
        ap.error(
            "--max-val-fpr must be between 0 and 1"
        )

    data = Path(args.data_dir)
    out = Path(args.output_dir)
    rep = Path(args.report_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rep.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read train and validation.
    tr = load(
        data / "bigvul_train.csv.gz",
        args.sample,
        args.seed,
    )

    va = load(
        data / "bigvul_val.csv.gz"
    )

    if args.sample:
        print(
            "SMOKE TEST: Do not use sampled "
            "results in final benchmarking."
        )

    ytr = tr.label.to_numpy()
    yva = va.label.to_numpy()

    if (
        len(np.unique(ytr)) < 2
        or len(np.unique(yva)) < 2
    ):
        raise ValueError(
            "Train and validation must "
            "contain both classes."
        )

    train_code = tr.code.str.slice(
        0, args.max_chars
    )

    val_code = va.code.str.slice(
        0, args.max_chars
    )

    best = None
    trials = []

    # Compare word and character TF-IDF.
    for kind in ["word", "char"]:
        print(
            f"\nBuilding {kind} TF-IDF features...",
            flush=True,
        )

        vec = make_vectorizer(
            kind, args.max_features
        )

        Xtr = vec.fit_transform(
            train_code
        )

        Xva = vec.transform(
            val_code
        )

        print(
            f"{kind}: {Xtr.shape[1]:,} features",
            flush=True,
        )

        # Different class weights and C values.
        for weight in [None, "balanced"]:
            for c in [0.1, 1.0, 10.0]:

                lr = LogisticRegression(
                    C=c,
                    class_weight=weight,
                    solver="liblinear",
                    max_iter=500,
                    random_state=args.seed,
                )

                lr.fit(Xtr, ytr)

                score = lr.predict_proba(
                    Xva
                )[:, 1]

                threshold = threshold_search(
                    yva,
                    score,
                    args.max_val_fpr,
                )

                m, _ = scores_to_metrics(
                    yva,
                    score,
                    threshold,
                )

                trial = {
                    "features": kind,
                    "C": c,
                    "class_weight": str(weight),
                    **m,
                }

                trials.append(trial)

                print(
                    f"{kind:4} "
                    f"C={c:<4} "
                    f"weight={str(weight):<8} "
                    f"val F1={m['f1']:.4f} "
                    f"FPR={m['fpr']:.4f} "
                    f"PR-AUC={m['pr_auc']:.4f} "
                    f"thr={threshold:.4f}",
                    flush=True,
                )

                # Select on validation only.
                key = (
                    m["f1"],
                    m["pr_auc"],
                    -m["fpr"],
                )

                if (
                    best is None
                    or key > best["key"]
                ):
                    best = {
                        "key": key,
                        "kind": kind,
                        "C": c,
                        "class_weight": weight,
                        "threshold": threshold,
                        "metrics": m,
                    }

        del Xtr, Xva, vec

    # Save all validation experiments.
    pd.DataFrame(trials).sort_values(
        ["f1", "pr_auc"],
        ascending=False,
    ).to_csv(
        rep / "validation_trials.csv",
        index=False,
    )

    print(
        "\nBest validation configuration:",
        best,
        flush=True,
    )

    # Refit selected configuration on TRAIN ONLY.
    vec = make_vectorizer(
        best["kind"],
        args.max_features,
    )

    Xtr = vec.fit_transform(
        train_code
    )

    lr = LogisticRegression(
        C=best["C"],
        class_weight=best["class_weight"],
        solver="liblinear",
        max_iter=500,
        random_state=args.seed,
    )

    lr.fit(Xtr, ytr)

    # Save improved model separately.
    joblib.dump(
        {
            "vectorizer": vec,
            "model": lr,
        },
        out / "baseline.joblib",
        compress=3,
    )

    config = {
        "model": "tfidf_logreg",
        "features": best["kind"],
        "C": best["C"],
        "class_weight": best["class_weight"],
        "threshold": best["threshold"],
        "max_chars": args.max_chars,
        "max_features": args.max_features,
        "max_val_fpr": args.max_val_fpr,
        "validation_metrics": best["metrics"],
        "train_rows": len(tr),
        "selection_data": "BigVul validation only",
        "seed": args.seed,
        "smoke_test": bool(args.sample),
    }

    (
        out / "config.json"
    ).write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    # Evaluate on untouched test datasets.
    datasets = [
        (
            "bigvul_test",
            data / "bigvul_test.csv.gz",
        ),
        (
            "cvefixes_strict",
            data / "cvefixes_external_strict.csv.gz",
        ),
    ]

    results = []

    for name, path in datasets:
        frame = load(path)

        y = frame.label.to_numpy()

        score = lr.predict_proba(
            vec.transform(
                frame.code.str.slice(
                    0, args.max_chars
                )
            )
        )[:, 1]

        metrics, pred = scores_to_metrics(
            y,
            score,
            best["threshold"],
        )

        results.append(
            {
                "dataset": name,
                **metrics,
            }
        )

        # Save predictions.
        pd.DataFrame(
            {
                "model": "TF-IDF + LR improved",
                "split": name,
                "y_true": y,
                "y_score": score,
                "y_pred": pred,
            }
        ).to_csv(
            rep / f"{name}_predictions.csv",
            index=False,
        )

        # Save confusion matrix image.
        save_cm(
            y,
            pred,
            f"Improved TF-IDF + LR - {name}",
            rep / f"{name}_confusion.png",
        )

        print(
            f"\n{name}: "
            f"{json.dumps(metrics, indent=2)}",
            flush=True,
        )

    pd.DataFrame(results).to_csv(
        rep / "holdout_metrics.csv",
        index=False,
    )

    print(
        f"\nSaved model: "
        f"{out / 'baseline.joblib'}"
    )

    print(
        f"Saved reports: {rep}"
    )

    print(
        "IMPORTANT: Do not tune using "
        "test or CVEfixes results."
    )


if __name__ == "__main__":
    main()
