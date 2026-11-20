
"""
Step 14 - Recall-focused vulnerability detection experiment.

Uses:
  DATA/processed/bigvul_train.csv.gz
  DATA/processed/bigvul_val.csv.gz

Does NOT use test or CVEfixes for training/model selection.

Run:
  python Scripts/14_train_recall_focused.py

Optional quick test:
  python Scripts/14_train_recall_focused.py --sample 8000

Outputs:
  models/baseline_recall_focused/baseline.joblib
  models/baseline_recall_focused/config.json
  reports/recall_focused/validation_results.csv
  reports/recall_focused/validation_summary.csv
  reports/recall_focused/validation_predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


LENGTH_BINS = [0, 200, 500, 1000, 2000, 5000, 10000, np.inf]


def prepare_data(path, sample=0, seed=42):
    df = pd.read_csv(path, usecols=["code", "label"])
    df["code"] = df["code"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    if sample:
        df = df.sample(
            n=min(sample, len(df)),
            random_state=seed,
        ).reset_index(drop=True)

    return df


def length_groups(codes):
    lengths = codes.str.len().to_numpy()

    return np.digitize(
        lengths,
        LENGTH_BINS[1:-1],
        right=True,
    )


def make_length_weights(train, strength=0.5):
    """
    Reweight vulnerable examples within each length bin.

    strength=0: no length adjustment
    strength=0.5: moderate adjustment
    strength=1: stronger adjustment

    Uses training labels only.
    """
    groups = length_groups(train["code"])
    labels = train["label"].to_numpy()

    weights = np.ones(len(train), dtype=np.float64)

    global_positive_rate = labels.mean()

    for group in np.unique(groups):
        mask = groups == group
        positives = mask & (labels == 1)

        n_group = int(mask.sum())
        n_positive = int(positives.sum())

        if n_positive == 0:
            continue

        local_positive_rate = n_positive / n_group

        multiplier = (
            global_positive_rate / local_positive_rate
        ) ** strength

        weights[positives] *= multiplier

    # Keep mean weight around 1 for comparability.
    weights /= weights.mean()

    return weights


def evaluate(y, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float((tn + tp) / len(y)),
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
        if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(
    y,
    scores,
    min_recall=0.50,
    max_fpr=0.10,
):
    # Validation-only threshold selection.
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 197),
                scores,
            ]
        )
    )

    rows = [
        evaluate(y, scores, t)
        for t in thresholds
    ]

    eligible = [
        row for row in rows
        if row["recall"] >= min_recall
        and row["fpr"] <= max_fpr
    ]

    if eligible:
        best = max(
            eligible,
            key=lambda row: (
                row["f1"],
                row["recall"],
                -row["fpr"],
            ),
        )
        return best, True

    # If constraints cannot be satisfied, choose the
    # closest trade-off without pretending targets were met.
    best = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -row["fpr"],
        ),
    )

    return best, False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="DATA/processed",
    )
    parser.add_argument(
        "--model-dir",
        default="models/baseline_recall_focused",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/recall_focused",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train = prepare_data(
        data_dir / "bigvul_train.csv.gz",
        args.sample,
        args.seed,
    )

    val = prepare_data(
        data_dir / "bigvul_val.csv.gz",
        max(args.sample // 4, 1000)
        if args.sample else 0,
        args.seed,
    )

    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training data needs both classes.")

    if len(np.unique(y_val)) != 2:
        raise ValueError("Validation data needs both classes.")

    print(
        f"Training rows: {len(train):,}, "
        f"validation rows: {len(val):,}"
    )

    train_code = train["code"].str.slice(
        0, args.max_chars
    )
    val_code = val["code"].str.slice(
        0, args.max_chars
    )

    print("\nBuilding word TF-IDF features...")

    word_vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[^\sA-Za-z0-9_]",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_features=100000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train_word = word_vec.fit_transform(train_code)
    X_val_word = word_vec.transform(val_code)

    print(
        f"Word features: {X_train_word.shape[1]:,}"
    )

    print("\nBuilding character TF-IDF features...")

    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=100000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train_char = char_vec.fit_transform(train_code)
    X_val_char = char_vec.transform(val_code)

    print(
        f"Character features: {X_train_char.shape[1]:,}"
    )

    X_train = hstack(
        [X_train_word, X_train_char],
        format="csr",
        dtype=np.float32,
    )

    X_val = hstack(
        [X_val_word, X_val_char],
        format="csr",
        dtype=np.float32,
    )

    print(
        f"\nCombined features: {X_train.shape[1]:,}"
    )

    results = []
    best_candidate = None

    strengths = [0.0, 0.5, 1.0]
    c_values = [0.1, 1.0, 10.0]
    class_weights = [None, "balanced"]

    for strength in strengths:
        weights = make_length_weights(
            train,
            strength=strength,
        )

        for class_weight in class_weights:
            for c_value in c_values:

                print(
                    f"\nTraining: strength={strength}, "
                    f"class_weight={class_weight}, "
                    f"C={c_value}",
                    flush=True,
                )

                model = LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=300,
                    random_state=args.seed,
                )

                model.fit(
                    X_train,
                    y_train,
                    sample_weight=weights,
                )

                scores = model.predict_proba(
                    X_val
                )[:, 1]

                metrics, feasible = choose_threshold(
                    y_val,
                    scores,
                    args.min_recall,
                    args.max_fpr,
                )

                pr_auc = average_precision_score(
                    y_val, scores
                )

                roc_auc = roc_auc_score(
                    y_val, scores
                )

                row = {
                    "strength": strength,
                    "class_weight": str(class_weight),
                    "C": c_value,
                    "feasible": feasible,
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    **metrics,
                }

                results.append(row)

                print(
                    f"Recall={metrics['recall']:.3f}, "
                    f"FPR={metrics['fpr']:.3f}, "
                    f"F1={metrics['f1']:.3f}, "
                    f"PR-AUC={pr_auc:.3f}, "
                    f"feasible={feasible}",
                    flush=True,
                )

                # Prefer models satisfying both constraints.
                # Within the same feasibility group,
                # maximize F1, then recall.
                candidate_key = (
                    int(feasible),
                    metrics["f1"],
                    metrics["recall"],
                    pr_auc,
                )

                if (
                    best_candidate is None
                    or candidate_key > best_candidate["key"]
                ):
                    best_candidate = {
                        "key": candidate_key,
                        "model": model,
                        "metrics": metrics,
                        "scores": scores.copy(),
                        "strength": strength,
                        "class_weight": class_weight,
                        "C": c_value,
                        "pr_auc": pr_auc,
                        "roc_auc": roc_auc,
                        "feasible": feasible,
                    }

    results_df = pd.DataFrame(results)

    results_df.sort_values(
        ["feasible", "f1", "recall"],
        ascending=[False, False, False],
    ).to_csv(
        report_dir / "validation_results.csv",
        index=False,
    )

    best = best_candidate

    summary = {
        "strength": best["strength"],
        "class_weight": str(best["class_weight"]),
        "C": best["C"],
        "pr_auc": best["pr_auc"],
        "roc_auc": best["roc_auc"],
        "feasible": best["feasible"],
        **best["metrics"],
    }

    pd.DataFrame([summary]).to_csv(
        report_dir / "validation_summary.csv",
        index=False,
    )

    predictions = pd.DataFrame({
        "y_true": y_val,
        "y_score": best["scores"],
        "y_pred": (
            best["scores"] >= best["metrics"]["threshold"]
        ).astype(int),
    })

    predictions.to_csv(
        report_dir / "validation_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "word_vectorizer": word_vec,
            "char_vectorizer": char_vec,
            "model": best["model"],
        },
        model_dir / "baseline.joblib",
        compress=3,
    )

    config = {
        "model": "word_char_tfidf_logreg",
        "threshold": best["metrics"]["threshold"],
        "max_chars": args.max_chars,
        "strength": best["strength"],
        "class_weight": best["class_weight"],
        "C": best["C"],
        "min_recall_target": args.min_recall,
        "max_fpr_target": args.max_fpr,
        "constraints_met": best["feasible"],
        "sampled_run": bool(args.sample),
        "seed": args.seed,
        "validation_metrics": best["metrics"],
        "validation_pr_auc": best["pr_auc"],
        "validation_roc_auc": best["roc_auc"],
    }

    with open(
        model_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\n" + "=" * 55)
    print("BEST VALIDATION MODEL")
    print("=" * 55)

    print(json.dumps(config, indent=2))

    if not best["feasible"]:
        print(
            "\nWARNING: No configuration met both "
            "recall and FPR targets."
        )

    print(
        "\nSaved model:", model_dir / "baseline.joblib"
    )

    print(
        "Saved reports:", report_dir
    )

    print(
        "\nIMPORTANT: Do not use test/CVEfixes "
        "to select a model or threshold."
    )


if __name__ == "__main__":
    main()
