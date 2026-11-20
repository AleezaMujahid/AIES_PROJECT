
"""
Step 15: Final evaluation of the recall-focused model.

Run:
    python Scripts/15_evaluate_recall_focused.py

Uses the frozen model and threshold from Step 14.

Outputs:
    reports/recall_focused/final_metrics.csv
    reports/recall_focused/final_predictions.csv
    reports/recall_focused/final_confusion_matrices.png
    reports/recall_focused/final_metrics_comparison.png

IMPORTANT:
Do not change the model or threshold based on test results.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path("DATA/processed")
MODEL_DIR = Path("models/baseline_recall_focused")
REPORT_DIR = Path("reports/recall_focused")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "bigvul_test": "bigvul_test.csv.gz",
    "cvefixes_strict": "cvefixes_external_strict.csv.gz",
}


def load_model():
    bundle = joblib.load(
        MODEL_DIR / "baseline.joblib"
    )

    with open(
        MODEL_DIR / "config.json",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    if config.get("sampled_run", False):
        raise RuntimeError(
            "This is a sampled model. "
            "Run full Step 14 training first."
        )

    return bundle, config


def predict_dataset(df, bundle, config):
    max_chars = int(config["max_chars"])
    threshold = float(config["threshold"])

    code = (
        df["code"]
        .fillna("")
        .astype(str)
        .str.slice(0, max_chars)
    )

    X_word = bundle[
        "word_vectorizer"
    ].transform(code)

    X_char = bundle[
        "char_vectorizer"
    ].transform(code)

    X = hstack(
        [X_word, X_char],
        format="csr",
        dtype=np.float32,
    )

    scores = bundle["model"].predict_proba(X)[:, 1]
    predictions = (scores >= threshold).astype(int)

    return scores, predictions


def evaluate(y, scores, pred, dataset, threshold):
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "dataset": dataset,
        "model": "Word+Char TF-IDF + LogReg",
        "threshold": threshold,
        "total": len(y),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "fpr": fp / (fp + tn)
        if fp + tn else np.nan,
        "fnr": fn / (fn + tp)
        if fn + tp else np.nan,
        "pr_auc": average_precision_score(
            y, scores
        ),
        "roc_auc": roc_auc_score(
            y, scores
        ) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("Loading frozen Step 14 model...")

    bundle, config = load_model()

    threshold = float(config["threshold"])

    print(f"Frozen threshold: {threshold:.6f}")

    all_metrics = []
    all_predictions = []
    confusion_data = []

    for dataset, filename in DATASETS.items():
        print(f"\nEvaluating {dataset}...")

        df = pd.read_csv(
            DATA_DIR / filename,
            usecols=["code", "label"],
        )

        y = df["label"].astype(int).to_numpy()

        scores, pred = predict_dataset(
            df, bundle, config
        )

        metrics = evaluate(
            y, scores, pred,
            dataset, threshold,
        )

        all_metrics.append(metrics)

        predictions_df = pd.DataFrame({
            "dataset": dataset,
            "model": "recall_focused",
            "y_true": y,
            "y_score": scores,
            "y_pred": pred,
        })

        all_predictions.append(predictions_df)

        cm = confusion_matrix(
            y, pred, labels=[0, 1]
        )

        confusion_data.append((dataset, cm))

        print(json.dumps(metrics, indent=2))

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df.to_csv(
        REPORT_DIR / "final_metrics.csv",
        index=False,
    )

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_csv(
        REPORT_DIR / "final_predictions.csv",
        index=False,
    )

    # Confusion matrix diagrams
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
    )

    for ax, (dataset, cm) in zip(
        axes, confusion_data
    ):
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Safe", "Vulnerable"],
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d",
        )

        ax.set_title(dataset)

    fig.suptitle(
        "Recall-Focused Model: Final Confusion Matrices"
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "final_confusion_matrices.png",
        dpi=180,
    )

    plt.close(fig)

    # Performance comparison between datasets
    plot_df = metrics_df.set_index("dataset")

    fig, ax = plt.subplots(figsize=(9, 5))

    plot_df[
        ["precision", "recall", "f1", "fpr"]
    ].plot.bar(ax=ax)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(
        "Recall-Focused Model: Holdout Performance"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "final_metrics_comparison.png",
        dpi=180,
    )

    plt.close(fig)

    print("\n" + "=" * 60)
    print("FINAL HOLDOUT RESULTS")
    print("=" * 60)

    print(
        metrics_df[
            [
                "dataset",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fpr",
                "fn",
                "fp",
                "tp",
                "tn",
                "pr_auc",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    print("\nSaved reports in:", REPORT_DIR)
    print(
        "IMPORTANT: Do not tune using holdout results."
    )


if __name__ == "__main__":
    main()
