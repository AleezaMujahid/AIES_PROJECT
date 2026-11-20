
"""
AIES Project - False Negative and Dataset Shift Investigation

Uses the saved improved TF-IDF + Logistic Regression model.
Does NOT train, tune, or modify the model.

Run:
    python Scripts/13_investigate_false_negatives.py

Outputs:
    reports/fn_investigation/
"""

import json
import re
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

DATA_DIR = Path("DATA/processed")
MODEL_DIR = Path("models/baseline_improved")
OUTPUT_DIR = Path("reports/fn_investigation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EXAMPLES = 30
RANDOM_SEED = 42


def normalize_code(code):
    return " ".join(str(code).split())


def code_hash(code):
    normalized = normalize_code(code)
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_dataset(filename, dataset_name):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset missing: {path}"
        )

    df = pd.read_csv(
        path,
        usecols=["code", "label"],
    )

    df["code"] = df["code"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["dataset"] = dataset_name

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Invalid labels in {filename}"
        )

    df["code_length"] = df["code"].str.len()

    df["token_count"] = df["code"].apply(
        lambda code: len(
            re.findall(r"\S+", code)
        )
    )

    df["code_hash"] = df["code"].apply(code_hash)

    return df


def evaluate_dataset(df, vectorizer, model, threshold, max_chars):
    X = vectorizer.transform(
        df["code"].str.slice(0, max_chars)
    )

    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= threshold).astype(int)

    result = df.copy()

    result["score"] = scores
    result["prediction"] = predictions

    result["outcome"] = np.select(
        [
            (result["label"] == 1)
            & (result["prediction"] == 1),

            (result["label"] == 1)
            & (result["prediction"] == 0),

            (result["label"] == 0)
            & (result["prediction"] == 1),

            (result["label"] == 0)
            & (result["prediction"] == 0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="UNKNOWN",
    )

    return result


def dataset_summary(df):
    return {
        "dataset": df["dataset"].iloc[0],
        "total": len(df),
        "safe": int((df["label"] == 0).sum()),
        "vulnerable": int((df["label"] == 1).sum()),
        "vulnerable_percent": round(
            100 * df["label"].mean(), 2
        ),
        "median_code_length": float(
            df["code_length"].median()
        ),
        "median_token_count": float(
            df["token_count"].median()
        ),
        "p95_code_length": float(
            df["code_length"].quantile(0.95)
        ),
    }


def performance_summary(df):
    y = df["label"].to_numpy()
    pred = df["prediction"].to_numpy()
    scores = df["score"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "dataset": df["dataset"].iloc[0],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
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
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores)
        if len(np.unique(y)) == 2 else np.nan,
    }


def length_analysis(df):
    bins = [0, 200, 500, 1000, 2000, 5000, 10000, np.inf]

    labels = [
        "0-200",
        "201-500",
        "501-1000",
        "1001-2000",
        "2001-5000",
        "5001-10000",
        "10000+",
    ]

    df = df.copy()

    df["length_group"] = pd.cut(
        df["code_length"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    vulnerable = df[df["label"] == 1]

    grouped = vulnerable.groupby(
        "length_group",
        observed=False,
    )

    summary = grouped.agg(
        vulnerable_count=("label", "size"),
        false_negatives=(
            "outcome",
            lambda x: int((x == "FN").sum()),
        ),
        true_positives=(
            "outcome",
            lambda x: int((x == "TP").sum()),
        ),
    ).reset_index()

    summary["fn_rate"] = (
        summary["false_negatives"]
        / summary["vulnerable_count"].replace(0, np.nan)
    )

    return summary


def plot_fn_rates(length_tables):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, table in length_tables.items():
        ax.plot(
            table["length_group"].astype(str),
            table["fn_rate"] * 100,
            marker="o",
            label=name,
        )

    ax.set_title(
        "False Negative Rate by Function Length"
    )

    ax.set_xlabel("Function length (characters)")
    ax.set_ylabel("Missed vulnerable functions (%)")

    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fn_rate_by_length.png",
        dpi=160,
    )

    plt.close(fig)


def plot_score_distributions(results):
    fig, axes = plt.subplots(
        1, len(results),
        figsize=(12, 4),
        squeeze=False,
    )

    for ax, (name, df) in zip(axes[0], results.items()):
        safe = df.loc[
            df["label"] == 0, "score"
        ]

        vulnerable = df.loc[
            df["label"] == 1, "score"
        ]

        ax.hist(
            safe,
            bins=30,
            alpha=0.55,
            density=True,
            label="Actually safe",
        )

        ax.hist(
            vulnerable,
            bins=30,
            alpha=0.55,
            density=True,
            label="Actually vulnerable",
        )

        ax.set_title(name)
        ax.set_xlabel("Predicted vulnerability score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "score_distributions.png",
        dpi=160,
    )

    plt.close(fig)


def save_fn_examples(df, dataset_name):
    missed = df[df["outcome"] == "FN"].copy()

    # Lowest scores = most confidently missed vulnerabilities.
    missed = missed.sort_values("score")

    examples = missed.head(MAX_EXAMPLES).copy()

    examples[
        [
            "dataset",
            "label",
            "prediction",
            "score",
            "code_length",
            "token_count",
            "code_hash",
            "code",
        ]
    ].to_csv(
        OUTPUT_DIR / f"{dataset_name}_false_negatives.csv",
        index=False,
    )


def overlap_analysis(train, test, name):
    train_hashes = set(train["code_hash"])

    matches = test["code_hash"].isin(train_hashes)

    return {
        "dataset": name,
        "total_rows": len(test),
        "exact_normalized_overlap": int(matches.sum()),
        "overlap_percent": round(
            100 * matches.mean(), 3
        ),
    }


def main():
    print("Loading saved model...")

    bundle = joblib.load(
        MODEL_DIR / "baseline.joblib"
    )

    config = json.loads(
        (MODEL_DIR / "config.json").read_text(
            encoding="utf-8"
        )
    )

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    threshold = float(config["threshold"])
    max_chars = int(config["max_chars"])

    print(f"Using threshold: {threshold:.4f}")

    print("Loading datasets...")

    train = load_dataset(
        "bigvul_train.csv.gz",
        "BigVul train",
    )

    bigvul = load_dataset(
        "bigvul_test.csv.gz",
        "BigVul test",
    )

    cvefixes = load_dataset(
        "cvefixes_external_strict.csv.gz",
        "CVEfixes strict",
    )

    print("Generating predictions...")

    results = {
        "BigVul test": evaluate_dataset(
            bigvul, vectorizer, model,
            threshold, max_chars,
        ),
        "CVEfixes strict": evaluate_dataset(
            cvefixes, vectorizer, model,
            threshold, max_chars,
        ),
    }

    summaries = []
    performances = []
    length_tables = {}
    overlaps = []

    for name, df in results.items():
        print(f"\nAnalyzing {name}...")

        summaries.append(dataset_summary(df))
        performances.append(performance_summary(df))

        length_table = length_analysis(df)
        length_tables[name] = length_table

        length_table.to_csv(
            OUTPUT_DIR
            / f"{name.lower().replace(' ', '_')}_length_analysis.csv",
            index=False,
        )

        save_fn_examples(
            df,
            name.lower().replace(" ", "_"),
        )

        overlaps.append(
            overlap_analysis(train, df, name)
        )

        # Keep row-level predictions for diagnosis.
        df.drop(
            columns=["code"],
        ).to_csv(
            OUTPUT_DIR
            / f"{name.lower().replace(' ', '_')}_diagnostics.csv",
            index=False,
        )

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False,
    )

    pd.DataFrame(performances).to_csv(
        OUTPUT_DIR / "performance_summary.csv",
        index=False,
    )

    pd.DataFrame(overlaps).to_csv(
        OUTPUT_DIR / "overlap_summary.csv",
        index=False,
    )

    plot_fn_rates(length_tables)
    plot_score_distributions(results)

    print("\nPERFORMANCE SUMMARY")
    print(
        pd.DataFrame(performances).to_string(
            index=False
        )
    )

    print("\nDATASET SUMMARY")
    print(
        pd.DataFrame(summaries).to_string(
            index=False
        )
    )

    print("\nTRAINING OVERLAP SUMMARY")
    print(
        pd.DataFrame(overlaps).to_string(
            index=False
        )
    )

    print(
        "\nDone! Reports saved in:",
        OUTPUT_DIR,
    )

    print(
        "\nNote: This script diagnoses errors. "
        "It does not improve the model automatically."
    )


if __name__ == "__main__":
    main()
