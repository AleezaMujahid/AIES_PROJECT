
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler


DATA = Path("DATA/processed")
OUT = Path("models/baseline_step16")
REPORT = Path("reports/step16")

TOKEN_PATTERN = (
    r"[A-Za-z_][A-Za-z0-9_]*|\d+|"
    r"->|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|"
    r"[^\sA-Za-z0-9_]"
)

FEATURE_NAMES = [
    "log_length",
    "log_lines",
    "log_tokens",
    "pointer_arrows",
    "dereferences",
    "array_access",
    "malloc_calls",
    "free_calls",
    "memcpy_calls",
    "strcpy_calls",
    "strncpy_calls",
    "sprintf_calls",
    "snprintf_calls",
    "scanf_calls",
    "if_count",
    "for_count",
    "while_count",
    "return_count",
    "null_checks",
    "comparison_ops",
]


def structure_features(code):
    s = code.fillna("").astype(str)

    patterns = [
        r"->",
        r"\*",
        r"\[",
        r"\bmalloc\s*\(",
        r"\bfree\s*\(",
        r"\bmemcpy\s*\(",
        r"\bstrcpy\s*\(",
        r"\bstrncpy\s*\(",
        r"\bsprintf\s*\(",
        r"\bsnprintf\s*\(",
        r"\bscanf\s*\(",
        r"\bif\s*\(",
        r"\bfor\s*\(",
        r"\bwhile\s*\(",
        r"\breturn\b",
        r"\bNULL\b|\bnullptr\b",
        r"==|!=|<=|>=",
    ]

    result = [
        np.log1p(s.str.len().to_numpy()),
        np.log1p(s.str.count(r"\n").to_numpy() + 1),
        np.log1p(
            s.str.count(r"\S+").to_numpy()
        ),
    ]

    for pattern in patterns:
        result.append(
            np.log1p(
                s.str.count(pattern).to_numpy()
            )
        )

    return np.column_stack(result).astype(np.float32)


def evaluate(y, score, threshold):
    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
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
        "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y, score):
    precision, recall, thresholds = (
        precision_recall_curve(y, score)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    # First priority:
    # Recall >= 50%, FPR <= 10%.
    # Among feasible thresholds, maximize F1.
    order = np.argsort(-f1)

    for idx in order:
        threshold = float(thresholds[idx])
        m = evaluate(y, score, threshold)

        if m["recall"] >= 0.50 and m["fpr"] <= 0.10:
            return threshold, m, True

    # If no threshold satisfies both constraints,
    # use the best validation F1.
    idx = int(np.argmax(f1))
    threshold = float(thresholds[idx])

    return (
        threshold,
        evaluate(y, score, threshold),
        False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=0
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    print("Loading training and validation datasets...")

    train = pd.read_csv(
        DATA / "bigvul_train.csv.gz",
        usecols=["code", "label"],
    )

    val = pd.read_csv(
        DATA / "bigvul_val.csv.gz",
        usecols=["code", "label"],
    )

    if args.sample:
        train = train.sample(
            min(args.sample, len(train)),
            random_state=42,
        )
        val = val.sample(
            min(max(args.sample // 4, 1000), len(val)),
            random_state=42,
        )
        print("SMOKE TEST: sampled results are not final.")

    y_train = train["label"].astype(int).to_numpy()
    y_val = val["label"].astype(int).to_numpy()

    code_train = (
        train["code"].fillna("").astype(str).str[:20000]
    )
    code_val = (
        val["code"].fillna("").astype(str).str[:20000]
    )

    print("Building word TF-IDF...")

    word_vec = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_features=200000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    Xw_train = word_vec.fit_transform(code_train)
    Xw_val = word_vec.transform(code_val)

    print("Building character TF-IDF...")

    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=200000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    Xc_train = char_vec.fit_transform(code_train)
    Xc_val = char_vec.transform(code_val)

    print("Building code structure features...")

    scaler = StandardScaler()

    Xs_train = scaler.fit_transform(
        structure_features(code_train)
    )
    Xs_val = scaler.transform(
        structure_features(code_val)
    )

    # Clip extreme values to reduce outlier influence.
    Xs_train = csr_matrix(
        np.clip(Xs_train, -5, 5)
    )
    Xs_val = csr_matrix(
        np.clip(Xs_val, -5, 5)
    )

    feature_sets = {
        "word_char": (
            hstack(
                [Xw_train, Xc_train],
                format="csr",
            ),
            hstack(
                [Xw_val, Xc_val],
                format="csr",
            ),
        ),
        "char_structure": (
            hstack(
                [Xc_train, Xs_train],
                format="csr",
            ),
            hstack(
                [Xc_val, Xs_val],
                format="csr",
            ),
        ),
        "word_char_structure": (
            hstack(
                [Xw_train, Xc_train, Xs_train],
                format="csr",
            ),
            hstack(
                [Xw_val, Xc_val, Xs_val],
                format="csr",
            ),
        ),
    }

    results = []
    best = None

    for name, (X_train, X_val) in feature_sets.items():
        print(f"\nTesting feature set: {name}")

        for C in [0.1, 1.0, 10.0]:
            for weight in [None, "balanced"]:

                model = LogisticRegression(
                    C=C,
                    class_weight=weight,
                    solver="liblinear",
                    max_iter=300,
                    random_state=42,
                )

                model.fit(X_train, y_train)

                scores = model.predict_proba(
                    X_val
                )[:, 1]

                threshold, metrics, feasible = (
                    choose_threshold(y_val, scores)
                )

                pr_auc = average_precision_score(
                    y_val, scores
                )
                roc_auc = roc_auc_score(
                    y_val, scores
                )

                row = {
                    "features": name,
                    "C": C,
                    "class_weight": str(weight),
                    "feasible": feasible,
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    **metrics,
                }

                results.append(row)

                print(
                    f"{name} C={C} weight={weight} "
                    f"recall={metrics['recall']:.3f} "
                    f"FPR={metrics['fpr']:.3f} "
                    f"F1={metrics['f1']:.3f} "
                    f"PR-AUC={pr_auc:.3f} "
                    f"feasible={feasible}"
                )

                # Prefer feasible configurations.
                # Then prefer F1, PR-AUC, and lower FPR.
                key = (
                    int(feasible),
                    metrics["f1"],
                    pr_auc,
                    -metrics["fpr"],
                )

                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "features": name,
                        "C": C,
                        "class_weight": weight,
                        "threshold": threshold,
                        "metrics": metrics,
                        "pr_auc": pr_auc,
                        "roc_auc": roc_auc,
                    }

    results_df = pd.DataFrame(results)

    results_df.sort_values(
        ["feasible", "f1", "pr_auc"],
        ascending=[False, False, False],
    ).to_csv(
        REPORT / "validation_experiments.csv",
        index=False,
    )

    print("\nBest validation configuration:")
    print(json.dumps(best, indent=2))

    # Retrain selected configuration on TRAIN only.
    # Validation remains the threshold-selection dataset.
    selected_train, _ = feature_sets[best["features"]]

    final_model = LogisticRegression(
        C=best["C"],
        class_weight=best["class_weight"],
        solver="liblinear",
        max_iter=300,
        random_state=42,
    )

    final_model.fit(selected_train, y_train)

    bundle = {
        "model": final_model,
        "word_vectorizer": word_vec,
        "char_vectorizer": char_vec,
        "structure_scaler": scaler,
        "feature_set": best["features"],
        "max_chars": 20000,
        "feature_names": FEATURE_NAMES,
    }

    joblib.dump(
        bundle,
        OUT / "baseline.joblib",
        compress=3,
    )

    config = {
        "model": "tfidf_logreg_step16",
        "feature_set": best["features"],
        "C": best["C"],
        "class_weight": best["class_weight"],
        "threshold": best["threshold"],
        "sampled_run": bool(args.sample),
        "max_chars": 20000,
        "validation_metrics": best["metrics"],
        "validation_pr_auc": best["pr_auc"],
        "validation_roc_auc": best["roc_auc"],
    }

    with open(
        OUT / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    print("\nSaved model:", OUT / "baseline.joblib")
    print("Saved config:", OUT / "config.json")
    print("Saved validation results:", REPORT)
    print(
        "\nIMPORTANT: No test or CVEfixes data "
        "was used for model selection."
    )


if __name__ == "__main__":
    main()
