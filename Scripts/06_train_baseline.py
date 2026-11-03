"""
06_train_baseline.py  --  Person B, Step 1: the baseline models.

Trains on  data/processed/bigvul_train.csv.gz
Tunes on   data/processed/bigvul_val.csv.gz      (the TEST file is never opened here)

Two baselines:
  1. Majority class: always answers "not vulnerable". It shows what accuracy alone hides.
  2. TF-IDF + Logistic Regression:
       - TF-IDF turns each function into numbers: how often each code token appears,
         down-weighted for tokens that appear in almost every function.
       - Tokens keep punctuation and operators (->, ==, *, [ ]), because they matter in C.
       - Logistic Regression learns a weight per token and outputs a vulnerable probability.
       - class_weight="balanced" makes the rare vulnerable class count more.
       - C (how strongly to limit big weights) is chosen on validation data by PR-AUC.
       - The decision threshold (the score above which we say "vulnerable") is chosen on
         validation data to maximise F1.

Outputs
    models/baseline/baseline.joblib        vectorizer + model together
    models/baseline/config.json            settings and the chosen threshold
    reports/baseline_val.md                validation results (copy into the report)
    reports/figures/baseline_pr_curve.png  precision-recall curve on validation data

Usage (one line):
    python Scripts/06_train_baseline.py
Quick smoke test on a small slice first (about a minute):
    python Scripts/06_train_baseline.py --sample 8000
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

# One token = an identifier, a number, a two-character operator, or any other single symbol.
TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[^\sA-Za-z0-9_]"


def clip(code, n):
    return code.fillna("").astype(str).str.slice(0, n)


def metrics_at(y, score, thr):
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(t[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models/baseline")
    ap.add_argument("--report", default="reports/baseline_val.md")
    ap.add_argument("--fig", default="reports/figures/baseline_pr_curve.png")
    ap.add_argument("--max-chars", type=int, default=20000, help="cut very long functions")
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--c-grid", default="0.1,1,10")
    ap.add_argument("--sample", type=int, default=0, help="smoke test: use only this many train rows")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tr_p = os.path.join(a.data_dir, "bigvul_train.csv.gz")
    va_p = os.path.join(a.data_dir, "bigvul_val.csv.gz")
    for p in (tr_p, va_p):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Finish Person A's steps first.")
    os.makedirs(a.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)

    print("Reading train and validation data ...", flush=True)
    cols = ["code", "label"]
    train = pd.read_csv(tr_p, usecols=cols)
    val = pd.read_csv(va_p, usecols=cols)
    if a.sample:
        train = train.sample(min(a.sample, len(train)), random_state=a.seed)
        val = val.sample(min(max(a.sample // 4, 1000), len(val)), random_state=a.seed)
    ytr, yva = train["label"].astype(int).values, val["label"].astype(int).values
    print(f"  train {len(train):,} rows ({ytr.mean():.1%} vulnerable); val {len(val):,} rows ({yva.mean():.1%})")

    # ---- baseline 1: majority class
    maj = {"accuracy": float(1 - yva.mean()), "precision": 0.0, "recall": 0.0, "f1": 0.0,
           "pr_auc": float(yva.mean()), "roc_auc": 0.5}

    # ---- baseline 2: TF-IDF + Logistic Regression
    t0 = time.time()
    print("Building TF-IDF features ...", flush=True)
    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, a.ngram_max),
                          min_df=a.min_df, max_features=a.max_features, sublinear_tf=True,
                          dtype=np.float32)
    Xtr = vec.fit_transform(clip(train["code"], a.max_chars))     # fitted on TRAIN only
    Xva = vec.transform(clip(val["code"], a.max_chars))
    print(f"  {Xtr.shape[1]:,} features, built in {time.time()-t0:.0f}s", flush=True)

    best = None
    rows = []
    for c in [float(x) for x in a.c_grid.split(",")]:
        t1 = time.time()
        lr = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=200)
        lr.fit(Xtr, ytr)
        s = lr.predict_proba(Xva)[:, 1]
        ap_ = average_precision_score(yva, s)
        rows.append((c, ap_, time.time() - t1))
        print(f"  C={c:<5} validation PR-AUC {ap_:.4f}  ({time.time()-t1:.0f}s)", flush=True)
        if best is None or ap_ > best[1]:
            best = (c, ap_, lr, s)
    c_best, ap_best, lr, sva = best

    thr = best_threshold(yva, sva)
    m05 = metrics_at(yva, sva, 0.5)
    mbest = metrics_at(yva, sva, thr)
    roc = roc_auc_score(yva, sva)

    joblib.dump({"vectorizer": vec, "model": lr}, os.path.join(a.model_dir, "baseline.joblib"), compress=3)
    cfg = {"model": "tfidf_logreg", "C": c_best, "threshold": thr, "max_chars": a.max_chars,
           "token_pattern": TOKEN_PATTERN, "ngram_range": [1, a.ngram_max], "min_df": a.min_df,
           "max_features": a.max_features, "class_weight": "balanced", "seed": a.seed,
           "val_pr_auc": ap_best, "val_prevalence": float(yva.mean()), "trained_on_rows": int(len(train))}
    with open(os.path.join(a.model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- PR curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, r, _ = precision_recall_curve(yva, sva)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, p, color="#c8553d", label=f"TF-IDF + LogReg (PR-AUC {ap_best:.3f})")
    ax.axhline(yva.mean(), color="gray", ls="--", label=f"majority / random ({yva.mean():.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Validation precision-recall curve")
    ax.legend(); fig.tight_layout(); fig.savefig(a.fig, dpi=150); plt.close(fig)

    # ---- report
    def row(name, m, pr, roc_):
        return (f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {pr:.3f} | {roc_:.3f} |")
    md = ["# Baseline results (validation data)", "",
          f"Generated by Scripts/06_train_baseline.py. Trained on {len(train):,} rows; validated on {len(val):,} rows "
          f"({yva.mean():.1%} vulnerable). The test set was not used.", "",
          "| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|---|---|---|",
          row("Majority class", maj, maj["pr_auc"], 0.5),
          row("TF-IDF + LogReg, threshold 0.5", m05, ap_best, roc),
          row(f"TF-IDF + LogReg, threshold {thr:.3f} (best F1)", mbest, ap_best, roc), "",
          "## Confusion matrix at the chosen threshold", "",
          "| | Predicted safe | Predicted vulnerable |", "|---|---|---|",
          f"| Actually safe | {mbest['tn']:,} | {mbest['fp']:,} |",
          f"| Actually vulnerable | {mbest['fn']:,} | {mbest['tp']:,} |", "",
          "## Regularisation search (validation PR-AUC)", "", "| C | PR-AUC | seconds |", "|---|---|---|"]
    md += [f"| {c} | {x:.4f} | {t:.0f} |" for c, x, t in rows]
    md += ["", f"Chosen C = {c_best}. A random or majority classifier has PR-AUC equal to the vulnerable share "
               f"({yva.mean():.3f}); anything clearly above that is real signal."]
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\nSaved model to {a.model_dir}/, report to {a.report}, chart to {a.fig}")


if __name__ == "__main__":
    main()