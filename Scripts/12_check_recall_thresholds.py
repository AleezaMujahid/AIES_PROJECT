
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

MODEL_DIR = Path("models/baseline_improved")
DATA_DIR = Path("DATA/processed")
REPORT_DIR = Path("reports/recall_analysis")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(MODEL_DIR / "baseline.joblib")

with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
    config = json.load(f)

vectorizer = bundle["vectorizer"]
model = bundle["model"]

max_chars = config["max_chars"]

# Use validation data to investigate thresholds.
df = pd.read_csv(
    DATA_DIR / "bigvul_val.csv.gz",
    usecols=["code", "label"],
)

X = vectorizer.transform(
    df["code"]
    .fillna("")
    .astype(str)
    .str.slice(0, max_chars)
)

y = df["label"].astype(int).to_numpy()

scores = model.predict_proba(X)[:, 1]

results = []

for threshold in np.arange(0.05, 0.951, 0.025):

    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": round(float(threshold), 3),
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
        "fpr": fp / (fp + tn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "threshold_analysis.csv",
    index=False,
)

print("\nTHRESHOLDS WITH RECALL >= 50% AND FPR <= 10%:\n")

candidates = results_df[
    (results_df["recall"] >= 0.50)
    & (results_df["fpr"] <= 0.10)
].sort_values("f1", ascending=False)

if candidates.empty:
    print(
        "No threshold satisfies both constraints. "
        "The model may need further training changes."
    )
else:
    print(candidates.head(10).to_string(index=False))

print("\nTOP 10 THRESHOLDS BY F1:\n")

print(
    results_df.sort_values(
        "f1", ascending=False
    ).head(10).to_string(index=False)
)

print("\nSaved: reports/recall_analysis/threshold_analysis.csv")
