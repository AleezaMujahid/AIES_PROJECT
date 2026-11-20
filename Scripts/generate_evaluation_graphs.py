"""
generate_evaluation_graphs.py -- turn the result CSVs in reports/ into report-ready graphs.

Reads (whichever exist):
    reports/before_after.csv      reports/val_before_after.csv
    reports/test_results.csv      reports/cvefixes_results.csv
Writes PNGs to reports/figures/:
    eval_<file>_metrics.png     grouped bars: accuracy, precision, recall, F1, PR-AUC per model/stage
    eval_<file>_prauc.png       PR-AUC per row, with the chance level (= vulnerable share) as a dashed line
    eval_<file>_accuracy_vs_f1.png   shows why accuracy alone misleads
    eval_<file>_confusion.png   confusion matrices (only if the CSV has TN, FP, FN, TP columns)

It works out the columns by name, so it does not matter if they are called "PR-AUC", "pr_auc" or
"average_precision". Any text column (model, stage, ...) becomes the row label. A column called
split / dataset / set makes one figure per value. Percent strings like "94.8%" are understood.
If something looks wrong it prints the columns it found.

Usage (from the project folder):
    python Scripts/generate_evaluation_graphs.py
    python Scripts/generate_evaluation_graphs.py --files reports/test_results.csv --prevalence 0.052
"""
import argparse
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FILES = ["reports/before_after.csv", "reports/val_before_after.csv",
                 "reports/test_results.csv", "reports/cvefixes_results.csv"]
ALIASES = {
    "accuracy": ["accuracy", "acc"],
    "precision": ["precision", "prec"],
    "recall": ["recall", "tpr", "sensitivity"],
    "f1": ["f1", "f1_score", "f1score"],
    "pr_auc": ["pr_auc", "prauc", "average_precision", "ap", "auprc"],
    "roc_auc": ["roc_auc", "rocauc", "auc"],
}
NICE = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "pr_auc": "PR-AUC"}
BAR_METRICS = ["accuracy", "precision", "recall", "f1", "pr_auc"]
SPLIT_NAMES = {"split", "dataset", "set", "data", "eval_set"}
SKIP_TEXT = {"note", "notes", "comment", "comments", "path", "file"}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]


def norm(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def to_num(s):
    """Numbers or strings like '94.8%' -> floats (percent -> fraction). Non-numbers -> NaN."""
    def one(v):
        if isinstance(v, str):
            v = v.strip()
            pct = v.endswith("%")
            v = v.rstrip("%").replace(",", "")
            try:
                x = float(v)
            except ValueError:
                return np.nan
            return x / 100 if pct else x
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    return s.map(one)


def load(path):
    df = pd.read_csv(path)
    df.columns = [norm(c) for c in df.columns]
    colmap = {}
    for key, names in ALIASES.items():
        for n in names:
            if n in df.columns:
                colmap[key] = n
                break
    for key, col in colmap.items():
        df[col] = to_num(df[col])
    for c in ("tn", "fp", "fn", "tp"):
        if c in df.columns:
            df[c] = to_num(df[c])
    return df, colmap


def label_columns(df, colmap):
    used = set(colmap.values()) | {"tn", "fp", "fn", "tp"}
    out = []
    for c in df.columns:
        if c in used or c in SKIP_TEXT or c in SPLIT_NAMES:
            continue
        if to_num(df[c]).notna().sum() == 0:      # genuinely text
            out.append(c)
    return out


def make_labels(df, label_cols):
    if not label_cols:
        return [f"row {i + 1}" for i in range(len(df))]
    labs = []
    for _, r in df.iterrows():
        parts = []
        for c in label_cols:
            v = str(r[c]).strip()
            if v and v.lower() != "nan" and v not in parts:
                parts.append(v)
        labs.append(" | ".join(parts) if parts else "row")
    return labs


def wrap(s, n=26):
    return "\n".join(textwrap.wrap(s, n)) or s


def grouped_bars(df, labels, colmap, title, out, prevalence):
    metrics = [m for m in BAR_METRICS if m in colmap]
    if not metrics:
        return False
    n = len(df)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(metrics) + 0.5 * n), 5.2))
    w = 0.8 / max(n, 1)
    x = np.arange(len(metrics))
    for i in range(n):
        vals = [df.iloc[i][colmap[m]] for m in metrics]
        bars = ax.bar(x + i * w - 0.4 + w / 2, [0 if pd.isna(v) else v for v in vals], w,
                      label=wrap(labels[i]), color=COLORS[i % len(COLORS)])
        for b, v in zip(bars, vals):
            if not pd.isna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (0 to 1)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(n, 3), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def prauc_chart(df, labels, colmap, title, out, prevalence):
    if "pr_auc" not in colmap:
        return False
    vals = df[colmap["pr_auc"]].tolist()
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.55 * len(vals) + 1.4)))
    y = np.arange(len(vals))
    ax.barh(y, [0 if pd.isna(v) else v for v in vals], color=[COLORS[i % len(COLORS)] for i in range(len(vals))])
    for yi, v in zip(y, vals):
        if not pd.isna(v):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=4)
    ax.axvline(prevalence, color="#444", ls="--", lw=1, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(l, 34) for l in labels], fontsize=8)
    ax.set_ylim(len(vals) - 0.4, -1.0)          # room above the first bar for the chance label
    ax.text(prevalence + 0.002, -0.62, f"chance level {prevalence:.3f}", fontsize=8, va="center", color="#444")
    ax.set_xlabel("PR-AUC (higher is better)")
    ax.set_title(title)
    ax.set_xlim(0, max([v for v in vals if not pd.isna(v)] + [prevalence]) * 1.25)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def acc_vs_f1(df, labels, colmap, title, out, prevalence):
    if "accuracy" not in colmap or "f1" not in colmap:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for i in range(len(df)):
        a, f = df.iloc[i][colmap["accuracy"]], df.iloc[i][colmap["f1"]]
        if pd.isna(a) or pd.isna(f):
            continue
        ax.scatter(a, f, s=80, color=COLORS[i % len(COLORS)], zorder=3, label=wrap(labels[i], 30))
    ax.axvline(1 - prevalence, color="#888", ls="--", lw=1)
    top = max(0.5, float(np.nanmax(df[colmap["f1"]])) * 1.3)
    ax.text(1 - prevalence - 0.008, top * 0.98, "accuracy of 'always safe'", fontsize=7, color="#555",
            rotation=90, va="top", ha="right")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("F1 (vulnerable class)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, top)
    ax.set_title(title + ": high accuracy does not mean a good detector", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def confusion_grid(df, labels, title, out):
    if not all(c in df.columns for c in ("tn", "fp", "fn", "tp")):
        return False
    sub = df.dropna(subset=["tn", "fp", "fn", "tp"]).head(8)
    if sub.empty:
        return False
    k = len(sub)
    cols = min(4, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (idx, r) in zip(axes.ravel(), sub.iterrows()):
        m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
        norm_m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.axis("on")
        ax.imshow(norm_m, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center", fontsize=9,
                        color="white" if norm_m[i, j] > 0.55 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred safe", "pred vuln"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["safe", "vulnerable"], fontsize=7)
        ax.set_title(wrap(labels[sub.index.get_loc(idx)], 24), fontsize=8)
    fig.suptitle(title + " (shade = share of each true class)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--prevalence", type=float, default=0.052,
                    help="share of vulnerable functions = PR-AUC of a random guess (default 0.052)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    found = [p for p in a.files if os.path.exists(p)]
    if not found:
        sys.exit("ERROR: none of these files exist: " + ", ".join(a.files))
    made = []
    for path in found:
        stem = os.path.splitext(os.path.basename(path))[0]
        df, colmap = load(path)
        print(f"\n{path}: columns {list(df.columns)}")
        if not colmap:
            print("  no metric columns found (accuracy, precision, recall, f1, pr_auc); skipped. "
                  "Paste me this line and I will adjust the script.")
            continue
        split_col = next((c for c in df.columns if c in SPLIT_NAMES), None)
        groups = [(None, df)] if split_col is None or df[split_col].nunique() < 2 else \
                 [(str(v), g) for v, g in df.groupby(split_col, sort=False)]
        for gname, g in groups:
            g = g.reset_index(drop=True)
            labels = make_labels(g, label_columns(g, colmap))
            tag = f"{stem}" + (f"_{norm(gname)}" if gname else "")
            title = stem.replace("_", " ") + (f" ({gname})" if gname else "")
            for fn, suffix in ((grouped_bars, "metrics"), (prauc_chart, "prauc"), (acc_vs_f1, "accuracy_vs_f1")):
                out = os.path.join(a.out_dir, f"eval_{tag}_{suffix}.png")
                if fn(g, labels, colmap, title, out, a.prevalence):
                    made.append(out)
            out = os.path.join(a.out_dir, f"eval_{tag}_confusion.png")
            if confusion_grid(g, labels, title, out):
                made.append(out)
    print("\nSaved:")
    for m in made:
        print("  ", m)
    if not made:
        print("  (nothing: check the column names printed above)")


if __name__ == "__main__":
    main()