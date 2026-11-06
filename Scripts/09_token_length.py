"""
09_token_length.py  --  how many functions are longer than CodeBERT's 512-token limit?

CodeBERT can read at most 512 tokens (we keep 2 slots for the start/end markers, so 510 of code).
Anything longer gets cut off. This script measures how often that happens, and whether the cut
hits vulnerable functions more than safe ones. The answer decides max_length for fine-tuning
and goes into the report.

Reads   data/processed/bigvul_train.csv.gz (and val/test if you pass --splits)
Writes  reports/token_length.md   and   reports/figures/token_length.png

Needs:  pip install transformers      (the tokenizer only, no torch needed)
        The first run downloads the CodeBERT tokenizer (about 1 MB) from Hugging Face.

Usage:
    python Scripts/09_token_length.py                  # train split, 30,000 random rows
    python Scripts/09_token_length.py --sample 0       # train split, ALL rows (slower)
    python Scripts/09_token_length.py --splits train val test --sample 0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

LIMIT = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--sample", type=int, default=30000, help="rows per split; 0 = all")
    ap.add_argument("--tokenizer", default="microsoft/codebert-base")
    ap.add_argument("--report", default="reports/token_length.md")
    ap.add_argument("--fig", default="reports/figures/token_length.png")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("ERROR: run  pip install transformers  first.")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    rows = []
    for sp in a.splits:
        path = os.path.join(a.data_dir, f"bigvul_{sp}.csv.gz")
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found.")
        df = pd.read_csv(path, usecols=["code", "label", "project_key"])
        if a.sample and len(df) > a.sample:
            df = df.sample(a.sample, random_state=a.seed)
        print(f"{sp}: tokenizing {len(df):,} functions ...", flush=True)
        codes = df["code"].fillna("").tolist()
        lens = []
        for i in range(0, len(codes), 2000):
            enc = tok(codes[i:i + 2000], add_special_tokens=True, truncation=False)["input_ids"]
            lens.extend(len(x) for x in enc)
        df = df.assign(split=sp, n_tokens=lens)
        rows.append(df)
    d = pd.concat(rows, ignore_index=True)
    d["too_long"] = d["n_tokens"] > LIMIT

    def line(name, x):
        q = x["n_tokens"].quantile([.5, .9, .95, .99])
        return (f"| {name} | {len(x):,} | {int(q[.5])} | {int(q[.9])} | {int(q[.95])} | {int(q[.99])} | "
                f"{x['n_tokens'].max():,} | {x['too_long'].mean():.1%} |")

    L = ["# CodeBERT token-length check", "",
         f"Tokenizer: `{a.tokenizer}`. Limit: {LIMIT} tokens (including the 2 special tokens). "
         f"Rows measured: {len(d):,}" + (f" (random sample of up to {a.sample:,} per split)" if a.sample else " (all rows)") + ".", "",
         "| Group | Functions | Median | 90th pct | 95th pct | 99th pct | Max | Longer than 512 |",
         "|---|---|---|---|---|---|---|---|", line("all", d)]
    for sp in a.splits:
        L.append(line(f"split: {sp}", d[d["split"] == sp]))
    L.append(line("label 0 (safe)", d[d["label"] == 0]))
    L.append(line("label 1 (vulnerable)", d[d["label"] == 1]))
    no_big = d[~d["project_key"].isin(["chrome", "linux"])]
    L.append(line("without chrome and linux", no_big))

    cut = d[d["too_long"]]
    L += ["", "## Does the cut hit vulnerable functions more?", "",
          f"- Vulnerable share among all measured functions: **{d['label'].mean():.1%}**",
          f"- Vulnerable share among functions that get cut: **{cut['label'].mean() if len(cut) else float('nan'):.1%}**",
          f"- Vulnerable share among functions that fit: **{d[~d['too_long']]['label'].mean():.1%}**", "",
          "## What this means", "",
          "- Functions over 512 tokens are truncated: CodeBERT sees only the first part and never the rest.",
          "- If a vulnerable line sits after the cut, the model cannot see it. Say this in the report as a limitation.",
          "- Use `max_length = 512` for fine-tuning (the model's hard limit). Longer is not possible without chunking."]
    text = "\n".join(L)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        top = max(2048, LIMIT * 2)
        ax.hist(np.clip(d["n_tokens"], 0, top), bins=80, color="#4C78A8")
        ax.axvline(LIMIT, color="#E45756", ls="--", label=f"CodeBERT limit ({LIMIT})")
        ax.set_xlabel(f"tokens per function (clipped at {top})")
        ax.set_ylabel("functions")
        ax.set_title("Function length in CodeBERT tokens")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.fig, dpi=150)
    except Exception as e:  # the plot is optional
        print("Plot skipped:", e)
    print("\n" + text)
    print(f"\nSaved {a.report} and {a.fig}")


if __name__ == "__main__":
    main()