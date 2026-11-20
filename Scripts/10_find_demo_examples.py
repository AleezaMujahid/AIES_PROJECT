"""
10_find_demo_examples.py -- find REAL vulnerable functions that the saved model flags,
so you have good examples to paste into the CodeSentinel app for the demo and the viva.

It scores the vulnerable functions (label 1) in the validation split, keeps the ones the model
flags (score >= its threshold), and saves the highest-scoring ones that are a sensible size to paste.
It also saves a few safe functions that score very low, for contrast.

Writes  reports/demo_examples/vuln_01.c, vuln_02.c, ...   (paste these into the app)
        reports/demo_examples/safe_01.c, ...
        reports/demo_examples/summary.md

Usage:
    python Scripts/10_find_demo_examples.py
    python Scripts/10_find_demo_examples.py --top 10 --include-big-projects
"""
import argparse
import json
import os
import re
import sys

import joblib
import pandas as pd

TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[^\sA-Za-z0-9_]")
IDENT = re.compile(r"[A-Za-z_]")


def score_codes(codes, bundle, cfg):
    mc = cfg["max_chars"]
    if bundle.get("mode") == "generic":
        keep = bundle["keep"]
        texts = [" ".join(t if (not IDENT.match(t) or t in keep) else "ID" for t in TOK.findall(c[:mc]))
                 for c in codes]
    else:
        texts = [c[:mc] for c in codes]
    return bundle["model"].predict_proba(bundle["vectorizer"].transform(texts))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/baseline_generic")
    ap.add_argument("--data", default="data/processed/bigvul_val.csv.gz")
    ap.add_argument("--out-dir", default="reports/demo_examples")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--safe", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars-show", type=int, default=3000)
    ap.add_argument("--include-big-projects", action="store_true",
                    help="also allow chrome and linux functions (excluded by default)")
    a = ap.parse_args()

    p = os.path.join(a.model_dir, "baseline.joblib")
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found.")
    bundle = joblib.load(p)
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    thr = cfg["threshold"]
    if not os.path.exists(a.data):
        sys.exit(f"ERROR: {a.data} not found.")

    df = pd.read_csv(a.data, usecols=["sample_id", "project_key", "func_name", "code", "label"])
    df["code"] = df["code"].fillna("")
    print(f"Scoring {len(df):,} validation functions with {a.model_dir} (threshold {thr:.3f}) ...", flush=True)
    df["score"] = score_codes(df["code"].tolist(), bundle, cfg)

    v = df[df["label"] == 1]
    flagged_all = int((v["score"] >= thr).sum())
    print(f"Vulnerable functions flagged: {flagged_all:,} of {len(v):,} ({flagged_all / max(len(v), 1):.1%}) -- this is the recall.")

    ok = (df["code"].str.len() >= a.min_chars) & (df["code"].str.len() <= a.max_chars_show)
    if not a.include_big_projects:
        ok &= ~df["project_key"].isin(["chrome", "linux"])
    pool = df[ok]
    vuln = pool[(pool["label"] == 1) & (pool["score"] >= thr)].sort_values("score", ascending=False).head(a.top)
    safe = pool[(pool["label"] == 0)].sort_values("score").head(a.safe)
    if vuln.empty:
        sys.exit("No vulnerable function in the size range was flagged. Try --include-big-projects or a larger --max-chars-show.")

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for prefix, part in (("vuln", vuln), ("safe", safe)):
        for i, (_, r) in enumerate(part.iterrows(), 1):
            name = f"{prefix}_{i:02d}.c"
            with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
                f.write(r["code"].rstrip() + "\n")
            rows.append((name, r["project_key"], r["func_name"], r["score"], len(r["code"]), prefix))

    md = ["# Demo examples", "",
          f"Model: `{a.model_dir}`, threshold {thr:.3f}. Source: `{a.data}` (validation split). "
          f"Model score is not a probability.", "",
          f"On the whole validation split the model flags {flagged_all:,} of {len(v):,} vulnerable functions "
          f"({flagged_all / max(len(v), 1):.1%}).", "",
          "| File | Project | Function | Score | Characters | Truth |", "|---|---|---|---|---|---|"]
    for name, pk, fn, sc, n, pre in rows:
        md.append(f"| {name} | {pk} | {fn} | {sc:.3f} | {n:,} | {'vulnerable (label 1)' if pre == 'vuln' else 'safe (label 0)'} |")
    md += ["", "Paste a `vuln_*.c` file into the app: the model should flag it. Paste a `safe_*.c` file: it should not.",
           "Labels come from BigVul commit context, so a flagged example is not proof of a real bug; read it before presenting it."]
    with open(os.path.join(a.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\nSaved to {a.out_dir}/")


if __name__ == "__main__":
    main()