"""
08_baseline_generic.py -- baseline that cannot memorise project-specific names (Person B, Step 3).

Problem found in Step 2: the first baseline's strongest tokens were names such as
PulseAudioMixer or WebContentsImpl. Those identify a project or file, not a bug.

Fix: keep an identifier only if it appears in at least --min-projects DIFFERENT projects
in the training data (keywords and library calls such as if, sizeof, memcpy, strcpy pass
easily). Every other identifier becomes the placeholder ID. Operators, numbers and
punctuation are kept. The list of kept identifiers is learned from TRAIN only.

Then the same TF-IDF + Logistic Regression is trained and compared with the first baseline,
on all validation rows and on validation rows with chrome and linux removed.
The test set is not opened.

Outputs
    models/baseline_generic/baseline.joblib, config.json
    reports/baseline_generic_val.md
Usage:   python Scripts/08_baseline_generic.py
Smoke:   python Scripts/08_baseline_generic.py --sample 8000
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[^\sA-Za-z0-9_]")
IDENT_RE = re.compile(r"[A-Za-z_]")


def tokens(code, max_chars):
    return TOKEN_RE.findall(code[:max_chars])


def to_generic(code, keep, max_chars):
    out = []
    for t in TOKEN_RE.findall(code[:max_chars]):
        out.append(t if (not IDENT_RE.match(t) or t in keep) else "ID")
    return " ".join(out)


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def prf(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return pr, rc, 2 * pr * rc / max(pr + rc, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model-dir", default="models/baseline_generic")
    ap.add_argument("--first-model-dir", default="models/baseline")
    ap.add_argument("--report", default="reports/baseline_generic_val.md")
    ap.add_argument("--min-projects", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=20000)
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--c-grid", default="0.1,1,10")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tr_p = os.path.join(a.data_dir, "bigvul_train.csv.gz"); va_p = os.path.join(a.data_dir, "bigvul_val.csv.gz")
    for p in (tr_p, va_p):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found.")
    os.makedirs(a.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)

    cols = ["code", "label", "project_key"]
    train = pd.read_csv(tr_p, usecols=cols); val = pd.read_csv(va_p, usecols=cols)
    if a.sample:
        train = train.sample(min(a.sample, len(train)), random_state=a.seed)
        val = val.sample(min(max(a.sample // 4, 1000), len(val)), random_state=a.seed)
    train["code"] = train["code"].fillna("").astype(str); val["code"] = val["code"].fillna("").astype(str)
    ytr, yva = train["label"].astype(int).values, val["label"].astype(int).values
    print(f"train {len(train):,} rows, val {len(val):,} rows", flush=True)

    # ---- which identifiers are generic? (train only)
    t0 = time.time()
    print("Counting in how many projects each identifier appears ...", flush=True)
    proj_count = Counter()
    for proj, grp in train.groupby("project_key"):
        seen = set()
        for code in grp["code"]:
            seen.update(t for t in tokens(code, a.max_chars) if IDENT_RE.match(t))
        proj_count.update(seen)
    keep = {t for t, n in proj_count.items() if n >= a.min_projects}
    print(f"  {len(proj_count):,} distinct identifiers; {len(keep):,} kept (in >= {a.min_projects} projects) "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("Rewriting code with generic tokens ...", flush=True)
    gtr = [to_generic(c, keep, a.max_chars) for c in train["code"]]
    gva = [to_generic(c, keep, a.max_chars) for c in val["code"]]
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2), min_df=a.min_df,
                          max_features=a.max_features, sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(gtr); Xva = vec.transform(gva)
    print(f"  {Xtr.shape[1]:,} features", flush=True)

    best = None; rows = []
    for c in [float(x) for x in a.c_grid.split(",")]:
        lr = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=200).fit(Xtr, ytr)
        s = lr.predict_proba(Xva)[:, 1]; ap_ = average_precision_score(yva, s)
        rows.append((c, ap_)); print(f"  C={c:<5} validation PR-AUC {ap_:.4f}", flush=True)
        if best is None or ap_ > best[1]:
            best = (c, ap_, lr, s)
    c_best, _, lr, s_gen = best
    thr = best_threshold(yva, s_gen)

    joblib.dump({"vectorizer": vec, "model": lr, "keep": keep, "mode": "generic"},
                os.path.join(a.model_dir, "baseline.joblib"), compress=3)
    cfg = {"model": "tfidf_logreg_generic", "C": c_best, "threshold": thr, "max_chars": a.max_chars,
           "min_projects": a.min_projects, "kept_identifiers": len(keep), "seed": a.seed,
           "val_pr_auc": float(best[1]), "trained_on_rows": int(len(train))}
    json.dump(cfg, open(os.path.join(a.model_dir, "config.json"), "w"), indent=2)

    # ---- compare with the first baseline, all val rows and val without chrome/linux
    subsets = {"all validation rows": np.ones(len(val), bool),
               "without chrome and linux": ~val["project_key"].isin(["chrome", "linux"]).values}
    first = None
    fp = os.path.join(a.first_model_dir, "baseline.joblib")
    if os.path.exists(fp):
        b = joblib.load(fp); fcfg = json.load(open(os.path.join(a.first_model_dir, "config.json")))
        s_first = b["model"].predict_proba(b["vectorizer"].transform(val["code"].str.slice(0, fcfg["max_chars"])))[:, 1]
        first = (s_first, fcfg["threshold"])

    md = ["# Generic-token baseline vs first baseline (validation data)", "",
          f"Generated by Scripts/08_baseline_generic.py. An identifier is kept only if it appears in at least "
          f"{a.min_projects} different training projects ({len(keep):,} of {len(proj_count):,} kept); all others become `ID`. "
          "The test set was not used.", "",
          "| Validation rows | Model | Rows | % vulnerable | PR-AUC | ROC-AUC | Precision | Recall | F1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, m in subsets.items():
        y = yva[m]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        entries = [("Generic tokens", s_gen[m], thr)]
        if first is not None:
            entries.insert(0, ("First baseline", first[0][m], first[1]))
        for mname, s, th in entries:
            pr, rc, f1 = prf(y, s, th)
            md.append(f"| {name} | {mname} | {m.sum():,} | {y.mean():.1%} | {average_precision_score(y, s):.3f} | "
                      f"{roc_auc_score(y, s):.3f} | {pr:.3f} | {rc:.3f} | {f1:.3f} |")
    md += ["", "Thresholds are each model's own best-F1 threshold on all validation rows. A random classifier's PR-AUC "
               "equals the % vulnerable. If the generic model is close to the first one, the first one was not relying "
               "much on project names; if it is lower, part of the first score was memorised names."]
    open(a.report, "w", encoding="utf-8").write("\n".join(md))
    print("\n" + "\n".join(md)); print(f"\nSaved {a.report} and {a.model_dir}/")


if __name__ == "__main__":
    main()