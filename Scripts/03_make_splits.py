"""
03_make_splits.py  --  freeze the train / validation / test splits (Person A, last step).

Reads   data/interim/clean_unified.csv.gz   (made by 02_clean_data.py)
Writes  data/processed/bigvul_train.csv.gz
        data/processed/bigvul_val.csv.gz
        data/processed/bigvul_test.csv.gz
        data/processed/cvefixes_external.csv.gz   (second-source test set, never used for training)
        data/processed/split_manifest.csv         (sample_id -> split, for every kept row)
        reports/split_report.md                   (counts to copy into the report)

Rules
  1. BigVul is split BY COMMIT: every function from one commit lands in the same split,
     so the vulnerable and fixed versions of one change can never sit on both sides.
     Each commit is placed greedily so row counts AND vulnerable counts stay close to
     the 70/15/15 targets.
  2. CVEfixes rows whose commit also appears in BigVul are removed, so the CVEfixes
     set is a real outside test.
  3. The seed is fixed, so the same input always gives the same split.
  4. Checks at the end STOP the script if a commit, a sample or a code hash appears in
     more than one split.

The data files hold only what a model may use as input or for analysis. CVE/CWE ids stay
in split_manifest.csv ONLY, for error analysis. They must never be model inputs.

Usage (one line):
    python Scripts/03_make_splits.py
"""
import argparse
import gzip
import os
import random
import sys

import pandas as pd

OUT_COLS = ["sample_id", "source", "project_key", "commit_id", "date",
            "language", "func_name", "file", "code", "label"]

# Obvious name variants inside BigVul. Hosts/organisations (savannah, launchpad, ...)
# are left alone on purpose: we cannot tell which project they are.
ALIASES = {"php-src": "php", "postgresql": "postgres", "imagemagick6": "imagemagick"}


def project_key(source, project):
    p = "" if pd.isna(project) else str(project).strip().rstrip("/")
    if source == "cvefixes":
        p = p.split("/")[-1]
    p = p.lower()
    return ALIASES.get(p, p)


def greedy_group_split(groups, ratios, seed):
    """groups: DataFrame [group, n, v]. Returns dict group -> split index."""
    total_n, total_v = groups["n"].sum(), groups["v"].sum()
    tn = [max(total_n * r, 1e-9) for r in ratios]
    tv = [max(total_v * r, 1e-9) for r in ratios]
    cur_n, cur_v = [0] * len(ratios), [0] * len(ratios)

    items = list(zip(groups["group"], groups["n"], groups["v"]))
    random.Random(seed).shuffle(items)
    items.sort(key=lambda x: -x[1])  # big groups first; shuffle above breaks ties fairly

    placed_n = placed_v = 0
    assign = {}
    for g, n, v in items:
        placed_n += n
        placed_v += v
        pn = placed_n / max(total_n, 1)   # share of all rows placed so far
        pv = placed_v / max(total_v, 1)   # share of all vulnerable rows placed so far
        best, best_cost = 0, None
        for s in range(len(ratios)):
            # try the group in split s; measure how uneven the fill levels would be
            cost = 0.0
            for t in range(len(ratios)):
                dn = cur_n[t] + (n if t == s else 0)
                dv = cur_v[t] + (v if t == s else 0)
                cost += (dn / tn[t] - pn) ** 2 + (dv / tv[t] - pv) ** 2
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        assign[g] = best
        cur_n[best] += n
        cur_v[best] += v
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/interim/clean_unified.csv.gz")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--report", default="reports/split_report.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--chunksize", type=int, default=20000)
    a = ap.parse_args()
    test_ratio = round(1 - a.train - a.val, 6)
    if test_ratio <= 0:
        sys.exit("ERROR: train + val must be less than 1")
    ratios = [a.train, a.val, test_ratio]
    names = ["train", "val", "test"]

    if not os.path.exists(a.clean):
        sys.exit(f"ERROR: {a.clean} not found. Run 02_clean_data.py first.")
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)

    # ---------- pass 1: small columns only (no code text)
    print("Reading ids ...", flush=True)
    meta = pd.read_csv(a.clean, dtype=str,
                       usecols=["sample_id", "source", "project", "commit_id", "label",
                                "code_hash", "cve_id", "cwe_id"])
    meta["label"] = meta["label"].astype(int)
    meta["project_key"] = [project_key(s, p) for s, p in zip(meta["source"], meta["project"])]

    bv = meta[meta["source"] == "bigvul"].copy()
    cf = meta[meta["source"] == "cvefixes"].copy()

    # ---------- CVEfixes: drop commits that also occur in BigVul
    bv_commits = set(bv["commit_id"].dropna())
    shared = cf["commit_id"].isin(bv_commits)
    cf_removed = int(shared.sum())
    cf = cf[~shared].copy()

    # ---------- BigVul: group by commit and split
    bv["group"] = bv["commit_id"].where(bv["commit_id"].notna() & (bv["commit_id"] != ""),
                                        "nocommit-" + bv["sample_id"])
    groups = (bv.groupby("group").agg(n=("label", "size"), v=("label", "sum")).reset_index())
    print(f"BigVul: {len(bv):,} rows in {len(groups):,} commit groups. Splitting ...", flush=True)
    assign = greedy_group_split(groups, ratios, a.seed)
    bv["split"] = bv["group"].map(assign).map(dict(enumerate(names)))
    cf["split"] = "cvefixes_external"

    manifest = pd.concat([bv, cf], ignore_index=True)

    # ---------- checks: nothing may leak across splits
    problems = []
    g = bv.groupby("group")["split"].nunique()
    if (g > 1).any():
        problems.append(f"{int((g > 1).sum())} BigVul commits appear in more than one split")
    if manifest["sample_id"].duplicated().any():
        problems.append("a sample_id appears twice")
    h = manifest.groupby("code_hash")["split"].nunique()
    if (h > 1).any():
        problems.append(f"{int((h > 1).sum())} code hashes appear in more than one split")
    if set(cf["commit_id"].dropna()) & bv_commits:
        problems.append("CVEfixes still shares commits with BigVul")
    if problems:
        sys.exit("SPLIT CHECK FAILED:\n  - " + "\n  - ".join(problems))
    print("Leak checks passed.", flush=True)

    manifest[["sample_id", "source", "split", "project_key", "commit_id",
              "label", "cve_id", "cwe_id"]].to_csv(
        os.path.join(a.out_dir, "split_manifest.csv"), index=False)

    # ---------- pass 2: stream the code text into one file per split
    split_of = dict(zip(manifest["sample_id"], manifest["split"]))
    key_of = dict(zip(manifest["sample_id"], manifest["project_key"]))
    files = {
        "train": "bigvul_train.csv.gz", "val": "bigvul_val.csv.gz", "test": "bigvul_test.csv.gz",
        "cvefixes_external": "cvefixes_external.csv.gz",
    }
    handles = {k: gzip.open(os.path.join(a.out_dir, v), "wt", encoding="utf-8", newline="")
               for k, v in files.items()}
    wrote = {k: 0 for k in files}
    first = {k: True for k in files}
    try:
        print("Writing split files ...", flush=True)
        for chunk in pd.read_csv(a.clean, dtype=str, chunksize=a.chunksize):
            chunk["split"] = chunk["sample_id"].map(split_of)
            chunk["project_key"] = chunk["sample_id"].map(key_of)
            for k, part in chunk.dropna(subset=["split"]).groupby("split"):
                part[OUT_COLS].to_csv(handles[k], index=False, header=first[k])
                first[k] = False
                wrote[k] += len(part)
    finally:
        for h_ in handles.values():
            h_.close()

    # ---------- report
    lines = ["# Split report", "",
             f"Generated by Scripts/03_make_splits.py (seed {a.seed}). BigVul is split by commit; "
             f"targets {a.train:.0%} / {a.val:.0%} / {test_ratio:.0%}.", "",
             "| Split | Rows | Vulnerable | % vulnerable | Commits | Projects |", "|---|---|---|---|---|---|"]
    for k in ["train", "val", "test", "cvefixes_external"]:
        d = manifest[manifest["split"] == k]
        if len(d):
            lines.append(f"| {k} | {len(d):,} | {int(d['label'].sum()):,} | "
                         f"{d['label'].mean():.1%} | {d['commit_id'].nunique():,} | "
                         f"{d['project_key'].nunique():,} |")
    lines += ["", f"- CVEfixes rows removed because their commit also appears in BigVul: **{cf_removed:,}**",
              f"- CVEfixes rows kept as the external test set: **{len(cf):,}**", "",
              "## Biggest projects per split (share of that split)", ""]
    for k in ["train", "val", "test", "cvefixes_external"]:
        d = manifest[manifest["split"] == k]
        if len(d):
            top = d["project_key"].value_counts(normalize=True).head(5)
            lines.append(f"- **{k}**: " + ", ".join(f"{p} {s:.0%}" for p, s in top.items()))
    lines += ["", "## Checks", "",
              "- No BigVul commit appears in more than one split: passed",
              "- No sample id or code hash appears in more than one split: passed",
              "- No CVEfixes commit also appears in BigVul: passed", "",
              "## Rules for Person B", "",
              "- Fit vectorizers, choose thresholds and tune only on train/val. Open test once.",
              "- Model input is the `code` column only. `commit_id`, `project_key`, `date` are for analysis.",
              "- CVE/CWE ids are in split_manifest.csv for error analysis only; never use them as inputs.",
              "- Report precision, recall, F1 and PR-AUC. Accuracy alone is misleading (about 95% by guessing 'safe').",
              "- Also report results with Chrome and linux rows excluded (filter on `project_key`)."]
    text = "\n".join(lines)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"\nSaved to {a.out_dir}/ and {a.report}")


if __name__ == "__main__":
    main()