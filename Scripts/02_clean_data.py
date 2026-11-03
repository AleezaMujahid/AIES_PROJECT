"""
02_clean_data.py  --  Person A's cleaning pipeline.

Reads the RAW data (never modifies it), cleans it step by step, and keeps a
count after every step so the numbers can go straight into the report.

Inputs
    BigVul   : the .zip (or plain .csv) -- read directly, no extraction needed
    CVEfixes : cvefixes_cpp_functions.csv  (+ optional commit->CVE/CWE lookup)

Outputs (all new files; raw data is untouched)
    data/interim/clean_unified.csv.gz      cleaned data, one common schema
    data/interim/rejected_conflicts.csv.gz functions with conflicting labels (kept for review)
    data/interim/audit_table.csv           counts after every step
    data/interim/project_counts.csv        rows and vulnerable rows per project (to review the names)
    reports/cleaning_audit.md              the same table, ready for the report

Step 0 (both datasets): tidy commit ids. Some BigVul ids carry web leftovers
    ('...?w=1', '...#diff-...'), are shortened hashes, or are not hashes at all.
    The tidy id goes in `commit_id`, the original is kept in `commit_id_raw`.
Steps, in this order, for EACH dataset
    1. remove rows with empty / missing code
    2. remove rows with an invalid label
    3. keep only C and C++
    4. remove very short functions (default under 20 characters)
    5. conflicting labels: same code (ignoring whitespace) with different labels -> remove ALL of them
    6. exact duplicates (ignoring whitespace) -> keep the first
Then across datasets
    7. functions that appear in both BigVul and CVEfixes -> removed from CVEfixes
       (CVEfixes is the second-source evaluation set, so it must not overlap with BigVul)

Only the columns needed for modelling, grouping and auditing are kept. Columns that
reveal the answer (patch, func_after, vul_func_with_fix, Summary, commit_message...)
are deliberately NOT loaded.

Usage (one line):
    python Scripts/02_clean_data.py --bigvul data/raw/bigvul/MSR_data_cleaned_download.zip --cvefixes data/raw/cvefixes/cvefixes_cpp_functions.csv --cvefixes-map data/raw/cvefixes/cvefixes_commit_cve_cwe.csv

Quick smoke test on a small part (NOT representative, BigVul is ordered by commit):
    add  --max-rows 5000
"""
import argparse
import bisect
import hashlib
import os
import sys
import zipfile
from contextlib import contextmanager

import pandas as pd

BV_COLS = ["Unnamed: 0", "CVE ID", "CWE ID", "Publish Date", "commit_id",
           "project", "lang", "func_before", "vul"]
CF_COLS = ["method_change_id", "name", "code", "before_change", "filename",
           "programming_language", "commit_hash", "repo_url", "author_date"]

UNIFIED = ["sample_id", "source", "orig_id", "project", "commit_id", "commit_id_raw", "date", "date_source",
           "language", "func_name", "file", "code", "label", "label_provenance",
           "cve_id", "cwe_id", "code_hash"]

LANG_MAP = {"c": "C", "c++": "C++", "cpp": "C++", "cc": "C++"}


# ----------------------------------------------------------------- helpers
@contextmanager
def open_binary(path, member=None):
    """Open a plain CSV, or one CSV inside a .zip, as a stream (no extraction)."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            if member is None:
                names = [n for n in z.namelist()
                         if n.lower().endswith(".csv")
                         and not os.path.basename(n).startswith("._")
                         and "__MACOSX" not in n]
                if len(names) != 1:
                    sys.exit(f"ERROR: expected exactly one CSV in the zip, found {names}. "
                             f"Use --bigvul-member.")
                member = names[0]
            print(f"  (reading '{member}' directly from the zip)", flush=True)
            with z.open(member) as f:
                yield f
    else:
        with open(path, "rb") as f:
            yield f


def code_hash(text: str) -> str:
    """Hash of the code with all whitespace collapsed, so layout-only copies match."""
    return hashlib.md5(" ".join(text.split()).encode("utf-8", "ignore")).hexdigest()


class Audit:
    def __init__(self):
        self.rows = []

    def add(self, dataset, step, before, after, note=""):
        self.rows.append({"dataset": dataset, "step": step, "rows_before": before,
                          "removed": before - after, "rows_after": after, "note": note})

    def info(self, dataset, step, value, note=""):
        self.rows.append({"dataset": dataset, "step": step, "rows_before": "",
                          "removed": "", "rows_after": value, "note": note})

    def frame(self):
        return pd.DataFrame(self.rows)


def read_columns(path, usecols, member, chunksize, max_rows, name):
    with open_binary(path, member) as f:
        have = list(pd.read_csv(f, nrows=0).columns)
    missing = [c for c in usecols if c not in have]
    if missing:
        sys.exit(f"ERROR: {name} file is missing columns {missing}.\nColumns found: {have}")
    parts, total = [], 0
    with open_binary(path, member) as f:
        for chunk in pd.read_csv(f, usecols=usecols, dtype=str, chunksize=chunksize,
                                 low_memory=False):
            parts.append(chunk)
            total += len(chunk)
            if len(parts) % 20 == 0:
                print(f"  ... {total:,} rows read", flush=True)
            if max_rows and total >= max_rows:
                break
    df = pd.concat(parts, ignore_index=True)
    return df.iloc[:max_rows] if max_rows else df


# ------------------------------------------------------- to common schema
def bigvul_to_unified(df):
    return pd.DataFrame({
        "source": "bigvul",
        "orig_id": df["Unnamed: 0"],
        "project": df["project"],
        "commit_id": df["commit_id"],
        "date": df["Publish Date"],
        "date_source": "cve_publish_date",
        "language": df["lang"],
        "func_name": "",
        "file": "",
        "code": df["func_before"],
        "label": df["vul"],
        "label_provenance": "bigvul_vul_column",
        "cve_id": df["CVE ID"],
        "cwe_id": df["CWE ID"],
    })


def cvefixes_to_unified(df, mapping):
    lab = df["before_change"].astype(str).str.strip().str.lower().map({"true": "1", "false": "0"})
    u = pd.DataFrame({
        "source": "cvefixes",
        "orig_id": df["method_change_id"],
        "project": df["repo_url"],
        "commit_id": df["commit_hash"],
        "date": df["author_date"],
        "date_source": "commit_author_date",
        "language": df["programming_language"],
        "func_name": df["name"],
        "file": df["filename"],
        "code": df["code"],
        "label": lab,
        # PROVISIONAL: before_change=True is the dataset's own flag, not a verified label.
        "label_provenance": "cvefixes_before_change_provisional",
        "cve_id": "",
        "cwe_id": "",
    })
    if mapping is not None:
        m = mapping.dropna(subset=["commit_hash"]).copy()
        for col in ("cve_id", "cwe_id"):
            m[col] = m[col].fillna("")
        agg = m.groupby("commit_hash").agg(
            cve_id=("cve_id", lambda s: ";".join(sorted({x for x in s if x}))),
            cwe_id=("cwe_id", lambda s: ";".join(sorted({x for x in s if x}))))
        u["cve_id"] = u["commit_id"].map(agg["cve_id"]).fillna("")
        u["cwe_id"] = u["commit_id"].map(agg["cwe_id"]).fillna("")
    return u


def normalise_commit_ids(frames, audit):
    """Tidy commit ids in place. frames = [(name, unified_df), ...]"""
    for name, df in frames:
        df["commit_id_raw"] = df["commit_id"]
        raw = df["commit_id"].fillna("").astype(str).str.strip()
        m = raw.str.extract(r"^([0-9a-fA-F]{7,40})(?![0-9a-zA-Z])", expand=False)
        df["commit_id"] = m.str.lower().where(m.notna(), raw)

    pool = sorted({c for _, df in frames
                   for c in df.loc[df["commit_id"].str.fullmatch(r"[0-9a-f]{40}"), "commit_id"]})

    for name, df in frames:
        rawstrip = df["commit_id_raw"].fillna("").astype(str).str.strip()
        tidied = int((df["commit_id"] != rawstrip).sum())
        expanded = 0
        short = df["commit_id"].str.fullmatch(r"[0-9a-f]{7,39}")
        if short.any():
            lookup = {}
            for sid in df.loc[short, "commit_id"].unique():
                i, hits = bisect.bisect_left(pool, sid), []
                while i < len(pool) and pool[i].startswith(sid) and len(hits) < 2:
                    hits.append(pool[i])
                    i += 1
                if len(hits) == 1:          # only expand when exactly one full id matches
                    lookup[sid] = hits[0]
            before = df.loc[short, "commit_id"]
            df.loc[short, "commit_id"] = before.map(lambda x: lookup.get(x, x))
            expanded = int((df.loc[short, "commit_id"] != before).sum())
        not_hash = int((~df["commit_id"].str.fullmatch(r"[0-9a-f]{40}")).sum())
        audit.info(name, "info: commit ids tidied (suffix removed or lowercased)", tidied)
        audit.info(name, "info: shortened commit ids expanded to the full hash", expanded,
                   "only when exactly one full id starts with it")
        audit.info(name, "info: commit ids still not a full 40-character hash", not_hash,
                   "left as they are; they still work as a grouping key")


# ------------------------------------------------------------ cleaning
def clean_source(u, source, audit, min_chars):
    """Steps 1-6 for one dataset. Returns (clean_df, conflict_rows_df)."""
    u = u.copy()
    u["code"] = u["code"].fillna("").astype(str)
    n = len(u)
    audit.add(source, "0. input rows", n, n, "rows read from the raw file")

    # 1. empty code
    keep = u["code"].str.strip() != ""
    audit.add(source, "1. remove empty or missing code", len(u), int(keep.sum()))
    u = u[keep]

    # 2. invalid label
    u["label"] = pd.to_numeric(u["label"], errors="coerce")
    keep = u["label"].isin([0, 1])
    audit.add(source, "2. remove invalid labels", len(u), int(keep.sum()), "label must be 0 or 1")
    u = u[keep].copy()
    u["label"] = u["label"].astype(int)

    # 3. language
    lang = u["language"].fillna("").astype(str).str.strip().str.lower().map(LANG_MAP)
    keep = lang.notna()
    audit.add(source, "3. keep only C and C++", len(u), int(keep.sum()))
    u = u[keep].copy()
    u["language"] = lang[keep]

    # 4. very short functions
    keep = u["code"].str.strip().str.len() >= min_chars
    audit.add(source, f"4. remove code shorter than {min_chars} characters", len(u), int(keep.sum()))
    u = u[keep].copy()

    # 5. conflicting labels
    u["code_hash"] = u["code"].map(code_hash)
    nlab = u.groupby("code_hash")["label"].transform("nunique")
    conflict = nlab > 1
    conflicts = u[conflict].copy()
    n_groups = conflicts["code_hash"].nunique()
    audit.add(source, "5. remove conflicting labels", len(u), int((~conflict).sum()),
              f"{n_groups:,} code groups with both labels; all their rows removed, saved for review")
    u = u[~conflict]

    # 6. exact duplicates
    before = len(u)
    u = u.drop_duplicates("code_hash", keep="first")
    audit.add(source, "6. remove exact duplicates", before, len(u),
              "same code ignoring whitespace; first copy kept")
    return u, conflicts


def main():
    ap = argparse.ArgumentParser(description="Cleaning pipeline for CodeSentinel (Person A).")
    ap.add_argument("--bigvul", required=True, help="BigVul .zip or .csv")
    ap.add_argument("--bigvul-member", help="CSV name inside the zip, if there are several")
    ap.add_argument("--cvefixes", required=True, help="cvefixes_cpp_functions.csv")
    ap.add_argument("--cvefixes-map", help="cvefixes_commit_cve_cwe.csv (optional)")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=0, help="smoke test: read only this many rows of each file")
    ap.add_argument("--out-dir", default="data/interim")
    ap.add_argument("--report", default="reports/cleaning_audit.md")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    audit = Audit()

    print("Reading BigVul ...", flush=True)
    bv_raw = read_columns(a.bigvul, BV_COLS, a.bigvul_member, a.chunksize, a.max_rows, "BigVul")
    print("Reading CVEfixes ...", flush=True)
    with open(a.cvefixes, "rb") as f:
        cf_raw = pd.read_csv(f, usecols=CF_COLS, dtype=str, low_memory=False,
                             nrows=a.max_rows or None)
    mapping = pd.read_csv(a.cvefixes_map, dtype=str) if a.cvefixes_map else None

    bv_u = bigvul_to_unified(bv_raw)
    del bv_raw
    cf_u = cvefixes_to_unified(cf_raw, mapping)
    del cf_raw
    print("Tidying commit ids ...", flush=True)
    normalise_commit_ids([("BigVul", bv_u), ("CVEfixes", cf_u)], audit)

    print("Cleaning BigVul ...", flush=True)
    bv, bv_conf = clean_source(bv_u, "BigVul", audit, a.min_chars)
    del bv_u
    print("Cleaning CVEfixes ...", flush=True)
    cf, cf_conf = clean_source(cf_u, "CVEfixes", audit, a.min_chars)
    del cf_u

    # 7. overlap between the two datasets
    shared = cf["code_hash"].isin(set(bv["code_hash"]))
    audit.add("CVEfixes", "7. remove functions that also appear in BigVul", len(cf), int((~shared).sum()),
              "CVEfixes is the second-source set, so the overlap is removed from it")
    cf = cf[~shared]
    shared_commits = len(set(bv["commit_id"].dropna()) & set(cf["commit_id"].dropna()))
    audit.info("both", "info: commits present in both datasets", shared_commits,
               "NOT removed; decide how to handle when freezing the splits")

    # dates: normalise, count what is missing
    final = pd.concat([bv, cf], ignore_index=True)
    # Parse each source on its own: the two sources use different date formats, and
    # parsing them together makes pandas guess one format and silently drop the rest.
    parsed = pd.Series([None] * len(final), index=final.index, dtype="object")
    for src in final["source"].unique():
        m = final["source"] == src
        parsed[m] = (pd.to_datetime(final.loc[m, "date"], errors="coerce", utc=True, format="mixed")
                     .dt.strftime("%Y-%m-%d"))
    final["date"] = parsed
    for src, g in final.groupby("source"):
        audit.info("BigVul" if src == "bigvul" else "CVEfixes", "info: rows with no usable date",
                   int(g["date"].isna().sum()), "matters for a chronological split")
    final["sample_id"] = final["source"] + "-" + final["orig_id"].astype(str)
    final = final[UNIFIED]

    # final counts per class
    for src, name in (("bigvul", "BigVul"), ("cvefixes", "CVEfixes")):
        g = final[final["source"] == src]
        audit.info(name, "FINAL rows kept", len(g),
                   f"label 1 (vulnerable): {int((g['label'] == 1).sum()):,}; "
                   f"label 0: {int((g['label'] == 0).sum()):,}")
    audit.info("both", "FINAL rows kept (both datasets)", len(final))

    # write outputs
    final.to_csv(os.path.join(a.out_dir, "clean_unified.csv.gz"), index=False, compression="gzip")
    conf = pd.concat([bv_conf.assign(dataset="BigVul"), cf_conf.assign(dataset="CVEfixes")],
                     ignore_index=True)
    conf.to_csv(os.path.join(a.out_dir, "rejected_conflicts.csv.gz"), index=False, compression="gzip")
    table = audit.frame()
    table.to_csv(os.path.join(a.out_dir, "audit_table.csv"), index=False)
    pc = (final.groupby(["source", "project"])
               .agg(rows=("label", "size"), vulnerable=("label", "sum")).reset_index()
               .sort_values(["source", "rows"], ascending=[True, False]))
    pc.to_csv(os.path.join(a.out_dir, "project_counts.csv"), index=False)

    md = ["# Cleaning audit", "", "Generated by Scripts/02_clean_data.py. Raw data was not modified.", "",
          "| Dataset | Step | Before | Removed | After | Note |", "|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        fmt = lambda v: f"{v:,}" if isinstance(v, (int,)) else str(v)
        md.append(f"| {r.dataset} | {r.step} | {fmt(r.rows_before)} | {fmt(r.removed)} | "
                  f"{fmt(r.rows_after)} | {r.note} |")
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n" + "\n".join(md))
    print(f"\nSaved: {a.out_dir}/clean_unified.csv.gz, rejected_conflicts.csv.gz, audit_table.csv, project_counts.csv and {a.report}")


if __name__ == "__main__":
    main()