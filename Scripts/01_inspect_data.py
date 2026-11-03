"""
01_inspect_data.py  --  first look at the raw datasets.

Read-only: this script never changes your data. It writes one report file
(default: reports/data_inspection.md) that you can use in the report as
evidence of what was wrong with the data BEFORE cleaning.

The BigVul input can be the .zip itself (no extraction needed).

Usage (from the project folder, all on one line):
    python Scripts/01_inspect_data.py --bigvul data/raw/bigvul/MSR_data_cleaned_download.zip --cvefixes data/raw/cvefixes/cvefixes_cpp_functions.csv

If a column name differs in your file, pass it explicitly, e.g.
    --bigvul-code func_before --bigvul-label vul
If the zip holds more than one CSV:  --bigvul-member NAME_INSIDE_ZIP
"""
import argparse
import hashlib
import os
import sys
import zipfile
from collections import Counter
from contextlib import ExitStack, contextmanager

import pandas as pd


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


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
                             f"Pass the right one with --bigvul-member.")
                member = names[0]
            print(f"  (reading '{member}' directly from the zip)")
            with z.open(member) as f:
                yield f
    else:
        with open(path, "rb") as f:
            yield f


def pick(columns, wanted, candidates, what):
    """Use the column the user named, else the first candidate that exists."""
    if wanted:
        if wanted not in columns:
            sys.exit(f"ERROR: {what} column '{wanted}' not found. Columns are: {list(columns)}")
        return wanted
    for c in candidates:
        if c in columns:
            return c
    sys.exit(f"ERROR: could not guess the {what} column. Columns are: {list(columns)}\n"
             f"Pass it explicitly with the matching --*-{what} option.")


def inspect_file(path, name, code_col, label_col, chunksize, member=None):
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")

    with open_binary(path, member) as f:
        columns = list(pd.read_csv(f, nrows=0).columns)
    code_col = pick(columns, code_col, ["func_before", "code", "func"], "code")
    label_col = pick(columns, label_col, ["vul", "before_change", "label"], "label")

    rows = 0
    nulls = Counter()
    empty_code = 0
    label_counts = Counter()
    first_label_by_hash = {}
    conflicting_hashes = set()
    hash_counts = Counter()
    lengths = []
    dtypes = None

    with ExitStack() as stack:
        stream = stack.enter_context(open_binary(path, member))
        # only the columns we need for counting, plus all columns for null counts
        for chunk in pd.read_csv(stream, chunksize=chunksize, low_memory=False):
            if dtypes is None:
                dtypes = {c: str(t) for c, t in chunk.dtypes.items()}
            rows += len(chunk)
            for c, n in chunk.isna().sum().items():
                nulls[c] += int(n)

            code = chunk[code_col].fillna("").astype(str)
            stripped = code.str.strip()
            empty_code += int((stripped == "").sum())
            lengths.extend(stripped.str.len().tolist())

            labels = chunk[label_col].astype(str)
            label_counts.update(labels.tolist())

            for text, lab in zip(stripped.tolist(), labels.tolist()):
                if not text:
                    continue
                h = md5(text)
                hash_counts[h] += 1
                seen = first_label_by_hash.get(h)
                if seen is None:
                    first_label_by_hash[h] = lab
                elif seen != lab:
                    conflicting_hashes.add(h)

            if rows % (chunksize * 10) == 0:
                print(f"  ... {rows:,} rows so far", flush=True)

    unique = len(hash_counts)
    non_empty = rows - empty_code
    dup_rows = non_empty - unique
    s = pd.Series(lengths) if lengths else pd.Series([0])

    return {
        "name": name, "path": path, "rows": rows, "columns": columns, "dtypes": dtypes,
        "code_col": code_col, "label_col": label_col,
        "nulls": {c: n for c, n in nulls.items() if n > 0},
        "empty_code": empty_code, "unique_code": unique, "duplicate_rows": dup_rows,
        "conflicting_label_groups": len(conflicting_hashes),
        "label_counts": dict(label_counts),
        "len_min": int(s.min()), "len_median": float(s.median()),
        "len_p95": float(s.quantile(0.95)), "len_max": int(s.max()),
    }


def to_markdown(r):
    out = [f"## {r['name']}", "",
           f"- File: `{r['path']}`",
           f"- Rows: **{r['rows']:,}**",
           f"- Code column used: `{r['code_col']}`   Label column used: `{r['label_col']}`",
           f"- Empty or whitespace-only code: **{r['empty_code']:,}**",
           f"- Unique code strings (by hash): **{r['unique_code']:,}**",
           f"- Exact duplicate rows (same code text): **{r['duplicate_rows']:,}**",
           f"- Code strings that appear with conflicting labels: **{r['conflicting_label_groups']:,}**",
           f"- Label counts: {r['label_counts']}",
           f"- Code length in characters: min {r['len_min']}, median {r['len_median']:.0f}, "
           f"95th percentile {r['len_p95']:.0f}, max {r['len_max']}",
           "", "### Missing values (columns with at least one)", ""]
    if r["nulls"]:
        out += ["| Column | Missing |", "|---|---|"]
        out += [f"| {c} | {n:,} |" for c, n in sorted(r["nulls"].items(), key=lambda x: -x[1])]
    else:
        out.append("None found.")
    out += ["", "### All columns and types", "", "| Column | Type |", "|---|---|"]
    out += [f"| {c} | {t} |" for c, t in r["dtypes"].items()]
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Read-only inspection of the raw CodeSentinel datasets.")
    ap.add_argument("--bigvul", required=True, help="BigVul CSV, or the .zip that contains it")
    ap.add_argument("--cvefixes", required=True, help="path to cvefixes_cpp_functions.csv")
    ap.add_argument("--bigvul-member", help="file name inside the BigVul zip, if it holds several CSVs")
    ap.add_argument("--bigvul-code"); ap.add_argument("--bigvul-label")
    ap.add_argument("--cvefixes-code"); ap.add_argument("--cvefixes-label")
    ap.add_argument("--out", default="reports/data_inspection.md")
    ap.add_argument("--chunksize", type=int, default=2000,
                    help="rows per chunk (small, because some BigVul rows are huge)")
    a = ap.parse_args()

    results = []
    for name, path, cc, lc, mem in [
        ("BigVul", a.bigvul, a.bigvul_code, a.bigvul_label, a.bigvul_member),
        ("CVEfixes (C/C++ slice)", a.cvefixes, a.cvefixes_code, a.cvefixes_label, None),
    ]:
        print(f"Inspecting {name} ...", flush=True)
        results.append(inspect_file(path, name, cc, lc, a.chunksize, mem))

    report = "# Raw data inspection\n\nGenerated by Scripts/01_inspect_data.py. " \
             "Nothing was modified.\n\n" + "\n".join(to_markdown(r) for r in results)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()