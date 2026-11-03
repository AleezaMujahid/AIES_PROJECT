"""
show_samples.py -- print a few REAL rows from the raw data, exactly as stored.

Read-only. Long text is cut after 300 characters so it stays readable
(the cut text is not removed from your data, only from this printout).
Also saves everything to reports/raw_samples.txt.

Usage (one line):
    python Scripts/show_samples.py --bigvul data/raw/bigvul/MSR_data_cleaned_download.zip --cvefixes data/raw/cvefixes/cvefixes_cpp_functions.csv
"""
import argparse
import os
import sys
import zipfile
from contextlib import contextmanager

import pandas as pd

MAX_CHARS = 300


@contextmanager
def open_binary(path):
    """Open a plain CSV, or the one CSV inside a .zip, without extracting."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith(".csv")
                     and not os.path.basename(n).startswith("._")
                     and "__MACOSX" not in n]
            if len(names) != 1:
                sys.exit(f"ERROR: expected exactly one CSV in the zip, found {names}")
            with z.open(names[0]) as f:
                yield f
    else:
        with open(path, "rb") as f:
            yield f


def short(value):
    if pd.isna(value):
        return "(empty)"
    s = str(value).replace("\r", "")
    if len(s) > MAX_CHARS:
        s = s[:MAX_CHARS] + f"  ...[cut here, {len(str(value)):,} characters in total]"
    return s


def render_row(row, title):
    lines = ["", "=" * 70, title, "=" * 70]
    for col, val in row.items():
        text = short(val)
        if "\n" in text:
            text = "\n    " + text.replace("\n", "\n    ")
        lines.append(f"{col}: {text}")
    return lines


def sample(path, label_col, label_values, name, nrows):
    with open_binary(path) as f:
        df = pd.read_csv(f, nrows=nrows, low_memory=False)
    out = [f"\n##### {name}: first {len(df):,} rows read, {len(df.columns)} columns #####"]
    out.append("Columns: " + ", ".join(df.columns))
    for lab, title in label_values:
        match = df[df[label_col].astype(str) == lab]
        if len(match):
            out += render_row(match.iloc[0], f"{name}: first row where {label_col} = {lab}  ({title})")
        else:
            out.append(f"\n(no row with {label_col} = {lab} among the first {len(df):,} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigvul", required=True)
    ap.add_argument("--cvefixes", required=True)
    ap.add_argument("--rows", type=int, default=2000, help="how many rows to scan for examples")
    ap.add_argument("--out", default="reports/raw_samples.txt")
    a = ap.parse_args()

    lines = []
    lines += sample(a.bigvul, "vul", [("1", "labeled vulnerable"), ("0", "labeled not vulnerable")],
                    "BigVul", a.rows)
    lines += sample(a.cvefixes, "before_change", [("True", "before the fix"), ("False", "after the fix")],
                    "CVEfixes", a.rows)

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()