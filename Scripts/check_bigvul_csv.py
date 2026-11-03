"""
check_bigvul_csv.py -- is the BigVul CSV complete and well-formed?

Works on a plain .csv OR directly on the .zip (no extraction needed).

Usage:
    python Scripts/check_bigvul_csv.py data/raw/bigvul/MSR_data_cleaned_download.zip
    python Scripts/check_bigvul_csv.py data/raw/bigvul/MSR_data_cleaned.csv
    (if a zip holds several CSVs:  --member NAME_INSIDE_ZIP)
"""
import argparse
import csv
import io
import os
import sys
import zipfile

csv.field_size_limit(20_000_000)  # stops a runaway quoted field from eating all memory


def open_text(path, member):
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        if member is None:
            names = [n for n in z.namelist()
                     if n.lower().endswith(".csv")
                     and not os.path.basename(n).startswith("._")
                     and "__MACOSX" not in n]
            if len(names) != 1:
                sys.exit(f"ERROR: expected exactly one CSV in the zip, found {names}. Use --member.")
            member = names[0]
        print("reading inside zip:", member)
        return io.TextIOWrapper(z.open(member), encoding="utf-8", errors="replace", newline="")
    return open(path, newline="", encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--member", help="file name inside the zip, if there is more than one CSV")
    a = ap.parse_args()

    print("file:", a.path)
    print("size on disk (GB):", round(os.path.getsize(a.path) / 1024**3, 2))

    rows, bad, last = 0, 0, None
    with open_text(a.path, a.member) as f:
        r = csv.reader(f)
        header = next(r)
        n = len(header)
        print("header has", n, "columns")
        try:
            for row in r:
                rows += 1
                last = row
                if len(row) != n:
                    bad += 1
                    if bad <= 5:
                        print("row", rows, "has", len(row), "fields, expected", n)
                if rows % 20000 == 0:
                    print("...", rows, "rows so far", flush=True)
        except csv.Error as e:
            print("csv.Error after row", rows, ":", e)

    print("rows read:", rows, " rows with wrong field count:", bad)
    if last:
        print("last row has", len(last), "fields")
        print("end of its last field:", repr(last[-1][-150:]))
    print("VERDICT:", "looks complete" if bad == 0 and rows > 150000 else "NOT OK, see above")


if __name__ == "__main__":
    main()