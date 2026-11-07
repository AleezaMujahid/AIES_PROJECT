"""
10_build_app_data.py -- Person C. Turn Person A/B's saved report files into the small JSON
files the Flask app shows on its Model comparison and Data audit pages.

Nothing is typed in by hand: every number is read from reports/*.md.
Run from the repo root:   python Scripts/10_build_app_data.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(ROOT, "app", "data")


def read(name):
    with open(os.path.join(REP, name), encoding="utf-8") as f:
        return f.read()


def tables(md):
    """Return a list of (section_heading, [row dicts]) for every markdown table in md."""
    found, heading, lines = [], "", md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("# ").strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-|]+\|$", lines[i + 1]):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                cells += [""] * (len(head) - len(cells))
                rows.append(dict(zip(head, cells)))
                i += 1
            found.append((heading, rows))
            continue
        i += 1
    return found


def main():
    os.makedirs(OUT, exist_ok=True)

    ba = tables(read("before_after.md"))
    by_section = {}
    for h, rows in ba:
        by_section.setdefault(h, []).append(rows)

    def sec(prefix, k=0):
        for h, v in by_section.items():
            if h.startswith(prefix):
                return v[k]
        return []

    results = {
        "validation": sec("2."),
        "test": sec("3.", 0),
        "test_confusion": sec("3.", 1),
        "slices": sec("4."),
        "cvefixes": sec("5.", 0),
        "cvefixes_confusion": sec("5.", 1),
        "pairs": sec("6."),
    }
    md = read("before_after.md")
    bootstrap = [ln.strip() for ln in md.split("\n") if ln.startswith("CodeBERT minus TF-IDF")]
    results["bootstrap_lines"] = bootstrap
    lim = md.split("## 8.")[-1].split("\n")[1:]
    results["limitations"] = [ln[2:].strip() for ln in lim if ln.startswith("- ")]
    setup = md.split("## 1. Setup")[1].split("##")[0].split("\n")
    results["setup"] = [ln[2:].strip() for ln in setup if ln.startswith("- ")]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)

    audit = {"cleaning": [], "splits": [], "split_checks": [], "split_projects": []}
    for h, rows in tables(read("cleaning_audit.md")):
        audit["cleaning"] = rows
    sp = read("split_report.md")
    st = tables(sp)
    audit["splits"] = st[0][1]
    audit["split_checks"] = [ln[2:].strip() for ln in sp.split("## Checks")[1].split("##")[0].split("\n")
                             if ln.startswith("- ")]
    audit["split_projects"] = [ln[2:].strip() for ln in sp.split("## Biggest projects per split")[1]
                               .split("##")[0].split("\n") if ln.startswith("- ")]
    audit["token_length"] = tables(read("token_length.md"))[0][1]
    with open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=1)
    print("wrote app/data/results.json and app/data/audit.json")


if __name__ == "__main__":
    main()
