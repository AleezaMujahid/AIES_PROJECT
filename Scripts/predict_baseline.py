"""
predict_baseline.py -- score ONE C/C++ function with the saved baseline (the Day 5 gate).

Usage:
    python Scripts/predict_baseline.py --file my_function.c
    python Scripts/predict_baseline.py --code "void f(char *s){ char b[8]; strcpy(b, s); }"
    (with neither, it reads the function from standard input)
"""
import argparse
import json
import os
import sys

import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/baseline")
    ap.add_argument("--file")
    ap.add_argument("--code")
    a = ap.parse_args()
    p = os.path.join(a.model_dir, "baseline.joblib")
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found. Run Scripts/06_train_baseline.py first.")
    if a.code is not None:
        code = a.code
    elif a.file:
        with open(a.file, encoding="utf-8", errors="replace") as f:
            code = f.read()
    else:
        code = sys.stdin.read()
    if not code.strip():
        sys.exit("ERROR: no code given.")
    bundle = joblib.load(p)
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    if bundle.get("mode") == "generic":
        import re
        tok = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[^\sA-Za-z0-9_]")
        keep = bundle["keep"]
        text = " ".join(t if (not re.match(r"[A-Za-z_]", t) or t in keep) else "ID"
                        for t in tok.findall(code[: cfg["max_chars"]]))
        X = bundle["vectorizer"].transform([text])
    else:
        X = bundle["vectorizer"].transform([code[: cfg["max_chars"]]])
    score = float(bundle["model"].predict_proba(X)[0, 1])
    verdict = "VULNERABLE" if score >= cfg["threshold"] else "not flagged"
    print(f"score {score:.3f}  (threshold {cfg['threshold']:.3f})  ->  {verdict}")


if __name__ == "__main__":
    main()