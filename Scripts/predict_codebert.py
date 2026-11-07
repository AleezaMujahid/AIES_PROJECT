"""
predict_codebert.py -- score ONE C/C++ function with the fine-tuned CodeBERT (Person B -> Person C gate).

The model folder is the "best" folder from Drive (weights, tokenizer, codebert_config.json). It is NOT in GitHub.

Usage:
    python Scripts/predict_codebert.py --model-dir path/to/best --file my_function.c
    python Scripts/predict_codebert.py --model-dir path/to/best --code "void f(char *s){ char b[8]; strcpy(b, s); }"
    (with neither --file nor --code it reads the function from standard input)

The output is a MODEL SCORE, not a probability. "not flagged" does not mean the code is secure.
"""
import argparse
import json
import os
import re
import sys


def normalize_code(c):
    """Same preprocessing as in training: per-line strip, collapse spaces/tabs, drop blank lines."""
    return "\n".join(re.sub(r"[ \t]+", " ", l.strip()) for l in str(c).split("\n") if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--file")
    ap.add_argument("--code")
    a = ap.parse_args()
    cfg_path = os.path.join(a.model_dir, "codebert_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"ERROR: {cfg_path} not found. --model-dir must be the 'best' folder.")
    if a.code is not None:
        code = a.code
    elif a.file:
        with open(a.file, encoding="utf-8", errors="replace") as f:
            code = f.read()
    else:
        code = sys.stdin.read()
    if not code.strip():
        sys.exit("ERROR: no code given.")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = json.load(open(cfg_path))
    max_len, thr = int(cfg["max_length"]), float(cfg["threshold"])
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    code_n = normalize_code(code)
    n_tok = len(tok(code_n, add_special_tokens=False, truncation=False)["input_ids"])
    enc = tok(code_n, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        score = float(torch.softmax(model(**enc).logits.float(), -1)[0, 1])
    verdict = "potentially vulnerable" if score >= thr else "not flagged"
    print(f"model score {score:.3f}  (threshold {thr:.2f})  ->  {verdict}")
    if n_tok > max_len - 2:
        print(f"WARNING: the input has {n_tok} tokens; only the first {max_len - 2} were analysed.")


if __name__ == "__main__":
    main()