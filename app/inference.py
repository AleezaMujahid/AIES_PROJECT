"""
Inference module.  Works WITHOUT Flask:

    from app.inference import Analyzer
    az = Analyzer.from_config()          # loads the model once
    print(az.analyze_function("void f(char *s){ char b[8]; strcpy(b, s); }"))

Backends
    CodeBERTBackend  - fine-tuned CodeBERT (the "best" folder from Drive).
    BaselineBackend  - saved TF-IDF + Logistic Regression (offline / fallback demo mode).

Rules that come from the model handoff notes (HANDOFF_B_to_C.md):
    * normalize_code() is applied BEFORE CodeBERT (same as training). No other cleaning.
    * max_length 512 -> 510 code tokens. Longer input: only the start is analysed, and we warn.
    * Frozen threshold comes from codebert_config.json (0.66). score >= threshold -> flagged.
    * The output is a MODEL SCORE, never a probability. Not flagged does not mean secure.
    * The submitted code is only tokenised and scored. It is never executed or compiled.
"""
import csv
import io
import json
import os
import re
import threading
import time

from . import rules

DISCLAIMER = ("This is a model score, not a probability, and it is not a security guarantee. "
              "Not flagged does not mean secure. A human must review the code.")


def normalize_code(c):
    """Same preprocessing as in training: per-line strip, collapse spaces/tabs, drop blank lines."""
    return "\n".join(re.sub(r"[ \t]+", " ", l.strip()) for l in str(c).split("\n") if l.strip())


# ------------------------------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------------------------------
class BaselineBackend:
    kind = "baseline"

    def __init__(self, model_dir):
        import joblib  # imported here so rules/tests do not need sklearn

        path = os.path.join(model_dir, "baseline.joblib")
        cfg_path = os.path.join(model_dir, "config.json")
        if not (os.path.exists(path) and os.path.exists(cfg_path)):
            raise FileNotFoundError("baseline files not found in " + model_dir)
        self.bundle = joblib.load(path)
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.threshold = float(self.cfg["threshold"])
        self.max_tokens = None  # TF-IDF has no token limit; it cuts at max_chars
        self.name = "TF-IDF + Logistic Regression (baseline)"
        self.version = self.cfg.get("model", "tfidf_logreg") + " (saved baseline)"
        self._tok = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|[^\sA-Za-z0-9_]")

    def score(self, code):
        code = code[: int(self.cfg["max_chars"])]
        if self.bundle.get("mode") == "generic":
            keep = self.bundle["keep"]
            text = " ".join(t if (not re.match(r"[A-Za-z_]", t) or t in keep) else "ID"
                            for t in self._tok.findall(code))
        else:
            text = code
        x = self.bundle["vectorizer"].transform([text])
        return float(self.bundle["model"].predict_proba(x)[0, 1]), None


class CodeBERTBackend:
    kind = "codebert"

    def __init__(self, model_dir):
        cfg_path = os.path.join(model_dir, "codebert_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("codebert_config.json not found in " + model_dir)
        import torch  # heavy imports only when CodeBERT is really used
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.max_length = int(self.cfg["max_length"])
        self.max_tokens = self.max_length - 2
        self.threshold = float(self.cfg["threshold"])
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.name = "Fine-tuned CodeBERT"
        self.version = "codebert-base fine-tuned, max_length %d, threshold %.2f" % (self.max_length, self.threshold)

    def score(self, code):
        torch = self._torch
        code_n = normalize_code(code)
        n_tok = len(self.tok(code_n, add_special_tokens=False, truncation=False)["input_ids"])
        enc = self.tok(code_n, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            s = float(torch.softmax(self.model(**enc).logits.float(), -1)[0, 1])
        return s, n_tok


# ------------------------------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------------------------------
class Analyzer:
    def __init__(self, backend, min_chars=20, max_chars=20000, load_note="", load_error=""):
        self.backend = backend
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.load_note = load_note      # e.g. "CodeBERT folder not found, using the baseline"
        self.load_error = load_error    # set when nothing could be loaded
        self._lock = threading.Lock()

    # ---- construction (model is loaded ONCE here) ----
    @classmethod
    def from_config(cls, config=None):
        """config can be a dict, a Flask config, or None (then app.config.Config is used)."""
        if config is None:
            from .config import Config
            config = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
        mode = str(config.get("BACKEND", "auto")).lower()
        cb_dir, bl_dir = config["CODEBERT_DIR"], config["BASELINE_DIR"]
        common = dict(min_chars=config.get("MIN_INPUT_CHARS", 20), max_chars=config.get("MAX_INPUT_CHARS", 20000))
        note, errors = "", []

        if mode in ("auto", "codebert"):
            if os.path.exists(os.path.join(cb_dir, "codebert_config.json")):
                try:
                    return cls(CodeBERTBackend(cb_dir), **common)
                except Exception as e:  # missing torch, corrupt weights, ...
                    errors.append("CodeBERT could not be loaded (%s)" % type(e).__name__)
            else:
                errors.append("CodeBERT folder not found")
            if mode == "codebert":
                return cls(None, load_error="; ".join(errors) + ". Set CODESENTINEL_MODEL_DIR to the 'best' folder.",
                           **common)
            note = "; ".join(errors) + ". Running in fallback mode with the saved TF-IDF baseline."
        try:
            return cls(BaselineBackend(bl_dir), load_note=note, **common)
        except Exception as e:
            errors.append("baseline could not be loaded (%s)" % type(e).__name__)
            return cls(None, load_error="; ".join(errors), **common)

    # ---- status for the pages ----
    def info(self):
        b = self.backend
        return {
            "loaded": b is not None,
            "model_name": b.name if b else "none",
            "model_version": b.version if b else "none",
            "threshold": round(b.threshold, 3) if b else None,
            "max_tokens": getattr(b, "max_tokens", None) if b else None,
            "is_fallback": bool(b and b.kind == "baseline"),
            "note": self.load_note,
            "error": self.load_error,
        }

    # ---- the main function ----
    def analyze_function(self, code):
        """Validate -> score -> rules -> structured result. Never raises for bad user input."""
        base = {"ok": False, "disclaimer": DISCLAIMER, "model": self.info()}

        # 1. validate input (R-02, blocking part)
        bad = rules.check_input(code, self.min_chars, self.max_chars)
        if bad is not None:
            base.update(error=bad["message"], rule_findings=[bad])
            return base
        if self.backend is None:
            base.update(error="No model is loaded, so nothing can be scored. " + (self.load_error or ""),
                        rule_findings=[])
            return base

        code = str(code)
        # 2. rule evidence (R-01) - kept separate from the model score
        hits = rules.find_risky_calls(code)
        findings = [rules.make_finding("R-01", "%s() on line %d: %s." % (h["function"], h["line"], h["reason"]),
                                       function=h["function"], line=h["line"]) for h in hits]

        # 3. model score (frozen threshold from the saved config)
        t0 = time.perf_counter()
        try:
            with self._lock:
                score, n_tokens = self.backend.score(code)
        except Exception:
            base.update(error="The model failed on this input. Nothing was scored.", rule_findings=findings)
            return base
        ms = (time.perf_counter() - t0) * 1000

        thr = self.backend.threshold
        flagged = score >= thr
        warnings = []
        trunc = rules.truncation_finding(n_tokens, self.backend.max_tokens) if self.backend.max_tokens else None
        if trunc:
            findings.append(trunc)
            warnings.append(trunc["message"])
        if self.backend.kind == "baseline":
            warnings.append("Fallback mode: the saved TF-IDF baseline scored this, not CodeBERT.")

        # 4. review policy (R-03)
        r3, priority, recommendation = rules.evaluate_model_rules(score, thr, hits)
        findings.extend(r3)

        return {
            "ok": True,
            "prediction": "Potentially vulnerable" if flagged else "Not flagged",
            "flagged": flagged,
            "model_score": round(score, 4),
            "threshold": round(thr, 3),
            "review_priority": priority,
            "review_recommendation": recommendation,
            "rule_findings": findings,
            "risky_call_count": len(hits),
            "model": self.info(),
            "input": {"characters": len(code), "lines": code.count("\n") + 1,
                      "tokens": n_tokens, "truncated": bool(trunc)},
            "warnings": warnings,
            "inference_ms": round(ms, 1),
            "disclaimer": DISCLAIMER,
        }


# ------------------------------------------------------------------------------------------
# Export (the submitted code itself is NOT included in the export)
# ------------------------------------------------------------------------------------------
def result_to_json(result):
    return json.dumps(result, indent=2, ensure_ascii=False)


def result_to_csv(result):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["field", "value"])
    m = result.get("model", {})
    rows = [("ok", result.get("ok")), ("error", result.get("error", "")),
            ("prediction", result.get("prediction", "")), ("model_score", result.get("model_score", "")),
            ("threshold", result.get("threshold", "")), ("review_priority", result.get("review_priority", "")),
            ("review_recommendation", result.get("review_recommendation", "")),
            ("model_name", m.get("model_name", "")), ("model_version", m.get("model_version", "")),
            ("rules_triggered", ";".join(sorted({f["rule_id"] for f in result.get("rule_findings", [])}))),
            ("warnings", " | ".join(result.get("warnings", []))),
            ("input_characters", result.get("input", {}).get("characters", "")),
            ("input_tokens", result.get("input", {}).get("tokens", "")),
            ("inference_ms", result.get("inference_ms", "")), ("disclaimer", result.get("disclaimer", ""))]
    for k, v in rows:
        w.writerow([k, _csv_safe(v)])
    for f in result.get("rule_findings", []):
        w.writerow(["finding", _csv_safe("%s: %s" % (f["rule_id"], f["message"]))])
    return buf.getvalue()


def _csv_safe(v):
    """Stop spreadsheet formula injection: a cell starting with = + - @ would be run by Excel."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s
