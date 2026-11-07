import json
import os

import pytest

from app import inference
from app.inference import Analyzer, normalize_code, result_to_csv, result_to_json
from tests.conftest import GOOD, FakeBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_normalize_code_matches_training_rule():
    assert normalize_code("  int a;\t\t b;  \n\n   \n  return 1;  ") == "int a; b;\nreturn 1;"


def test_flagged_when_score_above_threshold(analyzer, fake):
    fake.value = 0.9
    r = analyzer.analyze_function(GOOD)
    assert r["ok"] and r["prediction"] == "Potentially vulnerable" and r["review_priority"] == "HIGH"


def test_not_flagged_when_score_below_threshold(analyzer, fake):
    fake.value = 0.2
    r = analyzer.analyze_function("int add(int a, int b) { return a + b; }")
    assert r["prediction"] == "Not flagged" and r["review_priority"] == "LOW" and not r["rule_findings"]


def test_score_exactly_at_threshold_is_flagged(analyzer, fake):
    fake.value = 0.66
    assert analyzer.analyze_function(GOOD)["flagged"] is True


def test_result_has_all_fields(analyzer):
    r = analyzer.analyze_function(GOOD)
    for k in ("prediction", "model_score", "threshold", "review_priority", "review_recommendation",
              "rule_findings", "model", "input", "warnings", "inference_ms", "disclaimer"):
        assert k in r
    assert "not a security guarantee" in r["disclaimer"]
    assert "probability" in r["disclaimer"]


def test_long_input_gets_truncation_warning(analyzer, fake):
    fake.tokens = 900
    r = analyzer.analyze_function(GOOD)
    assert r["input"]["truncated"] and any("only analysed the first 510" in w for w in r["warnings"])


def test_empty_input_is_a_clear_error_and_model_not_called(analyzer, fake):
    r = analyzer.analyze_function("")
    assert not r["ok"] and "empty" in r["error"] and fake.calls == 0


def test_oversized_input_is_rejected_before_scoring(fake):
    az = Analyzer(fake, max_chars=100)
    r = az.analyze_function("void f(){" + "x" * 500 + "}")
    assert not r["ok"] and fake.calls == 0


def test_no_model_loaded_gives_safe_error():
    az = Analyzer(None, load_error="nothing found")
    r = az.analyze_function(GOOD)
    assert not r["ok"] and "No model" in r["error"]


def test_model_crash_does_not_leak_details(analyzer, fake):
    def boom(code):
        raise RuntimeError("secret path /home/x")
    fake.score = boom
    r = analyzer.analyze_function(GOOD)
    assert not r["ok"] and "secret" not in json.dumps(r)


def test_submitted_code_is_never_executed(analyzer, tmp_path):
    marker = tmp_path / "pwned"
    code = '#include <stdlib.h>\nvoid f(void) { system("touch %s"); }' % marker
    analyzer.analyze_function(code)
    assert not marker.exists()


def test_model_is_called_once_per_request(analyzer, fake):
    for _ in range(5):
        analyzer.analyze_function(GOOD)
    assert fake.calls == 5


def test_json_export_does_not_contain_the_code(analyzer):
    r = analyzer.analyze_function(GOOD)
    out = result_to_json(r)
    assert json.loads(out)["ok"] and "strcpy(b, s)" not in out


def test_csv_export_neutralises_formulas(analyzer):
    r = analyzer.analyze_function(GOOD)
    r["review_recommendation"] = "=HYPERLINK(\"http://evil\")"
    for line in result_to_csv(r).splitlines():
        assert not line.split(",", 1)[-1].startswith(("=", "+", "@"))
    assert "'=HYPERLINK" in result_to_csv(r)


def test_csv_export_has_rules_and_disclaimer(analyzer):
    out = result_to_csv(analyzer.analyze_function(GOOD))
    assert "R-01" in out and "disclaimer" in out


# ---------- loading behaviour ----------
def cfg(**kw):
    base = {"BACKEND": "auto", "CODEBERT_DIR": "/does/not/exist", "BASELINE_DIR": "/does/not/exist",
            "MIN_INPUT_CHARS": 20, "MAX_INPUT_CHARS": 20000}
    base.update(kw)
    return base


def test_nothing_available_means_no_backend_not_a_crash():
    az = Analyzer.from_config(cfg())
    assert az.backend is None and az.info()["loaded"] is False and az.load_error


def test_forced_codebert_with_missing_folder_does_not_silently_fall_back():
    az = Analyzer.from_config(cfg(BACKEND="codebert", BASELINE_DIR=os.path.join(ROOT, "models", "baseline_generic")))
    assert az.backend is None and "CODESENTINEL_MODEL_DIR" in az.load_error


BASELINE = os.path.join(ROOT, "models", "baseline_generic")


@pytest.mark.skipif(not os.path.exists(os.path.join(BASELINE, "baseline.joblib")), reason="baseline not in repo")
def test_auto_mode_falls_back_to_baseline_with_a_visible_note():
    az = Analyzer.from_config(cfg(BASELINE_DIR=BASELINE))
    info = az.info()
    assert info["is_fallback"] and "fallback" in info["note"].lower()
    r = az.analyze_function(GOOD)
    assert r["ok"] and 0.0 <= r["model_score"] <= 1.0 and any("Fallback" in w for w in r["warnings"])
