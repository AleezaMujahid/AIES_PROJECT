"""End-to-end checklist for the Flask app (Day 13)."""
import json

from app import create_app
from app.inference import Analyzer
from tests.conftest import GOOD


def test_all_pages_load(client):
    for url in ("/", "/models", "/data", "/research"):
        assert client.get(url).status_code == 200


def test_unknown_page_is_a_clean_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and b"Traceback" not in r.data


def test_analyze_shows_score_rules_and_disclaimer(client):
    r = client.post("/", data={"code": GOOD})
    html = r.data.decode()
    assert r.status_code == 200
    assert "Model score" in html and "R-01" in html and "not a security guarantee" in html
    assert "probability" in html  # it is explained as NOT a probability


def test_empty_input_message(client):
    r = client.post("/", data={"code": ""})
    assert r.status_code == 400 and b"The input is empty" in r.data


def test_whitespace_only_input_message(client):
    assert client.post("/", data={"code": "  \n\n "}).status_code == 400


def test_long_input_over_char_limit_is_rejected_cleanly(client):
    r = client.post("/", data={"code": "void f(){" + "a" * 30000 + "}"})
    assert r.status_code == 400 and b"limit" in r.data


def test_request_over_size_limit_gets_413(client):
    r = client.post("/", data={"code": "x" * (200 * 1024)})
    assert r.status_code == 413 and b"too large" in r.data


def test_html_and_javascript_in_code_are_escaped(client):
    evil = 'void f(){ puts("<script>alert(1)</script>"); } /* <img src=x onerror=alert(2)> */'
    html = client.post("/", data={"code": evil}).data.decode()
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_html_in_example_query_is_not_reflected(client):
    html = client.get("/?example=<script>alert(1)</script>").data.decode()
    assert "<script>alert(1)</script>" not in html


def test_example_loads_into_the_box(client):
    html = client.get("/?example=demo-gets").data.decode()
    assert "gets(name)" in html and "Hand-written demo" in html


def test_missing_model_app_still_starts_and_explains(tmp_path):
    az = Analyzer(None, load_error="CodeBERT folder not found")
    c = create_app({"TESTING": True}, analyzer=az).test_client()
    assert c.get("/").status_code == 200 and b"No model loaded" in c.get("/").data
    r = c.post("/", data={"code": GOOD})
    assert r.status_code == 400 and b"No model is loaded" in r.data and b"Traceback" not in r.data


def test_repeated_requests_give_same_answer_and_load_model_once(client, fake):
    outs = [client.post("/", data={"code": GOOD}).data.decode().count("Potentially vulnerable") for _ in range(10)]
    assert len(set(outs)) == 1 and fake.calls == 10


def test_export_json_and_csv(client):
    j = client.post("/export/json", data={"code": GOOD})
    assert j.status_code == 200 and j.mimetype == "application/json"
    assert "attachment" in j.headers["Content-Disposition"]
    assert json.loads(j.data)["prediction"] in ("Potentially vulnerable", "Not flagged")
    c = client.post("/export/csv", data={"code": GOOD})
    assert c.status_code == 200 and c.mimetype == "text/csv" and b"model_score" in c.data


def test_export_does_not_contain_submitted_code(client):
    assert b"strcpy(b, s)" not in client.post("/export/json", data={"code": GOOD}).data


def test_export_of_bad_input_and_bad_format(client):
    assert client.post("/export/json", data={"code": ""}).status_code == 400
    assert client.post("/export/exe", data={"code": GOOD}).status_code == 404
    assert client.get("/export/json").status_code == 405


def test_api_good_and_bad_requests(client):
    ok = client.post("/api/analyze", json={"code": GOOD})
    assert ok.status_code == 200 and ok.get_json()["ok"]
    assert client.post("/api/analyze", json={"nope": 1}).status_code == 400
    assert client.post("/api/analyze", data="not json", content_type="application/json").status_code == 400
    assert client.post("/api/analyze", json={"code": ""}).status_code == 400
    assert client.post("/api/analyze", json={"code": "x" * 200000}).status_code == 413


def test_security_headers_present(client):
    h = client.get("/").headers
    assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"


def test_pages_use_real_numbers_from_the_reports(client):
    models = client.get("/models").data.decode()
    assert "0.223" in models and "Majority guess" in models  # CodeBERT test F1 from before_after.md
    data = client.get("/data").data.decode()
    assert "184,800" in data and "FINAL" in data
