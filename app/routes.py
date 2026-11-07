"""Flask routes. All the real work is in inference.py and rules.py."""
import json
import os

from flask import (Blueprint, Response, abort, current_app, jsonify, render_template, request)

from . import inference
from .rules import RISKY_CALLS, RULES

bp = Blueprint("main", __name__)


def _analyzer():
    return current_app.extensions["analyzer"]


def _load_json(name, default):
    path = os.path.join(current_app.config["DATA_DIR"], name)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _examples():
    return _load_json("examples.json", [])


def _group_examples(examples):
    """Group the examples for the chips under the editor."""
    groups = [("Marked vulnerable in the dataset", []), ("Marked safe in the dataset", []),
              ("Same function before and after a fix", []), ("Rule demos (hand-written)", [])]
    n1 = n0 = 0
    for e in examples:
        if e["id"].startswith("demo-"):
            idx, label = 3, ("gets() call" if e["id"] == "demo-gets" else "gets in a comment")
        elif e["id"].startswith("cvefixes-"):
            idx, label = 2, ("Before the fix" if e["true_label"] else "After the fix")
        elif e["true_label"] == 1:
            n1 += 1
            idx, label = 0, "Function %d" % n1
        else:
            n0 += 1
            idx, label = 1, "Function %d" % n0
        groups[idx][1].append({"id": e["id"], "label": label, "title": e["title"]})
    return [g for g in groups if g[1]]


def _bars(results):
    """Accuracy / recall / F1 per model for the comparison chart (read from results.json)."""
    names = []
    for row in results.get("test", []):
        m, st = row.get("model", ""), row.get("stage", "")
        if "Majority" in m:
            n = "Always guess safe"
        elif "TF-IDF" in m:
            n = "TF-IDF baseline"
        elif "before" in st:
            n = "CodeBERT before training"
        else:
            n = "CodeBERT fine-tuned"
        try:
            names.append({"name": n, "accuracy": float(row["accuracy"]), "recall": float(row["recall"]),
                          "f1": float(row["f1"]), "precision": float(row["precision"])})
        except (KeyError, ValueError):
            continue
    return names


@bp.route("/", methods=["GET", "POST"])
def analyze():
    az = _analyzer()
    examples = _examples()
    code, result, error, shown_example = "", None, None, None

    if request.method == "POST":
        code = request.form.get("code", "")
        result = az.analyze_function(code)
        if not result["ok"]:
            error = result["error"]
            result = None if error else result
        status = 400 if error else 200
    else:
        wanted = request.args.get("example")
        if wanted:
            shown_example = next((e for e in examples if e["id"] == wanted), None)
            if shown_example:
                code = shown_example["code"]
        status = 200

    return render_template("analyze.html", code=code, result=result, error=error, groups=_group_examples(examples),
                           shown_example=shown_example, info=az.info(), max_chars=az.max_chars), status


@bp.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("code"), str):
        return jsonify({"ok": False, "error": 'Send JSON like {"code": "<one C/C++ function>"}.'}), 400
    result = _analyzer().analyze_function(data["code"])
    return jsonify(result), (200 if result["ok"] else 400)


@bp.route("/export/<fmt>", methods=["POST"])
def export(fmt):
    if fmt not in ("json", "csv"):
        abort(404)
    result = _analyzer().analyze_function(request.form.get("code", ""))
    if not result["ok"]:
        return render_template("message.html", title="Nothing to export", message=result["error"]), 400
    if fmt == "json":
        body, mime = inference.result_to_json(result), "application/json"
    else:
        body, mime = inference.result_to_csv(result), "text/csv"
    return Response(body, mimetype=mime,
                    headers={"Content-Disposition": "attachment; filename=codesentinel_assessment." + fmt})


@bp.route("/models")
def models():
    r = _load_json("results.json", {})
    return render_template("models.html", r=r, bars=_bars(r), info=_analyzer().info())


@bp.route("/data")
def data_audit():
    a = _load_json("audit.json", {})
    finals = [row for row in a.get("cleaning", []) if row.get("Step", "").startswith("FINAL")]
    return render_template("data_audit.html", a=a, finals=finals, info=_analyzer().info())


@bp.route("/research")
def research():
    return render_template("research.html", rules=RULES, risky=RISKY_CALLS,
                           r=_load_json("results.json", {}), info=_analyzer().info())
