"""
Expert rule layer.  Three small, independently testable rules.

    R-01  risky C library call        (static indicator)
    R-02  input check                 (input quality)
    R-03  model score / disagreement  (review policy)

Rule evidence is kept separate from the neural model's score.  A pattern match is NOT proof of a
vulnerability, and "no rule fired" does NOT mean the code is secure.
This module does not import Flask, torch or sklearn.
"""
import re
from dataclasses import asdict, dataclass

# --------------------------------------------------------------------------------------
# Rule definitions (the "knowledge base"; each rule has the fields the PRD asks for)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    condition: str
    review_weight: int            # 0 = info, 1 = low, 2 = medium, 3 = high
    recommended_review_action: str
    limitation: str


RULES = {
    "R-01": Rule(
        "R-01", "Risky C library call",
        "A call to one of the functions in RISKY_CALLS is found in the code, outside comments and string literals.",
        3, "Review how the buffer sizes / format strings / inputs reach this call.",
        "Text matching after removing comments and strings, not a full parser. A match does not establish "
        "exploitability; the call may be used safely."),
    "R-02": Rule(
        "R-02", "Input check",
        "Input is empty, too short, too long, has control characters, does not look like a C/C++ function, "
        "or is longer than the model limit and gets cut.",
        1, "Fix the input (paste one complete C/C++ function) or treat the result as partial.",
        "Only checks the form of the text, not whether the code is correct or complete."),
    "R-03": Rule(
        "R-03", "Model score / disagreement",
        "Model score >= frozen threshold, or the model and rule R-01 disagree.",
        2, "Prioritise this function for human security review.",
        "The score is not a calibrated probability. The detector is weak (test F1 about 0.22) and did not "
        "transfer to CVEfixes. Not flagged does not mean secure."),
}

# Functions that are well-known sources of memory-safety / format-string / injection bugs.
# value = short reason shown to the user.
RISKY_CALLS = {
    "gets": "reads input with no length limit (removed from C11)",
    "strcpy": "copies without checking the destination size",
    "strcat": "appends without checking the destination size",
    "sprintf": "formats into a buffer without a size limit",
    "vsprintf": "formats into a buffer without a size limit",
    "scanf": "can write past a buffer when %s has no width",
    "sscanf": "can write past a buffer when %s has no width",
    "memcpy": "copies a caller-supplied number of bytes; check the length",
    "system": "runs a shell command; check for command injection",
    "popen": "runs a shell command; check for command injection",
    "mktemp": "creates a predictable temporary file name",
}

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_STRIP = re.compile(
    r"""
      /\*.*?\*/                      # block comment
    | //[^\n]*                       # line comment
    | "(?:\\.|[^"\\\n])*"            # string literal
    | '(?:\\.|[^'\\\n])*'            # char literal
    """,
    re.VERBOSE | re.DOTALL,
)


def strip_comments_and_strings(code):
    """Blank out comments and string/char literals, keep newlines so line numbers stay right."""
    return _STRIP.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), code)


def _line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def find_risky_calls(code):
    """Return [{'function', 'line', 'reason'}] for real calls only.

    - ignores comments and string/char literals
    - ignores similar identifiers (my_strcpy(), strcpy_s(), mystrcpy()) because of the word boundary
    - ignores member calls such as obj.system( or p->gets(
    """
    clean = strip_comments_and_strings(code)
    hits = []
    for name, reason in RISKY_CALLS.items():
        for m in re.finditer(r"(?<![A-Za-z0-9_])" + name + r"\s*\(", clean):
            before = clean[: m.start()].rstrip()
            if before.endswith(".") or before.endswith("->"):
                continue
            hits.append({"function": name, "line": _line_of(clean, m.start()), "reason": reason})
    hits.sort(key=lambda h: (h["line"], h["function"]))
    return hits


# --------------------------------------------------------------------------------------
# Rule evaluation
# --------------------------------------------------------------------------------------


def _finding(rule_id, message, **extra):
    r = RULES[rule_id]
    d = asdict(r)
    d["message"] = message
    d.update(extra)
    return d


def check_input(code, min_chars, max_chars):
    """R-02 (blocking part). Returns None if the input is acceptable, else a finding dict."""
    if code is None or not str(code).strip():
        return _finding("R-02", "The input is empty. Paste one C/C++ function.", blocking=True)
    code = str(code)
    if "\x00" in code:
        return _finding("R-02", "The input contains a null byte, so it is not source text.", blocking=True)
    if len(code) > max_chars:
        return _finding("R-02", f"The input has {len(code):,} characters; the limit is {max_chars:,}.", blocking=True)
    if len(code.strip()) < min_chars:
        return _finding("R-02", f"The input is shorter than {min_chars} characters; paste a complete function.",
                        blocking=True)
    bare = strip_comments_and_strings(code)
    if "(" not in bare or "{" not in bare or "}" not in bare:
        return _finding("R-02", "This does not look like a C/C++ function (no parentheses and braces found). "
                                "CodeSentinel only analyses one C/C++ function.", blocking=True)
    return None


def truncation_finding(n_tokens, max_tokens):
    """R-02 (warning part): the model only sees the first max_tokens tokens."""
    if n_tokens is not None and n_tokens > max_tokens:
        return _finding("R-02", f"The function has {n_tokens:,} tokens; the model only analysed the first "
                                f"{max_tokens:,}. The rest was not scored.", blocking=False, truncated=True)
    return None


def evaluate_model_rules(score, threshold, risky_hits):
    """R-03 and the review-priority policy. Returns (findings, review_priority, recommendation).

    Policy (documented, also shown in the app):
        model flagged  AND R-01 hit  -> HIGH    (model and rule agree)
        model flagged  AND no R-01   -> MEDIUM  (model only; no rule evidence)
        not flagged    AND R-01 hit  -> MEDIUM  (rule only; model disagrees)
        not flagged    AND no R-01   -> LOW     (nothing found - NOT a security guarantee)
    """
    flagged = score >= threshold
    has_rule = bool(risky_hits)
    findings = []
    if flagged and has_rule:
        findings.append(_finding("R-03", "Model score is at or above the threshold and R-01 also matched: "
                                         "model and rule agree.", kind="agree"))
        return findings, "HIGH", "Prioritise for human security review: both the model and a rule flagged it."
    if flagged:
        findings.append(_finding("R-03", "Model score is at or above the threshold but no R-01 call was found: "
                                         "the model flag has no rule evidence.", kind="model_only"))
        return findings, "MEDIUM", "Review by a human. The flag comes from the model only; no rule evidence."
    if has_rule:
        findings.append(_finding("R-03", "R-01 matched a risky call but the model score is below the threshold: "
                                         "model and rule disagree.", kind="rule_only"))
        return findings, "MEDIUM", "Review the flagged call by hand; the model did not flag this function."
    return findings, "LOW", "No model flag and no rule match. This is not a guarantee that the code is secure."


# Public alias used by inference.py
make_finding = _finding
