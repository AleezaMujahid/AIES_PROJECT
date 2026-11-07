"""Unit tests for app/rules.py (no model, no Flask needed)."""
from app import rules


def names(code):
    return [h["function"] for h in rules.find_risky_calls(code)]


# ---------------- R-01: risky calls ----------------
def test_real_gets_call_is_found():
    assert names("void f(){ char b[8]; gets(b); }") == ["gets"]


def test_call_with_space_before_parenthesis():
    assert names("void f(){ strcpy (d, s); }") == ["strcpy"]


def test_call_in_line_comment_is_ignored():
    assert names("void f(){\n // gets(b);\n return; }") == []


def test_call_in_block_comment_is_ignored():
    assert names("void f(){ /* strcpy(a,b); \n system(x); */ return; }") == []


def test_call_in_string_literal_is_ignored():
    assert names('void f(){ puts("never call gets(x) or strcpy(a,b)"); }') == []


def test_call_in_char_literal_and_escaped_quote():
    assert names('void f(){ char c = \'"\'; puts("a\\"gets(b)"); }') == []


def test_similar_identifiers_are_ignored():
    code = "void f(){ my_strcpy(a,b); strcpy_s(a,1,b); mystrcpy(a,b); fgets(b,8,stdin); xsystem(c); }"
    assert names(code) == []


def test_member_calls_are_ignored():
    assert names("void f(){ obj.system(1); p->gets(2); }") == []


def test_function_name_without_call_is_ignored():
    assert names("void f(){ void (*p)(char*) = (void*)gets; }") == []


def test_line_numbers_are_correct():
    hits = rules.find_risky_calls("void f(){\n  int a;\n  strcpy(a,b);\n  gets(c);\n}")
    assert [(h["function"], h["line"]) for h in hits] == [("strcpy", 3), ("gets", 4)]


def test_line_numbers_survive_comments_with_newlines():
    hits = rules.find_risky_calls("void f(){\n/* a\nb\nc */\n gets(x);\n}")
    assert hits[0]["line"] == 5


def test_multiple_different_calls():
    assert set(names("void f(){ strcpy(a,b); sprintf(c,d); system(e); }")) == {"strcpy", "sprintf", "system"}


# ---------------- R-02: input check ----------------
def check(code):
    return rules.check_input(code, 20, 20000)


def test_empty_none_and_blank_are_rejected():
    for bad in ("", "   \n\t ", None):
        assert check(bad)["blocking"] is True


def test_too_short_is_rejected():
    assert "shorter" in check("int f(){}")["message"]


def test_too_long_is_rejected():
    assert "limit" in rules.check_input("int f(){" + "a" * 50 + "}", 20, 40)["message"]


def test_null_byte_is_rejected():
    assert "null byte" in check("void f(){ int a; \x00 }")["message"]


def test_plain_text_is_rejected():
    assert check("this is just some normal english text, not code")["blocking"]


def test_valid_function_is_accepted():
    assert check("int add(int a, int b) { return a + b; }") is None


def test_braces_only_in_comments_do_not_count():
    assert check("// f() { }\njust some words here to be long enough") is not None


def test_truncation_finding():
    assert rules.truncation_finding(600, 510)["truncated"] is True
    assert rules.truncation_finding(510, 510) is None
    assert rules.truncation_finding(None, 510) is None


# ---------------- R-03: model score / disagreement policy ----------------
HIT = [{"function": "gets", "line": 1, "reason": "x"}]


def test_policy_high_when_model_and_rule_agree():
    f, p, _ = rules.evaluate_model_rules(0.9, 0.66, HIT)
    assert p == "HIGH" and f[0]["kind"] == "agree"


def test_policy_medium_model_only():
    f, p, _ = rules.evaluate_model_rules(0.9, 0.66, [])
    assert p == "MEDIUM" and f[0]["kind"] == "model_only"


def test_policy_medium_rule_only_is_a_disagreement():
    f, p, _ = rules.evaluate_model_rules(0.1, 0.66, HIT)
    assert p == "MEDIUM" and f[0]["kind"] == "rule_only"


def test_policy_low_when_nothing_found_and_no_guarantee_text():
    f, p, rec = rules.evaluate_model_rules(0.1, 0.66, [])
    assert p == "LOW" and f == [] and "not a guarantee" in rec


def test_score_equal_to_threshold_counts_as_flagged():
    _, p, _ = rules.evaluate_model_rules(0.66, 0.66, [])
    assert p == "MEDIUM"


def test_every_rule_has_all_required_fields():
    for r in rules.RULES.values():
        for field in ("rule_id", "description", "condition", "review_weight",
                      "recommended_review_action", "limitation"):
            assert getattr(r, field) not in (None, "")
