from __future__ import annotations

from src.controller.score import calculate_importance
from src.mangle.rules import apply_ingest_rules
from src.redact import sanitize_text, should_quarantine


def test_importance_scoring_prefers_longer_user_messages():
    short = calculate_importance("hi", "user", {})
    long = calculate_importance("I prefer coffee over tea and never skip breakfast", "user", {})
    assert long > short


def test_redaction_masks_api_keys():
    clean, redacted = sanitize_text("My API key is sk-abcdef")
    assert redacted
    assert "sk-abcdef" not in clean


def test_rules_can_tag_user_facts():
    long_fact = (
        "I prefer hiking during long autumn weekends because it helps me unwind and "
        "clear my mind after intense work sprints."
    )
    rule_result = apply_ingest_rules(
        long_fact,
        0.5,
        {"role": "user"},
    )
    assert rule_result["keep"]
    assert "user_fact" in rule_result["tags"]
