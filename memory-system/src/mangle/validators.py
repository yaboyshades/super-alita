"""Validation helpers for rule definitions."""
from __future__ import annotations

from typing import Any, Dict, Iterable


REQUIRED_RULE_KEYS = {"id", "when", "then"}


def validate_rule_schema(rule: Dict[str, Any]) -> None:
    missing = REQUIRED_RULE_KEYS - rule.keys()
    if missing:
        raise ValueError(f"Rule missing keys: {sorted(missing)}")
    if not isinstance(rule["when"], Iterable) or isinstance(rule["when"], (str, bytes)):
        raise ValueError("Rule 'when' must be an iterable of conditions")
    if not isinstance(rule["then"], Iterable) or isinstance(rule["then"], (str, bytes)):
        raise ValueError("Rule 'then' must be an iterable of effects")


def validate_ruleset(ruleset: Dict[str, Any]) -> None:
    ingest = ruleset.get("ingest", [])
    for rule in ingest:
        validate_rule_schema(rule)
