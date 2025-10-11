"""Rule engine governing ingest and retrieval policies."""
from __future__ import annotations

import operator
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml

from src.models import Memory

BASE_DIR = Path(__file__).resolve().parents[2]
RULES_FILE = BASE_DIR / "rulesets" / "rules.default.yaml"

_rules_cache: Dict[str, Any] | None = None
_compiled_rules: Dict[str, Any] | None = None

OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda x, y: x in y,
    "contains": operator.contains,
}


def load_rules(reload: bool = False) -> Dict[str, Any]:
    """Load rule configuration from disk, compiling conditions for fast evaluation."""

    global _rules_cache, _compiled_rules
    if _rules_cache is None or reload:
        try:
            with RULES_FILE.open("r", encoding="utf-8") as handle:
                _rules_cache = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            _rules_cache = _default_rules()
        _compiled_rules = _compile_rules(_rules_cache)
    assert _compiled_rules is not None
    return _compiled_rules


def apply_ingest_rules(text: str, importance: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    rules = load_rules()
    context = _build_context(text, importance, meta)

    keep = importance >= 0.1
    tags: set[str] = set()
    ttl_days: int | None = None
    actions: set[str] = set()

    for rule in rules.get("ingest", []):
        if rule["condition"](context):
            for effect in rule["effects"]:
                effect_type = effect["type"]
                if effect_type == "keep":
                    keep = effect["value"]
                elif effect_type == "tag":
                    tags.add(effect["value"])
                elif effect_type == "ttl":
                    ttl_days = effect["value"]
                elif effect_type == "action":
                    actions.add(effect["value"])

    if "discard" in actions or "quarantine" in actions:
        keep = False

    return {"keep": keep, "tags": tags, "ttl_days": ttl_days, "actions": actions}


def apply_retrieval_rules(memories: List[Memory], k: int) -> List[Memory]:
    compiled = load_rules().get("retrieval", {})

    scored: List[Tuple[float, Memory]] = []
    for memory in memories:
        score = _retrieval_score(memory, compiled)
        scored.append((score, memory))

    scored.sort(key=lambda item: item[0], reverse=True)
    curated = _apply_diversity(scored, k, compiled.get("diversity_threshold", 0.8))
    curated = _apply_constraints(curated, compiled.get("ensure", []))
    return curated[:k]


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def _compile_rules(rules: Dict[str, Any]) -> Dict[str, Any]:
    compiled = {"ingest": [], "retrieval": rules.get("retrieval", {})}
    for rule in rules.get("ingest", []):
        compiled_rule = {
            "id": rule["id"],
            "condition": _compile_condition(rule["when"]),
            "effects": _compile_effects(rule["then"]),
        }
        compiled["ingest"].append(compiled_rule)
    return compiled


def _compile_condition(conditions: List[str]) -> Callable[[Dict[str, Any]], bool]:
    def _condition(context: Dict[str, Any]) -> bool:
        for condition in conditions:
            if not _evaluate_condition(condition, context):
                return False
        return True

    return _condition


def _compile_effects(effects: List[str]) -> List[Dict[str, Any]]:
    compiled_effects: List[Dict[str, Any]] = []
    for effect in effects:
        if effect.startswith("ttl_days="):
            compiled_effects.append({"type": "ttl", "value": int(effect.split("=", 1)[1])})
        elif effect.startswith("tag:"):
            compiled_effects.append({"type": "tag", "value": effect.split(":", 1)[1]})
        elif effect.startswith("action:"):
            compiled_effects.append({"type": "action", "value": effect.split(":", 1)[1]})
        elif effect == "keep":
            compiled_effects.append({"type": "keep", "value": True})
        elif effect == "discard":
            compiled_effects.append({"type": "keep", "value": False})
    return compiled_effects


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _build_context(text: str, importance: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": text,
        "content.len": len(text),
        "content.words": len(text.split()),
        "importance": importance,
        "meta": meta,
        "meta.role": meta.get("role", ""),
        "now": datetime.utcnow(),
    }


def _evaluate_condition(condition: str, context: Dict[str, Any]) -> bool:
    if " and " in condition:
        return all(_evaluate_condition(part.strip(), context) for part in condition.split(" and "))
    if " or " in condition:
        return any(_evaluate_condition(part.strip(), context) for part in condition.split(" or "))

    for op in [">=", "<=", "!=", "==", ">", "<"]:
        if op in condition:
            left, right = [segment.strip() for segment in condition.split(op, 1)]
            left_val = _resolve(left, context)
            right_val = _parse(right, context)
            comparator = OPERATORS[op]
            return comparator(left_val, right_val)

    if condition.startswith("content.contains:"):
        target = condition.split(":", 1)[1].strip().strip("'\"")
        return target in context["content"]
    if condition.startswith("content.regex:"):
        pattern = condition.split(":", 1)[1].strip().strip("'\"")
        return bool(re.search(pattern, context["content"], re.IGNORECASE))
    return False


def _resolve(path: str, context: Dict[str, Any]) -> Any:
    if path in context:
        return context[path]
    parts = path.split(".")
    value: Any = context
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _parse(value: str, context: Dict[str, Any]) -> Any:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.replace(".", "", 1).isdigit():
        return float(value) if "." in value else int(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if "." in value or value in context:
        resolved = _resolve(value, context)
        if resolved is not None:
            return resolved
    return value


# ---------------------------------------------------------------------------
# Retrieval scoring helpers
# ---------------------------------------------------------------------------

def _retrieval_score(memory: Memory, rules: Dict[str, Any]) -> float:
    score = memory.importance
    days_old = (datetime.utcnow() - memory.ts).days
    recency = max(0.1, 1.0 - days_old / 90.0)
    score += recency * 0.3
    score += min(0.2, memory.access_count * 0.05)

    for promotion in rules.get("promote", []):
        if _matches_promotion(memory, promotion):
            score += 0.2
    return min(score, 1.0)


def _matches_promotion(memory: Memory, rule: str) -> bool:
    if rule == "recent<=30d":
        return (datetime.utcnow() - memory.ts).days <= 30
    if rule == "importance>=0.5":
        return memory.importance >= 0.5
    if rule.startswith("tag:"):
        return rule.split(":", 1)[1] in memory.tags
    return False


def _apply_diversity(scored: List[Tuple[float, Memory]], k: int, threshold: float) -> List[Memory]:
    if not scored:
        return []
    selected: List[Memory] = [scored[0][1]]
    selected_embeddings = [selected[0].embeddings] if selected[0].embeddings else []

    for _, memory in scored[1:]:
        if len(selected) >= k:
            break
        is_diverse = True
        if memory.embeddings and selected_embeddings:
            for other in selected_embeddings:
                if other is None:
                    continue
                if _cosine_similarity(memory.embeddings, other) > threshold:
                    is_diverse = False
                    break
        if is_diverse:
            selected.append(memory)
            if memory.embeddings:
                selected_embeddings.append(memory.embeddings)
    return selected


def _apply_constraints(memories: List[Memory], ensure_rules: List[str]) -> List[Memory]:
    if not ensure_rules:
        return memories
    # For this MVP we do not enforce additional constraints beyond pass-through.
    return memories


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if not norm1 or not norm2:
        return 0.0
    return dot / (norm1 * norm2)


def _default_rules() -> Dict[str, Any]:
    return {
        "ingest": [
            {
                "id": "keep_code",
                "when": ["content.contains:```"],
                "then": ["ttl_days=365", "tag:code", "keep"],
            },
            {
                "id": "drop_secrets",
                "when": ["content.regex:(?i)(api[_-]?key|secret|token)\\s*[:=]"],
                "then": ["action:quarantine"],
            },
            {
                "id": "prefer_user_facts",
                "when": ["meta.role==user", "content.len>80"],
                "then": ["tag:user_fact", "keep"],
            },
        ],
        "retrieval": {
            "topk": 40,
            "promote": ["recent<=30d", "importance>=0.5", "tag:profile"],
            "demote": ["dup_sim>=0.9", "stale>365d"],
            "ensure": ["at_least_one:profile"],
            "diversity_threshold": 0.85,
        },
    }
