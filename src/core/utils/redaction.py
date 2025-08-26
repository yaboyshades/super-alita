from __future__ import annotations

import re
from typing import Any, Dict, Tuple

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HEX32PLUS = re.compile(r"\b[0-9a-fA-F]{32,}\b")
KEYWORDS = re.compile(r"(api[_-]?key|secret|token|password)\s*[:=]\s*([^\s,;]+)", re.IGNORECASE)


def _mask(s: str) -> str:
    if not s or len(s) < 6:
        return "•••"
    return s[:3] + "…" + s[-3:]


def _redact_str(s: str) -> Tuple[str, Dict[str, int]]:
    hits = {"email": 0, "hex": 0, "keyword": 0}

    def _repl_email(m):
        hits["email"] += 1
        return _mask(m.group(0))

    def _repl_hex(m):
        hits["hex"] += 1
        return _mask(m.group(0))

    def _repl_kw(m):
        hits["keyword"] += 1
        return f"{m.group(1)}={_mask(m.group(2))}"

    s = EMAIL.sub(_repl_email, s)
    s = HEX32PLUS.sub(_repl_hex, s)
    s = KEYWORDS.sub(_repl_kw, s)
    return s, hits


def _walk(obj):
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v) for v in obj]
    if isinstance(obj, str):
        red, _ = _redact_str(obj)
        return red
    return obj


def redact_prompt_and_context(prompt: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Redact sensitive patterns. Returns (prompt_redacted, context_redacted, report)."""
    red_prompt, hits_p = _redact_str(prompt or "")
    red_ctx = _walk(context or {})
    report = {"prompt_redactions": hits_p}
    return red_prompt, red_ctx, report
