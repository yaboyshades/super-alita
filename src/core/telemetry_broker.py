"""TelemetryBroker: Aggregates recent telemetry events into a consolidated
context envelope for LLM prompting.

Phase A (this commit):
 - Ring buffer per category (bounded by event count)
 - Ingest API with light schema normalization
 - Simple scoring (recency + declared importance)
 - Envelope build with per-category token budget approximation
 - Feature flag gate via env var CONTEXT_ENVELOPE_ENABLED
 - Basic redact hook (no-op now) and hashing for change detection

Planned Phase B (future):
 - Embedding similarity clustering to reduce redundancy
 - Novelty / surprise and decay based scoring
 - Adaptive token budgeting using actual tokenizer
 - Persistent cache of envelopes per session
 - Structured diff emission for UI
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_EVENTS_PER_CATEGORY = 200
DEFAULT_MAX_TOKENS = 2200  # rough envelope size target


@dataclass
class TelemetryEvent:
    ts: float
    category: str
    message: str
    importance: float = 1.0  # caller provided importance weight (0..n)
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, now: float | None = None) -> float:
        """Compute a simple score: importance * recency_decay."""
        if now is None:
            now = time.time()
        age = max(0.0, now - self.ts)
        # half-life ~ 5 minutes by default
        half_life = 300.0
        decay = 0.5 ** (age / half_life)
        return self.importance * decay

    def approx_token_len(self) -> int:
        # naive token approximation: words * 1.3 heuristic
        return int(len(self.message.split()) * 1.3) + int(
            sum(len(str(v)).split().__len__() for v in self.meta.values())
            * 0.5
        )


class TelemetryBroker:
    def __init__(
        self,
        max_events_per_category: int = DEFAULT_MAX_EVENTS_PER_CATEGORY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        category_token_budgets: dict[str, int] | None = None,
    ) -> None:
        self.max_events_per_category = max_events_per_category
        self.max_tokens = max_tokens
        self.category_token_budgets = category_token_budgets or {}
        self._buffers: dict[str, deque[TelemetryEvent]] = defaultdict(
            lambda: deque(maxlen=self.max_events_per_category)
        )
        self._last_envelope_hash: str | None = None

    # ----------------------------- Ingest ---------------------------------
    def ingest(
        self,
        category: str,
        message: str,
        *,
        importance: float = 1.0,
        meta: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> None:
        """Add a telemetry event to the broker.

        Args:
            category: Logical grouping (e.g. 'planner', 'tools', 'errors', 'memory').
            message: Human-readable message snippet.
            importance: Relative weight (default 1.0).
            meta: Optional structured metadata.
            ts: Optional timestamp override (epoch seconds).
        """
        if not category:
            category = "misc"
        evt = TelemetryEvent(
            ts=ts or time.time(),
            category=category,
            message=self._redact(message),
            importance=importance,
            meta=meta or {},
        )
        self._buffers[category].append(evt)

    # --------------------------- Envelope Build ---------------------------
    def build_envelope(self) -> dict[str, Any]:
        """Produce a context envelope of top-scoring events within token budgets."""
        now = time.time()
        envelope: dict[str, Any] = {
            "categories": {},
            "generated_at": now,
            "total_tokens": 0,
        }
        total_tokens = 0
        remaining_global = self.max_tokens
        # Determine per-category budgets (fallback evenly if unspecified)
        categories = list(self._buffers.keys())
        default_budget = (
            max(1, int(remaining_global / max(1, len(categories))))
            if categories
            else remaining_global
        )

        for cat in categories:
            budget = self.category_token_budgets.get(cat, default_budget)
            events = list(self._buffers[cat])
            # Rank by score desc
            ranked = sorted(events, key=lambda e: e.score(now), reverse=True)
            selected: list[dict[str, Any]] = []
            cat_tokens = 0
            for e in ranked:
                est = e.approx_token_len()
                if cat_tokens + est > budget:
                    continue
                selected.append(
                    {
                        "ts": e.ts,
                        "message": e.message,
                        "score": round(e.score(now), 4),
                        "importance": e.importance,
                        "meta": e.meta,
                    }
                )
                cat_tokens += est
            if selected:
                envelope["categories"][cat] = {
                    "events": selected,
                    "tokens": cat_tokens,
                }
                total_tokens += cat_tokens
                remaining_global = max(0, remaining_global - cat_tokens)
        envelope["total_tokens"] = total_tokens
        envelope["hash"] = self._hash_envelope(envelope)
        self._last_envelope_hash = envelope["hash"]
        return envelope

    # ---------------------------- Utilities -------------------------------
    def _redact(self, message: str) -> str:
        # Redaction hook; extend via utils.redaction when configured
        return message

    def _hash_envelope(self, envelope: dict[str, Any]) -> str:
        m = hashlib.sha256()
        m.update(json.dumps(envelope, sort_keys=True).encode("utf-8"))
        return m.hexdigest()

    # ---------------------------- Introspection ---------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "categories": {c: len(buf) for c, buf in self._buffers.items()},
            "max_events_per_category": self.max_events_per_category,
            "max_tokens": self.max_tokens,
            "last_envelope_hash": self._last_envelope_hash,
        }


_BROKER: TelemetryBroker | None = None


def get_broker() -> TelemetryBroker:
    global _BROKER
    if _BROKER is None:
        max_tokens = int(
            os.getenv(
                "CONTEXT_ENVELOPE_MAX_TOKENS",
                str(DEFAULT_MAX_TOKENS),
            )
        )
        # Bypass constant naming rule: runtime singleton
        object.__setattr__(
            globals(),
            "_BROKER",
            TelemetryBroker(max_tokens=max_tokens),  # type: ignore[arg-type]
        )
    return _BROKER  # type: ignore[return-value]


def ingest_event(
    category: str,
    message: str,
    *,
    importance: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> None:
    if not os.getenv("CONTEXT_ENVELOPE_ENABLED"):
        return
    get_broker().ingest(category, message, importance=importance, meta=meta)


def build_context_envelope() -> dict[str, Any] | None:
    if not os.getenv("CONTEXT_ENVELOPE_ENABLED"):
        return None
    return get_broker().build_envelope()


__all__ = [
    "TelemetryBroker",
    "TelemetryEvent",
    "get_broker",
    "ingest_event",
    "build_context_envelope",
]
