"""Complex event processor for conversational clarification detection."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Deque, Iterable, Mapping

FRUSTRATION_KEYWORDS = {
    "frustrated",
    "frustrating",
    "annoyed",
    "angry",
    "stuck",
    "irritated",
    "upset",
    "not working",
    "still broken",
    "why won't",
}

_RepetitionRecord = dict[str, Any]
_Event = Mapping[str, Any]


def _extract_event_text(event: Mapping[str, Any]) -> str:
    """Best-effort extraction of textual content from an event payload."""

    candidates: list[str] = []
    for key in ("text", "message", "content", "user_input", "prompt"):
        value = event.get(key)
        if isinstance(value, str):
            candidates.append(value)
    data = event.get("data")
    if isinstance(data, Mapping):
        for key in ("text", "message", "content", "user_input", "prompt"):
            value = data.get(key)
            if isinstance(value, str):
                candidates.append(value)
    if not candidates:
        return ""
    return candidates[0]


def _normalize_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _extract_iso_timestamp(event: Mapping[str, Any]) -> str | None:
    ts = event.get("ts") or event.get("timestamp") or event.get("time")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC).isoformat()
        except ValueError:
            return ts
    observed = event.get("_observed_at")
    if isinstance(observed, datetime):
        return observed.astimezone(UTC).isoformat()
    return None


def detect_frustration_pattern(events: Iterable[_Event]) -> str | None:
    """Return a reason string if frustration indicators are detected."""

    for event in events:
        metadata = event.get("metadata") if isinstance(event, Mapping) else None
        sentiment = ""
        if isinstance(metadata, Mapping):
            raw_sentiment = metadata.get("sentiment")
            if isinstance(raw_sentiment, str):
                sentiment = raw_sentiment.lower()
        direct_sentiment = event.get("sentiment") if isinstance(event, Mapping) else None
        if isinstance(direct_sentiment, str):
            sentiment = direct_sentiment.lower()
        if sentiment in {"frustrated", "angry", "upset", "annoyed"}:
            return f"sentiment:{sentiment}"
        text = _extract_event_text(event)
        normalized = text.lower()
        for keyword in FRUSTRATION_KEYWORDS:
            if keyword in normalized:
                return f"keyword:{keyword}"
    return None


def detect_repetition_pattern(
    events: Iterable[_Event], *, minimum_repetition: int = 2
) -> str | None:
    """Return the repeated text when the minimum repetition threshold is met."""

    counts: dict[str, _RepetitionRecord] = {}
    for event in events:
        text = _extract_event_text(event)
        if not text:
            continue
        normalized = _normalize_text(text)
        if not normalized:
            continue
        record = counts.setdefault(normalized, {"count": 0, "text": text})
        record["count"] += 1
        # Keep the most recent observed variant for context
        record["text"] = text
    for record in counts.values():
        if record["count"] >= minimum_repetition:
            return record["text"]
    return None


def build_clarification_context(
    events: Iterable[_Event], repeated_text: str, frustration_reason: str
) -> dict[str, Any]:
    """Construct a context payload for clarification opportunities."""

    history: list[dict[str, Any]] = []
    for event in list(events)[-5:]:
        history.append(
            {
                "type": event.get("type") or event.get("event_type"),
                "text": _extract_event_text(event) or None,
                "timestamp": _extract_iso_timestamp(event),
            }
        )
    return {
        "repeated_text": repeated_text,
        "frustration_reason": frustration_reason,
        "history": history,
    }


class ComplexEventProcessor:
    """Observes events and emits clarification opportunities when patterns align."""

    def __init__(
        self,
        emit_callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        *,
        relevant_event_types: Iterable[str] | None = None,
        window: timedelta | None = None,
        max_events: int = 20,
        minimum_repetition: int = 2,
        output_event_type: str = "clarification_opportunity",
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._emit_callback = emit_callback
        self._relevant_event_types = {
            _normalize_text(t) for t in (relevant_event_types or ("user_message", "conversation.user"))
        }
        self._window = window or timedelta(minutes=3)
        self._max_events = max_events
        self._minimum_repetition = minimum_repetition
        self.output_event_type = output_event_type
        self._history: dict[str, Deque[dict[str, Any]]] = defaultdict(deque)
        self._last_emitted: dict[tuple[str, str], datetime] = {}
        self._now = now_factory or (lambda: datetime.now(tz=UTC))

    def update_emit_callback(
        self, emit_callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    ) -> None:
        self._emit_callback = emit_callback

    async def observe(self, event: Mapping[str, Any]) -> None:
        if not self._emit_callback:
            return
        if not self._is_relevant(event):
            return
        bucket = self._resolve_bucket(event)
        observed_at = self._resolve_timestamp(event)
        record = dict(event)
        record["_observed_at"] = observed_at
        window = self._history[bucket]
        window.append(record)
        while len(window) > self._max_events:
            window.popleft()
        cutoff = observed_at - self._window
        while window and window[0].get("_observed_at", observed_at) < cutoff:
            window.popleft()
        frustration_reason = detect_frustration_pattern(window)
        repeated_text = detect_repetition_pattern(
            window, minimum_repetition=self._minimum_repetition
        )
        if not (frustration_reason and repeated_text):
            return
        normalized_repeated = _normalize_text(repeated_text)
        dedupe_key = (bucket, normalized_repeated)
        previous_emission = self._last_emitted.get(dedupe_key)
        if previous_emission and (observed_at - previous_emission) < self._window:
            return
        context = build_clarification_context(window, repeated_text, frustration_reason)
        payload = {
            "type": self.output_event_type,
            "event_type": self.output_event_type,
            "session_id": event.get("session_id"),
            "correlation_id": event.get("correlation_id"),
            "context": context,
            "source": "complex_event_processor",
            "ts": self._now().isoformat(),
        }
        await self._emit_callback(payload)
        self._last_emitted[dedupe_key] = observed_at

    def _is_relevant(self, event: Mapping[str, Any]) -> bool:
        event_type = self._extract_type(event)
        if not event_type:
            return False
        normalized = _normalize_text(event_type)
        if not self._relevant_event_types:
            return True
        return normalized in self._relevant_event_types

    def _extract_type(self, event: Mapping[str, Any]) -> str:
        raw_type = event.get("type") or event.get("event_type")
        if isinstance(raw_type, str):
            return raw_type
        return ""

    def _resolve_bucket(self, event: Mapping[str, Any]) -> str:
        for key in ("session_id", "conversation_id", "correlation_id"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value
        return "global"

    def _resolve_timestamp(self, event: Mapping[str, Any]) -> datetime:
        ts = event.get("timestamp") or event.get("ts") or event.get("time")
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=UTC)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                pass
        return self._now()
