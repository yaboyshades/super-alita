"""Prompt composition helpers for Copilot integration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256


def _truncate_text(text: str, token_budget: int) -> str:
    """Return text truncated to roughly ``token_budget`` tokens.

    Roughly estimates 1 token ≈ 4 characters. Keeps determinism by simple slicing.
    """
    max_chars = token_budget * 4
    return text[:max_chars]


def _stable_hash(parts: Iterable[str]) -> str:
    """Deterministic short hash for the given text parts."""
    joined = "\n".join(parts)
    return sha256(joined.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PromptPayload:
    system: str
    user: str
    hints: Mapping[str, object]


def compose_chat_prompt(
    banner: str,
    user_message: str,
    file_summaries: Sequence[str] | None = None,
    chat_signals: Sequence[str] | None = None,
    token_budget: int = 4000,
) -> PromptPayload:
    """Compose a chat prompt for Copilot.

    Parameters
    ----------
    banner:
        System banner injected at the start of the system message.
    user_message:
        The raw user message.
    file_summaries:
        Iterable of file summary strings.
    chat_signals:
        Iterable of prior chat signal strings.
    token_budget:
        Maximum number of tokens allowed for the hints section.
    """
    files = sorted(file_summaries or [])
    signals = sorted(chat_signals or [])
    hints_parts = []
    if files:
        hints_parts.append("Files:\n" + "\n".join(files))
    if signals:
        hints_parts.append("Chat:\n" + "\n".join(signals))
    hints_text = _truncate_text("\n\n".join(hints_parts), token_budget)
    content_hash = _stable_hash([banner, user_message, hints_text])
    hints = {
        "text": hints_text,
        "files": files,
        "chat": signals,
        "content_hash": content_hash,
    }
    return PromptPayload(system=banner, user=user_message, hints=hints)


def compose_inline_prompt(
    banner: str,
    code_snippet: str,
    file_summaries: Sequence[str] | None = None,
    token_budget: int = 4000,
) -> PromptPayload:
    """Compose an inline completion prompt.

    Acts as a thin wrapper over :func:`compose_chat_prompt` without chat signals.
    """
    return compose_chat_prompt(
        banner=banner,
        user_message=code_snippet,
        file_summaries=file_summaries,
        chat_signals=None,
        token_budget=token_budget,
    )
