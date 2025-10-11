from __future__ import annotations

from src.app import capture_messages, get_context
from src.models import Message, Role


def test_context_pack_contains_budgeted_text():
    capture_messages(
        [
            Message(role=Role.USER, content="I love sushi and blue colors."),
            Message(role=Role.USER, content="Remember that my meeting is on 2024-03-11."),
        ]
    )
    pack = get_context("sushi", k=3, budget=120)
    assert pack.citations
    assert "sushi" in pack.text.lower()
    assert pack.budget_used <= pack.budget_total
