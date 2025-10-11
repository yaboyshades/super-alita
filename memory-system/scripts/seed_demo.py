"""Seed the memory system with demo data."""
from __future__ import annotations

from src.app import capture_messages
from src.models import Message, Role


DEMO_MESSAGES = [
    Message(role=Role.USER, content="My favorite color is blue and I love sushi."),
    Message(role=Role.USER, content="Remember that my birthday is 1990-07-12."),
    Message(role=Role.USER, content="Here's some code: ```python\nprint('hello')\n```"),
]


if __name__ == "__main__":
    result = capture_messages(DEMO_MESSAGES)
    print(f"Seeded demo data: {result}")
