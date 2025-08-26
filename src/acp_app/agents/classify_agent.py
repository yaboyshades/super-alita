"""Classification agent returning structured response."""
import json
from typing import AsyncGenerator
from acp_sdk import Message, MessagePart


class ClassifyAgent:
    """Mock classifier agent."""

    async def run(self, messages: list[Message]) -> AsyncGenerator[Message, None]:
        if not messages:
            yield Message(parts=[MessagePart(text='{"error": "No input"}')])
            return

        text = ""
        for part in messages[0].parts:
            if getattr(part, "text", None):
                text += part.text

        result = {
            "input": text[:100],
            "classification": "neutral" if len(text) < 50 else "complex",
            "confidence": 0.85,
            "tokens": len(text.split()),
        }

        yield Message(parts=[MessagePart(text=json.dumps(result, indent=2))])
