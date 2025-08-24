"""Echo agent that mirrors input."""
from typing import AsyncGenerator
from acp_sdk import Message, MessagePart


class EchoAgent:
    """Simple echo agent for testing."""

    async def run(self, messages: list[Message]) -> AsyncGenerator[Message, None]:
        for msg in messages:
            yield Message(
                parts=[
                    MessagePart(text=f"Echo: {part.text}")
                    for part in msg.parts
                    if getattr(part, "text", None)
                ]
            )
