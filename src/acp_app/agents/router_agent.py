"""Router agent that chains other agents."""
from typing import AsyncGenerator
from acp_sdk import Message, MessagePart
from .classify_agent import ClassifyAgent


class RouterAgent:
    """Routes to other agents based on input."""

    def __init__(self):
        self.classifier = ClassifyAgent()

    async def run(self, messages: list[Message]) -> AsyncGenerator[Message, None]:
        yield Message(parts=[MessagePart(text="Routing: Sending to classifier...")])

        async for result in self.classifier.run(messages):
            yield result
