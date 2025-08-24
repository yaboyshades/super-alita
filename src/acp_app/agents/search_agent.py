"""Search agent integrating with Perplexica tool."""
import json
from typing import AsyncGenerator
from acp_sdk import Message, MessagePart

from src.tools.perplexica_tool import PerplexicaSearchTool


class SearchAgent:
    """AI-powered search agent."""

    def __init__(self):
        self.search_tool = PerplexicaSearchTool()

    async def run(self, messages: list[Message]) -> AsyncGenerator[Message, None]:
        if not messages:
            yield Message(parts=[MessagePart(text="Error: No search query provided")])
            return

        query = ""
        mode = "web"

        for part in messages[0].parts:
            if getattr(part, "text", None):
                query += part.text
            if getattr(part, "metadata", None) and "mode" in part.metadata:
                mode = part.metadata["mode"]

        if not query.strip():
            yield Message(parts=[MessagePart(text="Error: Empty search query")])
            return

        try:
            result = await self.search_tool(
                query=query,
                mode=mode,  # type: ignore[arg-type]
                max_results=10,
                rerank=True,
                cite=True,
                followups=2,
            )

            yield Message(parts=[MessagePart(text=f"## Search Results\n\n{result['summary']}")])

            if result.get("reasoning"):
                yield Message(
                    parts=[MessagePart(text=f"\n## Reasoning\n{result['reasoning']}")]
                )

            if result.get("citations"):
                citations_text = "\n## Sources\n"
                for i, citation in enumerate(result["citations"], 1):
                    citations_text += f"[{i}] {citation['title']} - {citation['url']}\n"
                yield Message(parts=[MessagePart(text=citations_text)])

            if result.get("followup_questions"):
                followups_text = "\n## Follow-up Questions\n"
                for q in result["followup_questions"]:
                    followups_text += f"• {q}\n"
                yield Message(parts=[MessagePart(text=followups_text)])

            yield Message(
                parts=[
                    MessagePart(
                        text=f"\n*Confidence: {result.get('confidence', 0):.1%}*"
                    )
                ]
            )

        except Exception as e:  # pragma: no cover - simple error path
            yield Message(parts=[MessagePart(text=f"Search error: {str(e)}")])
