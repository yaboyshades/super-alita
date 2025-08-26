import logging
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.functions import kernel_function
except ImportError as e:  # pragma: no cover - optional dependency path
    raise RuntimeError(
        "semantic-kernel must be installed to use semantic_kernel_agent service"
    ) from e

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("semantic-kernel-agent")


# --- Plugin Definition ---
class IssueTrackerPlugin:
    @kernel_function(
        description=("Searches for project issues based on status or labels.")
    )
    def search_issues(
        self,
        status: Annotated[str | None, "Filter by status."] = None,
        label: Annotated[str | None, "Filter by label."] = None,
    ) -> Annotated[str, "A formatted string of matching issues."]:
        logger.info("Searching issues with status='%s' label='%s'", status, label)
        mock_db = [
            {
                "id": "AL-101",
                "title": "UI bug on login screen",
                "status": "open",
                "label": "bug",
            }
        ]
        results = mock_db
        if status:
            results = [r for r in results if r.get("status") == status]
        if label:
            results = [r for r in results if r.get("label") == label]
        if not results:
            return "No issues found."
        return "\n".join(
            f"- {r['id']}: {r['title']} (Status: {r['status']})" for r in results
        )


# --- Agent and API Server ---
_agent_instance: ChatCompletionAgent | None = None
app = FastAPI(title="Alita Semantic Kernel Agent", version="1.0.0")


async def get_agent() -> ChatCompletionAgent:
    global _agent_instance
    if _agent_instance is None:
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise RuntimeError("Azure OpenAI env vars (AZURE_OPENAI_API_KEY) not set.")
        _agent_instance = ChatCompletionAgent(
            service=AzureChatCompletion(),
            name="AlitaRefactoringExpert",
            instructions=(
                "You are an expert software engineer. Refactor code as requested, "
                "providing only the refactored block."
            ),
            plugins=[IssueTrackerPlugin()],
        )
    return _agent_instance


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest) -> ChatResponse:
    try:
        agent = await get_agent()
    except RuntimeError as e:  # Configuration error
        raise HTTPException(status_code=500, detail=str(e)) from e
    try:
        response = await agent.get_response(messages=request.prompt)
    except Exception as e:  # pragma: no cover - upstream library behavior
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail="Agent invocation failed") from e
    content = getattr(response, "content", None) or str(response)
    return ChatResponse(reply=content)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
