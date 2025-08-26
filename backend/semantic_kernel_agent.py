import logging
import os
from typing import Annotated

import httpx
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


def _ollama_available(host: str) -> bool:
    try:
        r = httpx.get(host + "/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:  # pragma: no cover - network errors
        return False


async def get_agent() -> ChatCompletionAgent:
    """Instantiate (singleton) ChatCompletionAgent with provider selection.

    Provider precedence for this backend (simplified):
    1. Ollama if OLLAMA_HOST reachable and OLLAMA_MODEL set
    2. Azure OpenAI (env vars present)
    Future: Could extend to OpenAI / Anthropic etc. when SK connectors added here.
    """
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL")

    use_ollama = False
    if (
        provider in ("ollama", "auto")
        and ollama_model
        and _ollama_available(ollama_host)
    ):
        use_ollama = True

    if use_ollama:
        # Minimal Ollama chat wrapper (no SK native connector yet) using a lambda.
        class OllamaWrapper:  # pragma: no cover - simple adapter
            async def complete_chat(self, messages: list[dict]) -> str:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{ollama_host}/api/chat",
                        json={
                            "model": ollama_model,
                            "messages": messages,
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # ollama returns {message: {content: str}, ...}
                    return data.get("message", {}).get("content", "")

        async def _service_invoke(prompt: str) -> str:
            # SK ChatCompletionAgent expects the underlying service to provide a method
            # similar to openai style; we emulate with a single user message.
            wrapper = OllamaWrapper()
            return await wrapper.complete_chat([{"role": "user", "content": prompt}])

        class _ShimService:  # pragma: no cover - adapter
            async def get_chat_message_contents(self, *_, **__):  # type: ignore[override]
                raise NotImplementedError

            async def complete_chat(self, *_, **__):  # compatibility fallback
                raise NotImplementedError

        # We will inject by monkey patching ChatCompletionAgent.get_response call path
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(),  # placeholder to satisfy constructor
            name="AlitaOllamaExpert",
            instructions=(
                "You are an expert software engineer. Refactor code as requested, "
                "providing only the refactored block."
            ),
            plugins=[IssueTrackerPlugin()],
        )

        async def _patched_get_response(*_, messages: str, **__):  # type: ignore[override]
            return await _service_invoke(messages)

        agent.get_response = _patched_get_response  # type: ignore[assignment]
        _agent_instance = agent
        logger.info(
            "Semantic Kernel agent initialized with Ollama model '%s'", ollama_model
        )
        return _agent_instance

    # Fallback to Azure OpenAI
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        raise RuntimeError(
            "No supported provider available: missing Azure credentials and "
            "no reachable Ollama."
        )

    _agent_instance = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="AlitaRefactoringExpert",
        instructions=(
            "You are an expert software engineer. Refactor code as requested, "
            "providing only the refactored block."
        ),
        plugins=[IssueTrackerPlugin()],
    )
    logger.info("Semantic Kernel agent initialized with Azure OpenAI")
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
