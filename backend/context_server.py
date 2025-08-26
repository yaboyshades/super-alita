"""Lightweight context indexing & search service (scaffold).

Optional dependencies: chromadb, sentence_transformers.
This file degrades gracefully if deps missing, exposing /healthz only.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI  # type: ignore[import-not-found]
from pydantic import BaseModel  # type: ignore[import-not-found]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("context-server")

app = FastAPI(title="Alita Context Server", version="0.1.0")

# Use non-constant style name to allow reassignment without linter complaining
context_service_enabled: bool = False

try:  # pragma: no cover - optional heavy deps
    import chromadb  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    _client = chromadb.Client()
    _collection = _client.get_or_create_collection("alita_codebase")
    _feedback = _client.get_or_create_collection("alita_feedback")
    _client: Any = chromadb.Client()  # type: ignore[assignment]
    _collection: Any = _client.get_or_create_collection("alita_codebase")  # type: ignore[assignment]
    _feedback: Any = _client.get_or_create_collection("alita_feedback")  # type: ignore[assignment]
    _embed_model = SentenceTransformer(
        os.getenv("ALITA_EMBED_MODEL", "all-MiniLM-L6-v2"),
        device="cuda" if os.getenv("USE_GPU") else "cpu",
    )
    context_service_enabled = True  # type: ignore[assignment]
except Exception as e:  # noqa: BLE001
    logger.warning("Context indexing disabled: %s", e)


class IndexRequest(BaseModel):
    files: dict[str, str]


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class FeedbackRequest(BaseModel):
    prompt: str
    original_code: str
    final_code: str
    outcome: str  # accepted | modified | rejected


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {"status": "ok", "enabled": context_service_enabled}


@app.post("/index")
async def index_files(
    req: IndexRequest,
) -> dict[str, object]:  # pragma: no cover - IO path
    if not context_service_enabled:
        return {"status": "disabled"}
    if not req.files:
        return {"status": "no-files"}
    ids = list(req.files.keys())
    docs = list(req.files.values())
    embeds = _embed_model.encode(docs, show_progress_bar=False).tolist()
    _collection.add(ids=ids, documents=docs, embeddings=embeds)
    return {"status": "ok", "count": len(ids)}


@app.post("/search")
async def search(
    req: SearchRequest,
) -> dict[str, object]:  # pragma: no cover - IO path
    if not context_service_enabled:
        return {"results": []}
    q_emb = _embed_model.encode([req.query]).tolist()
    out = _collection.query(query_embeddings=q_emb, n_results=req.k)
    return {"results": out.get("documents", [[]])[0]}


@app.post("/feedback")
async def feedback(
    req: FeedbackRequest,
) -> dict[str, object]:  # pragma: no cover - IO path
    if not context_service_enabled:
        return {"status": "disabled"}
    doc = (
        f"Prompt: {req.prompt}\n"
        f"Original: {req.original_code}\n"
        f"Final: {req.final_code}\n"
        f"Outcome: {req.outcome}"
    )
    _feedback.add(ids=[f"fb_{len(_feedback.get().get('ids', []))}"], documents=[doc])
    return {"status": "ok"}
    if not hasattr(_feedback, "add") or not hasattr(_feedback, "get"):
        return {"status": "disabled"}
    try:
        existing = _feedback.get()  # type: ignore[call-arg]
        existing_ids = []
        if isinstance(existing, dict):
            existing_ids = existing.get("ids", []) or []
        new_id = f"fb_{len(existing_ids)}"
        _feedback.add(ids=[new_id], documents=[doc])  # type: ignore[call-arg]
    except Exception as e:  # noqa: BLE001
        logger.warning("Feedback store failure: %s", e)
        return {"status": "error", "detail": str(e)}
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # type: ignore[import-not-found]

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("CONTEXT_PORT", 5001)),
    )
