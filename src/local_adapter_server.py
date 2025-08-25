#!/usr/bin/env python3
"""Minimal local OpenAI-compatible adapter for a HF causal LM.

Endpoints:
  POST /v1/chat/completions  (supports messages, stream=true)
  POST /v1/completions       (supports prompt, stream=true)

Environment:
  LOCAL_HF_MODEL_ID  (default: gpt2)
  LOCAL_HF_MAX_NEW_TOKENS (default: 128)
  LOCAL_HF_DEVICE (auto|cuda|cpu) default auto

This is intentionally lightweight; for production consider batching,
KV caching, quantization, etc.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("transformers and torch are required for local adapter") from e

app = FastAPI(title="LocalHFAdapter", version="0.1.0")

MODEL_ID = os.getenv("LOCAL_HF_MODEL_ID", "gpt2")
MAX_NEW = int(os.getenv("LOCAL_HF_MAX_NEW_TOKENS", "128"))
DEVICE_PREF = os.getenv("LOCAL_HF_DEVICE", "auto")
DEVICE = 0 if (DEVICE_PREF in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)


async def generate_text(prompt: str) -> AsyncGenerator[str, None]:
    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    output = _model.generate(
        input_ids,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        pad_token_id=_tokenizer.eos_token_id,
    )
    gen_ids = output[0][input_ids.shape[-1] :]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Stream in 60 char chunks
    for i in range(0, len(text), 60):
        await asyncio.sleep(0)
        yield text[i : i + 60]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", MODEL_ID)
    # Compose a simple prompt from user messages
    user_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    prompt = user_parts[-1] if user_parts else ""

    if not stream:
        chunks = []
        async for part in generate_text(prompt):
            chunks.append(part)
        full = "".join(chunks)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full},
                    "finish_reason": "stop",
                }
            ],
        }

    async def streamer():
        async for piece in generate_text(prompt):
            data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)
    model = body.get("model", MODEL_ID)

    if not stream:
        chunks = []
        async for part in generate_text(prompt):
            chunks.append(part)
        full = "".join(chunks)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:10]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": full, "finish_reason": "stop"}],
        }

    async def streamer():
        async for piece in generate_text(prompt):
            data = {
                "id": f"cmpl-{uuid.uuid4().hex[:10]}",
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": piece, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.get("/healthz")
async def health():  # noqa: D401
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("local_adapter_server:app", host="0.0.0.0", port=8080)
