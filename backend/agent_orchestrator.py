"""OpenAI-compatible agent swarm orchestrator (no Azure dependency).

Provides a simple multi-agent execution model by calling a configured
OpenAI-compatible endpoint (e.g., local OSS 20B served via an adapter or
Ollama) with lightweight role prompts. If the endpoint is unavailable the
service still responds with graceful degraded output.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL = os.getenv("SWARM_MODEL", os.getenv("OLLAMA_MODEL", "oss-20b"))
ENDPOINT = os.getenv("SWARM_ENDPOINT", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/api/chat"))

ROLES: dict[str, str] = {
    "architect": "You are the Architect Agent. Provide high-level structural guidance, identify risks, and propose module boundaries.",
    "refactor": "You are the Refactor Agent. Suggest concrete, incremental refactors with code snippets maintaining behavior.",
    "testgen": "You are the TestGen Agent. Produce unit + integration test cases and edge cases succinctly.",
    "debug": "You are the Debug Agent. Identify likely root causes and propose minimal fix patches.",
}

GRAPHS: dict[str, list[str]] = {
    "complex_refactor": ["architect", "refactor", "testgen"],
    "bug_fix": ["debug", "testgen"],
    "performance_optimization": ["architect", "refactor", "debug"],
    "test_generation": ["testgen"],
}


class SwarmRequest(BaseModel):
    prompt: str
    context: dict[str, Any] = {}


app = FastAPI(title="Alita Swarm", version="0.1.0")


async def _call_model(role: str, user_prompt: str, ctx: dict[str, Any]) -> str:
    """Invoke OpenAI-compatible chat endpoint for a single role."""
    system_msg = ROLES[role]
    payload: dict[str, Any]
    if ENDPOINT.endswith("/api/chat") and "ollama" in ENDPOINT.lower():  # Ollama style
        payload = {"model": MODEL, "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context: {json.dumps(ctx)[:4000]}\nUser: {user_prompt}"}
        ], "stream": False}
    else:  # Assume OpenAI style /v1/chat/completions or adapter
        payload = {"model": MODEL, "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context: {json.dumps(ctx)[:4000]}\nUser: {user_prompt}"}
        ], "stream": False}
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(ENDPOINT, json=payload)
            if r.status_code >= 400:
                return f"error: upstream {r.status_code} {r.text[:120]}"
            data = r.json()
            # Ollama: { message: { content }} ; OpenAI: { choices: [ { message: { content }}] }
            if isinstance(data, dict) and "message" in data:
                return data.get("message", {}).get("content", "")
            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
            return json.dumps(data)[:800]
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"


async def _classify(prompt: str) -> str:
    # Tiny heuristic classifier to avoid extra round-trip; can be replaced by model call.
    p = prompt.lower()
    if any(k in p for k in ["benchmark", "slow", "optimiz", "performance"]):
        return "performance_optimization"
    if any(k in p for k in ["test", "coverage", "unit"]):
        return "test_generation"
    if any(k in p for k in ["bug", "error", "traceback", "exception"]):
        return "bug_fix"
    if any(k in p for k in ["refactor", "clean", "modular", "restructure"]):
        return "complex_refactor"
    return "complex_refactor"


@app.post("/swarm/execute")
async def execute(req: SwarmRequest):  # pragma: no cover - integration
    task_type = await _classify(req.prompt)
    chain = GRAPHS.get(task_type, ["refactor"])
    results: list[dict[str, Any]] = []
    async def run(role: str):
        content = await _call_model(role, req.prompt, req.context)
        results.append({"agent": role, "content": content})
    await asyncio.gather(*(run(r) for r in chain))
    synthesis = "\n---\n".join(
        f"[{r['agent']}]\n{r['content']}" for r in results
    )
    return {
        "task_type": task_type,
        "agents": chain,
        "results": results,
        "synthesis": synthesis[:8000],
    }


@app.get("/health")
async def health():  # pragma: no cover
    return {
        "status": "ok",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "roles": list(ROLES.keys()),
    }