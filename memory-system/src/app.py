"""FastAPI entry-point for the memory system."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - psutil optional for tests
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.controller.conflict import detect_and_resolve_conflicts
from src.controller.ace_evolver import ACEvolver
from src.controller.context_pack import compose_ace_context, compose_context_pack
from src.controller.consolidate import run_ace_informed_consolidation, run_consolidation
from src.controller.forget import run_forgetting_policy
from src.controller.inspector import make_decisions
from src.controller.self_improving import SelfImprovingMemoryController
from src.controller.score import calculate_importance
from src.mangle.rules import apply_ingest_rules, apply_retrieval_rules, load_rules
from src.models import Conflict, ContextPack, Decision, Memory, Message
from src.redact import sanitize_text, should_quarantine
from src.stores.episodic import episodic_store
from src.stores.semantic import semantic_store
from src.stores.working import working_buffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ace_evolver = ACEvolver()
strategy_controller = SelfImprovingMemoryController()
_last_context_pack: Optional[ContextPack] = None

app = FastAPI(
    title="Passive Memory + Mangle API",
    description="Deterministic memory system with rule-based ingestion and retrieval",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    load_rules()
    logger.info("Memory system initialized")


@app.post("/capture", response_model=Dict[str, int])
def capture_messages(messages: List[Message]) -> Dict[str, int]:
    kept = 0
    quarantined = 0
    for msg in messages:
        try:
            if msg.meta.get("x-no-store"):
                continue
            meta = dict(msg.meta)
            meta.setdefault("role", msg.role)
            clean_text, was_redacted = sanitize_text(msg.content)
            if not clean_text.strip():
                continue
            if should_quarantine(msg.content) or should_quarantine(clean_text):
                quarantined += 1
                continue
            imp_score = calculate_importance(clean_text, msg.role, meta)
            rule_result = apply_ingest_rules(clean_text, imp_score, meta)
            if not rule_result["keep"]:
                continue
            memory = Memory(
                text=clean_text,
                importance=imp_score,
                tags=list(rule_result["tags"]),
                ttl_days=rule_result.get("ttl_days", 90) or 90,
                source=f"msg_{msg.id}",
                meta={
                    "role": msg.role,
                    "was_redacted": was_redacted,
                    "original_length": len(msg.content),
                    **meta,
                },
            )
            episodic_store.add(memory)
            working_buffer.add(memory)
            kept += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to process message %s: %s", msg.id, exc)
    logger.info("Captured %s/%s messages (%s quarantined)", kept, len(messages), quarantined)
    return {"received": len(messages), "kept": kept, "quarantined": quarantined}


@app.get("/context", response_model=ContextPack)
def get_context(
    q: str = Query(..., min_length=1, description="Query for context retrieval"),
    k: int = Query(12, ge=1, le=50, description="Number of memories to consider"),
    budget: int = Query(700, ge=100, le=2000, description="Token budget for context"),
    include_semantic: bool = Query(True, description="Include semantic memories"),
    explain: bool = Query(False, description="Include detailed decisions"),
) -> ContextPack:
    try:
        episodic_candidates = episodic_store.search(q, k * 2)
        semantic_candidates: List[Memory] = []
        if include_semantic:
            semantic_candidates = semantic_store.search(q, k // 2)
        all_candidates = episodic_candidates + semantic_candidates
        curated = apply_retrieval_rules(all_candidates, k)
        decisions = make_decisions(curated, q) if explain else []
        pack = compose_context_pack(decisions, curated, budget=budget, query=q)
        global _last_context_pack
        _last_context_pack = pack
        return pack
    except Exception as exc:
        logger.error("Context retrieval failed for query '%s': %s", q, exc)
        raise HTTPException(500, f"Context retrieval failed: {exc}") from exc


@app.post("/context/evolve", response_model=ContextPack)
def get_evolved_context(
    q: str = Query(..., min_length=1),
    k: int = Query(12, ge=1, le=50),
    budget: int = Query(700, ge=100, le=2000),
    enable_ace: bool = Query(True),
    include_semantic: bool = Query(True),
    evolution_feedback: Optional[Dict[str, Any]] = Body(None),
) -> ContextPack:
    try:
        episodic_candidates = episodic_store.search(q, k * 2)
        semantic_candidates: List[Memory] = []
        if include_semantic:
            semantic_candidates = semantic_store.search(q, k // 2)
        curated = apply_retrieval_rules(episodic_candidates + semantic_candidates, k)
        decisions = make_decisions(curated, q)
        base_pack = compose_context_pack(decisions, curated, budget=budget, query=q)
        if not enable_ace:
            pack = base_pack
        else:
            feedback = evolution_feedback or {}
            pack = compose_ace_context(
                decisions,
                curated,
                budget=budget,
                query=q,
                feedback=feedback,
                evolver=ace_evolver,
            )
        global _last_context_pack
        _last_context_pack = pack
        return pack
    except Exception as exc:
        logger.error("ACE context retrieval failed for query '%s': %s", q, exc)
        raise HTTPException(500, f"ACE context retrieval failed: {exc}") from exc


@app.post("/ace/strategies/evaluate", response_model=Dict[str, Any])
def evaluate_strategies(feedback: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    if _last_context_pack is None:
        raise HTTPException(400, "No context available for evaluation")
    try:
        record = strategy_controller.evaluate_context_strategy(_last_context_pack, feedback)
        return record
    except Exception as exc:
        logger.error("Strategy evaluation failed: %s", exc)
        raise HTTPException(500, f"Strategy evaluation failed: {exc}") from exc


@app.get("/ace/evolution/history", response_model=Dict[str, Any])
def get_evolution_history(limit: int = Query(10, ge=1, le=100)) -> Dict[str, Any]:
    history = strategy_controller.evolution_history[-limit:]
    return {
        "history": list(reversed(history)),
        "total_cycles": strategy_controller.evolution_cycle,
    }


@app.post("/consolidate", response_model=Dict[str, Any])
def trigger_consolidation(
    background_tasks: BackgroundTasks,
    ace: bool = Query(False, description="Run ACE-informed consolidation"),
) -> Dict[str, Any]:
    task = run_ace_informed_consolidation if ace else run_consolidation
    background_tasks.add_task(task)
    return {
        "status": "started",
        "message": (
            "ACE-informed consolidation started"
            if ace
            else "Consolidation job started in background"
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/forget", response_model=Dict[str, Any])
def trigger_forgetting() -> Dict[str, Any]:
    try:
        result = run_forgetting_policy()
        return {"status": "completed", **result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as exc:
        raise HTTPException(500, f"Forgetting failed: {exc}") from exc


@app.get("/memory/{memory_id}", response_model=Memory)
def get_memory(memory_id: str) -> Memory:
    memory = episodic_store.get(memory_id) or semantic_store.get(memory_id)
    if not memory:
        raise HTTPException(404, f"Memory {memory_id} not found")
    return memory


@app.get("/explain/{memory_id}", response_model=Dict[str, Any])
def explain_memory(memory_id: str) -> Dict[str, Any]:
    memory = episodic_store.get(memory_id) or semantic_store.get(memory_id)
    if not memory:
        raise HTTPException(404, f"Memory {memory_id} not found")
    related = episodic_store.search(memory.text, k=5)
    return {
        "memory": memory,
        "provenance": {
            "source": memory.source,
            "ingest_time": memory.ts.isoformat(),
            "last_access": memory.last_access.isoformat(),
            "access_count": memory.access_count,
            "estimated_ttl": (memory.ts + timedelta(days=memory.ttl_days)).isoformat(),
        },
        "rule_effects": {
            "tags": memory.tags,
            "importance_factors": _analyze_importance(memory.text, memory.meta.get("role")),
            "ttl_reason": _explain_ttl(memory),
        },
        "related_memories": [mem.id for mem in related if mem.id != memory_id],
    }


@app.get("/conflicts", response_model=List[Conflict])
def get_conflicts(limit: int = Query(10, ge=1, le=50)) -> List[Conflict]:
    return detect_and_resolve_conflicts(limit=limit, auto_resolve=False)


@app.post("/conflicts/resolve", response_model=Dict[str, Any])
def resolve_conflicts() -> Dict[str, Any]:
    try:
        resolved = detect_and_resolve_conflicts(limit=50, auto_resolve=True)
        return {
            "resolved": len(resolved),
            "conflicts": resolved,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(500, f"Conflict resolution failed: {exc}") from exc


@app.post("/rules/reload", response_model=Dict[str, str])
def reload_rules() -> Dict[str, str]:
    try:
        load_rules(reload=True)
        return {"status": "success", "message": "Rules reloaded successfully"}
    except Exception as exc:
        raise HTTPException(500, f"Rule reload failed: {exc}") from exc


@app.get("/healthz", response_model=Dict[str, Any])
def health_check() -> Dict[str, Any]:
    try:
        episodic_count = episodic_store.count()
        semantic_count = semantic_store.count()
        working_count = working_buffer.count()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "stores": {
                "episodic": episodic_count,
                "semantic": semantic_count,
                "working": working_count,
            },
            "memory_usage": _get_memory_usage(),
            "version": "1.0.0",
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(500, "Health check failed") from exc


@app.get("/metrics", response_model=Dict[str, Any])
def get_metrics() -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "ingest": episodic_store.get_ingest_metrics(),
        "retrieval": episodic_store.get_retrieval_metrics(),
        "consolidation": semantic_store.get_consolidation_metrics(),
        "memory_characteristics": {
            "avg_importance": episodic_store.get_average_importance(),
            "age_distribution": episodic_store.get_age_distribution(),
            "tag_distribution": episodic_store.get_tag_distribution(),
        },
    }


def _analyze_importance(text: str, role: Optional[str]) -> List[str]:
    factors: List[str] = []
    if "```" in text:
        factors.append("contains_code")
    if "?" in text and len(text) > 40:
        factors.append("contains_question")
    if len(text) > 200:
        factors.append("long_content")
    if role == "user":
        factors.append("user_message")
    return factors


def _explain_ttl(memory: Memory) -> str:
    if memory.ttl_days >= 365:
        return "long_term_asset"
    if memory.ttl_days >= 180:
        return "important_memory"
    if memory.ttl_days <= 30:
        return "transient_memory"
    return "standard_retention"


def _get_memory_usage() -> Dict[str, Any]:
    try:
        process = psutil.Process()
        return {
            "process_rss_mb": process.memory_info().rss / 1024 / 1024,
            "system_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
        }
    except Exception:  # pragma: no cover - psutil optional on some systems
        return {"process_rss_mb": 0.0, "system_available_gb": 0.0}
