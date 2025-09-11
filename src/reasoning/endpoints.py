from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel


class ConsensusRequest(BaseModel):
    code: str
    context_before: list[str] = []
    context_after: list[str] = []
    language: str
    method: str
    include_alternatives: bool = True
    cursor_position: int = 0


class ConsensusResponse(BaseModel):
    method: str
    confidence: float
    reasoning: list[str]
    alternatives: list[str]
    samples: list[dict[str, Any]]


class DeepConfRequest(BaseModel):
    raw_confidence: list[float]
    domain: str
    context: dict[str, Any] = {}


class DeepConfResponse(BaseModel):
    calibrated_confidence: float
    calibration_factors: dict[str, float]


class MangleRequest(BaseModel):
    code: str
    language: str
    analyze_semantics: bool = True
    analyze_architecture: bool = True
    analyze_patterns: bool = True
    analyze_security: bool = True
    analyze_performance: bool = True


class MangleResponse(BaseModel):
    insights: list[dict[str, Any]]
    semantic_score: float
    confidence: float


def register_reasoning_endpoints(
    app: FastAPI,
    require_api_key=Depends,  # type: ignore[assignment]
    enforce_rate_limit=Depends,  # type: ignore[assignment]
):
    # Use dependencies from the enclosing app module if available
    try:
        from src.main import enforce_rate_limit as _rl  # type: ignore
        from src.main import require_api_key as _req  # type: ignore
    except Exception:  # pragma: no cover - fallback when importing outside app
        _req = lambda *_args, **_kwargs: None  # type: ignore
        _rl = lambda *_args, **_kwargs: None  # type: ignore

    @app.post("/reasoning/consensus", response_model=ConsensusResponse)  # type: ignore
    async def analyze_with_consensus(
        request: Request,  # type: ignore
        body: ConsensusRequest,  # type: ignore
        _auth: None = Depends(_req),  # type: ignore
        _rl_dep: None = Depends(_rl),  # type: ignore
    ) -> ConsensusResponse:  # type: ignore
        """Apply enhanced consensus via registered ability when available."""
        registry = getattr(app.state, "ability_registry", None)  # type: ignore[attr-defined]
        if not registry or not getattr(registry, "knows", lambda *_: False)("deepconf_consensus"):
            # Fallback minimal response
            return ConsensusResponse(
                method=body.method,
                confidence=0.5,
                reasoning=["Fallback consensus: deepconf_consensus not available"],
                alternatives=[],
                samples=[],
            )
        prompt = (
            "Analyze this code for completion suggestions.\n\n"
            f"Language: {body.language}\n"
            f"ContextBefore(last5): {body.context_before[-5:]}\n"
            f"ContextAfter(next3): {body.context_after[:3]}\n\n"
            f"Code:\n{body.code[:4000]}\n"
            "Return confidence, reasoning, alternatives, and samples."
        )
        args = {
            "prompt": prompt,
            "num_samples": 3,
            "temperature": 0.7,
            "max_tokens": 512,
            "method": body.method,
            "confidence_threshold": 0.5,
            "temperature_range": 0.2,
        }
        try:
            res: dict[str, Any] = await registry.execute("deepconf_consensus", args)  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Consensus failed: {e}")
        return ConsensusResponse(
            method=body.method,
            confidence=float(res.get("consensus_confidence", 0.5) or 0.5),
            reasoning=["Consensus aggregation complete"],
            alternatives=list(res.get("individual_responses", [])[:3])
            if isinstance(res.get("individual_responses", []), list)
            else [],
            samples=[],
        )

    @app.post("/reasoning/deepconf", response_model=DeepConfResponse)  # type: ignore
    async def calibrate_confidence(
        body: DeepConfRequest,  # type: ignore
        _auth: None = Depends(_req),  # type: ignore
        _rl_dep: None = Depends(_rl),  # type: ignore
    ) -> DeepConfResponse:  # type: ignore
        # Simple calibration fallback: average with light adjustments
        vals = body.raw_confidence or [0.5]
        avg = sum(vals) / max(1, len(vals))
        complexity = float(body.context.get("complexity", 0.5) or 0.5)
        consistency = float(body.context.get("consistency", 0.5) or 0.5)
        coverage = float(body.context.get("coverage", 0.5) or 0.5)
        calibrated = max(0.0, min(1.0, avg * (0.6 + 0.2 * consistency + 0.2 * coverage) - 0.1 * complexity))
        return DeepConfResponse(
            calibrated_confidence=calibrated,
            calibration_factors={
                "complexity": complexity,
                "consistency": consistency,
                "coverage": coverage,
            },
        )

    @app.post("/reasoning/mangle", response_model=MangleResponse)  # type: ignore
    async def analyze_with_mangle(
        body: MangleRequest,  # type: ignore
        _auth: None = Depends(_req),  # type: ignore
        _rl_dep: None = Depends(_rl),  # type: ignore
    ) -> MangleResponse:  # type: ignore
        insights: list[dict[str, Any]] = []
        code = body.code
        lang = (body.language or "").lower()
        if body.analyze_architecture and ("class " in code and lang in {"python", "javascript", "typescript", "java", "csharp"}):
            insights.append({
                "type": "architectural",
                "insight": "Class definition detected",
                "confidence": 0.9,
                "impact": "medium",
                "suggestion": "Ensure SOLID principles and proper encapsulation",
            })
        if body.analyze_performance and (" for " in code or "for(" in code):
            insights.append({
                "type": "performance",
                "insight": "Loop detected",
                "confidence": 0.7,
                "impact": "medium",
                "suggestion": "Consider vectorization/comprehensions where applicable",
            })
        if body.analyze_security and any(t in code.lower() for t in ["sql", "eval", "exec", "input("]):
            insights.append({
                "type": "security",
                "insight": "Potential security-sensitive usage",
                "confidence": 0.9,
                "impact": "high",
                "suggestion": "Validate and sanitize inputs; avoid eval/exec",
            })
        # Simple scoring
        if not insights:
            return MangleResponse(insights=[], semantic_score=0.5, confidence=0.5)
        score = sum(float(i.get("confidence", 0.5)) for i in insights) / len(insights)
        return MangleResponse(insights=insights, semantic_score=score, confidence=score)

