from __future__ import annotations

import re
import time
import uuid
from typing import Any

from src.contracts.gates.common_gates import (
    CombinedGate,
    PytestGate,
    RequiredPathsGate,
    SafetyGate,
)
from src.core.event_bus import EventBus
from src.core.events import create_event
from src.native_deepcode_api import NativeDeepCodeAPI, get_native_deepcode_api
from src.policies.need_detector import NeedDetector

# Capability templates: strictly ability-first, no domain specifics.
CAPABILITY_TEMPLATES: dict[str, dict[str, Any]] = {
    "web_scraper": {
        "task_kind": "web_scraper",
        "required_paths": [
            re.compile(r"^src/abilities/.*scrap", re.I),
            re.compile(r"^tests/abilities/.*scrap", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*scrap", re.I)],
        "requirements": lambda desc: (
            f"""Build a generic, spec-driven web scraping *ability*:
- Accept runtime selectors/regex/attrs (no site hardcoding).
- Output CSV+JSON; headers: url, sku, title, price, currency.
- Include tests (no network), docs, and Makefile target.
- Safety: no eval/os.system/subprocess(..., shell=True). 
  Retries, timeouts, UA, polite delays.
Task: {desc}
"""
        ),
    },
    "etl_task": {
        "task_kind": "etl_task",
        "required_paths": [
            re.compile(r"^src/abilities/.*etl", re.I),
            re.compile(r"^tests/abilities/.*etl", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*etl", re.I)],
        "requirements": lambda desc: f"""Build a configurable ETL *ability*:
- Declarative spec: inputs -> transforms -> outputs (csv/json).
- Idempotent; schema validation; unit tests; docs; Makefile target.
- Safety rules same as above.
Task: {desc}
""",
    },
    "api_client": {
        "task_kind": "api_client",
        "required_paths": [
            re.compile(r"^src/abilities/.*api_client", re.I),
            re.compile(r"^tests/abilities/.*api_client", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*api", re.I)],
        "requirements": lambda desc: f"""Build a typed API client *ability*:
- Declarative endpoints spec; retries/backoff; timeout; auth hooks; 
  tests; docs; Makefile target.
- No secrets in code; safety enforced.
Task: {desc}
""",
    },
    "report_generator": {
        "task_kind": "report_generator",
        "required_paths": [
            re.compile(r"^src/abilities/.*report", re.I),
            re.compile(r"^tests/abilities/.*report", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*report", re.I)],
        "requirements": lambda desc: f"""Build a report generator *ability*:
- Input data (csv/json) -> templated summary -> csv/xlsx/pdf; 
  tests; docs; Makefile target.
- Deterministic outputs; safety enforced.
Task: {desc}
""",
    },
    "paper2code": {
        "task_kind": "paper2code",
        "required_paths": [
            re.compile(r"^src/abilities/.*paper.*code", re.I),
            re.compile(r"^tests/abilities/.*paper.*code", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*paper.*code", re.I)],
        "requirements": lambda desc: f"""Build a research paper implementation *ability*:
- Extract algorithms and models from research papers
- Generate production-ready PyTorch/TensorFlow implementations
- Include comprehensive tests, documentation, and examples
- Preserve mathematical accuracy and computational complexity
- Support for attention mechanisms, transformers, neural networks
- Safety: no eval/exec, proper tensor operations, memory management
Task: {desc}
""",
    },
    "ml_algorithm": {
        "task_kind": "ml_algorithm", 
        "required_paths": [
            re.compile(r"^src/abilities/.*ml.*algo", re.I),
            re.compile(r"^tests/abilities/.*ml.*algo", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*ml.*algo", re.I)],
        "requirements": lambda desc: f"""Build a machine learning algorithm *ability*:
- Implement core ML algorithms (classification, regression, clustering)
- Include training loops, optimization, and evaluation metrics
- Support for different backends (numpy, sklearn, pytorch)
- Comprehensive tests with synthetic datasets
- Documentation with mathematical background
- Safety: validated inputs, numerical stability, memory efficiency
Task: {desc}
""",
    },
    "web_search": {
        "task_kind": "web_search",
        "required_paths": [
            re.compile(r"^src/abilities/.*search", re.I),
            re.compile(r"^tests/abilities/.*search", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*search", re.I)],
        "requirements": lambda desc: f"""Build a web search *ability*:
- Multi-mode search: web, academic, news, images, videos, reddit
- AI-powered analysis with reasoning and summarization
- Source citation and relevance scoring
- Configurable result limits and search parameters
- Include comprehensive tests, documentation, and examples
- Safety: proper URL validation, rate limiting, error handling
Task: {desc}
""",
    },
    "research_assistant": {
        "task_kind": "research_assistant",
        "required_paths": [
            re.compile(r"^src/abilities/.*research", re.I),
            re.compile(r"^tests/abilities/.*research", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*research", re.I)],
        "requirements": lambda desc: f"""Build a research assistant *ability*:
- Academic paper search and analysis
- Citation extraction and bibliography generation
- Research topic exploration with follow-up questions
- Multi-source knowledge synthesis
- Include comprehensive tests, documentation, and examples
- Safety: source verification, fact-checking, proper attribution
Task: {desc}
""",
    },
    "information_retrieval": {
        "task_kind": "information_retrieval",
        "required_paths": [
            re.compile(r"^src/abilities/.*info.*retrieval", re.I),
            re.compile(r"^tests/abilities/.*info.*retrieval", re.I),
        ],
        "required_docs": [re.compile(r"^docs/.*info.*retrieval", re.I)],
        "requirements": lambda desc: f"""Build an information retrieval *ability*:
- Multi-source search and aggregation
- Intelligent ranking and relevance scoring
- Context-aware result filtering and deduplication
- Structured data extraction from search results
- Include comprehensive tests, documentation, and examples
- Safety: content validation, bias detection, source verification
Task: {desc}
""",
    },
}


async def _emit(bus: EventBus, topic: str, payload: dict[str, Any]) -> None:
    """Emit event to EventBus."""
    # Add required source_plugin field
    payload_with_source = {"source_plugin": "autogen_pipeline", **payload}
    event = create_event(event_type=topic, **payload_with_source)
    await bus.publish(event)


def _refine_requirements(prev_reasons: list[str], base: str) -> str:
    """Add refinement requirements based on gate failures."""
    patch = "\n".join(f"- Fix: {r}" for r in prev_reasons[:12])
    return base + "\n### Refinements required by gate:\n" + patch + "\n"


async def autogen_any(
    description: str,
    repo_path: str = ".",
    iterations: int = 5,
    event_bus: EventBus | None = None,
    api: NativeDeepCodeAPI | None = None,
) -> dict[str, Any]:
    """
    Generic ability autogeneration:
    Detect kinds -> choose template(s) -> Native DeepCode -> Gates -> Iterate -> Apply.
    Emits telemetry + OaK and bandit signals.
    
    Now uses native DeepCode integration instead of external API.
    """
    bus = event_bus or EventBus()
    api = api or get_native_deepcode_api()
    convo_id = f"autogen-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    kinds = NeedDetector().detect(description)

    if not kinds:
        await _emit(
            bus, "autogen.skipped", {"reason": "no_signals", "desc": description}
        )
        return {"status": "skipped", "reason": "no_signals"}

    results: dict[str, Any] = {"status": "complete", "applied": []}

    for kind in kinds:
        tpl = CAPABILITY_TEMPLATES.get(kind)
        if not tpl:
            await _emit(bus, "autogen.unknown_kind", {"kind": kind})
            continue

        # Propose to OaK planning layer: this "option" (create ability) is available
        await _emit(
            bus,
            "oak.plan_proposed",
            {
                "kind": kind,
                "desc": description,
                "conversation_id": convo_id,
                "option": "autogen_create_ability",
            },
        )
        await _emit(
            bus,
            "autogen.started",
            {"kind": kind, "desc": description, "conversation_id": convo_id},
        )

        req = tpl["requirements"](description)
        await api.deepcode_request(
            task_kind=tpl["task_kind"],
            requirements=req,
            repo_path=repo_path,
            conversation_id=convo_id,
        )

        gate = CombinedGate(
            RequiredPathsGate(
                required_paths=tpl["required_paths"],
                required_docs=tpl["required_docs"],
            ),
            SafetyGate(api),
            PytestGate(api),
        )

        passed = False
        for i in range(1, iterations + 1):
            latest = await api.deepcode_latest()
            ok, info = gate.validate_latest(latest)
            await _emit(
                bus,
                "autogen.iteration_checked",
                {"kind": kind, "iteration": i, "ok": ok, "gate": info},
            )
            if ok:
                paths = info.get("paths") or []
                apply_res = await api.deepcode_apply(paths=paths if paths else None)
                await _emit(
                    bus,
                    "autogen.applied",
                    {
                        "kind": kind,
                        "iteration": i,
                        "paths": paths,
                        "apply_result": apply_res,
                    },
                )
                # Bandit reward: success
                await _emit(
                    bus,
                    "bandit.reward_event",
                    {
                        "source": "autogen",
                        "kind": kind,
                        "reward": 1.0,
                        "desc": description,
                    },
                )
                # OaK outcome feedback
                await _emit(
                    bus,
                    "oak.outcome_feedback",
                    {"kind": kind, "success": True, "paths": paths},
                )
                results["applied"].append(
                    {"kind": kind, "paths": paths, "apply": apply_res}
                )
                passed = True
                break

            # refine + request again
            req = _refine_requirements(info.get("reasons") or [], req)
            await _emit(
                bus,
                "autogen.refine",
                {"kind": kind, "iteration": i, "reasons": info.get("reasons")},
            )
            await api.deepcode_request(
                task_kind=tpl["task_kind"],
                requirements=req,
                repo_path=repo_path,
                conversation_id=convo_id,
            )

        if not passed:
            await _emit(
                bus,
                "autogen.failed",
                {"kind": kind, "iterations": iterations, "desc": description},
            )
            # Bandit reward: failure (soft penalty)
            await _emit(
                bus,
                "bandit.reward_event",
                {
                    "source": "autogen",
                    "kind": kind,
                    "reward": 0.0,
                    "desc": description,
                },
            )
            # OaK outcome feedback
            await _emit(bus, "oak.outcome_feedback", {"kind": kind, "success": False})
            results.setdefault("failed", []).append(kind)

    return results
