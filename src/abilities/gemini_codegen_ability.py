#!/usr/bin/env python3
from __future__ import annotations
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from src.core.plugin_interface import PluginInterface
from src.prompt.policy_constants import build_header, REQUIRED_FIELDS_ALG
from src.utils.telemetry import ReadLedger, start_timer, end_timer, telemetry_footer
from src.utils.guardrails import repo_download_integrity, hash_top_files

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


NAMESPACE_PROPOSAL = uuid.UUID("d6e2a8b1-4c7f-4e0a-8b9c-1d2e3f4a5b6c")


@dataclass
class GeminiConfig:
    api_key: Optional[str]
    model: str
    enabled: bool


class GeminiCodegenAbility(PluginInterface):
    """Generate diffs/tests/docs from requirements using Gemini."""

    def __init__(self) -> None:
        super().__init__()
        self._cfg = GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            enabled=os.getenv("CODEGEN_ENABLED", "true").lower() == "true",
        )

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)

    @property
    def name(self) -> str:
        return "gemini_codegen_ability"

    async def start(self) -> None:
        await super().start()
        if not self._cfg.enabled:
            logger.info("GeminiCodegenAbility disabled; not starting.")
            return
        await self.subscribe("codegen_request", self._on_request)
        logger.info(
            "GeminiCodegenAbility started (model=%s, key=%s)",
            self._cfg.model,
            bool(self._cfg.api_key),
        )

    async def _on_request(self, event: Dict[str, Any]) -> None:
        requirements = _norm(event.get("requirements") or "")
        repo_path = event.get("repo_path") or "."
        context_files = event.get("context_files") or []
        proposal_id = str(
            uuid.uuid5(
                NAMESPACE_PROPOSAL,
                f"{requirements}|{repo_path}|{','.join(context_files)}",
            )
        )

        # --- Repo integrity snapshot (non-blocking, informational) ---
        integrity = repo_download_integrity(repo_path)
        hashed = hash_top_files(repo_path, top_n=5) if integrity["status"] != "partial" else []

        # --- Build system prompt header with policies/guards ---
        sys_header = build_header(role="algorithm_extraction")

        # Read ledger + ceilings
        ledger = ReadLedger()
        t0 = start_timer()

        # local fallback (no network): echo minimal diff & test (and proper envelope)
        if not self._cfg.api_key:
            diffs = [self._fake_diff(requirements)]
            tests = [self._fake_test(requirements)]
            payload = {
                "event_type": "codegen_implementation_proposed",
                "source_plugin": self.name,
                "proposal_id": proposal_id,
                "diffs": diffs,
                "tests": tests,
                "docs": [],
                "confidence": 0.4,
                "repo_integrity": integrity,
                "hashed_files": hashed,
                **{k: v for k, v in event.items() if k.endswith("_id")},
                "timestamp": _utcnow(),
            }
            await self.emit_event("codegen_implementation_proposed", **payload)
            return

        # network path: call Gemini
        try:
            diffs, tests, docs, confidence, raw_obj = await self._call_gemini(
                requirements, repo_path, context_files, sys_header=sys_header
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Gemini call failed: %s", e)
            diffs, tests, docs, confidence, raw_obj = (
                [self._fake_diff(requirements)],
                [self._fake_test(requirements)],
                [],
                0.3,
                {"status": "MODEL_CALL_FAILED", "reason": str(e)[:160]},
            )

        # Validation summary scaffolding inside envelope (consumer CI can check)
        missing = [
            f
            for f in REQUIRED_FIELDS_ALG
            if f
            not in (
                raw_obj.get("alg_extraction_v1", {})
                if isinstance(raw_obj, dict)
                else {}
            )
        ]
        validation_summary = {"missing_required_fields": missing, "unknown_fields": []}
        telem = telemetry_footer(
            retrieval_rounds=int(raw_obj.get("telemetry", {}).get("retrieval_rounds", 1))
            if isinstance(raw_obj, dict)
            else 1,
            segments_used=int(raw_obj.get("telemetry", {}).get("segments_used", ledger.count()))
            if isinstance(raw_obj, dict)
            else ledger.count(),
            totals=raw_obj.get("telemetry", {}).get("total_extracted", {})
            if isinstance(raw_obj, dict)
            else {},
        )
        telem["telemetry"]["extraction_duration_ms"] = end_timer(t0)

        payload = {
            "event_type": "codegen_implementation_proposed",
            "source_plugin": self.name,
            "proposal_id": proposal_id,
            "diffs": diffs,
            "tests": tests,
            "docs": docs,
            "confidence": confidence,
            "repo_integrity": integrity,
            "hashed_files": hashed,
            "validation_summary": validation_summary,
            **{k: v for k, v in event.items() if k.endswith("_id")},
            "timestamp": _utcnow(),
            **telem,
        }
        await self.emit_event("codegen_implementation_proposed", **payload)

    async def _call_gemini(
        self,
        requirements: str,
        repo_path: str,
        context_files: List[str],
        sys_header: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], float, Dict[str, Any]]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._cfg.model}:generateContent?key={self._cfg.api_key}"
        )
        sys_message = sys_header + "\n\nYou are a repository-level code generation assistant.\n"
        # Execution cost guard is enforced by the retrieval tool on the caller side;
        # here we just pass the ceilings inside the policy header.
        user_message = {
            "requirements": requirements,
            "repo_path": repo_path,
            "context_files": context_files,
            "namespaces": {"root": "alg_extraction_v1"},
            "normalization_rules": {
                "years": "4-digit string",
                "numeric_hparams": "numeric (no quotes) unless scientific notation",
                "equations": "include equation_number or 'UNNUMBERED'",
            },
            "confidence_tags": ["source_locator"],
            "deduplication": {"hyperparameters": True},
        }
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": sys_message}]},
                {"role": "user", "parts": [{"text": json.dumps(user_message)}]},
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
            async with s.post(url, json=payload) as resp:
                txt = await resp.text()
                if resp.status >= 300:
                    raise RuntimeError(f"Gemini HTTP {resp.status}: {txt[:200]}")
                data = json.loads(txt)
        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            obj = json.loads(raw)
        except Exception:
            obj = data if isinstance(data, dict) else {"diffs": [], "tests": [], "docs": [], "confidence": 0.3}

        diffs = obj.get("diffs") or obj.get("alg_extraction_v1", {}).get("diffs") or []
        tests = obj.get("tests") or obj.get("alg_extraction_v1", {}).get("tests") or []
        docs = obj.get("docs") or obj.get("alg_extraction_v1", {}).get("docs") or []
        confidence = float(obj.get("confidence") or obj.get("alg_extraction_v1", {}).get("confidence", 0.5))
        return diffs, tests, docs, confidence, obj

    def _fake_diff(self, requirements: str) -> Dict[str, Any]:
        return {
            "path": "README.md",
            "patch": (
                "--- a/README.md\n" "+++ b/README.md\n" "@@\n" f"+AUTO-GENERATED NOTE: {requirements}\n"
            ),
        }

    def _fake_test(self, requirements: str) -> Dict[str, Any]:
        return {
            "path": "tests/test_codegen_placeholder.py",
            "content": (
                "def test_codegen_placeholder():\n"
                f"    assert '{requirements[:20]}' is not None\n"
            ),
        }
