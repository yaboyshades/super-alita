#!/usr/bin/env python3
"""Mangle integration validator utilities.

Provides high-level validation helpers that leverage Mangle rule execution for:
1. Output validation (violation detection)
2. Tool execution authorization (deny rules)
3. Consensus method selection
4. LLM claim verification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MangleValidator:
    """Validation and selection helpers backed by Mangle rules.

    The validator depends on an object (mangle_ability) that exposes an
    async method `run_rule(rule_text: str, query: str, temp_facts: list[str])`.
    """

    def __init__(self, mangle_ability: Any) -> None:
        self.mangle = mangle_ability

    # ------------------------------------------------------------------
    # Output Validation
    # ------------------------------------------------------------------
    async def validate_output(
        self,
        output_text: str,
        domain: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate generated output for policy violations.

        Returns dict with keys: valid, violations, confidence_penalty.
        """
        rules_file = Path("./data/mangle/rules.json")
        if not rules_file.exists():  # Fast path
            return {"valid": True, "violations": [], "confidence_penalty": 0.0}

        tfacts: list[str] = []
        safe_text = output_text[:500].replace("'", " ")
        tfacts.append(f"output_text('{safe_text}')")
        if domain:
            safe_domain = domain.replace("'", " ")
            tfacts.append(f"domain('{safe_domain}')")
        if meta:
            for k, v in meta.items():
                if isinstance(v, int | float):
                    tfacts.append(f"{k}({v})")
                elif isinstance(v, str):
                    safe_v = v.replace("'", " ")
                    tfacts.append(f"{k}('{safe_v}')")

        violations: list[str] = []
        confidence_penalty = 0.0

        try:
            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading rules file: %s", exc)
            return {"valid": True, "violations": [], "confidence_penalty": 0.0}

        for meta_rule in rules_data.values():
            body = (
                meta_rule.get("body") or meta_rule.get("rule") or ""
            ).strip()
            if not body:
                continue
            head = body.split(":-", 1)[0].strip().rstrip(".")
            if not head.startswith("violation("):
                continue
            try:
                res = await self.mangle.run_rule(
                    body, "violation(Reason)", temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    for result in res.get("results", []):
                        reason = result.get("Reason")
                        if reason:
                            violations.append(str(reason))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Violation rule execution failed: %s", exc)
                continue

        valid = not violations
        if not valid:
            confidence_penalty = min(0.5, 0.1 * len(violations))
        return {
            "valid": valid,
            "violations": violations,
            "confidence_penalty": confidence_penalty,
        }

    # ------------------------------------------------------------------
    # Tool Authorization
    # ------------------------------------------------------------------
    async def validate_tool_execution(
        self,
        tool_name: str,
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Determine if a tool execution should proceed.

        Looks for deny_tool(Reason) rule matches; if any found, returns
        authorized = False with first reason.
        """
        rules_file = Path("./data/mangle/rules.json")
        if not rules_file.exists():
            return {"authorized": True, "reason": None}

        tfacts: list[str] = []
        safe_tool_name = tool_name.replace("'", " ")
        tfacts.append(f"tool_name('{safe_tool_name}')")
        for k, v in params.items():
            if isinstance(v, int | float):
                tfacts.append(f"param('{k}', {v})")
            elif isinstance(v, str):
                safe_v = v.replace("'", " ")
                tfacts.append(f"param('{k}', '{safe_v}')")
            elif isinstance(v, bool):
                tfacts.append(f"param('{k}', {str(v).lower()})")
        if context:
            for k, v in context.items():
                if isinstance(v, int | float):
                    tfacts.append(f"context('{k}', {v})")
                elif isinstance(v, str):
                    safe_v = v.replace("'", " ")
                    tfacts.append(f"context('{k}', '{safe_v}')")
                elif isinstance(v, bool):
                    tfacts.append(f"context('{k}', {str(v).lower()})")

        try:
            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading rules file: %s", exc)
            return {"authorized": True, "reason": None}

        denied_reasons: list[str] = []
        for meta_rule in rules_data.values():
            body = (
                meta_rule.get("body") or meta_rule.get("rule") or ""
            ).strip()
            if not body:
                continue
            head = body.split(":-", 1)[0].strip().rstrip(".")
            if not head.startswith("deny_tool("):
                continue
            try:
                res = await self.mangle.run_rule(
                    body, "deny_tool(Reason)", temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    for result in res.get("results", []):
                        reason = result.get("Reason")
                        if reason:
                            denied_reasons.append(str(reason))
            except Exception as exc:  # noqa: BLE001
                logger.debug("deny_tool rule failed: %s", exc)
                continue

        authorized = not denied_reasons
        reason = denied_reasons[0] if denied_reasons else None
        return {"authorized": authorized, "reason": reason}

    # ------------------------------------------------------------------
    # Consensus Method Selection
    # ------------------------------------------------------------------
    async def select_consensus_method(
        self,
        domain: str | None = None,
        sample_count: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select a consensus method.

        Uses select_method(Method, Reason) rules; defaults to weighted_vote.
        """
        rules_file = Path("./data/mangle/rules.json")
        if not rules_file.exists():
            return {"method": "weighted_vote", "reason": "No rules file"}

        tfacts: list[str] = [f"sample_count({sample_count})"]
        if domain:
            safe_domain = domain.replace("'", " ")
            tfacts.append(f"domain('{safe_domain}')")
        if meta:
            for k, v in meta.items():
                if isinstance(v, int | float):
                    tfacts.append(f"{k}({v})")
                elif isinstance(v, str):
                    safe_v = v.replace("'", " ")
                    tfacts.append(f"{k}('{safe_v}')")

        try:
            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading rules file: %s", exc)
            return {"method": "weighted_vote", "reason": "Error reading rules"}

        methods: list[str] = []
        reasons: list[str] = []
        for meta_rule in rules_data.values():
            body = (
                meta_rule.get("body") or meta_rule.get("rule") or ""
            ).strip()
            if not body:
                continue
            head = body.split(":-", 1)[0].strip().rstrip(".")
            if not head.startswith("select_method("):
                continue
            try:
                res = await self.mangle.run_rule(
                    body, "select_method(Method, Reason)", temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    for result in res.get("results", []):
                        method = result.get("Method")
                        reason = result.get("Reason")
                        if method:
                            methods.append(str(method))
                            if reason:
                                reasons.append(str(reason))
            except Exception as exc:  # noqa: BLE001
                logger.debug("select_method rule failed: %s", exc)
                continue

        if methods:
            return {
                "method": methods[0],
                "reason": (
                    reasons[0]
                    if reasons
                    else f"Selected {methods[0]} via rule"
                ),
            }
        return {"method": "weighted_vote", "reason": "Default (no match)"}

    # ------------------------------------------------------------------
    # LLM Claim Verification
    # ------------------------------------------------------------------
    async def verify_llm_claims(
        self,
        output_text: str,
        claims_type: str = "factual",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Identify invalid claims using invalid_claim(Claim, Reason) rules."""
        rules_file = Path("./data/mangle/rules.json")
        if not rules_file.exists():
            return {
                "verified": True,
                "invalid_claims": [],
                "confidence_adjustment": 0.0,
            }

        tfacts: list[str] = []
        safe_text = output_text[:500].replace("'", " ")
        tfacts.append(f"output_text('{safe_text}')")
        tfacts.append(f"claims_type('{claims_type}')")
        if meta:
            for k, v in meta.items():
                if isinstance(v, int | float):
                    tfacts.append(f"{k}({v})")
                elif isinstance(v, str):
                    safe_v = v.replace("'", " ")
                    tfacts.append(f"{k}('{safe_v}')")

        try:
            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading rules file: %s", exc)
            return {
                "verified": True,
                "invalid_claims": [],
                "confidence_adjustment": 0.0,
            }

        invalid_claims: list[dict[str, str]] = []
        for meta_rule in rules_data.values():
            body = (
                meta_rule.get("body") or meta_rule.get("rule") or ""
            ).strip()
            if not body:
                continue
            head = body.split(":-", 1)[0].strip().rstrip(".")
            if not head.startswith("invalid_claim("):
                continue
            try:
                res = await self.mangle.run_rule(
                    body, "invalid_claim(Claim, Reason)", temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    for result in res.get("results", []):
                        claim = result.get("Claim")
                        reason = result.get("Reason")
                        if claim:
                            invalid_claims.append(
                                {
                                    "claim": str(claim),
                                    "reason": (
                                        str(reason)
                                        if reason
                                        else "Unverified claim"
                                    ),
                                }
                            )
            except Exception as exc:  # noqa: BLE001
                logger.debug("invalid_claim rule failed: %s", exc)
                continue

        verified = not invalid_claims
        confidence_adjustment = (
            0.0 if verified else max(-0.5, -0.1 * len(invalid_claims))
        )
        return {
            "verified": verified,
            "invalid_claims": invalid_claims,
            "confidence_adjustment": confidence_adjustment,
        }


__all__ = ["MangleValidator"]
