#!/usr/bin/env python3
"""
Mangle Integration Extensions for Output Validation

This module provides additional Mangle integration capabilities for:
1. Output Validation Gates - Rejecting output based on policy violations
2. Tool/Action Gating - Pre-execution checks for high-risk tools
3. Routing/Aggregation - Mangle-based consensus method selection
4. LLM Post-hoc Verification - Validating model claims
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MangleValidator:
    """Mangle-based validation extensions for Super Alita."""

    def __init__(self, mangle_ability):
        """Initialize the validator with a MangleAbility instance."""
        self.mangle = mangle_ability

    async def validate_output(
        self,
        output_text: str,
        domain: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate output against Mangle policy rules.

        Uses rules with head violation(Reason) to detect policy violations
        in the output. If violations are found, returns details with rejection
        flag set to True.

        Args:
            output_text: The output text to validate
            domain: Optional domain for domain-specific rules
            meta: Optional metadata about the generation context

        Returns:
            Dictionary with keys:
                - valid: Boolean indicating if the output passes validation
                - violations: List of violation reasons found
                - confidence_penalty: Suggested confidence penalty (0.0-1.0)
        """
        try:
            rules_file = Path("./data/mangle/rules.json")
            if not rules_file.exists():
                return {
                    "valid": True,
                    "violations": [],
                    "confidence_penalty": 0.0
                }

            # Build temp facts about the output and context
            tfacts = []

            # Add truncated output text fact (with simple escaping)
            safe_text = output_text[:500].replace("'", " ")
            tfacts.append(f"output_text('{safe_text}')")

            if domain:
                safe_domain = domain.replace("'", " ")
                tfacts.append(f"domain('{safe_domain}')")

            if meta:
                for k, v in meta.items():
                    if isinstance(v, (int, float)):
                        tfacts.append(f"{k}({v})")
                    elif isinstance(v, str):
                        safe_v = v.replace("'", " ")
                        tfacts.append(f"{k}('{safe_v}')")

            # Check for violation rules
            violations = []
            confidence_penalty = 0.0

            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
            for rule_id, meta_rule in rules_data.items():
                body = meta_rule.get("body") or meta_rule.get("rule") or ""
                if not body:
                    continue

                # Check if this is a violation rule
                head = body.split(":-", 1)[0].strip().rstrip(".")
                if not head.startswith("violation("):
                    continue

                # Run the violation check
                res = await self.mangle.run_rule(
                    body,
                    "violation(Reason)",
                    temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    try:
                        for result in res.get("results", []):
                            reason = result.get("Reason")
                            if reason:
                                violations.append(str(reason))
                    except Exception as e:
                        logger.warning(f"Error processing violation result: {e}")
                        continue

            # If any violations found, calculate penalty and return details
            valid = len(violations) == 0
            if not valid:
                # Scale penalty based on violation count, max 0.5
                confidence_penalty = min(0.5, 0.1 * len(violations))

            return {
                "valid": valid,
                "violations": violations,
                "confidence_penalty": confidence_penalty
            }
        except Exception as e:
            logger.warning(f"Output validation error: {e}")
            return {
                "valid": True,
                "violations": [],
                "confidence_penalty": 0.0
            }

    async def validate_tool_execution(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Pre-execution validation for tool/action requests.

        Uses rules with head deny_tool(Reason) to check if a tool execution
        should be permitted based on parameters and context.

        Args:
            tool_name: Name of the tool to be executed
            params: Parameters to be passed to the tool
            context: Optional execution context (user, session, etc.)

        Returns:
            Dictionary with keys:
                - authorized: Boolean indicating if tool execution is permitted
                - reason: Optional explanation if not authorized
        """
        try:
            rules_file = Path("./data/mangle/rules.json")
            if not rules_file.exists():
                return {"authorized": True, "reason": None}

            # Build temp facts about the tool execution request
            tfacts = []

            # Add tool name
            safe_tool_name = tool_name.replace("'", " ")
            tfacts.append(f"tool_name('{safe_tool_name}')")

            # Add parameter facts
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    tfacts.append(f"param('{k}', {v})")
                elif isinstance(v, str):
                    safe_v = v.replace("'", " ")
                    tfacts.append(f"param('{k}', '{safe_v}')")
                elif isinstance(v, bool):
                    tfacts.append(f"param('{k}', {str(v).lower()})")

            # Add context facts
            if context:
                for k, v in context.items():
                    if isinstance(v, (int, float)):
                        tfacts.append(f"context('{k}', {v})")
                    elif isinstance(v, str):
                        safe_v = v.replace("'", " ")
                        tfacts.append(f"context('{k}', '{safe_v}')")
                    elif isinstance(v, bool):
                        tfacts.append(f"context('{k}', {str(v).lower()})")

            # Check for tool authorization rules
            denied_reasons = []

            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
            for rule_id, meta_rule in rules_data.items():
                body = meta_rule.get("body") or meta_rule.get("rule") or ""
                if not body:
                    continue

                # Check if this is a deny_tool rule
                head = body.split(":-", 1)[0].strip().rstrip(".")
                if not head.startswith("deny_tool("):
                    continue

                # Run the authorization check
                res = await self.mangle.run_rule(
                    body,
                    "deny_tool(Reason)",
                    temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    try:
                        for result in res.get("results", []):
                            reason = result.get("Reason")
                            if reason:
                                denied_reasons.append(str(reason))
                    except Exception:
                        continue

            authorized = len(denied_reasons) == 0
            reason = denied_reasons[0] if denied_reasons else None

            return {
                "authorized": authorized,
                "reason": reason
            }
        except Exception as e:
            logger.warning(f"Tool validation error: {e}")
            return {"authorized": True, "reason": None}

    async def select_consensus_method(
        self,
        domain: Optional[str] = None,
        sample_count: int = 0,
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Select an aggregation method based on Mangle rules.

        Uses rules with head select_method(Method) to choose the appropriate
        consensus method based on domain, sample count, and other metadata.

        Args:
            domain: Optional domain for domain-specific method selection
            sample_count: Number of samples being aggregated
            meta: Optional metadata about the request

        Returns:
            Dictionary with keys:
                - method: Selected consensus method name
                - reason: Optional explanation for the selection
        """
        try:
            rules_file = Path("./data/mangle/rules.json")
            if not rules_file.exists():
                return {
                    "method": "weighted_vote",
                    "reason": "Default method (no rules)"
                }

            # Build temp facts for method selection
            tfacts = [f"sample_count({sample_count})"]

            if domain:
                safe_domain = domain.replace("'", " ")
                tfacts.append(f"domain('{safe_domain}')")

            if meta:
                for k, v in meta.items():
                    if isinstance(v, (int, float)):
                        tfacts.append(f"{k}({v})")
                    elif isinstance(v, str):
                        safe_v = v.replace("'", " ")
                        tfacts.append(f"{k}('{safe_v}')")

            # Check for method selection rules
            methods = []
            reasons = []

            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
            for rule_id, meta_rule in rules_data.items():
                body = meta_rule.get("body") or meta_rule.get("rule") or ""
                if not body:
                    continue

                # Check if this is a select_method rule
                head = body.split(":-", 1)[0].strip().rstrip(".")
                if not head.startswith("select_method("):
                    continue

                # Run the method selection
                res = await self.mangle.run_rule(
                    body,
                    "select_method(Method, Reason)",
                    temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    try:
                        for result in res.get("results", []):
                            method = result.get("Method")
                            reason = result.get("Reason")
                            if method:
                                methods.append(str(method))
                                if reason:
                                    reasons.append(str(reason))
                    except Exception:
                        continue

            # If multiple methods are selected, pick the first one
            if methods:
                return {
                    "method": methods[0],
                    "reason": reasons[0] if reasons else f"Selected via rule: {methods[0]}"
                }

            # Default to weighted_vote if no method selected
            return {
                "method": "weighted_vote",
                "reason": "Default method (no matching rules)"
            }
        except Exception as e:
            logger.warning(f"Method selection error: {e}")
            return {
                "method": "weighted_vote",
                "reason": f"Default method (error: {e})"
            }

    async def verify_llm_claims(
        self,
        output_text: str,
        claims_type: str = "factual",
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Verify claims made in LLM output using Mangle rules.

        Uses rules with head invalid_claim(Claim, Reason) to identify problematic
        claims in the LLM output based on verification rules.

        Args:
            output_text: The LLM output text containing claims to verify
            claims_type: Type of claims to verify (factual, code, etc.)
            meta: Optional metadata about the generation

        Returns:
            Dictionary with keys:
                - verified: Boolean indicating if all claims pass verification
                - invalid_claims: List of invalid claims with reasons
                - confidence_adjustment: Suggested confidence adjustment (-1.0 to 0.0)
        """
        try:
            rules_file = Path("./data/mangle/rules.json")
            if not rules_file.exists():
                return {
                    "verified": True,
                    "invalid_claims": [],
                    "confidence_adjustment": 0.0
                }

            # Build temp facts about the output
            tfacts = []

            # Add truncated output text fact (with simple escaping)
            safe_text = output_text[:500].replace("'", " ")
            tfacts.append(f"output_text('{safe_text}')")
            tfacts.append(f"claims_type('{claims_type}')")

            if meta:
                for k, v in meta.items():
                    if isinstance(v, (int, float)):
                        tfacts.append(f"{k}({v})")
                    elif isinstance(v, str):
                        safe_v = v.replace("'", " ")
                        tfacts.append(f"{k}('{safe_v}')")

            # Check for invalid claim rules
            invalid_claims = []

            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
            for rule_id, meta_rule in rules_data.items():
                body = meta_rule.get("body") or meta_rule.get("rule") or ""
                if not body:
                    continue

                # Check if this is an invalid_claim rule
                head = body.split(":-", 1)[0].strip().rstrip(".")
                if not head.startswith("invalid_claim("):
                    continue

                # Run the claim verification
                res = await self.mangle.run_rule(
                    body,
                    "invalid_claim(Claim, Reason)",
                    temp_facts=tfacts
                )
                if res.get("success") and res.get("count", 0) > 0:
                    try:
                        for result in res.get("results", []):
                            claim = result.get("Claim")
                            reason = result.get("Reason")
                            if claim:
                                invalid_claims.append({
                                    "claim": str(claim),
                                    "reason": str(reason) if reason else "Unverified claim"
                                })
                    except Exception:
                        continue

            # Calculate confidence adjustment based on invalid claims
            verified = len(invalid_claims) == 0
            confidence_adjustment = max(-0.5, -0.1 * len(invalid_claims))

            return {
                "verified": verified,
                "invalid_claims": invalid_claims,
                "confidence_adjustment": confidence_adjustment if not verified else 0.0
            }
        except Exception as e:
            logger.warning(f"Claim verification error: {e}")
            return {
                "verified": True,
                "invalid_claims": [],
                "confidence_adjustment": 0.0
            }
