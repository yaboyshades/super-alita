"""Constitutionally aligned capability code generation."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


class CodeGeneratorLike:
    """Protocol subset for text-to-code generators."""

    async def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class SpecificationValidatorLike:
    """Protocol subset for OpenSpec validators."""

    async def validate(self, code: str, spec: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class ConstitutionalCodeGenerator:
    """Generate code that adheres to OpenSpec and constitutional safeguards."""

    def __init__(
        self,
        code_generator: CodeGeneratorLike,
        *,
        validator: SpecificationValidatorLike | None = None,
        refinement_budget: int = 5,
    ) -> None:
        if code_generator is None:
            raise ValueError("code_generator must be provided")
        self.code_generator = code_generator
        self.validator = validator
        self.refinement_budget = refinement_budget

    async def generate_capability_code(
        self,
        capability_spec: dict[str, Any],
        openspec_constraints: dict[str, Any],
    ) -> str:
        """Generate code that satisfies the provided OpenSpec constraints."""

        prompt = await self._build_constitutional_prompt(
            capability_spec=capability_spec,
            constraints=openspec_constraints,
            architectural_patterns=await self._get_architectural_patterns(),
        )
        code = await self.code_generator.generate(prompt)
        validation = await self._validate_against_openspec(code, openspec_constraints)
        if validation.get("compliant"):
            return code
        violations = validation.get("violations", [])
        code = await self._refine_until_compliant(code, violations, openspec_constraints)
        # After refinement, validate again
        final_validation = await self._validate_against_openspec(code, openspec_constraints)
        if not final_validation.get("compliant"):
            unresolved_violations = final_validation.get("violations", [])
            logger.warning(
                "Code generation did not reach compliance after all refinement attempts. Unresolved violations: %s",
                unresolved_violations
            )
        return code

    async def _build_constitutional_prompt(
        self,
        *,
        capability_spec: dict[str, Any],
        constraints: dict[str, Any],
        architectural_patterns: Iterable[str],
    ) -> str:
        principles = [
            "Transparency",
            "Beneficence",
            "Autonomy",
            "Justice",
            "Non-maleficence",
        ]
        principle_text = "\n".join(f"- {p}" for p in principles)
        pattern_text = "\n".join(f"* {pattern}" for pattern in architectural_patterns)
        return (
            "You are a constitutional code generator.\n"
            "Follow these principles strictly:\n"
            f"{principle_text}\n\n"
            "Capability specification:\n"
            f"{capability_spec}\n\n"
            "OpenSpec constraints:\n"
            f"{constraints}\n\n"
            "Recommended architectural patterns:\n"
            f"{pattern_text}\n\n"
            "Return fully self-contained Python code that respects all constraints."
        )

    async def _get_architectural_patterns(self) -> list[str]:
        return [
            "Use dependency injection for runtime services",
            "Provide docstrings following Google style",
            "Validate inputs defensively",
        ]

    async def _validate_against_openspec(
        self, code: str, spec: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.validator:
            return {"compliant": True, "violations": []}
        try:
            result = await self.validator.validate(code, spec)
        except Exception:
            logger.exception("OpenSpec validation failed; treating as non-compliant")
            return {"compliant": False, "violations": ["validator_error"]}
        if not isinstance(result, dict):
            return {"compliant": False, "violations": ["invalid_validator_response"]}
        result.setdefault("violations", [])
        return result

    async def _refine_until_compliant(
        self,
        code: str,
        violations: Iterable[Any],
        spec: dict[str, Any],
    ) -> str:
        violations_text = "\n".join(str(v) for v in violations)
        current_code = code
        for _ in range(max(1, self.refinement_budget)):
            refinement_prompt = (
                "The following code violates OpenSpec checks.\n"
                f"Violations:\n{violations_text}\n\n"
                "Existing code:\n"
                f"{current_code}\n\n"
                "Produce a corrected version that resolves all issues."
            )
            current_code = await self.code_generator.generate(refinement_prompt)
            validation = await self._validate_against_openspec(current_code, spec)
            if validation.get("compliant"):
                return current_code
            violations_text = "\n".join(str(v) for v in validation.get("violations", []))
        return current_code
