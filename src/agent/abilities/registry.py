"""Ability registry responsible for managing agent abilities and metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import yaml


@dataclass(slots=True)
class AbilityMetadata:
    """Metadata describing an ability's execution constraints and costs."""

    name: str
    description: str
    parameters: Dict[str, Any]
    required_parameters: List[str]
    cost_estimate: float
    risk_level: str
    timeout_seconds: int
    constitutional_risk_factors: List[str]
    fallback_ability: Optional[str] = None


@dataclass(slots=True)
class Ability:
    """Ability definition with asynchronous execution hooks."""

    name: str
    metadata: AbilityMetadata
    execute_fn: Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Any]]
    validate_fn: Callable[[Dict[str, Any]], Awaitable[Any]]
    dry_run_fn: Callable[[Dict[str, Any]], Awaitable[Any]]

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Run the ability's execution function."""
        return await self.execute_fn(parameters, context)

    async def validate(self, parameters: Dict[str, Any]) -> Any:
        """Validate parameters for the ability."""
        return await self.validate_fn(parameters)

    async def dry_run(self, parameters: Dict[str, Any]) -> Any:
        """Perform a dry run of the ability."""
        return await self.dry_run_fn(parameters)


class AbilityRegistry:
    """Central registry for agent abilities with safety metadata."""

    def __init__(self, abilities_config_path: str = "config/abilities.yaml") -> None:
        self.abilities_config_path = abilities_config_path
        self.abilities: Dict[str, Ability] = {}
        self.logger = logging.getLogger(__name__)
        self._load_abilities_from_config()

    def _load_abilities_from_config(self) -> None:
        """Load ability definitions from configuration."""
        try:
            config_path = Path(self.abilities_config_path)
            if not config_path.exists():
                self.logger.warning(
                    "Abilities config not found at %s, using defaults", config_path
                )
                self._load_default_abilities()
                return

            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

            for ability_name, ability_config in config.get("abilities", {}).items():
                self._register_ability_from_config(ability_name, ability_config)

            self.logger.info("Loaded %s abilities from config", len(self.abilities))
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error("Failed to load abilities config: %s", exc)
            self._load_default_abilities()

    def _load_default_abilities(self) -> None:
        """Load a minimal ability set when configuration is missing."""
        default_abilities = {
            "code_analysis": {
                "description": "Analyze code for issues and improvements",
                "parameters": {"code": "string", "context": "dict"},
                "required_parameters": ["code"],
                "cost_estimate": 0.5,
                "risk_level": "low",
                "timeout_seconds": 30,
                "constitutional_risk_factors": ["code_review"],
            },
            "search_code": {
                "description": "Search for code patterns",
                "parameters": {"query": "string", "filters": "dict"},
                "required_parameters": ["query"],
                "cost_estimate": 0.3,
                "risk_level": "low",
                "timeout_seconds": 20,
                "constitutional_risk_factors": ["information_gathering"],
            },
            "code_generation": {
                "description": "Generate code based on requirements",
                "parameters": {"requirements": "string", "context": "dict"},
                "required_parameters": ["requirements"],
                "cost_estimate": 1.0,
                "risk_level": "medium",
                "timeout_seconds": 60,
                "constitutional_risk_factors": ["code_creation", "potential_misuse"],
            },
        }

        for ability_name, ability_config in default_abilities.items():
            self._register_ability_from_config(ability_name, ability_config)

    def _register_ability_from_config(
        self, ability_name: str, config: Dict[str, Any]
    ) -> None:
        """Create and register an ability from configuration metadata."""
        try:
            metadata = AbilityMetadata(
                name=ability_name,
                description=config.get("description", ""),
                parameters=config.get("parameters", {}),
                required_parameters=config.get("required_parameters", []),
                cost_estimate=float(config.get("cost_estimate", 1.0)),
                risk_level=config.get("risk_level", "medium"),
                timeout_seconds=int(config.get("timeout_seconds", 30)),
                constitutional_risk_factors=config.get(
                    "constitutional_risk_factors", []
                ),
                fallback_ability=config.get("fallback_ability"),
            )

            ability = Ability(
                name=ability_name,
                metadata=metadata,
                execute_fn=self._create_stub_executor(ability_name),
                validate_fn=self._create_stub_validator(ability_name),
                dry_run_fn=self._create_stub_dry_runner(ability_name),
            )
            self.abilities[ability_name] = ability
            self.logger.debug("Registered ability %s", ability_name)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to register ability %s: %s", ability_name, exc)

    def _create_stub_executor(
        self, ability_name: str
    ) -> Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Any]]:
        async def _stub(parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
            raise NotImplementedError(
                "Executor for ability '%s' not implemented. Provide an adapter." % ability_name
            )

        return _stub

    def _create_stub_validator(
        self, ability_name: str
    ) -> Callable[[Dict[str, Any]], Awaitable[Any]]:
        async def _stub(parameters: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "valid": True,
                "errors": [],
                "warnings": [f"Validation for {ability_name} not implemented"],
            }

        return _stub

    def _create_stub_dry_runner(
        self, ability_name: str
    ) -> Callable[[Dict[str, Any]], Awaitable[Any]]:
        async def _stub(parameters: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "would_execute": True,
                "estimated_cost": 1.0,
                "risk_assessment": "unknown",
                "steps": [f"Dry run for {ability_name} not implemented"],
            }

        return _stub

    def get_ability(self, ability_name: str) -> Optional[Ability]:
        """Retrieve an ability by name."""
        return self.abilities.get(ability_name)

    def list_abilities(self) -> List[AbilityMetadata]:
        """Return metadata for all registered abilities."""
        return [ability.metadata for ability in self.abilities.values()]

    def register_ability_adapter(self, ability_name: str, adapter: Any) -> None:
        """Attach an adapter implementation to an existing ability."""
        ability = self.abilities.get(ability_name)
        if not ability:
            self.logger.warning("Cannot register adapter for unknown ability: %s", ability_name)
            return

        if hasattr(adapter, "execute"):
            ability.execute_fn = adapter.execute  # type: ignore[assignment]
        if hasattr(adapter, "validate"):
            ability.validate_fn = adapter.validate  # type: ignore[assignment]
        if hasattr(adapter, "dry_run"):
            ability.dry_run_fn = adapter.dry_run  # type: ignore[assignment]

        self.logger.info("Registered adapter for ability %s", ability_name)

    async def validate_ability_execution(
        self,
        ability_name: str,
        parameters: Dict[str, Any],
        security_context: Any,
    ) -> Dict[str, Any]:
        """Validate ability execution using both ability metadata and security context."""
        ability = self.get_ability(ability_name)
        if not ability:
            return {
                "valid": False,
                "errors": [f"Unknown ability: {ability_name}"],
                "warnings": [],
            }

        validation_result = await ability.validate(parameters)

        if security_context and hasattr(security_context, "validate_input"):
            security_validation = await security_context.validate_input(
                {"ability": ability_name, "parameters": parameters},
                self._get_ability_schema(ability),
            )

            if not getattr(security_validation, "valid", True):
                validation_result["valid"] = False
                validation_result.setdefault("errors", []).extend(
                    getattr(security_validation, "errors", [])
                )
                validation_result.setdefault("warnings", []).extend(
                    getattr(security_validation, "warnings", [])
                )

        return validation_result

    def _get_ability_schema(self, ability: Ability) -> Dict[str, Any]:
        """Construct JSON schema for validating ability inputs."""
        return {
            "type": "object",
            "properties": {
                "ability": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": ability.metadata.parameters,
                    "required": ability.metadata.required_parameters,
                },
            },
            "required": ["ability", "parameters"],
        }

    def get_abilities_by_risk_level(self, risk_level: str) -> List[AbilityMetadata]:
        """Return abilities filtered by risk level."""
        return [
            ability.metadata
            for ability in self.abilities.values()
            if ability.metadata.risk_level == risk_level
        ]

    def get_abilities_by_cost_range(
        self, min_cost: float, max_cost: float
    ) -> List[AbilityMetadata]:
        """Return abilities whose cost estimate falls within a range."""
        return [
            ability.metadata
            for ability in self.abilities.values()
            if min_cost <= ability.metadata.cost_estimate <= max_cost
        ]


def create_ability_registry(abilities_config_path: str = "config/abilities.yaml") -> AbilityRegistry:
    """Factory for creating an ability registry."""
    return AbilityRegistry(abilities_config_path)
